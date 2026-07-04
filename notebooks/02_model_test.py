# Google Colab Interactive Notebook Cell Code - Model Shape and Forward Pass Test
# Copy and run these cells in your Jupyter/Colab notebook.

# %% Cell 1: Imports
import torch
import numpy as np
import yaml
from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.policy import DreamerActorCritic
from pcb_router.models.heatmap_decoder import HeatmapDecoder
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.data.graph_builder import GraphBuilder

# %% Cell 2: Load parameters and configs
with open('configs/model.yaml', 'r') as f:
    model_cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using test device: {device}")

# %% Cell 3: Instantiate all models
vit = ViTEncoder(
    image_channels=model_cfg['vit']['image_channels'],
    patch_size=model_cfg['vit']['patch_size'],
    embed_dim=model_cfg['vit']['embed_dim'],
    num_heads=model_cfg['vit']['num_heads'],
    num_layers=model_cfg['vit']['num_layers']
).to(device)

gnn = HeteroGATEncoder(
    hidden_dim=model_cfg['gnn']['hidden_dim'],
    out_dim=model_cfg['gnn']['out_dim'],
    num_layers=model_cfg['gnn']['num_layers'],
    num_heads=model_cfg['gnn']['num_heads']
).to(device)

fusion = CrossAttentionFusion(
    num_layers=model_cfg['fusion']['num_layers'],
    embed_dim=model_cfg['fusion']['embed_dim'],
    num_heads=model_cfg['fusion']['num_heads']
).to(device)

policy = DreamerActorCritic(
    embed_dim=model_cfg['vit']['embed_dim']
).to(device)

decoder = HeatmapDecoder(
    latent_dim=model_cfg['heatmap_decoder']['latent_dim'],
    spatial_dim=model_cfg['vit']['embed_dim']
).to(device)

print("All models successfully instantiated and moved to device.")
print(f"ViT params: {sum(p.numel() for p in vit.parameters())/1e6:.2f}M")
print(f"GNN params: {sum(p.numel() for p in gnn.parameters())/1e6:.2f}M")
print(f"Fusion params: {sum(p.numel() for p in fusion.parameters())/1e6:.2f}M")
print(f"Policy params: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M")
print(f"Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

# %% Cell 4: Test a forward pass
print("\nRunning simulated forward pass...")
B = 1
H, W = 256, 256 # grid dimensions (variable)
C = 13

# Create dummy input raster
raster_tensor = torch.randn(B, C, H, W, device=device)

# Generate dummy GNN data
generator = BoardGenerator()
board_cfg = BoardConfig(board_width=H, board_height=W, num_nets=4, num_layers=2)
board = generator.generate(board_cfg)
builder = GraphBuilder()
graph = builder.build_graph(board)

# Send graph tensors to device
x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

# 1. Forward ViT
spatial_patches, cls_spatial = vit(raster_tensor)
print(f"ViT Spatial Patches shape: {spatial_patches.shape} (B, N_patches, dim)")
print(f"ViT CLS token shape: {cls_spatial.shape} (B, dim)")

# 2. Forward GNN
node_embs = gnn(x_dict, edge_index_dict)
pads_emb = node_embs['pad'].unsqueeze(0) # (1, N_pads, dim)
print(f"GNN Pad Embeddings shape: {pads_emb.shape} (B, N_pads, dim)")

# 3. Forward Fusion
f_pads, f_spat = fusion(pads_emb, spatial_patches)
print(f"Fused Pads shape: {f_pads.shape}")
print(f"Fused Spatial shape: {f_spat.shape}")

# Average pad embeddings to net embeddings
num_nets = len(board.nets)
net_embs = torch.zeros((B, num_nets, vit.embed_dim), device=device)
for net_idx, net in enumerate(board.nets):
    pin_indices = [idx for idx, p in enumerate(board.pins.values()) if p.net_id == net.id]
    if pin_indices:
        net_embs[0, net_idx] = f_pads[0, pin_indices].mean(dim=0)
        
unrouted_mask = torch.ones((B, num_nets), dtype=torch.bool, device=device)

# 4. Forward Policy (Requires dummy latent states h and z)
h = torch.zeros(B, 512, device=device)
z = torch.zeros(B, 1024, device=device)
net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, value = policy(
    net_embs, unrouted_mask, h, z
)
print(f"Selected net action index: {net_idx.item()}")
print(f"Heatmap latent action shape: {heatmap_latent.shape} (B, latent_dim)")
print(f"Policy value estimation: {value.item():.3f}")

# 5. Forward Heatmap Decoder
layer_mask = torch.ones((B, 8), device=device)
heatmaps_via = decoder(
    heatmap_latent, f_spat,
    H, W, active_layers_mask=layer_mask
)
print(f"Decoded Heatmaps output shape: {heatmaps_via.shape} (B, N_layers+1, H, W)")
print("Forward pass completed successfully without errors!")
