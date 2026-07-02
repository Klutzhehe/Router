import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        embed_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        self.layers = nn.ModuleList([
            FusionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        gnn_embeddings: torch.Tensor,
        spatial_features: torch.Tensor,
        gnn_mask: torch.Tensor = None,
        spatial_mask: torch.Tensor = None
    ):
        """
        Args:
            gnn_embeddings: (B, N_nodes, embed_dim)
            spatial_features: (B, N_patches, embed_dim)
            gnn_mask: (B, N_nodes) boolean tensor (True for real, False for pad)
            spatial_mask: (B, N_patches) boolean tensor
        Returns:
            fused_node_embeddings: (B, N_nodes, embed_dim)
            fused_spatial_features: (B, N_patches, embed_dim)
        """
        # Convert bool mask (True=keep) to PyTorch's MHA key_padding_mask (True=ignore/mask out)
        key_padding_mask_gnn = ~gnn_mask if gnn_mask is not None else None
        key_padding_mask_spatial = ~spatial_mask if spatial_mask is not None else None
        
        h_nodes = gnn_embeddings
        h_spatial = spatial_features
        
        for layer in self.layers:
            h_nodes, h_spatial = layer(
                h_nodes, h_spatial,
                key_padding_mask_gnn, key_padding_mask_spatial
            )
            
        return h_nodes, h_spatial


class FusionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        # Node-to-Spatial (Nodes attend to Spatial)
        self.norm_nodes = nn.LayerNorm(dim)
        self.attn_nodes_to_spatial = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_nodes = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_nodes_2 = nn.LayerNorm(dim)
        
        # Spatial-to-Node (Spatial attends to Nodes)
        self.norm_spatial = nn.LayerNorm(dim)
        self.attn_spatial_to_nodes = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_spatial = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_spatial_2 = nn.LayerNorm(dim)

    def forward(
        self,
        nodes,
        spatial,
        key_padding_mask_nodes=None,
        key_padding_mask_spatial=None
    ):
        # 1. Nodes attend to Spatial
        nodes_norm = self.norm_nodes(nodes)
        spatial_norm = self.norm_spatial(spatial)
        
        # Node query, Spatial key/value
        nodes_fused, _ = self.attn_nodes_to_spatial(
            query=nodes_norm,
            key=spatial_norm,
            value=spatial_norm,
            key_padding_mask=key_padding_mask_spatial
        )
        nodes = nodes + nodes_fused
        nodes = nodes + self.mlp_nodes(self.norm_nodes_2(nodes))
        
        # 2. Spatial attends to Nodes
        nodes_norm_2 = self.norm_nodes(nodes)
        spatial_norm_2 = self.norm_spatial(spatial)
        
        # Spatial query, Node key/value
        spatial_fused, _ = self.attn_spatial_to_nodes(
            query=spatial_norm_2,
            key=nodes_norm_2,
            value=nodes_norm_2,
            key_padding_mask=key_padding_mask_nodes
        )
        spatial = spatial + spatial_fused
        spatial = spatial + self.mlp_spatial(self.norm_spatial_2(spatial))
        
        return nodes, spatial
