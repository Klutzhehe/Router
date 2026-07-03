import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.visualization.renderer import BoardRenderer
from pcb_router.visualization.heatmap_viz import HeatmapVisualizer
from pcb_router.training.trainer import DreamerJEPATrainer, PPOJEPATrainer

st.set_page_config(layout="wide", page_title="AI PCB Router Dashboard")

st.title("⚡ AI PCB Router — GNN + JEPA World Model Visualization")

# Load configs
with open('configs/model.yaml', 'r') as f:
    model_cfg = yaml.safe_load(f)
with open('configs/training.yaml', 'r') as f:
    train_cfg = yaml.safe_load(f)
with open('configs/curriculum.yaml', 'r') as f:
    cur_cfg = yaml.safe_load(f)

# Sidebar
st.sidebar.header("Control Panel")
stage_names = [s['name'] for s in cur_cfg['stages']]
selected_stage_name = st.sidebar.selectbox("Select Curriculum Stage", stage_names)
stage_idx = stage_names.index(selected_stage_name)
selected_stage = cur_cfg['stages'][stage_idx]

# Checkpoint path
ckpt_dir = train_cfg['checkpoint']['save_dir']
ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')] if os.path.exists(ckpt_dir) else []
ckpts = sorted(ckpts)

selected_ckpt = st.sidebar.selectbox("Select Model Checkpoint", ["Random (Untrained)"] + ckpts)

# Instantiate models and env
@st.cache_resource
def load_trainer(ckpt_name):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name) if ckpt_name != "Random (Untrained)" else None
    try:
        trainer = DreamerJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        is_dreamer = True
    except Exception as e:
        trainer = PPOJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        is_dreamer = False
    return trainer, is_dreamer

trainer, is_dreamer = load_trainer(selected_ckpt)
renderer = BoardRenderer()
viz = HeatmapVisualizer()

# Generator trigger
if st.sidebar.button("Generate New Board"):
    st.session_state.board_seed = np.random.randint(10000)
    if 'h' in st.session_state:
        del st.session_state.h
    if 'z' in st.session_state:
        del st.session_state.z
if 'board_seed' not in st.session_state:
    st.session_state.board_seed = 42

# Setup Env
board_config = BoardGenerator.from_curriculum_stage(selected_stage)
board_config.seed = st.session_state.board_seed
env = PCBRoutingEnv(board_config=board_config, reward_weights=selected_stage.get('reward_weights'))
obs, info = env.reset(seed=st.session_state.board_seed)

# Initialize RSSM state for Dreamer
if is_dreamer and ('h' not in st.session_state or 'z' not in st.session_state):
    h, z = trainer.jepa.initial_state(batch_size=1, device=trainer.device)
    st.session_state.h = h
    st.session_state.z = z

st.header(f"Curriculum Stage: **{selected_stage_name.replace('_', ' ').title()}**")
st.write(selected_stage['description'])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Board Layout")
    # Show initial state
    fig = renderer.render_board(env.board_state, env.board, show_all_layers=True)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Action Details")
    # List unrouted nets
    unrouted = [n for n in env.board.nets if n.id not in env.routed_nets]
    if unrouted:
        st.write(f"Total Nets: **{len(env.board.nets)}** | Unrouted Nets: **{len(unrouted)}**")
        
        selected_net_to_route = st.selectbox(
            "Select Net to Route",
            options=[n.id for n in unrouted],
            format_func=lambda x: next(n.name for n in env.board.nets if n.id == x)
        )
        
        # Trigger routing action
        if st.button("Route Selected Net"):
            net_idx = next(idx for idx, n in enumerate(env.board.nets) if n.id == selected_net_to_route)
            
            # Predict heatmap
            # Prepare observation inputs
            raster_t = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0)
            layer_mask = torch.tensor(obs['layer_mask'], dtype=torch.float32).unsqueeze(0)
            
            x_dict = {k: v for k, v in info['graph'].x_dict.items()} if hasattr(info['graph'], 'x_dict') else {k: v['x'] for k, v in info['graph'].items() if isinstance(v, dict) and 'x' in v}
            edge_index_dict = {k: v for k, v in info['graph'].edge_index_dict.items()} if hasattr(info['graph'], 'edge_index_dict') else {k: v for k, v in info['graph'].items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}
            
            with torch.no_grad():
                if is_dreamer:
                    context_emb = trainer.jepa.get_context_embedding(raster_t, x_dict, edge_index_dict, use_target=False)
                    net_embs, umask, fs = trainer._get_net_embeddings_and_mask(raster_t, x_dict, edge_index_dict)
                    selected_net_emb = net_embs[0, net_idx].unsqueeze(0)
                    
                    h = st.session_state.h
                    z = st.session_state.z
                    
                    heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                        selected_net_emb, h, z, deterministic=True
                    )
                    heatmaps_via = trainer.decoder(
                        heatmap_latent, fs,
                        env.H, env.W, active_layers_mask=layer_mask
                    )
                    
                    action_emb = trainer.jepa.get_action_embedding(
                        torch.tensor([net_idx], device=trainer.device),
                        heatmap_latent
                    )
                    h, z, _, _ = trainer.jepa.rssm_step(h, z, context_emb, action_emb)
                    st.session_state.h = h
                    st.session_state.z = z
                else:
                    spat_patches, cls_spat = trainer.vit(raster_t)
                    node_embs = trainer.gnn(x_dict, edge_index_dict)
                    f_pads, f_spat = trainer.fusion(node_embs['pad'].unsqueeze(0), spat_patches)
                    
                    # Mean pool pads to net
                    num_nets = len(env.board.nets)
                    net_embs = torch.zeros((1, num_nets, trainer.vit.embed_dim))
                    for n_i, net in enumerate(env.board.nets):
                        pin_indices = [idx for idx, p in enumerate(env.board.pins.values()) if p.net_id == net.id]
                        if pin_indices:
                            net_embs[0, n_i] = f_pads[0, pin_indices].mean(dim=0)
                            
                    unrouted_mask = torch.zeros((1, num_nets), dtype=torch.bool)
                    for n_i, net in enumerate(env.board.nets):
                        if net.id not in env.routed_nets:
                            unrouted_mask[0, n_i] = True
                            
                    # Policy heatmap latent output
                    heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                        net_embs[0, net_idx].unsqueeze(0), f_spat.mean(dim=1)
                    )
                    
                    # CNN decoder outputs
                    heatmaps_via = trainer.decoder(
                        heatmap_latent, f_spat,
                        env.H, env.W, active_layers_mask=layer_mask
                    )
                
            heatmaps_np = heatmaps_via[0, :env.board.num_layers].cpu().numpy()
            via_prob_np = heatmaps_via[0, 8].cpu().numpy()
            
            # Step environment with simulated values
            next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
                net_idx, heatmaps_np, via_prob_np
            )
            
            # Display heatmaps
            st.subheader("Decoded Planning Heatmap")
            h_fig = viz.render_heatmap(heatmaps_np[0], env.board_state, title="Current Layer Cost Heatmap")
            st.pyplot(h_fig)
            plt.close(h_fig)
            
            # Show update results
            st.success(f"Successfully routed Net {selected_net_to_route}! Reward: **{reward:.2f}**")
            st.write(f"New DRC Violations: **{next_info['drc_violations']}**")
            
            # Update state variables
            obs = next_obs
            info = next_info
    else:
        st.balloons()
        st.success("All Nets are routed successfully! Board fully routed.")
