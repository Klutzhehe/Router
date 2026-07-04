import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import os
import copy
import time

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.visualization.renderer import BoardRenderer
from pcb_router.visualization.heatmap_viz import HeatmapVisualizer
from pcb_router.training.trainer import DreamerJEPATrainer, PPOJEPATrainer

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="PCB Router Training",
    page_icon="🔲",
)

st.markdown("""
<style>
  /* Dark premium background */
  html, body, [data-testid="stAppViewContainer"] {
      background: #0A0B14;
      color: #E2E8F0;
  }
  [data-testid="stSidebar"] {
      background: #0E0F1E;
      border-right: 1px solid #1E2035;
  }
  /* Title styling */
  .main-title {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #6366F1 0%, #06B6D4 50%, #10B981 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 2px;
  }
  .main-subtitle {
      color: #64748B;
      font-size: 0.88rem;
      margin-top: 0;
  }
  /* Full board container */
  .full-board-banner {
      background: linear-gradient(135deg, #0F1629 0%, #111A2E 100%);
      border: 1px solid #6366F1;
      border-radius: 12px;
      padding: 18px 20px 12px 20px;
      margin-bottom: 18px;
  }
  .full-board-banner h3 {
      color: #A5B4FC !important;
      font-size: 1.1rem !important;
      margin-bottom: 4px !important;
  }
  /* Step card */
  .step-card {
      background: #161828;
      border: 1px solid #2A2D45;
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 8px;
      font-size: 13px;
      transition: border-color 0.2s;
  }
  .step-card.active {
      border-color: #6366F1;
      box-shadow: 0 0 0 1px #6366F1;
  }
  .step-card.success { border-left: 4px solid #10B981; }
  .step-card.fail    { border-left: 4px solid #EF4444; }
  .metric-pill {
      display: inline-block;
      background: #1E2035;
      border-radius: 99px;
      padding: 3px 10px;
      font-size: 12px;
      margin: 2px 2px;
  }
  .legend-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
  }
  .panel-label {
      color: #94A3B8;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
  }
  h1, h2, h3 { color: #E2E8F0 !important; }
  [data-testid="metric-container"] {
      background: #161828;
      border: 1px solid #2A2D45;
      border-radius: 8px;
      padding: 8px 12px;
  }
  /* Progress tracker */
  .progress-track {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      margin: 8px 0;
  }
  .progress-dot {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">🔲 PCB Router Training</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Visualize how the JEPA model routes each net — net-by-net — building the full PCB copper layout step by step.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Load configs
# ─────────────────────────────────────────────────────────
with open('configs/model.yaml', 'r') as f:
    model_cfg = yaml.safe_load(f)
with open('configs/training.yaml', 'r') as f:
    train_cfg = yaml.safe_load(f)
with open('configs/curriculum.yaml', 'r') as f:
    cur_cfg = yaml.safe_load(f)

# ─────────────────────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Control Panel")

stage_names = [s['name'] for s in cur_cfg['stages']]
selected_stage_name = st.sidebar.selectbox(
    "Curriculum Stage",
    stage_names,
    format_func=lambda n: n.replace('_', ' ').title()
)
stage_idx  = stage_names.index(selected_stage_name)
selected_stage = cur_cfg['stages'][stage_idx]

st.sidebar.markdown("---")

ckpt_dir = train_cfg['checkpoint']['save_dir']
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]) if os.path.exists(ckpt_dir) else []
selected_ckpt = st.sidebar.selectbox("Model Checkpoint", ["Random (Untrained)"] + ckpts)

st.sidebar.markdown("---")

board_seed = st.sidebar.number_input("Board Seed", value=42, step=1)
show_heatmap_layer = st.sidebar.selectbox("Heatmap Layer to Show", [0, 1, 2, 3], index=0,
                                           format_func=lambda l: f"Layer {l}")
overlay_existing = st.sidebar.checkbox("Overlay Existing Copper on Heatmap", value=True)
show_occupancy   = st.sidebar.checkbox("Show A* Occupancy Map", value=False)
show_heatmap     = st.sidebar.checkbox("Show JEPA Heatmap Panel", value=True)

# ─────────────────────────────────────────────────────────
#  Load model (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_trainer(ckpt_name):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name) if ckpt_name != "Random (Untrained)" else None
    try:
        trainer = DreamerJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        return trainer, True
    except Exception:
        trainer = PPOJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        return trainer, False

trainer, is_dreamer = load_trainer(selected_ckpt)
renderer   = BoardRenderer(theme_dark=True)
hmap_viz   = HeatmapVisualizer(theme_dark=True)

NET_COLORS = [
    '#3B82F6','#10B981','#EC4899','#8B5CF6','#06B6D4',
    '#F59E0B','#14B8A6','#6366F1','#A855F7','#F43F5E',
]

# ─────────────────────────────────────────────────────────
#  Session-state helpers
# ─────────────────────────────────────────────────────────

def _fresh_state_key():
    return f"state_{selected_stage_name}_{board_seed}_{selected_ckpt}"

def reset_routing_session():
    """Build a brand-new env + routing history in session_state."""
    board_config = BoardGenerator.from_curriculum_stage(selected_stage)
    board_config.seed = int(board_seed)
    env = PCBRoutingEnv(board_config=board_config, reward_weights=selected_stage.get('reward_weights'))
    obs, info = env.reset(seed=int(board_seed))

    h_state, z_state = None, None
    if is_dreamer:
        h_state, z_state = trainer.jepa.initial_state(batch_size=1, device=trainer.device)

    st.session_state['routing_env']      = env
    st.session_state['routing_obs']      = obs
    st.session_state['routing_info']     = info
    st.session_state['routing_h']        = h_state
    st.session_state['routing_z']        = z_state
    st.session_state['routing_history']  = []   # list of step dicts
    st.session_state['routing_step']     = 0    # index of current displayed step
    st.session_state['routing_done']     = False
    st.session_state['state_key']        = _fresh_state_key()


# Reset if stage / seed / checkpoint changed
if st.session_state.get('state_key') != _fresh_state_key():
    reset_routing_session()

# ─────────────────────────────────────────────────────────
#  JEPA inference helper — run one routing step
# ─────────────────────────────────────────────────────────

def run_one_step():
    """Route the next unrouted net using JEPA and record the step."""
    env   = st.session_state['routing_env']
    obs   = st.session_state['routing_obs']
    info  = st.session_state['routing_info']
    h     = st.session_state['routing_h']
    z     = st.session_state['routing_z']

    unrouted = [n for n in env.board.nets if n.id not in env.routed_nets]
    if not unrouted:
        st.session_state['routing_done'] = True
        return

    # Pick next net (sequential ordering for determinism in viz)
    net   = unrouted[0]
    net_idx = next(i for i, n in enumerate(env.board.nets) if n.id == net.id)
    net_id  = net.id

    raster_t   = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0)
    layer_mask = torch.tensor(obs['layer_mask'],   dtype=torch.float32).unsqueeze(0)

    # Build graph dicts
    graph = info['graph']
    if hasattr(graph, 'x_dict'):
        x_dict         = {k: v for k, v in graph.x_dict.items()}
        edge_index_dict = {k: v for k, v in graph.edge_index_dict.items()}
    else:
        x_dict         = {k: v['x'] for k, v in graph.items() if isinstance(v, dict) and 'x' in v}
        edge_index_dict = {k: v for k, v in graph.items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}

    # Snapshot board BEFORE routing this net
    board_state_before = env.board_state.clone()

    with torch.no_grad():
        if is_dreamer:
            context_emb = trainer.jepa.get_context_embedding(raster_t, x_dict, edge_index_dict, use_target=False)
            net_embs, _, fs = trainer._get_net_embeddings_and_mask(raster_t, x_dict, edge_index_dict)
            selected_net_emb = net_embs[0, net_idx].unsqueeze(0)

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
            new_h, new_z, _, _ = trainer.jepa.rssm_step(h, z, context_emb, action_emb)
        else:
            spat_patches, _ = trainer.vit(raster_t)
            node_embs       = trainer.gnn(x_dict, edge_index_dict)
            f_pads, f_spat  = trainer.fusion(node_embs['pad'].unsqueeze(0), spat_patches)

            num_nets_total = len(env.board.nets)
            net_embs_ppo   = torch.zeros((1, num_nets_total, trainer.vit.embed_dim))
            for n_i, n in enumerate(env.board.nets):
                pin_indices = [idx for idx, p in enumerate(env.board.pins.values()) if p.net_id == n.id]
                if pin_indices:
                    net_embs_ppo[0, n_i] = f_pads[0, pin_indices].mean(dim=0)

            heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                net_embs_ppo[0, net_idx].unsqueeze(0), f_spat.mean(dim=1)
            )
            heatmaps_via = trainer.decoder(
                heatmap_latent, f_spat,
                env.H, env.W, active_layers_mask=layer_mask
            )
            new_h, new_z = None, None

    heatmaps_np  = heatmaps_via[0, :env.board.num_layers].cpu().numpy()
    via_prob_np  = heatmaps_via[0, 8].cpu().numpy()

    # Step environment
    next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
        net_idx, heatmaps_np, via_prob_np
    )

    # Build occupancy snapshot (post step)
    layer_to_show = min(show_heatmap_layer, env.board.num_layers - 1)
    occupancy_map = env.board_state.get_occupancy(layer_to_show)

    # Snapshot board AFTER routing this net (so we can show cumulative board at this step)
    board_state_after = env.board_state.clone()

    # Record step
    step_dict = {
        'step_num':          len(st.session_state['routing_history']) + 1,
        'net_id':            net_id,
        'net_name':          net.name,
        'net_idx':           net_idx,
        'success':           next_info.get('connected', False),
        'reward':            reward,
        'drc_violations':    next_info['drc_violations'],
        'completion':        next_info['completion_rate'],
        'path':              next_info.get('path', []),
        'heatmaps_np':       heatmaps_np,      # (num_layers, H, W)
        'via_prob_np':       via_prob_np,       # (H, W)
        'occupancy_map':     occupancy_map,     # (H, W)
        'board_before':      board_state_before,  # BoardState BEFORE this net
        'board_after':       board_state_after,   # BoardState AFTER this net (cumulative)
    }
    st.session_state['routing_history'].append(step_dict)
    st.session_state['routing_step']  = len(st.session_state['routing_history']) - 1
    st.session_state['routing_obs']   = next_obs
    st.session_state['routing_info']  = next_info
    st.session_state['routing_h']     = new_h
    st.session_state['routing_z']     = new_z

    if terminated or truncated or not unrouted[1:]:
        st.session_state['routing_done'] = True

# ─────────────────────────────────────────────────────────
#  Render helpers
# ─────────────────────────────────────────────────────────

def draw_board_state(ax, env, board_state, highlight_net_id=None, path=None,
                     title=None, show_all_bright=False):
    """Draw board (components, traces, pads, vias) onto an existing matplotlib Axes.
    
    Args:
        show_all_bright: If True, render ALL traces at full brightness (for the final full board view).
    """
    board = env.board
    bg    = '#0C0D1A'
    layer_colors = ['#F43F5E','#06B6D4','#8B5CF6','#F59E0B','#10B981','#EC4899']

    ax.set_facecolor(bg)

    # Grid
    ax.set_xlim(0, board.width)
    ax.set_ylim(0, board.height)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color='#181A2E', linewidth=0.4, linestyle='--')

    # Obstacles
    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle(
            (obs.x, obs.y), obs.width, obs.height,
            facecolor='#EF4444', alpha=0.2, linewidth=0, hatch='//'
        ))

    # Keep-out zones
    for ko in board.keep_out_zones:
        ax.add_patch(patches.Rectangle(
            (ko.x, ko.y), ko.width, ko.height,
            edgecolor='#F59E0B', facecolor='none', linewidth=1, alpha=0.5, linestyle='--'
        ))

    # Components
    for comp in board.components:
        ax.add_patch(patches.Rectangle(
            (comp.x, comp.y), comp.width, comp.height,
            facecolor='#1A1D38', edgecolor='#374151', linewidth=1.2, alpha=0.85
        ))
        ax.text(
            comp.x + comp.width / 2, comp.y + comp.height / 2,
            comp.name, color='#9CA3AF', fontsize=7, ha='center', va='center'
        )

    # Traces — colour by layer; highlight the active net in white if stepping
    for seg in board_state.traces:
        net_col = NET_COLORS[seg.net_id % len(NET_COLORS)]
        layer_col = layer_colors[seg.layer % len(layer_colors)]
        lw    = max(1.5, seg.width / board_state.resolution)

        if show_all_bright:
            # Full board: draw each trace in its net colour, full brightness
            col   = net_col
            alpha = 1.0
        elif highlight_net_id is not None and seg.net_id == highlight_net_id:
            col   = '#FFFFFF'
            lw    = lw * 1.8
            alpha = 1.0
        else:
            col   = layer_col
            alpha = 0.55  # slightly dim non-active nets during step view

        ax.plot(
            [seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
            color=col, linewidth=lw, alpha=alpha, solid_capstyle='round'
        )

    # Vias
    for via in board_state.vias:
        r_outer = (via.drill_size / 2.0 + via.annular_ring) / board_state.resolution
        r_inner = (via.drill_size / 2.0) / board_state.resolution
        ax.add_patch(patches.Circle((via.x, via.y), r_outer, facecolor='#EAB308', edgecolor='#fff', linewidth=0.4, alpha=0.9))
        ax.add_patch(patches.Circle((via.x, via.y), r_inner, facecolor=bg,        edgecolor='#EAB308', linewidth=0.4))

    # Pads
    for pin in board.pins.values():
        col    = NET_COLORS[pin.net_id % len(NET_COLORS)]
        radius = 3
        is_cur = (highlight_net_id is not None and pin.net_id == highlight_net_id)
        alpha  = 1.0 if (is_cur or show_all_bright) else 0.65
        ew     = 1.5 if is_cur else 0.6
        ec     = '#FFFFFF' if is_cur else '#555570'
        if pin.pad_shape == 0:
            ax.add_patch(patches.Circle((pin.global_x, pin.global_y), radius=radius,
                                        facecolor=col, edgecolor=ec, linewidth=ew, alpha=alpha, zorder=8))
        else:
            ax.add_patch(patches.Rectangle((pin.global_x - 3, pin.global_y - 3), 6, 6,
                                           facecolor=col, edgecolor=ec, linewidth=ew, alpha=alpha, zorder=8))

    # Draw unrouted/failed nets as light dotted red lines
    unrouted_nets = [net for net in board.nets if net.id not in board_state.routed_net_ids]
    for net in unrouted_nets:
        pins = [board.pins[pid] for pid in net.pin_ids if pid in board.pins]
        for idx in range(len(pins) - 1):
            p1, p2 = pins[idx], pins[idx+1]
            ax.plot([p1.global_x, p2.global_x], [p1.global_y, p2.global_y],
                    color='#EF4444', linestyle=':', linewidth=1.2, alpha=0.6)

    # Draw routed path for the active net (current step only)
    if path and not show_all_bright:
        layer_to_show = min(show_heatmap_layer, env.board.num_layers - 1)
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == layer_to_show]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#FFFFFF', linewidth=2.5, alpha=0.85,
                    solid_capstyle='round', zorder=9)
            ax.plot(xs[0], ys[0], marker='*', color='#10B981', markersize=10, zorder=10)
            ax.plot(xs[-1], ys[-1], marker='s', color='#EF4444', markersize=8, zorder=10)

    if title:
        ax.set_title(title, color='#E2E8F0', fontsize=12, pad=10)


def draw_heatmap_panel(ax, heatmap, board_state, env, path=None,
                        overlay_copper=True, title="JEPA Cost Heatmap"):
    """Draw JEPA heatmap with optional copper overlay and path."""
    board = env.board
    bg    = '#0C0D1A'

    ax.set_facecolor(bg)
    ax.imshow(heatmap, cmap='inferno', origin='lower', alpha=0.92,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width)
    ax.set_ylim(0, board.height)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if overlay_copper:
        for seg in board_state.traces:
            ax.plot(
                [seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                color='#FFFFFF', linewidth=max(1.0, seg.width / board_state.resolution * 0.6),
                alpha=0.45, solid_capstyle='round', zorder=5
            )
        for via in board_state.vias:
            r = (via.drill_size / 2.0 + via.annular_ring) / board_state.resolution
            ax.add_patch(patches.Circle((via.x, via.y), radius=r,
                                        facecolor='none', edgecolor='#FFFFFF',
                                        linewidth=1.0, alpha=0.4, zorder=5))

    for pin in board.pins.values():
        ax.add_patch(patches.Circle((pin.global_x, pin.global_y), radius=3.5,
                                    facecolor='none', edgecolor='#FFFFFF',
                                    linewidth=0.6, alpha=0.5, zorder=6))

    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle(
            (obs.x, obs.y), obs.width, obs.height,
            facecolor='#EF4444', alpha=0.18, linewidth=0, hatch='//'))

    if board_state.current_net_id is not None:
        net = next((n for n in board.nets if n.id == board_state.current_net_id), None)
        if net and net.pin_ids:
            src  = board.pins.get(net.pin_ids[0])
            tgts = [board.pins.get(pid) for pid in net.pin_ids[1:] if board.pins.get(pid)]
            if src:
                ax.plot(src.global_x, src.global_y, marker='*', color='#10B981',
                        markersize=14, zorder=10, markeredgecolor='white', markeredgewidth=0.6)
            for tgt in tgts:
                ax.plot(tgt.global_x, tgt.global_y, marker='s', color='#EF4444',
                        markersize=9, zorder=10, markeredgecolor='white', markeredgewidth=0.6)

    if path:
        layer_to_show = min(show_heatmap_layer, env.board.num_layers - 1)
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == layer_to_show]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#06B6D4', linewidth=2.0, alpha=0.95,
                    solid_capstyle='round', zorder=9)

    ax.set_title(title, color='#E2E8F0', fontsize=11, pad=8)


def draw_occupancy_panel(ax, occ_map, env, board_state):
    """Draw A* occupancy map (white = blocked, black = free)."""
    board = env.board
    ax.set_facecolor('#0C0D1A')
    ax.imshow(occ_map, cmap='Greys', origin='lower', vmin=0, vmax=1,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width)
    ax.set_ylim(0, board.height)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A* Occupancy (white=blocked)", color='#E2E8F0', fontsize=11, pad=8)


# ─────────────────────────────────────────────────────────
#  Status bar
# ─────────────────────────────────────────────────────────

env  = st.session_state['routing_env']
hist = st.session_state['routing_history']
done = st.session_state['routing_done']

total_nets   = len(env.board.nets)
routed_count = len(env.routed_nets)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Curriculum Stage", selected_stage_name.replace('_', ' ').title())
mc2.metric("Total Nets", total_nets)
mc3.metric("Routed So Far", routed_count)
mc4.metric("Completion", f"{routed_count / max(total_nets,1)*100:.0f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────
#  Control buttons
# ─────────────────────────────────────────────────────────

ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 1, 2])

with ctrl1:
    if st.button("🔄 New Board", use_container_width=True):
        reset_routing_session()
        st.rerun()

with ctrl2:
    step_back_clicked = st.button("⬅ Prev Step", use_container_width=True,
                                   disabled=(len(hist) == 0))

with ctrl3:
    step_fwd_clicked = st.button("➡ Next Step", use_container_width=True,
                                  disabled=(done and routed_count >= total_nets))

with ctrl4:
    autoplay = st.button("▶ Route All", use_container_width=True,
                          disabled=(done and routed_count >= total_nets))

with ctrl5:
    if done or routed_count >= total_nets:
        st.success("✅ All nets routed!" if routed_count >= total_nets else "⚠️ Routing complete (some nets may have failed)")
    else:
        remaining = total_nets - routed_count
        st.info(f"🔌 {remaining} net{'s' if remaining != 1 else ''} remaining — press **Next Step** or **Route All**")

# Handle button actions
if step_back_clicked and len(hist) >= 1:
    st.session_state['routing_step'] = max(0, st.session_state['routing_step'] - 1)

if step_fwd_clicked and not (done and routed_count >= total_nets):
    if st.session_state['routing_step'] < len(hist) - 1:
        st.session_state['routing_step'] = st.session_state['routing_step'] + 1
    elif not done:
        run_one_step()
        st.rerun()

if autoplay and not (done and routed_count >= total_nets):
    progress_bar = st.progress(0, text="Routing all nets…")
    while not st.session_state['routing_done']:
        run_one_step()
        n_done = len(st.session_state['routing_history'])
        progress_bar.progress(min(n_done / total_nets, 1.0),
                               text=f"Routing net {n_done}/{total_nets}…")
        time.sleep(0.05)
    progress_bar.empty()
    st.rerun()

# ─────────────────────────────────────────────────────────
#  Refresh local refs after any button actions
# ─────────────────────────────────────────────────────────

hist = st.session_state['routing_history']
cur_step_idx = st.session_state['routing_step']
done = st.session_state['routing_done']
routed_count = len(env.routed_nets)

# ─────────────────────────────────────────────────────────
#  FULL ROUTED BOARD — shown when routing is done (or any step exists)
# ─────────────────────────────────────────────────────────

if len(hist) > 0:
    final_board_state = env.board_state  # always the most up-to-date full state

    success_count = sum(1 for s in hist if s['success'])
    fail_count    = len(hist) - success_count

    st.markdown('<div class="full-board-banner">', unsafe_allow_html=True)

    if done:
        st.markdown("### 🏁 Fully Routed Board")
        st.caption(f"All {len(hist)} nets attempted — **{success_count} routed** ✅  |  **{fail_count} failed** ❌  |  DRC: {hist[-1]['drc_violations']} violations")
    else:
        st.markdown(f"### 📊 Board State — {routed_count}/{total_nets} Nets Routed")
        st.caption("This shows all traces placed so far. Use **Next Step** below to route more.")

    fig_full, ax_full = plt.subplots(figsize=(14, 7), dpi=110)
    fig_full.patch.set_facecolor('#0A0B14')
    draw_board_state(
        ax_full, env, final_board_state,
        highlight_net_id=None,
        path=None,
        show_all_bright=True,
        title=None
    )
    # Annotate each net's pads with its colour legend
    ax_full.set_title(
        "Complete Routed Board — all copper layers" if done else f"Current Board — {routed_count}/{total_nets} nets placed",
        color='#A5B4FC', fontsize=14, pad=12, fontweight='bold'
    )

    # Add a net colour legend in the corner
    board = env.board
    legend_handles = []
    for i, net in enumerate(board.nets):
        col = NET_COLORS[net.id % len(NET_COLORS)]
        is_routed = net.id in env.routed_nets
        marker = '●' if is_routed else '○'
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=col if is_routed else '#333',
                       markeredgecolor=col,
                       markersize=8, label=f"{marker} {net.name or f'Net {net.id}'}")
        )
    if legend_handles:
        ax_full.legend(
            handles=legend_handles, loc='upper right',
            framealpha=0.7, facecolor='#0E0F1E', edgecolor='#2A2D45',
            labelcolor='#E2E8F0', fontsize=8,
            ncol=max(1, len(legend_handles) // 8 + 1)
        )

    st.pyplot(fig_full, use_container_width=True)
    plt.close(fig_full)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

# ─────────────────────────────────────────────────────────
#  STEP-BY-STEP VIEW — board at current step + heatmap
# ─────────────────────────────────────────────────────────

st.subheader("🔬 Step-by-Step Routing Detail")

if len(hist) == 0:
    st.info("Press **Next Step** or **Route All** above to start routing nets on this board.")
else:
    # Slider to navigate steps
    if len(hist) > 1:
        nav = st.slider(
            "Navigate routing steps",
            min_value=1, max_value=len(hist),
            value=cur_step_idx + 1, step=1,
            key="history_slider",
            help="Drag to replay any routing step"
        )
        if nav - 1 != cur_step_idx:
            st.session_state['routing_step'] = nav - 1
            cur_step_idx = nav - 1

    step = hist[cur_step_idx]

    # --- Net info header ---
    net_color = NET_COLORS[step['net_id'] % len(NET_COLORS)]
    status_label = "Successfully Routed" if step['success'] else "Routing Failed"
    status_icon2 = "✅" if step['success'] else "❌"
    net_name_disp = step['net_name'] or f"Net {step['net_id']}"
    st.markdown(
        f'<div style="background:#161828;border-left:4px solid {net_color};'
        f'border-radius:8px;padding:10px 16px;margin-bottom:12px;">'
        f'<b style="color:{net_color};font-size:1.05rem">Step {step["step_num"]}/{total_nets}'
        f' &mdash; {net_name_disp}</b>&nbsp;&nbsp;'
        f'<span style="color:#94A3B8">{status_icon2} {status_label}</span>'
        f'<br/><span class="metric-pill">Reward: {step["reward"]:+.2f}</span>'
        f'<span class="metric-pill">DRC: {step["drc_violations"]}</span>'
        f'<span class="metric-pill">Completion: {step["completion"]*100:.0f}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Two-column layout: board | heatmap
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<p class="panel-label">Board after this step (cumulative traces)</p>', unsafe_allow_html=True)

        # Use board_after so all traces placed UP TO AND INCLUDING this step are shown
        display_state = step['board_after']

        fig_step, ax_step = plt.subplots(figsize=(7, 7), dpi=100)
        fig_step.patch.set_facecolor('#0A0B14')
        step_title_net = step['net_name'] or f"Net {step['net_id']}"
        step_title_status = 'Routed' if step['success'] else 'Failed'
        draw_board_state(
            ax_step, env, display_state,
            highlight_net_id=step['net_id'],   # highlight the net just routed
            path=step['path'],
            show_all_bright=False,
            title=f"Step {step['step_num']}/{total_nets} — {step_title_net} — {step_title_status}"
        )
        st.pyplot(fig_step, use_container_width=True)
        plt.close(fig_step)

    if show_heatmap:
        with right_col:
            st.markdown('<p class="panel-label">JEPA routing heatmap (before this step)</p>', unsafe_allow_html=True)

            before_state = step['board_before']
            before_state.set_current_net(step['net_id'])

            layer = min(show_heatmap_layer, step['heatmaps_np'].shape[0] - 1)
            num_layers_board = step['heatmaps_np'].shape[0]

            if num_layers_board == 1 and not show_occupancy:
                fig_hm, ax_hm = plt.subplots(figsize=(7, 7), dpi=100)
                fig_hm.patch.set_facecolor('#0A0B14')
                draw_heatmap_panel(
                    ax_hm, step['heatmaps_np'][layer], before_state, env,
                    path=step['path'],
                    overlay_copper=overlay_existing,
                    title=f"JEPA Cost Heatmap — Layer {layer}"
                )
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)
            else:
                extra_cols = 1 if show_occupancy else 0
                n_panels   = num_layers_board + extra_cols
                cols_per_row = min(2, n_panels)
                rows_needed  = int(np.ceil(n_panels / cols_per_row))

                fig_hm, axes = plt.subplots(rows_needed, cols_per_row,
                                             figsize=(cols_per_row * 5, rows_needed * 5), dpi=90)
                fig_hm.patch.set_facecolor('#0A0B14')
                axes_flat = np.array(axes).flatten()

                for li in range(num_layers_board):
                    draw_heatmap_panel(
                        axes_flat[li], step['heatmaps_np'][li], before_state, env,
                        path=step['path'],
                        overlay_copper=overlay_existing,
                        title=f"Layer {li} — JEPA Cost Heatmap"
                    )

                if show_occupancy and num_layers_board < len(axes_flat):
                    draw_occupancy_panel(axes_flat[num_layers_board], step['occupancy_map'], env, before_state)

                for i in range(n_panels, len(axes_flat)):
                    axes_flat[i].axis('off')

                plt.tight_layout(pad=1.5)
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)

            # Via probability map
            st.markdown("**Via Placement Probability Map**")
            fig_via, ax_via = plt.subplots(figsize=(6, 2.5), dpi=90)
            fig_via.patch.set_facecolor('#0A0B14')
            ax_via.set_facecolor('#0A0B14')
            ax_via.imshow(step['via_prob_np'], cmap='viridis', origin='lower',
                          extent=[0, env.board.width, 0, env.board.height])
            ax_via.set_xlim(0, env.board.width)
            ax_via.set_ylim(0, env.board.height)
            ax_via.set_aspect('equal')
            ax_via.set_xticks([]); ax_via.set_yticks([])
            ax_via.set_title("Via Placement Confidence", color='#E2E8F0', fontsize=10, pad=6)
            st.pyplot(fig_via, use_container_width=True)
            plt.close(fig_via)

# ─────────────────────────────────────────────────────────
#  ROUTING HISTORY TIMELINE
# ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Routing History Timeline")

if not hist:
    st.caption("No steps recorded yet. Press **Next Step** or **Route All** to begin.")
else:
    dot_html = '<div class="progress-track">'
    for s in hist:
        col = NET_COLORS[s['net_id'] % len(NET_COLORS)]
        is_active = (s['step_num'] - 1 == cur_step_idx)
        border = f"box-shadow:0 0 0 2px #fff" if is_active else ""
        opacity = "1.0" if s['success'] else "0.4"
        title_str = s['net_name'] or f"Net {s['net_id']}"
        dot_html += (
            f'<span class="progress-dot" title="Step {s["step_num"]}: {title_str}" '
            f'style="background:{col};opacity:{opacity};{border}"></span>'
        )
    dot_html += '</div>'
    st.markdown(dot_html, unsafe_allow_html=True)
    st.caption("🟢 Filled = routed  ◻ Faded = failed  |  Border = currently viewed step")

    st.markdown("")

    cols_per_row = 4
    rows = [hist[i:i+cols_per_row] for i in range(0, len(hist), cols_per_row)]

    for row in rows:
        row_cols = st.columns(cols_per_row)
        for col, s in zip(row_cols, row):
            with col:
                is_active = (s['step_num'] - 1 == cur_step_idx)
                status_icon = "✅" if s['success'] else "❌"
                border_class = "success" if s['success'] else "fail"
                active_class = " active" if is_active else ""

                net_color = NET_COLORS[s['net_id'] % len(NET_COLORS)]
                st.markdown(
                    f"""<div class="step-card {border_class}{active_class}">
                    <span style="color:{net_color};font-weight:700">
                        {status_icon} Step {s['step_num']}
                    </span><br/>
                    <b>{s['net_name'] or f"Net {s['net_id']}"}</b><br/>
                    <span class="metric-pill">R: {s['reward']:+.2f}</span>
                    <span class="metric-pill">DRC: {s['drc_violations']}</span>
                    <span class="metric-pill">{s['completion']*100:.0f}%</span>
                    </div>""",
                    unsafe_allow_html=True
                )
                if st.button(f"View", key=f"view_step_{s['step_num']}"):
                    st.session_state['routing_step'] = s['step_num'] - 1
                    st.rerun()

# ─────────────────────────────────────────────────────────
#  Curriculum stage info
# ─────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("📚 Curriculum Stage Details", expanded=False):
    st.json(selected_stage)
