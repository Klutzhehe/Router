"""
visualize_routing_gradio.py
===========================
Gradio-based step-by-step JEPA routing visualizer for Google Colab.

Launches a Gradio app with a public URL (share=True) — no port tunneling needed.

Each step renders a panel showing:
  • Board layout BEFORE routing (with all prior copper visible)
  • JEPA cost heatmap per layer (white overlay = existing copper the model sees)
  • A* occupancy map (what the pathfinder sees as blocked)
  • Board layout AFTER routing
  • Via placement probability map

Inline Colab Usage:
-------------------
    import sys; sys.path.insert(0, '/content/Router')
    import os; os.chdir('/content/Router')

    from scripts.visualize_routing_gradio import launch_gradio_visualizer
    launch_gradio_visualizer(
        checkpoint_path='/content/drive/MyDrive/pcb_router/checkpoints/checkpoint_50000.pt',
        share=True   # gives a public *.gradio.live URL
    )

What the panels reveal:
-----------------------
  JEPA heatmap bright (yellow/white) → model learned to AVOID this region
  JEPA heatmap dark (purple/black)   → model prefers routing here
  White overlaid lines on heatmap    → previously routed copper (channel 10)
  If bright zones match copper lines → JEPA IS correctly avoiding existing traces
  If heatmap ignores copper entirely → model hasn't learned avoidance yet
"""

import os
import sys
import io
import copy
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

# ── allow running from project root ──────────────────────────────────────────
_here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _here not in sys.path:
    sys.path.insert(0, _here)

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.training.trainer import DreamerJEPATrainer, PPOJEPATrainer

# ── Color palette ─────────────────────────────────────────────────────────────
NET_COLORS   = ['#3B82F6','#10B981','#EC4899','#8B5CF6','#06B6D4',
                '#F59E0B','#14B8A6','#6366F1','#A855F7','#F43F5E']
LAYER_COLORS = ['#F43F5E','#06B6D4','#8B5CF6','#F59E0B','#10B981','#EC4899']
BG  = '#111222'
FG  = '#E2E8F0'


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers (identical to wandb script so we can share the logic)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_board_on_ax(ax, env, board_state, highlight_net_id=None, path=None):
    board = env.board
    ax.set_facecolor(BG)
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, color='#1A1C2E', linewidth=0.4, linestyle='--')

    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle((obs.x, obs.y), obs.width, obs.height,
                                        facecolor='#EF4444', alpha=0.2, lw=0, hatch='//'))
    for ko in board.keep_out_zones:
        ax.add_patch(patches.Rectangle((ko.x, ko.y), ko.width, ko.height,
                                        edgecolor='#F59E0B', facecolor='none', lw=1, alpha=0.5, ls='--'))
    for comp in board.components:
        ax.add_patch(patches.Rectangle((comp.x, comp.y), comp.width, comp.height,
                                        facecolor='#1E2035', edgecolor='#374151', lw=1.2, alpha=0.85))
        ax.text(comp.x + comp.width/2, comp.y + comp.height/2, comp.name,
                color='#9CA3AF', fontsize=7, ha='center', va='center')

    for seg in board_state.traces:
        col = LAYER_COLORS[seg.layer % len(LAYER_COLORS)]
        lw  = max(1.2, seg.width / board_state.resolution)
        if highlight_net_id is not None and seg.net_id == highlight_net_id:
            col, lw = '#FFFFFF', lw * 1.8
        ax.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                color=col, lw=lw, alpha=0.9, solid_capstyle='round')

    for via in board_state.vias:
        ro = (via.drill_size/2.0 + via.annular_ring) / board_state.resolution
        ri = (via.drill_size/2.0) / board_state.resolution
        ax.add_patch(patches.Circle((via.x, via.y), ro, fc='#EAB308', ec='white', lw=0.4, alpha=0.9))
        ax.add_patch(patches.Circle((via.x, via.y), ri, fc=BG, ec='#EAB308', lw=0.4))

    for pin in board.pins.values():
        col   = NET_COLORS[pin.net_id % len(NET_COLORS)]
        is_hi = (highlight_net_id is not None and pin.net_id == highlight_net_id)
        a, ew, ec = (1.0, 1.5, '#FFF') if is_hi else (0.65, 0.6, '#555570')
        if pin.pad_shape == 0:
            ax.add_patch(patches.Circle((pin.global_x, pin.global_y), 3,
                                        fc=col, ec=ec, lw=ew, alpha=a, zorder=8))
        else:
            ax.add_patch(patches.Rectangle((pin.global_x-3, pin.global_y-3), 6, 6,
                                           fc=col, ec=ec, lw=ew, alpha=a, zorder=8))

    if path:
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#FFF', lw=2.5, alpha=0.8, solid_capstyle='round', zorder=9)
            ax.plot(xs[0],  ys[0],  marker='*', color='#10B981', ms=12, zorder=10)
            ax.plot(xs[-1], ys[-1], marker='s', color='#EF4444', ms=8,  zorder=10)


def _draw_heatmap_on_ax(ax, heatmap, board_state, env, path=None,
                         show_copper=True, title='JEPA Cost Heatmap'):
    board = env.board
    ax.set_facecolor(BG)
    ax.imshow(heatmap, cmap='inferno', origin='lower', alpha=0.92,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])

    if show_copper:
        for seg in board_state.traces:
            ax.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                    color='white', lw=max(1.0, seg.width / board_state.resolution * 0.6),
                    alpha=0.4, solid_capstyle='round', zorder=5)
        for via in board_state.vias:
            r = (via.drill_size/2.0 + via.annular_ring) / board_state.resolution
            ax.add_patch(patches.Circle((via.x, via.y), r, fc='none', ec='white', lw=1.0, alpha=0.35, zorder=5))

    for pin in board.pins.values():
        ax.add_patch(patches.Circle((pin.global_x, pin.global_y), 3.5,
                                    fc='none', ec='white', lw=0.6, alpha=0.45, zorder=6))

    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle((obs.x, obs.y), obs.width, obs.height,
                                        fc='#EF4444', alpha=0.15, lw=0, hatch='//'))

    if board_state.current_net_id is not None:
        net = next((n for n in board.nets if n.id == board_state.current_net_id), None)
        if net and net.pin_ids:
            src  = board.pins.get(net.pin_ids[0])
            tgts = [board.pins.get(p) for p in net.pin_ids[1:] if board.pins.get(p)]
            if src:
                ax.plot(src.global_x, src.global_y, marker='*', color='#10B981',
                        ms=14, zorder=10, mec='white', mew=0.6)
            for tgt in tgts:
                ax.plot(tgt.global_x, tgt.global_y, marker='s', color='#EF4444',
                        ms=9, zorder=10, mec='white', mew=0.6)

    if path:
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#06B6D4', lw=2.0, alpha=0.95, solid_capstyle='round', zorder=9)

    ax.set_title(title, color=FG, fontsize=10, pad=6)


def _draw_occupancy_on_ax(ax, occ, env):
    board = env.board
    ax.set_facecolor(BG)
    ax.imshow(occ, cmap='Greys', origin='lower', vmin=0, vmax=1,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("A* Occupancy\n(white=blocked, includes routed copper)", color=FG, fontsize=9, pad=6)


def _fig_to_pil(fig):
    """Convert matplotlib figure → PIL Image for Gradio."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Routing session state (holds env + history between Gradio interactions)
# ─────────────────────────────────────────────────────────────────────────────

class RoutingSession:
    """Holds one active routing session: env, history, model refs."""

    def __init__(self, trainer, is_dreamer, selected_stage, seed):
        self.trainer      = trainer
        self.is_dreamer   = is_dreamer
        self.stage        = selected_stage
        self.seed         = seed
        self.env          = None
        self.obs          = None
        self.info         = None
        self.h            = None
        self.z            = None
        self.history      = []   # list of step dicts
        self._init_env()

    def _init_env(self):
        board_config = BoardGenerator.from_curriculum_stage(self.stage)
        board_config.seed = self.seed
        self.env = PCBRoutingEnv(
            board_config=board_config,
            reward_weights=self.stage.get('reward_weights')
        )
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.history.clear()
        if self.is_dreamer:
            self.h, self.z = self.trainer.jepa.initial_state(
                batch_size=1, device=self.trainer.device)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self._init_env()

    @property
    def total_nets(self):
        return len(self.env.board.nets) if self.env else 0

    @property
    def routed_count(self):
        return len(self.env.routed_nets) if self.env else 0

    @property
    def done(self):
        return self.routed_count >= self.total_nets

    def step_one_net(self):
        """Route the next unrouted net. Returns step dict or None if done."""
        if self.done:
            return None

        env = self.env
        unrouted = [n for n in env.board.nets if n.id not in env.routed_nets]
        if not unrouted:
            return None

        net     = unrouted[0]
        net_idx = next(i for i, n in enumerate(env.board.nets) if n.id == net.id)

        raster_t   = torch.tensor(self.obs['board_raster'], dtype=torch.float32).unsqueeze(0)
        layer_mask = torch.tensor(self.obs['layer_mask'],   dtype=torch.float32).unsqueeze(0)

        graph = self.info['graph']
        if hasattr(graph, 'x_dict'):
            x_dict          = {k: v for k, v in graph.x_dict.items()}
            edge_index_dict = {k: v for k, v in graph.edge_index_dict.items()}
        else:
            x_dict          = {k: v['x'] for k, v in graph.items() if isinstance(v, dict) and 'x' in v}
            edge_index_dict = {k: v for k, v in graph.items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}

        board_before = env.board_state.clone()
        board_before.set_current_net(net.id)

        with torch.no_grad():
            if self.is_dreamer:
                ctx = self.trainer.jepa.get_context_embedding(raster_t, x_dict, edge_index_dict, use_target=False)
                ne, _, fs = self.trainer._get_net_embeddings_and_mask(raster_t, x_dict, edge_index_dict)
                hl, _, _ = self.trainer.policy.get_heatmap_latent(
                    ne[0, net_idx].unsqueeze(0), self.h, self.z, deterministic=True)
                hv = self.trainer.decoder(hl, fs, env.H, env.W, active_layers_mask=layer_mask)
                ae = self.trainer.jepa.get_action_embedding(
                    torch.tensor([net_idx], device=self.trainer.device), hl)
                self.h, self.z, _, _ = self.trainer.jepa.rssm_step(self.h, self.z, ctx, ae)
            else:
                sp, _ = self.trainer.vit(raster_t)
                ne2   = self.trainer.gnn(x_dict, edge_index_dict)
                fp, fs = self.trainer.fusion(ne2['pad'].unsqueeze(0), sp)
                nN    = len(env.board.nets)
                nEmbs = torch.zeros((1, nN, self.trainer.vit.embed_dim))
                for ni, n in enumerate(env.board.nets):
                    pi = [i for i, p in enumerate(env.board.pins.values()) if p.net_id == n.id]
                    if pi:
                        nEmbs[0, ni] = fp[0, pi].mean(0)
                hl, _, _ = self.trainer.policy.get_heatmap_latent(
                    nEmbs[0, net_idx].unsqueeze(0), fs.mean(1))
                hv = self.trainer.decoder(hl, fs, env.H, env.W, active_layers_mask=layer_mask)

        heatmaps_np = hv[0, :env.board.num_layers].cpu().numpy()
        via_prob_np = hv[0, 8].cpu().numpy()

        # Occupancy BEFORE routing this net
        occ_map = env.board_state.get_occupancy(0)

        next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
            net_idx, heatmaps_np, via_prob_np)

        board_after = env.board_state.clone()
        path = next_info.get('path', [])

        step = {
            'step_num':    len(self.history) + 1,
            'net':         net,
            'net_id':      net.id,
            'net_name':    net.name,
            'success':     next_info.get('connected', False),
            'reward':      reward,
            'drc':         next_info['drc_violations'],
            'completion':  next_info['completion_rate'],
            'heatmaps_np': heatmaps_np,
            'via_prob_np': via_prob_np,
            'occ_map':     occ_map,
            'board_before': board_before,
            'board_after':  board_after,
            'path':         path,
        }
        self.history.append(step)
        self.obs  = next_obs
        self.info = next_info
        return step


# ─────────────────────────────────────────────────────────────────────────────
#  Render a single step → PIL image (for Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def render_step_image(session: RoutingSession, step_dict: dict, layer_idx: int = 0) -> tuple:
    """
    Build a 2×N panel figure for one routing step.
    Returns (PIL board_panel, PIL heatmap_panel, PIL occupancy_panel, status_text)
    """
    env    = session.env
    net    = step_dict['net']
    before = step_dict['board_before']
    after  = step_dict['board_after']
    heatmaps_np = step_dict['heatmaps_np']
    via_prob_np = step_dict['via_prob_np']
    occ_map     = step_dict['occ_map']
    path        = step_dict['path']
    num_layers  = heatmaps_np.shape[0]
    layer       = min(layer_idx, num_layers - 1)

    # ── Panel A: Board before vs after ────────────────────────────────────────
    fig_board, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig_board.patch.set_facecolor(BG)
    fig_board.suptitle(
        f"Step {step_dict['step_num']} — Net '{net.name}'  "
        f"{'✅ Routed' if step_dict['success'] else '❌ Failed'}  |  "
        f"Reward: {step_dict['reward']:+.2f}  DRC: {step_dict['drc']}  "
        f"Completion: {step_dict['completion']*100:.0f}%",
        color=FG, fontsize=12, fontweight='bold'
    )
    before.set_current_net(net.id)
    _draw_board_on_ax(ax_b, env, before, highlight_net_id=net.id)
    ax_b.set_title(
        f"Board BEFORE (step {step_dict['step_num']})\n"
        "White lines in heatmap = this copper",
        color=FG, fontsize=10, pad=8
    )
    _draw_board_on_ax(ax_a, env, after, highlight_net_id=net.id, path=path)
    ax_a.set_title(
        f"Board AFTER routing '{net.name}'",
        color=FG, fontsize=10, pad=8
    )
    plt.tight_layout(pad=1.5)
    board_pil = _fig_to_pil(fig_board)
    plt.close(fig_board)

    # ── Panel B: JEPA heatmap (all layers) + occupancy ────────────────────────
    num_cols = num_layers + 1  # layers + occupancy
    fig_heat, axes = plt.subplots(1, num_cols, figsize=(num_cols * 5, 5), dpi=100)
    fig_heat.patch.set_facecolor(BG)
    if num_cols == 1:
        axes = [axes]

    for li in range(num_layers):
        before.set_current_net(net.id)
        _draw_heatmap_on_ax(
            axes[li], heatmaps_np[li], before, env,
            path=path, show_copper=True,
            title=(f"JEPA Cost — Layer {li}\n"
                   "(bright=avoid, dark=prefer, white=existing copper)")
        )

    _draw_occupancy_on_ax(axes[-1], occ_map, env)
    plt.tight_layout(pad=1.5)
    heat_pil = _fig_to_pil(fig_heat)
    plt.close(fig_heat)

    # ── Panel C: Via probability ───────────────────────────────────────────────
    fig_via, ax_via = plt.subplots(figsize=(5, 5), dpi=100)
    fig_via.patch.set_facecolor(BG)
    ax_via.set_facecolor(BG)
    ax_via.imshow(via_prob_np, cmap='viridis', origin='lower',
                  extent=[0, env.board.width, 0, env.board.height])
    ax_via.set_xlim(0, env.board.width); ax_via.set_ylim(0, env.board.height)
    ax_via.set_aspect('equal'); ax_via.set_xticks([]); ax_via.set_yticks([])
    ax_via.set_title("Via Placement Confidence", color=FG, fontsize=11, pad=8)
    plt.tight_layout()
    via_pil = _fig_to_pil(fig_via)
    plt.close(fig_via)

    status = (
        f"**Step {step_dict['step_num']}** — Net `{net.name}` (id={net.id})\n\n"
        f"{'✅ Routed successfully' if step_dict['success'] else '❌ Routing failed'}\n\n"
        f"- Reward: `{step_dict['reward']:+.2f}`\n"
        f"- DRC violations total: `{step_dict['drc']}`\n"
        f"- Completion: `{step_dict['completion']*100:.0f}%`\n"
        f"- Nets routed: `{session.routed_count}` / `{session.total_nets}`"
    )
    return board_pil, heat_pil, via_pil, status


# ─────────────────────────────────────────────────────────────────────────────
#  History panel (all steps as thumbnails)
# ─────────────────────────────────────────────────────────────────────────────

def render_history_grid(session: RoutingSession):
    """Render a compact grid showing all routed nets so far."""
    hist = session.history
    if not hist:
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No steps yet', color=FG, ha='center', va='center', fontsize=12)
        pil = _fig_to_pil(fig)
        plt.close(fig)
        return pil

    cols = min(len(hist), 6)
    rows = int(np.ceil(len(hist) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5), dpi=80)
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if len(hist) > 1 else [axes]

    for i, step in enumerate(hist):
        ax = axes_flat[i]
        env = session.env
        _draw_board_on_ax(ax, env, step['board_after'], highlight_net_id=step['net_id'])
        status_icon = '✅' if step['success'] else '❌'
        ax.set_title(
            f"{status_icon} Step {step['step_num']}: {step['net_name']}\n"
            f"R={step['reward']:+.2f}  DRC={step['drc']}",
            color=FG, fontsize=8, pad=4
        )

    for i in range(len(hist), len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(pad=0.8)
    pil = _fig_to_pil(fig)
    plt.close(fig)
    return pil


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio app builder
# ─────────────────────────────────────────────────────────────────────────────

def build_gradio_app(trainer, is_dreamer, cur_cfg):
    """Build and return a Gradio Blocks app."""
    import gradio as gr

    stage_names = [s['name'] for s in cur_cfg['stages']]
    stage_map   = {s['name']: s for s in cur_cfg['stages']}

    # Shared mutable session reference
    _session: list = [None]  # use list so closure can rebind

    def _get_or_make_session(stage_name, seed):
        stage = stage_map[stage_name]
        if (_session[0] is None
                or _session[0].stage != stage
                or _session[0].seed != int(seed)):
            _session[0] = RoutingSession(trainer, is_dreamer, stage, int(seed))
        return _session[0]

    # ── Gradio event handlers ──────────────────────────────────────────────────

    def handle_new_board(stage_name, seed):
        stage = stage_map[stage_name]
        _session[0] = RoutingSession(trainer, is_dreamer, stage, int(seed))
        s = _session[0]
        return (
            f"🎲 New board generated — {s.total_nets} nets to route. Press **Route Next Net**.",
            None, None, None, render_history_grid(s),
            gr.update(value=f"0 / {s.total_nets} nets routed")
        )

    def handle_step_one(stage_name, seed, layer_idx):
        s = _get_or_make_session(stage_name, int(seed))
        if s.done:
            return (
                f"✅ All {s.total_nets} nets routed!",
                None, None, None, render_history_grid(s),
                gr.update(value=f"{s.routed_count} / {s.total_nets} nets routed  ✅")
            )
        step = s.step_one_net()
        if step is None:
            return (
                "Nothing to route.",
                None, None, None, render_history_grid(s),
                gr.update(value=f"{s.routed_count} / {s.total_nets} nets routed")
            )
        board_pil, heat_pil, via_pil, status = render_step_image(s, step, int(layer_idx))
        hist_pil = render_history_grid(s)
        return (
            status, board_pil, heat_pil, via_pil, hist_pil,
            gr.update(value=f"{s.routed_count} / {s.total_nets} nets routed")
        )

    def handle_route_all(stage_name, seed, layer_idx):
        s = _get_or_make_session(stage_name, int(seed))
        last_step = None
        while not s.done:
            last_step = s.step_one_net()
            if last_step is None:
                break
        if last_step is None:
            board_pil, heat_pil, via_pil = None, None, None
            status = "Nothing to route."
        else:
            board_pil, heat_pil, via_pil, status = render_step_image(s, last_step, int(layer_idx))
        hist_pil = render_history_grid(s)
        return (
            status, board_pil, heat_pil, via_pil, hist_pil,
            gr.update(value=f"{s.routed_count} / {s.total_nets} nets routed  ✅")
        )

    def handle_view_step(stage_name, seed, layer_idx, step_num):
        s = _get_or_make_session(stage_name, int(seed))
        idx = int(step_num) - 1
        if idx < 0 or idx >= len(s.history):
            return "Invalid step number.", None, None, None, render_history_grid(s), gr.update()
        step = s.history[idx]
        board_pil, heat_pil, via_pil, status = render_step_image(s, step, int(layer_idx))
        return status, board_pil, heat_pil, via_pil, render_history_grid(s), gr.update()

    def handle_live_poll(stage_name, seed, layer_idx, last_step):
        current_timesteps = getattr(trainer, 'total_timesteps', 0)
        if current_timesteps == last_step:
            # Skip updating to prevent flicker/loading when no training progress has been made
            return [gr.skip()] * 6 + [last_step]
            
        s = _get_or_make_session(stage_name, int(seed))
        s.reset()
        
        last_step_dict = None
        while not s.done:
            res = s.step_one_net()
            if res is not None:
                last_step_dict = res
            else:
                break
                
        hist_pil = render_history_grid(s)
        progress_text = f"{s.routed_count} / {s.total_nets} nets routed (Step {current_timesteps:,})"
        
        if last_step_dict is None:
            board_pil, heat_pil, via_pil = None, None, None
            status = f"🎲 Board reset, ready. Step {current_timesteps:,}."
        else:
            board_pil, heat_pil, via_pil, status = render_step_image(s, last_step_dict, int(layer_idx))
            status = f"### 📡 Live Training Update (Step {current_timesteps:,})\n\n" + status
            
        return status, board_pil, heat_pil, via_pil, hist_pil, progress_text, current_timesteps

    # ── Gradio Blocks UI ──────────────────────────────────────────────────────
    with gr.Blocks(
        title="JEPA PCB Router — Step Visualizer",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { background: #0D0F1A !important; }
        .prose h1, .prose h2, .prose h3 { color: #E2E8F0; }
        .label-wrap span { color: #94A3B8; }
        """
    ) as demo:

        last_rendered_step = gr.State(value=-1)
        timer = gr.Timer(value=3, active=False)

        gr.Markdown("""
        # ⚡ JEPA PCB Router — Step-by-Step Routing Visualizer
        Watch the JEPA world model plan routes **net-by-net**, seeing how it responds to previously placed copper.

        > **How to read the heatmap:**
        > - 🟡 **Bright (yellow/white)** = JEPA learned to *avoid* this area
        > - 🟣 **Dark (purple/black)** = JEPA *prefers* routing here
        > - ⬜ **White overlay lines** = previously routed copper (channel 10 of the board raster)
        > - If bright zones correlate with white lines → JEPA **is** avoiding existing traces ✅
        > - If heatmap looks the same regardless → model hasn't learned avoidance yet ❌
        """)

        with gr.Row():
            with gr.Column(scale=1):
                stage_dd   = gr.Dropdown(stage_names, value=stage_names[0],
                                          label="Curriculum Stage")
                seed_num   = gr.Number(value=42, label="Board Seed", precision=0)
                layer_dd   = gr.Dropdown([0, 1, 2, 3], value=0, label="Heatmap Layer to Show")
                progress   = gr.Textbox(value="0 / ? nets routed", label="Progress",
                                         interactive=False)
                
                live_monitor = gr.Checkbox(
                    label="📡 Live Training Monitor (Auto-refresh on training progress)", 
                    value=False
                )

                with gr.Row():
                    btn_new  = gr.Button("🔄 New Board",     variant="secondary")
                    btn_step = gr.Button("➡ Route Next Net", variant="primary")

                with gr.Row():
                    btn_all  = gr.Button("▶ Route All Nets", variant="primary")

                step_view_num = gr.Number(value=1, label="View Specific Step #", precision=0)
                btn_view      = gr.Button("🔍 View This Step")

            with gr.Column(scale=3):
                status_md = gr.Markdown("Press **New Board** then **Route Next Net** to begin.")
                board_img = gr.Image(label="Board Layout (Before → After)", type="pil",
                                      height=400, show_label=True)
                heat_img  = gr.Image(label="JEPA Cost Heatmap + Occupancy", type="pil",
                                      height=300, show_label=True)
                via_img   = gr.Image(label="Via Placement Probability", type="pil",
                                      height=200, show_label=True)

        gr.Markdown("### 📋 Routing History (all steps)")
        hist_img = gr.Image(label="All Routing Steps", type="pil",
                             height=350, show_label=False)

        # ── Wire events ──────────────────────────────────────────────────────
        outputs = [status_md, board_img, heat_img, via_img, hist_img, progress]

        btn_new.click(handle_new_board, inputs=[stage_dd, seed_num], outputs=outputs)
        btn_step.click(handle_step_one, inputs=[stage_dd, seed_num, layer_dd], outputs=outputs)
        btn_all.click(handle_route_all, inputs=[stage_dd, seed_num, layer_dd], outputs=outputs)
        btn_view.click(handle_view_step,
                       inputs=[stage_dd, seed_num, layer_dd, step_view_num],
                       outputs=outputs)

        # Live monitor timer toggle & tick
        live_monitor.change(
            lambda active: gr.Timer(active=active), 
            inputs=[live_monitor], 
            outputs=[timer]
        )
        
        timer.tick(
            handle_live_poll,
            inputs=[stage_dd, seed_num, layer_dd, last_rendered_step],
            outputs=outputs + [last_rendered_step]
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def launch_gradio_visualizer(
    checkpoint_path: str = None,
    stage: str = None,
    share: bool = True,
    server_port: int = 7860,
    curriculum_config_path: str = 'configs/curriculum.yaml',
):
    """
    Load model + launch Gradio app.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to .pt checkpoint. None = untrained random model.
    stage : str or None
        Default curriculum stage to preselect (cosmetic only).
    share : bool
        True = generate a public *.gradio.live URL (works from Colab).
    server_port : int
        Local port (only matters when share=False).
    curriculum_config_path : str
        Path to curriculum YAML.
    """
    with open(curriculum_config_path, 'r') as f:
        cur_cfg = yaml.safe_load(f)

    # Load model
    try:
        trainer = DreamerJEPATrainer(device='cpu', load_checkpoint_path=checkpoint_path)
        is_dreamer = True
        print("✅ Loaded DreamerJEPA model")
    except Exception as e:
        print(f"DreamerJEPA failed ({e}), using PPOJEPATrainer…")
        trainer = PPOJEPATrainer(device='cpu', load_checkpoint_path=checkpoint_path)
        is_dreamer = False
        print("✅ Loaded PPO-JEPA model")

    demo = build_gradio_app(trainer, is_dreamer, cur_cfg)
    print("\nLaunching Gradio app…")
    demo.launch(share=share, server_port=server_port, quiet=False)
    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--stage',      default='multi_net_single_layer')
    parser.add_argument('--share',      action='store_true', default=True)
    parser.add_argument('--port',       type=int, default=7860)
    args = parser.parse_args()

    launch_gradio_visualizer(
        checkpoint_path=args.checkpoint,
        stage=args.stage,
        share=args.share,
        server_port=args.port,
    )
