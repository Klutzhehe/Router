"""
visualize_routing_wandb.py
==========================
Run JEPA step-by-step routing on a single board and log every net's
planning visuals to Weights & Biases.

Each step produces:
  • Board layout BEFORE routing this net  (showing all prior copper)
  • JEPA cost heatmap per layer           (with prior copper overlay)
  • A* occupancy map                      (what the pathfinder sees as blocked)
  • Board layout AFTER routing this net
  • Via placement probability map

All images are logged to a W&B run so you can inspect them in any browser,
even from Colab, without needing to expose a port.

CLI Usage (Terminal / Colab cell via subprocess):
-------------------------------------------------
    python scripts/visualize_routing_wandb.py \\
        --checkpoint checkpoints/checkpoint_50000.pt \\
        --stage multi_net_single_layer \\
        --seed 42 \\
        --wandb_project pcb-router \\
        --num_boards 3

Inline Colab Usage (import and call directly):
----------------------------------------------
    import sys; sys.path.insert(0, '/content/Router')
    from scripts.visualize_routing_wandb import run_visualization

    import argparse
    args = argparse.Namespace(
        checkpoint='/content/drive/MyDrive/pcb_router/checkpoints/checkpoint_50000.pt',
        stage='multi_net_single_layer',
        seed=42,
        num_boards=3,
        wandb_project='pcb-router',
        wandb_run_name=None,
    )
    run_visualization(args)

What each heatmap panel reveals about JEPA's routing decisions:
---------------------------------------------------------------
  • HIGH-cost (bright yellow/white) areas → JEPA learned to avoid these
  • LOW-cost (dark purple/black) areas    → JEPA prefers routing through these
  • White overlay lines on heatmap        → previously routed copper in channel 10
  • If high-cost regions coincide with existing copper → JEPA IS avoiding traces
  • If heatmap looks the same regardless of copper → model hasn't learned yet
"""


import argparse
import os
import sys
import io
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.training.trainer import DreamerJEPATrainer, PPOJEPATrainer

# ── Color palette ─────────────────────────────────────────────────────────────
NET_COLORS = [
    '#3B82F6','#10B981','#EC4899','#8B5CF6','#06B6D4',
    '#F59E0B','#14B8A6','#6366F1','#A855F7','#F43F5E',
]
LAYER_COLORS = ['#F43F5E','#06B6D4','#8B5CF6','#F59E0B','#10B981','#EC4899']
BG_COLOR = '#111222'
TEXT_COLOR = '#E2E8F0'


# ─────────────────────────────────────────────────────────────────────────────
#  Board drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_board_on_ax(ax, env, board_state, highlight_net_id=None, path=None):
    board = env.board
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, color='#1A1C2E', linewidth=0.4, linestyle='--')

    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle(
            (obs.x, obs.y), obs.width, obs.height,
            facecolor='#EF4444', alpha=0.2, linewidth=0, hatch='//'
        ))
    for ko in board.keep_out_zones:
        ax.add_patch(patches.Rectangle(
            (ko.x, ko.y), ko.width, ko.height,
            edgecolor='#F59E0B', facecolor='none', linewidth=1, alpha=0.5, linestyle='--'
        ))
    for comp in board.components:
        ax.add_patch(patches.Rectangle(
            (comp.x, comp.y), comp.width, comp.height,
            facecolor='#1E2035', edgecolor='#374151', linewidth=1.2, alpha=0.85
        ))
        ax.text(comp.x + comp.width / 2, comp.y + comp.height / 2, comp.name,
                color='#9CA3AF', fontsize=7, ha='center', va='center')

    for seg in board_state.traces:
        col = LAYER_COLORS[seg.layer % len(LAYER_COLORS)]
        lw  = max(1.2, seg.width / board_state.resolution)
        if highlight_net_id is not None and seg.net_id == highlight_net_id:
            col, lw = '#FFFFFF', lw * 1.8
        ax.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                color=col, linewidth=lw, alpha=0.9, solid_capstyle='round')

    for via in board_state.vias:
        ro = (via.drill_size / 2.0 + via.annular_ring) / board_state.resolution
        ri = (via.drill_size / 2.0) / board_state.resolution
        ax.add_patch(patches.Circle((via.x, via.y), ro, facecolor='#EAB308', edgecolor='white', lw=0.4, alpha=0.9))
        ax.add_patch(patches.Circle((via.x, via.y), ri, facecolor=BG_COLOR,  edgecolor='#EAB308', lw=0.4))

    for pin in board.pins.values():
        col   = NET_COLORS[pin.net_id % len(NET_COLORS)]
        is_cur = (highlight_net_id is not None and pin.net_id == highlight_net_id)
        alpha, ew, ec = (1.0, 1.5, '#FFF') if is_cur else (0.65, 0.6, '#555570')
        if pin.pad_shape == 0:
            ax.add_patch(patches.Circle((pin.global_x, pin.global_y), 3,
                                        facecolor=col, edgecolor=ec, lw=ew, alpha=alpha, zorder=8))
        else:
            ax.add_patch(patches.Rectangle((pin.global_x-3, pin.global_y-3), 6, 6,
                                           facecolor=col, edgecolor=ec, lw=ew, alpha=alpha, zorder=8))

    if path:
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == 0]  # layer 0 path
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#FFF', linewidth=2.5, alpha=0.8, solid_capstyle='round', zorder=9)
            ax.plot(xs[0],  ys[0],  marker='*', color='#10B981', markersize=12, zorder=10)
            ax.plot(xs[-1], ys[-1], marker='s', color='#EF4444', markersize=8,  zorder=10)


def _draw_heatmap_on_ax(ax, heatmap, board_state, env, path=None,
                         show_existing_copper=True, title='JEPA Cost Heatmap'):
    board = env.board
    ax.set_facecolor(BG_COLOR)
    ax.imshow(heatmap, cmap='inferno', origin='lower', alpha=0.92,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])

    # White overlay of already-routed copper — shows the model seeing prior traces
    if show_existing_copper:
        for seg in board_state.traces:
            ax.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                    color='white', linewidth=max(1.0, seg.width / board_state.resolution * 0.6),
                    alpha=0.4, solid_capstyle='round', zorder=5)
        for via in board_state.vias:
            r = (via.drill_size / 2.0 + via.annular_ring) / board_state.resolution
            ax.add_patch(patches.Circle((via.x, via.y), radius=r,
                                        facecolor='none', edgecolor='white', lw=1.0, alpha=0.35, zorder=5))

    # Pad outlines
    for pin in board.pins.values():
        ax.add_patch(patches.Circle((pin.global_x, pin.global_y), 3.5,
                                    facecolor='none', edgecolor='white', lw=0.6, alpha=0.45, zorder=6))

    # Obstacles
    for obs in board.obstacles:
        ax.add_patch(patches.Rectangle((obs.x, obs.y), obs.width, obs.height,
                                       facecolor='#EF4444', alpha=0.15, linewidth=0, hatch='//'))

    # Source / target markers for current net
    if board_state.current_net_id is not None:
        net = next((n for n in board.nets if n.id == board_state.current_net_id), None)
        if net and net.pin_ids:
            src  = board.pins.get(net.pin_ids[0])
            tgts = [board.pins.get(p) for p in net.pin_ids[1:] if board.pins.get(p)]
            if src:
                ax.plot(src.global_x, src.global_y, marker='*', color='#10B981',
                        markersize=14, zorder=10, markeredgecolor='white', markeredgewidth=0.6)
            for tgt in tgts:
                ax.plot(tgt.global_x, tgt.global_y, marker='s', color='#EF4444',
                        markersize=9, zorder=10, markeredgecolor='white', markeredgewidth=0.6)

    # Planned path
    if path:
        pts = [(wp[0], wp[1]) for wp in path if wp[2] == 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='#06B6D4', linewidth=2.0, alpha=0.95, solid_capstyle='round', zorder=9)

    ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)


def _draw_occupancy_on_ax(ax, occ, env):
    board = env.board
    ax.set_facecolor(BG_COLOR)
    ax.imshow(occ, cmap='Greys', origin='lower', vmin=0, vmax=1,
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("A* Occupancy (white=blocked)", color=TEXT_COLOR, fontsize=11, pad=8)


def _draw_via_prob_on_ax(ax, via_prob, env):
    board = env.board
    ax.set_facecolor(BG_COLOR)
    ax.imshow(via_prob, cmap='viridis', origin='lower',
              extent=[0, board.width, 0, board.height])
    ax.set_xlim(0, board.width);  ax.set_ylim(0, board.height)
    ax.set_aspect('equal');       ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Via Placement Confidence", color=TEXT_COLOR, fontsize=11, pad=8)


# ─────────────────────────────────────────────────────────────────────────────
#  Render one routing step → figure
# ─────────────────────────────────────────────────────────────────────────────

def render_routing_step(
    step_num, net, env,
    board_before, board_after,
    heatmaps_np, via_prob_np, occ_map, path,
    num_layers
) -> plt.Figure:
    """
    Build a 2×(num_layers+1) panel figure for one routing step:
      Row 0: Board-before | Heatmap L0 | Heatmap L1 …
      Row 1: Board-after  | Occupancy  | Via-prob   …
    """
    cols = max(2, num_layers + 1)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 4.5, 9), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"Step {step_num} — Net '{net.name}' (id={net.id})",
        color=TEXT_COLOR, fontsize=14, fontweight='bold', y=1.02
    )

    # ── Row 0 ─────────────────────────────────────────────
    # Col 0: board BEFORE (what the model sees)
    _draw_board_on_ax(axes[0, 0], env, board_before, highlight_net_id=net.id)
    axes[0, 0].set_title(f"Board BEFORE step {step_num}\n"
                          f"(Routed copper visible in raster channel 10)",
                          color=TEXT_COLOR, fontsize=10, pad=8)

    # Cols 1…: JEPA heatmap per layer
    for li in range(num_layers):
        col = li + 1
        if col < cols:
            board_before.set_current_net(net.id)  # ensure source/target markers are set
            _draw_heatmap_on_ax(
                axes[0, col], heatmaps_np[li], board_before, env,
                path=path, show_existing_copper=True,
                title=f"JEPA Cost — Layer {li}\n(white = existing copper)"
            )

    # Fill any extra cols in row 0
    for col in range(num_layers + 1, cols):
        axes[0, col].axis('off')

    # ── Row 1 ─────────────────────────────────────────────
    # Col 0: board AFTER (result of routing)
    _draw_board_on_ax(axes[1, 0], env, board_after, highlight_net_id=net.id, path=path)
    axes[1, 0].set_title(f"Board AFTER routing net '{net.name}'",
                          color=TEXT_COLOR, fontsize=10, pad=8)

    # Col 1: A* occupancy (hard obstacles + routed copper the pathfinder sees)
    if 1 < cols:
        _draw_occupancy_on_ax(axes[1, 1], occ_map, env)

    # Col 2: Via probability
    if 2 < cols:
        _draw_via_prob_on_ax(axes[1, 2], via_prob_np, env)

    # Fill remaining
    for col in range(3, cols):
        axes[1, col].axis('off')

    plt.tight_layout(pad=1.5)
    return fig


def fig_to_wandb_image(fig, caption=''):
    """Convert matplotlib figure to wandb.Image without saving to disk."""
    import wandb
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', dpi=fig.get_dpi())
    buf.seek(0)
    return wandb.Image(buf, caption=caption)


# ─────────────────────────────────────────────────────────────────────────────
#  Main routing loop
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(args):
    import wandb

    # ── Load configs ──────────────────────────────────────
    with open('configs/training.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
    with open('configs/curriculum.yaml', 'r') as f:
        cur_cfg = yaml.safe_load(f)

    stage_map = {s['name']: s for s in cur_cfg['stages']}
    if args.stage not in stage_map:
        raise ValueError(f"Unknown stage '{args.stage}'. Available: {list(stage_map)}")
    selected_stage = stage_map[args.stage]

    # ── W&B init ──────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"routing-viz-{args.stage}-seed{args.seed}",
        config={
            'checkpoint': args.checkpoint,
            'stage': args.stage,
            'seed': args.seed,
            'num_boards': args.num_boards,
        },
        tags=['visualization', 'routing-steps', args.stage]
    )
    print(f"W&B run: {run.url}")

    # ── Load model ─────────────────────────────────────────
    ckpt_path = args.checkpoint if args.checkpoint != 'none' else None
    try:
        trainer = DreamerJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        is_dreamer = True
        print("Loaded DreamerJEPA model")
    except Exception as e:
        print(f"DreamerJEPA load failed ({e}), trying PPOJEPATrainer…")
        trainer = PPOJEPATrainer(device='cpu', load_checkpoint_path=ckpt_path)
        is_dreamer = False
        print("Loaded PPO-JEPA model")

    # ── Route multiple boards ──────────────────────────────
    summary_table = wandb.Table(columns=[
        'board', 'step', 'net_name', 'net_id', 'success',
        'reward', 'drc_violations', 'completion_pct', 'routing_panel'
    ])

    for board_i in range(args.num_boards):
        seed = args.seed + board_i
        print(f"\n{'='*60}")
        print(f"  Board {board_i+1}/{args.num_boards}  seed={seed}")
        print(f"{'='*60}")

        board_config = BoardGenerator.from_curriculum_stage(selected_stage)
        board_config.seed = seed
        env = PCBRoutingEnv(
            board_config=board_config,
            reward_weights=selected_stage.get('reward_weights')
        )
        obs, info = env.reset(seed=seed)

        h_state, z_state = None, None
        if is_dreamer:
            h_state, z_state = trainer.jepa.initial_state(batch_size=1, device=trainer.device)

        step_num = 0

        while True:
            unrouted = [n for n in env.board.nets if n.id not in env.routed_nets]
            if not unrouted:
                break

            net     = unrouted[0]
            net_idx = next(i for i, n in enumerate(env.board.nets) if n.id == net.id)

            raster_t   = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0)
            layer_mask = torch.tensor(obs['layer_mask'],   dtype=torch.float32).unsqueeze(0)

            graph = info['graph']
            if hasattr(graph, 'x_dict'):
                x_dict          = {k: v for k, v in graph.x_dict.items()}
                edge_index_dict = {k: v for k, v in graph.edge_index_dict.items()}
            else:
                x_dict          = {k: v['x'] for k, v in graph.items() if isinstance(v, dict) and 'x' in v}
                edge_index_dict = {k: v for k, v in graph.items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}

            # Snapshot board BEFORE routing
            board_before = env.board_state.clone()
            board_before.set_current_net(net.id)

            # ── Inference ─────────────────────────────────
            with torch.no_grad():
                if is_dreamer:
                    context_emb = trainer.jepa.get_context_embedding(
                        raster_t, x_dict, edge_index_dict, use_target=False)
                    net_embs, _, fs = trainer._get_net_embeddings_and_mask(
                        raster_t, x_dict, edge_index_dict)
                    sel_emb = net_embs[0, net_idx].unsqueeze(0)
                    heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                        sel_emb, h_state, z_state, deterministic=True)
                    heatmaps_via = trainer.decoder(
                        heatmap_latent, fs, env.H, env.W, active_layers_mask=layer_mask)
                    action_emb = trainer.jepa.get_action_embedding(
                        torch.tensor([net_idx], device=trainer.device), heatmap_latent)
                    h_state, z_state, _, _ = trainer.jepa.rssm_step(
                        h_state, z_state, context_emb, action_emb)
                else:
                    sp, _ = trainer.vit(raster_t)
                    ne    = trainer.gnn(x_dict, edge_index_dict)
                    fp, fsp = trainer.fusion(ne['pad'].unsqueeze(0), sp)
                    nN    = len(env.board.nets)
                    ne2   = torch.zeros((1, nN, trainer.vit.embed_dim))
                    for ni2, n2 in enumerate(env.board.nets):
                        pi = [i for i, p in enumerate(env.board.pins.values()) if p.net_id == n2.id]
                        if pi:
                            ne2[0, ni2] = fp[0, pi].mean(0)
                    heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                        ne2[0, net_idx].unsqueeze(0), fsp.mean(1))
                    heatmaps_via = trainer.decoder(
                        heatmap_latent, fsp, env.H, env.W, active_layers_mask=layer_mask)

            heatmaps_np = heatmaps_via[0, :env.board.num_layers].cpu().numpy()
            via_prob_np = heatmaps_via[0, 8].cpu().numpy()

            # Occupancy: what A* actually sees as blocked BEFORE this step
            board_before.set_current_net(net.id)
            occ_map = env.board_state.get_occupancy(0)  # layer 0

            # ── Step env ──────────────────────────────────
            next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
                net_idx, heatmaps_np, via_prob_np
            )
            path = next_info.get('path', [])
            board_after = env.board_state.clone()

            step_num += 1
            success = next_info.get('connected', False)
            drc_n   = next_info['drc_violations']
            comp    = next_info['completion_rate']

            print(f"  Step {step_num:02d} | Net '{net.name}' | "
                  f"{'✅' if success else '❌'} | R={reward:+.2f} | "
                  f"DRC={drc_n} | {comp*100:.0f}%")

            # ── Render & log ──────────────────────────────
            fig = render_routing_step(
                step_num, net, env,
                board_before, board_after,
                heatmaps_np, via_prob_np, occ_map, path,
                num_layers=env.board.num_layers
            )
            wb_img = fig_to_wandb_image(
                fig,
                caption=(f"Board {board_i+1} Step {step_num} — "
                         f"Net '{net.name}' | {'SUCCESS' if success else 'FAILED'}")
            )
            plt.close(fig)

            # Log per-step metrics + image
            wandb.log({
                f'routing/board_{board_i+1}/step_{step_num:03d}_panel': wb_img,
                f'routing/board_{board_i+1}/reward':       reward,
                f'routing/board_{board_i+1}/drc':          drc_n,
                f'routing/board_{board_i+1}/completion':   comp,
                f'routing/board_{board_i+1}/step':         step_num,
            })

            # Add to summary table
            summary_table.add_data(
                board_i + 1, step_num, net.name, net.id,
                success, reward, drc_n, comp * 100, wb_img
            )

            obs  = next_obs
            info = next_info

            if terminated or truncated:
                break

        # Log final board state for this board
        fig_final, ax_final = plt.subplots(figsize=(7, 7), dpi=100)
        fig_final.patch.set_facecolor(BG_COLOR)
        _draw_board_on_ax(ax_final, env, env.board_state)
        ax_final.set_title(
            f"Board {board_i+1} Final — {len(env.routed_nets)}/{len(env.board.nets)} nets routed",
            color=TEXT_COLOR, fontsize=12, pad=10
        )
        wandb.log({
            f'routing/board_{board_i+1}/final_board': fig_to_wandb_image(
                fig_final,
                caption=f"Board {board_i+1} final routing result"
            )
        })
        plt.close(fig_final)

    # Log summary table
    wandb.log({'routing/step_summary_table': summary_table})
    print(f"\n✅ Visualization complete! View at: {run.url}")
    wandb.finish()


def log_training_rollout_viz(trainer, is_dreamer, stage_cfg, seed, current_step):
    """
    Run one full board routing episode on the current stage config using 
    the trainer's current model weights, and log the step-by-step panels to W&B.
    
    This visualizes the whole model (ViT + GNN + Fusion + Policy + Decoder)
    step-by-step at the current curriculum stage during training.
    """
    import wandb
    
    board_config = BoardGenerator.from_curriculum_stage(stage_cfg)
    board_config.seed = seed
    
    # Use temporary PCBRoutingEnv to avoid mutating the trainer's training environment state
    env = PCBRoutingEnv(
        board_config=board_config,
        reward_weights=stage_cfg.get('reward_weights')
    )
    obs, info = env.reset(seed=seed)
    
    h_state, z_state = None, None
    if is_dreamer:
        h_state, z_state = trainer.jepa.initial_state(batch_size=1, device=trainer.device)
        
    step_num = 0
    stage_name = stage_cfg.get('name', 'unknown')
    
    summary_table = wandb.Table(columns=[
        'step', 'net_name', 'net_id', 'success',
        'reward', 'drc_violations', 'completion_pct', 'routing_panel'
    ])
    
    # We use AMP context if training is on GPU
    amp_ctx = torch.autocast(device_type=trainer.device.type, enabled=(trainer.device.type == 'cuda'))
    
    while True:
        unrouted = [n for n in env.board.nets if n.id not in env.routed_nets]
        if not unrouted:
            break
            
        net = unrouted[0]
        net_idx = next(i for i, n in enumerate(env.board.nets) if n.id == net.id)
        
        raster_t   = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(trainer.device)
        layer_mask = torch.tensor(obs['layer_mask'],   dtype=torch.float32).unsqueeze(0).to(trainer.device)
        
        graph = info['graph']
        if hasattr(graph, 'x_dict'):
            x_dict          = {k: v.to(trainer.device) for k, v in graph.x_dict.items()}
            edge_index_dict = {k: v.to(trainer.device) for k, v in graph.edge_index_dict.items()}
        else:
            x_dict          = {k: v['x'].to(trainer.device) for k, v in graph.items() if isinstance(v, dict) and 'x' in v}
            edge_index_dict = {k: v.to(trainer.device) for k, v in graph.items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}
            
        board_before = env.board_state.clone()
        board_before.set_current_net(net.id)
        
        with torch.no_grad(), amp_ctx:
            if is_dreamer:
                context_emb = trainer.jepa.get_context_embedding(
                    raster_t, x_dict, edge_index_dict, use_target=False)
                net_embs, _, fs = trainer._get_net_embeddings_and_mask(
                    raster_t, x_dict, edge_index_dict)
                sel_emb = net_embs[0, net_idx].unsqueeze(0)
                heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                    sel_emb, h_state, z_state, deterministic=True)
                heatmaps_via = trainer.decoder(
                    heatmap_latent, fs, env.H, env.W, active_layers_mask=layer_mask)
                action_emb = trainer.jepa.get_action_embedding(
                    torch.tensor([net_idx], device=trainer.device), heatmap_latent)
                h_state, z_state, _, _ = trainer.jepa.rssm_step(
                    h_state, z_state, context_emb, action_emb)
            else:
                sp, _ = trainer.vit(raster_t)
                ne    = trainer.gnn(x_dict, edge_index_dict)
                fp, fsp = trainer.fusion(ne['pad'].unsqueeze(0), sp)
                nN    = len(env.board.nets)
                ne2   = torch.zeros((1, nN, trainer.vit.embed_dim), device=trainer.device)
                for ni2, n2 in enumerate(env.board.nets):
                    pi = [i for i, p in enumerate(env.board.pins.values()) if p.net_id == n2.id]
                    if pi:
                        ne2[0, ni2] = fp[0, pi].mean(0)
                heatmap_latent, _, _ = trainer.policy.get_heatmap_latent(
                    ne2[0, net_idx].unsqueeze(0), fsp.mean(1))
                heatmaps_via = trainer.decoder(
                    heatmap_latent, fsp, env.H, env.W, active_layers_mask=layer_mask)
                    
        heatmaps_np = heatmaps_via[0, :env.board.num_layers].cpu().numpy()
        via_prob_np = heatmaps_via[0, 8].cpu().numpy()
        occ_map = env.board_state.get_occupancy(0)
        
        next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
            net_idx, heatmaps_np, via_prob_np
        )
        path = next_info.get('path', [])
        board_after = env.board_state.clone()
        
        step_num += 1
        success = next_info.get('connected', False)
        drc_n   = next_info['drc_violations']
        comp    = next_info['completion_rate']
        
        fig = render_routing_step(
            step_num, net, env,
            board_before, board_after,
            heatmaps_np, via_prob_np, occ_map, path,
            num_layers=env.board.num_layers
        )
        wb_img = fig_to_wandb_image(
            fig,
            caption=f"Eval Step {step_num} | Net '{net.name}'"
        )
        plt.close(fig)
        
        summary_table.add_data(
            step_num, net.name, net.id, success,
            reward, drc_n, comp * 100, wb_img
        )
        
        obs  = next_obs
        info = next_info
        if terminated or truncated:
            break
            
    wandb.log({
        f"eval_training/step_summary_table": summary_table,
        f"eval_training/timesteps": current_step,
        f"eval_training/stage_name": stage_name
    })


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JEPA routing step visualizer → W&B')
    parser.add_argument('--checkpoint',      default='none',
                        help='Path to .pt checkpoint (or "none" for untrained)')
    parser.add_argument('--stage',           default='multi_net_single_layer',
                        help='Curriculum stage name')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--num_boards',      type=int, default=3,
                        help='Number of different boards to visualize')
    parser.add_argument('--wandb_project',   default='pcb-router')
    parser.add_argument('--wandb_run_name',  default=None)
    args = parser.parse_args()
    run_visualization(args)
