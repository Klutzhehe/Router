# ================================================================
#  ALL-NETS HEATMAP CELL — paste this into a new cell below Cell 6
#  (or run it standalone after training has started)
#
#  It reads trainer.all_episode_heatmaps which is populated
#  automatically every episode by the updated trainer.py
# ================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import IPython.display as ipydisplay
from IPython.display import clear_output
import numpy as np

BG     = "#0d0e1a"
PANEL  = "#13142b"
BORDER = "#2a2d50"
WHITE  = "#e8eaf6"
net_colors = [
    "#3B82F6", "#10B981", "#EC4899", "#8B5CF6", "#06B6D4",
    "#F59E0B", "#14B8A6", "#6366F1", "#A855F7", "#F43F5E",
]

def show_all_nets_heatmaps(trainer, layer=0):
    """
    Display every net's JEPA heatmap from the last episode in a grid.
    
    Args:
        trainer: DreamerJEPATrainer instance
        layer:   Which copper layer index to display (default 0)
    """
    all_hmaps = getattr(trainer, 'all_episode_heatmaps', [])
    if not all_hmaps:
        print("No heatmaps recorded yet. Make sure training has run at least one episode.")
        return

    board = getattr(trainer, 'last_completed_board', trainer.env.board)

    n_nets  = len(all_hmaps)
    n_cols  = min(4, n_nets)
    n_rows  = int(np.ceil(n_nets / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.8, n_rows * 3.8),
        facecolor=BG,
        squeeze=False
    )
    ts = trainer.total_timesteps
    fig.suptitle(
        f"🗺️  All-Nets Heatmap Overview — Layer {layer}  |  Step {ts:,}",
        color=WHITE, fontsize=14, fontweight="bold", y=1.02
    )

    axes_flat = axes.flatten()
    for i, entry in enumerate(all_hmaps):
        ax = axes_flat[i]
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.set_xticks([]); ax.set_yticks([])

        hm      = entry['heatmaps_np']
        lyr     = min(layer, hm.shape[0] - 1)
        ni      = entry['net_idx']
        col     = net_colors[ni % len(net_colors)]

        ax.imshow(
            hm[lyr], cmap='magma', origin='lower',
            extent=(0, board.width, 0, board.height),
            alpha=0.88, vmin=0, vmax=1
        )
        ax.set_title(
            f"{entry['net_name']}  (L{lyr})",
            color=col, fontsize=8, pad=4
        )

        # Overlay all pins — highlight current net's pins brighter
        for pin in board.pins.values():
            pc    = net_colors[pin.net_id % len(net_colors)]
            is_me = (pin.net_id == (board.nets[ni].id if ni < len(board.nets) else -1))
            ax.add_patch(mpatches.Circle(
                (pin.global_x, pin.global_y),
                radius=2.2 if is_me else 1.4,
                fc=pc, ec=WHITE,
                lw=0.8 if is_me else 0.3,
                alpha=1.0 if is_me else 0.5,
                zorder=5 if is_me else 3
            ))

        ax.set_xlim(0, board.width)
        ax.set_ylim(0, board.height)
        ax.set_aspect('equal')

    # Hide unused tiles
    for j in range(n_nets, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(pad=1.5)
    ipydisplay.display(fig)
    plt.close(fig)
    print(f"Showing {n_nets} nets — Layer {layer}")


# ── Run it ──────────────────────────────────────────────────────────────
show_all_nets_heatmaps(trainer, layer=0)
