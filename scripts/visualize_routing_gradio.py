"""
visualize_routing_gradio.py
===========================
Gradio live dashboard for the autoregressive JEPA PCB router (Google Colab friendly).

Launches a Gradio app with a public URL (share=True) — no port tunneling needed.

The dashboard streams the *active training board* as the step policy draws traces
one cell at a time in the training environment. It reads two attributes that the
training loop (see the notebook's live-viz cell) publishes on the trainer:

    trainer.last_training_board_image  -> PIL.Image of the current board
    trainer.live_training_status       -> markdown status string

and lets you toggle live-step visualization on/off (CONFIG.LIVE_STEP_VIZ) plus its
per-step delay.

NOTE: The old net-by-net "Interactive Evaluator" tab and its JEPA cost-heatmap panels
were removed together with the neural HeatmapDecoder. Routing is autoregressive now —
the policy moves a cursor under a full-resolution valid-move mask; there is no dense
cost heatmap to display. A* pathfinding still exists, but only for offline BC dataset
generation (scripts/generate_bc_dataset.py), not for live routing.

Inline Colab Usage:
-------------------
    import sys; sys.path.insert(0, '/content/Router')
    import os; os.chdir('/content/Router')

    from scripts.visualize_routing_gradio import launch_gradio_visualizer
    launch_gradio_visualizer(
        checkpoint_path='/content/drive/MyDrive/pcb_router/checkpoints/checkpoint_50000.pt',
        share=True   # gives a public *.gradio.live URL
    )
"""

import os
import sys
import io
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# ── allow running from project root ──────────────────────────────────────────
_here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _here not in sys.path:
    sys.path.insert(0, _here)

from pcb_router.training.trainer import DreamerJEPATrainer

# ── Color palette ─────────────────────────────────────────────────────────────
BG = '#111222'
FG = '#E2E8F0'


def _fig_to_pil(fig):
    """Convert a matplotlib figure → PIL Image for Gradio."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf).copy()


def _placeholder_image(message):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.text(0.5, 0.5, message, color=FG, ha='center', va='center', wrap=True)
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio app builder
# ─────────────────────────────────────────────────────────────────────────────

def build_gradio_app(trainer, is_dreamer=True, cur_cfg=None):
    """Build and return a Gradio Blocks app with the live training stream."""
    import gradio as gr

    with gr.Blocks(
        title="JEPA PCB Router — Live Training Dashboard",
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

        gr.Markdown("# ⚡ JEPA PCB Router — Live Training Dashboard")

        with gr.Tab("📡 Live Training Stream"):
            gr.Markdown("### Watch Training Rollouts Step-by-Step")
            gr.Markdown(
                "This panel streams the active board layout as the autoregressive step "
                "policy draws traces in the training environment. Enable live-step viz "
                "below to start streaming (this slows training slightly)."
            )

            with gr.Row():
                btn_enable_viz = gr.Checkbox(
                    label="Enable Live Step viz (overrides CONFIG.LIVE_STEP_VIZ)", value=False)
                btn_delay = gr.Slider(0.01, 0.5, value=0.02, label="Live Step Delay (seconds)")

            with gr.Row():
                live_board_img = gr.Image(label="Active Board State", type="pil", height=500)
                with gr.Column():
                    live_metrics_md = gr.Markdown("Waiting for training steps...")

            def toggle_live_step_viz(checked, delay_val):
                trainer.train_cfg.get('training', {})['LIVE_STEP_VIZ'] = checked
                trainer.train_cfg.get('training', {})['LIVE_STEP_DELAY'] = delay_val
                trainer.live_step_viz = checked
                trainer.live_step_delay = delay_val
                return f"Live viz: {'Enabled' if checked else 'Disabled'} (delay={delay_val}s)"

            btn_enable_viz.change(toggle_live_step_viz,
                                  inputs=[btn_enable_viz, btn_delay], outputs=live_metrics_md)
            btn_delay.change(toggle_live_step_viz,
                             inputs=[btn_enable_viz, btn_delay], outputs=live_metrics_md)

            live_timer = gr.Timer(value=0.5, active=True)

            def update_live_stream_view():
                img = getattr(trainer, 'last_training_board_image', None)
                if img is None:
                    img = _placeholder_image(
                        "Training hasn't started or LIVE_STEP_VIZ is False.\n"
                        "Enable the checkbox above to start streaming.")
                status = getattr(trainer, 'live_training_status', "Waiting for training steps...")
                return img, status

            live_timer.tick(update_live_stream_view, outputs=[live_board_img, live_metrics_md])

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
    Load model + launch the live Gradio dashboard.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to .pt checkpoint. None = untrained random model.
    stage : str or None
        Unused; kept for backward-compatible call sites.
    share : bool
        True = generate a public *.gradio.live URL (works from Colab).
    server_port : int
        Local port (only matters when share=False).
    curriculum_config_path : str
        Path to curriculum YAML (optional; loaded if present).
    """
    cur_cfg = None
    if curriculum_config_path and os.path.exists(curriculum_config_path):
        with open(curriculum_config_path, 'r') as f:
            cur_cfg = yaml.safe_load(f)

    trainer = DreamerJEPATrainer(device='cpu', load_checkpoint_path=checkpoint_path)
    print("✅ Loaded DreamerJEPA model")

    demo = build_gradio_app(trainer, is_dreamer=True, cur_cfg=cur_cfg)
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
