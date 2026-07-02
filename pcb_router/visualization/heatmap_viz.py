import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pcb_router.env.board_state import BoardState
from pcb_router.visualization.renderer import BoardRenderer
from pcb_router.data.board_generator import Board

class HeatmapVisualizer:
    def __init__(self, theme_dark: bool = True):
        self.theme_dark = theme_dark
        self.renderer = BoardRenderer(theme_dark)
        
        if theme_dark:
            self.bg_color = '#111222'
            self.text_color = '#FFFFFF'
            self.grid_color = '#222336'
        else:
            self.bg_color = '#FFFFFF'
            self.text_color = '#111827'
            self.grid_color = '#E5E7EB'

    def render_heatmap(
        self,
        heatmap: np.ndarray,
        board_state: Optional[BoardState] = None,
        title: str = ''
    ) -> plt.Figure:
        """
        Render cost heatmap as an overlay or stand-alone grid
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        
        # Show heatmap
        im = ax.imshow(heatmap, cmap='inferno', origin='lower', alpha=0.95)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=self.text_color)
        cbar.ax.tick_params(labelcolor=self.text_color)
        cbar.set_label('Routing Cost/Difficulty', color=self.text_color, labelpad=10)
        
        # Overlay board elements (components/pads) for reference if provided
        if board_state is not None:
            # We add transparent components and pads
            board = board_state.board
            for comp in board.components:
                rect = patches.Rectangle(
                    (comp.x, comp.y), comp.width, comp.height,
                    linewidth=1, edgecolor='#4B5563', facecolor='none', alpha=0.5, zorder=5
                )
                ax.add_patch(rect)
            for pin in board.pins.values():
                circle = patches.Circle(
                    (pin.global_x, pin.global_y), radius=3,
                    facecolor='none', edgecolor='#FFFFFF', linewidth=0.5, alpha=0.8, zorder=6
                )
                ax.add_patch(circle)
                
        ax.set_title(title if title else "Cost Heatmap", color=self.text_color, fontsize=14, pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig

    def render_heatmap_with_path(
        self,
        heatmap: np.ndarray,
        path: List[Tuple[int, int, int]],
        board_state: Optional[BoardState] = None,
        layer: int = 0
    ) -> plt.Figure:
        """Overlays the generated A* path on top of the cost heatmap"""
        fig = self.render_heatmap(heatmap, board_state, title=f"Heatmap with Planned Path (Layer {layer})")
        ax = fig.gca()
        
        if path:
            # Filter waypoints for current layer
            pts = [(wp[0], wp[1]) for wp in path if wp[2] == layer]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color='#06B6D4', linewidth=2.5, marker='o', markersize=3, label='Planned Trace')
                # Mark start and end
                ax.plot(xs[0], ys[0], marker='*', color='#10B981', markersize=10, label='Start')
                ax.plot(xs[-1], ys[-1], marker='s', color='#EF4444', markersize=8, label='Target')
                
                ax.legend(facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
                
        return fig

    def render_multi_layer_heatmaps(
        self,
        layer_heatmaps: np.ndarray, # (num_active_layers, H, W)
        via_prob: np.ndarray,       # (H, W)
        active_layers: List[int]
    ) -> plt.Figure:
        """Show subplots of cost heatmaps for all active layers + via probabilities side-by-side"""
        num_plots = len(active_layers) + 1
        cols = min(3, num_plots)
        rows = int(np.ceil(num_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.patch.set_facecolor(self.bg_color)
        axes = axes.flatten() if num_plots > 1 else [axes]
        
        for idx, layer in enumerate(active_layers):
            ax = axes[idx]
            ax.set_facecolor(self.bg_color)
            im = ax.imshow(layer_heatmaps[layer], cmap='inferno', origin='lower')
            ax.set_title(f"Layer {layer} Cost", color=self.text_color, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Last plot: via probability map
        ax = axes[num_plots - 1]
        ax.set_facecolor(self.bg_color)
        im = ax.imshow(via_prob, cmap='viridis', origin='lower')
        ax.set_title("Via Place Confidence", color=self.text_color, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Turn off remaining unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        return fig

    def render_routing_comparison(
        self,
        board_before: BoardState,
        board_after: BoardState,
        heatmap: np.ndarray,
        path: List[Tuple[int, int, int]]
    ) -> plt.Figure:
        """Side-by-side dashboard of: Before Routing -> Planned Heatmap Path -> Routed Result"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(self.bg_color)
        
        # Plot 1: Before Routing
        # Use our existing rendering tools by temporarily swapping axes
        fig_temp = self.renderer.render_board(board_before, board_before.board, show_all_layers=True)
        ax1.imshow(fig_temp.gca().images[0].get_array() if fig_temp.gca().images else np.zeros((10, 10)), origin='lower')
        # Actually, copy elements to axes
        plt.close(fig_temp)
        
        # We can implement a clean rendering loop here
        self._plot_on_ax(ax1, board_before, board_before.board)
        ax1.set_title("Before Step", color=self.text_color, fontsize=12)
        
        # Plot 2: Heatmap Overlay
        im = ax2.imshow(heatmap, cmap='inferno', origin='lower')
        
        # Overlay transparent components and pads for reference
        for comp in board_before.board.components:
            rect = patches.Rectangle(
                (comp.x, comp.y), comp.width, comp.height,
                linewidth=1, edgecolor='#4B5563', facecolor='none', alpha=0.5, zorder=5
            )
            ax2.add_patch(rect)
        for pin in board_before.board.pins.values():
            circle = patches.Circle(
                (pin.global_x, pin.global_y), radius=3,
                facecolor='none', edgecolor='#FFFFFF', linewidth=0.5, alpha=0.8, zorder=6
            )
            ax2.add_patch(circle)

        if path:
            xs, ys = zip(*[(wp[0], wp[1]) for wp in path])
            ax2.plot(xs, ys, color='#06B6D4', linewidth=2.0, zorder=10)
        ax2.set_title("Planned Heatmap & Path", color=self.text_color, fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot 3: Routed Result
        self._plot_on_ax(ax3, board_after, board_after.board)
        ax3.set_title("After Routing", color=self.text_color, fontsize=12)
        
        plt.tight_layout()
        return fig

    def _plot_on_ax(self, ax, state: BoardState, board: Board):
        ax.set_facecolor(self.bg_color)
        # Components
        for comp in board.components:
            rect = patches.Rectangle((comp.x, comp.y), comp.width, comp.height, facecolor='#374151', edgecolor='#4B5563', alpha=0.8)
            ax.add_patch(rect)
        # Traces
        for seg in state.traces:
            color = self.renderer._get_net_color(seg.net_id)
            ax.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y], color=color, linewidth=2)
        # Pads
        for pin in board.pins.values():
            color = self.renderer._get_net_color(pin.net_id)
            circle = patches.Circle((pin.global_x, pin.global_y), radius=3, facecolor=color, edgecolor='#FFFFFF', linewidth=0.5)
            ax.add_patch(circle)
        ax.set_xlim(0, board.width)
        ax.set_ylim(0, board.height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def create_training_dashboard(self, metrics: Dict[str, List[float]]) -> plt.Figure:
        """Create 2x2 training metric curves visualizer"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor(self.bg_color)
        
        def plot_curve(ax, data, title, ylabel, color):
            ax.set_facecolor(self.bg_color)
            ax.grid(True, color=self.grid_color, linestyle='--', linewidth=0.5)
            if data:
                steps = np.arange(len(data))
                ax.plot(steps, data, color=color, linewidth=0.8, alpha=0.25)
                # Calculate and plot contiguous moving average (max window 20)
                ma = [np.mean(data[max(0, i - 19) : i + 1]) for i in range(len(data))]
                ax.plot(steps, ma, color='#F59E0B', linewidth=2.0, label='MA')
            ax.set_title(title, color=self.text_color, fontsize=12)
            ax.set_ylabel(ylabel, color=self.text_color)
            ax.set_xlabel('Episodes / Updates', color=self.text_color)
            ax.tick_params(colors=self.text_color)
            
        plot_curve(ax1, metrics.get('reward', []), "Episode Reward", "Reward", '#3B82F6')
        plot_curve(ax2, metrics.get('completion', []), "Routing Completion Rate", "Rate", '#10B981')
        plot_curve(ax3, metrics.get('violations', []), "DRC Violation Rate", "Violations / Net", '#EF4444')
        plot_curve(ax4, metrics.get('loss_jepa', []), "JEPA Prediction Loss", "Loss", '#8B5CF6')
        
        plt.tight_layout()
        return fig
