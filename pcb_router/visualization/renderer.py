import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any, Optional
from pcb_router.env.board_state import BoardState
from pcb_router.data.board_generator import Board, Component, Pin
from pcb_router.routing.trace_generator import TraceSegment, Via
from pcb_router.env.drc_checker import DRCViolation

class BoardRenderer:
    def __init__(self, theme_dark: bool = True):
        self.theme_dark = theme_dark
        if theme_dark:
            self.bg_color = '#111222'
            self.text_color = '#FFFFFF'
            self.grid_color = '#222336'
            self.obstacle_color = '#EF4444' # soft red
            self.keepout_color = '#F59E0B' # amber
        else:
            self.bg_color = '#FFFFFF'
            self.text_color = '#111827'
            self.grid_color = '#E5E7EB'
            self.obstacle_color = '#FCA5A5'
            self.keepout_color = '#FDE68A'
            
        # Curated net color palette
        self.net_colors = [
            '#3B82F6', '#10B981', '#EC4899', '#8B5CF6', '#06B6D4',
            '#F59E0B', '#14B8A6', '#6366F1', '#A855F7', '#10B981'
        ]

    def _get_net_color(self, net_id: int) -> str:
        if net_id == 0:
            return '#6B7280' # unassigned gray
        return self.net_colors[net_id % len(self.net_colors)]

    def render_board(
        self,
        board_state: BoardState,
        board: Board,
        layer: int = 0,
        show_all_layers: bool = False
    ):
        """
        Draws the board state including outline, components, pads, traces, and vias
        """
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        
        # Draw grid lines
        ax.grid(True, color=self.grid_color, linestyle='--', linewidth=0.5)
        
        # 1. Draw obstacles (channel 9 of board_state)
        # We can also draw them from the board object directly for clear dimensions
        for obs in board.obstacles:
            if show_all_layers or obs.layer == layer or obs.layer == -1:
                rect = patches.Rectangle(
                    (obs.x, obs.y), obs.width, obs.height,
                    linewidth=0, facecolor=self.obstacle_color,
                    alpha=0.35, hatch='//', label='Obstacle' if 'Obstacle' not in ax.get_legend_handles_labels()[1] else ""
                )
                ax.add_patch(rect)
                
        for ko in board.keep_out_zones:
            rect = patches.Rectangle(
                (ko.x, ko.y), ko.width, ko.height,
                linewidth=1, edgecolor=self.keepout_color, facecolor='none',
                alpha=0.6, hatch='\\\\', label='Keep-Out' if 'Keep-Out' not in ax.get_legend_handles_labels()[1] else ""
            )
            ax.add_patch(rect)

        # 2. Draw components
        for comp in board.components:
            rect = patches.Rectangle(
                (comp.x, comp.y), comp.width, comp.height,
                linewidth=1.5, edgecolor='#4B5563', facecolor='#374151' if self.theme_dark else '#F3F4F6',
                alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(
                comp.x + comp.width/2, comp.y + comp.height/2, comp.name,
                color=self.text_color, fontsize=10, ha='center', va='center', weight='bold'
            )

        # 3. Draw traces
        layer_colors = ['#F43F5E', '#06B6D4', '#8B5CF6', '#F59E0B', '#10B981', '#EC4899']
        for seg in board_state.traces:
            if show_all_layers or seg.layer == layer:
                if show_all_layers:
                    color = layer_colors[seg.layer % len(layer_colors)]
                else:
                    color = self._get_net_color(seg.net_id)
                # Convert width from mm to cells for thickness
                width_cells = max(1.5, seg.width / board_state.resolution)
                
                # Check layer for alpha/style if showing all
                alpha = 1.0 if (not show_all_layers or seg.layer == layer) else 0.4
                ax.plot(
                    [seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                    color=color, linewidth=width_cells, alpha=alpha, solid_capstyle='round'
                )

        # 4. Draw vias
        for via in board_state.vias:
            if show_all_layers or (min(via.from_layer, via.to_layer) <= layer <= max(via.from_layer, via.to_layer)):
                # Concentric circles
                r_outer = (via.drill_size/2.0 + via.annular_ring) / board_state.resolution
                r_inner = (via.drill_size/2.0) / board_state.resolution
                
                outer_circle = patches.Circle(
                    (via.x, via.y), r_outer,
                    facecolor='#EAB308', edgecolor='#FFFFFF', linewidth=0.5, alpha=0.9
                )
                inner_circle = patches.Circle(
                    (via.x, via.y), r_inner,
                    facecolor=self.bg_color, edgecolor='#EAB308', linewidth=0.5, alpha=1.0
                )
                ax.add_patch(outer_circle)
                ax.add_patch(inner_circle)

        # 5. Draw pads
        for pin in board.pins.values():
            if show_all_layers:
                color = layer_colors[pin.layer % len(layer_colors)] if pin.layer >= 0 else '#10B981'
            else:
                color = self._get_net_color(pin.net_id)
            pad_w = 6
            if pin.pad_shape == 0: # circular
                circle = patches.Circle(
                    (pin.global_x, pin.global_y), radius=3,
                    facecolor=color, edgecolor='#FFFFFF', linewidth=0.8, alpha=0.95
                )
                ax.add_patch(circle)
            else: # rectangular
                rect = patches.Rectangle(
                    (pin.global_x - 3, pin.global_y - 3), pad_w, pad_w,
                    facecolor=color, edgecolor='#FFFFFF', linewidth=0.8, alpha=0.95
                )
                ax.add_patch(rect)
                
        # Set bounds and title
        ax.set_xlim(0, board.width)
        ax.set_ylim(0, board.height)
        ax.set_aspect('equal')
        
        layer_title = "All Layers" if show_all_layers else f"Layer {layer}"
        ax.set_title(f"PCB Layout Visualizer — {layer_title}", color=self.text_color, fontsize=14, pad=15)
        
        # Color scale info / legends
        if show_all_layers:
            from matplotlib.lines import Line2D
            layer_legends = [
                Line2D([0], [0], color='#F43F5E', lw=3, label='Top Layer (L0)'),
                Line2D([0], [0], color='#06B6D4', lw=3, label='Bottom Layer (L1)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#EAB308', markersize=8, label='Via')
            ]
            ax.legend(handles=layer_legends, loc='upper right', facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
        else:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
            
        fig.tight_layout()
        return fig

    def render_state_channels(self, board_state: BoardState) -> plt.Figure:
        """Render grid showing all 13 channels of the board state raster"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.patch.set_facecolor(self.bg_color)
        
        channels = board_state.get_raster().numpy()
        
        channel_names = [
            "Copper L1", "Copper L2", "Copper L3", "Copper L4",
            "Copper L5", "Copper L6", "Copper L7", "Copper L8",
            "Pads Location", "Obstacles", "Routed Traces", "Current Net Markers",
            "Board Outline", "Unused 1", "Unused 2", "Unused 3"
        ]
        
        for i in range(16):
            r, c = i // 4, i % 4
            ax = axes[r, c]
            ax.set_facecolor(self.bg_color)
            
            if i < 13:
                ax.imshow(channels[i], cmap='magma', origin='lower')
                ax.set_title(channel_names[i], color=self.text_color, fontsize=10)
            else:
                ax.axis('off')
                
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        return fig

    def render_drc_violations(self, board_state: BoardState, violations: List[DRCViolation]) -> plt.Figure:
        """Draw board layout and mark violation spots with standard red crosshairs"""
        fig = self.render_board(board_state, board_state.board, show_all_layers=True)
        ax = fig.gca()
        
        # Draw violation crosshairs
        for v in violations:
            ax.plot(
                v.x, v.y, marker='x', color='#DC2626', markersize=12,
                markeredgewidth=2.5, label='DRC Violation' if 'DRC Violation' not in ax.get_legend_handles_labels()[1] else ""
            )
            # circle halo
            circle = patches.Circle(
                (v.x, v.y), radius=10,
                edgecolor='#DC2626', facecolor='none', linewidth=1.5, linestyle='--'
            )
            ax.add_patch(circle)
            
        # Refresh legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
        
        return fig

    def save_animation(self, frames: List[np.ndarray], path: str, fps: int = 2):
        """Save animated list of RGB frames as GIF (using matplotlib or pillow)"""
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        if imgs:
            imgs[0].save(
                path, save_all=True, append_images=imgs[1:],
                duration=int(1000 / fps), loop=0
            )
            print(f"Animated routing replay saved to {path}")
