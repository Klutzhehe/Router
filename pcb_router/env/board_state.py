import torch
import numpy as np
import copy
from typing import List, Tuple, Dict, Any
from pcb_router.data.board_generator import Board, Pin, Component, Obstacle
from pcb_router.routing.trace_generator import TraceSegment, Via

class BoardState:
    def __init__(self, board: Board, resolution: float = 0.1):
        self.board = board
        self.resolution = resolution
        self.height = board.height
        self.width = board.width
        self.num_layers = board.num_layers
        
        # Max active layer index
        self.active_layers_mask = torch.zeros(8)
        self.active_layers_mask[:self.num_layers] = 1.0
        
        # 13 channels tensor: 8 copper + 5 semantic
        self.raster = torch.zeros((13, self.height, self.width), dtype=torch.float32)
        
        # Keep track of physical traces and vias separately for DRC checks
        self.traces = []
        self.vias = []
        self.routed_net_ids = set()
        
        # Render static elements
        self._render_initial_state()

    def clone(self) -> 'BoardState':
        cloned = copy.copy(self)
        cloned.raster = self.raster.clone()
        cloned.traces = list(self.traces)
        cloned.vias = list(self.vias)
        cloned.routed_net_ids = set(self.routed_net_ids)
        return cloned

    def reset(self):
        """Reset raster to initial state, clearing routed traces"""
        self.raster.zero_()
        self.traces.clear()
        self.vias.clear()
        self.routed_net_ids.clear()
        self._render_initial_state()

    def get_raster(self) -> torch.Tensor:
        return self.raster

    def get_occupancy(self, layer: int) -> np.ndarray:
        """Returns binary occupancy grid for A* pathfinding on a given layer.
        
        Only hard obstacles (keep-outs, ch9) and already-routed copper (ch10)
        block the path. Pad copper (ch0-ch7) is intentionally excluded because
        pads are valid routing endpoints, not obstacles.
        """
        obstacles = self.raster[9].numpy()   # keepout zones / board edge
        routed    = self.raster[10].numpy()  # ch10 = all routed traces (not pads)
        return np.clip(obstacles + routed, 0, 1)

    def _render_initial_state(self):
        # Channel 9: Obstacles and Keep-outs
        for obs in self.board.obstacles:
            self._rasterize_rect(obs.x, obs.y, obs.width, obs.height, channel=9, val=1.0)
        for ko in self.board.keep_out_zones:
            self._rasterize_rect(ko.x, ko.y, ko.width, ko.height, channel=9, val=1.0)
            
        # Channel 8: Pads
        max_net_id = max([n.id for n in self.board.nets]) if self.board.nets else 1
        for pin in self.board.pins.values():
            pad_val = pin.net_id / max_net_id if pin.net_id > 0 else 0.1
            
            # Draw pads into copper layer 0 (top) and channel 8 (pads channel)
            pad_w = 6 # pad width in cells (approx 0.6mm)
            if pin.pad_shape == 0: # circular
                self._rasterize_circle(pin.global_x, pin.global_y, radius=3, channel=8, val=pad_val)
                # All pads are on copper layer 0 by default
                self._rasterize_circle(pin.global_x, pin.global_y, radius=3, channel=0, val=1.0)
            else: # rectangular
                self._rasterize_rect(pin.global_x - 3, pin.global_y - 3, pad_w, pad_w, channel=8, val=pad_val)
                self._rasterize_rect(pin.global_x - 3, pin.global_y - 3, pad_w, pad_w, channel=0, val=1.0)
                
        # Channel 12: Board outline (1.0 on-board, 0.0 off-board)
        # Rectangular outline for now
        self.raster[12, :, :] = 1.0

    def set_current_net(self, net_id: int):
        """Update channel 11 with current net source and target markers"""
        # Clear channel 11
        self.raster[11].zero_()
        
        # Find pins for this net
        net = next((n for n in self.board.nets if n.id == net_id), None)
        if net:
            # Source pad (first pin)
            src_pin_id = net.pin_ids[0]
            src_pin = self.board.pins.get(src_pin_id)
            if src_pin:
                self._rasterize_circle(src_pin.global_x, src_pin.global_y, radius=4, channel=11, val=1.0)
                
            # Target pads (other pins)
            for tgt_pin_id in net.pin_ids[1:]:
                tgt_pin = self.board.pins.get(tgt_pin_id)
                if tgt_pin:
                    self._rasterize_circle(tgt_pin.global_x, tgt_pin.global_y, radius=4, channel=11, val=0.5)

    def add_routed_trace(self, trace_segments: List[TraceSegment], vias: List[Via]):
        """Render new trace segments and vias into copper and routed channels"""
        for seg in trace_segments:
            self.traces.append(seg)
            # Render trace into its specific copper layer and channel 10 (all routed traces)
            self._rasterize_trace(seg, channel=seg.layer, val=1.0)
            self._rasterize_trace(seg, channel=10, val=1.0)
            
        for via in vias:
            self.vias.append(via)
            # Render via as a small circular pad on all layers it transitions through
            r = int(round((via.drill_size / 2.0 + via.annular_ring) / self.resolution))
            for layer in range(min(via.from_layer, via.to_layer), max(via.from_layer, via.to_layer) + 1):
                self._rasterize_circle(via.x, via.y, radius=r, channel=layer, val=1.0)
                self._rasterize_circle(via.x, via.y, radius=r, channel=10, val=1.0)
                
        if trace_segments:
            self.routed_net_ids.add(trace_segments[0].net_id)

    def get_congestion_map(self) -> np.ndarray:
        """Estimated congestion based on straight paths of unrouted nets"""
        congestion = np.zeros((self.height, self.width), dtype=np.float32)
        for net in self.board.nets:
            if net.id not in self.routed_net_ids:
                # Add straight line density between net pins
                pins = [self.board.pins[pid] for pid in net.pin_ids]
                for i in range(len(pins) - 1):
                    p1 = pins[i]
                    p2 = pins[i+1]
                    # Draw simple approximate line in numpy
                    y_indices, x_indices = self._get_line_coords(p1.global_x, p1.global_y, p2.global_x, p2.global_y)
                    congestion[y_indices, x_indices] += 1.0
        return congestion

    def _rasterize_rect(self, x: int, y: int, w: int, h: int, channel: int, val: float):
        # Clip rect boundaries
        x1 = max(0, min(self.width - 1, x))
        y1 = max(0, min(self.height - 1, y))
        x2 = max(0, min(self.width - 1, x + w))
        y2 = max(0, min(self.height - 1, y + h))
        self.raster[channel, y1:y2+1, x1:x2+1] = val

    def _rasterize_circle(self, cx: int, cy: int, radius: int, channel: int, val: float):
        # Create grid coordinates relative to center
        y_min = max(0, cy - radius)
        y_max = min(self.height - 1, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(self.width - 1, cx + radius)
        
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    self.raster[channel, y, x] = val

    def _rasterize_trace(self, seg: TraceSegment, channel: int, val: float):
        """Draw trace segment with its physical width converted to cells"""
        width_cells = int(round(seg.width / self.resolution))
        radius = max(1, width_cells // 2)
        
        x1, y1 = seg.start_x, seg.start_y
        x2, y2 = seg.end_x, seg.end_y
        
        # Calculate bounding box plus padding for width
        padding = radius + 1
        min_x = max(0, min(x1, x2) - padding)
        max_x = min(self.width - 1, max(x1, x2) + padding)
        min_y = max(0, min(y1, y2) - padding)
        max_y = min(self.height - 1, max(y1, y2) + padding)
        
        # Calculate segment vectors
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        
        if seg_len_sq == 0:
            self._rasterize_circle(x1, y1, radius, channel, val)
            return
            
        # Draw cells whose distance to line segment is less than radius
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Projection factor t
                t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                
                # Closest point on segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                
                # Check distance
                dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
                if dist_sq <= radius ** 2:
                    self.raster[channel, y, x] = val

    def _get_line_coords(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simple line coords generator using integer steps"""
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x = np.linspace(x1, x2, num_points).astype(np.int32)
        y = np.linspace(y1, y2, num_points).astype(np.int32)
        # Clip to board boundaries
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        return y, x
