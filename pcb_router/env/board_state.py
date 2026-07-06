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
        self.current_net_id = None
        
        # Render static elements
        self._render_initial_state()

    def clone(self) -> 'BoardState':
        cloned = copy.copy(self)
        cloned.raster = self.raster.clone()
        cloned.traces = list(self.traces)
        cloned.vias = list(self.vias)
        cloned.routed_net_ids = set(self.routed_net_ids)
        cloned.current_net_id = self.current_net_id
        return cloned

    def reset(self):
        """Reset raster to initial state, clearing routed traces"""
        self.raster.zero_()
        self.traces.clear()
        self.vias.clear()
        self.routed_net_ids.clear()
        self.current_net_id = None
        self._render_initial_state()

    def get_raster(self) -> torch.Tensor:
        return self.raster

    def get_occupancy(self, layer: int) -> np.ndarray:
        """Returns binary occupancy grid for A* pathfinding on a given layer.
        
        Only hard obstacles (keep-outs, ch9) and already-routed copper (ch10)
        block the path. Pad copper (ch0-ch7) is intentionally excluded because
        pads are valid routing endpoints, not obstacles.
        """
        # 1. Start with routed copper (channel 10)
        occ = self.raster[10].numpy().copy()
        
        # 2. Add obstacles that exist on this specific layer (or all layers: -1)
        for obs in self.board.obstacles:
            if obs.layer == -1 or obs.layer == layer:
                x1 = max(0, obs.x)
                y1 = max(0, obs.y)
                x2 = min(self.width, obs.x + obs.width)
                y2 = min(self.height, obs.y + obs.height)
                occ[y1:y2, x1:x2] = 1.0
                
        # 3. Add keep-out zones
        for ko in self.board.keep_out_zones:
            if ko.layer == -1 or ko.layer == layer:
                x1 = max(0, ko.x)
                y1 = max(0, ko.y)
                x2 = min(self.width, ko.x + ko.width)
                y2 = min(self.height, ko.y + ko.height)
                occ[y1:y2, x1:x2] = 1.0
                
        # 4. Add pads of other nets with DRC clearance buffer
        if self.current_net_id is not None:
            # Pre-compute net class rules for lookup
            net_rules = {}
            for net in self.board.nets:
                rules = self.board.design_rules.get(net.net_class, self.board.design_rules.get('default', {}))
                net_rules[net.id] = rules
            default_rules = self.board.design_rules.get('default', {})
            
            current_net = next((n for n in self.board.nets if n.id == self.current_net_id), None)
            current_rules = net_rules.get(self.current_net_id, default_rules)
            current_width = current_rules.get('width', 0.15)
            
            exempt_net_ids = {self.current_net_id}
            if current_net and current_net.is_diff_pair and current_net.diff_pair_id is not None:
                exempt_net_ids.add(current_net.diff_pair_id)
            
            # 2 cells (0.2mm) of extra clearance for length-matched nets to reserve space
            length_matched_inflation = 2.0 if (current_net and current_net.target_length > 0) else 0.0
            
            for pin in self.board.pins.values():
                if pin.net_id not in exempt_net_ids:
                    # Block all layers for through-hole pins (-1), block specific layer for SMD pads
                    if pin.layer == -1 or pin.layer == layer:
                        rules = net_rules.get(pin.net_id, default_rules)
                        clearance = max(rules.get('clearance', 0.15), current_rules.get('clearance', 0.15))
                        
                        # Pad radius is 3. We inflate the pad obstacle by:
                        # clearance + trace_radius to ensure the trace center stays far enough away.
                        pad_r = 3.0
                        clearance_cells = clearance / self.resolution
                        trace_r_cells = (current_width / 2.0) / self.resolution
                        avoid_r = pad_r + clearance_cells + trace_r_cells + length_matched_inflation
                        
                        cx, cy = pin.global_x, pin.global_y
                        
                        # Calculate bounds based on inflated avoidance radius
                        r_ceil = int(avoid_r + 0.99)
                        y_min = max(0, cy - r_ceil)
                        y_max = min(self.height - 1, cy + r_ceil)
                        x_min = max(0, cx - r_ceil)
                        x_max = min(self.width - 1, cx + r_ceil)
                        
                        if pin.pad_shape == 0:  # circular
                            for y in range(y_min, y_max + 1):
                                for x in range(x_min, x_max + 1):
                                    if (x - cx)**2 + (y - cy)**2 <= avoid_r**2:
                                        occ[y, x] = 1.0
                        else:  # rectangular / oval (approximated as square of size 2*avoid_r)
                            occ[y_min:y_max+1, x_min:x_max+1] = 1.0
                            
        return np.clip(occ, 0, 1)

    def get_foreign_pad_discs(self, layer: int) -> list:
        """Geometry (not raster) version of the foreign-pad obstacles get_occupancy rasterizes.

        Returns a list of (cx, cy, radius) discs in grid cells for pads belonging to OTHER nets
        that block routing on `layer`. Used by obstacle-aware imagination so the imagined cursor
        must route around the very same pad discs the real valid-move mask blocks — without
        rasterizing and storing a full occupancy grid per step. Radius matches get_occupancy's
        inflated avoidance radius (pad + clearance + trace half-width).
        """
        discs = []
        if self.current_net_id is None:
            return discs
        net_rules = {}
        for net in self.board.nets:
            net_rules[net.id] = self.board.design_rules.get(net.net_class, self.board.design_rules.get('default', {}))
        default_rules = self.board.design_rules.get('default', {})
        
        current_net = next((n for n in self.board.nets if n.id == self.current_net_id), None)
        exempt_net_ids = {self.current_net_id}
        if current_net and current_net.is_diff_pair and current_net.diff_pair_id is not None:
            exempt_net_ids.add(current_net.diff_pair_id)
            
        for pin in self.board.pins.values():
            if pin.net_id not in exempt_net_ids and (pin.layer == -1 or pin.layer == layer):
                rules = net_rules.get(pin.net_id, default_rules)
                clearance = rules.get('clearance', 0.15)
                trace_width = rules.get('width', 0.15)
                pad_r = 3.0
                clearance_cells = clearance / self.resolution
                trace_r_cells = (trace_width / 2.0) / self.resolution
                avoid_r = pad_r + clearance_cells + trace_r_cells
                discs.append((float(pin.global_x), float(pin.global_y), float(avoid_r)))
        return discs

    def get_pin_exclusion_mask(self, layer: int, extra_margin: int = 2) -> np.ndarray:
        """Returns a mask where 1.0 = routable, 0.0 = within clearance of another net's pad.
        
        Applied to heatmaps BEFORE passing to A* so the decoder learns that areas
        near other-net pads should never have high routing preference.
        
        Args:
            layer: copper layer index
            extra_margin: additional cells beyond DRC clearance to discourage routing
                          near foreign pads (soft margin for heatmap suppression)
        """
        mask = np.ones((self.height, self.width), dtype=np.float32)
        
        if self.current_net_id is None:
            return mask
            
        # Look up design rules once
        default_rules = self.board.design_rules.get('default', {})
        net_rules = {}
        for net in self.board.nets:
            net_rules[net.id] = self.board.design_rules.get(
                net.net_class, default_rules
            )
        
        current_net = next((n for n in self.board.nets if n.id == self.current_net_id), None)
        exempt_net_ids = {self.current_net_id}
        if current_net and current_net.is_diff_pair and current_net.diff_pair_id is not None:
            exempt_net_ids.add(current_net.diff_pair_id)
        
        for pin in self.board.pins.values():
            if pin.net_id in exempt_net_ids:
                continue  # Skip our own net's pads
                
            # Block all layers for through-hole pins (-1), specific layer for SMD
            if pin.layer != -1 and pin.layer != layer:
                continue
                
            rules = net_rules.get(pin.net_id, default_rules)
            clearance = rules.get('clearance', 0.15)
            trace_width = rules.get('width', 0.15)
            
            pad_r = 3.0
            clearance_cells = clearance / self.resolution
            trace_r_cells = (trace_width / 2.0) / self.resolution
            # Use a wider exclusion radius than occupancy: pad + clearance + trace + extra margin
            avoid_r = pad_r + clearance_cells + trace_r_cells + extra_margin
            
            cx, cy = pin.global_x, pin.global_y
            r_ceil = int(avoid_r + 0.99)
            y_min = max(0, cy - r_ceil)
            y_max = min(self.height - 1, cy + r_ceil)
            x_min = max(0, cx - r_ceil)
            x_max = min(self.width - 1, cx + r_ceil)
            
            if pin.pad_shape == 0:  # circular
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if (x - cx)**2 + (y - cy)**2 <= avoid_r**2:
                            mask[y, x] = 0.0
            else:  # rectangular
                mask[y_min:y_max+1, x_min:x_max+1] = 0.0
                
        return mask

    def get_via_in_pad_boost(self) -> np.ndarray:
        """Returns a via probability boost map for via-in-pad routing.
        
        Marks the current net's own pad locations with 1.0 so A* treats them
        as excellent via placement sites (enables via-in-pad routing when
        source and target are on different layers).
        """
        boost = np.zeros((self.height, self.width), dtype=np.float32)
        
        if self.current_net_id is None:
            return boost
            
        for pin in self.board.pins.values():
            if pin.net_id == self.current_net_id:
                cx, cy = pin.global_x, pin.global_y
                pad_r = 3
                y_min = max(0, cy - pad_r)
                y_max = min(self.height - 1, cy + pad_r)
                x_min = max(0, cx - pad_r)
                x_max = min(self.width - 1, cx + pad_r)
                
                if pin.pad_shape == 0:  # circular
                    for y in range(y_min, y_max + 1):
                        for x in range(x_min, x_max + 1):
                            if (x - cx)**2 + (y - cy)**2 <= pad_r**2:
                                boost[y, x] = 1.0
                else:
                    boost[y_min:y_max+1, x_min:x_max+1] = 1.0
                    
        return boost

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
        self.current_net_id = net_id
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
        elif vias:
            self.routed_net_ids.add(vias[0].net_id)

    def rasterize_partial_move(self, x1, y1, l1, x2, y2, l2, net_class='default'):
        """Render a single partial step segment/via incrementally into raster.
        Does NOT append to self.traces or self.vias to avoid duplicates/failed route pollution.
        """
        # Look up default rules for width and via dimensions
        rules = self.board.design_rules.get(net_class, self.board.design_rules.get('default', {}))
        
        if l1 == l2:
            # Same layer trace segment
            trace_width = rules.get('width', 0.15)
            # Create temporary TraceSegment
            seg = TraceSegment(
                start_x=x1, start_y=y1,
                end_x=x2, end_y=y2,
                layer=l1, width=trace_width,
                net_id=self.current_net_id
            )
            # Rasterize trace into its layer and channel 10
            self._rasterize_trace(seg, channel=l1, val=1.0)
            self._rasterize_trace(seg, channel=10, val=1.0)
        else:
            # Layer transition (via)
            via_drill = rules.get('via_drill', 0.3)
            via_annular = rules.get('via_annular', 0.15)
            # Create temporary Via
            via = Via(
                x=x1, y=y1,
                from_layer=l1, to_layer=l2,
                drill_size=via_drill,
                annular_ring=via_annular,
                net_id=self.current_net_id
            )
            # Rasterize via on all traversed layers and channel 10
            r = int(round((via.drill_size / 2.0 + via.annular_ring) / self.resolution))
            for layer in range(min(l1, l2), max(l1, l2) + 1):
                self._rasterize_circle(via.x, via.y, radius=r, channel=layer, val=1.0)
                self._rasterize_circle(via.x, via.y, radius=r, channel=10, val=1.0)

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
