"""
rip_up_router.py
================
Stage 1 of the "completing router": a classical rip-up-and-reroute engine built on the existing
A* pathfinder. This is the executor — it guarantees legal, complete routing where a solution
exists — and it needs no ML. (The learned demand predictor plugs in later as a cost bias.)

Algorithm — negotiated congestion (the "PathFinder" approach):
  * Nets are routed in a given order (the order is an input; we do NOT decide it here).
  * Other nets' *pads and obstacles* are HARD blocks (a trace may never cross them), but other
    nets' *traces* are SOFT: cells may be shared temporarily. Sharing is discouraged by a cost
    that rises on contested cells.
  * Each iteration rips up and reroutes every net on the current cost map. Cells used by more
    than one net get a growing "history" penalty, so on the next pass the nets peel apart.
  * Repeat until no cell is shared (converged) or the iteration budget is hit.

Cost is injected through the A* pathfinder's existing heatmap channel: h_val in [0,1] is a
*preference* (high = cheap), so we map congestion -> low preference:
    h_val = 1 / (1 + alpha * (history + present_usage))
A free cell has h_val=1 (base cost); a congested cell approaches h_val=0 (up to
(1 + heatmap_weight)x cost). See AStarPathfinder.find_path.
"""

from typing import List, Dict, Optional, Any
import math
import numpy as np
import scipy.ndimage as ndimage

from pcb_router.routing.pathfinder import AStarPathfinder
from pcb_router.routing.trace_generator import TraceGenerator
from pcb_router.routing.meander import MeanderInserter
from pcb_router.env.board_state import BoardState


class RipUpRerouteRouter:
    def __init__(
        self,
        board,
        heatmap_weight: float = 20.0,
        congestion_alpha: float = 1.0,
        history_increment: float = 1.0,
        max_iterations: int = 20,
        resolution: float = 0.1,
    ):
        self.board = board
        self.H = board.height
        self.W = board.width
        self.L = board.num_layers
        self.resolution = resolution
        self.active_layers = list(range(self.L))

        # A* with a strong heatmap weight so congestion actually pushes traces apart.
        self.pathfinder = AStarPathfinder(heatmap_weight=heatmap_weight)
        self.trace_gen = TraceGenerator(resolution)
        self.meander_inserter = MeanderInserter(resolution)

        self.congestion_alpha = congestion_alpha
        self.history_increment = history_increment
        self.max_iterations = max_iterations

    # ── internals ────────────────────────────────────────────────────────────────
    def _pins_of(self, net):
        return [self.board.pins[pid] for pid in net.pin_ids]

    def _route_one_net(self, net, heatmaps, via_prob, base_state) -> Optional[list]:
        """Route a (possibly multi-pin) net sequentially through its pins.
        Returns the full cell path [(x, y, layer), ...] or None if any segment fails."""
        base_state.set_current_net(net.id)
        pins = self._pins_of(net)
        src = pins[0]
        curr = (src.global_x, src.global_y, src.layer if src.layer != -1 else 0)
        full_path = [curr]
        for p in pins[1:]:
            tgt = (p.global_x, p.global_y, p.layer if p.layer != -1 else 0)
            path, _cost = self.pathfinder.find_path(
                heatmaps, via_prob, curr, tgt, self.active_layers, board_state=base_state
            )
            if not path:
                return None
            full_path.extend(path[1:])
            curr = path[-1]
        return full_path

    def _stamp(self, usage: np.ndarray, cells: list, delta: int):
        for (x, y, l) in cells:
            xi, yi, li = int(x), int(y), int(l)
            if 0 <= li < self.L and 0 <= yi < self.H and 0 <= xi < self.W:
                usage[li, yi, xi] += delta

    # ── public API ───────────────────────────────────────────────────────────────
    def _get_footprint_kernel(self, net) -> np.ndarray:
        rules = self.board.design_rules.get(net.net_class, self.board.design_rules.get('default', {}))
        width = rules.get('width', 0.15)
        clearance = rules.get('clearance', 0.15) + 0.07 # Add 0.07mm discretization margin
        
        # If length matched, inflate clearance to reserve space
        if net.target_length > 0:
            clearance += 0.20 # 2 cells extra clearance (0.2mm)
            
        radius = (width / 2.0 + clearance) / self.resolution
        r_int = int(math.ceil(radius))
        
        # Create circular kernel of size (2*r_int+1, 2*r_int+1)
        size = 2 * r_int + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        for dx in range(-r_int, r_int + 1):
            for dy in range(-r_int, r_int + 1):
                if dx*dx + dy*dy <= radius*radius:
                    kernel[r_int + dy, r_int + dx] = 1.0
        return kernel

    def route(self, net_order: Optional[List[int]] = None) -> Dict[str, Any]:
        """Route all nets. `net_order` is a list of net ids (defaults to board order).
        Returns a result dict with the routed BoardState and completion stats."""
        if net_order is None:
            nets = list(self.board.nets)
        else:
            id_to_net = {n.id: n for n in self.board.nets}
            nets = [id_to_net[nid] for nid in net_order if nid in id_to_net]

        # Base state carries pads/obstacles/keep-outs only (no traces), so other nets' *traces*
        # stay soft. set_current_net(id) per net exempts that net's own pads.
        base_state = BoardState(self.board, self.resolution)

        history = np.zeros((self.L, self.H, self.W), dtype=np.float32)
        usage = np.zeros((self.L, self.H, self.W), dtype=np.float32)
        routes: Dict[int, Optional[list]] = {n.id: None for n in nets}
        via_prob = np.ones((self.H, self.W), dtype=np.float32)

        iteration = 0
        for iteration in range(self.max_iterations):
            routed_diff_pairs = set()
            for net in nets:
                if net.is_diff_pair and net.id in routed_diff_pairs:
                    continue

                # Rip up this net's current contribution so it doesn't penalize itself.
                if routes[net.id]:
                    self._stamp(usage, routes[net.id], -1)
                    routes[net.id] = None

                # If diff pair, also rip up the other net
                is_pair = net.is_diff_pair and net.diff_pair_id is not None
                if is_pair:
                    other_id = net.diff_pair_id
                    if routes[other_id]:
                        self._stamp(usage, routes[other_id], -1)
                        routes[other_id] = None

                # Get footprint kernel and spread usage congestion using convolve
                kernel = self._get_footprint_kernel(net)
                spread_usage = np.zeros_like(usage)
                for l in range(self.L):
                    spread_usage[l] = ndimage.convolve(usage[l], kernel, mode='constant', cval=0.0)

                cong = history + spread_usage
                heatmaps = (1.0 - self.congestion_alpha * cong).astype(np.float32)

                if is_pair:
                    other_net = next((n for n in self.board.nets if n.id == net.diff_pair_id), None)
                    if other_net:
                        # Determine Positive and Negative
                        if "DIFF_N" in net.name or "_N" in net.name:
                            p_net, n_net = other_net, net
                        else:
                            p_net, n_net = net, other_net

                        pins_p = self._pins_of(p_net)
                        pins_n = self._pins_of(n_net)

                        src_p = pins_p[0]
                        curr_p = (src_p.global_x, src_p.global_y, src_p.layer if src_p.layer != -1 else 0)
                        tgt_p = pins_p[1]
                        curr_tgt_p = (tgt_p.global_x, tgt_p.global_y, tgt_p.layer if tgt_p.layer != -1 else 0)

                        src_n = pins_n[0]
                        curr_n = (src_n.global_x, src_n.global_y, src_n.layer if src_n.layer != -1 else 0)
                        tgt_n = pins_n[1]
                        curr_tgt_n = (tgt_n.global_x, tgt_n.global_y, tgt_n.layer if tgt_n.layer != -1 else 0)

                        rules = self.board.design_rules.get(net.net_class, self.board.design_rules.get('default', {}))
                        width = rules.get('width', 0.12)
                        clearance = rules.get('clearance', 0.12)
                        gap_cells = int(math.ceil((width + clearance) / self.resolution))

                        base_state.set_current_net(p_net.id)
                        path_p, path_n, _cost = self.pathfinder.find_path_coupled(
                            heatmaps, via_prob, curr_p, curr_tgt_p, curr_n, curr_tgt_n,
                            self.active_layers, board_state=base_state, gap_cells=gap_cells
                        )

                        routes[p_net.id] = path_p
                        routes[n_net.id] = path_n

                        if path_p:
                            self._stamp(usage, path_p, +1)
                        if path_n:
                            self._stamp(usage, path_n, +1)

                        routed_diff_pairs.add(p_net.id)
                        routed_diff_pairs.add(n_net.id)
                else:
                    # Single net routing
                    path = self._route_one_net(net, heatmaps, via_prob, base_state)
                    routes[net.id] = path
                    if path:
                        self._stamp(usage, path, +1)

            shared = usage > 1.5  # a centerline cell claimed by >= 2 nets
            failed = [nid for nid, p in routes.items() if p is None]
            if not shared.any() and not failed:
                break
            # Grow the penalty on contested cells so the nets separate next pass.
            history[shared] += self.history_increment

        # Materialize the final routes into a board state (sequential; reports residual DRC).
        result_state = BoardState(self.board, self.resolution)
        for net in nets:
            path = routes[net.id]
            if not path:
                continue
            result_state.set_current_net(net.id)
            new_traces, new_vias, _ = self.trace_gen.generate_traces(
                path, net.id, self.board.design_rules, net.net_class,
                result_state.traces, result_state.vias,
            )

            # Post-route length tuning with meander inserter
            if net.target_length > 0:
                rules = self.board.design_rules.get(net.net_class, self.board.design_rules.get('default', {}))
                clearance = rules.get('clearance', 0.15)
                tuned_traces, actual_len = self.meander_inserter.insert_meanders(
                    new_traces, net.target_length, net.length_tolerance,
                    result_state.traces, clearance
                )
                new_traces = tuned_traces

            result_state.add_routed_trace(new_traces, new_vias)

        completed = sum(1 for p in routes.values() if p)
        total = len(nets)
        return {
            "board_state": result_state,
            "routes": routes,
            "completed": completed,
            "total": total,
            "completion_rate": (completed / total) if total else 0.0,
            "iterations": iteration + 1,
            "shared_cells": int((usage > 1.5).sum()),
            "converged": bool((usage <= 1.5).all() and completed == total),
        }
