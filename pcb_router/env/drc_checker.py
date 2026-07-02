import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set
from pcb_router.data.board_generator import Board, Pin, Net, Obstacle
from pcb_router.routing.trace_generator import TraceSegment, Via, TraceGenerator
from pcb_router.env.board_state import BoardState

@dataclass
class DRCViolation:
    type: str          # 'clearance', 'short_circuit', 'width', 'keep_out', 'via_annular_ring', 'unconnected'
    severity: str      # 'error', 'warning'
    x: int
    y: int
    layer: int
    description: str
    net_id_a: int
    net_id_b: int

class DRCChecker:
    def __init__(self, design_rules: Dict[str, Any], resolution: float = 0.1):
        self.design_rules = design_rules
        self.resolution = resolution
        self.trace_gen = TraceGenerator(resolution)

    def check_all(self, board_state: BoardState, traces: List[TraceSegment], vias: List[Via], board: Board) -> List[DRCViolation]:
        """
        Run all design rule checks on current board layout
        """
        violations = []
        
        # 1. Check Short Circuits (different nets overlapping on same layer)
        violations.extend(self._check_short_circuits(traces, vias, board.pins))
        
        # 2. Check Clearances (different nets too close but not shorted)
        violations.extend(self._check_clearances(traces, vias, board.pins))
        
        # 3. Check Width Rules (trace widths vs design rules)
        violations.extend(self._check_widths(traces, board.nets))
        
        # 4. Check Keep Out Zones
        violations.extend(self._check_keep_outs(traces, vias, board.keep_out_zones))
        
        # 5. Check Via Rules
        violations.extend(self._check_vias(vias))
        
        # 6. Check Electrical Connectivity
        violations.extend(self._check_connectivity(board, traces, vias))
        
        return violations

    def _check_short_circuits(self, traces: List[TraceSegment], vias: List[Via], pins: Dict[int, Pin]) -> List[DRCViolation]:
        violations = []
        # Check trace-to-trace shorts
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                t1, t2 = traces[i], traces[j]
                if t1.net_id != t2.net_id and t1.layer == t2.layer:
                    dist = self.trace_gen._segment_to_segment_distance(t1, t2)
                    if dist <= 0.001: # actual intersection / overlap
                        violations.append(DRCViolation(
                            type='short_circuit', severity='error',
                            x=int((t1.start_x + t1.end_x)/2), y=int((t1.start_y + t1.end_y)/2),
                            layer=t1.layer,
                            description=f"Short circuit: net {t1.net_id} overlaps trace of net {t2.net_id}",
                            net_id_a=t1.net_id, net_id_b=t2.net_id
                        ))
        return violations

    def _check_clearances(self, traces: List[TraceSegment], vias: List[Via], pins: Dict[int, Pin]) -> List[DRCViolation]:
        violations = []
        # Check trace-to-trace clearance
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                t1, t2 = traces[i], traces[j]
                if t1.net_id != t2.net_id and t1.layer == t2.layer:
                    dist = self.trace_gen._segment_to_segment_distance(t1, t2)
                    # Min clearance is default or specific class rule
                    rule_1 = self.design_rules.get('default') # simplifed
                    min_clearance = rule_1['clearance']
                    allowed_dist = min_clearance + (t1.width + t2.width) / 2.0
                    
                    if 0.001 < dist < allowed_dist:
                        violations.append(DRCViolation(
                            type='clearance', severity='error',
                            x=int((t1.start_x + t1.end_x)/2), y=int((t1.start_y + t1.end_y)/2),
                            layer=t1.layer,
                            description=f"Clearance violation: net {t1.net_id} and {t2.net_id} too close ({dist:.3f}mm < {allowed_dist:.3f}mm)",
                            net_id_a=t1.net_id, net_id_b=t2.net_id
                        ))
        return violations

    def _check_widths(self, traces: List[TraceSegment], nets: List[Net]) -> List[DRCViolation]:
        violations = []
        for t in traces:
            net = next((n for n in nets if n.id == t.net_id), None)
            if net:
                rules = self.design_rules.get(net.net_class, self.design_rules.get('default'))
                target_w = rules.get('width', 0.15)
                if t.width < target_w - 0.001:
                    violations.append(DRCViolation(
                        type='width', severity='error',
                        x=t.start_x, y=t.start_y, layer=t.layer,
                        description=f"Trace width violation: net {t.net_id} width {t.width}mm < target {target_w}mm",
                        net_id_a=t.net_id, net_id_b=0
                    ))
        return violations

    def _check_keep_outs(self, traces: List[TraceSegment], vias: List[Via], keep_outs: List[Obstacle]) -> List[DRCViolation]:
        violations = []
        for ko in keep_outs:
            # Check traces
            for t in traces:
                if t.layer == ko.layer or ko.layer == -1:
                    # check intersection with rect
                    if self._segment_intersects_rect(t, ko):
                        violations.append(DRCViolation(
                            type='keep_out', severity='error',
                            x=t.start_x, y=t.start_y, layer=t.layer,
                            description=f"Keep-out violation: trace net {t.net_id} crosses keep-out zone",
                            net_id_a=t.net_id, net_id_b=0
                        ))
            # Check vias
            for via in vias:
                # via spans layers from min to max
                v_layers = range(min(via.from_layer, via.to_layer), max(via.from_layer, via.to_layer) + 1)
                if ko.layer == -1 or ko.layer in v_layers:
                    if ko.x <= via.x <= ko.x + ko.width and ko.y <= via.y <= ko.y + ko.height:
                        violations.append(DRCViolation(
                            type='keep_out', severity='error',
                            x=via.x, y=via.y, layer=via.from_layer,
                            description=f"Keep-out violation: via net {via.net_id} placed inside keep-out zone",
                            net_id_a=via.net_id, net_id_b=0
                        ))
        return violations

    def _check_vias(self, vias: List[Via]) -> List[DRCViolation]:
        violations = []
        for via in vias:
            rules = self.design_rules.get('default')
            min_drill = rules.get('via_drill', 0.3)
            min_annular = rules.get('via_annular', 0.15)
            
            if via.drill_size < min_drill - 0.001:
                violations.append(DRCViolation(
                    type='via_annular_ring', severity='error',
                    x=via.x, y=via.y, layer=via.from_layer,
                    description=f"Via drill violation: net {via.net_id} drill size {via.drill_size}mm < {min_drill}mm",
                    net_id_a=via.net_id, net_id_b=0
                ))
            if via.annular_ring < min_annular - 0.001:
                violations.append(DRCViolation(
                    type='via_annular_ring', severity='error',
                    x=via.x, y=via.y, layer=via.from_layer,
                    description=f"Via annular ring violation: net {via.net_id} ring {via.annular_ring}mm < {min_annular}mm",
                    net_id_a=via.net_id, net_id_b=0
                ))
        return violations

    def _check_connectivity(self, board: Board, traces: List[TraceSegment], vias: List[Via]) -> List[DRCViolation]:
        violations = []
        # Check connectivity for each net
        for net in board.nets:
            net_pins = [board.pins[pid] for pid in net.pin_ids]
            if len(net_pins) < 2:
                continue
                
            net_traces = [t for t in traces if t.net_id == net.id]
            net_vias = [v for v in vias if v.net_id == net.id]
            
            # Build union find over pins, trace endpoints, and vias
            parent = {}
            def find(x):
                if parent[x] == x:
                    return x
                parent[x] = find(parent[x])
                return parent[x]
                
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry
                    
            # Initialize union find nodes
            # Pin nodes
            for pin in net_pins:
                parent[(pin.global_x, pin.global_y, pin.layer)] = (pin.global_x, pin.global_y, pin.layer)
                
            # Trace segment endpoints
            for t in net_traces:
                parent[(t.start_x, t.start_y, t.layer)] = (t.start_x, t.start_y, t.layer)
                parent[(t.end_x, t.end_y, t.layer)] = (t.end_x, t.end_y, t.layer)
                
            # Via nodes
            for v in net_vias:
                # A via connects across all layers it spans, so we link all points (v.x, v.y, l) together
                for l in range(min(v.from_layer, v.to_layer), max(v.from_layer, v.to_layer) + 1):
                    parent[(v.x, v.y, l)] = (v.x, v.y, l)
                    
            # Connect trace segments endpoints
            for t in net_traces:
                union((t.start_x, t.start_y, t.layer), (t.end_x, t.end_y, t.layer))
                
            # Connect vias across their layer range
            for v in net_vias:
                p_base = (v.x, v.y, v.from_layer)
                for l in range(min(v.from_layer, v.to_layer), max(v.from_layer, v.to_layer) + 1):
                    union(p_base, (v.x, v.y, l))
                    
            # Connect overlapping elements
            # Connect pins to traces/vias if they sit at the same coordinate
            all_keys = list(parent.keys())
            for i in range(len(all_keys)):
                for j in range(i + 1, len(all_keys)):
                    k1, k2 = all_keys[i], all_keys[j]
                    # If same grid location and same layer (or overlaps)
                    if k1[0] == k2[0] and k1[1] == k2[1] and k1[2] == k2[2]:
                        union(k1, k2)
                        
            # Verify if all net pins belong to the same component
            pin_roots = []
            for pin in net_pins:
                pin_key = (pin.global_x, pin.global_y, pin.layer)
                if pin_key in parent:
                    pin_roots.append(find(pin_key))
                else:
                    pin_roots.append(None)
                    
            # If any pin is not found, or roots differ, we have unconnected segments
            unconnected = False
            first_root = pin_roots[0]
            if first_root is None:
                unconnected = True
            else:
                for r in pin_roots[1:]:
                    if r is None or r != first_root:
                        unconnected = True
                        break
                        
            if unconnected:
                violations.append(DRCViolation(
                    type='unconnected', severity='error',
                    x=net_pins[0].global_x, y=net_pins[0].global_y, layer=net_pins[0].layer,
                    description=f"Unconnected Net: pins of net {net.id} are not fully routed/connected",
                    net_id_a=net.id, net_id_b=0
                ))
                
        return violations

    def _segment_intersects_rect(self, t: TraceSegment, rect: Obstacle) -> bool:
        # Check if segment crosses rectangular boundary
        rx1, rx2 = rect.x, rect.x + rect.width
        ry1, ry2 = rect.y, rect.y + rect.height
        
        # Check endpoints
        if (rx1 <= t.start_x <= rx2 and ry1 <= t.start_y <= ry2) or \
           (rx1 <= t.end_x <= rx2 and ry1 <= t.end_y <= ry2):
            return True
            
        # Check intersection with 4 sides of the rectangle
        def intersect(p1, p2, p3, p4):
            # standard intersection check
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
            
        p1, p2 = (t.start_x, t.start_y), (t.end_x, t.end_y)
        sides = [
            ((rx1, ry1), (rx2, ry1)), # Bottom
            ((rx2, ry1), (rx2, ry2)), # Right
            ((rx2, ry2), (rx1, ry2)), # Top
            ((rx1, ry2), (rx1, ry1))  # Left
        ]
        
        for s1, s2 in sides:
            if intersect(p1, p2, s1, s2):
                return True
                
        return False

    def get_violation_rate(self, violations: List[DRCViolation], total_nets: int) -> float:
        if total_nets == 0:
            return 0.0
        # Average number of violations per net
        return len(violations) / total_nets
