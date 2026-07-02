import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class TraceSegment:
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    width: float       # in mm
    layer: int
    net_id: int

@dataclass
class Via:
    x: int
    y: int
    from_layer: int
    to_layer: int
    drill_size: float   # in mm
    annular_ring: float # in mm
    net_id: int

class TraceGenerator:
    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution # 0.1mm per grid cell

    def grid_to_mm(self, grid_units: float) -> float:
        return grid_units * self.resolution

    def mm_to_grid(self, mm_units: float) -> int:
        return int(round(mm_units / self.resolution))

    def simplify_waypoints(self, waypoints: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Merge collinear adjacent waypoints on the same layer"""
        if len(waypoints) < 3:
            return waypoints
            
        simplified = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            prev_wp = simplified[-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i+1]
            
            # If same layer and directions are collinear
            if prev_wp[2] == curr_wp[2] == next_wp[2]:
                dx1, dy1 = curr_wp[0] - prev_wp[0], curr_wp[1] - prev_wp[1]
                dx2, dy2 = next_wp[0] - curr_wp[0], next_wp[1] - curr_wp[1]
                
                # Check collinearity: cross product is 0 and signs match
                # Also, we check if they are moving in the same direction
                len1 = math.hypot(dx1, dy1)
                len2 = math.hypot(dx2, dy2)
                
                if len1 > 0 and len2 > 0:
                    cross_product = dx1 * dy2 - dy1 * dx2
                    dot_product = dx1 * dx2 + dy1 * dy2
                    
                    if abs(cross_product) < 1e-5 and dot_product > 0:
                        # Collinear, skip curr_wp and continue
                        continue
                        
            simplified.append(curr_wp)
            
        simplified.append(waypoints[-1])
        return simplified

    def generate_traces(
        self,
        waypoints: List[Tuple[int, int, int]],
        net_id: int,
        design_rules: Dict[str, Any],
        net_class: str = 'default',
        existing_segments: List[TraceSegment] = None,
        existing_vias: List[Via] = None
    ) -> Tuple[List[TraceSegment], List[Via], List[Dict[str, Any]]]:
        """
        Convert waypoints to TraceSegments and Vias, checking design rules
        """
        if not waypoints:
            return [], [], []
            
        rules = design_rules.get(net_class, design_rules.get('default'))
        width = rules.get('width', 0.15)
        clearance = rules.get('clearance', 0.15)
        via_drill = rules.get('via_drill', 0.3)
        via_annular = rules.get('via_annular', 0.15)
        
        simplified_wps = self.simplify_waypoints(waypoints)
        
        new_segments = []
        new_vias = []
        violations = []
        
        for i in range(len(simplified_wps) - 1):
            w1 = simplified_wps[i]
            w2 = simplified_wps[i+1]
            
            # Same layer trace
            if w1[2] == w2[2]:
                seg = TraceSegment(
                    start_x=w1[0], start_y=w1[1],
                    end_x=w2[0], end_y=w2[1],
                    width=width, layer=w1[2], net_id=net_id
                )
                new_segments.append(seg)
            else:
                # Layer transition via
                via = Via(
                    x=w1[0], y=w1[1],
                    from_layer=w1[2], to_layer=w2[2],
                    drill_size=via_drill, annular_ring=via_annular,
                    net_id=net_id
                )
                new_vias.append(via)
                
                # If there's still a spatial move to be done, handle it
                if w1[0] != w2[0] or w1[1] != w2[1]:
                    # Create trace on the target layer
                    seg = TraceSegment(
                        start_x=w1[0], start_y=w1[1],
                        end_x=w2[0], end_y=w2[1],
                        width=width, layer=w2[2], net_id=net_id
                    )
                    new_segments.append(seg)
                    
        # Check clearance against existing geometry
        if existing_segments is None:
            existing_segments = []
        if existing_vias is None:
            existing_vias = []
            
        # 1. Check new segments against existing segments
        for seg in new_segments:
            for ex_seg in existing_segments:
                if seg.net_id != ex_seg.net_id and seg.layer == ex_seg.layer:
                    dist = self._segment_to_segment_distance(seg, ex_seg)
                    min_allowed = clearance + (seg.width + ex_seg.width) / 2.0
                    if dist < min_allowed:
                        violations.append({
                            'type': 'clearance',
                            'severity': 'error',
                            'description': f'Clearance violation between net {seg.net_id} and {ex_seg.net_id}: {dist:.3f}mm < {min_allowed:.3f}mm',
                            'x': int((seg.start_x + seg.end_x) / 2),
                            'y': int((seg.start_y + seg.end_y) / 2),
                            'layer': seg.layer,
                            'net_id_a': seg.net_id,
                            'net_id_b': ex_seg.net_id
                        })
                        
        # 2. Check new vias against existing segments and vias
        for via in new_vias:
            # Check via annular ring clearance to other nets
            via_radius = via.drill_size / 2.0 + via.annular_ring
            
            for ex_seg in existing_segments:
                if via.net_id != ex_seg.net_id:
                    # If segment is on one of the layers the via goes through
                    v_layers = range(min(via.from_layer, via.to_layer), max(via.from_layer, via.to_layer) + 1)
                    if ex_seg.layer in v_layers:
                        dist = self._point_to_segment_distance(via.x, via.y, ex_seg)
                        min_allowed = clearance + via_radius + ex_seg.width / 2.0
                        if dist < min_allowed:
                            violations.append({
                                'type': 'clearance',
                                'severity': 'error',
                                'description': f'Clearance violation between via (net {via.net_id}) and trace (net {ex_seg.net_id}): {dist:.3f}mm < {min_allowed:.3f}mm',
                                'x': via.x, 'y': via.y, 'layer': ex_seg.layer,
                                'net_id_a': via.net_id, 'net_id_b': ex_seg.net_id
                            })
                            
            for ex_via in existing_vias:
                if via.net_id != ex_via.net_id:
                    # Check overlap of layer ranges
                    v1_range = set(range(min(via.from_layer, via.to_layer), max(via.from_layer, via.to_layer) + 1))
                    v2_range = set(range(min(ex_via.from_layer, ex_via.to_layer), max(ex_via.from_layer, ex_via.to_layer) + 1))
                    
                    if v1_range.intersection(v2_range):
                        dx = self.grid_to_mm(via.x - ex_via.x)
                        dy = self.grid_to_mm(via.y - ex_via.y)
                        dist = math.hypot(dx, dy)
                        ex_via_radius = ex_via.drill_size / 2.0 + ex_via.annular_ring
                        min_allowed = clearance + via_radius + ex_via_radius
                        if dist < min_allowed:
                            violations.append({
                                'type': 'clearance',
                                'severity': 'error',
                                'description': f'Clearance violation between via (net {via.net_id}) and via (net {ex_via.net_id}): {dist:.3f}mm < {min_allowed:.3f}mm',
                                'x': via.x, 'y': via.y, 'layer': list(v1_range.intersection(v2_range))[0],
                                'net_id_a': via.net_id, 'net_id_b': ex_via.net_id
                            })
                            
        return new_segments, new_vias, violations

    def _point_to_segment_distance(self, px: int, py: int, seg: TraceSegment) -> float:
        """Find distance in mm from point (px, py) in grid units to a segment"""
        # Convert all to mm
        x = self.grid_to_mm(px)
        y = self.grid_to_mm(py)
        x1 = self.grid_to_mm(seg.start_x)
        y1 = self.grid_to_mm(seg.start_y)
        x2 = self.grid_to_mm(seg.end_x)
        y2 = self.grid_to_mm(seg.end_y)
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
            
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.hypot(x - closest_x, y - closest_y)

    def _segment_to_segment_distance(self, seg1: TraceSegment, seg2: TraceSegment) -> float:
        """Find minimum distance in mm between two TraceSegments"""
        # Check intersection in grid space
        if self._segments_intersect(seg1, seg2):
            return 0.0
            
        # Distance is min of endpoint-to-segment distances
        d1 = self._point_to_segment_distance(seg1.start_x, seg1.start_y, seg2)
        d2 = self._point_to_segment_distance(seg1.end_x, seg1.end_y, seg2)
        d3 = self._point_to_segment_distance(seg2.start_x, seg2.start_y, seg1)
        d4 = self._point_to_segment_distance(seg2.end_x, seg2.end_y, seg1)
        
        return min(d1, d2, d3, d4)

    def _segments_intersect(self, s1: TraceSegment, s2: TraceSegment) -> bool:
        """Check if two line segments S1 and S2 intersect in 2D"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        A = (s1.start_x, s1.start_y)
        B = (s1.end_x, s1.end_y)
        C = (s2.start_x, s2.start_y)
        D = (s2.end_x, s2.end_y)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
