import math
from typing import List, Tuple, Dict, Any
from pcb_router.routing.trace_generator import TraceSegment, TraceGenerator

class MeanderInserter:
    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.trace_gen = TraceGenerator(resolution)

    def calculate_trace_length(self, traces: List[TraceSegment]) -> float:
        """Calculate total physical length of trace segments in mm"""
        length = 0.0
        for seg in traces:
            dx = (seg.end_x - seg.start_x) * self.resolution
            dy = (seg.end_y - seg.start_y) * self.resolution
            length += math.hypot(dx, dy)
        return length

    def insert_meanders(
        self,
        traces: List[TraceSegment],
        target_length: float, # in mm
        tolerance: float,     # in mm
        existing_traces: List[TraceSegment],
        min_clearance: float  # in mm
    ) -> Tuple[List[TraceSegment], float]:
        """
        Inserts serpentine meanders to increase trace length to target_length
        """
        current_length = self.calculate_trace_length(traces)
        deficit = target_length - current_length
        
        # If length is already sufficient or within tolerance, return original traces
        if deficit <= tolerance:
            return traces, current_length
            
        # Find candidates for meanders (straight horizontal or vertical segments)
        # Sort by length descending to meander the longest segment first
        candidates = []
        for i, seg in enumerate(traces):
            dx = abs(seg.end_x - seg.start_x)
            dy = abs(seg.end_y - seg.start_y)
            length = math.hypot(dx, dy) * self.resolution
            
            # Straight horizontal or vertical segments only, must be long enough
            if (dx == 0 or dy == 0) and length > 2.0:
                candidates.append((i, seg, length))
                
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        modified_traces = list(traces)
        
        # Work through candidates to add length
        for idx, seg, length in candidates:
            current_length = self.calculate_trace_length(modified_traces)
            deficit = target_length - current_length
            if deficit <= tolerance:
                break
                
            # Let's construct a meander for this segment
            # We split the segment into cycles.
            # E.g. segment is from (x1, y1) to (x2, y2).
            # Let's say horizontal. Direction is along X.
            is_horizontal = (seg.start_y == seg.end_y)
            
            x1, y1 = seg.start_x, seg.start_y
            x2, y2 = seg.end_x, seg.end_y
            
            # Ensure coordinates go from low to high for simplicity
            flipped = False
            if is_horizontal and x1 > x2:
                x1, x2 = x2, x1
                flipped = True
            elif not is_horizontal and y1 > y2:
                y1, y2 = y2, y1
                flipped = True
                
            seg_len_cells = (x2 - x1) if is_horizontal else (y2 - y1)
            
            # Define meander cycle spacing (e.g. 0.8mm = 8 cells)
            cycle_spacing = 8
            num_cycles = max(1, int(seg_len_cells // (cycle_spacing * 2)))
            
            # Each cycle adds deficit / num_cycles of length (in mm)
            # Cycle adds 2 * amplitude_mm of length
            added_per_cycle = deficit / num_cycles
            amplitude_mm = added_per_cycle / 2.0
            
            # Limit amplitude to prevent ridiculously long zigzags, e.g. max 3.0mm
            amplitude_mm = min(3.0, max(0.4, amplitude_mm))
            amplitude_cells = int(round(amplitude_mm / self.resolution))
            
            # Generate meander segments
            meander_segs = []
            
            curr_pos = x1 if is_horizontal else y1
            step = seg_len_cells / (2 * num_cycles + 1)
            
            # Start generating
            for c in range(num_cycles):
                # Flat segment
                next_flat = int(round(x1 + (2 * c + 0.5) * step)) if is_horizontal else int(round(y1 + (2 * c + 0.5) * step))
                
                # Up (or left)
                # Down (or right)
                # We alternate directions to avoid drifting
                direction = 1 if c % 2 == 0 else -1
                
                if is_horizontal:
                    # Flat to start of pulse
                    p_start = next_flat
                    p_end = p_start + int(round(step))
                    
                    meander_segs.append(TraceSegment(x1 + int(round(2*c*step)), y1, p_start, y1, seg.width, seg.layer, seg.net_id))
                    # Pulse Up/Down
                    meander_segs.append(TraceSegment(p_start, y1, p_start, y1 + direction * amplitude_cells, seg.width, seg.layer, seg.net_id))
                    # Pulse Flat
                    meander_segs.append(TraceSegment(p_start, y1 + direction * amplitude_cells, p_end, y1 + direction * amplitude_cells, seg.width, seg.layer, seg.net_id))
                    # Pulse Back
                    meander_segs.append(TraceSegment(p_end, y1 + direction * amplitude_cells, p_end, y1, seg.width, seg.layer, seg.net_id))
                else:
                    p_start = next_flat
                    p_end = p_start + int(round(step))
                    
                    meander_segs.append(TraceSegment(x1, y1 + int(round(2*c*step)), x1, p_start, seg.width, seg.layer, seg.net_id))
                    # Pulse Left/Right
                    meander_segs.append(TraceSegment(x1, p_start, x1 + direction * amplitude_cells, p_start, seg.width, seg.layer, seg.net_id))
                    # Pulse Flat
                    meander_segs.append(TraceSegment(x1 + direction * amplitude_cells, p_start, x1 + direction * amplitude_cells, p_end, seg.width, seg.layer, seg.net_id))
                    # Pulse Back
                    meander_segs.append(TraceSegment(x1 + direction * amplitude_cells, p_end, x1, p_end, seg.width, seg.layer, seg.net_id))
                    
            # Final flat segment to end
            last_flat_start = int(round(x1 + (2 * num_cycles) * step)) if is_horizontal else int(round(y1 + (2 * num_cycles) * step))
            if is_horizontal:
                meander_segs.append(TraceSegment(last_flat_start, y1, x2, y1, seg.width, seg.layer, seg.net_id))
            else:
                meander_segs.append(TraceSegment(x1, last_flat_start, x1, y2, seg.width, seg.layer, seg.net_id))
                
            # Verify clearance of new meander segments
            clearance_ok = True
            for m_seg in meander_segs:
                for ex_seg in existing_traces:
                    if ex_seg.net_id != seg.net_id and ex_seg.layer == m_seg.layer:
                        dist = self.trace_gen._segment_to_segment_distance(m_seg, ex_seg)
                        if dist < min_clearance + (m_seg.width + ex_seg.width) / 2.0:
                            clearance_ok = False
                            break
                if not clearance_ok:
                    break
                    
            if clearance_ok:
                # Replace the original segment with the meandered segments
                modified_traces.pop(idx)
                # Insert all new segments at the index
                for offset, m_seg in enumerate(meander_segs):
                    modified_traces.insert(idx + offset, m_seg)
            else:
                # Keep original and try next candidate
                pass
                
        final_length = self.calculate_trace_length(modified_traces)
        return modified_traces, final_length

    def match_differential_pair(
        self,
        traces_p: List[TraceSegment],
        traces_n: List[TraceSegment],
        target_length: float,
        tolerance: float,
        existing_traces: List[TraceSegment],
        min_clearance: float
    ) -> Tuple[List[TraceSegment], List[TraceSegment], float, float]:
        """Tuning both traces of a differential pair to match length"""
        # Tune positive line
        tuned_p, len_p = self.insert_meanders(traces_p, target_length, tolerance, existing_traces + traces_n, min_clearance)
        # Tune negative line
        tuned_n, len_n = self.insert_meanders(traces_n, target_length, tolerance, existing_traces + tuned_p, min_clearance)
        
        return tuned_p, tuned_n, len_p, len_n
