import numpy as np
import heapq
import math

class AStarPathfinder:
    def __init__(self, direction_change_penalty: float = 15.0, base_via_cost: float = 15.0, heatmap_weight: float = 10.0):
        self.direction_change_penalty = direction_change_penalty
        self.base_via_cost = base_via_cost
        self.heatmap_weight = heatmap_weight
        # Occupancy threshold above which a cell is treated as blocked
        self.obstacle_threshold = 0.5
        
        # 8-directional moves: (dx, dy, is_diagonal)
        self.moves = [
            (0, 1, False),   # N
            (0, -1, False),  # S
            (1, 0, False),   # E
            (-1, 0, False),  # W
            (1, 1, True),    # NE
            (1, -1, True),   # SE
            (-1, 1, True),   # NW
            (-1, -1, True)   # SW
        ]

    def _heuristic(self, p1, p2):
        """3D heuristic: 2D distance + layer transition cost estimation"""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        # Use diagonal distance for 2D
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        d2d = (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)
        
        # If either coordinate is on layer -1 (through-hole), layer transition cost to connect is 0
        z_dist = 0 if (z1 == -1 or z2 == -1) else abs(z1 - z2)
        d3d = d2d + z_dist * self.base_via_cost
        # Scale heuristic to match step cost scale, preventing search timeouts (200k max_iterations)
        return d3d * (1.0 + self.heatmap_weight)

    def find_path(self, heatmaps, via_prob, source, target, active_layers, max_iterations=200000, board_state=None):
        """
        Find path from source (x, y, layer) to target (x, y, layer)
        Args:
            heatmaps: numpy array of shape (N_layers, H, W) — high values = PREFERRED routing areas
            via_prob: numpy array of shape (H, W) containing via placing confidence [0, 1]
            source: tuple (x, y, layer)
            target: tuple (x, y, layer)
            active_layers: list of active layer indices
            board_state: optional BoardState for obstacle blocking
        """
        N_layers, H, W = heatmaps.shape
        active_layers_set = set(active_layers)

        # Early-exit: source/target out of board bounds
        sx, sy, sl = source
        tx, ty, tl = target
        if not (0 <= sx < W and 0 <= sy < H):
            return None, float('inf')
        if not (0 <= tx < W and 0 <= ty < H):
            return None, float('inf')
        # Early-exit: already at destination
        if (tl == -1 and sx == tx and sy == ty) or (tl != -1 and source == target):
            p_sl = 0 if sl == -1 else sl
            return [(sx, sy, p_sl)], 0.0

        # Pre-build obstacle maps per layer (cells that are blocked to traversal).
        # Source and target cells are always passable regardless of occupancy.
        exempt = {(sx, sy), (tx, ty)}
        obstacle_maps = {}  # layer -> 2D bool array, True = blocked
        if board_state is not None:
            for l in active_layers:
                occ = board_state.get_occupancy(l)  # (H, W) float, 1.0 = occupied
                obstacle_maps[l] = occ >= self.obstacle_threshold

        # Compute via clearance offsets
        via_clearance_offsets = []
        if board_state is not None:
            rules = board_state.board.design_rules.get('default', {})
            via_drill = rules.get('via_drill', 0.3)
            via_annular = rules.get('via_annular', 0.15)
            clearance = rules.get('clearance', 0.15)
            # Via total radius (drill + 2 * annular_ring) / 2
            via_radius = (via_drill + 2 * via_annular) / 2.0
            # Total clearance boundary from via center
            required_clearance = via_radius + clearance
            
            # Convert to grid units (resolution = 0.1mm)
            res = getattr(board_state, 'resolution', 0.1)
            radius_cells = int(math.ceil(required_clearance / res))
            
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx*dx + dy*dy <= radius_cells*radius_cells:
                        via_clearance_offsets.append((dx, dy))
        else:
            via_clearance_offsets = [(0, 0)]

        # Priority Queue: stores tuples of (f_score, g_score, (x, y, layer), last_direction)
        pq = []
        if sl == -1:
            for l in active_layers:
                start_node = (sx, sy, l)
                heapq.heappush(pq, (self._heuristic(start_node, target), 0.0, start_node, (0, 0, 0)))
        else:
            start_node = source
            heapq.heappush(pq, (self._heuristic(start_node, target), 0.0, start_node, (0, 0, 0)))
        
        came_from = {}  # key: (pos, last_dir) -> parent (pos, last_dir)
        visited = {}    # key: (pos, last_dir) -> g_score
        
        iterations = 0
        min_h_seen = min(self._heuristic((sx, sy, l if sl == -1 else sl), target) for l in active_layers)
        total_pushes = len(pq)
        total_pops = 0
        first_pops = []
        last_pops = []
        
        while pq and iterations < max_iterations:
            iterations += 1
            f, g, curr, last_dir = heapq.heappop(pq)
            total_pops += 1
            
            # Record some popped states for debugging
            state_info = (f"{f:.2f}", f"{g:.2f}", curr, last_dir)
            if len(first_pops) < 15:
                first_pops.append(state_info)
            else:
                last_pops.append(state_info)
                if len(last_pops) > 15:
                    last_pops.pop(0)
            
            h_curr = self._heuristic(curr, target)
            if h_curr < min_h_seen:
                min_h_seen = h_curr
            
            # Target reached check
            is_reached = (curr[0] == tx and curr[1] == ty) if tl == -1 else (curr == target)
            if is_reached:
                # Reconstruct path
                path = []
                temp = (curr, last_dir)
                while temp in came_from:
                    path.append(temp[0])
                    temp = came_from[temp]
                path.append(temp[0])  # Append the physical start node
                path.reverse()
                return path, g
                
            state_key = (curr, last_dir)
            if state_key in visited and visited[state_key] < g:
                continue
            visited[state_key] = g
            
            cx, cy, cl = curr
            
            # 1. Check Spatial Neighbors (same layer)
            for dx, dy, is_diag in self.moves:
                nx, ny = cx + dx, cy + dy
                
                # Check boundaries
                if 0 <= nx < W and 0 <= ny < H:
                    # Skip obstacle cells (but always allow source/target)
                    if cl in obstacle_maps and obstacle_maps[cl][ny, nx] and (nx, ny) not in exempt:
                        continue
                    
                    # FIX: high heatmap value = policy PREFERS this cell → low cost
                    # Invert: cost = 1 + w*(1 - h_val) so h_val=1 → cost=1, h_val=0 → cost=1+w
                    h_val = heatmaps[cl, ny, nx]
                    step_len = math.sqrt(2.0) if is_diag else 1.0
                    step_cost = step_len * (1.0 + self.heatmap_weight * (1.0 - h_val))
                    
                    # Direction change penalty
                    new_dir = (dx, dy, 0)
                    if last_dir != (0, 0, 0) and last_dir != new_dir:
                        step_cost += self.direction_change_penalty
                        
                    next_g = g + step_cost
                    next_pos = (nx, ny, cl)
                    
                    next_state = (next_pos, new_dir)
                    if next_state not in visited or visited[next_state] > next_g:
                        visited[next_state] = next_g
                        came_from[next_state] = state_key
                        next_f = next_g + self._heuristic(next_pos, target)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir))
                        total_pushes += 1
            
            # 2. Check Layer Transitions (vias)
            for dl in [-1, 1]:
                nl = cl + dl
                if nl in active_layers_set:
                    # Check via clearance on both current layer (cl) and target layer (nl)
                    via_valid = True
                    for dx, dy in via_clearance_offsets:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if (cl in obstacle_maps and obstacle_maps[cl][ny, nx] and (nx, ny) not in exempt) or \
                               (nl in obstacle_maps and obstacle_maps[nl][ny, nx] and (nx, ny) not in exempt):
                                via_valid = False
                                break
                        else:
                            via_valid = False
                            break
                    if not via_valid:
                        continue
                        
                    # Via cost: base cost + penalty for low via probability
                    v_prob = via_prob[cy, cx]
                    via_cost = self.base_via_cost + (1.0 - v_prob) * self.base_via_cost
                    
                    new_dir = (0, 0, dl)
                    if last_dir != (0, 0, 0) and last_dir != new_dir:
                        via_cost += self.direction_change_penalty
                        
                    next_g = g + via_cost
                    next_pos = (cx, cy, nl)
                    
                    next_state = (next_pos, new_dir)
                    if next_state not in visited or visited[next_state] > next_g:
                        visited[next_state] = next_g
                        came_from[next_state] = state_key
                        next_f = next_g + self._heuristic(next_pos, target)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir))
                        total_pushes += 1
                        


        return None, float('inf')

    def find_path_multi_target(self, heatmaps, via_prob, source, targets, active_layers, max_iterations=200000):
        """
        Finds the shortest path from source to the nearest of multiple targets
        (Useful for connecting a new pin to an already routed net of pads/traces)
        """
        if not targets:
            return None, float('inf')
            
        targets_set = set(targets)
        N_layers, H, W = heatmaps.shape
        active_layers_set = set(active_layers)
        
        # Custom multi-target heuristic: min heuristic to any target
        def multi_target_heuristic(p):
            return min(self._heuristic(p, t) for t in targets_set)
            
        pq = []
        heapq.heappush(pq, (multi_target_heuristic(source), 0.0, source, (0, 0, 0)))
        
        came_from = {}
        visited = {}
        iterations = 0
        
        while pq and iterations < max_iterations:
            iterations += 1
            f, g, curr, last_dir = heapq.heappop(pq)
            
            if curr in targets_set:
                # Reconstruct path
                path = []
                temp = (curr, last_dir)
                while temp in came_from:
                    path.append(temp[0])
                    temp = came_from[temp]
                path.append(source)
                path.reverse()
                return path, g
                
            state_key = (curr, last_dir)
            if state_key in visited and visited[state_key] < g:
                continue
            visited[state_key] = g
            
            cx, cy, cl = curr
            
            # Spatial Neighbors
            for dx, dy, is_diag in self.moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    h_val = heatmaps[cl, ny, nx]
                    step_len = math.sqrt(2.0) if is_diag else 1.0
                    # FIX: match find_path convention — high h_val = preferred = low cost
                    step_cost = step_len * (1.0 + self.heatmap_weight * (1.0 - h_val))
                    
                    new_dir = (dx, dy, 0)
                    if last_dir != (0, 0, 0) and last_dir != new_dir:
                        step_cost += self.direction_change_penalty
                        
                    next_g = g + step_cost
                    next_pos = (nx, ny, cl)
                    
                    next_state = (next_pos, new_dir)
                    if next_state not in visited or visited[next_state] > next_g:
                        visited[next_state] = next_g
                        came_from[next_state] = state_key
                        next_f = next_g + multi_target_heuristic(next_pos)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir))
                        
            # Layer Transitions
            for dl in [-1, 1]:
                nl = cl + dl
                if nl in active_layers_set:
                    v_prob = via_prob[cy, cx]
                    via_cost = self.base_via_cost + (1.0 - v_prob) * self.base_via_cost
                    
                    new_dir = (0, 0, dl)
                    if last_dir != (0, 0, 0) and last_dir != new_dir:
                        via_cost += self.direction_change_penalty
                        
                    next_g = g + via_cost
                    next_pos = (cx, cy, nl)
                    
                    next_state = (next_pos, new_dir)
                    if next_state not in visited or visited[next_state] > next_g:
                        visited[next_state] = next_g
                        came_from[next_state] = state_key
                        next_f = next_g + multi_target_heuristic(next_pos)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir))
                        
        return None, float('inf')
