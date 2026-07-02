import numpy as np
import heapq
import math

class AStarPathfinder:
    def __init__(self, direction_change_penalty: float = 1.0, base_via_cost: float = 15.0, heatmap_weight: float = 10.0):
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
        d3d = d2d + abs(z1 - z2) * self.base_via_cost
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
        if source == target:
            return [source], 0.0

        # Pre-build obstacle maps per layer (cells that are blocked to traversal).
        # Source and target cells are always passable regardless of occupancy.
        exempt = {(source[0], source[1]), (target[0], target[1])}
        obstacle_maps = {}  # layer -> 2D bool array, True = blocked
        if board_state is not None:
            for l in active_layers:
                occ = board_state.get_occupancy(l)  # (H, W) float, 1.0 = occupied
                obstacle_maps[l] = occ >= self.obstacle_threshold

        # Priority Queue: stores tuples of (f_score, g_score, (x, y, layer), last_direction)
        start_node = source
        pq = []
        heapq.heappush(pq, (self._heuristic(start_node, target), 0.0, start_node, (0, 0, 0)))
        
        came_from = {}  # key: (pos, last_dir) -> parent (pos, last_dir)
        visited = {}    # key: (pos, last_dir) -> g_score
        
        iterations = 0
        min_h_seen = self._heuristic(start_node, target)
        total_pushes = 1
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
            
            if curr == target:
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
                        
        # Debug fail information
        print(f"[A* DEBUG] Failed to find path from {source} to {target} after {iterations} iterations.")
        print(f"  Board size: {W}x{H}, Active layers: {active_layers}")
        print(f"  Total pushes: {total_pushes}, Total pops: {total_pops}, PQ size at end: {len(pq)}")
        print(f"  Start heuristic: {self._heuristic(source, target):.2f}, Min heuristic seen: {min_h_seen:.2f}")
        print(f"  First 15 popped states (f, g, pos, dir): {first_pops}")
        print(f"  Last 15 popped states (f, g, pos, dir): {last_pops}")
        if board_state is not None:
            for l in active_layers:
                occ_map = obstacle_maps.get(l)
                if occ_map is not None:
                    src_blocked = occ_map[sy, sx]
                    tgt_blocked = occ_map[ty, tx]
                    print(f"  Layer {l}: source blocked={src_blocked}, target blocked={tgt_blocked}")
                    
                    # Print 5x5 neighborhood occupancy around source
                    y_min, y_max = max(0, sy - 2), min(H - 1, sy + 2)
                    x_min, x_max = max(0, sx - 2), min(W - 1, sx + 2)
                    # Slice slice bounds inclusive
                    neighborhood_src = occ_map[y_min:y_max+1, x_min:x_max+1].astype(int)
                    print(f"  Layer {l} source neighborhood (5x5, center is source):\n{neighborhood_src}")
                    
                    # Print 5x5 neighborhood occupancy around target
                    y_min, y_max = max(0, ty - 2), min(H - 1, ty + 2)
                    x_min, x_max = max(0, tx - 2), min(W - 1, tx + 2)
                    neighborhood_tgt = occ_map[y_min:y_max+1, x_min:x_max+1].astype(int)
                    print(f"  Layer {l} target neighborhood (5x5, center is target):\n{neighborhood_tgt}")
        else:
            print("  board_state is None!")

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
                    step_cost = step_len * (1.0 + self.heatmap_weight * h_val)
                    
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
