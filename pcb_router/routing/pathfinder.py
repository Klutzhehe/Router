import numpy as np
import heapq
import math

class AStarPathfinder:
    def __init__(self, direction_change_penalty: float = 15.0, base_via_cost: float = 15.0, heatmap_weight: float = 10.0, debug: bool = False):
        self.direction_change_penalty = direction_change_penalty
        self.base_via_cost = base_via_cost
        self.heatmap_weight = heatmap_weight
        self.obstacle_threshold = 0.5
        self.moves = [
            (0, 1, False), (0, -1, False), (1, 0, False), (-1, 0, False),
            (1, 1, True), (1, -1, True), (-1, 1, True), (-1, -1, True)
        ]
        self.SQRT2_MINUS_2 = math.sqrt(2.0) - 2.0
        self._visited_buf = None
        self._parent_x_buf = None
        self._parent_y_buf = None
        self._parent_l_buf = None
        self._parent_dir_buf = None
        self.debug = debug

    def _heuristic(self, p1, p2):
        """3D heuristic: 2D distance + layer transition cost estimation"""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        d2d = (dx + dy) + self.SQRT2_MINUS_2 * min(dx, dy)
        z_dist = 0 if (z1 == -1 or z2 == -1) else abs(z1 - z2)
        d3d = d2d + z_dist * self.base_via_cost
        return d3d * (1.0 + self.heatmap_weight)

    def find_path(self, heatmaps, via_prob, source, target, active_layers, max_iterations=200000, board_state=None):
        """
        Find path from source (x, y, layer) to target (x, y, layer)
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

        exempt = {(sx, sy), (tx, ty)}

        # Pre-build obstacle maps per layer (cells that are blocked to traversal).
        # Source and target cells are always passable regardless of occupancy.
        temp_obstacle_maps = {}
        for l in active_layers:
            if board_state is not None:
                occ = board_state.get_occupancy(l)  # (H, W) float, 1.0 = occupied
                t_map = (occ >= self.obstacle_threshold).copy()
                for ex_x, ex_y in exempt:
                    t_map[ex_y, ex_x] = False
                temp_obstacle_maps[l] = t_map
            else:
                temp_obstacle_maps[l] = np.zeros((H, W), dtype=bool)

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

        # Precompute via blockage per layer using NumPy shift-based dilation
        via_blocked = {}
        for l in active_layers:
            obs = temp_obstacle_maps[l]
            blocked = np.zeros_like(obs)
            for dx, dy in via_clearance_offsets:
                shifted = np.ones_like(obs)
                y_start_dst = max(0, -dy)
                y_end_dst = min(H, H - dy)
                y_start_src = max(0, dy)
                y_end_src = min(H, H + dy)
                x_start_dst = max(0, -dx)
                x_end_dst = min(W, W - dx)
                x_start_src = max(0, dx)
                x_end_src = min(W, W + dx)
                if y_start_dst < y_end_dst and x_start_dst < x_end_dst:
                    shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = obs[y_start_src:y_end_src, x_start_src:x_end_src]
                blocked |= shifted
            via_blocked[l] = blocked

        # Allocate or reuse 4D visited and parent arrays
        if (self._visited_buf is None or 
            self._visited_buf.shape[0] < N_layers or 
            self._visited_buf.shape[1] < H or 
            self._visited_buf.shape[2] < W):
            pad_layers = max(N_layers, 8)
            pad_H = max(H, 512)
            pad_W = max(W, 512)
            self._visited_buf = np.full((pad_layers, pad_H, pad_W, 11), np.inf, dtype=np.float64)
            self._parent_x_buf = np.full((pad_layers, pad_H, pad_W, 11), -1, dtype=np.int16)
            self._parent_y_buf = np.full((pad_layers, pad_H, pad_W, 11), -1, dtype=np.int16)
            self._parent_l_buf = np.full((pad_layers, pad_H, pad_W, 11), -1, dtype=np.int8)
            self._parent_dir_buf = np.full((pad_layers, pad_H, pad_W, 11), -1, dtype=np.int8)

        visited = self._visited_buf[:N_layers, :H, :W, :]
        visited.fill(np.inf)
        
        parent_x = self._parent_x_buf[:N_layers, :H, :W, :]
        parent_y = self._parent_y_buf[:N_layers, :H, :W, :]
        parent_l = self._parent_l_buf[:N_layers, :H, :W, :]
        parent_dir = self._parent_dir_buf[:N_layers, :H, :W, :]

        # We only need to reset start node parents to -1, which they default to.
        if sl == -1:
            for l in active_layers:
                parent_x[l, sy, sx, 0] = -1
        else:
            parent_x[sl, sy, sx, 0] = -1

        # Priority Queue: stores tuples of (f_score, g_score, (x, y, layer), last_direction_idx)
        pq = []
        if sl == -1:
            for l in active_layers:
                start_node = (sx, sy, l)
                heapq.heappush(pq, (self._heuristic(start_node, target), 0.0, start_node, 0))
        else:
            start_node = source
            heapq.heappush(pq, (self._heuristic(start_node, target), 0.0, start_node, 0))
        
        iterations = 0
        
        while pq and iterations < max_iterations:
            iterations += 1
            f, g, curr, last_dir_idx = heapq.heappop(pq)
            cx, cy, cl = curr

            # Target reached check
            is_reached = (cx == tx and cy == ty) if tl == -1 else (curr == target)
            if is_reached:
                # Reconstruct path
                path = []
                rx, ry, rl, rdir = cx, cy, cl, last_dir_idx
                while rx != -1:
                    path.append((int(rx), int(ry), int(rl)))
                    px = parent_x[rl, ry, rx, rdir]
                    py = parent_y[rl, ry, rx, rdir]
                    pl = parent_l[rl, ry, rx, rdir]
                    pdir = parent_dir[rl, ry, rx, rdir]
                    rx, ry, rl, rdir = px, py, pl, pdir
                path.reverse()
                return path, g
                
            if visited[cl, cy, cx, last_dir_idx] < g:
                continue
            visited[cl, cy, cx, last_dir_idx] = g
            
            # 1. Check Spatial Neighbors (same layer)
            for i, (dx, dy, is_diag) in enumerate(self.moves):
                nx, ny = cx + dx, cy + dy
                
                # Check boundaries
                if 0 <= nx < W and 0 <= ny < H:
                    # Skip obstacle cells
                    if temp_obstacle_maps[cl][ny, nx]:
                        continue
                    
                    h_val = heatmaps[cl, ny, nx]
                    step_len = 1.4142135623730951 if is_diag else 1.0
                    step_cost = step_len * (1.0 + self.heatmap_weight * (1.0 - h_val))
                    
                    new_dir_idx = 1 + i
                    if last_dir_idx != 0 and last_dir_idx != new_dir_idx:
                        step_cost += self.direction_change_penalty
                        
                    next_g = g + step_cost
                    next_pos = (nx, ny, cl)
                    
                    if visited[cl, ny, nx, new_dir_idx] > next_g:
                        visited[cl, ny, nx, new_dir_idx] = next_g
                        parent_x[cl, ny, nx, new_dir_idx] = cx
                        parent_y[cl, ny, nx, new_dir_idx] = cy
                        parent_l[cl, ny, nx, new_dir_idx] = cl
                        parent_dir[cl, ny, nx, new_dir_idx] = last_dir_idx
                        next_f = next_g + self._heuristic(next_pos, target)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir_idx))
            
            # 2. Check Layer Transitions (vias)
            for dl_idx, dl in enumerate([-1, 1]):
                nl = cl + dl
                if nl in active_layers_set:
                    # Check precomputed via clearance
                    if via_blocked[cl][cy, cx] or via_blocked[nl][cy, cx]:
                        continue
                        
                    # Via cost: base cost + penalty for low via probability
                    v_prob = via_prob[cy, cx]
                    via_cost = self.base_via_cost + (1.0 - v_prob) * self.base_via_cost
                    
                    new_dir_idx = 9 + dl_idx
                    if last_dir_idx != 0 and last_dir_idx != new_dir_idx:
                        via_cost += self.direction_change_penalty
                        
                    next_g = g + via_cost
                    next_pos = (cx, cy, nl)
                    
                    if visited[nl, cy, cx, new_dir_idx] > next_g:
                        visited[nl, cy, cx, new_dir_idx] = next_g
                        parent_x[nl, cy, cx, new_dir_idx] = cx
                        parent_y[nl, cy, cx, new_dir_idx] = cy
                        parent_l[nl, cy, cx, new_dir_idx] = cl
                        parent_dir[nl, cy, cx, new_dir_idx] = last_dir_idx
                        next_f = next_g + self._heuristic(next_pos, target)
                        heapq.heappush(pq, (next_f, next_g, next_pos, new_dir_idx))

        if self.debug:
            print(f"[A* DEBUG] Failed to find path from {source} to {target} after {iterations} iterations.")

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
