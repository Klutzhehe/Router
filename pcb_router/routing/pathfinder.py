import numpy as np
import heapq
import math
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

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
        self.debug = debug

        # Precomputed turn-severity table (8x8, indices 0..7 matching self.moves) used to scale
        # direction_change_penalty by the ACTUAL angle between the previous and next move, instead
        # of a flat "changed direction at all" cost. Previously a gentle 45-degree bend and a full
        # 180-degree U-turn cost the exact same penalty, so once local congestion made a straight
        # push expensive, A* had no extra reason to prefer a small bend over a full reversal -
        # producing paths that bend, overlap themselves, and U-turn back. Severity in [0, 1]:
        # 0.0 = straight (same direction), 0.5 = 90-degree turn, 1.0 = full 180-degree reversal.
        # A 45-degree bend now costs ~0.15x the old flat penalty; a reversal still costs the full
        # penalty, so reversals are now much more expensive RELATIVE to gentle bends than before.
        n = len(self.moves)
        self._turn_severity = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dx1, dy1, _ = self.moves[i]
            len1 = math.hypot(dx1, dy1)
            for j in range(n):
                dx2, dy2, _ = self.moves[j]
                len2 = math.hypot(dx2, dy2)
                cos_theta = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                self._turn_severity[i][j] = (1.0 - cos_theta) / 2.0

    def _turn_penalty(self, last_dir_idx: int, d_idx: int) -> float:
        """direction_change_penalty scaled by turn angle. last_dir_idx/d_idx are 1..8 (index into
        self.moves + 1); 0 means no established direction yet (path start) and 9/10 mean the
        previous move was a via (no in-plane direction to compare against) - both are unpenalized,
        matching prior behavior."""
        if last_dir_idx <= 0 or last_dir_idx > 8:
            return 0.0
        severity = self._turn_severity[last_dir_idx - 1][d_idx - 1]
        return self.direction_change_penalty * severity

    def _heuristic(self, p1, p2):
        """3D heuristic: 2D distance + layer transition cost estimation"""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        d2d = (dx + dy) + self.SQRT2_MINUS_2 * min(dx, dy)
        z_dist = 0 if (z1 == -1 or z2 == -1) else abs(z1 - z2)
        return d2d * (1.0 + self.heatmap_weight) + z_dist * self.base_via_cost

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
        target_pin = None
        if board_state and board_state.board:
            for pin in board_state.board.pins.values():
                if pin.global_x == tx and pin.global_y == ty:
                    target_pin = pin
                    break

        def check_target_reached(x, y, l):
            if target_pin:
                dx = abs(x - tx)
                dy = abs(y - ty)
                w_half = 3
                h_half = 3
                reached = (dx <= w_half and dy <= h_half)
                if tl != -1 and l != tl:
                    reached = False
                return reached
            else:
                return (tl == -1 and x == tx and y == ty) or (x == tx and y == ty and l == tl)

        # Early-exit: already at destination
        if check_target_reached(sx, sy, sl):
            p_sl = 0 if sl == -1 else sl
            return [(sx, sy, p_sl)], 0.0

        exempt = {(sx, sy), (tx, ty)}

        temp_obstacle_maps = build_obstacle_maps(
            board_state, active_layers, exempt, shape=(H, W), obstacle_threshold=self.obstacle_threshold
        )
        via_blocked = build_via_blocked_maps(
            board_state, temp_obstacle_maps, active_layers, shape=(H, W)
        )

        # 1D Indexing Setup
        stride_y = W * 11
        stride_l = H * W * 11
        total_size = N_layers * H * W * 11
        
        if not hasattr(self, '_visited_flat') or self._visited_flat is None or len(self._visited_flat) < total_size:
            self._visited_flat = [float('inf')] * total_size
            self._generation_flat = [0] * total_size
            self._parent_flat = [-1] * total_size
            self._generation_id = 0

        self._generation_id += 1
        gen_id = self._generation_id
        
        visited_flat = self._visited_flat
        generation_flat = self._generation_flat
        parent_flat = self._parent_flat

        pq = []
        if sl == -1:
            for l in active_layers:
                start_idx = l * stride_l + sy * stride_y + sx * 11 + 0
                generation_flat[start_idx] = gen_id
                visited_flat[start_idx] = 0.0
                parent_flat[start_idx] = -1
                heapq.heappush(pq, (self._heuristic((sx, sy, l), target), 0.0, sx, sy, l, 0))
        else:
            start_idx = sl * stride_l + sy * stride_y + sx * 11 + 0
            generation_flat[start_idx] = gen_id
            visited_flat[start_idx] = 0.0
            parent_flat[start_idx] = -1
            heapq.heappush(pq, (self._heuristic(source, target), 0.0, sx, sy, sl, 0))

        iterations = 0
        while pq and iterations < max_iterations:
            iterations += 1
            f, g, cx, cy, cl, last_dir_idx = heapq.heappop(pq)

            if check_target_reached(cx, cy, cl):
                # Reconstruct path
                path = []
                curr_idx = cl * stride_l + cy * stride_y + cx * 11 + last_dir_idx
                while curr_idx != -1:
                    rl = curr_idx // stride_l
                    ry = (curr_idx % stride_l) // (W * 11)
                    rx = (curr_idx % (W * 11)) // 11
                    path.append((rx, ry, rl))
                    curr_idx = parent_flat[curr_idx]
                path.reverse()
                return path, g

            curr_idx = cl * stride_l + cy * stride_y + cx * 11 + last_dir_idx
            if visited_flat[curr_idx] < g:
                continue

            # 1. Spatial moves
            for i, (dx, dy, is_diag) in enumerate(self.moves):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if temp_obstacle_maps[cl][ny, nx]:
                        continue

                    move_cost = 1.4142135623730951 if is_diag else 1.0
                    d_idx = i + 1
                    penalty = self._turn_penalty(last_dir_idx, d_idx)

                    h_val = heatmaps[cl, ny, nx]
                    edge_cost = move_cost * (1.0 + self.heatmap_weight * (1.0 - h_val)) + penalty
                    next_g = g + edge_cost

                    next_idx = cl * stride_l + ny * stride_y + nx * 11 + d_idx
                    if generation_flat[next_idx] != gen_id:
                        visited_flat[next_idx] = float('inf')
                        generation_flat[next_idx] = gen_id

                    if next_g < visited_flat[next_idx]:
                        visited_flat[next_idx] = next_g
                        parent_flat[next_idx] = curr_idx
                        next_f = next_g + self._heuristic((nx, ny, cl), target)
                        heapq.heappush(pq, (next_f, next_g, nx, ny, cl, d_idx))

            # 2. Layer changes (Vias)
            if not via_blocked[cl][cy, cx]:
                for layer_idx in active_layers:
                    if layer_idx != cl:
                        via_cost = self.base_via_cost * (1.0 + self.heatmap_weight * (1.0 - via_prob[cy, cx]))
                        next_g = g + via_cost
                        d_idx = 9 if layer_idx < cl else 10

                        next_idx = layer_idx * stride_l + cy * stride_y + cx * 11 + d_idx
                        if generation_flat[next_idx] != gen_id:
                            visited_flat[next_idx] = float('inf')
                            generation_flat[next_idx] = gen_id

                        if next_g < visited_flat[next_idx]:
                            visited_flat[next_idx] = next_g
                            parent_flat[next_idx] = curr_idx
                            next_f = next_g + self._heuristic((cx, cy, layer_idx), target)
                            heapq.heappush(pq, (next_f, next_g, cx, cy, layer_idx, d_idx))

        return None, float('inf')

    def find_path_coupled(self, heatmaps, via_prob, source_p, target_p, source_n, target_n, active_layers, board_state, gap_cells=2):
        N_layers, H, W = heatmaps.shape
        active_layers_set = set(active_layers)

        sx_p, sy_p, sl_p = source_p
        sx_n, sy_n, sl_n = source_n
        tx_p, ty_p, tl_p = target_p
        tx_n, ty_n, tl_n = target_n

        sx_c = int(round((sx_p + sx_n) / 2.0))
        sy_c = int(round((sy_p + sy_n) / 2.0))
        sl_c = sl_p if sl_p != -1 else 0

        tx_c = int(round((tx_p + tx_n) / 2.0))
        ty_c = int(round((ty_p + ty_n) / 2.0))
        tl_c = tl_p if tl_p != -1 else 0

        if not (0 <= sx_c < W and 0 <= sy_c < H) or not (0 <= tx_c < W and 0 <= ty_c < H):
            return None, None, float('inf')

        exempt = {
            (source_p[0], source_p[1]), (target_p[0], target_p[1]),
            (source_n[0], source_n[1]), (target_n[0], target_n[1]),
            (sx_c, sy_c), (tx_c, ty_c)
        }

        temp_obstacle_maps = build_obstacle_maps(
            board_state, active_layers, exempt, shape=(H, W), obstacle_threshold=self.obstacle_threshold
        )
        via_blocked = build_via_blocked_maps(
            board_state, temp_obstacle_maps, active_layers, shape=(H, W)
        )

        ref_dx = target_p[0] - source_p[0]
        ref_dy = target_p[1] - source_p[1]
        if ref_dx == 0 and ref_dy == 0:
            ref_dx, ref_dy = 1, 0
            
        length_ref = math.hypot(ref_dx, ref_dy)
        perp_x = -ref_dy / length_ref
        perp_y = ref_dx / length_ref
        
        pin_dx = source_p[0] - sx_c
        pin_dy = source_p[1] - sy_c
        
        dot = pin_dx * perp_x + pin_dy * perp_y
        sign = 1 if dot >= 0 else -1

        def get_offsets(dx, dy, gap):
            if dx == 0 and dy == 0:
                return 0, 0
            length = math.hypot(dx, dy)
            ox = int(round(-dy / length * gap / 2.0))
            oy = int(round(dx / length * gap / 2.0))
            return sign * ox, sign * oy

        # Precompute spatial offsets
        move_offsets = []
        for dx, dy, is_diag in self.moves:
            ox, oy = get_offsets(dx, dy, gap_cells)
            move_offsets.append((ox, oy))

        def is_segment_free(x1, y1, x2, y2, obs_map):
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if not (0 <= x_min < W and 0 <= x_max < W and 0 <= y_min < H and 0 <= y_max < H):
                return False
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if obs_map[y, x]:
                        return False
            return True

        # Pre-allocate flat arrays
        stride_y = W * 11
        stride_l = H * W * 11
        total_size = N_layers * H * W * 11
        
        if not hasattr(self, '_visited_flat') or self._visited_flat is None or len(self._visited_flat) < total_size:
            self._visited_flat = [float('inf')] * total_size
            self._generation_flat = [0] * total_size
            self._parent_flat = [-1] * total_size
            self._generation_id = 0

        self._generation_id += 1
        gen_id = self._generation_id
        
        visited_flat = self._visited_flat
        generation_flat = self._generation_flat
        parent_flat = self._parent_flat

        init_ox = source_p[0] - sx_c
        init_oy = source_p[1] - sy_c

        if temp_obstacle_maps[sl_c][source_p[1], source_p[0]] or temp_obstacle_maps[sl_c][source_n[1], source_n[0]]:
            return None, None, float('inf')

        pq = []
        start_idx = sl_c * stride_l + sy_c * stride_y + sx_c * 11 + 0
        generation_flat[start_idx] = gen_id
        visited_flat[start_idx] = 0.0
        parent_flat[start_idx] = -1

        heapq.heappush(pq, (self._heuristic((sx_c, sy_c, sl_c), (tx_c, ty_c, tl_c)), 0.0, sx_c, sy_c, sl_c, 0))

        iterations = 0
        max_iterations = 200000
        while pq and iterations < max_iterations:
            iterations += 1
            f, g, cx, cy, cl, last_dir_idx = heapq.heappop(pq)

            if cx == tx_c and cy == ty_c and cl == tl_c:
                # Reconstruct full path candidate first
                center_path = []
                curr_idx = cl * stride_l + cy * stride_y + cx * 11 + last_dir_idx
                while curr_idx != -1:
                    rl = curr_idx // stride_l
                    ry = (curr_idx % stride_l) // (W * 11)
                    rx = (curr_idx % (W * 11)) // 11
                    rdir = curr_idx % 11
                    center_path.append((rx, ry, rl, rdir))
                    curr_idx = parent_flat[curr_idx]
                center_path.reverse()

                path_p = [source_p]
                path_n = [source_n]
                curr_ox, curr_oy = init_ox, init_oy

                for idx, (ccx, ccy, ccl, rdir) in enumerate(center_path):
                    if idx == 0:
                        continue
                    prev_cx, prev_cy, prev_cl, _ = center_path[idx-1]
                    dx = ccx - prev_cx
                    dy = ccy - prev_cy
                    if dx != 0 or dy != 0:
                        curr_ox, curr_oy = get_offsets(dx, dy, gap_cells)

                    px_cell = max(0, min(W - 1, ccx + curr_ox))
                    py_cell = max(0, min(H - 1, ccy + curr_oy))
                    nx_cell = max(0, min(W - 1, ccx - curr_ox))
                    ny_cell = max(0, min(H - 1, ccy - curr_oy))

                    path_p.append((px_cell, py_cell, ccl))
                    path_n.append((nx_cell, ny_cell, ccl))

                if path_p[-1] != target_p:
                    path_p.append(target_p)
                if path_n[-1] != target_n:
                    path_n.append(target_n)

                dedup_p = [path_p[0]]
                for p in path_p[1:]:
                    if p != dedup_p[-1]:
                        dedup_p.append(p)
                dedup_n = [path_n[0]]
                for n in path_n[1:]:
                    if n != dedup_n[-1]:
                        dedup_n.append(n)

                # Calculate final endpoint positions
                dx_last, dy_last = 0, 0
                if last_dir_idx > 0 and last_dir_idx <= 8:
                    dx_last, dy_last, _ = self.moves[last_dir_idx - 1]
                elif last_dir_idx > 8:
                    curr_idx = cl * stride_l + cy * stride_y + cx * 11 + last_dir_idx
                    parent_idx = parent_flat[curr_idx]
                    if parent_idx != -1:
                        pdir = parent_idx % 11
                        if pdir > 0 and pdir <= 8:
                            dx_last, dy_last, _ = self.moves[pdir - 1]
                ox_last, oy_last = get_offsets(dx_last, dy_last, gap_cells)
                p_end = (cx + ox_last, cy + oy_last)
                n_end = (cx - ox_last, cy - oy_last)

                # Check if final connection segments intersect/cross or overlap
                def segments_overlap_or_intersect(p1, p2, q1, q2):
                    def ccw(A, B, C):
                        val = (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
                        if abs(val) < 1e-9:
                            return 0
                        return 1 if val > 0 else -1
                    A, B, C, D = p1, p2, q1, q2
                    ccw_acd = ccw(A, C, D)
                    ccw_bcd = ccw(B, C, D)
                    ccw_abc = ccw(A, B, C)
                    ccw_abd = ccw(A, B, D)
                    if ccw_acd != ccw_bcd and ccw_abc != ccw_abd:
                        return True
                    if ccw_acd == 0 and ccw_bcd == 0 and ccw_abc == 0 and ccw_abd == 0:
                        dx = B[0] - A[0]
                        dy = B[1] - A[1]
                        if abs(dx) > abs(dy):
                            min_ab, max_ab = min(A[0], B[0]), max(A[0], B[0])
                            min_cd, max_cd = min(C[0], D[0]), max(C[0], D[0])
                        else:
                            min_ab, max_ab = min(A[1], B[1]), max(A[1], B[1])
                            min_cd, max_cd = min(C[1], D[1]), max(C[1], D[1])
                        if max_ab > min_cd + 0.001 and max_cd > min_ab + 0.001:
                            return True
                    return False

                # Check if final segments intersect/cross or overlap with the OTHER path
                def segment_intersects_path(s1, s2, path):
                    for idx in range(len(path) - 1):
                        q1, q2 = path[idx][:2], path[idx+1][:2]
                        if segments_overlap_or_intersect(s1, s2, q1, q2):
                            return True
                    return False

                if segments_overlap_or_intersect(p_end, target_p[:2], n_end, target_n[:2]):
                    continue
                if segment_intersects_path(p_end, target_p[:2], path_n):
                    continue
                if segment_intersects_path(n_end, target_n[:2], path_p):
                    continue

                if not is_segment_free(p_end[0], p_end[1], target_p[0], target_p[1], temp_obstacle_maps[cl]):
                    continue
                if not is_segment_free(n_end[0], n_end[1], target_n[0], target_n[1], temp_obstacle_maps[cl]):
                    continue

                return dedup_p, dedup_n, g

            curr_idx = cl * stride_l + cy * stride_y + cx * 11 + last_dir_idx
            if visited_flat[curr_idx] < g:
                continue

            if last_dir_idx == 0:
                ox_curr, oy_curr = init_ox, init_oy
            else:
                if last_dir_idx <= 8:
                    ox_curr, oy_curr = move_offsets[last_dir_idx - 1]
                else:
                    ox_curr, oy_curr = init_ox, init_oy

            # 1. Spatial moves
            for i, (dx, dy, is_diag) in enumerate(self.moves):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    ox_next, oy_next = move_offsets[i]
                    P_curr_x, P_curr_y = cx + ox_curr, cy + oy_curr
                    P_next_x, P_next_y = nx + ox_next, ny + oy_next
                    N_curr_x, N_curr_y = cx - ox_curr, cy - oy_curr
                    N_next_x, N_next_y = nx - ox_next, ny - oy_next

                    if not is_segment_free(P_curr_x, P_curr_y, P_next_x, P_next_y, temp_obstacle_maps[cl]):
                        continue
                    if not is_segment_free(N_curr_x, N_curr_y, N_next_x, N_next_y, temp_obstacle_maps[cl]):
                        continue

                    move_cost = 1.4142135623730951 if is_diag else 1.0
                    d_idx = i + 1
                    penalty = self._turn_penalty(last_dir_idx, d_idx)

                    h_val_p = heatmaps[cl, P_next_y, P_next_x]
                    h_val_n = heatmaps[cl, N_next_y, N_next_x]
                    edge_cost = move_cost * (1.0 + self.heatmap_weight * (2.0 - h_val_p - h_val_n) / 2.0) + penalty
                    next_g = g + edge_cost

                    next_idx = cl * stride_l + ny * stride_y + nx * 11 + d_idx
                    if generation_flat[next_idx] != gen_id:
                        visited_flat[next_idx] = float('inf')
                        generation_flat[next_idx] = gen_id

                    if next_g < visited_flat[next_idx]:
                        visited_flat[next_idx] = next_g
                        parent_flat[next_idx] = curr_idx
                        next_f = next_g + self._heuristic((nx, ny, cl), (tx_c, ty_c, tl_c))
                        heapq.heappush(pq, (next_f, next_g, nx, ny, cl, d_idx))

            # 2. Layer changes (Vias)
            if not via_blocked[cl][cy, cx]:
                for layer_idx in active_layers:
                    if layer_idx != cl:
                        P_x, P_y = cx + ox_curr, cy + oy_curr
                        N_x, N_y = cx - ox_curr, cy - oy_curr

                        if not (0 <= P_x < W and 0 <= P_y < H and 0 <= N_x < W and 0 <= N_y < H):
                            continue

                        if via_blocked[cl][P_y, P_x] or via_blocked[layer_idx][P_y, P_x]:
                            continue
                        if via_blocked[cl][N_y, N_x] or via_blocked[layer_idx][N_y, N_x]:
                            continue

                        via_cost = self.base_via_cost * (1.0 + self.heatmap_weight * (1.0 - via_prob[cy, cx]))
                        next_g = g + via_cost
                        d_idx = 9 if layer_idx < cl else 10

                        next_idx = layer_idx * stride_l + cy * stride_y + cx * 11 + d_idx
                        if generation_flat[next_idx] != gen_id:
                            visited_flat[next_idx] = float('inf')
                            generation_flat[next_idx] = gen_id

                        if next_g < visited_flat[next_idx]:
                            visited_flat[next_idx] = next_g
                            parent_flat[next_idx] = curr_idx
                            next_f = next_g + self._heuristic((cx, cy, layer_idx), (tx_c, ty_c, tl_c))
                            heapq.heappush(pq, (next_f, next_g, cx, cy, layer_idx, d_idx))

        return None, None, float('inf')
