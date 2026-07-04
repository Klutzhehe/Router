import numpy as np
import math

def build_obstacle_maps(board_state, active_layers, exempt_cells=None, shape=None, obstacle_threshold=0.5) -> dict[int, np.ndarray]:
    """
    Pre-build obstacle maps per layer (cells that are blocked to traversal).
    Source and target cells are always passable regardless of occupancy.
    """
    if exempt_cells is None:
        exempt_cells = set()
    else:
        exempt_cells = set(exempt_cells)

    temp_obstacle_maps = {}
    if board_state is not None:
        H, W = board_state.height, board_state.width
    else:
        if shape is None:
            raise ValueError("Either board_state or shape must be provided")
        H, W = shape

    for l in active_layers:
        if board_state is not None:
            occ = board_state.get_occupancy(l)  # (H, W) float, 1.0 = occupied
            t_map = (occ >= obstacle_threshold).copy()
            for ex_x, ex_y in exempt_cells:
                if 0 <= ex_x < W and 0 <= ex_y < H:
                    t_map[ex_y, ex_x] = False
            temp_obstacle_maps[l] = t_map
        else:
            temp_obstacle_maps[l] = np.zeros((H, W), dtype=bool)

    return temp_obstacle_maps

def build_via_blocked_maps(board_state, obstacle_maps, active_layers, shape=None) -> dict[int, np.ndarray]:
    """
    Precompute via blockage per layer using NumPy shift-based dilation
    """
    if board_state is not None:
        H, W = board_state.height, board_state.width
    else:
        if shape is None:
            raise ValueError("Either board_state or shape must be provided")
        H, W = shape

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
        obs = obstacle_maps[l]
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

    return via_blocked
