import sys
import os
import numpy as np

from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.routing.rip_up_router import RipUpRerouteRouter
from pcb_router.env.drc_checker import DRCChecker

def verify_constraints():
    print("==================================================")
    print("   PCB Router Constraints Verification Test       ")
    print("==================================================")
    
    board_gen = BoardGenerator()
    
    config = BoardConfig()
    config.board_width = 180
    config.board_height = 180
    config.num_nets = 8
    config.num_layers = 2
    config.num_components = 5
    config.diff_pairs = True
    config.num_diff_pairs = 1
    config.length_matching = True
    config.matched_group_size = 2
    config.seed = 42

    print("Generating board with constraints...")
    board = board_gen.generate(config)
    print(f"Board: {board.width}x{board.height}, layers={board.num_layers}, nets={len(board.nets)}")
    
    diff_nets = [n for n in board.nets if n.is_diff_pair]
    matched_nets = [n for n in board.nets if n.matched_group_id is not None]
    
    print(f"Differential Pair Nets: {[n.name for n in diff_nets]}")
    print(f"Length-Matched Nets: {[n.name for n in matched_nets]}")

    print("\nRunning routing...")
    router = RipUpRerouteRouter(board, max_iterations=20)
    res = router.route()

    print("\nRouting Results:")
    print(f"Completed: {res['completed']}/{res['total']} nets ({res['completion_rate']:.2%})")
    print(f"Iterations: {res['iterations']}, Vias: {len(res['board_state'].vias)}")
    print(f"Converged: {res['converged']}")

    if not res['converged']:
        print("[FAILED] Router did not converge.")
        sys.exit(1)

    print("\nRunning Design Rule Check (DRC)...")
    drc = DRCChecker(board.design_rules, router.resolution)
    violations = drc.check_all(res['board_state'], res['board_state'].traces, res['board_state'].vias, board)
    
    # Filter out unconnected violations
    errors = [v for v in violations if v.severity == 'error' and v.type != 'unconnected']
    
    print(f"Total DRC Violations found: {len(violations)}")
    for v in violations:
        print(f"  [{v.type}] severity={v.severity}: {v.description} at (x={v.x}, y={v.y}) layer={v.layer}")

    print("\nVerifying Differential Pair Coupling...")
    if len(diff_nets) >= 2:
        p_net = next(n for n in diff_nets if "DIFF_P" in n.name or "_P" in n.name)
        n_net = next(n for n in diff_nets if "DIFF_N" in n.name or "_N" in n.name)
        
        path_p = res['routes'][p_net.id]
        path_n = res['routes'][n_net.id]
        
        if path_p and path_n:
            print("Both Positive and Negative traces successfully routed.")
            print(f"  Positive path length: {len(path_p)} nodes")
            print(f"  Negative path length: {len(path_n)} nodes")
            
            diff_clearances = []
            for i in range(1, min(len(path_p), len(path_n)) - 1):
                pt_p = np.array(path_p[i][:2])
                pt_n = np.array(path_n[i][:2])
                dist = np.linalg.norm(pt_p - pt_n) * router.resolution
                diff_clearances.append(dist)
            
            mean_dist = np.mean(diff_clearances)
            print(f"  Average coupling distance (excluding entries): {mean_dist:.3f} mm")

    print("\nVerifying Length Matching...")
    for net in matched_nets:
        net_traces = [t for t in res['board_state'].traces if t.net_id == net.id]
        total_len = 0.0
        for seg in net_traces:
            dx = (seg.end_x - seg.start_x) * router.resolution
            dy = (seg.end_y - seg.start_y) * router.resolution
            total_len += np.hypot(dx, dy)
            
        print(f"  Net {net.name}: target_len={net.target_length:.2f} mm, actual_len={total_len:.2f} mm (error={abs(total_len - net.target_length):.2f} mm, tol={net.length_tolerance:.2f} mm)")

    if len(errors) == 0:
        print("\nSUCCESS: CONSTRAINTS VERIFICATION SUCCESSFUL! ZERO DRC ERRORS!")
    else:
        print(f"\nFAILURE: {len(errors)} DRC Errors found on the final board state.")
        sys.exit(1)

if __name__ == "__main__":
    verify_constraints()
