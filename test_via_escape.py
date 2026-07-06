import sys
import os
import numpy as np

from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.routing.rip_up_router import RipUpRerouteRouter

def run_verification():
    print("==================================================")
    print("   PCB Router Via Escape Verification Test        ")
    print("==================================================")
    
    board_gen = BoardGenerator()
    seeds = [40, 42, 44, 45]
    all_success = True
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        config = BoardConfig()
        config.board_width = 150
        config.board_height = 150
        config.num_nets = 10
        config.num_layers = 2
        config.num_components = 5
        config.seed = seed

        board = board_gen.generate(config)
        print(f"Board: {board.width}x{board.height}, layers={board.num_layers}, nets={len(board.nets)}")

        router = RipUpRerouteRouter(board, max_iterations=20)
        res = router.route()

        vias_count = len(res['board_state'].vias)
        print(f"Result: Completed={res['completed']}/{res['total']} nets ({res['completion_rate']:.2%})")
        print(f"        Iterations={res['iterations']}, Vias={vias_count}, Shared cells={res['shared_cells']}")
        print(f"        Converged={res['converged']}")
        
        if not res['converged'] or res['completion_rate'] < 1.0:
            print("[FAILED] Verification FAILED for this seed.")
            all_success = False
        else:
            print("[PASSED] Verification PASSED.")
            
    print("\n==================================================")
    if all_success:
        print("SUCCESS: ALL VERIFICATION SEEDS PASSED SUCCESSFULLY!")
    else:
        print("FAILURE: SOME SEEDS FAILED. Please check logs.")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
