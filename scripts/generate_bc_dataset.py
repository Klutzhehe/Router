import os
import pickle
import math
import numpy as np
import torch
from tqdm import tqdm

from pcb_router.data.board_generator import BoardGenerator
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

def cell_delta_to_action(dx, dy, dl):
    if dl == -1:
        return 8
    elif dl == 1:
        return 9
    else:
        moves = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for idx, (mdx, mdy) in enumerate(moves):
            if mdx == dx and mdy == dy:
                return idx
    raise ValueError(f"Invalid move delta: {(dx, dy, dl)}")

def generate_dataset(out_dir="data/bc_dataset"):
    # Setup directories
    os.makedirs(out_dir, exist_ok=True)
    
    curriculum = CurriculumManager("configs/curriculum.yaml")
    # We will generate dataset for stages s00 to s06 (Blocks A and B)
    stages_to_generate = [
        "s00_single_net_empty_board",
        "s01_single_net_sparse_obstacles",
        "s02_single_net_moderate_obstacles",
        "s03_two_nets",
        "s04_three_nets",
        "s05_four_nets",
        "s06_five_nets_congestion"
    ]
    
    episodes_per_stage = 40 # 40 boards per stage
    
    # Map from stage name to config index
    stage_idx_map = {s["name"]: idx for idx, s in enumerate(curriculum.stages)}
    
    for stage_name in stages_to_generate:
        if stage_name not in stage_idx_map:
            print(f"Skipping {stage_name} (not found in curriculum)")
            continue
            
        import glob
        import gzip
        existing_eps = glob.glob(f"{out_dir}/{stage_name}_ep*.pkl.gz") + glob.glob(f"{out_dir}/{stage_name}_ep*.pkl")
        if len(existing_eps) >= episodes_per_stage:
            print(f"Dataset for {stage_name} already generated ({len(existing_eps)} episodes). Skipping...")
            continue
            
        stage_idx = stage_idx_map[stage_name]
        curriculum.current_stage_idx = stage_idx
        
        print(f"\n--- Generating BC Dataset for Curriculum Stage: {stage_name} ---")
        
        # Use PCBRoutingEnv in astar_guided mode to run A* expert
        env = PCBRoutingEnv(
            board_config=curriculum.get_board_config(),
            curriculum_stage=curriculum.current_stage,
            reward_weights=curriculum.get_reward_weights(),
            routing_mode='astar_guided'
        )
        # Instantiate validation environment once per stage
        val_env = PCBRoutingEnv(
            board_config=curriculum.get_board_config(),
            curriculum_stage=curriculum.current_stage,
            reward_weights=curriculum.get_reward_weights(),
            routing_mode='autoregressive'
        )
        
        successful_episodes = len(existing_eps)
        pbar = tqdm(total=episodes_per_stage, initial=successful_episodes, desc=f"Stage {stage_name}")
        
        seed = 42
        while successful_episodes < episodes_per_stage:
            ep_shard_path = f"{out_dir}/{stage_name}_ep{seed}.pkl.gz"
            fallback_path = f"{out_dir}/{stage_name}_ep{seed}.pkl"
            if (os.path.exists(ep_shard_path) and os.path.getsize(ep_shard_path) > 0) or \
               (os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0):
                seed += 1
                continue
                
            print(f"\n  [Seed {seed}] Generating environment...", end="", flush=True)
            obs, info = env.reset(seed=seed)
            print(" generated successfully.", flush=True)
            
            episode_board = env.board
            episode_nets = list(env.board.nets)
            
            episode_transitions = []
            episode_success = True
            
            # Cache active layers
            active_layers = list(range(env.board.num_layers))
            
            # Temporary copy of board state for our cell-by-cell rasterization
            temp_board_state = env.board_state.clone()
            
            for net_idx, net in enumerate(episode_nets):
                net_id = net.id
                temp_board_state.set_current_net(net_id)
                
                # Mock heatmaps/via probability for A*
                layer_heatmaps = np.zeros((env.board.num_layers, env.H, env.W), dtype=np.float32)
                layer_heatmaps.fill(1.0)
                via_prob_map = np.zeros((env.H, env.W), dtype=np.float32)
                via_prob_map.fill(1.0)
                
                # Apply pin exclusions and via boost
                for l_idx in range(env.board.num_layers):
                    pin_mask = temp_board_state.get_pin_exclusion_mask(l_idx)
                    layer_heatmaps[l_idx] *= pin_mask
                
                via_pad_boost = temp_board_state.get_via_in_pad_boost()
                via_prob_map = np.clip(via_prob_map + via_pad_boost, 0.0, 1.0)
                
                # Run A* to find path
                net_pins = [env.board.pins[pid] for pid in net.pin_ids]
                src_pin = net_pins[0]
                source_pos = (src_pin.global_x, src_pin.global_y, src_pin.layer if src_pin.layer != -1 else 0)
                target_positions = [(p.global_x, p.global_y, p.layer if p.layer != -1 else 0) for p in net_pins[1:]]
                
                curr_source = source_pos
                all_routed_path = [curr_source]
                
                print(f"    [Net {net_idx}] Routing A* from {source_pos} to targets...", end="", flush=True)
                astar_success = True
                for target_pos in target_positions:
                    path, cost = env.pathfinder.find_path(
                        layer_heatmaps, via_prob_map,
                        curr_source, target_pos,
                        active_layers,
                        board_state=temp_board_state
                    )
                    if path:
                        all_routed_path.extend(path[1:])
                        curr_source = target_pos
                    else:
                        astar_success = False
                        break
                        
                if not astar_success:
                    print(" failed (no A* path found).", flush=True)
                    episode_success = False
                    break
                else:
                    print(f" success (path length: {len(all_routed_path)} cells).", flush=True)
                    
                # Keep a clean copy of the board state before starting this net
                clean_board_state = temp_board_state.clone()
                
                # Walk consecutive path cells and capture transitions
                net_start_board_state = temp_board_state.clone()
                
                cursor = source_pos
                remaining_targets = list(target_positions)
                curr_target = remaining_targets.pop(0)
                
                # Calculate budget
                total_manhattan = 0.0
                pins_to_route = [net_pins[0]] + [env.board.pins[pid] for pid in net.pin_ids[1:]]
                for i in range(len(pins_to_route) - 1):
                    p1 = pins_to_route[i]
                    p2 = pins_to_route[i+1]
                    total_manhattan += abs(p1.global_x - p2.global_x) + abs(p1.global_y - p2.global_y) + abs(p1.layer - p2.layer)
                max_moves = int(math.ceil(total_manhattan * 4.0))
                max_moves = max(max_moves, 20)
                
                current_net_path_so_far = [cursor]
                
                for step_idx in range(len(all_routed_path) - 1):
                    p_curr = all_routed_path[step_idx]
                    p_next = all_routed_path[step_idx + 1]
                    
                    cx, cy, cl = p_curr
                    nx, ny, nl = p_next
                    
                    # Compute action label
                    dx = nx - cx
                    dy = ny - cy
                    dl = nl - cl
                    action_label = cell_delta_to_action(dx, dy, dl)
                    
                    # Compute valid mask at current cursor
                    exempt = {(cx, cy), (curr_target[0], curr_target[1])}
                    temp_obs = build_obstacle_maps(
                        clean_board_state, active_layers, exempt, shape=(env.H, env.W)
                    )
                    temp_via = build_via_blocked_maps(
                        clean_board_state, temp_obs, active_layers, shape=(env.H, env.W)
                    )
                    
                    valid_mask = np.zeros(10, dtype=bool)
                    for a_idx in range(8):
                        mdx, mdy, _ = env.pathfinder.moves[a_idx]
                        mx, my = cx + mdx, cy + mdy
                        if 0 <= mx < env.W and 0 <= my < env.H:
                            if not temp_obs[cl][my, mx]:
                                valid_mask[a_idx] = True
                    for dl_idx, v_dl in enumerate([-1, 1]):
                        v_nl = cl + v_dl
                        if v_nl in active_layers:
                            if not temp_via[cl][cy, cx] and not temp_via[v_nl][cy, cx]:
                                valid_mask[8 + dl_idx] = True
                                
                    assert valid_mask[action_label], f"Expert action {action_label} is invalid at step {step_idx}!"
                    
                    cursor_norm = np.array([cx / env.W, cy / env.H, cl / env.board.num_layers], dtype=np.float32)
                    target_norm = np.array([curr_target[0] / env.W, curr_target[1] / env.H, curr_target[2] / env.board.num_layers], dtype=np.float32)
                    moves_remaining_frac = np.array([max(0.0, (max_moves - step_idx) / max_moves)], dtype=np.float32)
                    
                    transition = {
                        'raster': net_start_board_state.get_raster().clone().numpy().astype(np.bool_),
                        'graph': env.graph,
                        'layer_mask': net_start_board_state.active_layers_mask.clone().numpy().astype(np.bool_),
                        'cursor_pos': cursor_norm,
                        'target_pos': target_norm,
                        'moves_remaining_frac': moves_remaining_frac,
                        'action': action_label,
                        'valid_mask': valid_mask,
                        'seed': seed
                    }
                    episode_transitions.append(transition)
                    
                    net_start_board_state.rasterize_partial_move(cx, cy, cl, nx, ny, nl, net_class=net.net_class)
                    current_net_path_so_far.append(p_next)
                    
                    if p_next == curr_target and remaining_targets:
                        curr_target = remaining_targets.pop(0)
                
                new_traces, new_vias, p_violations = env.trace_gen.generate_traces(
                    all_routed_path, net_id, env.board.design_rules, net.net_class,
                    temp_board_state.traces, temp_board_state.vias
                )
                if net.target_length > 0.0:
                    tuned_traces, actual_len = env.meander_inserter.insert_meanders(
                        new_traces, net.target_length, net.length_tolerance,
                        temp_board_state.traces, env.board.design_rules.get('default')['clearance']
                    )
                    new_traces = tuned_traces
                temp_board_state.add_routed_trace(new_traces, new_vias)
                
                env.graph = env.graph_builder.update_graph(
                    env.graph, env.board, net_id, new_traces, new_vias
                )
                
            if episode_success:
                print("    Running deterministic reconstruction check...", end="", flush=True)
                val_env.set_board(episode_board)
                
                t_idx = 0
                val_success = True
                
                for net_idx, net in enumerate(episode_nets):
                    val_env.start_routing_net(net_idx)
                    
                    net_done = False
                    while not net_done:
                        action = episode_transitions[t_idx]['action']
                        obs_v, reward_v, term_v, trunc_v, info_v = val_env.step({'action_id': action})
                        
                        net_done = (val_env.current_net_index is None)
                        t_idx += 1
                        
                        if trunc_v:
                            val_success = False
                            break
                            
                    if not val_success:
                        break
                        
                if val_success and len(val_env.routed_nets) == len(episode_nets):
                    print(" PASSED.", flush=True)
                    
                    # Save immediately to disk, freeing memory!
                    with gzip.open(ep_shard_path, "wb") as f:
                        pickle.dump([episode_transitions], f)
                        
                    successful_episodes += 1
                    pbar.update(1)
                else:
                    print(" FAILED (skipping episode).", flush=True)
            seed += 1
                    
        pbar.close()
        print(f"Finished generating stage {stage_name}.\n")
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/bc_dataset', help='Directory to save the generated dataset shards')
    args = parser.parse_args()
    generate_dataset(args.out_dir)
