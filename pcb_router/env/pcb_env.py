import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from typing import Dict, Any, Tuple, Optional

from pcb_router.data.board_generator import BoardGenerator, BoardConfig, Board
from pcb_router.data.graph_builder import GraphBuilder
from pcb_router.env.board_state import BoardState
from pcb_router.env.drc_checker import DRCChecker
from pcb_router.routing.pathfinder import AStarPathfinder
from pcb_router.routing.trace_generator import TraceGenerator
from pcb_router.routing.meander import MeanderInserter
from pcb_router.training.rewards import RewardCalculator
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

class PCBRoutingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board_config: Optional[BoardConfig] = None,
        curriculum_stage: Optional[Dict[str, Any]] = None,
        reward_weights: Optional[Dict[str, Any]] = None,
        routing_mode: str = 'astar_guided'
    ):
        super().__init__()
        
        self.board_generator = BoardGenerator()
        self.graph_builder = GraphBuilder()
        self.pathfinder = AStarPathfinder()
        self.trace_gen = TraceGenerator()
        self.meander_inserter = MeanderInserter()
        self.reward_calculator = RewardCalculator(reward_weights)
        
        # Set config/curriculum
        self.board_config = board_config if board_config is not None else BoardConfig()
        self.curriculum_stage = curriculum_stage
        self.routing_mode = routing_mode
        
        # Maximum nets budget for allocation
        self.max_nets = 100
        
        # Board dimensions
        self.H = self.board_config.board_height
        self.W = self.board_config.board_width
        
        # Spaces
        self.observation_space = spaces.Dict({
            'board_raster': spaces.Box(low=0.0, high=1.0, shape=(13, self.H, self.W), dtype=np.float32),
            'layer_mask': spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            'num_unrouted': spaces.Discrete(self.max_nets + 1),
            'cursor_pos': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            'target_pos': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            'moves_remaining_frac': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        self.action_space = spaces.Dict({
            'net_index': spaces.Discrete(self.max_nets),
            'heatmap_latent': spaces.Box(low=-5.0, high=5.0, shape=(256,), dtype=np.float32)
        })
        
        self.board = None
        self.board_state = None
        self.graph = None
        self.step_count = 0
        self.max_steps = 2 * self.board_config.num_nets
        
        self.routed_nets = set()
        self.drc_violations = []

        # Incremental DRC cache: avoids re-scanning already-checked trace pairs
        # on every step (see DRCChecker.check_incremental). Reset whenever
        # board_state.traces is reset/replaced (reset(), set_board()).
        self._drc_cache_trace_count = 0
        self._drc_cache_pairwise = []

        # Step routing state
        self.current_net_index = None
        self.cursor_pos = None
        self.target_pos = None
        self.remaining_targets = []
        self.moves_taken = 0
        self.max_moves_per_net = 0
        self.start_board_state_clone = None
        self.current_net_path = []

    def set_board_config(self, config: BoardConfig):
        self.board_config = config
        self.H = config.board_height
        self.W = config.board_width
        # Re-initialize spaces to match new board size
        self.observation_space = spaces.Dict({
            'board_raster': spaces.Box(low=0.0, high=1.0, shape=(13, self.H, self.W), dtype=np.float32),
            'layer_mask': spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            'num_unrouted': spaces.Discrete(self.max_nets + 1),
            'cursor_pos': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            'target_pos': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            'moves_remaining_frac': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.max_steps = 2 * config.num_nets

    def set_board(self, board: Board):
        """Force the environment to load a specific board (useful for evaluation)"""
        self.board = board
        self.H = board.height
        self.W = board.width
        self.board_state = BoardState(board)
        self.graph = self.graph_builder.build_graph(board)
        self.routed_nets.clear()
        self.drc_violations.clear()
        self._drc_cache_trace_count = 0
        self._drc_cache_pairwise = []
        self.step_count = 0
        self.max_steps = 2 * len(board.nets)

    def start_routing_net(self, net_index: int):
        """Initialize step-by-step routing for a selected net."""
        if net_index >= len(self.board.nets) or self.board.nets[net_index].id in self.routed_nets:
            raise ValueError(f"Invalid net index {net_index} for step-by-step routing.")
            
        selected_net = self.board.nets[net_index]
        net_id = selected_net.id
        
        # Set current net in board state
        self.board_state.set_current_net(net_id)
        
        # Get pins
        net_pins = [self.board.pins[pid] for pid in selected_net.pin_ids]
        src_layer = net_pins[0].layer if net_pins[0].layer != -1 else 0
        source_pos = (net_pins[0].global_x, net_pins[0].global_y, src_layer)
        
        self.cursor_pos = source_pos
        self.remaining_targets = [(p.global_x, p.global_y, p.layer if p.layer != -1 else 0) for p in net_pins[1:]]
        self.target_pos = self.remaining_targets.pop(0)
        
        self.current_net_index = net_index
        self.moves_taken = 0
        self.current_net_path = [source_pos]
        
        # Calculate step budget
        total_manhattan = 0.0
        pins_to_route = [net_pins[0]] + [self.board.pins[pid] for pid in selected_net.pin_ids[1:]]
        for i in range(len(pins_to_route) - 1):
            p1 = pins_to_route[i]
            p2 = pins_to_route[i+1]
            total_manhattan += abs(p1.global_x - p2.global_x) + abs(p1.global_y - p2.global_y) + abs(p1.layer - p2.layer)
            
        budget_multiplier = 4.0
        self.max_moves_per_net = int(math.ceil(total_manhattan * budget_multiplier))
        self.max_moves_per_net = max(self.max_moves_per_net, 20)
        
        # Clone board state for rollback in case of failure
        self.start_board_state_clone = self.board_state.clone()

    def step_move(self, action_id: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the current net's cursor state by one cell or via.
        
        Args:
            action_id: discrete action (0-9)
        """
        if self.current_net_index is None:
            raise RuntimeError("Must call start_routing_net before step_move.")
            
        self.moves_taken += 1
        
        cx, cy, cl = self.cursor_pos
        tx, ty, tl = self.target_pos
        
        active_layers = list(range(self.board.num_layers))
        
        # Rebuild obstacle and via maps based on start board state (to exclude current net's own partial traces)
        exempt = {(cx, cy), (tx, ty)}
        temp_obstacle_maps = build_obstacle_maps(
            self.start_board_state_clone, active_layers, exempt, shape=(self.H, self.W)
        )
        via_blocked = build_via_blocked_maps(
            self.start_board_state_clone, temp_obstacle_maps, active_layers, shape=(self.H, self.W)
        )
        
        # Determine movement
        invalid_move = False
        nx, ny, nl = cx, cy, cl
        is_via = False
        direction_changed = False
        
        if 0 <= action_id <= 7:
            # Grid move
            dx, dy, is_diag = self.pathfinder.moves[action_id]
            nx, ny = cx + dx, cy + dy
            
            # Check boundaries
            if not (0 <= nx < self.W and 0 <= ny < self.H):
                invalid_move = True
            # Check obstacles
            elif temp_obstacle_maps[cl][ny, nx]:
                invalid_move = True
        elif action_id == 8 or action_id == 9:
            # Via transition
            is_via = True
            dl = -1 if action_id == 8 else 1
            nl = cl + dl
            
            # Check valid layer
            if nl not in active_layers:
                invalid_move = True
            # Check via blockage
            elif via_blocked[cl][cy, cx] or via_blocked[nl][cy, cx]:
                invalid_move = True
        else:
            invalid_move = True
            
        # Check direction change penalty
        if len(self.current_net_path) >= 2 and not invalid_move:
            px, py, pl = self.current_net_path[-2]
            prev_dx = cx - px
            prev_dy = cy - py
            prev_dl = cl - pl
            
            curr_dx = nx - cx
            curr_dy = ny - cy
            curr_dl = nl - cl
            
            # Normalize directions
            prev_dir = (np.sign(prev_dx), np.sign(prev_dy), np.sign(prev_dl))
            curr_dir = (np.sign(curr_dx), np.sign(curr_dy), np.sign(curr_dl))
            
            if prev_dir != (0, 0, 0) and prev_dir != curr_dir:
                direction_changed = True

        # Calculate previous distance to target
        dist_prev = abs(cx - tx) + abs(cy - ty) + abs(cl - (0 if tl == -1 else tl))
        
        # Process the move
        if not invalid_move:
            # Apply incremental occupancy update to board raster
            self.board_state.rasterize_partial_move(cx, cy, cl, nx, ny, nl, net_class=self.board.nets[self.current_net_index].net_class)
            self.cursor_pos = (nx, ny, nl)
            self.current_net_path.append(self.cursor_pos)
            cx, cy, cl = nx, ny, nl
        
        # Calculate new distance to target
        dist_curr = abs(cx - tx) + abs(cy - ty) + abs(cl - (0 if tl == -1 else tl))
        dist_delta = dist_prev - dist_curr
        
        # Calculate step reward
        step_info = {
            'dist_delta': float(dist_delta),
            'invalid_move': invalid_move,
            'direction_changed': direction_changed,
            'is_via': is_via
        }
        step_reward = self.reward_calculator.calculate_step(step_info)
        
        # Check if current segment target reached
        target_pin = None
        for p in self.board.nets[self.current_net_index].pin_ids:
            pin = self.board.pins[p]
            if pin.global_x == tx and pin.global_y == ty:
                target_pin = pin
                break
                
        if target_pin:
            w_half = 3
            h_half = 3
            target_reached = (abs(cx - tx) <= w_half and abs(cy - ty) <= h_half)
            if tl != -1 and cl != tl:
                target_reached = False
        else:
            target_reached = (cx == tx and cy == ty) if tl == -1 else (cx == tx and cy == ty and cl == tl)
        
        terminated = False
        truncated = False
        success = False
        
        if target_reached:
            if self.remaining_targets:
                # Move to next target pin of same net
                self.target_pos = self.remaining_targets.pop(0)
            else:
                # All target pins for this net are reached! Success!
                success = True
                net_id = self.board.nets[self.current_net_index].id
                
                # Commit final physical traces & vias using TraceGenerator
                drc_checker = DRCChecker(self.board.design_rules, self.resolution_mm())
                selected_net = self.board.nets[self.current_net_index]
                
                # Generate Trace segments and Vias from cell path
                new_traces, new_vias, p_violations = self.trace_gen.generate_traces(
                    self.current_net_path, net_id, self.board.design_rules, selected_net.net_class,
                    self.start_board_state_clone.traces, self.start_board_state_clone.vias
                )
                
                # Revert temporary rasterizations by restoring the clean start_board_state_clone,
                # then commit the properly generated and optionally tuned traces
                self.board_state = self.start_board_state_clone
                
                # Length tuning post-processing
                if selected_net.target_length > 0.0:
                    tuned_traces, actual_len = self.meander_inserter.insert_meanders(
                        new_traces, selected_net.target_length, selected_net.length_tolerance,
                        self.board_state.traces, self.board.design_rules.get('default')['clearance']
                    )
                    new_traces = tuned_traces
                
                # Render final proper traces/vias
                self.board_state.add_routed_trace(new_traces, new_vias)
                self.routed_nets.add(net_id)
                
                # Calculate terminal bonus/penalties using normal calculate()
                all_violations, self._drc_cache_pairwise = drc_checker.check_incremental(
                    self.board_state, self.board_state.traces, self.board_state.vias, self.board,
                    self._drc_cache_trace_count, self._drc_cache_pairwise
                )
                self._drc_cache_trace_count = len(self.board_state.traces)
                num_new_violations = len(all_violations) - len(self.drc_violations)
                self.drc_violations = all_violations
                
                actual_wirelen = self.meander_inserter.calculate_trace_length(new_traces)
                net_pins = [self.board.pins[pid] for pid in selected_net.pin_ids]
                manhattan_dist = self._calculate_manhattan_distance(net_pins)
                
                length_error = 0.0
                if selected_net.target_length > 0.0:
                    length_error = abs(actual_wirelen - selected_net.target_length) / selected_net.target_length
                    
                all_nets_complete = (len(self.routed_nets) == len(self.board.nets))
                
                reward_dict = {
                    'connected': True,
                    'wirelength': actual_wirelen,
                    'manhattan_distance': manhattan_dist,
                    'drc_violations': num_new_violations,
                    'congestion_increase': 0.05,
                    'length_error': length_error,
                    'all_nets_complete': all_nets_complete,
                    'total_nets': len(self.board.nets),
                    'routed_nets': len(self.routed_nets)
                }
                terminal_reward = self.reward_calculator.calculate(reward_dict)
                step_reward += terminal_reward
                
                # Update GNN Graph
                self.graph = self.graph_builder.update_graph(
                    self.graph, self.board, net_id, new_traces, new_vias
                )
                
                # Clear step state for this net
                self.last_cursor_pos = (cx, cy, cl)
                self.current_net_index = None
                self.cursor_pos = None
                self.target_pos = None
                self.start_board_state_clone = None
                
                terminated = all_nets_complete
        
        # Check budget truncation
        if self.current_net_index is not None and self.moves_taken >= self.max_moves_per_net:
            # Failed to route within budget: revert board_state to clean cloned state!
            self.board_state = self.start_board_state_clone
            
            # Apply failure terminal penalty
            reward_dict = {
                'connected': False,
                'wirelength': 0.0,
                'manhattan_distance': 1.0,
                'drc_violations': 0,
                'congestion_increase': 0.0,
                'length_error': 0.0,
                'all_nets_complete': False,
                'total_nets': len(self.board.nets),
                'routed_nets': len(self.routed_nets)
            }
            terminal_penalty = self.reward_calculator.calculate(reward_dict)
            step_reward += terminal_penalty
            
            # Clear step state
            self.last_cursor_pos = (cx, cy, cl)
            self.current_net_index = None
            self.cursor_pos = None
            self.target_pos = None
            self.start_board_state_clone = None
            
            truncated = True
            
        obs = self._get_obs()
        info = self._get_info()
        info['connected'] = success
        info['path'] = self.current_net_path if success else []
        info['dist_delta'] = float(dist_delta)
        info['invalid_move'] = invalid_move
        info['direction_changed'] = direction_changed
        info['is_via'] = is_via
        
        return obs, step_reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        
        # 1. Generate new board
        if options is not None and 'board_config' in options:
            self.board_config = options['board_config']
        elif self.curriculum_stage is not None:
            self.board_config = self.board_generator.from_curriculum_stage(self.curriculum_stage)
            
        if seed is not None:
            self.board_config.seed = seed
            
        self.set_board_config(self.board_config)
        self.board = self.board_generator.generate(self.board_config)
        
        # 2. Setup board state & GNN Graph
        self.board_state = BoardState(self.board)
        self.graph = self.graph_builder.build_graph(self.board)
        
        self.routed_nets.clear()
        self.drc_violations.clear()
        self._drc_cache_trace_count = 0
        self._drc_cache_pairwise = []
        self.step_count = 0

        # Reset step routing state
        self.current_net_index = None
        self.cursor_pos = None
        self.last_cursor_pos = None
        self.target_pos = None
        self.remaining_targets = []
        self.moves_taken = 0
        self.max_moves_per_net = 0
        self.start_board_state_clone = None
        self.current_net_path = []
        
        # Re-init reward weights if curriculum changed
        if self.curriculum_stage:
            self.reward_calculator.update_weights(self.curriculum_stage.get('reward_weights'))
            
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.routing_mode == 'autoregressive':
            if 'action_id' in action:
                return self.step_move(action['action_id'])
            elif 'net_index' in action:
                net_idx = int(action['net_index'])
                self.start_routing_net(net_idx)
                return self._get_obs(), 0.0, False, False, self._get_info()
            else:
                raise ValueError("In autoregressive routing mode, action must contain 'action_id' or 'net_index'")
        
        # Traditional step handles Gymnasium interface.
        # However, because action includes heatmap_latent which must be decoded
        # by the CNN decoder (which sits inside the model on GPU), we usually
        # invoke `step_with_heatmaps` during the training loop.
        # Here we provide a default mock/fallback for standard gym compatibility
        net_idx = int(action['net_index'])
        # Draw a uniform mock heatmap since we don't have the model inside the env
        mock_heatmaps = np.zeros((self.board.num_layers, self.H, self.W), dtype=np.float32)
        mock_via_prob = np.zeros((self.H, self.W), dtype=np.float32)
        return self.step_with_heatmaps(net_idx, mock_heatmaps, mock_via_prob)

    def step_with_heatmaps(
        self,
        net_index: int,
        layer_heatmaps: np.ndarray, # (num_active_layers, H, W)
        via_prob_map: np.ndarray    # (H, W)
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        import time
        self.step_count += 1
        
        # Validate net_index selection
        valid_nets = [n for n in self.board.nets if n.id not in self.routed_nets]
        
        # If model selects an out-of-bounds index or already routed net, penalize
        if net_index >= len(self.board.nets) or self.board.nets[net_index].id in self.routed_nets:
            reward = -2.0 # Invalid choice penalty
            terminated = False
            truncated = (self.step_count >= self.max_steps)
            info = self._get_info()
            info['connected'] = False
            info['time_astar'] = 0.0
            info['time_post'] = 0.0
            info['time_drc'] = 0.0
            info['time_graph'] = 0.0
            return self._get_obs(), reward, terminated, truncated, info
            
        selected_net = self.board.nets[net_index]
        net_id = selected_net.id
        
        # Set source/target markers in raster
        self.board_state.set_current_net(net_id)
        
        # Apply pin exclusion mask: zero out heatmap near other-net pads
        # so A* never sees those areas as "preferred" routing zones.
        # Also apply via-in-pad boost so A* can place vias at own pads.
        for l_idx in range(layer_heatmaps.shape[0]):
            pin_mask = self.board_state.get_pin_exclusion_mask(l_idx)
            layer_heatmaps[l_idx] *= pin_mask
        
        via_pad_boost = self.board_state.get_via_in_pad_boost()
        via_prob_map = np.clip(via_prob_map + via_pad_boost, 0.0, 1.0)
        
        # 1. Find pins for this net
        net_pins = [self.board.pins[pid] for pid in selected_net.pin_ids]
        src_pin = net_pins[0]
        
        # 2. Route net using A*
        # For multi-pin nets, route to nearest unrouted target sequentially
        active_layers = list(range(self.board.num_layers))
        
        waypoints = []
        path_cost = 0.0
        success = False
        
        # First route source to target 1
        source_pos = (src_pin.global_x, src_pin.global_y, src_pin.layer)
        target_positions = [(p.global_x, p.global_y, p.layer) for p in net_pins[1:]]
        
        # Sequentially route targets
        curr_source = source_pos
        all_routed_path = [curr_source]
        
        t_astar_start = time.perf_counter()
        for target_pos in target_positions:
            path, cost = self.pathfinder.find_path(
                layer_heatmaps, via_prob_map,
                curr_source, target_pos,
                active_layers,
                board_state=self.board_state  # enables obstacle blocking
            )
            if path:
                all_routed_path.extend(path[1:])
                path_cost += cost
                # Next target is routed from the exact point the current path ended
                curr_source = path[-1]
                success = True
            else:
                success = False
                break
        t_astar = time.perf_counter() - t_astar_start
                
        # 3. Generate Trace segments and Vias if path found
        new_traces = []
        new_vias = []
        drc_checker = DRCChecker(self.board.design_rules, self.resolution_mm())
        
        t_post_start = time.perf_counter()
        if success:
            # Build physical trace segments and vias
            new_traces, new_vias, p_violations = self.trace_gen.generate_traces(
                all_routed_path, net_id, self.board.design_rules, selected_net.net_class,
                self.board_state.traces, self.board_state.vias
            )
            
            # 4. Length tuning post-processing
            if selected_net.target_length > 0.0:
                tuned_traces, actual_len = self.meander_inserter.insert_meanders(
                    new_traces, selected_net.target_length, selected_net.length_tolerance,
                    self.board_state.traces, self.board.design_rules.get('default')['clearance']
                )
                new_traces = tuned_traces
                
            # Render trace into board raster layers
            self.board_state.add_routed_trace(new_traces, new_vias)
            self.routed_nets.add(net_id)
        t_post = time.perf_counter() - t_post_start
            
        t_drc_start = time.perf_counter()
        # 5. Check DRC violations (incremental: avoids re-scanning already-checked
        # trace pairs every step, see DRCChecker.check_incremental)
        all_violations, self._drc_cache_pairwise = drc_checker.check_incremental(
            self.board_state, self.board_state.traces, self.board_state.vias, self.board,
            self._drc_cache_trace_count, self._drc_cache_pairwise
        )
        self._drc_cache_trace_count = len(self.board_state.traces)
        # Find violations introduced by this step
        num_new_violations = len(all_violations) - len(self.drc_violations)
        self.drc_violations = all_violations
        t_drc = time.perf_counter() - t_drc_start
        
        # 6. Calculate Reward
        # Prepare dict for reward calculator
        actual_wirelen = self.meander_inserter.calculate_trace_length(new_traces) if success else 0.0
        manhattan_dist = self._calculate_manhattan_distance(net_pins)
        
        congestion_increase = 0.05 # placeholder, can estimate from board_state
        
        length_error = 0.0
        if selected_net.target_length > 0.0 and success:
            length_error = abs(actual_wirelen - selected_net.target_length) / selected_net.target_length
            
        all_nets_complete = (len(self.routed_nets) == len(self.board.nets))
        
        reward_dict = {
            'connected': success,
            'wirelength': actual_wirelen,
            'manhattan_distance': manhattan_dist,
            'drc_violations': num_new_violations,
            'congestion_increase': congestion_increase,
            'length_error': length_error,
            'all_nets_complete': all_nets_complete,
            'total_nets': len(self.board.nets),
            'routed_nets': len(self.routed_nets)
        }
        
        step_reward = self.reward_calculator.calculate(reward_dict)
        
        t_graph_start = time.perf_counter()
        # 7. Update GNN Graph node features with the new routing decision
        self.graph = self.graph_builder.update_graph(
            self.graph, self.board, net_id, new_traces, new_vias
        )
        t_graph = time.perf_counter() - t_graph_start
        
        terminated = all_nets_complete
        truncated = (self.step_count >= self.max_steps)
        
        obs = self._get_obs()
        info = self._get_info()
        info['connected'] = success
        info['time_astar'] = t_astar
        info['time_post'] = t_post
        info['time_drc'] = t_drc
        info['time_graph'] = t_graph
        info['path'] = all_routed_path if success else []
        
        return obs, step_reward, terminated, truncated, info

    def resolution_mm(self) -> float:
        return self.board_state.resolution

    def _get_obs(self) -> Dict[str, Any]:
        if self.cursor_pos is not None:
            cx, cy, cl = self.cursor_pos
            c_pos = np.array([cx / max(1, self.W), cy / max(1, self.H), cl / max(1, self.board.num_layers)], dtype=np.float32)
        else:
            c_pos = np.zeros(3, dtype=np.float32)
            
        if self.target_pos is not None:
            tx, ty, tl = self.target_pos
            t_layer = 0 if tl == -1 else tl
            t_pos = np.array([tx / max(1, self.W), ty / max(1, self.H), t_layer / max(1, self.board.num_layers)], dtype=np.float32)
        else:
            t_pos = np.zeros(3, dtype=np.float32)
            
        if self.max_moves_per_net > 0:
            frac_val = (self.max_moves_per_net - self.moves_taken) / self.max_moves_per_net
            frac = np.array([max(0.0, min(1.0, frac_val))], dtype=np.float32)
        else:
            frac = np.array([0.0], dtype=np.float32)

        return {
            'board_raster': self.board_state.get_raster().clone().numpy(),
            'layer_mask': self.board_state.active_layers_mask.clone().numpy(),
            'num_unrouted': len(self.board.nets) - len(self.routed_nets),
            'cursor_pos': c_pos,
            'target_pos': t_pos,
            'moves_remaining_frac': frac
        }

    def _get_info(self) -> Dict[str, Any]:
        # Completion stats
        comp_rate = len(self.routed_nets) / len(self.board.nets) if self.board.nets else 0.0
        return {
            'routed_nets': list(self.routed_nets),
            'drc_violations': len(self.drc_violations),
            'completion_rate': comp_rate,
            'current_step': self.step_count,
            'graph': self.graph, # Expose PyG HeteroData graph via info dict
            'max_moves_per_net': self.max_moves_per_net
        }

    def _calculate_manhattan_distance(self, pins) -> float:
        total = 0.0
        for i in range(len(pins) - 1):
            p1, p2 = pins[i], pins[i+1]
            total += (abs(p1.global_x - p2.global_x) + abs(p1.global_y - p2.global_y)) * self.resolution_mm()
        return total

    def render(self) -> Optional[np.ndarray]:
        # Return canvas of the current state, rendered using BoardRenderer
        from pcb_router.visualization.renderer import BoardRenderer
        renderer = BoardRenderer()
        fig = renderer.render_board(self.board_state, self.board, show_all_layers=True)
        
        # Convert fig to RGB array
        fig.canvas.draw()
        rgba_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgb_img = rgba_img.reshape((h, w, 3))
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        return rgb_img

    def validate_final_board(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Hard correctness gate to verify the final board routing in production.
        Checks:
        1. All nets must be fully routed/connected.
        2. Zero DRC (clearance/boundary) violations must exist.
        
        Returns:
            Tuple (is_valid, report):
                - is_valid: True if board passes all gates, else False.
                - report: Dictionary detailing unrouted nets and any DRC violations.
        """
        if self.board is None or self.board_state is None:
            return False, {"error": "No board loaded"}

        # 1. Run fresh, full DRC check first to get physical unconnected violations
        drc_checker = DRCChecker(self.board.design_rules, self.resolution_mm())
        all_violations = drc_checker.check_all(
            self.board_state, self.board_state.traces, self.board_state.vias, self.board
        )
        
        # Update cache
        self.drc_violations = all_violations

        # 2. Determine physical connectivity from DRC unconnected violations
        unconnected_nets = {v.net_id_a for v in all_violations if v.type == 'unconnected'}
        unrouted_nets = list(unconnected_nets)
        
        total_nets = len(self.board.nets)
        connected_nets_count = len([net for net in self.board.nets if net.id not in unconnected_nets])
        all_connected = (connected_nets_count == total_nets)
        
        # 3. Determine validity
        is_valid = all_connected and (len(all_violations) == 0)
        
        report = {
            "is_valid": is_valid,
            "total_nets": total_nets,
            "routed_nets_count": connected_nets_count,
            "unrouted_nets": unrouted_nets,
            "drc_violations_count": len(all_violations),
            "drc_violations": [
                {
                    "type": v.type,
                    "severity": v.severity,
                    "position": (v.x, v.y, v.layer),
                    "description": v.description,
                    "net_id_a": v.net_id_a,
                    "net_id_b": v.net_id_b
                } for v in all_violations
            ]
        }
        
        return is_valid, report
