import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional

from pcb_router.data.board_generator import BoardGenerator, BoardConfig, Board
from pcb_router.data.graph_builder import GraphBuilder
from pcb_router.env.board_state import BoardState
from pcb_router.env.drc_checker import DRCChecker
from pcb_router.routing.pathfinder import AStarPathfinder
from pcb_router.routing.trace_generator import TraceGenerator
from pcb_router.routing.meander import MeanderInserter
from pcb_router.training.rewards import RewardCalculator

class PCBRoutingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board_config: Optional[BoardConfig] = None,
        curriculum_stage: Optional[Dict[str, Any]] = None,
        reward_weights: Optional[Dict[str, Any]] = None
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
        
        # Maximum nets budget for allocation
        self.max_nets = 100
        
        # Board dimensions
        self.H = self.board_config.board_height
        self.W = self.board_config.board_width
        
        # Spaces
        self.observation_space = spaces.Dict({
            'board_raster': spaces.Box(low=0.0, high=1.0, shape=(13, self.H, self.W), dtype=np.float32),
            'layer_mask': spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            'num_unrouted': spaces.Discrete(self.max_nets + 1)
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

    def set_board_config(self, config: BoardConfig):
        self.board_config = config
        self.H = config.board_height
        self.W = config.board_width
        # Re-initialize spaces to match new board size
        self.observation_space = spaces.Dict({
            'board_raster': spaces.Box(low=0.0, high=1.0, shape=(13, self.H, self.W), dtype=np.float32),
            'layer_mask': spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            'num_unrouted': spaces.Discrete(self.max_nets + 1)
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
        self.step_count = 0
        self.max_steps = 2 * len(board.nets)

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
        self.step_count = 0
        
        # Re-init reward weights if curriculum changed
        if self.curriculum_stage:
            self.reward_calculator.update_weights(self.curriculum_stage.get('reward_weights'))
            
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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
                # Next target is routed from the current target or trace points
                curr_source = target_pos
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
        # 5. Check DRC violations
        all_violations = drc_checker.check_all(
            self.board_state, self.board_state.traces, self.board_state.vias, self.board
        )
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
        return {
            'board_raster': self.board_state.get_raster().clone().numpy(),
            'layer_mask': self.board_state.active_layers_mask.clone().numpy(),
            'num_unrouted': len(self.board.nets) - len(self.routed_nets)
        }

    def _get_info(self) -> Dict[str, Any]:
        # Completion stats
        comp_rate = len(self.routed_nets) / len(self.board.nets) if self.board.nets else 0.0
        return {
            'routed_nets': list(self.routed_nets),
            'drc_violations': len(self.drc_violations),
            'completion_rate': comp_rate,
            'current_step': self.step_count,
            'graph': self.graph # Expose PyG HeteroData graph via info dict
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
