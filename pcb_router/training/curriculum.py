import yaml
import os
import numpy as np
from typing import Dict, Any, Optional, List
from pcb_router.data.board_generator import BoardConfig

class CurriculumManager:
    def __init__(self, config_path: str = 'configs/curriculum.yaml'):
        self.config_path = config_path
        self.stages = []
        self.progression_cfg = {}
        
        # Load curriculum config
        self._load_config()
        
        self.current_stage_idx = 0
        
        # Rolling windows for episode stats
        self.eval_window = self.progression_cfg.get('eval_window', 100)
        self.completion_history = []
        self.violation_history = []
        self.episodes_in_stage = 0

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.stages = cfg.get('stages', [])
                self.progression_cfg = cfg.get('progression', {})
        else:
            # Hardcoded fallback config matching curriculum.yaml specs
            self.stages = [
                {
                    'name': 'single_net_single_layer',
                    'board_generator': {'num_nets': 1, 'num_layers': 2, 'board_size_range': [200, 400], 'obstacle_density': 0.1},
                    'reward_weights': {'completion': 1.5, 'wirelength': 0.05, 'drc_violations': 0.2, 'congestion': 0.0, 'length_error': 0.0, 'all_complete_bonus': 0.3}
                }
            ]
            self.progression_cfg = {
                'completion_threshold': 0.95,
                'drc_violation_threshold': 0.02,
                'min_episodes': 500,
                'eval_window': 100
            }

    @property
    def current_stage(self) -> Dict[str, Any]:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]

    @property
    def current_stage_name(self) -> str:
        return self.current_stage.get('name', 'unknown')

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def get_board_config(self) -> BoardConfig:
        """Sample and create BoardConfig for current stage parameters"""
        stage_cfg = self.current_stage
        board_gen_cfg = stage_cfg.get('board_generator', {})
        
        def resolve_val(val, default):
            if isinstance(val, list):
                return np.random.randint(val[0], val[1] + 1)
            return val if val is not None else default
            
        width = resolve_val(board_gen_cfg.get('board_size_range'), 400)
        height = width
        
        num_nets = resolve_val(board_gen_cfg.get('num_nets_range'), resolve_val(board_gen_cfg.get('num_nets'), 5))
        num_layers = resolve_val(board_gen_cfg.get('num_layers_range'), board_gen_cfg.get('num_layers', 2))
        num_components = resolve_val(board_gen_cfg.get('num_components_range'), board_gen_cfg.get('num_components', 4))
        
        # Keepout zones
        num_keep_out_zones = 0
        if board_gen_cfg.get('keep_out_zones', False):
            num_keep_out_zones = resolve_val(board_gen_cfg.get('num_keep_out_zones_range'), 2)
            
        # Diff pairs
        diff_pairs = board_gen_cfg.get('diff_pairs', False)
        num_diff_pairs = 0
        if diff_pairs:
            num_diff_pairs = resolve_val(board_gen_cfg.get('num_diff_pairs_range'), 1)
            
        return BoardConfig(
            board_width=width,
            board_height=height,
            num_nets=num_nets,
            num_layers=num_layers,
            num_components=num_components,
            obstacle_density=board_gen_cfg.get('obstacle_density', 0.1),
            num_keep_out_zones=num_keep_out_zones,
            diff_pairs=diff_pairs,
            num_diff_pairs=num_diff_pairs,
            length_matching=board_gen_cfg.get('length_matching', False),
            length_tolerance_mm=board_gen_cfg.get('length_tolerance_mm', 1.0),
            net_classes=board_gen_cfg.get('net_classes', ['signal']),
            design_rules=board_gen_cfg.get('design_rules', stage_cfg.get('design_rules'))
        )

    def get_reward_weights(self) -> Dict[str, float]:
        return self.current_stage.get('reward_weights', {})

    def record_episode(self, completion_rate: float, drc_violation_rate: float):
        self.episodes_in_stage += 1
        self.completion_history.append(completion_rate)
        self.violation_history.append(drc_violation_rate)
        
        # Keep history within rolling window length
        if len(self.completion_history) > self.eval_window:
            self.completion_history.pop(0)
            self.violation_history.pop(0)

    def should_advance(self) -> bool:
        min_episodes = self.progression_cfg.get('min_episodes', 500)
        if self.episodes_in_stage < min_episodes:
            return False
            
        if len(self.completion_history) < self.eval_window:
            return False
            
        mean_comp = np.mean(self.completion_history)
        mean_viol = np.mean(self.violation_history)
        
        comp_thresh = self.progression_cfg.get('completion_threshold', 0.95)
        viol_thresh = self.progression_cfg.get('drc_violation_threshold', 0.02)
        
        return mean_comp >= comp_thresh and mean_viol <= viol_thresh

    def advance(self) -> bool:
        """Advance index and clear progress stats"""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.completion_history.clear()
            self.violation_history.clear()
            self.episodes_in_stage = 0
            print(f"--- CURRICULUM ADVANCED TO STAGE {self.current_stage_idx}: {self.current_stage_name} ---")
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            'current_stage_idx': self.current_stage_idx,
            'episodes_in_stage': self.episodes_in_stage,
            'completion_history': self.completion_history,
            'violation_history': self.violation_history
        }

    def load_state(self, state: Dict[str, Any]):
        self.current_stage_idx = state.get('current_stage_idx', 0)
        self.episodes_in_stage = state.get('episodes_in_stage', 0)
        self.completion_history = state.get('completion_history', [])
        self.violation_history = state.get('violation_history', [])

    def get_progress_summary(self) -> Dict[str, Any]:
        mean_comp = np.mean(self.completion_history) if self.completion_history else 0.0
        mean_viol = np.mean(self.violation_history) if self.violation_history else 0.0
        
        return {
            'stage_name': self.current_stage_name,
            'stage_idx': self.current_stage_idx,
            'episodes_completed': self.episodes_in_stage,
            'rolling_mean_completion': mean_comp,
            'rolling_mean_violations': mean_viol,
            'target_completion': self.progression_cfg.get('completion_threshold', 0.95),
            'target_violations': self.progression_cfg.get('drc_violation_threshold', 0.02)
        }
