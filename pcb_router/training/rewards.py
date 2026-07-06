from typing import Dict, Any, Optional

# Proximity attractor for the final approach to a target pad. The plain distance reward gives the
# same weak pull whether the cursor is far from or one cell short of the target, so near the pad
# that pull is drowned out by exploration noise and the turn penalty — the agent jitters next to
# the pad and never lands, so the net never completes. This potential ramps up steeply within
# NEAR_TARGET_RADIUS cells, giving the last few steps a strong, committing reward. It is applied as
# a per-step DELTA (potential-based shaping), so it cannot be farmed by hovering near the target.
# Keep these constants in sync with DreamerJEPATrainer._imagine_autoregressive_rollout, which
# imports them so the imagined objective matches the real one.
NEAR_TARGET_RADIUS = 6.0
NEAR_TARGET_GAIN = 1.5


def near_target_potential(dist: float) -> float:
    """Potential that is high on the target and decays to zero beyond NEAR_TARGET_RADIUS cells."""
    return NEAR_TARGET_GAIN * max(0.0, NEAR_TARGET_RADIUS - dist)


class RewardCalculator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default weight parameters
        self.weights = {
            'completion': 1.0,
            'wirelength': 0.1,
            'drc_violations': 0.5,
            'congestion': 0.2,
            'length_error': 0.3,
            'all_complete_bonus': 0.5
        }
        if weights is not None:
            self.update_weights(weights)

    def update_weights(self, weights: Optional[Dict[str, float]]):
        if weights:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = float(v)

    def calculate(self, routing_result: Dict[str, Any]) -> float:
        """
        Calculate dense routing step reward
        Args:
            routing_result: Dict containing:
                - 'connected': bool
                - 'wirelength': float (mm)
                - 'manhattan_distance': float (mm)
                - 'drc_violations': int
                - 'congestion_increase': float
                - 'length_error': float
                - 'all_nets_complete': bool
        """
        w = self.weights
        r = 0.0
        
        # 1. Completion reward
        if routing_result.get('connected', False):
            r += w['completion'] * 1.0
        else:
            r -= w['completion'] * 0.5 # Penalty for pathfinding failure
            
        # 2. Detour ratio penalty (penalize long, winding trace segments)
        wirelength = routing_result.get('wirelength', 0.0)
        manhattan_dist = routing_result.get('manhattan_distance', 1.0)
        
        if wirelength > 0 and manhattan_dist > 0:
            detour_ratio = max(0.0, (wirelength / manhattan_dist) - 1.0)
            r -= w['wirelength'] * detour_ratio
            
        # 3. DRC violation penalty
        drc_violations = routing_result.get('drc_violations', 0)
        r -= w['drc_violations'] * drc_violations
        
        # 4. Congestion penalty
        congestion_inc = routing_result.get('congestion_increase', 0.0)
        r -= w['congestion'] * congestion_inc
        
        # 5. Length tuning error penalty
        length_err = routing_result.get('length_error', 0.0)
        r -= w['length_error'] * length_err
        
        # 6. All nets complete bonus
        if routing_result.get('all_nets_complete', False):
            r += w['all_complete_bonus']
            
        return float(r)

    def calculate_step(self, step_info: dict) -> float:
        """
        Calculate step-level reward for per-cell routing decisions.
        Args:
            step_info: Dict containing:
                - 'dist_delta': float (dist_prev - dist_curr)
                - 'invalid_move': bool
                - 'direction_changed': bool
                - 'is_via': bool
        """
        # Direction change penalty and via cost for RL step decisions.
        # NOTE: Unlike pathfinder.py's static A* search costs (where 15.0 acts as a tie-breaker),
        # these values act as dense step rewards. They are scaled relative to dist_delta (+/-1.0)
        # to prevent via/turn avoidance from overpowering the progress signal.
        # The turn penalty is 0.5 (was 0.2): at 0.2 turns were nearly free relative to ~1.0 progress,
        # so the policy produced jagged staircase traces. 0.5 favors straight runs while still
        # allowing a necessary turn (progress ~1.0 - 0.5 = net positive). Keep this in sync with the
        # imagined-rollout turn penalty in DreamerJEPATrainer._imagine_autoregressive_rollout.
        direction_change_penalty = 0.5
        base_via_cost = 0.5
        invalid_move_penalty = 1.0  # matches step size order of magnitude
        
        r = 0.0

        # 1. Distance progress (positive is good, negative is bad)
        r += step_info.get('dist_delta', 0.0)

        # 1b. Proximity attractor (potential-based): a steep extra pull within NEAR_TARGET_RADIUS
        # of the target so the agent commits to landing on the pad instead of jittering next to it.
        dist_prev = step_info.get('dist_prev')
        dist_curr = step_info.get('dist_curr')
        if dist_prev is not None and dist_curr is not None:
            r += near_target_potential(dist_curr) - near_target_potential(dist_prev)

        # 2. Invalid move penalty
        if step_info.get('invalid_move', False):
            r -= invalid_move_penalty
            
        # 3. Direction change penalty
        if step_info.get('direction_changed', False):
            r -= direction_change_penalty
            
        # 4. Via cost
        if step_info.get('is_via', False):
            r -= base_via_cost
            
        return float(r)
