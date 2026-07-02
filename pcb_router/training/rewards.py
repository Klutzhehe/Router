from typing import Dict, Any, Optional

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
