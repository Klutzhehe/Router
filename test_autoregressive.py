import sys
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.data.board_generator import BoardConfig

if __name__ == '__main__':
    # Generate a tiny board with 1 net to test autoregressive targeting
    config = BoardConfig()
    config.grid_width = 20
    config.grid_height = 20
    config.num_nets = 1
    config.num_components = 2

    env = PCBRoutingEnv(config)
    obs, info = env.reset(seed=42)

    env.start_routing_net(0)
    target_x, target_y, target_l = env.target_pos

    # Manually step the agent exactly towards the target edge
    print(f"Agent starting at: {env.cursor_pos}")
    print(f"Target is at: {env.target_pos}")

    done = False
    success = False
    for _ in range(50):
        cx, cy, cl = env.cursor_pos
        
        # Move towards target
        dx = target_x - cx
        dy = target_y - cy
        
        if abs(dx) <= 3 and abs(dy) <= 3:
            # We are at the edge!
            print(f"Reached edge of target at {cx}, {cy}!")
            
        action = 0 # default up
        if dx > 0: action = 2 # right
        elif dx < 0: action = 3 # left
        elif dy > 0: action = 0 # up
        elif dy < 0: action = 1 # down
            
        obs, reward, terminated, truncated, info = env.step_move(action)
        if terminated or truncated:
            print(f"Episode ended. Completion rate: {info['completion_rate']}")
            success = info['completion_rate'] == 1.0
            break

    # Run final validation to be absolutely sure
    is_valid, report = env.validate_final_board()
    print(f"Final Validation Valid? {is_valid}")
    print(f"Report: {report}")
