# Google Colab Interactive Notebook Cell Code - Environment Smoke Test
# Copy and run these cells in your Jupyter/Colab notebook.

# %% Cell 1: Clone and install dependencies
# !git clone <your-repo>
# %cd Router
# !pip install -r requirements.txt

# %% Cell 2: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pcb_router.data.board_generator import BoardGenerator, BoardConfig
from pcb_router.env.board_state import BoardState
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.visualization.renderer import BoardRenderer

# %% Cell 3: Generate a procedural board
print("Generating procedural PCB...")
generator = BoardGenerator()
config = BoardConfig(
    board_width=400, board_height=400,
    num_nets=5, num_layers=2,
    num_components=4, obstacle_density=0.1
)
board = generator.generate(config)

print(f"Board size: {board.width}x{board.height} cells")
print(f"Nets to route: {len(board.nets)}")
print(f"Pads/Pins: {len(board.pins)}")
print(f"Obstacles: {len(board.obstacles)}")

# %% Cell 4: Initialize and render the board state
board_state = BoardState(board)
renderer = BoardRenderer(theme_dark=True)

fig = renderer.render_board(board_state, board, show_all_layers=True)
plt.show()

# %% Cell 5: Setup Gym Environment and test step
env = PCBRoutingEnv(board_config=config)
obs, info = env.reset(seed=42)

print("\nGym Environment Observation Space Details:")
for k, v in obs.items():
    if isinstance(v, np.ndarray):
        print(f" - {k}: Shape {v.shape}, dtype {v.dtype}")
    else:
        print(f" - {k}: {v}")

# Route first net with mock flat heatmap
print("\nRouting first net...")
mock_heatmaps = np.zeros((board.num_layers, board.height, board.width), dtype=np.float32)
mock_via_prob = np.zeros((board.height, board.width), dtype=np.float32)

next_obs, reward, terminated, truncated, next_info = env.step_with_heatmaps(
    net_index=0, layer_heatmaps=mock_heatmaps, via_prob_map=mock_via_prob
)

print(f"Routing result: Reward = {reward:.2f}")
print(f"Remaining unrouted nets: {next_obs['num_unrouted']}")
print(f"Active DRC violations: {next_info['drc_violations']}")

# Render routed board
fig2 = renderer.render_board(env.board_state, env.board, show_all_layers=True)
plt.show()
