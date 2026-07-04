"""
patch_notebook_use_cache.py
===========================
Patches Train_PCB_Router.ipynb to render the last completed board state
(if available) instead of the active environment's intermediate board state,
ensuring we always display fully finished boards on the training dashboard.
"""
import json, sys

NOTEBOOK = "notebooks/Train_PCB_Router.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 5 (board panel cell)
target_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "ax_board = fig.add_subplot(sub_gs[0, :])" in src and "Draw traces" in src:
        target_cell_idx = i
        break

if target_cell_idx is None:
    print("ERROR: Could not find board panel cell!")
    sys.exit(1)

print(f"Found board panel in cell {target_cell_idx}")

cell_lines = nb["cells"][target_cell_idx]["source"]
cell_src = "".join(cell_lines)

# Target substitutions
old_block = """    # ── Board state panel (Fully routed board + Layer heatmaps) ──
    try:
        state = trainer.env.board_state
        board = trainer.env.board"""

new_block = """    # ── Board state panel (Fully routed board + Layer heatmaps) ──
    try:
        # Use the last completed board state if available to show a fully routed result,
        # otherwise fall back to the active environment state.
        if hasattr(trainer, 'last_completed_board_state') and trainer.last_completed_board_state is not None:
            state = trainer.last_completed_board_state
            board = trainer.last_completed_board
        else:
            state = trainer.env.board_state
            board = trainer.env.board"""

if old_block in cell_src:
    cell_src = cell_src.replace(old_block, new_block)
    print("Successfully patched notebook to use last_completed_board_state!")
else:
    # Try finding target line by line
    print("Direct block match failed, searching for lines...")
    found_idx = -1
    for idx, line in enumerate(cell_lines):
        if "state = trainer.env.board_state" in line:
            found_idx = idx
            break
    if found_idx != -1:
        # Replace the state/board lines
        cell_lines[found_idx:found_idx+2] = [
            "        # Use the last completed board state if available to show a fully routed result,\n",
            "        # otherwise fall back to the active environment state.\n",
            "        if hasattr(trainer, 'last_completed_board_state') and trainer.last_completed_board_state is not None:\n",
            "            state = trainer.last_completed_board_state\n",
            "            board = trainer.last_completed_board\n",
            "        else:\n",
            "            state = trainer.env.board_state\n",
            "            board = trainer.env.board\n"
        ]
        cell_src = "".join(cell_lines)
        print("Successfully patched notebook via list splice!")
    else:
        print("ERROR: Could not find target lines!")
        sys.exit(1)

new_lines = [line + "\n" for line in cell_src.split("\n")]
if new_lines and new_lines[-1] == "\n":
    new_lines.pop()

nb["cells"][target_cell_idx]["source"] = new_lines

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Saved notebooks/Train_PCB_Router.ipynb")
