"""
patch_notebook_speed_config.py
==============================
Patches default configuration overrides in Train_PCB_Router.ipynb
to optimize RAM and GPU VRAM resource usage for higher training speed.
"""
import json, sys

NOTEBOOK = "notebooks/Train_PCB_Router.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 2 (configuration cell)
target_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if 'CONFIG = {' in src and '"REAL_STEPS_PER_ITERATION"' in src:
        target_cell_idx = i
        break

if target_cell_idx is None:
    print("ERROR: Could not find configuration cell!")
    sys.exit(1)

print(f"Found configuration in cell {target_cell_idx}")

cell_lines = nb["cells"][target_cell_idx]["source"]
cell_src = "".join(cell_lines)

# Target substitutions
replacements = [
    (
        '    "REAL_STEPS_PER_ITERATION":  64,',
        '    "REAL_STEPS_PER_ITERATION":  512,  # increased from 64; faster collection, less loop overhead'
    ),
    (
        '    "REPLAY_BUFFER_SIZE":        5000,',
        '    "REPLAY_BUFFER_SIZE":        20000, # increased from 5000; utilizes more system RAM for memory storage'
    ),
    (
        '    "IMAGINE_BATCH_SIZE":        512,',
        '    "IMAGINE_BATCH_SIZE":        2048,  # increased from 512; scales up GPU VRAM utilization and speeds up training'
    )
]

applied = 0
for old, new in replacements:
    if old in cell_src:
        cell_src = cell_src.replace(old, new)
        applied += 1
        print(f"  Applied: {old} -> {new}")
    else:
        print(f"  SKIP (not found): {old}")

if applied > 0:
    new_lines = [line + "\n" for line in cell_src.split("\n")]
    if new_lines and new_lines[-1] == "\n":
        new_lines.pop()
    nb["cells"][target_cell_idx]["source"] = new_lines
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully saved notebooks/Train_PCB_Router.ipynb")
else:
    print("No changes made.")
