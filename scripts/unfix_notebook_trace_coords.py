"""
unfix_notebook_trace_coords.py
==============================
Reverts the * res scaling on trace and via coordinates in the notebook Train_PCB_Router.ipynb.
Everything (components, pins, obstacles, traces, vias) is already in grid cell units (0-400),
so we should plot them directly in grid cells without scaling down by res.
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

# Target substitutions to restore original coordinates (grid cell units)
replacements = [
    # 1. Traces: remove * res
    (
        '            ax_board.plot(\n                [seg.start_x * res, seg.end_x * res],\n                [seg.start_y * res, seg.end_y * res],',
        '            ax_board.plot(\n                [seg.start_x, seg.end_x],\n                [seg.start_y, seg.end_y],'
    ),
    # 2. Trace line width: seg.width / res -> seg.width / state.resolution
    (
        '            lw = max(1.2, seg.width / res)',
        '            lw = max(1.2, seg.width / state.resolution)'
    ),
    # 3. Vias: remove * res from coordinates, restore default radius
    (
        '        # Draw vias -- via.x/y are grid integers, multiply by res\n        for via in state.vias:\n            r_outer = (via.drill_size / 2.0 + via.annular_ring) / res\n            r_inner = via.drill_size / 2.0 / res\n            ax_board.add_patch(mpatches.Circle(\n                (via.x * res, via.y * res), radius=r_outer,\n                fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))\n            ax_board.add_patch(mpatches.Circle(\n                (via.x * res, via.y * res), radius=r_inner,\n                fc=BG, zorder=6))',
        '        # Draw vias\n        for via in state.vias:\n            ax_board.add_patch(mpatches.Circle(\n                (via.x, via.y), radius=3.5,\n                fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))\n            ax_board.add_patch(mpatches.Circle(\n                (via.x, via.y), radius=1.2,\n                fc=BG, zorder=6))'
    ),
    # 4. Pins: restore default radius
    (
        '            ax_board.add_patch(mpatches.Circle(\n                (pin.global_x, pin.global_y), radius=2.5 * res,',
        '            ax_board.add_patch(mpatches.Circle(\n                (pin.global_x, pin.global_y), radius=2.5,'
    )
]

applied = 0
for old, new in replacements:
    if old in cell_src:
        cell_src = cell_src.replace(old, new)
        applied += 1
        print(f"  Applied: {old[:60].strip()!r}...")
    else:
        print(f"  SKIP (not found): {old[:60].strip()!r}...")

if applied > 0:
    new_lines = [line + "\n" for line in cell_src.split("\n")]
    if new_lines and new_lines[-1] == "\n":
        new_lines.pop()
    nb["cells"][target_cell_idx]["source"] = new_lines
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully restored notebooks/Train_PCB_Router.ipynb")
else:
    print("No changes made.")
