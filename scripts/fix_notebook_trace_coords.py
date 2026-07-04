"""
fix_notebook_trace_coords.py — v2
Patches the board drawing code in Train_PCB_Router.ipynb to fix
the grid-integer vs mm coordinate mismatch for traces and vias.
"""
import json, sys

NOTEBOOK = "notebooks/Train_PCB_Router.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell with the board panel
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

# Reconstruct the source as a single string
cell_lines = nb["cells"][target_cell_idx]["source"]
cell_src = "".join(cell_lines)

# ── Targeted line-by-line replacements ───────────────────────────────────────
# We'll do exact substring replacements that are unambiguous

changes = [
    # Fix title to show routed count
    (
        'ax_board.set_title("Fully Routed Board (All Layers & Vias)", color=WHITE, fontsize=10, pad=6)',
        'ax_board.set_title(f"Fully Routed Board -- {len(trainer.env.routed_nets)}/{len(board.nets)} nets (all layers)", color=WHITE, fontsize=10, pad=6)'
    ),
    # Add resolution variable after title block
    (
        '        for spine in ax_board.spines.values():\n            spine.set_color(BORDER)\n            \n        # Draw obstacles',
        '        for spine in ax_board.spines.values():\n            spine.set_color(BORDER)\n            \n        # res: mm per grid unit. Trace/via coords are grid ints; pin coords are already mm.\n        res = state.resolution\n        \n        # Draw obstacles'
    ),
    # Fix Draw traces comment
    (
        '        # Draw traces\n        net_colors',
        '        # Draw traces -- seg coords are grid integers, multiply by res to get mm\n        net_colors'
    ),
    # Fix trace colour: net colour instead of layer colour
    (
        '        for seg in state.traces:\n            c = layer_colors[seg.layer % len(layer_colors)]\n            lw = max(1.2, seg.width / state.resolution)\n            ax_board.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],\n                          color=c, linewidth=lw, alpha=0.9, solid_capstyle="round")',
        '        for seg in state.traces:\n            c = net_colors[seg.net_id % len(net_colors)]\n            lw = max(1.2, seg.width / res)\n            ax_board.plot(\n                [seg.start_x * res, seg.end_x * res],\n                [seg.start_y * res, seg.end_y * res],\n                color=c, linewidth=lw, alpha=0.9, solid_capstyle="round")'
    ),
    # Fix Draw vias comment and coordinate scaling
    (
        '        # Draw vias\n        for via in state.vias:\n            ax_board.add_patch(mpatches.Circle(\n                (via.x, via.y), radius=3.5,\n                fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))\n            ax_board.add_patch(mpatches.Circle(\n                (via.x, via.y), radius=1.2,\n                fc=BG, zorder=6))',
        '        # Draw vias -- via.x/y are grid integers, multiply by res\n        for via in state.vias:\n            r_outer = (via.drill_size / 2.0 + via.annular_ring) / res\n            r_inner = via.drill_size / 2.0 / res\n            ax_board.add_patch(mpatches.Circle(\n                (via.x * res, via.y * res), radius=r_outer,\n                fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))\n            ax_board.add_patch(mpatches.Circle(\n                (via.x * res, via.y * res), radius=r_inner,\n                fc=BG, zorder=6))'
    ),
    # Fix pin radius to scale with resolution
    (
        '                (pin.global_x, pin.global_y), radius=2.5,',
        '                (pin.global_x, pin.global_y), radius=2.5 * res,'
    ),
]

applied = 0
for old, new in changes:
    if old in cell_src:
        cell_src = cell_src.replace(old, new)
        applied += 1
        print(f"  Applied: {old[:60].strip()!r}...")
    else:
        print(f"  SKIP (not found): {old[:60].strip()!r}...")

if applied == 0:
    print("No changes applied - notebook may already be patched or has drifted.")
    sys.exit(0)

print(f"\n{applied}/{len(changes)} changes applied.")

# Write back: split into lines, preserving newlines
new_lines = []
for line in cell_src.split("\n"):
    new_lines.append(line + "\n")
# Remove extra trailing newline artifact
if new_lines and new_lines[-1] == "\n":
    new_lines.pop()

nb["cells"][target_cell_idx]["source"] = new_lines

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Saved: {NOTEBOOK}")
