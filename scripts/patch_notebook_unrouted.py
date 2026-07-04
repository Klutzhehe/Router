"""
patch_notebook_unrouted.py
==========================
Patches the board drawing code in Train_PCB_Router.ipynb to show
unrouted/failed nets as light dotted red lines, and also show n_routed/n_total
in the title.
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

# Target substring to insert the unrouted nets visualization
target_pins_block = """        # Draw pins
        for pin in board.pins.values():
            c = net_colors[pin.net_id % len(net_colors)]
            ax_board.add_patch(mpatches.Circle(
                (pin.global_x, pin.global_y), radius=2.5 * res,
                fc=c, ec=WHITE, lw=0.6, alpha=0.95, zorder=8))"""

replacement_pins_block = target_pins_block + """
                
        # Draw unrouted/failed nets as light dotted red lines
        unrouted_nets = [net for net in board.nets if net.id not in state.routed_net_ids]
        for net in unrouted_nets:
            pins = [board.pins[pid] for pid in net.pin_ids if pid in board.pins]
            for idx in range(len(pins) - 1):
                p1, p2 = pins[idx], pins[idx+1]
                ax_board.plot([p1.global_x, p2.global_x], [p1.global_y, p2.global_y],
                               color="#EF4444", linestyle=":", linewidth=1.2, alpha=0.6)"""

if target_pins_block in cell_src:
    cell_src = cell_src.replace(target_pins_block, replacement_pins_block)
    print("Successfully replaced target pins block to include unrouted lines!")
else:
    # Try with raw string representation
    target_pins_block_escaped = target_pins_block.replace("\n", "\\n")
    print("Target block not found as plain text, trying escaped search...")
    # Let's do exact match on lines
    found_start = -1
    for idx, line in enumerate(cell_lines):
        if "Draw pins" in line:
            found_start = idx
            break
    if found_start != -1:
        # Insert the lines
        insert_idx = found_start + 6
        unrouted_lines = [
            "        # Draw unrouted/failed nets as light dotted red lines\n",
            "        unrouted_nets = [net for net in board.nets if net.id not in state.routed_net_ids]\n",
            "        for net in unrouted_nets:\n",
            "            pins = [board.pins[pid] for pid in net.pin_ids if pid in board.pins]\n",
            "            for idx in range(len(pins) - 1):\n",
            "                p1, p2 = pins[idx], pins[idx+1]\n",
            "                ax_board.plot([p1.global_x, p2.global_x], [p1.global_y, p2.global_y],\n",
            "                               color=\"#EF4444\", linestyle=\":\", linewidth=1.2, alpha=0.6)\n"
        ]
        cell_lines[insert_idx:insert_idx] = unrouted_lines
        cell_src = "".join(cell_lines)
        print("Successfully inserted unrouted lines using list splice!")
    else:
        print("ERROR: Could not find 'Draw pins' block!")
        sys.exit(1)

# Split back into lines preserving newline characters
new_lines = []
for line in cell_src.split("\n"):
    new_lines.append(line + "\n")
if new_lines and new_lines[-1] == "\n":
    new_lines.pop()

nb["cells"][target_cell_idx]["source"] = new_lines

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Saved notebooks/Train_PCB_Router.ipynb")
