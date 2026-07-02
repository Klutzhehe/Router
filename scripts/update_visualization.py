import json

notebook_path = "notebooks/Train_PCB_Router.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 5 (Training Visualization)
cell5_src = "".join(nb["cells"][5]["source"])

target_render_block = """    # ── Board state panel ─────────────────────────────────────────
    ax_board = fig.add_subplot(gs[:, 2:])
    ax_board.set_facecolor(PANEL)
    ax_board.set_title(
        f"Board State   {len(trainer.env.routed_nets)}/{len(trainer.env.board.nets)} nets routed   "
        f"{trainer.env.W}x{trainer.env.H}px",
        color=WHITE, fontsize=10, pad=6
    )
    ax_board.set_xticks([]); ax_board.set_yticks([])
    for spine in ax_board.spines.values():
        spine.set_color(BORDER)

    try:
        state  = trainer.env.board_state
        board  = trainer.env.board
        import matplotlib.patches as mpatches

        for obs in board.obstacles:
            ax_board.add_patch(mpatches.Rectangle(
                (obs.x, obs.y), obs.width, obs.height,
                fc="#EF4444", alpha=0.25, lw=0))
        for comp in board.components:
            ax_board.add_patch(mpatches.Rectangle(
                (comp.x, comp.y), comp.width, comp.height,
                fc="#1e2040", ec="#4B5563", lw=1.2, alpha=0.85))
            ax_board.text(comp.x + comp.width / 2, comp.y + comp.height / 2,
                          comp.name, color=WHITE, fontsize=7, ha="center", va="center")
        net_colors = ["#3B82F6","#10B981","#EC4899","#8B5CF6","#06B6D4",
                      "#F59E0B","#14B8A6","#6366F1","#A855F7","#10B981"]
        for seg in state.traces:
            c = net_colors[seg.net_id % len(net_colors)]
            ax_board.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                          color=c, linewidth=2.0, solid_capstyle="round")
        for pin in board.pins.values():
            c = net_colors[pin.net_id % len(net_colors)]
            ax_board.add_patch(mpatches.Circle(
                (pin.global_x, pin.global_y), radius=2.5,
                fc=c, ec=WHITE, lw=0.6, alpha=0.95))
        ax_board.set_xlim(0, board.width)
        ax_board.set_ylim(0, board.height)
        ax_board.set_aspect("equal")
    except Exception as e:
        ax_board.text(0.5, 0.5, f"Board render error:\\n{e}",
                      transform=ax_board.transAxes, color=WHITE,
                      ha="center", va="center", fontsize=9)"""

replacement_render_block = """    # ── Board state panel (split by active layers) ─────────────────
    try:
        state = trainer.env.board_state
        board = trainer.env.board
        num_layers = board.num_layers
        import matplotlib.patches as mpatches

        if num_layers == 1:
            rows, cols = 1, 1
        elif num_layers == 2:
            rows, cols = 1, 2
        else:
            rows = int(np.ceil(num_layers / 2))
            cols = 2

        sub_gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=gs[:, 2:], hspace=0.3, wspace=0.25
        )

        layer_names = ["Top Layer (L0)", "Bottom Layer (L1)"] + [f"Layer {l}" for l in range(2, num_layers)]
        layer_colors_title = ["#F43F5E", "#06B6D4"] + ["#8B5CF6", "#F59E0B"]

        for l in range(num_layers):
            r_idx = l // cols
            c_idx = l % cols
            ax_lay = fig.add_subplot(sub_gs[r_idx, c_idx])
            ax_lay.set_facecolor(PANEL)
            
            t_color = layer_colors_title[l] if l < len(layer_colors_title) else WHITE
            active_traces = [s for s in state.traces if s.layer == l]
            ax_lay.set_title(
                f"{layer_names[l] if l < len(layer_names) else f'Layer {l}'}   "
                f"({len(active_traces)} traces)",
                color=t_color, fontsize=10, pad=6
            )
            ax_lay.set_xticks([]); ax_lay.set_yticks([])
            for spine in ax_lay.spines.values():
                spine.set_color(BORDER)

            # Draw obstacles
            for obs in board.obstacles:
                if obs.layer == -1 or obs.layer == l:
                    ax_lay.add_patch(mpatches.Rectangle(
                        (obs.x, obs.y), obs.width, obs.height,
                        fc="#EF4444", alpha=0.15, lw=0))

            # Draw keepout zones
            for ko in board.keep_out_zones:
                if ko.layer == -1 or ko.layer == l:
                    ax_lay.add_patch(mpatches.Rectangle(
                        (ko.x, ko.y), ko.width, ko.height,
                        fc="#F59E0B", alpha=0.15, lw=0))

            # Draw components
            for comp in board.components:
                ax_lay.add_patch(mpatches.Rectangle(
                    (comp.x, comp.y), comp.width, comp.height,
                    fc="#1e2040", ec="#4B5563", lw=1.0, alpha=0.7))
                ax_lay.text(comp.x + comp.width / 2, comp.y + comp.height / 2,
                            comp.name, color="#888899", fontsize=6, ha="center", va="center")

            # Draw traces
            net_colors = ["#3B82F6","#10B981","#EC4899","#8B5CF6","#06B6D4",
                          "#F59E0B","#14B8A6","#6366F1","#A855F7","#10B981"]
            for seg in state.traces:
                if seg.layer == l:
                    c = net_colors[seg.net_id % len(net_colors)]
                    ax_lay.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                                color=c, linewidth=2.0, solid_capstyle="round")

            # Draw vias connecting to/from this layer
            for via in state.vias:
                if via.from_layer == l or via.to_layer == l:
                    ax_lay.add_patch(mpatches.Circle(
                        (via.x, via.y), radius=3.5,
                        fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))
                    ax_lay.add_patch(mpatches.Circle(
                        (via.x, via.y), radius=1.2,
                        fc=BG, zorder=6))

            # Draw pins (solid on active layer, faded/dashed on other layer)
            for pin in board.pins.values():
                c = net_colors[pin.net_id % len(net_colors)]
                is_active = (pin.layer == l)
                alpha = 0.95 if is_active else 0.25
                ls = 'solid' if is_active else 'dashed'
                ax_lay.add_patch(mpatches.Circle(
                    (pin.global_x, pin.global_y), radius=2.5,
                    fc=c, ec=WHITE if is_active else "#555566", lw=0.6, alpha=alpha, linestyle=ls))

            ax_lay.set_xlim(0, board.width)
            ax_lay.set_ylim(0, board.height)
            ax_lay.set_aspect("equal")
    except Exception as e:
        ax_err = fig.add_subplot(gs[:, 2:])
        ax_err.set_facecolor(PANEL)
        ax_err.text(0.5, 0.5, f"Board render error:\\n{e}",
                     transform=ax_err.transAxes, color=WHITE,
                     ha="center", va="center", fontsize=9)"""

if target_render_block in cell5_src:
    cell5_src = cell5_src.replace(target_render_block, replacement_render_block)
    nb["cells"][5]["source"] = [line + "\n" for line in cell5_src.split("\n")]
    if nb["cells"][5]["source"][-1] == "\n":
        nb["cells"][5]["source"].pop()
    print("Cell 5 board render block successfully updated!")
else:
    # Try normalizing line endings if not found
    normalized_cell5 = cell5_src.replace("\r\n", "\n")
    normalized_target = target_render_block.replace("\r\n", "\n")
    if normalized_target in normalized_cell5:
        normalized_cell5 = normalized_cell5.replace(normalized_target, replacement_render_block)
        nb["cells"][5]["source"] = [line + "\n" for line in normalized_cell5.split("\n")]
        if nb["cells"][5]["source"][-1] == "\n":
            nb["cells"][5]["source"].pop()
        print("Cell 5 board render block successfully updated (after normalization)!")
    else:
        print("Warning: could not find target render block in Cell 5!")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook Train_PCB_Router.ipynb updated successfully!")
