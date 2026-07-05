import json
import os

notebook_path = "notebooks/Train_PCB_Router.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Deduplicate cells by header
seen_headers = set()
unique_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    header = None
    for line in cell.get("source", []):
        if "CELL" in line:
            header = line.strip()
            break
    if header and header in seen_headers:
        print(f"Deduplicating notebook: removing extra cell '{header}'")
        continue
    if header:
        seen_headers.add(header)
    unique_cells.append(cell)
nb["cells"] = unique_cells

# Find cell indices dynamically
config_cell_idx = 1
init_cell_idx = 4
training_cell_idx = 5
eval_cell_idx = 6

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "CELL 2 — CONFIGURATION" in src:
        config_cell_idx = i
    elif "CELL 5 — INITIALIZE TRAINER" in src:
        init_cell_idx = i
    elif "CELL 6 — TRAINING WITH LIVE VISUALS" in src:
        training_cell_idx = i
    elif "CELL 7 — LOAD CHECKPOINT + EVALUATE" in src:
        eval_cell_idx = i

# Cell 1 (CONFIG cell)
# Find the config cell (Cell 1) and update CONFIG dictionary
config_source = nb["cells"][config_cell_idx]["source"]
# Let's rebuild the CONFIG dictionary source lines
new_config_source = [
    "# ================================================================\n",
    "#  CELL 2 — CONFIGURATION  (Edit this cell before running)\n",
    "# ================================================================\n",
    "\n",
    "CONFIG = {\n",
    "    # -----------------------------------------------------------\n",
    "    #  How to get the code into Colab — pick ONE method:\n",
    "    #\n",
    "    #  Method A (easiest): Upload a zip of the project.\n",
    "    #    Set REPO_URL = None  →  Cell 4 will prompt you to upload a zip.\n",
    "    #\n",
    "    #  Method B: Clone from GitHub.\n",
    "    #    Set REPO_URL = \"https://github.com/YOUR_USERNAME/Router.git\"\n",
    "    # -----------------------------------------------------------\n",
    "    \"REPO_URL\": None,   # <-- None = zip upload, or paste your GitHub URL here\n",
    "    \"REPO_DIR\": \"/content/Router\",\n",
    "\n",
    "    # -----------------------------------------------------------\n",
    "    #  Checkpoints\n",
    "    # -----------------------------------------------------------\n",
    "    # Google Drive path (survives session restarts — recommended):\n",
    "    \"CHECKPOINT_DIR\": \"/content/drive/MyDrive/pcb_router/checkpoints\",\n",
    "    # Local Colab path (lost when session ends):\n",
    "    # \"CHECKPOINT_DIR\": \"/content/checkpoints\",\n",
    "\n",
    "    # Resume from a previous checkpoint, or None to start fresh:\n",
    "    \"LOAD_CHECKPOINT\": None,\n",
    "    # Example: \"/content/drive/MyDrive/pcb_router/checkpoints/checkpoint_50000.pt\"\n",
    "\n",
    "    # -----------------------------------------------------------\n",
    "    #  Training\n",
    "    # -----------------------------------------------------------\n",
    "    \"TOTAL_TIMESTEPS\": 5_000_000,\n",
    "    \"SAVE_INTERVAL\":   10_000,     # Save a .pt checkpoint every N timesteps\n",
    "    \"VIZ_INTERVAL\":    1,          # Show live dashboard every N rollout updates\n",
    "\n",
    "    # -----------------------------------------------------------\n",
    "    #  DreamerV3 Hyperparameters  (override configs/training.yaml)\n",
    "    # -----------------------------------------------------------\n",
    "    \"REAL_STEPS_PER_ITERATION\":  64,\n",
    "    \"TRAIN_RATIO\":               100,\n",
    "    \"REPLAY_BUFFER_SIZE\":        5000,\n",
    "    \"IMAGINE_BATCH_SIZE\":        512,\n",
    "    \"IMAGINATION_HORIZON_START\": 5,\n",
    "    \"IMAGINATION_HORIZON_END\":   15,\n",
    "    \"WORLD_MODEL_LR\":            3e-4,\n",
    "    \"ACTOR_LR\":                  8e-5,\n",
    "    \"CRITIC_LR\":                 8e-5,\n",
    "    \"GAMMA\":                     0.997,\n",
    "    \"LAMBDA\":                    0.95,\n",
    "\n",
    "    # -----------------------------------------------------------\n",
    "    #  Model  (override configs/model.yaml)\n",
    "    # -----------------------------------------------------------\n",
    "    \"MAX_GRID_SIZE\": 256,          # Do NOT exceed 512 without 16GB+ VRAM\n",
    "\n",
    "    # -----------------------------------------------------------\n",
    "    #  Logging\n",
    "    # -----------------------------------------------------------\n",
    "    \"USE_WANDB\":      False,\n",
    "    \"WANDB_PROJECT\":  \"pcb-router\",\n",
    "    \"WANDB_RUN_NAME\": None,\n",
    "    \"COMPILE_MODELS\": True,\n",
    "    \"LAUNCH_GRADIO_DURING_TRAINING\": False,\n",
    "}\n",
    "\n",
    "print(\"Config loaded:\")\n",
    "for k, v in CONFIG.items():\n",
    "    print(f\"  {k}: {v}\")\n"
]
nb["cells"][config_cell_idx]["source"] = new_config_source

# Cell 4 (Initialize Trainer)
new_init_source = [
    "# ================================================================\n",
    "#  CELL 5 — INITIALIZE TRAINER\n",
    "# ================================================================\n",
    "import os, sys\n",
    "\n",
    "# ── Clear Python import cache to force reloading from disk ─────\n",
    "for key in list(sys.modules.keys()):\n",
    "    if key in [\"pcb_router\", \"scripts\"] or key.startswith(\"pcb_router.\") or key.startswith(\"scripts.\"):\n",
    "        del sys.modules[key]\n",
    "\n",
    "sys.path.insert(0, CONFIG[\"REPO_DIR\"])\n",
    "os.chdir(CONFIG[\"REPO_DIR\"])\n",
    "\n",
    "from pcb_router.training.trainer import DreamerJEPATrainer\n",
    "\n",
    "# Initialize the DreamerJEPATrainer\n",
    "trainer = DreamerJEPATrainer(\n",
    "    config_path=\"configs/training.yaml\",\n",
    "    model_config_path=\"configs/model.yaml\",\n",
    "    curriculum_config_path=\"configs/curriculum.yaml\",\n",
    "    device=\"auto\",                             # auto-selects GPU if available\n",
    "    checkpoint_dir=CONFIG[\"CHECKPOINT_DIR\"],\n",
    "    load_checkpoint_path=CONFIG[\"LOAD_CHECKPOINT\"],\n",
    ")\n",
    "\n",
    "# Apply CONFIG overrides dynamically\n",
    "trainer.real_steps_per_iteration = CONFIG.get(\"REAL_STEPS_PER_ITERATION\", 64)\n",
    "trainer.train_ratio = CONFIG.get(\"TRAIN_RATIO\", 100)\n",
    "trainer.replay_buffer.capacity_episodes = CONFIG.get(\"REPLAY_BUFFER_SIZE\", 5000)\n",
    "trainer.imagine_batch_size = CONFIG.get(\"IMAGINE_BATCH_SIZE\", 512)\n",
    "trainer.imagination_horizon_start = CONFIG.get(\"IMAGINATION_HORIZON_START\", 5)\n",
    "trainer.imagination_horizon_end = CONFIG.get(\"IMAGINATION_HORIZON_END\", 15)\n",
    "trainer.gamma = CONFIG.get(\"GAMMA\", 0.997)\n",
    "trainer.lambda_ = CONFIG.get(\"LAMBDA\", 0.95)\n",
    "trainer.compile_models = CONFIG.get(\"COMPILE_MODELS\", True)\n",
    "\n",
    "for pg in trainer.wm_opt.param_groups:\n",
    "    pg[\"lr\"] = CONFIG.get(\"WORLD_MODEL_LR\", 3e-4)\n",
    "for pg in trainer.actor_opt.param_groups:\n",
    "    pg[\"lr\"] = CONFIG.get(\"ACTOR_LR\", 8e-5)\n",
    "for pg in trainer.critic_opt.param_groups:\n",
    "    pg[\"lr\"] = CONFIG.get(\"CRITIC_LR\", 8e-5)\n",
    "\n",
    "trainer.train_cfg[\"training\"][\"save_interval\"] = CONFIG[\"SAVE_INTERVAL\"]\n",
    "\n",
    "if CONFIG[\"USE_WANDB\"]:\n",
    "    import wandb\n",
    "    wandb.init(\n",
    "        project=CONFIG[\"WANDB_PROJECT\"],\n",
    "        name=CONFIG[\"WANDB_RUN_NAME\"],\n",
    "        config=CONFIG\n",
    "    )\n",
    "    print(f\"W&B run: {wandb.run.url}\")\n",
    "\n",
    "print(\"Trainer ready!\")\n",
    "print(f\"  Device:           {trainer.device}\")\n",
    "print(f\"  Checkpoint dir:   {trainer.checkpoint_dir}\")\n",
    "print(f\"  Resuming from:    {CONFIG['LOAD_CHECKPOINT'] or 'scratch'}\")\n",
    "print(f\"  Timesteps so far: {trainer.total_timesteps:,}\")\n"
]

nb["cells"][init_cell_idx]["source"] = new_init_source

# Cell 5 (Training Cell)
training_source_str = "".join(nb["cells"][training_cell_idx]["source"])
target_metrics_str = """    metrics_to_plot = [
        (h["completion_rate"], "Routing Completion Rate", TEAL,   "Rate",  (0, 0)),
        (h["loss_policy"],     "Policy Loss (PPO)",       BLUE,   "Loss",  (0, 1)),
        (h["loss_jepa"],       "JEPA World Model Loss",   PURPLE, "Loss",  (1, 0)),
        (h["loss_value"],      "Value Function Loss",     AMBER,  "Loss",  (1, 1)),
    ]"""

replacement_metrics_str = """    is_dreamer = "loss_wm" in h
    if is_dreamer:
        metrics_to_plot = [
            (h["completion_rate"], "Routing Completion Rate", TEAL,   "Rate",  (0, 0)),
            (h["loss_actor"],      "Actor Loss (Dreamer)",    BLUE,   "Loss",  (0, 1)),
            (h["loss_wm"],         "JEPA World Model Loss",   PURPLE, "Loss",  (1, 0)),
            (h["loss_critic"],     "Critic Loss (Dreamer)",   AMBER,  "Loss",  (1, 1)),
        ]
    else:
        metrics_to_plot = [
            (h["completion_rate"], "Routing Completion Rate", TEAL,   "Rate",  (0, 0)),
            (h["loss_policy"],     "Policy Loss (PPO)",       BLUE,   "Loss",  (0, 1)),
            (h["loss_jepa"],       "JEPA World Model Loss",   PURPLE, "Loss",  (1, 0)),
            (h["loss_value"],      "Value Function Loss",     AMBER,  "Loss",  (1, 1)),
        ]"""

if target_metrics_str in training_source_str:
    training_source_str = training_source_str.replace(target_metrics_str, replacement_metrics_str)
else:
    print("Warning: could not find target metrics_to_plot block in training cell!")

target_on_update_end = """    clear_output(wait=True)
    ipydisplay.display(fig)
    plt.close(fig)"""

replacement_on_update_end = """    clear_output(wait=True)
    ipydisplay.display(fig)
    if CONFIG["USE_WANDB"]:
        import wandb
        try:
            # Log training dashboard figure as an image
            wandb.log({"training/dashboard": wandb.Image(fig, caption=f"Training Dashboard at Step {ts}")})
            
            # Periodically log step-by-step routing viz of the whole model (first update + every 20 updates)
            if _update_count[0] == 1 or _update_count[0] % 20 == 0:
                from scripts.visualize_routing_wandb import log_training_rollout_viz
                stage_cfg = trainer.curriculum.current_stage
                is_dreamer = hasattr(trainer, 'jepa')
                log_training_rollout_viz(
                    trainer=trainer,
                    is_dreamer=is_dreamer,
                    stage_cfg=stage_cfg,
                    seed=42, # stable seed to compare progress
                    current_step=ts
                )
        except Exception as e:
            print(f"Warning: Failed to log W&B visuals: {e}")
    plt.close(fig)"""

if target_on_update_end in training_source_str:
    training_source_str = training_source_str.replace(target_on_update_end, replacement_on_update_end)
else:
    # Try with single quotes if double quotes were used in notebook
    target_on_update_end_alt = """    clear_output(wait=True)
    ipydisplay.display(fig)
    plt.close(fig)"""
    training_source_str = training_source_str.replace(target_on_update_end_alt, replacement_on_update_end)

target_training_start = 'print(f"Starting training for {CONFIG[\'TOTAL_TIMESTEPS\']:,} timesteps...")'
replacement_training_start = """# ── Optional live Gradio server during training ───────────────────
if CONFIG.get("LAUNCH_GRADIO_DURING_TRAINING", False):
    try:
        import gradio
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    try:
        from scripts.visualize_routing_gradio import build_gradio_app
        is_dreamer = hasattr(trainer, 'jepa')
        cur_cfg = {"stages": trainer.curriculum.stages}
        print("\\n[Gradio] Launching live step-by-step routing visualizer...")
        demo = build_gradio_app(trainer, is_dreamer, cur_cfg)
        # share=True creates a public URL.
        demo.launch(share=True)
        print("[Gradio] Live server is running! Open the public URL in a new tab to route boards interactively with active weights.\\n")
    except Exception as e:
        print(f"\\n[Gradio] Warning: Failed to launch live visualizer: {e}\\n")

print(f"Starting training for {CONFIG['TOTAL_TIMESTEPS']:,} timesteps...")"""

if target_training_start in training_source_str:
    training_source_str = training_source_str.replace(target_training_start, replacement_training_start)
else:
    # Try alt format single quotes
    target_training_start_alt = "print(f\\\"Starting training for {CONFIG['TOTAL_TIMESTEPS']:,} timesteps...\\\")"
    training_source_str = training_source_str.replace(target_training_start_alt, replacement_training_start)

# ── Dynamic replacement for the right board state panel ──────────────────
try_start_idx = training_source_str.find("    # \\u2500\\u2500 Board state panel")
if try_start_idx == -1:
    try_start_idx = training_source_str.find("    # ── Board state panel")

end_idx = training_source_str.find("    clear_output(wait=True)")

if try_start_idx != -1 and end_idx != -1:
    new_render_code = """    # ── Board state panel (Fully routed board + Layer heatmaps) ──
    try:
        state = trainer.env.board_state
        board = trainer.env.board
        num_layers = board.num_layers
        import matplotlib.patches as mpatches
        
        # Grid layout for the right side: Row 0 is the fully routed board, Row 1 is the layer heatmaps
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[:, 2:], hspace=0.35, wspace=0.25
        )
        
        # 1. Draw the Fully Routed Board (taking the entire Row 0)
        ax_board = fig.add_subplot(sub_gs[0, :])
        ax_board.set_facecolor(PANEL)
        ax_board.set_title("Fully Routed Board (All Layers & Vias)", color=WHITE, fontsize=10, pad=6)
        ax_board.set_xticks([]); ax_board.set_yticks([])
        for spine in ax_board.spines.values():
            spine.set_color(BORDER)
            
        # Draw obstacles
        for obs in board.obstacles:
            ax_board.add_patch(mpatches.Rectangle(
                (obs.x, obs.y), obs.width, obs.height,
                fc="#EF4444", alpha=0.15, lw=0, hatch='//'))
                
        # Draw keepout zones
        for ko in board.keep_out_zones:
            ax_board.add_patch(mpatches.Rectangle(
                (ko.x, ko.y), ko.width, ko.height,
                ec="#F59E0B", fc="none", lw=1.0, alpha=0.5, linestyle="--"))
                
        # Draw components
        for comp in board.components:
            ax_board.add_patch(mpatches.Rectangle(
                (comp.x, comp.y), comp.width, comp.height,
                fc="#1e2040", ec="#4B5563", lw=1.0, alpha=0.7))
            ax_board.text(comp.x + comp.width/2, comp.y + comp.height/2, comp.name,
                          color="#9CA3AF", fontsize=7, ha='center', va='center')
                          
        # Draw traces
        net_colors = ["#3B82F6","#10B981","#EC4899","#8B5CF6","#06B6D4",
                      "#F59E0B","#14B8A6","#6366F1","#A855F7","#F43F5E"]
        layer_colors = ['#F43F5E', '#06B6D4', '#8B5CF6', '#F59E0B', '#10B981', '#EC4899']
        for seg in state.traces:
            c = layer_colors[seg.layer % len(layer_colors)]
            lw = max(1.2, seg.width / state.resolution)
            ax_board.plot([seg.start_x, seg.end_x], [seg.start_y, seg.end_y],
                          color=c, linewidth=lw, alpha=0.9, solid_capstyle="round")
                          
        # Draw active net path if in autoregressive mode
        is_ar = (getattr(trainer, 'routing_mode', 'astar_guided') == 'autoregressive')
        if is_ar and hasattr(trainer.env, 'current_net_path') and len(trainer.env.current_net_path) > 0:
            active_path = trainer.env.current_net_path
            xs = [p[0] for p in active_path]
            ys = [p[1] for p in active_path]
            ax_board.plot(xs, ys, color='#10B981', linewidth=1.5, alpha=0.95, solid_capstyle="round", zorder=9)
            for wp in active_path:
                wx, wy, wl = wp
                c = layer_colors[wl % len(layer_colors)]
                ax_board.add_patch(mpatches.Circle((wx, wy), radius=0.6, fc=c, ec=WHITE, lw=0.3, alpha=0.9, zorder=9))
            
        # Draw cursor and target if available
        if is_ar and hasattr(trainer.env, 'cursor_pos') and trainer.env.cursor_pos is not None:
            cx, cy, cl = trainer.env.cursor_pos
            ax_board.add_patch(mpatches.Circle(
                (cx, cy), radius=1.8,
                fc="#F59E0B", ec=WHITE, lw=1.0, alpha=1.0, zorder=10))
        if is_ar and hasattr(trainer.env, 'target_pos') and trainer.env.target_pos is not None:
            tx, ty, tl = trainer.env.target_pos
            ax_board.add_patch(mpatches.Circle(
                (tx, ty), radius=1.8,
                fc="#EF4444", ec=WHITE, lw=1.0, alpha=1.0, zorder=10))

        # Draw vias
        for via in state.vias:
            ax_board.add_patch(mpatches.Circle(
                (via.x, via.y), radius=3.5,
                fc="#EAB308", ec=WHITE, lw=0.8, alpha=0.9, zorder=5))
            ax_board.add_patch(mpatches.Circle(
                (via.x, via.y), radius=1.2,
                fc=BG, zorder=6))
                
        # Draw pins
        for pin in board.pins.values():
            c = net_colors[pin.net_id % len(net_colors)]
            ax_board.add_patch(mpatches.Circle(
                (pin.global_x, pin.global_y), radius=2.5,
                fc=c, ec=WHITE, lw=0.6, alpha=0.95, zorder=8))
                
        ax_board.set_xlim(0, board.width)
        ax_board.set_ylim(0, board.height)
        ax_board.set_aspect("equal")
        
        # 2. Draw Layer Heatmaps or Active Traces (in Row 1)
        for l in range(min(2, num_layers)):
            ax_hm = fig.add_subplot(sub_gs[1, l])
            ax_hm.set_facecolor(PANEL)
            if is_ar:
                ax_hm.set_title(f"Layer {l} Active Trace", color=layer_colors[l % len(layer_colors)], fontsize=9, pad=4)
            else:
                ax_hm.set_title(f"Layer {l} Heatmap (AI Cost)", color=layer_colors[l % len(layer_colors)], fontsize=9, pad=4)
                
            ax_hm.set_xticks([]); ax_hm.set_yticks([])
            for spine in ax_hm.spines.values():
                spine.set_color(BORDER)
                
            # Draw obstacles on this layer
            for obs in board.obstacles:
                ax_hm.add_patch(mpatches.Rectangle(
                    (obs.x, obs.y), obs.width, obs.height,
                    fc="#EF4444", alpha=0.1, lw=0, hatch='//'))
            
            if not is_ar and hasattr(trainer, 'last_heatmap') and trainer.last_heatmap is not None:
                ax_hm.imshow(
                    trainer.last_heatmap[l],
                    cmap='magma', origin='lower',
                    extent=(0, board.width, 0, board.height),
                    alpha=0.85
                )
            elif is_ar and hasattr(trainer.env, 'current_net_path') and len(trainer.env.current_net_path) > 0:
                # Draw segments on layer l
                active_path = trainer.env.current_net_path
                for i in range(len(active_path) - 1):
                    p1 = active_path[i]
                    p2 = active_path[i+1]
                    if p1[2] == l or p2[2] == l:
                        c = layer_colors[l % len(layer_colors)]
                        ax_hm.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=2.0, alpha=0.9, marker='o', markersize=2)
                
                # Draw cursor/target if they are on layer l
                if hasattr(trainer.env, 'cursor_pos') and trainer.env.cursor_pos is not None:
                    cx, cy, cl = trainer.env.cursor_pos
                    if cl == l:
                        ax_hm.add_patch(mpatches.Circle((cx, cy), radius=1.5, fc="#F59E0B", ec=WHITE, lw=0.5, alpha=1.0, zorder=10))
                if hasattr(trainer.env, 'target_pos') and trainer.env.target_pos is not None:
                    tx, ty, tl = trainer.env.target_pos
                    if tl == l:
                        ax_hm.add_patch(mpatches.Circle((tx, ty), radius=1.5, fc="#EF4444", ec=WHITE, lw=0.5, alpha=1.0, zorder=10))
            else:
                text_str = "No Active Path" if is_ar else "No Heatmap Yet"
                ax_hm.text(0.5, 0.5, text_str, color="#888",
                           transform=ax_hm.transAxes, ha="center", va="center")
                           
            # Overlay active pins on heatmap/trace for reference
            for pin in board.pins.values():
                if pin.layer == l:
                    c = net_colors[pin.net_id % len(net_colors)]
                    ax_hm.add_patch(mpatches.Circle(
                        (pin.global_x, pin.global_y), radius=2.0,
                        fc=c, ec=WHITE, lw=0.5, alpha=0.9))
                        
            ax_hm.set_xlim(0, board.width)
            ax_hm.set_ylim(0, board.height)
            ax_hm.set_aspect("equal")
    except Exception as e:
        ax_err = fig.add_subplot(gs[:, 2:])
        ax_err.set_facecolor(PANEL)
        ax_err.text(0.5, 0.5, f"Board render error:\\n{e}",
                     transform=ax_err.transAxes, color=WHITE,
                     ha="center", va="center", fontsize=9)
                     
"""
    training_source_str = training_source_str[:try_start_idx] + new_render_code + training_source_str[end_idx:]
else:
    print("Warning: Could not locate board rendering panel for replacement in training cell!")

# Split back into lines
nb["cells"][training_cell_idx]["source"] = [line + "\n" for line in training_source_str.split("\n")]
# remove trailing newline duplicate from split
if nb["cells"][training_cell_idx]["source"][-1] == "\n":
    nb["cells"][training_cell_idx]["source"].pop()

# Cell 6 (Evaluation Cell)
eval_source_str = "".join(nb["cells"][eval_cell_idx]["source"])
target_eval_import = "    from pcb_router.training.trainer import DreamerJEPATrainer"
replacement_eval_import = "    from pcb_router.training.trainer import DreamerJEPATrainer"

target_eval_instantiation = """    eval_trainer = PPOJEPATrainer(
        checkpoint_dir=ckpt_dir,
        load_checkpoint_path=EVAL_CHECKPOINT
    )"""

replacement_eval_instantiation = """    eval_trainer = DreamerJEPATrainer(
        checkpoint_dir=ckpt_dir,
        load_checkpoint_path=EVAL_CHECKPOINT
    )
    is_dreamer = True"""

target_eval_loop = """    print("\\nRunning evaluation episode...")
    while not done and step < 200:
        raster     = torch.tensor(obs["board_raster"], dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        layer_mask = torch.tensor(obs["layer_mask"],   dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        graph      = info["graph"]
        x_dict          = {k: v.to(eval_trainer.device) for k, v in graph.x_dict.items()}         if hasattr(graph, "x_dict")         else {}
        edge_index_dict = {k: v.to(eval_trainer.device) for k, v in graph.edge_index_dict.items()} if hasattr(graph, "edge_index_dict") else {}

        with torch.no_grad():
            sp, cls = eval_trainer.vit(raster)
            ne       = eval_trainer.gnn(x_dict, edge_index_dict)
            fp, fs   = eval_trainer.fusion(ne["pad"].unsqueeze(0), sp)
            num_nets  = len(eval_trainer.env.board.nets)
            net_embs  = torch.zeros((1, 100, eval_trainer.vit.embed_dim), device=eval_trainer.device)
            umask     = torch.zeros((1, 100), dtype=torch.bool, device=eval_trainer.device)
            for ni, net in enumerate(eval_trainer.env.board.nets):
                pidx = [pi for pi, p in enumerate(eval_trainer.env.board.pins.values()) if p.net_id == net.id]
                if pidx: net_embs[0, ni] = fp[0, pidx].mean(0)
                if net.id not in eval_trainer.env.routed_nets: umask[0, ni] = True
            net_idx, hlat, _, _, _ = eval_trainer.policy(net_embs, umask, fs, cls, deterministic=True)
            hv = eval_trainer.decoder(hlat, fs, eval_trainer.env.H, eval_trainer.env.W, active_layers_mask=layer_mask)"""

replacement_eval_loop = """    h, z = eval_trainer.jepa.initial_state(batch_size=1, device=eval_trainer.device)
    num_nets = len(eval_trainer.env.board.nets)
        
    print("\\nRunning evaluation episode...")
    while not done and step < 200:
        raster     = torch.tensor(obs["board_raster"], dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        layer_mask = torch.tensor(obs["layer_mask"],   dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        graph      = info["graph"]
        x_dict          = {k: v.to(eval_trainer.device) for k, v in graph.x_dict.items()}         if hasattr(graph, "x_dict")         else {}
        edge_index_dict = {k: v.to(eval_trainer.device) for k, v in graph.edge_index_dict.items()} if hasattr(graph, "edge_index_dict") else {}

        with torch.no_grad():
            context_emb = eval_trainer.jepa.get_context_embedding(raster, x_dict, edge_index_dict, use_target=False)
            net_embs, umask, fs = eval_trainer._get_net_embeddings_and_mask(raster, x_dict, edge_index_dict)
            
            net_idx, heatmap_latent, _, _ = eval_trainer.policy.act(net_embs, umask, h, z, explore=False)
            
            action_emb = eval_trainer.jepa.get_action_embedding(net_idx, heatmap_latent)
            h, z, _, _ = eval_trainer.jepa.rssm_step(h, z, context_emb, action_emb)
            
            hv = eval_trainer.decoder(heatmap_latent, fs, eval_trainer.env.H, eval_trainer.env.W, active_layers_mask=layer_mask)"""

# Apply evaluation updates
if target_eval_import in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_import, replacement_eval_import)
if target_eval_instantiation in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_instantiation, replacement_eval_instantiation)
if target_eval_loop in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_loop, replacement_eval_loop)

nb["cells"][eval_cell_idx]["source"] = [line + "\n" for line in eval_source_str.split("\n")]
if nb["cells"][eval_cell_idx]["source"][-1] == "\n":
    nb["cells"][eval_cell_idx]["source"].pop()

# ─────────────────────────────────────────────────────────────────────────────
#  Cell 7 (NEW) — Step-by-Step Routing Visualizer (Gradio OR W&B)
# ─────────────────────────────────────────────────────────────────────────────
viz_cell_source = [
    "# ================================================================\n",
    "#  CELL 7 — JEPA STEP-BY-STEP ROUTING VISUALIZER\n",
    "#\n",
    "#  Choose your visualizer:\n",
    "#\n",
    "#   VIZ_MODE = 'gradio'   →  Full interactive UI with a public URL\n",
    "#                             (works in Colab — no port forwarding needed)\n",
    "#\n",
    "#   VIZ_MODE = 'wandb'    →  Logs every routing step as images to W&B\n",
    "#                             (persistent, shareable, viewable anywhere)\n",
    "#\n",
    "#  Each step shows:\n",
    "#    • Board BEFORE routing (existing copper visible)\n",
    "#    • JEPA cost heatmap (white overlay = existing copper the model sees)\n",
    "#    • A* occupancy map  (what the pathfinder sees as blocked)\n",
    "#    • Board AFTER routing\n",
    "#    • Via placement probability map\n",
    "# ================================================================\n",
    "\n",
    "# ── Config ────────────────────────────────────────────────────────\n",
    "VIZ_MODE        = 'gradio'                              # 'gradio' or 'wandb'\n",
    "VIZ_CHECKPOINT  = CONFIG.get('LOAD_CHECKPOINT', None)  # None = untrained model\n",
    "VIZ_STAGE       = 'multi_net_single_layer'             # curriculum stage\n",
    "VIZ_SEED        = 42                                   # board seed\n",
    "VIZ_NUM_BOARDS  = 3                                    # W&B only: boards to route\n",
    "VIZ_WANDB_PROJ  = CONFIG.get('WANDB_PROJECT', 'pcb-router')\n",
    "\n",
    "# ── Launch ────────────────────────────────────────────────────────\n",
    "import sys, os\n",
    "sys.path.insert(0, CONFIG['REPO_DIR'])\n",
    "os.chdir(CONFIG['REPO_DIR'])\n",
    "\n",
    "if VIZ_MODE == 'gradio':\n",
    "    # ── Gradio: interactive UI with a public *.gradio.live URL ──────\n",
    "    try:\n",
    "        import gradio\n",
    "    except ImportError:\n",
    "        import subprocess\n",
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=True)\n",
    "\n",
    "    from scripts.visualize_routing_gradio import launch_gradio_visualizer\n",
    "\n",
    "    print('Launching Gradio step-by-step routing visualizer...')\n",
    "    print('A public URL will appear below. Click it to open the UI.')\n",
    "    launch_gradio_visualizer(\n",
    "        checkpoint_path=VIZ_CHECKPOINT,\n",
    "        stage=VIZ_STAGE,\n",
    "        share=True,             # ← generates public *.gradio.live URL\n",
    "    )\n",
    "\n",
    "elif VIZ_MODE == 'wandb':\n",
    "    # ── W&B: log all steps as images to the W&B cloud dashboard ────\n",
    "    import argparse\n",
    "    from scripts.visualize_routing_wandb import run_visualization\n",
    "\n",
    "    args = argparse.Namespace(\n",
    "        checkpoint=str(VIZ_CHECKPOINT) if VIZ_CHECKPOINT else 'none',\n",
    "        stage=VIZ_STAGE,\n",
    "        seed=VIZ_SEED,\n",
    "        num_boards=VIZ_NUM_BOARDS,\n",
    "        wandb_project=VIZ_WANDB_PROJ,\n",
    "        wandb_run_name=None,\n",
    "    )\n",
    "    print(f'Logging routing steps to W&B project: {VIZ_WANDB_PROJ}')\n",
    "    run_visualization(args)\n",
    "\n",
    "else:\n",
    "    raise ValueError(f\"Unknown VIZ_MODE '{VIZ_MODE}'. Use 'gradio' or 'wandb'.\")\n",
]

# Find or insert the step-by-step visualizer cell dynamically to avoid overwriting other cells.
viz_cell_idx = None
checkpoint_mgmt_idx = None

for idx, cell in enumerate(nb["cells"]):
    cell_src = "".join(cell.get("source", []))
    if "JEPA STEP-BY-STEP ROUTING VISUALIZER" in cell_src:
        viz_cell_idx = idx
        break
    if "CHECKPOINT MANAGEMENT" in cell_src:
        checkpoint_mgmt_idx = idx

if viz_cell_idx is not None:
    nb["cells"][viz_cell_idx]["source"] = viz_cell_source
    print(f"Updated existing step-by-step routing visualizer cell at index {viz_cell_idx}")
else:
    viz_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_cell_source,
    }
    if checkpoint_mgmt_idx is not None:
        nb["cells"].insert(checkpoint_mgmt_idx, viz_cell)
        print(f"Inserted new step-by-step routing visualizer cell at index {checkpoint_mgmt_idx}")
    else:
        nb["cells"].append(viz_cell)
        print("Appended new step-by-step routing visualizer cell to the end")

# Find or insert the BC pretraining cell dynamically
bc_cell_idx = None
init_trainer_idx = None

for idx, cell in enumerate(nb["cells"]):
    cell_src = "".join(cell.get("source", []))
    if "CELL 4b — OPTIONAL: BEHAVIOR CLONING" in cell_src:
        bc_cell_idx = idx
        break
    if "CELL 5 — INITIALIZE TRAINER" in cell_src:
        init_trainer_idx = idx

bc_cell_source = [
    "# ================================================================\n",
    "#  CELL 4b — OPTIONAL: BEHAVIOR CLONING (BC) PRETRAINING\n",
    "#  Generates expert trajectories using A* and pretrains the\n",
    "#  RouteStepPolicy before running RL fine-tuning.\n",
    "# ================================================================\n",
    "\n",
    "# 1. Generate the dataset for stages s00-s06\n",
    "!python -u scripts/generate_bc_dataset.py\n",
    "\n",
    "# 2. Run supervised BC pretraining on the policy\n",
    "!python -u scripts/train_bc_policy.py --epochs 20 --batch_size 128\n"
]

if bc_cell_idx is not None:
    nb["cells"][bc_cell_idx]["source"] = bc_cell_source
    print(f"Updated existing BC pretraining cell at index {bc_cell_idx}")
else:
    bc_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": bc_cell_source,
    }
    if init_trainer_idx is not None:
        nb["cells"].insert(init_trainer_idx, bc_cell)
        print(f"Inserted new BC pretraining cell at index {init_trainer_idx}")
    else:
        nb["cells"].append(bc_cell)
        print("Appended new BC pretraining cell to the end")

# Save notebook back to file
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Jupyter Notebook Train_PCB_Router.ipynb successfully updated!")

