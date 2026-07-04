import json
import os

notebook_path = "notebooks/Train_PCB_Router.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1 (CONFIG cell)
# Find the config cell (Cell 1) and update CONFIG dictionary
config_source = nb["cells"][1]["source"]
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
    "}\n",
    "\n",
    "print(\"Config loaded:\")\n",
    "for k, v in CONFIG.items():\n",
    "    print(f\"  {k}: {v}\")\n"
]
nb["cells"][1]["source"] = new_config_source

# Cell 4 (Initialize Trainer)
new_init_source = [
    "# ================================================================\n",
    "#  CELL 5 — INITIALIZE TRAINER\n",
    "# ================================================================\n",
    "import os, sys\n",
    "\n",
    "# ── Clear Python import cache to force reloading from disk ─────\n",
    "for key in list(sys.modules.keys()):\n",
    "    if key == \"pcb_router\" or key.startswith(\"pcb_router.\"):\n",
    "        del sys.modules[key]\n",
    "\n",
    "sys.path.insert(0, CONFIG[\"REPO_DIR\"])\n",
    "os.chdir(CONFIG[\"REPO_DIR\"])\n",
    "\n",
    "from pcb_router.training.trainer import PPOJEPATrainer, DreamerJEPATrainer\n",
    "\n",
    "# Initialize the new DreamerJEPATrainer by default\n",
    "trainer = DreamerJEPATrainer(\n",
    "    config_path=\"configs/training.yaml\",\n",
    "    model_config_path=\"configs/model.yaml\",\n",
    "    curriculum_config_path=\"configs/curriculum.yaml\",\n",
    "    device=\"auto\",                             # auto-selects GPU if available\n",
    "    checkpoint_dir=CONFIG[\"CHECKPOINT_DIR\"],\n",
    "    load_checkpoint_path=CONFIG[\"LOAD_CHECKPOINT\"],\n",
    ")\n",
    "\n",
    "# Apply CONFIG overrides dynamically depending on trainer type\n",
    "if isinstance(trainer, DreamerJEPATrainer):\n",
    "    trainer.real_steps_per_iteration = CONFIG.get(\"REAL_STEPS_PER_ITERATION\", 64)\n",
    "    trainer.train_ratio = CONFIG.get(\"TRAIN_RATIO\", 100)\n",
    "    trainer.replay_buffer.capacity_episodes = CONFIG.get(\"REPLAY_BUFFER_SIZE\", 5000)\n",
    "    trainer.imagine_batch_size = CONFIG.get(\"IMAGINE_BATCH_SIZE\", 512)\n",
    "    trainer.imagination_horizon_start = CONFIG.get(\"IMAGINATION_HORIZON_START\", 5)\n",
    "    trainer.imagination_horizon_end = CONFIG.get(\"IMAGINATION_HORIZON_END\", 15)\n",
    "    trainer.gamma = CONFIG.get(\"GAMMA\", 0.997)\n",
    "    trainer.lambda_ = CONFIG.get(\"LAMBDA\", 0.95)\n",
    "    trainer.compile_models = CONFIG.get(\"COMPILE_MODELS\", True)\n",
    "    \n",
    "    for pg in trainer.wm_opt.param_groups:\n",
    "        pg[\"lr\"] = CONFIG.get(\"WORLD_MODEL_LR\", 3e-4)\n",
    "    for pg in trainer.actor_opt.param_groups:\n",
    "        pg[\"lr\"] = CONFIG.get(\"ACTOR_LR\", 8e-5)\n",
    "    for pg in trainer.critic_opt.param_groups:\n",
    "        pg[\"lr\"] = CONFIG.get(\"CRITIC_LR\", 8e-5)\n",
    "else:\n",
    "    trainer.train_cfg[\"ppo\"][\"batch_size\"]         = CONFIG.get(\"BATCH_SIZE\", 16)\n",
    "    trainer.train_cfg[\"ppo\"][\"num_rollout_steps\"]  = CONFIG.get(\"NUM_ROLLOUT_STEPS\", 64)\n",
    "    trainer.train_cfg[\"ppo\"][\"num_epochs\"]         = CONFIG.get(\"NUM_EPOCHS\", 4)\n",
    "    trainer.train_cfg[\"ppo\"][\"learning_rate\"]      = CONFIG.get(\"LEARNING_RATE\", 3e-4)\n",
    "    trainer.train_cfg[\"ppo\"][\"gamma\"]              = CONFIG.get(\"GAMMA\", 0.99)\n",
    "    trainer.train_cfg[\"ppo\"][\"gae_lambda\"]         = CONFIG.get(\"GAE_LAMBDA\", 0.95)\n",
    "    for pg in trainer.optimizer.param_groups:\n",
    "        pg[\"lr\"] = CONFIG.get(\"LEARNING_RATE\", 3e-4)\n",
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
nb["cells"][4]["source"] = new_init_source

# Cell 5 (Training Cell)
training_source_str = "".join(nb["cells"][5]["source"])
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
            
            # Periodically log step-by-step routing viz of the whole model (every 20 updates)
            if _update_count[0] % 20 == 0:
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

# Split back into lines
nb["cells"][5]["source"] = [line + "\n" for line in training_source_str.split("\n")]
# remove trailing newline duplicate from split
if nb["cells"][5]["source"][-1] == "\n":
    nb["cells"][5]["source"].pop()

# Cell 6 (Evaluation Cell)
eval_source_str = "".join(nb["cells"][6]["source"])
target_eval_import = "    from pcb_router.training.trainer import PPOJEPATrainer"
replacement_eval_import = "    from pcb_router.training.trainer import PPOJEPATrainer, DreamerJEPATrainer"

target_eval_instantiation = """    eval_trainer = PPOJEPATrainer(
        checkpoint_dir=ckpt_dir,
        load_checkpoint_path=EVAL_CHECKPOINT
    )"""

replacement_eval_instantiation = """    # Load using DreamerJEPATrainer or fallback
    try:
        eval_trainer = DreamerJEPATrainer(
            checkpoint_dir=ckpt_dir,
            load_checkpoint_path=EVAL_CHECKPOINT
        )
        is_dreamer = True
    except Exception as e:
        print(f"Dreamer load failed, falling back to PPO: {e}")
        eval_trainer = PPOJEPATrainer(
            checkpoint_dir=ckpt_dir,
            load_checkpoint_path=EVAL_CHECKPOINT
        )
        is_dreamer = False"""

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

replacement_eval_loop = """    if is_dreamer:
        h, z = eval_trainer.jepa.initial_state(batch_size=1, device=eval_trainer.device)
        num_nets = len(eval_trainer.env.board.nets)
        
    print("\\nRunning evaluation episode...")
    while not done and step < 200:
        raster     = torch.tensor(obs["board_raster"], dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        layer_mask = torch.tensor(obs["layer_mask"],   dtype=torch.float32).unsqueeze(0).to(eval_trainer.device)
        graph      = info["graph"]
        x_dict          = {k: v.to(eval_trainer.device) for k, v in graph.x_dict.items()}         if hasattr(graph, "x_dict")         else {}
        edge_index_dict = {k: v.to(eval_trainer.device) for k, v in graph.edge_index_dict.items()} if hasattr(graph, "edge_index_dict") else {}

        with torch.no_grad():
            if is_dreamer:
                context_emb = eval_trainer.jepa.get_context_embedding(raster, x_dict, edge_index_dict, use_target=False)
                net_embs, umask, fs = eval_trainer._get_net_embeddings_and_mask(raster, x_dict, edge_index_dict)
                
                net_idx, heatmap_latent, _, _ = eval_trainer.policy.act(net_embs, umask, h, z, explore=False)
                
                action_emb = eval_trainer.jepa.get_action_embedding(net_idx, heatmap_latent)
                h, z, _, _ = eval_trainer.jepa.rssm_step(h, z, context_emb, action_emb)
                
                hv = eval_trainer.decoder(heatmap_latent, fs, eval_trainer.env.H, eval_trainer.env.W, active_layers_mask=layer_mask)
            else:
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

# Apply evaluation updates
if target_eval_import in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_import, replacement_eval_import)
if target_eval_instantiation in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_instantiation, replacement_eval_instantiation)
if target_eval_loop in eval_source_str:
    eval_source_str = eval_source_str.replace(target_eval_loop, replacement_eval_loop)

nb["cells"][6]["source"] = [line + "\n" for line in eval_source_str.split("\n")]
if nb["cells"][6]["source"][-1] == "\n":
    nb["cells"][6]["source"].pop()

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

# Save notebook back to file
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Jupyter Notebook Train_PCB_Router.ipynb successfully updated!")

