# PCB Router — AI-Powered PCB Routing

An AI agent that learns to route PCB traces using a **PPO reinforcement learning** loop with a **Spatial JEPA world model** for self-supervised representation learning.

## Architecture

```
Board Raster (13ch) ──► ViT-Small Encoder ──┐
                                             ├──► Cross-Attention Fusion ──► PPO Policy ──► Net Selection + Heatmap
Graph (Pads/Nets) ───► Hetero-GAT Encoder ──┘                                          └──► Heatmap Decoder ──► A* Router
                                                                                        
Spatial JEPA: ViT Online ──► Predictor ──► Predict z_{t+1}  (EMA Target Encoder)
```

## Quick Start

### Local
```bash
git clone <repo-url>
cd Router
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
pip install -e .

python -c "
from pcb_router.training.trainer import PPOJEPATrainer
trainer = PPOJEPATrainer()
trainer.train(total_timesteps=5_000_000)
"
```

### Google Colab
Open `notebooks/Train_PCB_Router.ipynb` in Colab. Set your config in Cell 2, mount Drive in Cell 3, then run all cells.

## Configuration

| File | Purpose |
|------|---------|
| `configs/training.yaml` | PPO hyperparameters, batch size, rollout steps |
| `configs/model.yaml` | ViT, GNN, JEPA, policy architecture dims |
| `configs/curriculum.yaml` | Board complexity curriculum stages |

Key settings in `configs/training.yaml`:
- `batch_size`: Reduce if you get OOM (default: 16)
- `num_rollout_steps`: Steps before each PPO update (default: 64)
- `num_epochs`: PPO update epochs (default: 4)

Key settings in `configs/model.yaml`:
- `max_grid_size`: Max board resolution in pixels (default: 256 — do **not** increase past 512 without a large GPU)

## Project Structure

```
pcb_router/
├── models/
│   ├── vit_encoder.py      # ViT-Small spatial encoder
│   ├── gnn_encoder.py      # Heterogeneous GAT graph encoder
│   ├── fusion.py           # Cross-attention fusion
│   ├── jepa.py             # Spatial JEPA world model
│   ├── policy.py           # PPO actor-critic policy
│   └── heatmap_decoder.py  # CNN heatmap decoder
├── env/
│   ├── pcb_env.py          # Gymnasium environment
│   ├── board_state.py      # Board state raster
│   └── drc_checker.py      # DRC violation checker
├── training/
│   ├── trainer.py          # PPOJEPATrainer main loop
│   ├── curriculum.py       # Difficulty curriculum
│   └── rewards.py          # Reward shaping
├── visualization/
│   ├── renderer.py         # Board renderer
│   └── heatmap_viz.py      # Heatmap + training dashboard
configs/
notebooks/
│   └── Train_PCB_Router.ipynb  # Colab training notebook
```

## Hardware Requirements

| Hardware | Status | Notes |
|----------|--------|-------|
| CPU only | Works | ~10-30s/step, fine for debugging |
| GPU (8GB VRAM) | Recommended | ~0.5-2s/step |
| GPU (16GB+ VRAM) | Ideal | Can increase batch_size and max_grid_size |

## Checkpointing

Checkpoints are saved to `configs/training.yaml → checkpoint.save_dir` (default: `checkpoints/`), or override at runtime:

```python
trainer = PPOJEPATrainer(checkpoint_dir='/path/to/save', load_checkpoint_path='/path/to/resume.pt')
```
