# DreamerJEPA PCB Router — Standalone Training Script
# Mirrors the Colab notebook (Train_PCB_Router.ipynb) but runs from the CLI.
# Usage:
#   python notebooks/03_training.py
#   python notebooks/03_training.py --checkpoint-dir /tmp/ckpts --load-checkpoint /tmp/ckpts/checkpoint_50000.pt

# %% Cell 1: Imports and logging setup
import torch
import numpy as np
import os
import yaml
from pcb_router.training.trainer import DreamerJEPATrainer  # ← latest model

import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="DreamerJEPA PCB Router Training")
parser.add_argument('--checkpoint-dir', type=str, default=None,
                    help='Directory to save checkpoints')
parser.add_argument('--load-checkpoint', type=str, default=None,
                    help='Path to load an existing checkpoint')
parser.add_argument('--total-timesteps', type=int, default=1_000_000,
                    help='Total environment timesteps to train for')
# parse_known_args avoids crashing if executed inside interactive notebook cells
args, unknown = parser.parse_known_args()

# Set up logging directories
checkpoint_save_dir = args.checkpoint_dir if args.checkpoint_dir else 'checkpoints'
os.makedirs(checkpoint_save_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# %% Cell 2: Initialize DreamerJEPA trainer
print("Initializing DreamerJEPATrainer...")
trainer = DreamerJEPATrainer(
    config_path='configs/training.yaml',
    model_config_path='configs/model.yaml',
    curriculum_config_path='configs/curriculum.yaml',
    device='auto',
    checkpoint_dir=checkpoint_save_dir,
    load_checkpoint_path=args.load_checkpoint,
)

# Display active curriculum details
progress = trainer.curriculum.get_progress_summary()
print(f"\nCurriculum loaded at stage {progress['stage_idx']}: {progress['stage_name']}")
print(f"Board size: {trainer.env.board_config.board_width}x{trainer.env.board_config.board_height}")
print(f"Target completion rate to advance: {progress['target_completion'] * 100:.0f}%")
print(f"Timesteps so far: {trainer.total_timesteps:,}")

# %% Cell 3: Run quick training smoke test (one iteration of all 3 phases)
print("\nRunning training smoke test (1 DreamerJEPA iteration)...")

print("  Phase 1 — collecting real environment transitions...")
mean_completion = trainer._phase1_collect_real(num_steps=trainer.real_steps_per_iteration, explore=True)
print(f"  Phase 1 done. Mean completion: {mean_completion:.3f}")

print("  Phase 2 — training world model...")
wm_metrics = trainer._phase2_train_world_model()
print("  World-model losses:")
for k, v in wm_metrics.items():
    print(f"    {k}: {v:.5f}")

print("  Phase 3 — training actor-critic in imagination...")
ac_metrics = trainer._phase3_train_actor_critic()
print("  Actor-critic losses:")
for k, v in ac_metrics.items():
    print(f"    {k}: {v:.5f}")

print("\nSmoke test passed!")

# %% Cell 4: Launch full training loop
print(f"\nStarting full DreamerJEPA training loop ({args.total_timesteps:,} timesteps)...")
print(f"Checkpoints -> {checkpoint_save_dir}")
print("-" * 60)
try:
    trainer.train(total_timesteps=args.total_timesteps)
    print("Training complete!")
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving emergency checkpoint...")
    ep = os.path.join(checkpoint_save_dir, f"interrupted_{trainer.total_timesteps}.pt")
    trainer.save_checkpoint(ep)
    print(f"Saved to: {ep}")
