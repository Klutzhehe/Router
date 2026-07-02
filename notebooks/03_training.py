# Google Colab Interactive Notebook Cell Code - PPO + JEPA Training Run
# Copy and run these cells in your Jupyter/Colab notebook.

# %% Cell 1: Imports and logging setup
import torch
import numpy as np
import os
import yaml
from pcb_router.training.trainer import PPOJEPATrainer

import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="PPO + JEPA PCB Router Training")
parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to save checkpoints')
parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to load an existing checkpoint')
# parse_known_args avoids crashing if executed directly in interactive notebook cells
args, unknown = parser.parse_known_args()

# Set up logging directories
checkpoint_save_dir = args.checkpoint_dir if args.checkpoint_dir else 'checkpoints'
os.makedirs(checkpoint_save_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# %% Cell 2: Initialize trainer
print("Initializing trainer...")
# Initialize with CLI arguments
trainer = PPOJEPATrainer(
    device='auto',
    checkpoint_dir=args.checkpoint_dir,
    load_checkpoint_path=args.load_checkpoint
)

# Display active curriculum details
progress = trainer.curriculum.get_progress_summary()
print(f"\nCurriculum loaded at stage {progress['stage_idx']}: {progress['stage_name']}")
print(f"Board sizes generated at this stage: {trainer.env.board_config.board_width}x{trainer.env.board_config.board_height}")
print(f"Target completion rate to advance: {progress['target_completion'] * 100}%")

# %% Cell 3: Run quick training smoke test
# We run 2 rollout collections and updates to verify weight changes
print("\nRunning training smoke test...")
print("Collecting rollout transitions...")
eval_completion = trainer.collect_rollouts(num_steps=64)

print(f"Rollout collection finished. Mean completion: {eval_completion:.2f}")
print("Updating policy and JEPA world-model weights...")
losses = trainer.update()

print("Losses after update:")
for k, v in losses.items():
    print(f" - {k}: {v:.5f}")

# %% Cell 4: Launch full training loop
# You can set this running in Google Colab
print("\nStarting full training loop (1,000,000 steps)...")
try:
    trainer.train(total_timesteps=1000000)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving checkpoint...")
    trainer.save_checkpoint("checkpoints/interrupted_checkpoint.pt")
