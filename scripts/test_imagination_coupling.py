import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pcb_router.training.trainer import DreamerJEPATrainer
from pcb_router.training.replay_buffer import Episode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running imagination coupling test on: {device}")
    
    bc_path = 'checkpoints/bc_pretrained_policy.pt'
    created_bc = False
    
    # 1. Create a dummy file to satisfy os.path.exists check
    if not os.path.exists(bc_path):
        os.makedirs('checkpoints', exist_ok=True)
        with open(bc_path, 'w') as f:
            f.write('')
        created_bc = True
        
    # 2. Monkeypatch torch.load and load_state_dict to make loading a safe no-op
    original_load = torch.load
    original_load_state_dict = nn.Module.load_state_dict
    
    torch.load = lambda *args, **kwargs: {}
    nn.Module.load_state_dict = lambda self, state_dict, strict=True: ([], [])
    
    try:
        # 3. Initialize trainer in autoregressive mode
        trainer = DreamerJEPATrainer(
            config_path='configs/training.yaml',
            model_config_path='configs/model.yaml',
            curriculum_config_path='configs/curriculum.yaml',
            device='auto'
        )
        trainer.routing_mode = 'autoregressive'
        
        # Restore mock objects
        torch.load = original_load
        nn.Module.load_state_dict = original_load_state_dict
        
        # 4. Seed replay buffer with a dummy episode
        episode = Episode()
        episode.length = 20
        episode._finalized = True
        
        crop_size = trainer.policy.step_policy.crop_size
        embed_dim = trainer.policy.step_policy.embed_dim
        crop_dim = crop_size * crop_size * embed_dim
        
        episode.cropped_spatials_tensor = torch.zeros(20, crop_dim)
        episode.cursor_poses_tensor = torch.zeros(20, 3)
        episode.target_poses_tensor = torch.zeros(20, 3)
        episode.moves_remaining_fracs_tensor = torch.zeros(20, 1)
        
        # New required fields for dynamic updates
        episode.max_moves_fracs_tensor = torch.ones(20) * 0.05
        episode.fused_spatials_tensor = torch.zeros(20, 16, embed_dim) # N_patches = 16
        episode.board_dims = (20, 20, 2)
        
        # Fill in cursor poses with distinct coordinates
        for t in range(20):
            episode.cursor_poses_tensor[t] = torch.tensor([t / 20.0, t / 20.0, 0.0])
            
        episode.net_embeddings = torch.zeros(100, embed_dim)
        episode.unrouted_masks = torch.ones(20, 100, dtype=torch.bool)
        
        # Add to buffer
        trainer.replay_buffer.add_episode(episode)
        
        # 5. Setup rollout inputs
        h0, z0 = trainer.jepa.initial_state(batch_size=1, device=trainer.device)
        
        # 6. Rollout twice with different actions at t=0
        original_sample = torch.distributions.Categorical.sample
        
        # Run 1: force action 2 (Move Right)
        torch.distributions.Categorical.sample = lambda self: torch.tensor([2], device=trainer.device)
        rollout1 = trainer._imagine_autoregressive_rollout(h0, z0, [episode], [0], current_horizon=2)
        
        # Run 2: force action 3 (Move Left)
        torch.distributions.Categorical.sample = lambda self: torch.tensor([3], device=trainer.device)
        rollout2 = trainer._imagine_autoregressive_rollout(h0, z0, [episode], [0], current_horizon=2)
        
        # Restore Categorical.sample
        torch.distributions.Categorical.sample = original_sample
        
        # 7. Assert coupling
        # In a coupled setup, action 2 vs action 3 must result in different cursor_poses at t=1.
        # In decoupled code, they will be identical because they are read from the same cached slice index 1.
        cursor1_t1 = rollout1['traj_cursor_poses'][1]
        cursor2_t1 = rollout2['traj_cursor_poses'][1]
        
        print(f"Run 1 cursor at t=1: {cursor1_t1.cpu().numpy()}")
        print(f"Run 2 cursor at t=1: {cursor2_t1.cpu().numpy()}")
        
        if torch.allclose(cursor1_t1, cursor2_t1):
            print("BUG CONFIRMED: cursor at t=1 is identical regardless of action taken at t=0 (decoupled).")
            sys.exit(1)
        else:
            print("SUCCESS: cursor at t=1 is coupled to the action taken at t=0!")
            sys.exit(0)
            
    finally:
        # Restore in case of exception before restoration
        torch.load = original_load
        nn.Module.load_state_dict = original_load_state_dict
        
        # Clean up dummy BC checkpoint if we created it
        if created_bc and os.path.exists(bc_path):
            os.remove(bc_path)

if __name__ == "__main__":
    main()
