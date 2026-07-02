import os
import sys
import torch
import yaml
import copy

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pcb_router.training.trainer import DreamerJEPATrainer
from pcb_router.training.replay_buffer import Episode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running gradient isolation test on: {device}")
    
    # 1. Initialize trainer
    trainer = DreamerJEPATrainer(
        config_path='configs/training.yaml',
        model_config_path='configs/model.yaml',
        curriculum_config_path='configs/curriculum.yaml',
        device='auto'
    )
    
    # 2. Seed replay buffer with a dummy episode
    episode = Episode()
    
    # Create mock context embeddings (T=20, dim=768)
    for _ in range(20):
        ctx = torch.randn(768)
        tgt_ctx = torch.randn(768)
        # Action is net_idx (long), heatmap_latent (256)
        action = (torch.tensor(0, dtype=torch.long), torch.randn(256))
        reward = 0.5
        done = False
        
        if not hasattr(episode, 'target_context_embeddings'):
            episode.target_context_embeddings = []
        if not hasattr(episode, 'net_embeddings_list'):
            episode.net_embeddings_list = []
        if not hasattr(episode, 'unrouted_masks_list'):
            episode.unrouted_masks_list = []
            
        episode.append(ctx, action, reward, done)
        episode.target_context_embeddings.append(tgt_ctx)
        
        # Net embeddings (1, 100, 384), unrouted_mask (1, 100)
        episode.net_embeddings_list.append(torch.randn(100, 384))
        episode.unrouted_masks_list.append(torch.ones(100, dtype=torch.bool))
        
    episode.net_embeddings = episode.net_embeddings_list[0]
    episode.unrouted_masks = episode.unrouted_masks_list
    
    trainer.replay_buffer.add_episode(episode)
    
    # Cache dummy latents so the actor-critic can sample them
    h_cache = torch.randn(10, 512)
    z_cache = torch.randn(10, 1024)
    trainer.replay_buffer.cache_latents(h_cache, z_cache)
    
    # 3. Record all world model parameter values before Phase 3
    wm_params_before = {}
    for name, param in trainer.jepa.named_parameters():
        wm_params_before[name] = param.clone().detach()
        
    # Keep copies of vit, gnn, fusion params too
    vit_params_before = {name: param.clone().detach() for name, param in trainer.vit.named_parameters()}
    gnn_params_before = {name: param.clone().detach() for name, param in trainer.gnn.named_parameters()}
    fusion_params_before = {name: param.clone().detach() for name, param in trainer.fusion.named_parameters()}

    print("Running Phase 3 (Actor-Critic imagined updates)...")
    # Set train_ratio=2 for quick test
    trainer.train_ratio = 2
    trainer.imagine_batch_size = 4
    trainer.imagination_horizon = 5
    
    trainer._phase3_train_actor_critic()
    
    print("Verifying gradient isolation...")
    
    # 4. Assert all parameters are bit-identical
    mismatches = 0
    
    for name, param in trainer.jepa.named_parameters():
        before = wm_params_before[name]
        if not torch.allclose(before, param):
            print(f"Mismatch in world model parameter: {name}")
            mismatches += 1
            
    for name, param in trainer.vit.named_parameters():
        before = vit_params_before[name]
        if not torch.allclose(before, param):
            print(f"Mismatch in vit parameter: {name}")
            mismatches += 1
            
    for name, param in trainer.gnn.named_parameters():
        before = gnn_params_before[name]
        if not torch.allclose(before, param):
            print(f"Mismatch in gnn parameter: {name}")
            mismatches += 1
            
    for name, param in trainer.fusion.named_parameters():
        before = fusion_params_before[name]
        if not torch.allclose(before, param):
            print(f"Mismatch in fusion parameter: {name}")
            mismatches += 1
            
    if mismatches == 0:
        print("SUCCESS: World model, ViT, GNN, and Fusion parameters are 100% bit-identical after Phase 3 updates!")
    else:
        print(f"FAILURE: Detected {mismatches} parameter mismatches. Gradient leakage occurred!")
        sys.exit(1)

if __name__ == "__main__":
    main()
