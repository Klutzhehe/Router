import os
import sys
import torch
import torch.nn as nn
import yaml
import copy

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pcb_router.training.trainer import DreamerJEPATrainer
from pcb_router.training.replay_buffer import Episode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running gradient isolation test on: {device}")
    
    bc_path = 'checkpoints/bc_pretrained_policy.pt'
    created_bc = False
    if not os.path.exists(bc_path):
        os.makedirs('checkpoints', exist_ok=True)
        with open(bc_path, 'w') as f:
            f.write('')
        created_bc = True
        
    original_load = torch.load
    original_load_state_dict = nn.Module.load_state_dict
    
    torch.load = lambda *args, **kwargs: {}
    nn.Module.load_state_dict = lambda self, state_dict, strict=True: ([], [])
    
    try:
        # 1. Initialize trainer
        trainer = DreamerJEPATrainer(
            config_path='configs/training.yaml',
            model_config_path='configs/model.yaml',
            curriculum_config_path='configs/curriculum.yaml',
            device='auto'
        )
        trainer.routing_mode = 'heatmap'
        
        # Restore mock objects for the rest of the test
        torch.load = original_load
        nn.Module.load_state_dict = original_load_state_dict
        
        # Query dimensions dynamically
        embed_dim = trainer.policy.embed_dim
        h_dim = trainer.policy.h_dim
        z_dim = trainer.policy.z_dim
        heatmap_latent_dim = trainer.policy.heatmap_latent_dim
        ctx_dim = embed_dim * 2 # Fusion outputs global_spatial + global_graph, each is embed_dim
        
        # 2. Seed replay buffer with a dummy episode
        episode = Episode()
        
        # Create mock context embeddings (T=20, dim=ctx_dim)
        for _ in range(20):
            ctx = torch.randn(ctx_dim)
            tgt_ctx = torch.randn(ctx_dim)
            # Action is net_idx (long), heatmap_latent (heatmap_latent_dim)
            action = (torch.tensor(0, dtype=torch.long), torch.randn(heatmap_latent_dim))
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
            
            # Net embeddings (1, 100, embed_dim), unrouted_mask (1, 100)
            episode.net_embeddings_list.append(torch.randn(100, embed_dim))
            episode.unrouted_masks_list.append(torch.ones(100, dtype=torch.bool))
            
        episode.net_embeddings = episode.net_embeddings_list[0]
        episode.unrouted_masks = episode.unrouted_masks_list
        
        trainer.replay_buffer.add_episode(episode)
        
        # Cache dummy latents so the actor-critic can sample them
        h_cache = torch.randn(10, h_dim)
        z_cache = torch.randn(10, z_dim)
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
            
    finally:
        torch.load = original_load
        nn.Module.load_state_dict = original_load_state_dict
        if created_bc and os.path.exists(bc_path):
            os.remove(bc_path)

if __name__ == "__main__":
    main()
