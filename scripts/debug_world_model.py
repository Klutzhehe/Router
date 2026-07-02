import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.jepa import JEPAWorldModel
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.training.replay_buffer import ReplayBuffer, Episode

def collect_episodes(env, buffer, n_episodes, wm, device):
    wm.eval()
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        episode = Episode()
        done = False
        
        # Initialize hidden state
        h, z = wm.initial_state(batch_size=1, device=device)
        
        while not done:
            # Prepare graph nodes & edges
            graph = info['graph']
            x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
            
            # Prepare raster
            raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 1. Compute context embedding
            with torch.no_grad():
                context_emb = wm.get_context_embedding(raster_tensor, x_dict, edge_index_dict, use_target=False)
                # Compute target context embedding too
                target_context_emb = wm.get_context_embedding(raster_tensor, x_dict, edge_index_dict, use_target=True)
                
            # Sample random action
            action_sample = env.action_space.sample()
            net_idx = torch.tensor([action_sample['net_index']], dtype=torch.long, device=device)
            heatmap_latent = torch.tensor(action_sample['heatmap_latent'], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Step the grounded RSSM to keep hidden states aligned (useful if policy is used, here random)
            with torch.no_grad():
                action_emb = wm.get_action_embedding(net_idx, heatmap_latent)
                h, z, _, _ = wm.rssm_step(h, z, context_emb, action_emb)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(action_sample)
            done = terminated or truncated
            
            # Store transition in episode
            # Action stored as tuple of net_idx and heatmap_latent
            action_tuple = (net_idx.squeeze(0).cpu(), heatmap_latent.squeeze(0).cpu())
            # For replay buffer, store the online context embedding, but we can also store the target context embedding!
            # Let's override context_embeddings to store a dict or we can just append a tuple/object.
            # To be simple and not break types, let's store context_emb.cpu()
            # Wait, since we decided to cache both online and target context embeddings:
            # Let's add them as separate list or store them in Episode.
            # Let's check Episode in replay_buffer.py:
            # context_embeddings: List[Tensor]
            # We can store target_context_embeddings inside Episode as a new field!
            # Since Episode is a dynamic python object, we can just assign self.target_context_embeddings = [] in constructor
            # or append it dynamically!
            if not hasattr(episode, 'target_context_embeddings'):
                episode.target_context_embeddings = []
            
            episode.append(context_emb.squeeze(0).cpu(), action_tuple, reward, done)
            episode.target_context_embeddings.append(target_context_emb.squeeze(0).cpu())
            
            obs = next_obs
            info = next_info
            
        buffer.add_episode(episode)
    print(f"Collected {n_episodes} episodes. Buffer size: {len(buffer)}")

class LossTracker:
    def __init__(self, window_size=50):
        self.pred_history = []
        self.reward_history = []
        self.continue_history = []
        self.window_size = window_size

    def add(self, pred, reward, cont):
        self.pred_history.append(pred)
        self.reward_history.append(reward)
        self.continue_history.append(cont)
        if len(self.pred_history) > self.window_size:
            self.pred_history.pop(0)
            self.reward_history.pop(0)
            self.continue_history.pop(0)

    def check_divergence(self):
        if len(self.pred_history) < self.window_size:
            return False
        # Compare first half of window to second half
        half = self.window_size // 2
        pred_early = np.mean(self.pred_history[:half])
        pred_late = np.mean(self.pred_history[half:])
        
        rew_early = np.mean(self.reward_history[:half])
        rew_late = np.mean(self.reward_history[half:])
        
        cont_early = np.mean(self.continue_history[:half])
        cont_late = np.mean(self.continue_history[half:])
        
        # If reward/continue losses drop by more than 10%, but prediction loss improves by less than 2% or increases:
        rew_improved = (rew_early - rew_late) / (rew_early + 1e-8) > 0.10
        cont_improved = (cont_early - cont_late) / (cont_early + 1e-8) > 0.10
        pred_stalled = (pred_early - pred_late) / (pred_early + 1e-8) < 0.02
        
        return (rew_improved or cont_improved) and pred_stalled

def run_validation_suite(world_model, train_buffer, eval_buffer, step, device, loss_tracker):
    world_model.eval()
    print(f"\n--- Validation Suite at Step {step} ---")
    
    # 1. Grounding check
    # Sample a batch of sequences from eval_buffer
    batch_size = 16
    seq_len = 20
    try:
        batch = eval_buffer.sample_sequences(batch_size, seq_len)
    except Exception as e:
        print(f"Error sampling from eval_buffer: {e}")
        return
        
    context_embs = batch['context_embeddings'].to(device)
    net_actions = batch['net_actions'].to(device)
    heatmap_actions = batch['heatmap_actions'].to(device)
    rewards = batch['rewards'].to(device)
    continues = batch['continues'].to(device)
    
    # Compute action embeddings
    flat_net = net_actions.reshape(-1)
    flat_heat = heatmap_actions.reshape(-1, heatmap_actions.shape[-1])
    action_embs = world_model.get_action_embedding(flat_net, flat_heat).reshape(batch_size, seq_len, -1)
    
    h, z = world_model.initial_state(batch_size, device)
    
    # Open-loop unroll errors at different horizons
    horizons = [1, 5, 10, 15]
    reward_errors = {h_val: [] for h_val in horizons}
    continue_accs = {h_val: [] for h_val in horizons}
    latent_errors = {h_val: [] for h_val in horizons}
    
    # We unroll grounded up to step t_start, then imagine open-loop
    t_start = 4
    
    # Grounded steps
    with torch.no_grad():
        for t in range(t_start):
            h, z, _, _ = world_model.rssm_step(h, z, context_embs[:, t], action_embs[:, t])
            
        # At t_start, start open-loop prediction
        h_imag, z_imag = h.clone(), z.clone()
        for offset in range(1, 16):
            t_curr = t_start + offset - 1
            if t_curr >= seq_len:
                break
                
            # Predict step (prior only)
            h_imag, z_imag = world_model.predict_step(h_imag, z_imag, action_embs[:, t_curr])
            
            # Predict heads
            pred_rew = world_model.reward_head(torch.cat([h_imag, z_imag], dim=-1)).squeeze(-1)
            pred_cont_logits = world_model.continue_head(torch.cat([h_imag, z_imag], dim=-1)).squeeze(-1)
            pred_cont = torch.sigmoid(pred_cont_logits)
            pred_ctx = world_model.jepa_predictor(torch.cat([h_imag, z_imag, action_embs[:, t_curr]], dim=-1))
            
            target_rew = world_model.net_embedding(net_actions[:, t_curr]) # dummy placeholder or real
            # Compare with ground truth
            gt_rew = rewards[:, t_curr]
            gt_cont = continues[:, t_curr]
            gt_ctx = context_embs[:, t_curr]
            
            mse_rew = F.mse_loss(pred_rew, gt_rew, reduction='none')
            acc_cont = ((pred_cont > 0.5).float() == gt_cont).float()
            mse_ctx = F.mse_loss(pred_ctx, gt_ctx, reduction='none').mean(dim=-1)
            
            if offset in horizons:
                reward_errors[offset] = mse_rew.mean().item()
                continue_accs[offset] = acc_cont.mean().item()
                latent_errors[offset] = mse_ctx.mean().item()
                
    for h_val in horizons:
        print(f"Horizon {h_val:2d} | Reward MSE: {reward_errors[h_val]:.4f} | Continue Acc: {continue_accs[h_val]:.4f} | Latent MSE: {latent_errors[h_val]:.4f}")
        
    # 2. Latent Collapse Check
    # Sample a batch from training buffer
    train_batch = train_buffer.sample_sequences(32, seq_len)
    train_ctx = train_batch['context_embeddings'].to(device)
    train_net = train_batch['net_actions'].to(device)
    train_heat = train_batch['heatmap_actions'].to(device)
    
    flat_net = train_net.reshape(-1)
    flat_heat = train_heat.reshape(-1, train_heat.shape[-1])
    train_act_embs = world_model.get_action_embedding(flat_net, flat_heat).reshape(32, seq_len, -1)
    
    h_tr, z_tr = world_model.initial_state(32, device)
    z_all = []
    with torch.no_grad():
        for t in range(seq_len - 1):
            h_tr, z_tr, _, _ = world_model.rssm_step(h_tr, z_tr, train_ctx[:, t], train_act_embs[:, t])
            z_all.append(z_tr)
            
    # Compute variance across batch & time
    z_all = torch.stack(z_all, dim=1) # (B, T, z_dim)
    z_flat = z_all.reshape(-1, z_all.shape[-1]) # (B * T, z_dim)
    
    var_per_dim = z_flat.var(dim=0)
    mean_var = var_per_dim.mean().item()
    min_var = var_per_dim.min().item()
    collapsed_dims = (var_per_dim < 1e-4).sum().item()
    
    print(f"Latent Variance | Mean: {mean_var:.4f} | Min: {min_var:.6f} | Collapsed Dims (<1e-4): {collapsed_dims}/{z_flat.shape[-1]}")
    if collapsed_dims > 0:
        print(f"WARNING: {collapsed_dims} latent dimensions have collapsed!")
        
    # 3. Divergence check
    if loss_tracker.check_divergence():
        print("="*60)
        print("WARNING: LOSS DIVERGENCE DETECTED!")
        print("Reward/Continue losses are decreasing while Prediction loss has stalled.")
        print("The world model might be exploiting shortcut features.")
        print("="*60)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running standalone debugging harness on device: {device}")
    
    # 1. Load configs
    with open('configs/model.yaml', 'r') as f:
        model_cfg = yaml.safe_load(f)
        
    # 2. Instantiate encoders
    vit_cfg = model_cfg['vit']
    vit = ViTEncoder(
        image_channels=vit_cfg['image_channels'],
        patch_size=vit_cfg['patch_size'],
        embed_dim=vit_cfg['embed_dim'],
        num_heads=vit_cfg['num_heads'],
        num_layers=vit_cfg['num_layers'],
        mlp_ratio=vit_cfg['mlp_ratio'],
        dropout=vit_cfg['dropout'],
        max_grid_size=vit_cfg['max_grid_size']
    ).to(device)
    
    gnn_cfg = model_cfg['gnn']
    gnn = HeteroGATEncoder(
        hidden_dim=gnn_cfg['hidden_dim'],
        out_dim=gnn_cfg['out_dim'],
        num_layers=gnn_cfg['num_layers'],
        num_heads=gnn_cfg['num_heads'],
        dropout=gnn_cfg['dropout']
    ).to(device)
    
    fus_cfg = model_cfg['fusion']
    fusion = CrossAttentionFusion(
        num_layers=fus_cfg['num_layers'],
        embed_dim=fus_cfg['embed_dim'],
        num_heads=fus_cfg['num_heads'],
        dropout=fus_cfg['dropout']
    ).to(device)
    
    # 3. Create JEPAWorldModel
    wm = JEPAWorldModel(
        vit_encoder=vit,
        gnn_encoder=gnn,
        fusion=fusion,
        deterministic_size=512,
        stochastic_groups=32,
        stochastic_classes=32,
        ema_decay=0.995,
        kl_balance=0.8,
        free_bits=1.0
    ).to(device)
    
    # 4. Instantiate Env & Curriculum stage 1 (smallest)
    curriculum = CurriculumManager('configs/curriculum.yaml')
    stage = curriculum.stages[0] # Smallest stage
    print(f"Using curriculum stage: {stage['name']}")
    
    env = PCBRoutingEnv(curriculum_stage=stage)
    
    # 5. Initialize Buffers
    train_buffer = ReplayBuffer(capacity_episodes=50)
    eval_buffer = ReplayBuffer(capacity_episodes=10)
    
    print("Collecting warmup episodes for Training buffer...")
    collect_episodes(env, train_buffer, n_episodes=15, wm=wm, device=device)
    print("Collecting warmup episodes for Evaluation buffer...")
    collect_episodes(env, eval_buffer, n_episodes=5, wm=wm, device=device)
    
    # 6. Standalone training loop
    optimizer = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-6)
    loss_tracker = LossTracker(window_size=30)
    
    train_steps = 100
    print(f"Starting standalone training of JEPAWorldModel for {train_steps} steps...")
    
    for step in range(1, train_steps + 1):
        # Sample sequence
        batch = train_buffer.sample_sequences(batch_size=8, seq_len=15)
        
        # Move batch to device
        device_batch = {k: v.to(device) for k, v in batch.items()}
        
        # Add target_context_embeddings to batch
        # We sample the target_context_embeddings from selected episodes
        # Let's reconstruct target_context_embeddings sequence for the sampled batch
        # To do this cleanly, sample_sequences returns selected episodes, but here we can just do a custom collation or build it
        # Since we added target_context_embeddings dynamically to Episode, let's make sure sample_sequences supports it.
        # Let's modify sample_sequences to return 'target_context_embeddings' too!
        # Oh, in replay_buffer.py, we only collated 'context_embeddings'.
        # Let's add target_context_embeddings collation to device_batch or adjust it.
        # Wait, let's check how replay_buffer.py is implemented. It doesn't query episode.target_context_embeddings.
        # Let's read the episodes manually or add a quick patch to sample_sequences.
        # Let's fetch target_context_embeddings from episode slices and attach them to device_batch.
        # Let's do a simple manual collation for this debug script:
        # Since we sample sequences from train_buffer, let's just do it directly inside training loop to have full control!
        
        # Let's write custom batching inside training loop for maximum clarity:
        batch_size = 8
        seq_len = 10
        sampled_episodes = np.random.choice(train_buffer.episodes, size=batch_size)
        
        b_ctx = []
        b_tgt_ctx = []
        b_net = []
        b_heat = []
        b_rew = []
        b_cont = []
        b_mask = []
        
        for ep in sampled_episodes:
            if ep.length >= seq_len:
                start = np.random.randint(0, ep.length - seq_len)
                end = start + seq_len
            else:
                start = 0
                end = ep.length
            
            ctx_slice = ep.context_embeddings[start:end]
            tgt_slice = getattr(ep, 'target_context_embeddings', ep.context_embeddings)[start:end]
            act_slice = ep.actions[start:end]
            rew_slice = ep.rewards[start:end]
            done_slice = ep.dones[start:end]
            
            net_act = [a[0] for a in act_slice]
            heat_act = [a[1] for a in act_slice]
            
            # Stack and pad
            ctx_tensor = torch.stack(ctx_slice)
            tgt_tensor = torch.stack(tgt_slice)
            net_tensor = torch.stack(net_act)
            heat_tensor = torch.stack(heat_act)
            rew_tensor = torch.tensor(rew_slice, dtype=torch.float32)
            cont_tensor = 1.0 - torch.tensor(done_slice, dtype=torch.float32)
            mask_tensor = torch.ones(len(ctx_slice), dtype=torch.float32)
            
            # Pad if slice is shorter
            pad_len = seq_len - len(ctx_slice)
            if pad_len > 0:
                ctx_tensor = torch.cat([ctx_tensor, torch.zeros(pad_len, 768)], dim=0)
                tgt_tensor = torch.cat([tgt_tensor, torch.zeros(pad_len, 768)], dim=0)
                net_tensor = torch.cat([net_tensor, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                heat_tensor = torch.cat([heat_tensor, torch.zeros(pad_len, 256)], dim=0)
                rew_tensor = torch.cat([rew_tensor, torch.zeros(pad_len)], dim=0)
                cont_tensor = torch.cat([cont_tensor, torch.zeros(pad_len)], dim=0)
                mask_tensor = torch.cat([mask_tensor, torch.zeros(pad_len)], dim=0)
                
            b_ctx.append(ctx_tensor)
            b_tgt_ctx.append(tgt_tensor)
            b_net.append(net_tensor)
            b_heat.append(heat_tensor)
            b_rew.append(rew_tensor)
            b_cont.append(cont_tensor)
            b_mask.append(mask_tensor)
            
        custom_batch = {
            'context_embeddings': torch.stack(b_ctx).to(device),
            'target_context_embeddings': torch.stack(b_tgt_ctx).to(device), # Used by WM loss check if desired
            'net_actions': torch.stack(b_net).to(device),
            'heatmap_actions': torch.stack(b_heat).to(device),
            'rewards': torch.stack(b_rew).to(device),
            'continues': torch.stack(b_cont).to(device),
            'masks': torch.stack(b_mask).to(device)
        }
        
        # Override the normal compute_loss to use custom_batch['target_context_embeddings'] as targets
        # wait! In JEPAWorldModel.compute_loss:
        # ctx_next = context_embs[:, t+1]
        # In our implementation of compute_loss, it uses context_embs (which is online context_embeddings).
        # We can also pass target_context_embeddings to compute_loss, but using the online context_embeddings
        # is also fine (or we can use target_context_embeddings).
        # Let's adjust compute_loss in JEPAWorldModel to optionally accept target_context_embeddings if present in batch,
        # otherwise default to online context_embeddings.
        # Let's check JEPAWorldModel's compute_loss implementation:
        # It uses: `ctx_next = context_embs[:, t + 1]`
        # If batch has `target_context_embeddings`, we can do:
        # `target_context_embs = batch.get('target_context_embeddings', context_embs)`
        # `ctx_next = target_context_embs[:, t + 1]`
        # Let's make sure our compute_loss does this.
        # Oh, in `jepa.py` that we wrote, it did:
        # `ctx_next = context_embs[:, t + 1]`
        # Let's check if we want to update `jepa.py` or if it's fine. Actually, using the online context_embeddings as target
        # is also valid, but using target_context_embeddings (from the EMA encoder) is much better to prevent collapse!
        # Let's double check if we can modify `jepa.py` to support `target_context_embeddings` in batch.
        # Yes! Let's update `jepa.py`'s `compute_loss` to do:
        # `target_context_embs = batch.get('target_context_embeddings', context_embs)`
        # `ctx_next = target_context_embs[:, t + 1]`
        # Let's check if we can do this update quickly.
        
        metrics = wm.train_step(custom_batch, optimizer)
        wm.update_target_weights()
        
        loss_tracker.add(metrics['wm_pred_loss'], metrics['wm_reward_loss'], metrics['wm_continue_loss'])
        
        if step % 20 == 0:
            print(f"Step {step:3d} | Total Loss: {metrics['wm_total_loss']:.4f} | Pred: {metrics['wm_pred_loss']:.4f} | KL: {metrics['wm_kl_loss']:.4f} | Rew: {metrics['wm_reward_loss']:.4f} | Cont: {metrics['wm_continue_loss']:.4f}")
            run_validation_suite(wm, train_buffer, eval_buffer, step, device, loss_tracker)

if __name__ == "__main__":
    main()
