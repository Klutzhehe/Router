import os
import glob
import pickle
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.route_step_policy import RouteStepPolicy
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

from torch.utils.data import IterableDataset

class BCIterableDataset(IterableDataset):
    def __init__(self, file_paths, shuffle=False):
        self.file_paths = file_paths
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            paths = list(self.file_paths)
        else:
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            paths = self.file_paths[worker_id * per_worker : (worker_id + 1) * per_worker]
            
        import random
        if self.shuffle:
            random.shuffle(paths)
            
        for p in paths:
            import gzip, pickle
            open_func = gzip.open if p.endswith('.gz') else open
            try:
                with open_func(p, "rb") as f:
                    episodes = pickle.load(f)
            except Exception:
                continue
                
            transitions = []
            for ep in episodes:
                L = len(ep)
                for i, step in enumerate(ep):
                    step['steps_remaining'] = float(L - 1 - i)
                    transitions.append(step)
                    
            if self.shuffle:
                random.shuffle(transitions)
                
            for t in transitions:
                yield {
                    'raster': torch.tensor(t['raster'], dtype=torch.float32),
                    'layer_mask': torch.tensor(t['layer_mask'], dtype=torch.float32),
                    'cursor_pos': torch.tensor(t['cursor_pos'], dtype=torch.float32),
                    'target_pos': torch.tensor(t['target_pos'], dtype=torch.float32),
                    'moves_remaining_frac': torch.tensor(t['moves_remaining_frac'], dtype=torch.float32),
                    'action': torch.tensor(t['action'], dtype=torch.long),
                    'valid_mask': torch.tensor(t['valid_mask'], dtype=torch.bool),
                    'steps_remaining': torch.tensor(t['steps_remaining'], dtype=torch.float32),
                    'graph': t['graph'],
                    'orig_size': torch.tensor(t['raster'].shape[1:], dtype=torch.float32)
                }

def collate_fn(batch):
    # Find max H and W in this batch
    max_h = max(x['raster'].shape[1] for x in batch)
    max_w = max(x['raster'].shape[2] for x in batch)
    
    padded_rasters = []
    cursor_poses = []
    target_poses = []
    
    for x in batch:
        r = x['raster']
        h, w = r.shape[1], r.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            r = F.pad(r, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        padded_rasters.append(r)
        
        # Re-normalize coordinates based on padded dimensions
        cp = x['cursor_pos'].clone()
        cp[0] = cp[0] * x['orig_size'][1] / max_w
        cp[1] = cp[1] * x['orig_size'][0] / max_h
        cursor_poses.append(cp)
        
        tp = x['target_pos'].clone()
        tp[0] = tp[0] * x['orig_size'][1] / max_w
        tp[1] = tp[1] * x['orig_size'][0] / max_h
        target_poses.append(tp)
        
    rasters = torch.stack(padded_rasters)
    layer_masks = torch.stack([x['layer_mask'] for x in batch])
    cursor_poses = torch.stack(cursor_poses)
    target_poses = torch.stack(target_poses)
    moves_remaining_fracs = torch.stack([x['moves_remaining_frac'] for x in batch])
    actions = torch.stack([x['action'] for x in batch])
    valid_masks = torch.stack([x['valid_mask'] for x in batch])
    steps_remainings = torch.stack([x['steps_remaining'] for x in batch])
    
    graphs = [x['graph'] for x in batch]
    
    return {
        'raster': rasters,
        'layer_mask': layer_masks,
        'cursor_pos': cursor_poses,
        'target_pos': target_poses,
        'moves_remaining_frac': moves_remaining_fracs,
        'action': actions,
        'valid_mask': valid_masks,
        'steps_remaining': steps_remainings,
        'graphs': graphs
    }

def get_valid_mask(env):
    cx, cy, cl = env.cursor_pos
    tx, ty, tl = env.target_pos
    active_layers = list(range(env.board.num_layers))
    exempt = {(cx, cy), (tx, ty)}
    temp_obs = build_obstacle_maps(env.board_state, active_layers, exempt, shape=(env.H, env.W))
    temp_via = build_via_blocked_maps(env.board_state, temp_obs, active_layers, shape=(env.H, env.W))
    
    valid_mask = np.zeros(10, dtype=bool)
    for a_idx in range(8):
        mdx, mdy, _ = env.pathfinder.moves[a_idx]
        mx, my = cx + mdx, cy + mdy
        if 0 <= mx < env.W and 0 <= my < env.H:
            if not temp_obs[cl][my, mx]:
                valid_mask[a_idx] = True
    for dl_idx, v_dl in enumerate([-1, 1]):
        v_nl = cl + v_dl
        if v_nl in active_layers:
            if not temp_via[cl][cy, cx] and not temp_via[v_nl][cy, cx]:
                valid_mask[8 + dl_idx] = True
    return valid_mask

def evaluate_closed_loop(env, policy, vit, gnn, fusion, device, num_episodes=5):
    policy.eval()
    vit.eval()
    gnn.eval()
    fusion.eval()
    
    env.routing_mode = 'autoregressive'
    completed_nets = 0
    total_nets = 0
    total_drc = 0
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset(seed=1000 + ep)
            episode_nets = list(env.board.nets)
            total_nets += len(episode_nets)
            
            for net_idx, net in enumerate(episode_nets):
                env.start_routing_net(net_idx)
                
                net_done = False
                max_steps = env.W * env.H * 2  # generous cap: 2x board cells
                steps = 0
                while not net_done and steps < max_steps:
                    steps += 1
                    # Get observations once per step
                    obs_dict = env._get_obs()
                    raster_tensor = torch.tensor(obs_dict['board_raster'], dtype=torch.float32).unsqueeze(0).to(device)
                    layer_mask = torch.tensor(obs_dict['layer_mask'], dtype=torch.float32).unsqueeze(0).to(device)
                    cursor_norm = torch.tensor(obs_dict['cursor_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    target_norm = torch.tensor(obs_dict['target_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    moves_frac = torch.tensor(obs_dict['moves_remaining_frac'], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    graph = env.graph
                    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
                    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
                    
                    # Run encoders
                    spatial_patches, cls_spatial = vit(raster_tensor)
                    node_embs = gnn(x_dict, edge_index_dict)
                    pad_embs = node_embs['pad'].unsqueeze(0)
                    fused_pads, fused_spatial = fusion(pad_embs, spatial_patches)
                    
                    # Get action from policy
                    logits, value = policy(fused_spatial, cursor_norm, target_norm, moves_frac)
                    
                    # Mask logits dynamically
                    v_mask = torch.tensor(get_valid_mask(env), dtype=torch.bool, device=device).unsqueeze(0)
                    logits = logits.masked_fill(~v_mask, -1e4)
                    
                    action = logits.argmax(dim=-1).item()
                    obs, reward, term, trunc, info = env.step({'action_id': action})
                    net_done = (env.current_net_index is None)
                    
            completed_nets += len(env.routed_nets)
            total_drc += len(env.drc_violations)
            
    comp_rate = completed_nets / max(1, total_nets)
    mean_drc = total_drc / max(1, num_episodes)
    return comp_rate, mean_drc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--unfreeze_encoders', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='data/bc_dataset', help='Directory to load dataset shards from')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Enable PyTorch speed optimizations for modern GPUs
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Uses TF32 on TensorCores for huge speedup
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Gather all file paths
    shard_paths = glob.glob(os.path.join(args.data_dir, "*.pkl.gz")) + glob.glob(os.path.join(args.data_dir, "*.pkl"))
    if not shard_paths:
        raise FileNotFoundError(f"No dataset shards found in {args.data_dir}/. Run scripts/generate_bc_dataset.py first.")
        
    # 2. Train/val split by file
    random.seed(42)
    random.shuffle(shard_paths)
    split_idx = int(len(shard_paths) * 0.8)
    train_paths = shard_paths[:split_idx]
    val_paths = shard_paths[split_idx:]
    
    # No need to count sizes upfront, we'll keep a running tally during the epoch loop
    train_dataset = BCIterableDataset(train_paths, shuffle=True)
    val_dataset = BCIterableDataset(val_paths, shuffle=False)
    
    # num_workers=0: runs in main thread to avoid Colab shared memory crashes
    # pin_memory=True: enables faster CPU→GPU tensor transfers
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # 3. Init encoders and policy
    # We load default model config to setup encoders
    import yaml
    with open("configs/model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    vit_cfg = model_cfg['vit']
    gnn_cfg = model_cfg['gnn']
    fus_cfg = model_cfg['fusion']
    
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
    
    gnn = HeteroGATEncoder(
        hidden_dim=gnn_cfg['hidden_dim'],
        out_dim=gnn_cfg['out_dim'],
        num_layers=gnn_cfg['num_layers'],
        num_heads=gnn_cfg['num_heads'],
        dropout=gnn_cfg['dropout']
    ).to(device)
    
    fusion = CrossAttentionFusion(
        num_layers=fus_cfg['num_layers'],
        embed_dim=fus_cfg['embed_dim'],
        num_heads=fus_cfg['num_heads'],
        dropout=fus_cfg['dropout']
    ).to(device)
    
    policy = RouteStepPolicy(embed_dim=vit_cfg['embed_dim']).to(device)
    

    # Freezing logic
    if not args.unfreeze_encoders:
        print("Freezing encoder (ViT, GNN, Fusion) parameters...")
        for p in list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters()):
            p.requires_grad = False
            
    # Setup optimizer
    params = list(policy.parameters())
    if args.unfreeze_encoders:
        params += list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters())
    # fused=True drastically speeds up the optimizer step on the GPU
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4, fused=torch.cuda.is_available())
    
    # Cosine annealing with linear warmup (standard for transformer training)
    warmup_epochs = max(1, int(args.epochs * 0.05))  # 5% warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)  # linear warmup
        progress = float(epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_epoch = 0
    best_val_acc = 0.0
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Optional load checkpoint for resume
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'vit' in ckpt:
            vit.load_state_dict(ckpt['vit'])
        if 'gnn' in ckpt:
            gnn.load_state_dict(ckpt['gnn'])
        if 'fusion' in ckpt:
            fusion.load_state_dict(ckpt['fusion'])
        if 'policy' in ckpt:
            policy.load_state_dict(ckpt['policy'])
        if 'optimizer' in ckpt and args.unfreeze_encoders:
            # Only load optimizer if we aren't changing freezing logic mid-run
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state (likely param group mismatch): {e}")
                
        if 'scaler' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler'])
                print("Loaded AMP GradScaler state.")
            except Exception as e:
                print(f"Warning: Could not load scaler state: {e}")
        
        start_epoch = ckpt.get('epoch', -1) + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        print(f"Resuming from epoch {start_epoch} (Best Val Acc: {best_val_acc*100:.2f}%)")

    
    criterion_action = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # Init validation env (use a simple stage config)
    curriculum = CurriculumManager("configs/curriculum.yaml")
    # stage 0 is easy empty board, stage 2 is moderate obstacles
    curriculum.current_stage_idx = 1
    eval_env = PCBRoutingEnv(
        board_config=curriculum.get_board_config(),
        curriculum_stage=curriculum.current_stage,
        reward_weights=curriculum.get_reward_weights(),
        routing_mode='autoregressive'
    )
    
    print("\nStarting BC pretraining loop...")
    for epoch in range(start_epoch, args.epochs):
        policy.train()
        if args.unfreeze_encoders:
            vit.train()
            gnn.train()
            fusion.train()
        else:
            vit.eval()
            gnn.eval()
            fusion.eval()
            
        train_loss = 0.0
        train_acc = 0.0
        train_steps_count = 0
        
        # Guard check: confirm no gradients flow to encoders if frozen
        if not args.unfreeze_encoders:
            for p in list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters()):
                assert p.grad is None, "Gradients are flowing into frozen encoders!"
                
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train"):
            rasters = batch['raster'].to(device)
            layer_masks = batch['layer_mask'].to(device)
            cursor_poses = batch['cursor_pos'].to(device)
            target_poses = batch['target_pos'].to(device)
            moves_fracs = batch['moves_remaining_frac'].to(device)
            actions = batch['action'].to(device)
            valid_masks = batch['valid_mask'].to(device)
            steps_remainings = batch['steps_remaining'].to(device)
            
            with torch.amp.autocast('cuda'):
                # Run ViT on the entire batch at once
                if not args.unfreeze_encoders:
                    with torch.no_grad():
                        spatial_patches, _ = vit(rasters)
                else:
                    spatial_patches, _ = vit(rasters)
                    
                # Combine individual graphs into a single batched graph
                batched_graph = Batch.from_data_list(batch['graphs']).to(device)
                pad_batch = batched_graph['pad'].batch
                
                # Forward GNN and dense collate pads in a single batched call
                if not args.unfreeze_encoders:
                    with torch.no_grad():
                        node_embs = gnn(batched_graph.x_dict, batched_graph.edge_index_dict)
                        pad_embs_dense, pad_mask = to_dense_batch(node_embs['pad'], pad_batch)
                        fused_pads, fused_spatial = fusion(pad_embs_dense, spatial_patches, gnn_mask=pad_mask)
                else:
                    node_embs = gnn(batched_graph.x_dict, batched_graph.edge_index_dict)
                    pad_embs_dense, pad_mask = to_dense_batch(node_embs['pad'], pad_batch)
                    fused_pads, fused_spatial = fusion(pad_embs_dense, spatial_patches, gnn_mask=pad_mask)
                
                # Forward policy
                logits, value = policy(fused_spatial, cursor_poses, target_poses, moves_fracs)
                
                # Action masking
                masked_logits = logits.masked_fill(~valid_masks, -1e4)
                
            # Compute loss outside autocast in float32 to prevent MSE overflow
            loss_action = criterion_action(masked_logits.float(), actions)
            loss_val = criterion_value(value.float().squeeze(-1), steps_remainings)
            
            loss = loss_action + 0.001 * loss_val
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * rasters.size(0)
            preds = masked_logits.argmax(dim=-1)
            train_acc += (preds == actions).sum().item()
            train_steps_count += rasters.size(0)
            
        if train_steps_count > 0:
            train_loss /= train_steps_count
            train_acc /= train_steps_count
        
        # Validation pass
        policy.eval()
        vit.eval()
        gnn.eval()
        fusion.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} Val  ", leave=False):
                rasters = batch['raster'].to(device)
                cursor_poses = batch['cursor_pos'].to(device)
                target_poses = batch['target_pos'].to(device)
                moves_fracs = batch['moves_remaining_frac'].to(device)
                actions = batch['action'].to(device)
                valid_masks = batch['valid_mask'].to(device)
                steps_remainings = batch['steps_remaining'].to(device)
                
                with torch.amp.autocast('cuda'):
                    # Run ViT on the entire batch at once
                    spatial_patches, _ = vit(rasters)
                    
                    # Combine graphs into a single batched graph
                    batched_graph = Batch.from_data_list(batch['graphs']).to(device)
                    pad_batch = batched_graph['pad'].batch
                    
                    # Single batched forward pass for GNN and Fusion
                    node_embs = gnn(batched_graph.x_dict, batched_graph.edge_index_dict)
                    pad_embs_dense, pad_mask = to_dense_batch(node_embs['pad'], pad_batch)
                    fused_pads, fused_spatial = fusion(pad_embs_dense, spatial_patches, gnn_mask=pad_mask)
                    
                    logits, value = policy(fused_spatial, cursor_poses, target_poses, moves_fracs)
                    masked_logits = logits.masked_fill(~valid_masks, -1e4)
                    
                # Compute loss outside autocast in float32
                loss_action = criterion_action(masked_logits.float(), actions)
                loss_val = criterion_value(value.float().squeeze(-1), steps_remainings)
                loss = loss_action + 0.001 * loss_val
                
                val_loss += loss.item() * rasters.size(0)
                preds = masked_logits.argmax(dim=-1)
                val_acc += (preds == actions).sum().item()
                val_steps_count += rasters.size(0)
                
        if val_steps_count > 0:
            val_loss /= val_steps_count
            val_acc /= val_steps_count
        
        # Closed-loop evaluation — expensive, run every 10 epochs only
        if (epoch + 1) % 10 == 0 or epoch == 0:
            comp_rate, mean_drc = evaluate_closed_loop(eval_env, policy, vit, gnn, fusion, device, num_episodes=5)
        else:
            comp_rate, mean_drc = float('nan'), float('nan')
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        eval_str = f"{comp_rate*100:.1f}% | DRC: {mean_drc:.2f}" if not math.isnan(comp_rate) else "-- (skipped)"
        print(f"Epoch {epoch+1:02d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Eval Rollout: {eval_str}")
        
        # Free memory cache and clean GPU memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Save checkpoints
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, "bc_policy_latest.pt")
        torch.save({
            'epoch': epoch,
            'policy': policy.state_dict(),
            'vit': vit.state_dict(),
            'gnn': gnn.state_dict(),
            'fusion': fusion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'val_acc': val_acc
        }, latest_path)
        
        # Save best checkpoint
        if 'best_val_acc' not in locals() or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, "bc_policy_best.pt")
            torch.save({
                'epoch': epoch,
                'policy': policy.state_dict(),
                'vit': vit.state_dict(),
                'gnn': gnn.state_dict(),
                'fusion': fusion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'val_acc': val_acc
            }, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (Val Acc: {val_acc*100:.2f}%)")
        
    # Also save the final one to the expected path for Dreamer
    final_path = os.path.join(args.save_dir, "bc_pretrained_policy.pt")
    import shutil
    best_path = os.path.join(args.save_dir, "bc_policy_best.pt")
    if os.path.exists(best_path):
        shutil.copy(best_path, final_path)
        print(f"Pretraining complete. Copied best weights to {final_path}")
    else:
        # Fallback if best doesn't exist for some reason
        torch.save({
            'policy': policy.state_dict(),
            'vit': vit.state_dict(),
            'gnn': gnn.state_dict(),
            'fusion': fusion.state_dict()
        }, final_path)
        print(f"Pretraining complete. Saved final weights to {final_path}")

if __name__ == '__main__':
    main()
