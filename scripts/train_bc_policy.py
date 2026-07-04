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

from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.route_step_policy import RouteStepPolicy
from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

class BCDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        
        return {
            'raster': torch.tensor(t['raster'], dtype=torch.float32),
            'layer_mask': torch.tensor(t['layer_mask'], dtype=torch.float32),
            'cursor_pos': torch.tensor(t['cursor_pos'], dtype=torch.float32),
            'target_pos': torch.tensor(t['target_pos'], dtype=torch.float32),
            'moves_remaining_frac': torch.tensor(t['moves_remaining_frac'], dtype=torch.float32),
            'action': torch.tensor(t['action'], dtype=torch.long),
            'valid_mask': torch.tensor(t['valid_mask'], dtype=torch.bool),
            'steps_remaining': torch.tensor(t.get('steps_remaining', 0.0), dtype=torch.float32),
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
                    # Get observations
                    raster_tensor = torch.tensor(env._get_obs()['board_raster'], dtype=torch.float32).unsqueeze(0).to(device)
                    layer_mask = torch.tensor(env._get_obs()['layer_mask'], dtype=torch.float32).unsqueeze(0).to(device)
                    cursor_norm = torch.tensor(env._get_obs()['cursor_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    target_norm = torch.tensor(env._get_obs()['target_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    moves_frac = torch.tensor(env._get_obs()['moves_remaining_frac'], dtype=torch.float32).unsqueeze(0).to(device)
                    
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--unfreeze_encoders', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load dataset shards
    all_episodes = []
    shard_paths = glob.glob("data/bc_dataset/*.pkl")
    if not shard_paths:
        raise FileNotFoundError("No dataset shards found in data/bc_dataset/. Run scripts/generate_bc_dataset.py first.")
        
    for p in shard_paths:
        if os.path.getsize(p) == 0:
            print(f"Skipping empty dataset shard: {p}")
            continue
        try:
            with open(p, "rb") as f:
                stage_episodes = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Failed to load corrupted dataset shard {p}: {e}. Skipping...")
            continue
            
        # Add steps_remaining labels
        for ep in stage_episodes:
            L = len(ep)
            for i, step in enumerate(ep):
                step['steps_remaining'] = L - 1 - i
        all_episodes.extend(stage_episodes)
        print(f"Loaded {len(stage_episodes)} episodes from {p}")
        
    if not all_episodes:
        raise ValueError("No valid episodes could be loaded from dataset shards! Ensure scripts/generate_bc_dataset.py completes successfully.")
            
    # 2. Train/val split by episode
    random.seed(42)
    random.shuffle(all_episodes)
    split_idx = int(len(all_episodes) * 0.8)
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:]
    
    # Flatten episodes to transition datasets
    train_transitions = [step for ep in train_episodes for step in ep]
    val_transitions = [step for ep in val_episodes for step in ep]
    
    train_dataset = BCDataset(train_transitions)
    val_dataset = BCDataset(val_transitions)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Training set: {len(train_dataset)} steps, Validation set: {len(val_dataset)} steps")
    
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
    
    # Optional load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        vit.load_state_dict(ckpt.get('vit', ckpt))
        gnn.load_state_dict(ckpt.get('gnn', ckpt))
        fusion.load_state_dict(ckpt.get('fusion', ckpt))
        
    # Freezing logic
    if not args.unfreeze_encoders:
        print("Freezing encoder (ViT, GNN, Fusion) parameters...")
        for p in list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters()):
            p.requires_grad = False
            
    # Setup optimizer
    params = list(policy.parameters())
    if args.unfreeze_encoders:
        params += list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
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
    for epoch in range(args.epochs):
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
        
        # Guard check: confirm no gradients flow to encoders if frozen
        if not args.unfreeze_encoders:
            for p in list(vit.parameters()) + list(gnn.parameters()) + list(fusion.parameters()):
                assert p.grad is None, "Gradients are flowing into frozen encoders!"
                
        for batch in train_loader:
            rasters = batch['raster'].to(device)
            layer_masks = batch['layer_mask'].to(device)
            cursor_poses = batch['cursor_pos'].to(device)
            target_poses = batch['target_pos'].to(device)
            moves_fracs = batch['moves_remaining_frac'].to(device)
            actions = batch['action'].to(device)
            valid_masks = batch['valid_mask'].to(device)
            steps_remainings = batch['steps_remaining'].to(device)
            
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
            
            loss_action = criterion_action(masked_logits, actions)
            loss_val = criterion_value(value.squeeze(-1), steps_remainings)
            
            loss = loss_action + 0.001 * loss_val
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * rasters.size(0)
            preds = masked_logits.argmax(dim=-1)
            train_acc += (preds == actions).sum().item()
            
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)
        
        # Validation pass
        policy.eval()
        vit.eval()
        gnn.eval()
        fusion.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                rasters = batch['raster'].to(device)
                cursor_poses = batch['cursor_pos'].to(device)
                target_poses = batch['target_pos'].to(device)
                moves_fracs = batch['moves_remaining_frac'].to(device)
                actions = batch['action'].to(device)
                valid_masks = batch['valid_mask'].to(device)
                steps_remainings = batch['steps_remaining'].to(device)
                
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
                
                loss_action = criterion_action(masked_logits, actions)
                loss_val = criterion_value(value.squeeze(-1), steps_remainings)
                loss = loss_action + 0.001 * loss_val
                
                val_loss += loss.item() * rasters.size(0)
                preds = masked_logits.argmax(dim=-1)
                val_acc += (preds == actions).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)
        
        # Closed-loop evaluation
        comp_rate, mean_drc = evaluate_closed_loop(eval_env, policy, vit, gnn, fusion, device, num_episodes=5)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Eval Rollout Comp Rate: {comp_rate*100:.1f}% | DRC Violations: {mean_drc:.2f}")
        
        # Free memory cache and clean GPU memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    # Save pretrained policy checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/bc_pretrained_policy.pt"
    torch.save({
        'policy': policy.state_dict(),
        'vit': vit.state_dict(),
        'gnn': gnn.state_dict(),
        'fusion': fusion.state_dict()
    }, save_path)
    print(f"Pretraining complete. Saved weights to {save_path}")

if __name__ == '__main__':
    main()
