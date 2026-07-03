import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.jepa import SpatialJEPA
from pcb_router.models.policy import PPOPolicy
from pcb_router.models.heatmap_decoder import HeatmapDecoder

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.training.rewards import RewardCalculator

class RolloutBuffer:
    def __init__(self):
        self.rasters = []
        self.graphs = []
        self.layer_masks = []
        
        self.net_actions = []
        self.heatmap_actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        self.log_probs_net = []
        self.log_probs_heatmap = []
        
        self.advantages = []
        self.returns = []
        
        # Pre-computed net embeddings (CPU tensors) cached during rollout
        # Shape per step: (max_nets, embed_dim). Avoids storing full board deep copies.
        self.net_embs_cache = []
        self.unrouted_masks = []

    def clear(self):
        self.rasters.clear()
        self.graphs.clear()
        self.layer_masks.clear()
        self.net_actions.clear()
        self.heatmap_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs_net.clear()
        self.log_probs_heatmap.clear()
        self.advantages.clear()
        self.returns.clear()
        self.net_embs_cache.clear()
        self.unrouted_masks.clear()

    def compute_gae(self, last_value: float, last_done: bool, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.advantages = [0.0] * len(self.rewards)
        self.returns = [0.0] * len(self.rewards)
        
        last_gae_lam = 0.0
        val_next = last_value
        done_next = last_done
        
        for t in reversed(range(len(self.rewards))):
            val_curr = self.values[t]
            rew = self.rewards[t]
            non_terminal = 1.0 - float(self.dones[t])
            
            delta = rew + gamma * val_next * (1.0 - float(done_next)) - val_curr
            last_gae_lam = delta + gamma * gae_lambda * (1.0 - float(done_next)) * last_gae_lam
            
            self.advantages[t] = last_gae_lam
            self.returns[t] = self.advantages[t] + val_curr
            
            val_next = val_curr
            done_next = self.dones[t]


class PPOJEPATrainer:
    def __init__(
        self,
        config_path: str = 'configs/training.yaml',
        model_config_path: str = 'configs/model.yaml',
        curriculum_config_path: str = 'configs/curriculum.yaml',
        device: str = 'auto',
        checkpoint_dir: Optional[str] = None,
        load_checkpoint_path: Optional[str] = None
    ):
        # 1. Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using training device: {self.device}")
        
        # Load configs
        with open(config_path, 'r') as f:
            self.train_cfg = yaml.safe_load(f)
        with open(model_config_path, 'r') as f:
            self.model_cfg = yaml.safe_load(f)
            
        # 2. Init Curriculum Manager
        self.curriculum = CurriculumManager(curriculum_config_path)
        
        # 3. Create Gym Env
        self.env = PCBRoutingEnv(
            board_config=self.curriculum.get_board_config(),
            reward_weights=self.curriculum.get_reward_weights()
        )
        
        # 4. Init Models
        # ViT Encoder
        vit_cfg = self.model_cfg['vit']
        self.vit = ViTEncoder(
            image_channels=vit_cfg['image_channels'],
            patch_size=vit_cfg['patch_size'],
            embed_dim=vit_cfg['embed_dim'],
            num_heads=vit_cfg['num_heads'],
            num_layers=vit_cfg['num_layers'],
            mlp_ratio=vit_cfg['mlp_ratio'],
            dropout=vit_cfg['dropout'],
            max_grid_size=vit_cfg['max_grid_size']
        ).to(self.device)
        
        # GNN Encoder
        gnn_cfg = self.model_cfg['gnn']
        self.gnn = HeteroGATEncoder(
            hidden_dim=gnn_cfg['hidden_dim'],
            out_dim=gnn_cfg['out_dim'],
            num_layers=gnn_cfg['num_layers'],
            num_heads=gnn_cfg['num_heads'],
            dropout=gnn_cfg['dropout']
        ).to(self.device)
        
        # Cross Attention Fusion
        fus_cfg = self.model_cfg['fusion']
        self.fusion = CrossAttentionFusion(
            num_layers=fus_cfg['num_layers'],
            embed_dim=fus_cfg['embed_dim'],
            num_heads=fus_cfg['num_heads'],
            dropout=fus_cfg['dropout']
        ).to(self.device)
        
        # Spatial-JEPA
        jepa_cfg = self.model_cfg['jepa']
        self.jepa = SpatialJEPA(
            vit_encoder=self.vit, # target encoder is copy of self.vit
            predictor_layers=jepa_cfg['predictor_layers'],
            predictor_dim=jepa_cfg['predictor_dim'],
            predictor_heads=jepa_cfg['predictor_heads'],
            ema_decay=jepa_cfg['ema_decay'],
            vicreg_weight=jepa_cfg['vicreg_weight'],
            variance_weight=jepa_cfg['variance_weight'],
            invariance_weight=jepa_cfg['invariance_weight'],
            covariance_weight=jepa_cfg['covariance_weight']
        ).to(self.device)
        
        # Policy
        pol_cfg = self.model_cfg['policy']
        self.policy = PPOPolicy(
            embed_dim=vit_cfg['embed_dim'],
            net_selector_dim=pol_cfg['net_selector_dim'],
            heatmap_latent_dim=pol_cfg['heatmap_latent_dim'],
            value_hidden_dim=pol_cfg['value_hidden_dim']
        ).to(self.device)
        
        # Heatmap Decoder
        dec_cfg = self.model_cfg['heatmap_decoder']
        self.decoder = HeatmapDecoder(
            latent_dim=dec_cfg['latent_dim'],
            spatial_dim=vit_cfg['embed_dim'],
            max_layers=dec_cfg['max_layers']
        ).to(self.device)
        
        # 5. Optimizers
        self.optimizer = torch.optim.AdamW(
            list(self.vit.parameters()) +
            list(self.gnn.parameters()) +
            list(self.fusion.parameters()) +
            list(self.jepa.predictor_blocks.parameters()) +
            list(self.jepa.action_proj.parameters()) +
            list(self.policy.parameters()) +
            list(self.decoder.parameters()),
            lr=float(self.train_cfg['ppo']['learning_rate']),
            weight_decay=float(self.train_cfg['optimizer']['weight_decay'])
        )
        
        self.buffer = RolloutBuffer()
        self.total_timesteps = 0
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else self.train_cfg['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Metrics history for live plotting in Colab / external hooks
        self.metrics_history = {
            'timesteps': [],
            'completion_rate': [],
            'loss_policy': [],
            'loss_value': [],
            'loss_jepa': [],
            'stage': [],
        }
        
        self.last_heatmap = None
        self.last_net_idx = None
        
        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)

    def collect_rollouts(self, num_steps: int) -> float:
        import time
        self.buffer.clear()
        
        obs, info = self.env.reset()
        done = False
        
        eval_completion_rates = []
        
        # A* success rate tracking (Fix #3)
        astar_connected_count = 0
        astar_attempted_count = 0
        
        step_timings = {
            'obs': [], 'vit': [], 'gnn': [], 'fusion': [],
            'net_emb': [], 'policy': [], 'decoder': [], 'env': [],
            'astar': [], 'post': [], 'drc': [], 'graph': []
        }
        
        for step in range(num_steps):
            t_start = time.perf_counter()
            # Move observation to PyTorch
            raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 13, H, W)
            layer_mask = torch.tensor(obs['layer_mask'], dtype=torch.float32).unsqueeze(0).to(self.device)     # (1, 8)
            
            # GNN inputs
            graph = info['graph']
            # Convert PyG HeteroData dicts to device
            x_dict = {k: v.to(self.device) for k, v in graph.x_dict.items()} if hasattr(graph, 'x_dict') else {k: v['x'].to(self.device) for k, v in graph.items() if isinstance(v, dict) and 'x' in v}
            edge_index_dict = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()} if hasattr(graph, 'edge_index_dict') else {k: v.to(self.device) for k, v in graph.items() if isinstance(v, torch.Tensor) and v.shape[0] == 2}
            t_obs = time.perf_counter() - t_start
            
            with torch.no_grad():
                # 1. Spatial encoding
                t_vit_start = time.perf_counter()
                spatial_patches, cls_spatial = self.vit(raster_tensor) # (1, N_patches, 384)
                t_vit = time.perf_counter() - t_vit_start
                
                # 2. Graph encoding
                t_gnn_start = time.perf_counter()
                node_embs = self.gnn(x_dict, edge_index_dict)
                pad_embs = node_embs['pad'].unsqueeze(0) # add batch dim -> (1, N_pads, 384)
                t_gnn = time.perf_counter() - t_gnn_start
                
                # 3. Bidirectional Fusion
                t_fusion_start = time.perf_counter()
                fused_pads, fused_spatial = self.fusion(pad_embs, spatial_patches)
                t_fusion = time.perf_counter() - t_fusion_start
                
                # Group pad embeddings to net embeddings by average
                t_net_emb_start = time.perf_counter()
                num_nets = len(self.env.board.nets)
                max_nets = 100
                net_embs = torch.zeros((1, max_nets, self.vit.embed_dim), device=self.device)
                unrouted_mask = torch.zeros((1, max_nets), dtype=torch.bool, device=self.device)
                
                temp_net_embs = torch.zeros((num_nets, self.vit.embed_dim), device=self.device)
                for net_idx, net in enumerate(self.env.board.nets):
                    # Average pad embeddings belonging to this net
                    pin_indices = [idx for idx, p in enumerate(self.env.board.pins.values()) if p.net_id == net.id]
                    if pin_indices:
                        temp_net_embs[net_idx] = fused_pads[0, pin_indices].mean(dim=0)
                        
                net_embs[0, :num_nets] = temp_net_embs
                
                # Create mask for unrouted nets
                for net_idx, net in enumerate(self.env.board.nets):
                    if net.id not in self.env.routed_nets:
                        unrouted_mask[0, net_idx] = True
                t_net_emb = time.perf_counter() - t_net_emb_start
                        
                # 4. Policy forward
                t_policy_start = time.perf_counter()
                net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, value = self.policy(
                    net_embs, unrouted_mask, fused_spatial, cls_spatial
                )
                t_policy = time.perf_counter() - t_policy_start
                
                # 5. Decode heatmap
                t_decoder_start = time.perf_counter()
                heatmaps_via = self.decoder(
                    heatmap_latent, fused_spatial,
                    self.env.H, self.env.W, active_layers_mask=layer_mask
                ) # (1, 9, H, W)
                t_decoder = time.perf_counter() - t_decoder_start
                
            # Convert heatmaps to numpy for environment pathfinding
            heatmaps_np = heatmaps_via[0, :self.env.board.num_layers].cpu().numpy()
            via_prob_np = heatmaps_via[0, 8].cpu().numpy()
            
            # Step environment with actions
            t_env_start = time.perf_counter()
            next_obs, reward, terminated, truncated, next_info = self.env.step_with_heatmaps(
                net_idx.item(), heatmaps_np, via_prob_np
            )
            t_env = time.perf_counter() - t_env_start
            
            # Track A* connection success rate (Fix #3)
            astar_attempted_count += 1
            if next_info.get('connected', False):
                astar_connected_count += 1
            
            # Get selected net and start pin for visualization layer selection
            selected_net = self.env.board.nets[net_idx.item()] if net_idx.item() < len(self.env.board.nets) else None
            src_pin = self.env.board.pins[selected_net.pin_ids[0]] if selected_net else None
            
            # Record timings
            step_timings['obs'].append(t_obs)
            step_timings['vit'].append(t_vit)
            step_timings['gnn'].append(t_gnn)
            step_timings['fusion'].append(t_fusion)
            step_timings['net_emb'].append(t_net_emb)
            step_timings['policy'].append(t_policy)
            step_timings['decoder'].append(t_decoder)
            step_timings['env'].append(t_env)
            step_timings['astar'].append(next_info.get('time_astar', 0.0))
            step_timings['post'].append(next_info.get('time_post', 0.0))
            step_timings['drc'].append(next_info.get('time_drc', 0.0))
            step_timings['graph'].append(next_info.get('time_graph', 0.0))
            
            if step == 0 or (step + 1) % 10 == 0 or step == num_steps - 1:
                t_inf = (t_vit + t_gnn + t_fusion + t_net_emb + t_policy + t_decoder) * 1000
                t_as = next_info.get('time_astar', 0.0) * 1000
                t_po = next_info.get('time_post', 0.0) * 1000
                t_dr = next_info.get('time_drc', 0.0) * 1000
                t_gr = next_info.get('time_graph', 0.0) * 1000
                t_en = t_env * 1000
                # Running A* success rate at this point in the rollout
                astar_rate_so_far = astar_connected_count / max(astar_attempted_count, 1)
                connected_flag = '✓' if next_info.get('connected', False) else '✗'
                
                print(f"    [Rollout Step {step + 1}/{num_steps}] A*: {connected_flag} ({astar_connected_count}/{astar_attempted_count} = {astar_rate_so_far:.0%} connected) | Reward: {reward:+.3f}\n"
                      f"      +-- Model Inference: {t_inf:.1f}ms [ViT: {t_vit*1000:.1f}ms | GNN: {t_gnn*1000:.1f}ms | Fusion: {t_fusion*1000:.1f}ms | Policy: {t_policy*1000:.1f}ms | Dec: {t_decoder*1000:.1f}ms]\n"
                      f"      +-- Environment Step: {t_en:.1f}ms [A* Search: {t_as:.1f}ms | Post-Proc: {t_po:.1f}ms | DRC: {t_dr:.1f}ms | GraphUpdate: {t_gr:.1f}ms]")
                
                # Render and display comparison dashboard in Jupyter/Colab
                try:
                    from pcb_router.visualization.heatmap_viz import HeatmapVisualizer
                    import matplotlib.pyplot as plt
                    import IPython.display as ipydisplay
                    
                    viz = HeatmapVisualizer(theme_dark=True)
                    # Select the heatmap slice corresponding to starting pin layer
                    h_idx = src_pin.layer if (src_pin and src_pin.layer < len(heatmaps_np)) else 0
                    # Only deepcopy board state when visualization actually runs
                    board_before = copy.deepcopy(self.env.board_state)
                    fig = viz.render_routing_comparison(
                        board_before=board_before,
                        board_after=copy.deepcopy(self.env.board_state),
                        heatmap=heatmaps_np[h_idx],
                        path=next_info.get('path', [])
                    )
                    ipydisplay.display(fig)
                    plt.close(fig)
                except Exception as e:
                    pass
            
            done = terminated or truncated
            
            # Store in rollout buffer
            self.buffer.rasters.append(obs['board_raster'])
            self.buffer.graphs.append(graph)
            self.buffer.layer_masks.append(obs['layer_mask'])
            self.buffer.net_actions.append(net_idx.item())
            self.buffer.heatmap_actions.append(heatmap_latent.squeeze(0).cpu().numpy())
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            self.buffer.values.append(value.item())
            self.buffer.log_probs_net.append(log_prob_net.item())
            self.buffer.log_probs_heatmap.append(log_prob_heatmap.item())
            # Cache pre-computed net embeddings on CPU instead of deep-copying the full board.
            # net_embs shape: (1, max_nets, embed_dim) → squeeze to (max_nets, embed_dim)
            self.buffer.net_embs_cache.append(net_embs.squeeze(0).cpu())
            self.buffer.unrouted_masks.append(unrouted_mask.squeeze(0).cpu().numpy())
            
            self.total_timesteps += 1
            
            if done:
                # Record completion rate to curriculum
                self.curriculum.record_episode(next_info['completion_rate'], next_info['drc_violations'] / len(self.env.board.nets))
                eval_completion_rates.append(next_info['completion_rate'])
                obs, info = self.env.reset()
            else:
                obs = next_obs
                info = next_info
                
        # GAE Advantages computation
        self.buffer.compute_gae(
            last_value=value.item(),
            last_done=done,
            gamma=self.train_cfg['ppo']['gamma'],
            gae_lambda=self.train_cfg['ppo']['gae_lambda']
        )
        
        # Print summary of timings for this rollout collection
        m_inf = (np.mean(step_timings['vit']) + np.mean(step_timings['gnn']) + np.mean(step_timings['fusion']) + 
                 np.mean(step_timings['net_emb']) + np.mean(step_timings['policy']) + np.mean(step_timings['decoder'])) * 1000
        m_vit = np.mean(step_timings['vit']) * 1000
        m_gnn = np.mean(step_timings['gnn']) * 1000
        m_fusion = np.mean(step_timings['fusion']) * 1000
        m_policy = np.mean(step_timings['policy']) * 1000
        m_decoder = np.mean(step_timings['decoder']) * 1000
        
        m_env = np.mean(step_timings['env']) * 1000
        m_astar = np.mean(step_timings['astar']) * 1000
        m_post = np.mean(step_timings['post']) * 1000
        m_drc = np.mean(step_timings['drc']) * 1000
        m_graph = np.mean(step_timings['graph']) * 1000
        
        # A* success rate summary for the full rollout (Fix #3)
        astar_success_rate = astar_connected_count / max(astar_attempted_count, 1)
        astar_diagnosis = (
            "CRITICAL — A* almost never finds a path. Check env/pathfinding, not policy."
            if astar_success_rate < 0.05 else
            "LOW — pathfinding is struggling. Heatmap guidance may be inverted/uninformative."
            if astar_success_rate < 0.30 else
            "MODERATE — pathfinding works sometimes. Policy is learning to guide A*."
            if astar_success_rate < 0.70 else
            "GOOD — A* connects most of the time. Completion rate should be rising."
        )
        print(f"    [Rollout Collection Finished] Mean Timings across {num_steps} steps:\n"
              f"      +-- Model Inference: {m_inf:.1f}ms [ViT: {m_vit:.1f}ms | GNN: {m_gnn:.1f}ms | Fusion: {m_fusion:.1f}ms | Policy: {m_policy:.1f}ms | Dec: {m_decoder:.1f}ms]\n"
              f"      +-- Environment Step: {m_env:.1f}ms [A* Search: {m_astar:.1f}ms | Post-Proc: {m_post:.1f}ms | DRC: {m_drc:.1f}ms | GraphUpdate: {m_graph:.1f}ms]\n"
              f"      +-- A* Success Rate: {astar_connected_count}/{astar_attempted_count} ({astar_success_rate:.1%}) — {astar_diagnosis}")
        
        return np.mean(eval_completion_rates) if eval_completion_rates else 0.0

    def update(self) -> Dict[str, float]:
        """Runs PPO update and JEPA representation update"""
        import gc
        
        # Convert rollout buffer to tensors
        rasters = torch.tensor(np.array(self.buffer.rasters), dtype=torch.float32).to(self.device) # (T, 13, H, W)
        layer_masks = torch.tensor(np.array(self.buffer.layer_masks), dtype=torch.float32).to(self.device) # (T, 8)
        
        net_actions = torch.tensor(self.buffer.net_actions, dtype=torch.long).to(self.device)
        heatmap_actions = torch.tensor(np.array(self.buffer.heatmap_actions), dtype=torch.float32).to(self.device)
        
        old_log_probs_net = torch.tensor(self.buffer.log_probs_net, dtype=torch.float32).to(self.device)
        old_log_probs_heatmap = torch.tensor(self.buffer.log_probs_heatmap, dtype=torch.float32).to(self.device)
        
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(self.buffer.returns, dtype=torch.float32).to(self.device)
        
        # Load pre-computed net embeddings from buffer (no GNN+Fusion re-forward needed)
        # Shape: (T, max_nets, embed_dim) — cached as CPU tensors during rollout collection
        all_net_embs = torch.stack(self.buffer.net_embs_cache, dim=0).to(self.device)  # (T, max_nets, embed_dim)
        all_unrouted_masks = torch.tensor(
            np.array(self.buffer.unrouted_masks), dtype=torch.bool
        ).to(self.device)  # (T, max_nets)
        
        # Convert dones to a tensor on the correct device once
        dones = torch.tensor(self.buffer.dones, dtype=torch.bool, device=self.device)
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        num_epochs = self.train_cfg['ppo']['num_epochs']
        batch_size = self.train_cfg['ppo']['batch_size']
        T = len(self.buffer.rewards)
        
        # --- Pre-compute JEPA embeddings ONCE before epoch loop ---
        # Run ViT on rasters under no_grad in small chunks to avoid memory spikes / RAM OOM in Colab
        with torch.no_grad():
            chunk_size = 8
            
            all_z_curr_list = []
            for idx in range(0, T - 1, chunk_size):
                chunk = rasters[idx:min(idx + chunk_size, T - 1)]
                z_curr, _ = self.vit(chunk)
                all_z_curr_list.append(z_curr)
            all_z_curr_all = torch.cat(all_z_curr_list, dim=0) if all_z_curr_list else torch.empty((0, self.vit.embed_dim), device=self.device)
            
            all_z_next_target_list = []
            for idx in range(1, T, chunk_size):
                chunk = rasters[idx:min(idx + chunk_size, T)]
                z_next, _ = self.jepa.target_encoder(chunk)
                all_z_next_target_list.append(z_next)
            all_z_next_target_all = torch.cat(all_z_next_target_list, dim=0) if all_z_next_target_list else torch.empty((0, self.vit.embed_dim), device=self.device)
            
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        jepa_loss_epoch = 0.0
        num_updates = 0
        
        # PPO update epochs
        for epoch in range(num_epochs):
            # Create random minibatches on the device
            permutation = torch.randperm(T, device=self.device)
            for i in range(0, T, batch_size):
                indices = permutation[i:i+batch_size]
                if len(indices) < batch_size // 2:
                    continue
                
                b_rasters = rasters[indices]
                b_net_acts = net_actions[indices]
                b_heatmap_acts = heatmap_actions[indices]
                
                b_old_log_net = old_log_probs_net[indices]
                b_old_log_heat = old_log_probs_heatmap[indices]
                
                b_advantages = advantages[indices]
                b_returns = returns[indices]
                
                # Use cached net embeddings — no GNN/Fusion re-forward per sample
                b_net_embs = all_net_embs[indices]       # (B, max_nets, embed_dim)
                b_unrouted_mask = all_unrouted_masks[indices]  # (B, max_nets)
                
                # Run ViT forward in batch (needed for gradient flow through spatial features)
                b_spatial_patches, b_cls_spatial = self.vit(b_rasters) # (B, N_patches, embed_dim), (B, embed_dim)
                
                # Policy evaluation
                log_prob_net, log_prob_heatmap, value, ent_net, ent_heatmap = self.policy.evaluate_actions(
                    b_net_embs, b_unrouted_mask, b_spatial_patches, b_cls_spatial,
                    b_net_acts, b_heatmap_acts
                )
                
                # PPO Loss — Net selector
                ratio_net = torch.exp(log_prob_net - b_old_log_net)
                surr1_net = ratio_net * b_advantages
                surr2_net = torch.clamp(ratio_net, 1.0 - self.train_cfg['ppo']['clip_epsilon'], 1.0 + self.train_cfg['ppo']['clip_epsilon']) * b_advantages
                loss_policy_net = -torch.min(surr1_net, surr2_net).mean()
                
                # PPO Loss — Heatmap
                ratio_heat = torch.exp(log_prob_heatmap - b_old_log_heat)
                surr1_heat = ratio_heat * b_advantages
                surr2_heat = torch.clamp(ratio_heat, 1.0 - self.train_cfg['ppo']['clip_epsilon'], 1.0 + self.train_cfg['ppo']['clip_epsilon']) * b_advantages
                loss_policy_heat = -torch.min(surr1_heat, surr2_heat).mean()
                
                loss_policy = loss_policy_net + loss_policy_heat
                
                # Value Loss
                loss_val = F.mse_loss(value, b_returns)
                
                # ── Batched JEPA Loss ────────────────────────────────────
                # Map minibatch indices to the transition indices (0 to T-2 and not done)
                valid_idx_mask = (indices < T - 1) & (~dones[indices])
                jepa_indices = indices[valid_idx_mask].to(self.device)
                
                if len(jepa_indices) > 0:
                    b_z_curr = all_z_curr_all[jepa_indices]
                    b_z_next_target = all_z_next_target_all[jepa_indices]
                    b_act_net = net_actions[jepa_indices]
                    b_act_heat = heatmap_actions[jepa_indices]
                    
                    # Predict next state embedding in a single batch call
                    b_z_next_pred = self.jepa.predict(b_z_curr, (b_act_net, b_act_heat))
                    
                    # Batch VICReg losses
                    loss_inv = F.mse_loss(b_z_next_pred, b_z_next_target.detach())
                    loss_var = self.jepa.compute_variance_loss(b_z_next_pred) + self.jepa.compute_variance_loss(b_z_next_target.detach())
                    loss_cov = self.jepa.compute_covariance_loss(b_z_next_pred) + self.jepa.compute_covariance_loss(b_z_next_target.detach())
                    
                    loss_jepa = (
                        self.jepa.invariance_weight * loss_inv +
                        self.jepa.variance_weight * loss_var +
                        self.jepa.covariance_weight * loss_cov
                    )
                else:
                    loss_jepa = torch.tensor(0.0, device=self.device)
                
                # Total Combined Loss
                loss_total = (
                    loss_policy +
                    self.train_cfg['ppo']['value_loss_coef'] * loss_val -
                    self.train_cfg['ppo']['entropy_coef'] * (ent_net.mean() + ent_heatmap.mean()) +
                    self.train_cfg['jepa_loss']['prediction_weight'] * loss_jepa
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss_total.backward()
                # Clip gradients for all optimized parameters to prevent gradient explosion
                nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g['params']],
                    self.train_cfg['ppo']['max_grad_norm']
                )
                self.optimizer.step()
                
                # Update EMA target encoder weights
                self.jepa.update_target_weights(self.vit)
                
                policy_loss_epoch += loss_policy.item()
                value_loss_epoch += loss_val.item()
                jepa_loss_epoch += loss_jepa.item()
                num_updates += 1
        
        # Free pre-computed JEPA pairs and flush GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        denom = max(num_updates, 1)
        return {
            'loss_policy': policy_loss_epoch / denom,
            'loss_value': value_loss_epoch / denom,
            'loss_jepa': jepa_loss_epoch / denom
        }

    def train(self, total_timesteps: int, on_update=None):
        """
        Main training loop.
        Args:
            total_timesteps: Total env steps to train for.
            on_update: Optional callback called after each rollout+update cycle.
                       Signature: on_update(trainer, metrics_dict) -> None
                       Use this in Colab to inject live visualization between updates.
        """
        print("Starting GNN + JEPA PCB Router joint training...")
        progress_bar = tqdm(total=total_timesteps, desc="Training")
        
        while self.total_timesteps < total_timesteps:
            # 1. Collect Rollouts
            rollout_steps = self.train_cfg['ppo']['num_rollout_steps']
            mean_completion = self.collect_rollouts(rollout_steps)
            
            # 2. Run Updates
            metrics = self.update()
            
            # 3. Record metrics history
            self.metrics_history['timesteps'].append(self.total_timesteps)
            self.metrics_history['completion_rate'].append(mean_completion)
            self.metrics_history['loss_policy'].append(metrics['loss_policy'])
            self.metrics_history['loss_value'].append(metrics['loss_value'])
            self.metrics_history['loss_jepa'].append(metrics['loss_jepa'])
            self.metrics_history['stage'].append(self.curriculum.current_stage_name)
            
            # Print epoch logs
            print(f"[Step {self.total_timesteps}/{total_timesteps}] "
                  f"Stage: '{self.curriculum.current_stage_name}' | "
                  f"Board size: {self.env.W}x{self.env.H} | "
                  f"Completion rate: {mean_completion:.2f} | "
                  f"Loss Policy: {metrics['loss_policy']:.4f} | "
                  f"Loss JEPA: {metrics['loss_jepa']:.4f}")
            
            # Update progress bar
            progress_bar.n = self.total_timesteps
            progress_bar.set_postfix({
                'stage': self.curriculum.current_stage_name,
                'comp_rate': f"{mean_completion:.2f}",
                'loss_policy': f"{metrics['loss_policy']:.3f}",
                'loss_jepa': f"{metrics['loss_jepa']:.3f}"
            })
            progress_bar.refresh()
            
            # 4. Check Curriculum Advancement
            if self.curriculum.should_advance():
                self.curriculum.advance()
                # Update environment board size and components matching new stage
                self.env.reset()
                
            # 5. Checkpoint
            if self.total_timesteps % self.train_cfg['training']['save_interval'] == 0:
                self.save_checkpoint(f"{self.checkpoint_dir}/checkpoint_{self.total_timesteps}.pt")
                
            # 6. Visual Checkpoint (Every Rollout Update)
            self.save_visual_checkpoint(f"{self.checkpoint_dir}/visuals/step_{self.total_timesteps}.png")
            
            # 7. Optional external callback (for Colab live viz)
            if on_update is not None:
                on_update(self, {
                    'timesteps': self.total_timesteps,
                    'completion_rate': mean_completion,
                    **metrics
                })

    def save_checkpoint(self, path: str):
        state = {
            'vit': self.vit.state_dict(),
            'gnn': self.gnn.state_dict(),
            'fusion': self.fusion.state_dict(),
            'jepa': self.jepa.state_dict(),
            'policy': self.policy.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'curriculum': self.curriculum.get_state(),
            'total_timesteps': self.total_timesteps
        }
        
        # Save atomically to prevent file corruption if interrupted
        import tempfile
        temp_dir = os.path.dirname(path)
        if not temp_dir:
            temp_dir = "."
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".tmp")
        try:
            os.close(temp_fd)
            torch.save(state, temp_path)
            os.replace(temp_path, path)
            print(f"Checkpoint saved atomically to {path}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            # weights_only=False is used because checkpoints contain custom python structures (e.g. curriculum state and numpy multiarray scalars)
            try:
                state = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                state = torch.load(path, map_location=self.device)
            self.vit.load_state_dict(state['vit'])
            self.gnn.load_state_dict(state['gnn'])
            self.fusion.load_state_dict(state['fusion'])
            self.jepa.load_state_dict(state['jepa'])
            self.policy.load_state_dict(state['policy'])
            self.decoder.load_state_dict(state['decoder'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.curriculum.load_state(state['curriculum'])
            self.total_timesteps = state['total_timesteps']
            print(f"Checkpoint loaded successfully from {path} (Step {self.total_timesteps})")
        else:
            print(f"No checkpoint found at {path}")

    def save_visual_checkpoint(self, path: str):
        try:
            from pcb_router.visualization.renderer import BoardRenderer
            import matplotlib
            matplotlib.use('Agg') # Use non-interactive backend to prevent GUI issues
            import matplotlib.pyplot as plt
            
            renderer = BoardRenderer(theme_dark=True)
            # Render layout (showing all layers since it is multi-layer routing)
            fig = renderer.render_board(
                board_state=self.env.board_state,
                board=self.env.board,
                show_all_layers=True
            )
            
            # Add a title
            if len(fig.axes) > 0:
                fig.axes[0].set_title(
                    f"Step {self.total_timesteps} | Stage: {self.curriculum.current_stage_name}\n"
                    f"Mean Completion: {self.curriculum.completion_rate_ma:.2f}",
                    color='white', fontsize=12
                )
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            fig.clear() # Clear figure memory to prevent leaks
            print(f"Visual training snapshot saved to {path}")
            
            # If inside Colab or Jupyter, display it inline
            try:
                import IPython.display as ipydisplay
                ipydisplay.display(ipydisplay.Image(filename=path))
            except Exception as e:
                # Do not spam if IPython is not available, but log other issues
                pass
        except Exception as e:
            print(f"Warning: Failed to save visual checkpoint: {e}")


from pcb_router.models.jepa import JEPAWorldModel
from pcb_router.models.policy import DreamerActorCritic
from pcb_router.training.replay_buffer import ReplayBuffer, Episode
from collections import defaultdict
import random

def compute_lambda_returns(rewards, values, continues, bootstrap, gamma, lam):
    H, B = rewards.shape
    returns = torch.zeros_like(rewards)
    last_return = bootstrap
    for t in reversed(range(H)):
        returns[t] = rewards[t] + gamma * continues[t] * ((1.0 - lam) * values[t] + lam * last_return)
        last_return = returns[t]
    return returns

class DreamerJEPATrainer(PPOJEPATrainer):
    def __init__(
        self,
        config_path: str = 'configs/training.yaml',
        model_config_path: str = 'configs/model.yaml',
        curriculum_config_path: str = 'configs/curriculum.yaml',
        device: str = 'auto',
        checkpoint_dir: Optional[str] = None,
        load_checkpoint_path: Optional[str] = None
    ):
        super().__init__(
            config_path=config_path,
            model_config_path=model_config_path,
            curriculum_config_path=curriculum_config_path,
            device=device,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint_path=None
        )
        
        jepa_cfg = self.model_cfg.get('jepa', {})
        self.jepa = JEPAWorldModel(
            vit_encoder=self.vit,
            gnn_encoder=self.gnn,
            fusion=self.fusion,
            deterministic_size=512,
            stochastic_groups=32,
            stochastic_classes=32,
            ema_decay=jepa_cfg.get('ema_decay', 0.995)
        ).to(self.device)
        
        pol_cfg = self.model_cfg.get('policy', {})
        vit_cfg = self.model_cfg.get('vit', {})
        self.policy = DreamerActorCritic(
            h_dim=512,
            z_dim=1024,
            embed_dim=vit_cfg.get('embed_dim', 384),
            net_selector_dim=pol_cfg.get('net_selector_dim', 256),
            heatmap_latent_dim=pol_cfg.get('heatmap_latent_dim', 256),
            value_hidden_dim=pol_cfg.get('value_hidden_dim', 256)
        ).to(self.device)
        
        t_cfg = self.train_cfg.get('training', {})
        wm_lr = float(t_cfg.get('world_model_lr', 3e-4))
        actor_lr = float(t_cfg.get('actor_lr', 8e-5))
        critic_lr = float(t_cfg.get('critic_lr', 8e-5))
        weight_decay = float(t_cfg.get('wm_weight_decay', 1e-6))
        
        self.wm_opt = torch.optim.AdamW(
            list(self.vit.parameters()) +
            list(self.gnn.parameters()) +
            list(self.fusion.parameters()) +
            list(self.jepa.parameters()),
            lr=wm_lr,
            weight_decay=weight_decay
        )
        
        actor_params = (
            list(self.policy.state_proj.parameters()) +
            list(self.policy.net_scorer.parameters()) +
            list(self.policy.heatmap_mlp.parameters()) +
            list(self.policy.heatmap_mean.parameters()) +
            list(self.decoder.parameters()) +
            [self.policy.heatmap_log_std]
        )
        self.actor_opt = torch.optim.AdamW(actor_params, lr=actor_lr)
        self.critic_opt = torch.optim.AdamW(self.policy.value_head.parameters(), lr=critic_lr)
        
        self.use_amp = t_cfg.get('use_amp', True)
        self.scaler_wm = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_ac = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.replay_buffer = ReplayBuffer(capacity_episodes=t_cfg.get('replay_buffer_size', 5000))
        
        self.imagination_horizon_start = t_cfg.get('imagination_horizon_start', 5)
        self.imagination_horizon_end = t_cfg.get('imagination_horizon_end', 15)
        self.imagination_horizon_ramp_iters = t_cfg.get('imagination_horizon_ramp_iters', 20000)
        self.imagination_horizon = self.imagination_horizon_start
        
        self.entropy_coef_start = float(t_cfg.get('entropy_coef_start', 3e-3))
        self.entropy_coef_end = float(t_cfg.get('entropy_coef_end', 3e-4))
        self.entropy_coef_decay_iters = t_cfg.get('entropy_coef_decay_iters', 50000)
        
        self.real_steps_per_iteration = t_cfg.get('real_steps_per_iteration', 64)
        self.train_ratio = t_cfg.get('train_ratio', 100)
        self.imagine_batch_size = t_cfg.get('imagine_batch_size', 512)
        
        self.gamma = t_cfg.get('gamma', 0.997)
        self.lambda_ = t_cfg.get('lambda_', 0.95)
        
        # Optional torch.compile() for PyTorch 2.0+ GPU acceleration
        self.compile_models = t_cfg.get('compile_models', True)
        if self.compile_models and hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("Compiling world model and policy with torch.compile()...")
            try:
                self.jepa = torch.compile(self.jepa)
                self.policy = torch.compile(self.policy)
            except Exception as e:
                print(f"torch.compile failed (falling back to uncompiled execution): {e}")
        
        self.metrics_history = {
            'timesteps': [],
            'completion_rate': [],
            'loss_wm': [],
            'loss_actor': [],
            'loss_critic': [],
            'stage': [],
        }
        
        self.last_heatmap = None
        self.last_net_idx = None
        
        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)

    def save_checkpoint(self, path: str):
        state = {
            'vit': self.vit.state_dict(),
            'gnn': self.gnn.state_dict(),
            'fusion': self.fusion.state_dict(),
            'jepa': self.jepa.state_dict(),
            'policy': self.policy.state_dict(),
            'decoder': self.decoder.state_dict(),
            'wm_opt': self.wm_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'scaler_wm': self.scaler_wm.state_dict(),
            'scaler_ac': self.scaler_ac.state_dict(),
            'curriculum': self.curriculum.get_state(),
            'total_timesteps': self.total_timesteps
        }
        
        # Save atomically to prevent file corruption if interrupted
        import tempfile
        temp_dir = os.path.dirname(path)
        if not temp_dir:
            temp_dir = "."
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".tmp")
        try:
            os.close(temp_fd)
            torch.save(state, temp_path)
            os.replace(temp_path, path)
            print(f"Dreamer checkpoint saved atomically to {path}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                state = torch.load(path, map_location=self.device)
            self.vit.load_state_dict(state['vit'])
            self.gnn.load_state_dict(state['gnn'])
            self.fusion.load_state_dict(state['fusion'])
            self.jepa.load_state_dict(state['jepa'])
            self.policy.load_state_dict(state['policy'])
            self.decoder.load_state_dict(state['decoder'])
            if 'wm_opt' in state:
                self.wm_opt.load_state_dict(state['wm_opt'])
            if 'actor_opt' in state:
                self.actor_opt.load_state_dict(state['actor_opt'])
            if 'critic_opt' in state:
                self.critic_opt.load_state_dict(state['critic_opt'])
            if 'scaler_wm' in state:
                self.scaler_wm.load_state_dict(state['scaler_wm'])
            if 'scaler_ac' in state:
                self.scaler_ac.load_state_dict(state['scaler_ac'])
            self.curriculum.load_state(state['curriculum'])
            self.total_timesteps = state['total_timesteps']
            print(f"Dreamer checkpoint loaded successfully from {path} (Step {self.total_timesteps})")
        else:
            print(f"No checkpoint found at {path}")

    def _get_net_embeddings_and_mask(self, raster_tensor, x_dict, edge_index_dict):
        spatial_patches, cls_spatial = self.vit(raster_tensor)
        node_embs = self.gnn(x_dict, edge_index_dict)
        pad_embs = node_embs['pad'].unsqueeze(0)
        fused_pads, fused_spatial = self.fusion(pad_embs, spatial_patches)
        
        num_nets = len(self.env.board.nets)
        max_nets = 100
        net_embs = torch.zeros((1, max_nets, self.vit.embed_dim), device=self.device)
        unrouted_mask = torch.zeros((1, max_nets), dtype=torch.bool, device=self.device)
        
        temp_net_embs = torch.zeros((num_nets, self.vit.embed_dim), device=self.device)
        for net_idx, net in enumerate(self.env.board.nets):
            pin_indices = [idx for idx, p in enumerate(self.env.board.pins.values()) if p.net_id == net.id]
            if pin_indices:
                temp_net_embs[net_idx] = fused_pads[0, pin_indices].mean(dim=0)
        net_embs[0, :num_nets] = temp_net_embs
        
        for net_idx, net in enumerate(self.env.board.nets):
            if net.id not in self.env.routed_nets:
                unrouted_mask[0, net_idx] = True
                
        return net_embs, unrouted_mask, fused_spatial

    def _phase1_collect_real(self, num_steps: int, explore: bool = True) -> float:
        self.vit.train()
        self.gnn.train()
        self.fusion.train()
        self.jepa.train()
        self.policy.train()
        self.decoder.train()
        
        steps_collected = 0
        completion_rates = []
        
        while steps_collected < num_steps:
            obs, info = self.env.reset()
            episode = Episode()
            h, z = self.jepa.initial_state(batch_size=1, device=self.device)
            done = False
            
            while not done and steps_collected < num_steps:
                raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                graph = info['graph']
                x_dict = {k: v.to(self.device) for k, v in graph.x_dict.items()}
                edge_index_dict = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()}
                layer_mask = torch.tensor(obs['layer_mask'], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 1. Forward pass WITH gradients enabled for supervised path training
                context_emb = self.jepa.get_context_embedding(raster_tensor, x_dict, edge_index_dict, use_target=False)
                target_context_emb = self.jepa.get_context_embedding(raster_tensor, x_dict, edge_index_dict, use_target=True)
                net_embs, unrouted_mask, fused_spatial = self._get_net_embeddings_and_mask(raster_tensor, x_dict, edge_index_dict)
                
                # Run policy forward to select net and sample heatmap_latent (differentiable rsample)
                net_idx_tensor, log_prob_net, ent_net = self.policy.select_net(net_embs, unrouted_mask, h, z, deterministic=not explore)
                
                selected_net_emb = net_embs[0, net_idx_tensor.item()].unsqueeze(0)
                state = torch.cat([h, z], dim=-1)
                x_feat = torch.cat([selected_net_emb, state], dim=-1)
                h_feat = self.policy.heatmap_mlp(x_feat)
                mean = self.policy.heatmap_mean(h_feat)
                log_std = torch.clamp(self.policy.heatmap_log_std, min=-20.0, max=2.0).expand_as(mean)
                std = torch.exp(log_std)
                
                dist = torch.distributions.Normal(mean, std)
                if not explore:
                    heatmap_latent = mean
                else:
                    heatmap_latent = dist.rsample()  # Differentiable path!
                
                # Decode heatmap using current fused_spatial
                heatmaps_via = self.decoder(
                    heatmap_latent, fused_spatial,
                    self.env.H, self.env.W, active_layers_mask=layer_mask
                )
                
                # Step the environment using detached numpy arrays
                heatmaps_np = heatmaps_via[0, :self.env.board.num_layers].detach().cpu().numpy()
                via_prob_np = heatmaps_via[0, 8].detach().cpu().numpy()
                
                self.last_heatmap = heatmaps_np
                self.last_net_idx = net_idx_tensor.item()
                
                next_obs, reward, terminated, truncated, next_info = self.env.step_with_heatmaps(
                    net_idx_tensor.item(), heatmaps_np, via_prob_np
                )
                done = terminated or truncated
                
                # Supervised update for decoder and encoders on successful paths
                if next_info.get('connected', False) and 'path' in next_info and len(next_info['path']) > 1:
                    all_routed_path = next_info['path']
                    # target has same layers as env + 1 (for via)
                    target_heatmap = torch.zeros((self.env.board.num_layers + 1, self.env.H, self.env.W), device=self.device)
                    for idx, wp in enumerate(all_routed_path):
                        wx, wy, wl = wp
                        if 0 <= wx < self.env.W and 0 <= wy < self.env.H and 0 <= wl < self.env.board.num_layers:
                            target_heatmap[wl, wy, wx] = 1.0
                            # Mark via
                            if idx > 0 and all_routed_path[idx-1][2] != wl:
                                target_heatmap[-1, wy, wx] = 1.0
                                
                    # Match channels: pred has shape (9, H, W). We take layers 0..num_layers, and layer 8 (via map)
                    pred_layers = heatmaps_via[0, :self.env.board.num_layers]
                    pred_via = heatmaps_via[0, 8:9]
                    pred_selected = torch.cat([pred_layers, pred_via], dim=0)
                    
                    # Use weighted BCE to handle the massive class imbalance of sparse path pixels
                    bce_loss = F.binary_cross_entropy(pred_selected, target_heatmap, reduction='none')
                    weight_mask = torch.where(target_heatmap > 0, torch.tensor(50.0, device=self.device), torch.tensor(1.0, device=self.device))
                    loss_dec = (bce_loss * weight_mask).mean()
                    
                    self.actor_opt.zero_grad(set_to_none=True)
                    self.wm_opt.zero_grad(set_to_none=True)
                    
                    self.scaler_ac.scale(loss_dec).backward()
                    
                    self.scaler_ac.step(self.actor_opt)
                    self.scaler_ac.step(self.wm_opt)  # Backprop to vit/gnn/fusion
                    self.scaler_ac.update()
                
                # Detached RSSM update and Replay Buffer appending
                with torch.no_grad():
                    action_tuple = (net_idx_tensor.squeeze(0).cpu(), heatmap_latent.squeeze(0).cpu())
                    action_emb = self.jepa.get_action_embedding(net_idx_tensor, heatmap_latent.detach())
                    h, z, _, _ = self.jepa.rssm_step(h, z, context_emb.detach(), action_emb)
                    
                    if not hasattr(episode, 'target_context_embeddings'):
                        episode.target_context_embeddings = []
                    if not hasattr(episode, 'net_embeddings_list'):
                        episode.net_embeddings_list = []
                    if not hasattr(episode, 'unrouted_masks_list'):
                        episode.unrouted_masks_list = []
                        
                    episode.append(context_emb.squeeze(0).cpu(), action_tuple, reward, done)
                    episode.target_context_embeddings.append(target_context_emb.squeeze(0).cpu())
                    episode.net_embeddings_list.append(net_embs.squeeze(0).cpu())
                    episode.unrouted_masks_list.append(unrouted_mask.squeeze(0).cpu())
                
                obs = next_obs
                info = next_info
                steps_collected += 1
                self.total_timesteps += 1
                if steps_collected % 5 == 0 or steps_collected == num_steps:
                    print(f"  Collected {steps_collected}/{num_steps} steps...")
                
            if episode.length > 0:
                episode.net_embeddings = episode.net_embeddings_list[0]
                episode.unrouted_masks = episode.unrouted_masks_list
                self.replay_buffer.add_episode(episode)
                completion_rates.append(info.get('completion_rate', 0.0))
                
        return np.mean(completion_rates) if completion_rates else 0.0

    def _phase2_train_world_model(self):
        self.jepa.train()
        self.vit.train()
        self.gnn.train()
        self.fusion.train()
        
        batch_size = self.train_cfg.get('training', {}).get('batch_size', 64)
        seq_len = self.train_cfg.get('training', {}).get('seq_len', 50)
        
        sampled_episodes = random.choices(self.replay_buffer.episodes, k=batch_size)
        
        b_ctx = []
        b_tgt_ctx = []
        b_net = []
        b_heat = []
        b_rew = []
        b_cont = []
        b_mask = []
        
        for ep in sampled_episodes:
            if ep.length >= seq_len:
                start = random.randint(0, ep.length - seq_len)
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
            
            ctx_tensor = torch.stack(ctx_slice)
            tgt_tensor = torch.stack(tgt_slice)
            net_tensor = torch.stack(net_act)
            heat_tensor = torch.stack(heat_act)
            rew_tensor = torch.tensor(rew_slice, dtype=torch.float32)
            cont_tensor = 1.0 - torch.tensor(done_slice, dtype=torch.float32)
            mask_tensor = torch.ones(len(ctx_slice), dtype=torch.float32)
            
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
            
        batch = {
            'context_embeddings': torch.stack(b_ctx).to(self.device),
            'target_context_embeddings': torch.stack(b_tgt_ctx).to(self.device),
            'net_actions': torch.stack(b_net).to(self.device),
            'heatmap_actions': torch.stack(b_heat).to(self.device),
            'rewards': torch.stack(b_rew).to(self.device),
            'continues': torch.stack(b_cont).to(self.device),
            'masks': torch.stack(b_mask).to(self.device)
        }
        
        grad_clip = self.train_cfg.get('training', {}).get('wm_grad_clip', 100.0)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            losses = self.jepa.compute_loss(batch)
            total_loss = (
                self.jepa.invariance_weight * losses['loss_pred'] +
                self.jepa.variance_weight * losses['loss_variance'] +
                self.jepa.covariance_weight * losses['loss_covariance'] +
                losses['loss_kl'] +
                losses['loss_reward'] +
                losses['loss_continue']
            )

        self.wm_opt.zero_grad(set_to_none=True)
        self.scaler_wm.scale(total_loss).backward()
        self.scaler_wm.unscale_(self.wm_opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), grad_clip)
        self.scaler_wm.step(self.wm_opt)
        self.scaler_wm.update()
        
        self.jepa.update_target_weights()
        
        with torch.no_grad():
            B_b, T_b = batch['context_embeddings'].shape[0], batch['context_embeddings'].shape[1]
            h, z = self.jepa.initial_state(B_b, self.device)
            flat_net = batch['net_actions'].reshape(-1)
            flat_heat = batch['heatmap_actions'].reshape(-1, batch['heatmap_actions'].shape[-1])
            action_embs = self.jepa.get_action_embedding(flat_net, flat_heat).reshape(B_b, T_b, -1)
            
            all_h, all_z = [], []
            for t in range(T_b - 1):
                h, z, _, _ = self.jepa.rssm_step(h, z, batch['context_embeddings'][:, t], action_embs[:, t])
                all_h.append(h)
                all_z.append(z)
            if all_h:
                self.replay_buffer.cache_latents(torch.stack(all_h, dim=1), torch.stack(all_z, dim=1))
                
        return {
            'loss_wm': total_loss.item(),
            'loss_wm_pred': losses['loss_pred'].item(),
            'loss_wm_kl': losses['loss_kl'].item(),
            'loss_wm_reward': losses['loss_reward'].item(),
            'loss_wm_continue': losses['loss_continue'].item(),
            'wm_grad_norm': grad_norm.item()
        }

    def _phase3_train_actor_critic(self):
        self.jepa.eval()
        self.vit.eval()
        self.gnn.eval()
        self.fusion.eval()
        
        for p in self.jepa.parameters():
            p.requires_grad = False
            
        metrics = defaultdict(list)
        
        for update_step in range(self.train_ratio):
            init_states = self.replay_buffer.sample_latents(self.imagine_batch_size, device=self.device)
            
            if init_states is None:
                h0, z0 = self.jepa.initial_state(self.imagine_batch_size, device=self.device)
            else:
                h0, z0 = init_states['h'], init_states['z']
                
            h0 = h0.detach()
            z0 = z0.detach()
            
            sampled_episodes = random.choices(self.replay_buffer.episodes, k=self.imagine_batch_size)
            
            net_embs_list = []
            unrouted_mask_list = []
            for ep in sampled_episodes:
                net_embs_list.append(ep.net_embeddings)
                idx = random.randint(0, ep.length - 1)
                unrouted_mask_list.append(ep.unrouted_masks[idx])
                
            net_embeddings = torch.stack(net_embs_list).to(self.device)
            unrouted_mask = torch.stack(unrouted_mask_list).to(self.device)
            
            h, z = h0, z0
            traj_h = []
            traj_z = []
            traj_actions_net = []
            traj_actions_heat = []
            traj_rewards = []
            traj_continues = []
            traj_values = []
            traj_log_probs_net = []
            traj_log_probs_heat = []
            
            current_horizon = self._current_imagination_horizon()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                for t in range(current_horizon):
                    net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, value = self.policy(
                        net_embeddings, unrouted_mask, h, z, deterministic=False
                    )
                    
                    pred_reward = self.jepa.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)
                    pred_continue_logits = self.jepa.continue_head(torch.cat([h, z], dim=-1)).squeeze(-1)
                    pred_continue = torch.sigmoid(pred_continue_logits)
                    
                    traj_h.append(h)
                    traj_z.append(z)
                    traj_actions_net.append(net_idx)
                    traj_actions_heat.append(heatmap_latent)
                    traj_rewards.append(pred_reward)
                    traj_continues.append(pred_continue)
                    traj_values.append(value)
                    traj_log_probs_net.append(log_prob_net)
                    traj_log_probs_heat.append(log_prob_heatmap)
                    
                    action_emb = self.jepa.get_action_embedding(net_idx, heatmap_latent)
                    h, z = self.jepa.predict_step(h, z, action_emb)

                    # Out-of-place mask update: mark the chosen net as routed.
                    # scatter_() is inplace and corrupts autograd version counters when
                    # the tensor participates in the backward graph — use scatter() instead.
                    update = torch.zeros_like(unrouted_mask, dtype=torch.bool)
                    update = update.scatter(1, net_idx.unsqueeze(-1), True)
                    unrouted_mask = unrouted_mask & ~update
                    
                bootstrap_value = self.policy.get_value(h, z)
                
                traj_h = torch.stack(traj_h, dim=0)
                traj_z = torch.stack(traj_z, dim=0)
                traj_rewards = torch.stack(traj_rewards, dim=0)
                traj_continues = torch.stack(traj_continues, dim=0)
                traj_values = torch.stack(traj_values, dim=0)
                traj_log_probs_net = torch.stack(traj_log_probs_net, dim=0)
                traj_log_probs_heat = torch.stack(traj_log_probs_heat, dim=0)
                
                lambda_returns = compute_lambda_returns(
                    rewards=traj_rewards,
                    values=traj_values,
                    continues=traj_continues,
                    bootstrap=bootstrap_value,
                    gamma=self.gamma,
                    lam=self.lambda_
                )
                
                targets = lambda_returns.detach()
            
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                critic_loss = F.mse_loss(traj_values, targets)
                advantages = (targets - traj_values).detach()
                loss_policy = -(traj_log_probs_net + traj_log_probs_heat) * advantages
                loss_policy = loss_policy.mean()
                total_loss = loss_policy + critic_loss
            
            self.scaler_ac.scale(total_loss).backward()
            self.scaler_ac.unscale_(self.actor_opt)
            self.scaler_ac.unscale_(self.critic_opt)
            
            grad_norm_crit = torch.nn.utils.clip_grad_norm_(self.policy.value_head.parameters(), 100.0)
            
            all_actor_params = []
            for group in self.actor_opt.param_groups:
                all_actor_params.extend(group['params'])
            grad_norm_act = torch.nn.utils.clip_grad_norm_(all_actor_params, 100.0)
            
            self.scaler_ac.step(self.actor_opt)
            self.scaler_ac.step(self.critic_opt)
            self.scaler_ac.update()
            
            self.policy.update_target_critic(ema_decay=self.train_cfg.get('training', {}).get('critic_target_ema', 0.98))
            
            metrics['loss_actor'].append(loss_policy.item())
            metrics['loss_critic'].append(critic_loss.item())
            metrics['imagined_return_mean'].append(targets.mean().item())
            metrics['actor_grad_norm'].append(grad_norm_act.item())
            metrics['critic_grad_norm'].append(grad_norm_crit.item())
            
        for p in self.jepa.parameters():
            p.requires_grad = True
            
        return {
            'loss_actor': np.mean(metrics['loss_actor']),
            'loss_critic': np.mean(metrics['loss_critic']),
            'imagined_return_mean': np.mean(metrics['imagined_return_mean']),
            'actor_grad_norm': np.mean(metrics['actor_grad_norm']),
            'critic_grad_norm': np.mean(metrics['critic_grad_norm'])
        }

    def _current_entropy_coef(self) -> float:
        if self.total_timesteps >= self.entropy_coef_decay_iters:
            return self.entropy_coef_end
        frac = self.total_timesteps / self.entropy_coef_decay_iters
        return self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start)

    def _current_imagination_horizon(self) -> int:
        if self.total_timesteps >= self.imagination_horizon_ramp_iters:
            return self.imagination_horizon_end
        frac = self.total_timesteps / self.imagination_horizon_ramp_iters
        return int(self.imagination_horizon_start + frac * (self.imagination_horizon_end - self.imagination_horizon_start))

    def train(self, total_timesteps: int, on_update=None):
        print("Starting GNN + JEPA DreamerV3-style training...")
        progress_bar = tqdm(total=total_timesteps, desc="Training")
        
        if len(self.replay_buffer) < 5:
            print("Collecting initial warmup episodes for replay buffer...")
            self._phase1_collect_real(num_steps=64, explore=True)
            
        while self.total_timesteps < total_timesteps:
            mean_completion = self._phase1_collect_real(num_steps=self.real_steps_per_iteration, explore=True)
            wm_metrics = self._phase2_train_world_model()
            ac_metrics = self._phase3_train_actor_critic()
            
            self.metrics_history['timesteps'].append(self.total_timesteps)
            self.metrics_history['completion_rate'].append(mean_completion)
            self.metrics_history['loss_wm'].append(wm_metrics['loss_wm'])
            self.metrics_history['loss_actor'].append(ac_metrics['loss_actor'])
            self.metrics_history['loss_critic'].append(ac_metrics['loss_critic'])
            self.metrics_history['stage'].append(self.curriculum.current_stage_name)
            
            print(f"[Step {self.total_timesteps}/{total_timesteps}] "
                  f"Stage: '{self.curriculum.current_stage_name}' | "
                  f"Completion: {mean_completion:.2f} | "
                  f"Loss WM: {wm_metrics['loss_wm']:.4f} | "
                  f"Loss Actor: {ac_metrics['loss_actor']:.4f} | "
                  f"Loss Critic: {ac_metrics['loss_critic']:.4f}")
            
            progress_bar.n = self.total_timesteps
            progress_bar.set_postfix({
                'stage': self.curriculum.current_stage_name,
                'comp_rate': f"{mean_completion:.2f}",
                'loss_wm': f"{wm_metrics['loss_wm']:.3f}",
                'loss_actor': f"{ac_metrics['loss_actor']:.3f}"
            })
            progress_bar.refresh()
            
            self.curriculum.completion_history.append(mean_completion)
            self.curriculum.episodes_in_stage += 1
            
            if self.curriculum.should_advance():
                self.curriculum.advance()
                self.env.reset()
                
            save_interval = self.train_cfg.get('training', {}).get('save_interval', 50000)
            if self.total_timesteps % save_interval < self.real_steps_per_iteration:
                self.save_checkpoint(f"{self.checkpoint_dir}/checkpoint_{self.total_timesteps}.pt")
                
            self.save_visual_checkpoint(f"{self.checkpoint_dir}/visuals/step_{self.total_timesteps}.png")
            
            if on_update is not None:
                on_update(self, {
                    'timesteps': self.total_timesteps,
                    'completion_rate': mean_completion,
                    **wm_metrics,
                    **ac_metrics
                })

