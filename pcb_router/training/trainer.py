import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

from pcb_router.models.vit_encoder import ViTEncoder
from pcb_router.models.gnn_encoder import HeteroGATEncoder
from pcb_router.models.fusion import CrossAttentionFusion
from pcb_router.models.heatmap_decoder import HeatmapDecoder

from pcb_router.env.pcb_env import PCBRoutingEnv
from pcb_router.training.curriculum import CurriculumManager
from pcb_router.training.rewards import RewardCalculator
from pcb_router.routing.obstacle_maps import build_obstacle_maps, build_via_blocked_maps

class RolloutBuffer:
    def __init__(self):
        self.rasters = []
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

class BaseRoutingTrainer:
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
        
        # Cache for last completed board state (for visualization)
        self.last_completed_board_state = None
        self.last_completed_board = None
        
        # 3. Create Gym Env
        self.env = PCBRoutingEnv(
            board_config=self.curriculum.get_board_config(),
            curriculum_stage=self.curriculum.current_stage,
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
        
        # Heatmap Decoder
        dec_cfg = self.model_cfg['heatmap_decoder']
        self.decoder = HeatmapDecoder(
            latent_dim=dec_cfg['latent_dim'],
            spatial_dim=vit_cfg['embed_dim'],
            max_layers=dec_cfg['max_layers']
        ).to(self.device)

        self.total_timesteps = 0
        self.last_action_probs = None
        self.current_net_values = None
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else self.train_cfg['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Cache pin→net mapping (built once per episode, invalidated on reset)
        self._pin_to_net_idx: Optional[torch.Tensor] = None
        self._pin_to_net_valid: bool = False

        # Auto-detect Jupyter / Colab: enable inline heatmap plots when running inside a
        # kernel (interactive), disable in headless script runs to avoid deepcopy overhead.
        try:
            from IPython import get_ipython
            _ip = get_ipython()
            self.enable_viz: bool = (_ip is not None and hasattr(_ip, 'kernel'))
        except Exception:
            self.enable_viz: bool = False

        # Metrics history for live plotting in Colab / external hooks
        self.metrics_history = {
            'timesteps': [],
            'completion_rate': [],
            'stage': [],
        }

        self.last_heatmap = None
        self.last_net_idx = None
        self.all_episode_heatmaps = []  # list of {'net_name', 'net_idx', 'heatmaps_np'} per net in last episode

        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)

    def _build_pin_to_net_idx(self) -> torch.Tensor:
        """Build a (num_pads,) int64 tensor mapping each pad index to its net index.
        Cached per episode; call _invalidate_pin_cache() on env.reset()."""
        num_nets = len(self.env.board.nets)
        net_id_to_idx = {net.id: idx for idx, net in enumerate(self.env.board.nets)}
        pins_list = list(self.env.board.pins.values())
        pin_to_net = torch.zeros(len(pins_list), dtype=torch.long, device=self.device)
        for pad_idx, p in enumerate(pins_list):
            net_idx = net_id_to_idx.get(p.net_id, 0)
            pin_to_net[pad_idx] = net_idx
        return pin_to_net

    def _invalidate_pin_cache(self):
        self._pin_to_net_valid = False
        self._pin_to_net_idx = None

    def _get_net_embs_vectorized(
        self,
        fused_pads: torch.Tensor,  # (1, num_pads, embed_dim)
        num_nets: int,
    ) -> torch.Tensor:
        """Aggregate pad embeddings into per-net embeddings via scatter_add (no Python loop)."""
        if not self._pin_to_net_valid or self._pin_to_net_idx is None:
            self._pin_to_net_idx = self._build_pin_to_net_idx()
            self._pin_to_net_valid = True

        embed_dim = fused_pads.shape[-1]
        pads = fused_pads[0]  # (num_pads, embed_dim)
        pin_to_net = self._pin_to_net_idx  # (num_pads,)

        # Accumulate sum of pad embeddings per net
        net_sum = torch.zeros(num_nets, embed_dim, device=self.device)
        net_sum.scatter_add_(0, pin_to_net.unsqueeze(1).expand_as(pads), pads)

        # Count pads per net for averaging
        pad_counts = torch.zeros(num_nets, device=self.device)
        pad_counts.scatter_add_(0, pin_to_net, torch.ones(len(pin_to_net), device=self.device))
        pad_counts = pad_counts.clamp(min=1.0)

        net_mean = net_sum / pad_counts.unsqueeze(1)  # (num_nets, embed_dim)
        return net_mean

    @staticmethod
    def _unwrap_compiled(module):
        """Return the underlying nn.Module even if it has been wrapped by torch.compile().
        torch.compile stores the original module in ._orig_mod. Without unwrapping,
        calling state_dict() on a doubly-compiled model raises AttributeError."""
        while hasattr(module, '_orig_mod'):
            module = module._orig_mod
        return module

    def _safe_load(self, module, ckpt_state):
        # Unwrap torch.compile wrapper so state_dict() always works
        module = self._unwrap_compiled(module)
        cleaned = {}
        model_keys = set(module.state_dict().keys())
        for k, v in ckpt_state.items():
            new_key = k
            if k.startswith('_orig_mod.') and k not in model_keys:
                stripped = k[10:] # len('_orig_mod.') is 10
                if stripped in model_keys:
                    new_key = stripped
            elif not k.startswith('_orig_mod.') and k not in model_keys:
                prepended = '_orig_mod.' + k
                if prepended in model_keys:
                    new_key = prepended
            cleaned[new_key] = v
        module.load_state_dict(cleaned)

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

class DreamerJEPATrainer(BaseRoutingTrainer):
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
        self.routing_mode = self.train_cfg.get('training', {}).get('routing_mode', 'astar_guided')
        self.env.routing_mode = self.routing_mode
        
        pol_cfg = self.model_cfg.get('policy', {})
        self.jepa = JEPAWorldModel(
            vit_encoder=self.vit,
            gnn_encoder=self.gnn,
            fusion=self.fusion,
            deterministic_size=512,
            stochastic_groups=32,
            stochastic_classes=32,
            heatmap_latent_dim=pol_cfg.get('heatmap_latent_dim', 256),
            ema_decay=jepa_cfg.get('ema_decay', 0.995)
        ).to(self.device)
        
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
        
        if self.routing_mode == 'autoregressive':
            actor_params_raw = (
                list(self.policy.state_proj.parameters()) +
                list(self.policy.net_scorer.parameters()) +
                list(self.policy.step_policy.parameters())
            )
        else:
            actor_params_raw = (
                list(self.policy.state_proj.parameters()) +
                list(self.policy.net_scorer.parameters()) +
                list(self.policy.heatmap_mlp.parameters()) +
                list(self.policy.heatmap_mean.parameters()) +
                list(self.decoder.parameters()) +
                [self.policy.heatmap_log_std]
            )
        
        # Remove duplicates while preserving order
        actor_params = list(dict.fromkeys(actor_params_raw))
        
        self.actor_opt = torch.optim.AdamW(actor_params, lr=actor_lr, fused=torch.cuda.is_available())
        self.critic_opt = torch.optim.AdamW(self.policy.value_head.parameters(), lr=critic_lr, fused=torch.cuda.is_available())
        
        self.use_amp = t_cfg.get('use_amp', True)
        self.scaler_wm = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_ac = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        self.replay_buffer = ReplayBuffer(capacity_episodes=t_cfg.get('replay_buffer_size', 5000))
        self.replay_buffer.latent_cache_capacity = t_cfg.get('latent_cache_capacity', 10000)
        
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
        
        # Optional torch.compile() for PyTorch 2.0+ GPU acceleration.
        # Only compile jepa and policy — they have fully static shapes (fixed latent dims)
        # and are called the most (train_ratio x imagination_horizon times per iteration).
        #
        # Models intentionally excluded from compilation:
        #  - vit: interpolate_pos_encoding() calls F.interpolate with dynamic (H,W) per
        #         curriculum stage → triggers expensive recompilation on stage transitions.
        #  - decoder: forward() takes env.H, env.W as args → same dynamic shape issue.
        #  - gnn: PyG HeteroConv has dynamic node/edge counts per board → cannot compile.
        #  - fusion: cross-attention over dynamic N_nodes → cannot compile.
        self.compile_models = t_cfg.get('compile_models', True)
        if self.compile_models and hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("Compiling world model with torch.compile()...")
            try:
                self.jepa = torch.compile(self.jepa)
            except Exception as e:
                print(f"torch.compile failed (falling back to uncompiled execution): {e}")
        
        self.metrics_history = {
            'timesteps': [],
            'completion_rate': [],
            'loss_wm': [],
            'loss_wm_reward': [],
            'loss_actor': [],
            'loss_critic': [],
            'entropy': [],
            'mean_dist_delta': [],
            'stage': [],
        }
        
        self.last_heatmap = None
        self.last_net_idx = None
        self.all_episode_heatmaps = []  # list of {'net_name', 'net_idx', 'heatmaps_np'} per net in last episode
        
        self.autoregressive_fallback_count = 0
        self.autoregressive_total_count = 0
        
        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)
        elif self.routing_mode == 'autoregressive':
            bc_path = os.path.join(self.checkpoint_dir, "bc_pretrained_policy.pt")
            if os.path.exists(bc_path):
                print(f"Starting fresh training run. Loading pretrained BC policy weights from {bc_path}...")
                try:
                    ckpt = torch.load(bc_path, map_location=self.device)
                    policy_state = ckpt.get('policy', ckpt)
                    self.policy.step_policy.load_state_dict(policy_state)
                    
                    # Also load the frozen visual and graph encoders!
                    if 'vit' in ckpt: self.vit.load_state_dict(ckpt['vit'])
                    if 'gnn' in ckpt: self.gnn.load_state_dict(ckpt['gnn'])
                    if 'fusion' in ckpt: self.fusion.load_state_dict(ckpt['fusion'])
                    
                    print("Pretrained BC weights (Policy + Encoders) successfully loaded! ✓")
                except Exception as e:
                    print(f"Warning: Failed to load pretrained BC weights: {e}")
            else:
                raise FileNotFoundError(
                    f"Pretrained BC policy checkpoint not found at: {bc_path}\n"
                    "In autoregressive routing mode, you MUST run CELL 4b successfully first to pretrain the step policy.\n"
                    "Starting RL training from random initialization is blocked to prevent wasting training time."
                )

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
            self._safe_load(self.vit, state['vit'])
            self._safe_load(self.gnn, state['gnn'])
            self._safe_load(self.fusion, state['fusion'])
            self._safe_load(self.jepa, state['jepa'])
            self._safe_load(self.policy, state['policy'])
            self._safe_load(self.decoder, state['decoder'])
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
            self.env.curriculum_stage = self.curriculum.current_stage
            self.env.reset(options={'board_config': self.curriculum.get_board_config()})
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
        dist_deltas = []
        
        if self.routing_mode == 'heatmap':
            while steps_collected < num_steps:
                obs, info = self.env.reset(options={'board_config': self.curriculum.get_board_config()})
                episode = Episode()
                h, z = self.jepa.initial_state(batch_size=1, device=self.device)
                done = False
                self.all_episode_heatmaps = []  # reset per-episode heatmap log
    
                
                while not done and steps_collected < num_steps:
                    raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    graph = info['graph']
                    x_dict = {k: v.to(self.device) for k, v in graph.x_dict.items()}
                    edge_index_dict = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()}
                    layer_mask = torch.tensor(obs['layer_mask'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
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
                    net_idx_int = net_idx_tensor.item()
                    nets_list = self.env.board.nets
                    net_name = nets_list[net_idx_int].name if net_idx_int < len(nets_list) else f"Net {net_idx_int}"
                    self.all_episode_heatmaps.append({
                        'net_name': net_name or f"Net {net_idx_int}",
                        'net_idx': net_idx_int,
                        'heatmaps_np': heatmaps_np,
                    })
                    
                    next_obs, reward, terminated, truncated, next_info = self.env.step_with_heatmaps(
                        net_idx_tensor.item(), heatmaps_np, via_prob_np
                    )
                    done = terminated or truncated
                    steps_collected += 1
                    
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
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            # Use weighted BCE to handle the massive class imbalance of sparse path pixels
                            # Cast to float32 to prevent underflow or NaN issues with BCE in float16
                            with torch.amp.autocast('cuda', enabled=False):
                                bce_loss = F.binary_cross_entropy(pred_selected.float(), target_heatmap.float(), reduction='none')
                            weight_mask = torch.where(target_heatmap > 0, torch.tensor(50.0, device=self.device), torch.tensor(1.0, device=self.device))
                            loss_dec = (bce_loss * weight_mask).mean()
                        
                        self.actor_opt.zero_grad(set_to_none=True)
        if self.routing_mode == 'astar_guided':
            while steps_collected < num_steps:
                obs, info = self.env.reset(options={'board_config': self.curriculum.get_board_config()})
                episode = Episode()
                h, z = self.jepa.initial_state(batch_size=1, device=self.device)
                
                done = False
                while not done and steps_collected < num_steps:
                    raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    graph = info['graph']
                    x_dict = {k: v.to(self.device) for k, v in graph.x_dict.items()}
                    edge_index_dict = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()}
                    
                    with torch.no_grad():
                        spatial_patches, cls_spatial = self.vit(raster_tensor)
                        node_embs = self.gnn(x_dict, edge_index_dict)
                        pad_embs = node_embs['pad'].unsqueeze(0)
                        fused_pads, fused_spatial = self.fusion(pad_embs, spatial_patches)
                        
                        global_spatial = cls_spatial
                        global_graph = fused_pads.mean(dim=1)
                        context_emb = torch.cat([global_spatial, global_graph], dim=-1)
                        context_emb = F.layer_norm(context_emb, (context_emb.shape[-1],))
                        
                    net_embs, unrouted_mask, _ = self._get_net_embeddings_and_mask(raster_tensor, x_dict, edge_index_dict)
                    
                    with torch.no_grad():
                        net_idx_tensor, heatmap_latent, log_prob_net, log_prob_heatmap, value = self.policy.act(
                            net_embs, unrouted_mask, h, z, explore=explore
                        )
                        
                    with torch.no_grad():
                        heatmaps = self.heatmap_decoder(heatmap_latent)
                        heatmaps_np = heatmaps.cpu().squeeze(0).numpy()
                        via_prob_map = torch.sigmoid(heatmaps[:, -1]).cpu().squeeze(0).numpy()
                        
                    next_obs, reward, terminated, truncated, next_info = self.env.step_with_heatmaps(
                        net_idx_tensor.item(), heatmaps_np, via_prob_map
                    )
                    
                    done = terminated or truncated
                    if next_info and 'dist_delta' in next_info:
                        dist_deltas.append(next_info['dist_delta'])
                    
                    with torch.no_grad():
                        next_raster = torch.tensor(next_obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                        next_graph = next_info['graph']
                        next_x_dict = {k: v.to(self.device) for k, v in next_graph.x_dict.items()}
                        next_edge_index_dict = {k: v.to(self.device) for k, v in next_graph.edge_index_dict.items()}
                        
                        next_sp, next_cls = self.vit(next_raster)
                        next_node = self.gnn(next_x_dict, next_edge_index_dict)
                        next_pad = next_node['pad'].unsqueeze(0)
                        next_fused_pads, _ = self.fusion(next_pad, next_sp)
                        
                        next_global_spatial = next_cls
                        next_global_graph = next_fused_pads.mean(dim=1)
                        target_context_emb = torch.cat([next_global_spatial, next_global_graph], dim=-1)
                        target_context_emb = F.layer_norm(target_context_emb, (target_context_emb.shape[-1],))
                        
                    with torch.no_grad():
                        action_tuple = (net_idx_tensor.detach().squeeze(0).cpu(), heatmap_latent.detach().squeeze(0).cpu())
                        action_emb = self.jepa.get_action_embedding(net_idx_tensor, heatmap_latent.detach())
                        h, z, _, _ = self.jepa.rssm_step(h, z, context_emb.detach(), action_emb)
                        h = h.detach()
                        z = z.detach()
                        
                        if not hasattr(episode, 'target_context_embeddings'):
                            episode.target_context_embeddings = []
                        if not hasattr(episode, 'net_embeddings_list'):
                            episode.net_embeddings_list = []
                        if not hasattr(episode, 'unrouted_masks_list'):
                            episode.unrouted_masks_list = []
                            
                        episode.append(context_emb.detach().squeeze(0).cpu(), action_tuple, reward, done)
                        episode.target_context_embeddings.append(target_context_emb.detach().squeeze(0).cpu())
                        episode.net_embeddings_list.append(net_embs.detach().squeeze(0).cpu())
                        episode.unrouted_masks_list.append(unrouted_mask.detach().squeeze(0).cpu())
                        
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
                    cr = info.get('completion_rate', 0.0)
                    drc_viol = info.get('drc_violations', 0)
                    num_nets = len(self.env.board.nets)
                    drc_rate = drc_viol / num_nets if num_nets > 0 else 0.0
                    self.curriculum.record_episode(cr, drc_rate)
                    completion_rates.append(cr)
                    
                    self.last_completed_board_state = copy.deepcopy(self.env.board_state)
                    self.last_completed_board = copy.deepcopy(self.env.board)
                    
            self.mean_dist_delta = np.mean(dist_deltas) if dist_deltas else 0.0
            return np.mean(completion_rates) if completion_rates else 0.0
        else:
            # Autoregressive mode
            while steps_collected < num_steps:
                obs, info = self.env.reset(options={'board_config': self.curriculum.get_board_config()})
                episode = Episode()
                episode.cropped_spatials = []
                episode.cursor_poses = []
                episode.target_poses = []
                episode.moves_remaining_fracs = []
                episode.fused_spatials = []
                episode.max_moves_fracs = []
                episode.board_dims = (self.env.W, self.env.H, self.env.board.num_layers)
                
                h, z = self.jepa.initial_state(batch_size=1, device=self.device)
                
                done = False
                for net_idx in range(len(self.env.board.nets)):
                    if steps_collected >= num_steps or done:
                        break
                        
                    raster_tensor = torch.tensor(obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    graph = info['graph']
                    x_dict = {k: v.to(self.device) for k, v in graph.x_dict.items()}
                    edge_index_dict = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()}
                    
                    with torch.no_grad():
                        net_embs, unrouted_mask, fused_spatial = self._get_net_embeddings_and_mask(raster_tensor, x_dict, edge_index_dict)
                        net_idx_tensor, _, _ = self.policy.select_net(net_embs, unrouted_mask, h, z, deterministic=not explore)
                        
                    self.env.start_routing_net(net_idx_tensor.item())
                    self.current_net_values = []
                    
                    net_done = False
                    while not net_done and steps_collected < num_steps:
                        curr_obs = self.env._get_obs()
                        curr_info = self.env._get_info()
                        
                        r_tensor = torch.tensor(curr_obs['board_raster'], dtype=torch.float32).unsqueeze(0).to(self.device)
                        cursor_norm = torch.tensor(curr_obs['cursor_pos'], dtype=torch.float32).unsqueeze(0).to(self.device)
                        target_norm = torch.tensor(curr_obs['target_pos'], dtype=torch.float32).unsqueeze(0).to(self.device)
                        moves_frac = torch.tensor(curr_obs['moves_remaining_frac'], dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        graph_t = curr_info['graph']
                        x_dict_t = {k: v.to(self.device) for k, v in graph_t.x_dict.items()}
                        edge_index_dict_t = {k: v.to(self.device) for k, v in graph_t.edge_index_dict.items()}
                        
                        with torch.no_grad():
                            spatial_patches, cls_spatial = self.vit(r_tensor)
                            node_embs = self.gnn(x_dict_t, edge_index_dict_t)
                            pad_embs = node_embs['pad'].unsqueeze(0)
                            fused_pads, fused_spatial = self.fusion(pad_embs, spatial_patches)
                            
                            logits, value = self.policy.forward_step(fused_spatial, cursor_norm, target_norm, moves_frac)
                            if not hasattr(self, 'current_net_values') or self.current_net_values is None:
                                self.current_net_values = []
                            self.current_net_values.append(value.item())
                            
                            v_mask = torch.tensor(get_valid_mask(self.env), dtype=torch.bool, device=self.device).unsqueeze(0)
                            masked_logits = logits.masked_fill(~v_mask, -1e4)
                            
                            probs = F.softmax(masked_logits, dim=-1)
                            self.last_action_probs = probs.squeeze(0).detach().cpu().numpy()
                            if not explore:
                                action = masked_logits.argmax(dim=-1).item()
                            else:
                                dist = torch.distributions.Categorical(probs)
                                action = dist.sample().item()
                                
                        cursor_prev = self.env.cursor_pos
                        next_obs, reward, terminated, truncated, next_info = self.env.step({'action_id': action})
                        if next_info and 'dist_delta' in next_info:
                            dist_deltas.append(next_info['dist_delta'])
                        cursor_curr = self.env.cursor_pos
                        if cursor_curr is None:
                            cursor_curr = getattr(self.env, 'last_cursor_pos', cursor_prev)
                            if cursor_curr is None:
                                cursor_curr = cursor_prev
                        
                        net_done = (self.env.current_net_index is None)
                        done = terminated or truncated
                        
                        dx = cursor_curr[0] - cursor_prev[0]
                        dy = cursor_curr[1] - cursor_prev[1]
                        dl = cursor_curr[2] - cursor_prev[2]
                        cursor_delta = np.array([dx, dy, dl], dtype=np.float32)
                        
                        with torch.no_grad():
                            action_onehot = F.one_hot(torch.tensor([action], device=self.device), num_classes=10).float()
                            cursor_delta_tensor = torch.tensor(cursor_delta, dtype=torch.float32, device=self.device).unsqueeze(0)
                            
                            action_emb = self.jepa.get_action_embedding_move(action_onehot, cursor_delta_tensor)
                            
                            global_spatial = cls_spatial
                            global_graph = fused_pads.mean(dim=1)
                            context_emb = torch.cat([global_spatial, global_graph], dim=-1)
                            context_emb = F.layer_norm(context_emb, (context_emb.shape[-1],))
                            
                            h, z, _, _ = self.jepa.rssm_step(h, z, context_emb.detach(), action_emb)
                            h = h.detach()
                            z = z.detach()
                            
                            cropped_spatial = self.policy.step_policy.crop_spatial(fused_spatial, cursor_norm)
                            
                            if not hasattr(episode, 'target_context_embeddings'):
                                episode.target_context_embeddings = []
                            if not hasattr(episode, 'net_embeddings_list'):
                                episode.net_embeddings_list = []
                            if not hasattr(episode, 'unrouted_masks_list'):
                                episode.unrouted_masks_list = []
                                
                            action_tuple = (torch.tensor(action, dtype=torch.long).cpu(), cursor_delta_tensor.detach().squeeze(0).cpu())
                            
                            episode.append(context_emb.detach().squeeze(0).cpu(), action_tuple, reward, done)
                            episode.target_context_embeddings.append(context_emb.detach().squeeze(0).cpu())
                            episode.net_embeddings_list.append(net_embs.detach().squeeze(0).cpu())
                            episode.unrouted_masks_list.append(unrouted_mask.detach().squeeze(0).cpu())
                            
                            episode.cropped_spatials.append(cropped_spatial.detach().squeeze(0).cpu())
                            episode.cursor_poses.append(cursor_norm.detach().squeeze(0).cpu())
                            episode.target_poses.append(target_norm.detach().squeeze(0).cpu())
                            episode.moves_remaining_fracs.append(moves_frac.detach().squeeze(0).cpu())
                            episode.fused_spatials.append(fused_spatial.detach().squeeze(0).cpu())
                            
                            max_moves = curr_info.get('max_moves_per_net', 0)
                            moves_frac_val = 1.0 / max(1, max_moves)
                            episode.max_moves_fracs.append(moves_frac_val)
                            
                        obs = next_obs
                        info = next_info
                        steps_collected += 1
                        self.total_timesteps += 1
                        if steps_collected % 50 == 0 or steps_collected == num_steps:
                            print(f"  Collected {steps_collected}/{num_steps} steps...")
                            
                if episode.length > 0:
                    episode.net_embeddings = episode.net_embeddings_list[0]
                    episode.unrouted_masks = episode.unrouted_masks_list
                    self.replay_buffer.add_episode(episode)
                    cr = info.get('completion_rate', 0.0)
                    drc_viol = info.get('drc_violations', 0)
                    num_nets = len(self.env.board.nets)
                    drc_rate = drc_viol / num_nets if num_nets > 0 else 0.0
                    self.curriculum.record_episode(cr, drc_rate)
                    completion_rates.append(cr)
                    
                    self.last_completed_board_state = copy.deepcopy(self.env.board_state)
                    self.last_completed_board = copy.deepcopy(self.env.board)
                    
            self.mean_dist_delta = np.mean(dist_deltas) if dist_deltas else 0.0
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
        
        b_crop = []
        b_cursor = []
        b_target = []
        b_moves = []
        
        sampled_ep_info = []
        for ep in sampled_episodes:
            if ep.length >= seq_len:
                start = random.randint(0, ep.length - seq_len)
                end = start + seq_len
            else:
                start = 0
                end = ep.length
            sampled_ep_info.append((ep, start, end))
            
            ctx_tensor = ep.context_embeddings_tensor[start:end]
            tgt_tensor = ep.target_context_embeddings_tensor[start:end]
            net_tensor = ep.net_actions_tensor[start:end]
            heat_tensor = ep.heatmap_actions_tensor[start:end]
            rew_tensor = ep.rewards_tensor[start:end]
            cont_tensor = 1.0 - ep.dones_tensor[start:end].to(torch.float32)
            mask_tensor = torch.ones(end - start, dtype=torch.float32)
            
            pad_len = seq_len - (end - start)
            if pad_len > 0:
                ctx_tensor = torch.cat([ctx_tensor, torch.zeros(pad_len, ctx_tensor.shape[-1], dtype=ctx_tensor.dtype, device=ctx_tensor.device)], dim=0)
                tgt_tensor = torch.cat([tgt_tensor, torch.zeros(pad_len, tgt_tensor.shape[-1], dtype=tgt_tensor.dtype, device=tgt_tensor.device)], dim=0)
                net_tensor = torch.cat([net_tensor, torch.zeros(pad_len, dtype=net_tensor.dtype, device=net_tensor.device)], dim=0)
                heat_tensor = torch.cat([heat_tensor, torch.zeros(pad_len, heat_tensor.shape[-1], dtype=heat_tensor.dtype, device=heat_tensor.device)], dim=0)
                rew_tensor = torch.cat([rew_tensor, torch.zeros(pad_len, dtype=rew_tensor.dtype, device=rew_tensor.device)], dim=0)
                cont_tensor = torch.cat([cont_tensor, torch.zeros(pad_len, dtype=cont_tensor.dtype, device=cont_tensor.device)], dim=0)
                mask_tensor = torch.cat([mask_tensor, torch.zeros(pad_len, dtype=mask_tensor.dtype, device=mask_tensor.device)], dim=0)
                
            b_ctx.append(ctx_tensor)
            b_tgt_ctx.append(tgt_tensor)
            b_net.append(net_tensor)
            b_heat.append(heat_tensor)
            b_rew.append(rew_tensor)
            b_cont.append(cont_tensor)
            b_mask.append(mask_tensor)
            
            if self.routing_mode == 'autoregressive':
                crop_tensor = ep.cropped_spatials_tensor[start:end]
                cursor_tensor = ep.cursor_poses_tensor[start:end]
                target_tensor = ep.target_poses_tensor[start:end]
                moves_tensor = ep.moves_remaining_fracs_tensor[start:end]
                
                if pad_len > 0:
                    crop_tensor = torch.cat([crop_tensor, torch.zeros(pad_len, crop_tensor.shape[-1], dtype=crop_tensor.dtype, device=crop_tensor.device)], dim=0)
                    cursor_tensor = torch.cat([cursor_tensor, torch.zeros(pad_len, cursor_tensor.shape[-1], dtype=cursor_tensor.dtype, device=cursor_tensor.device)], dim=0)
                    target_tensor = torch.cat([target_tensor, torch.zeros(pad_len, target_tensor.shape[-1], dtype=target_tensor.dtype, device=target_tensor.device)], dim=0)
                    moves_tensor = torch.cat([moves_tensor, torch.zeros(pad_len, moves_tensor.shape[-1], dtype=moves_tensor.dtype, device=moves_tensor.device)], dim=0)
                    
                b_crop.append(crop_tensor)
                b_cursor.append(cursor_tensor)
                b_target.append(target_tensor)
                b_moves.append(moves_tensor)
            
        batch = {
            'context_embeddings': torch.stack(b_ctx).to(self.device),
            'target_context_embeddings': torch.stack(b_tgt_ctx).to(self.device),
            'net_actions': torch.stack(b_net).to(self.device),
            'heatmap_actions': torch.stack(b_heat).to(self.device),
            'rewards': torch.stack(b_rew).to(self.device),
            'continues': torch.stack(b_cont).to(self.device),
            'masks': torch.stack(b_mask).to(self.device)
        }
        if self.routing_mode == 'autoregressive':
            batch['cropped_spatials'] = torch.stack(b_crop).to(self.device)
            batch['cursor_poses'] = torch.stack(b_cursor).to(self.device)
            batch['target_poses'] = torch.stack(b_target).to(self.device)
            batch['moves_remaining_fracs'] = torch.stack(b_moves).to(self.device)
            
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
            h_init = h.clone()
            z_init = z.clone()
            flat_net = batch['net_actions'].reshape(-1)
            flat_heat = batch['heatmap_actions'].reshape(-1, batch['heatmap_actions'].shape[-1])
            if flat_heat.shape[-1] == 3:
                move_action_onehot = F.one_hot(flat_net.long(), num_classes=10).float()
                action_embs = self.jepa.get_action_embedding_move(move_action_onehot, flat_heat).reshape(B_b, T_b, -1)
            else:
                action_embs = self.jepa.get_action_embedding(flat_net, flat_heat).reshape(B_b, T_b, -1)
            
            all_h, all_z = [], []
            for t in range(T_b - 1):
                h, z, _, _ = self.jepa.rssm_step(h, z, batch['context_embeddings'][:, t], action_embs[:, t])
                all_h.append(h)
                all_z.append(z)
            if all_h:
                h_seq = torch.stack(all_h, dim=1)
                z_seq = torch.stack(all_z, dim=1)
                self.replay_buffer.cache_latents(h_seq, z_seq)
                
                h_init_cpu = h_init.detach().cpu()
                z_init_cpu = z_init.detach().cpu()
                h_seq_cpu = h_seq.detach().cpu()
                z_seq_cpu = z_seq.detach().cpu()
                
                h_dim = h_seq_cpu.shape[-1]
                z_dim = z_seq_cpu.shape[-1]
                
                for i in range(B_b):
                    ep, start, end = sampled_ep_info[i]
                    if ep.cached_h is None:
                        ep.cached_h = torch.zeros((ep.length, h_dim), dtype=torch.float32)
                        ep.cached_z = torch.zeros((ep.length, z_dim), dtype=torch.float32)
                    ep.cached_h[start] = h_init_cpu[i]
                    ep.cached_z[start] = z_init_cpu[i]
                    valid_len = end - start - 1
                    if valid_len > 0:
                        ep.cached_h[start + 1:end] = h_seq_cpu[i, :valid_len]
                        ep.cached_z[start + 1:end] = z_seq_cpu[i, :valid_len]
                
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
            sampled_episodes = random.choices(self.replay_buffer.episodes, k=self.imagine_batch_size)
            sampled_indices = [random.randint(0, max(0, ep.length - 1)) for ep in sampled_episodes]
            
            if self.routing_mode == 'autoregressive':
                h0_list = []
                z0_list = []
                for i, ep in enumerate(sampled_episodes):
                    idx = sampled_indices[i]
                    if ep.cached_h is not None:
                        h0_list.append(ep.cached_h[idx].to(self.device))
                        z0_list.append(ep.cached_z[idx].to(self.device))
                        self.autoregressive_total_count += 1
                    else:
                        single_h, single_z = self.jepa.initial_state(batch_size=1, device=self.device)
                        h0_list.append(single_h.squeeze(0))
                        z0_list.append(single_z.squeeze(0))
                        self.autoregressive_fallback_count += 1
                        self.autoregressive_total_count += 1
                h0 = torch.stack(h0_list).detach()
                z0 = torch.stack(z0_list).detach()
                
                # Debug assertions check
                if getattr(self, 'debug_assertions', False):
                    for i in range(min(5, self.imagine_batch_size)):
                        ep = sampled_episodes[i]
                        idx = sampled_indices[i]
                        if ep.cached_h is not None and not torch.all(ep.cached_h[0] == 0):
                            h_check, z_check = self.jepa.initial_state(batch_size=1, device=self.device)
                            for t in range(idx):
                                ctx_t = ep.context_embeddings_tensor[t].unsqueeze(0).to(self.device)
                                net_act_t = ep.net_actions_tensor[t].unsqueeze(0).to(self.device)
                                heat_act_t = ep.heatmap_actions_tensor[t].unsqueeze(0).to(self.device)
                                
                                if heat_act_t.shape[-1] == 3:
                                    move_action_onehot = F.one_hot(net_act_t.long(), num_classes=10).float()
                                    action_emb_t = self.jepa.get_action_embedding_move(move_action_onehot, heat_act_t)
                                else:
                                    action_emb_t = self.jepa.get_action_embedding(net_act_t, heat_act_t)
                                    
                                h_check, z_check, _, _ = self.jepa.rssm_step(h_check, z_check, ctx_t, action_emb_t)
                                
                            h_cached = ep.cached_h[idx].to(self.device)
                            assert torch.allclose(h_check.squeeze(0), h_cached, atol=1e-3, rtol=1e-3), \
                                f"Debug assertion failed: cached h at idx {idx} does not match recomputed h!"
                
                net_embeddings = None
                unrouted_mask = None
            else:
                init_states = self.replay_buffer.sample_latents(self.imagine_batch_size, device=self.device)
                
                if init_states is None:
                    h0, z0 = self.jepa.initial_state(self.imagine_batch_size, device=self.device)
                else:
                    h0, z0 = init_states['h'], init_states['z']
                    
                h0 = h0.detach()
                z0 = z0.detach()
                
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
            
            if self.routing_mode == 'autoregressive':
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    rollout = self._imagine_autoregressive_rollout(h, z, sampled_episodes, sampled_indices, current_horizon)
                traj_h = rollout['traj_h']
                traj_z = rollout['traj_z']
                traj_actions_net = rollout['traj_actions_net']
                traj_rewards = rollout['traj_rewards']
                traj_continues = rollout['traj_continues']
                traj_values = rollout['traj_values']
                traj_log_probs_net = rollout['traj_log_probs_net']
                traj_entropy = rollout['traj_entropy']
                h = rollout['final_h']
                z = rollout['final_z']
            else:
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
            if self.routing_mode == 'autoregressive':
                traj_entropy = torch.stack(traj_entropy, dim=0)
            if self.routing_mode != 'autoregressive':
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
                # Normalize advantages to stabilize training and maintain correct scale relative to entropy
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.routing_mode == 'autoregressive':
                    loss_policy = -traj_log_probs_net * advantages
                    loss_policy = loss_policy.mean() - self._current_entropy_coef() * traj_entropy.mean()
                else:
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
            if self.routing_mode == 'autoregressive':
                metrics['entropy'].append(traj_entropy.mean().item())
            else:
                metrics['entropy'].append(0.0)
            
        for p in self.jepa.parameters():
            p.requires_grad = True
            
        return {
            'loss_actor': np.mean(metrics['loss_actor']),
            'loss_critic': np.mean(metrics['loss_critic']),
            'imagined_return_mean': np.mean(metrics['imagined_return_mean']),
            'actor_grad_norm': np.mean(metrics['actor_grad_norm']),
            'critic_grad_norm': np.mean(metrics['critic_grad_norm']),
            'entropy': np.mean(metrics['entropy'])
        }

    def _imagine_autoregressive_rollout(
        self,
        h0: torch.Tensor,
        z0: torch.Tensor,
        sampled_episodes: List[Episode],
        sampled_indices: List[int],
        current_horizon: int
    ) -> Dict[str, Any]:
        h, z = h0, z0
        traj_h = []
        traj_z = []
        traj_actions_net = []
        traj_rewards = []
        traj_continues = []
        traj_values = []
        traj_log_probs_net = []
        traj_entropy = []
        traj_cursor_poses = []
        
        # 1. Initialize per-episode values at t = 0
        cursor_pos_list = []
        target_pos_list = []
        moves_frac_list = []
        moves_dec_list = []
        fused_sp_list = []
        
        for i, ep in enumerate(sampled_episodes):
            idx = sampled_indices[i]
            cursor_pos_list.append(ep.cursor_poses_tensor[idx])
            target_pos_list.append(ep.target_poses_tensor[idx])
            moves_frac_list.append(ep.moves_remaining_fracs_tensor[idx])
            moves_dec_list.append(ep.max_moves_fracs_tensor[idx])
            fused_sp_list.append(ep.fused_spatials_tensor[idx])
            
        cursor_pos_img = torch.stack(cursor_pos_list).to(self.device)
        target_pos_img = torch.stack(target_pos_list).to(self.device)
        moves_remaining_frac_img = torch.stack(moves_frac_list).to(self.device)
        moves_decrement = torch.stack(moves_dec_list).to(self.device).unsqueeze(-1)
        
        # Static Map/Moving Window Approximation:
        # We hold the global board representation static across the imagined horizon,
        # which means the policy crops its window based on where it moves (dynamic cursor),
        # but the underlying board occupancy (wires/obstacles) is not dynamically updated.
        # Since different episodes can have different board sizes (varying N_patches),
        # we keep them as a list of raw tensors and group/crop them dynamically inside the loop.
        
        board_dims = torch.tensor([ep.board_dims for ep in sampled_episodes], dtype=torch.float32, device=self.device)
        B_size = len(sampled_episodes)
        
        # Store the cursor at t=0
        traj_cursor_poses.append(cursor_pos_img.clone())
        
        # 2. Rollout loop
        for t in range(current_horizon):
            # Group by N_patches to batch-crop efficiently (handles varying board/grid sizes in the batch)
            crop_size = self.policy.step_policy.crop_size
            embed_dim = self.policy.step_policy.embed_dim
            crop_dim = crop_size * crop_size * embed_dim
            
            cropped_spatial = torch.zeros((B_size, crop_dim), device=self.device, dtype=cursor_pos_img.dtype)
            
            groups = {}
            for idx_in_batch, f_sp in enumerate(fused_sp_list):
                N_patches = f_sp.shape[0]
                if N_patches not in groups:
                    groups[N_patches] = []
                groups[N_patches].append(idx_in_batch)
                
            for N_patches, indices in groups.items():
                group_fused = torch.stack([fused_sp_list[idx] for idx in indices]).to(self.device) # shape (G, N_patches, C)
                group_cursor = cursor_pos_img[indices] # shape (G, 3)
                group_cropped = self.policy.step_policy.crop_spatial(group_fused, group_cursor) # shape (G, crop_dim)
                cropped_spatial[indices] = group_cropped
            
            # Policy forward step using current dynamically evolved observations
            logits, value = self.policy.forward_step_cropped(
                cropped_spatial, cursor_pos_img, target_pos_img, moves_remaining_frac_img
            )
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_id = dist.sample()
            log_prob = dist.log_prob(action_id)
            entropy = dist.entropy()
            
            pred_reward = self.jepa.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)
            pred_continue_logits = self.jepa.continue_head(torch.cat([h, z], dim=-1)).squeeze(-1)
            pred_continue = torch.sigmoid(pred_continue_logits)
            
            traj_h.append(h)
            traj_z.append(z)
            traj_actions_net.append(action_id)
            traj_rewards.append(pred_reward)
            traj_continues.append(pred_continue)
            traj_values.append(value)
            traj_log_probs_net.append(log_prob)
            traj_entropy.append(entropy)
            
            device = action_id.device
            moves_delta = torch.tensor([
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]
            ], device=device)
            
            B_size = action_id.shape[0]
            cursor_delta = torch.zeros((B_size, 3), device=device)
            
            mask_grid = (action_id < 8)
            if mask_grid.any():
                cursor_delta[mask_grid] = moves_delta[action_id[mask_grid]]
            mask_up = (action_id == 8)
            if mask_up.any():
                cursor_delta[mask_up, 2] = -1.0
            mask_down = (action_id == 9)
            if mask_down.any():
                cursor_delta[mask_down, 2] = 1.0
                
            # Convert delta to normalized units using per-episode board_dims
            cursor_delta_norm = cursor_delta / board_dims
            
            # Evolve observation inputs for the next step (t+1)
            cursor_pos_img = torch.clamp(cursor_pos_img + cursor_delta_norm, 0.0, 1.0)
            moves_remaining_frac_img = torch.clamp(moves_remaining_frac_img - moves_decrement, 0.0, 1.0)
            
            # Store next cursor position
            traj_cursor_poses.append(cursor_pos_img.clone())
            
            # Evolve model states h, z
            action_onehot = F.one_hot(action_id, num_classes=10).float()
            action_emb = self.jepa.get_action_embedding_move(action_onehot, cursor_delta)
            h, z = self.jepa.predict_step(h, z, action_emb)
            
        return {
            'traj_h': traj_h,
            'traj_z': traj_z,
            'traj_actions_net': traj_actions_net,
            'traj_rewards': traj_rewards,
            'traj_continues': traj_continues,
            'traj_values': traj_values,
            'traj_log_probs_net': traj_log_probs_net,
            'traj_entropy': traj_entropy,
            'traj_cursor_poses': traj_cursor_poses,
            'final_h': h,
            'final_z': z
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
            self.metrics_history['loss_wm_reward'].append(wm_metrics.get('loss_wm_reward', 0.0))
            self.metrics_history['loss_actor'].append(ac_metrics['loss_actor'])
            self.metrics_history['loss_critic'].append(ac_metrics['loss_critic'])
            self.metrics_history['entropy'].append(ac_metrics.get('entropy', 0.0))
            self.metrics_history['mean_dist_delta'].append(self.mean_dist_delta)
            self.metrics_history['stage'].append(self.curriculum.current_stage_name)
            
            print(f"[Step {self.total_timesteps}/{total_timesteps}] "
                  f"Stage: '{self.curriculum.current_stage_name}' | "
                  f"Completion: {mean_completion:.2f} | "
                  f"Loss WM: {wm_metrics['loss_wm']:.4f} (Reward: {wm_metrics.get('loss_wm_reward', 0.0):.4f}) | "
                  f"Loss Actor: {ac_metrics['loss_actor']:.4f} | "
                  f"Loss Critic: {ac_metrics['loss_critic']:.4f} | "
                  f"Entropy: {ac_metrics.get('entropy', 0.0):.4f} | "
                  f"Dist Delta: {self.mean_dist_delta:.4f}")
            
            progress_bar.n = self.total_timesteps
            progress_bar.set_postfix({
                'stage': self.curriculum.current_stage_name,
                'comp_rate': f"{mean_completion:.2f}",
                'loss_wm_rew': f"{wm_metrics.get('loss_wm_reward', 0.0):.3f}",
                'entropy': f"{ac_metrics.get('entropy', 0.0):.3f}",
                'dist_delta': f"{self.mean_dist_delta:.3f}",
                'loss_actor': f"{ac_metrics['loss_actor']:.3f}"
            })
            progress_bar.refresh()
            # Note: Curriculum episodes and completion history are updated automatically
            # inside _phase1_collect_real for every actual episode completed.
            
            if self.curriculum.should_advance():
                self.curriculum.advance()
                self.env.curriculum_stage = self.curriculum.current_stage
                self.env.reset(options={'board_config': self.curriculum.get_board_config()})
                
            save_interval = self.train_cfg.get('training', {}).get('save_interval', 50000)
            if self.total_timesteps % save_interval < self.real_steps_per_iteration:
                self.save_checkpoint(f"{self.checkpoint_dir}/checkpoint_{self.total_timesteps}.pt")

            # Visual checkpoint gated by visual_save_interval (not every iteration)
            visual_interval = self.train_cfg.get('training', {}).get('visual_save_interval', 5000)
            if self.total_timesteps % visual_interval < self.real_steps_per_iteration:
                self.save_visual_checkpoint(f"{self.checkpoint_dir}/visuals/step_{self.total_timesteps}.png")
            
            if on_update is not None:
                on_update(self, {
                    'timesteps': self.total_timesteps,
                    'completion_rate': mean_completion,
                    **wm_metrics,
                    **ac_metrics
                })

