import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Tuple, Any
from pcb_router.models.vit_encoder import TransformerBlock, ViTEncoder

# Symlog transform for stable targets
def symlog(x):
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def straight_through_categorical(logits, num_groups, num_classes):
    B = logits.shape[0]
    logits = logits.reshape(B, num_groups, num_classes)
    probs = F.softmax(logits, dim=-1)
    
    if logits.requires_grad:
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = F.softmax((logits + gumbels) / 1.0, dim=-1)
        index = y_soft.max(-1, keepdim=True)[1]
        # Out-of-place scatter: scatter_() would mutate y_soft in-place and corrupt
        # the autograd version counter; use the functional form instead.
        y_hard = torch.zeros_like(logits).scatter(-1, index, 1.0)
        sample = y_hard - y_soft.detach() + y_soft
    else:
        index = probs.max(-1, keepdim=True)[1]
        sample = torch.zeros_like(logits).scatter(-1, index, 1.0)
        
    return sample.reshape(B, -1), probs.reshape(B, -1)


class JEPAWorldModel(nn.Module):
    def __init__(
        self,
        vit_encoder: nn.Module,
        gnn_encoder: nn.Module,
        fusion: nn.Module,
        deterministic_size: int = 512,
        stochastic_groups: int = 32,
        stochastic_classes: int = 32,
        net_embed_dim: int = 128,
        heatmap_latent_dim: int = 256,
        reward_head_hidden: int = 256,
        continue_head_hidden: int = 256,
        ema_decay: float = 0.995,
        kl_balance: float = 0.8,
        free_bits: float = 1.0,
        num_nets_max: int = 100,
        vicreg_weight: float = 0.1,
        variance_weight: float = 25.0,
        invariance_weight: float = 25.0,
        covariance_weight: float = 1.0
    ):
        super().__init__()
        self.online_vit = vit_encoder
        self.online_gnn = gnn_encoder
        self.online_fusion = fusion

        self.ema_decay = ema_decay
        self.kl_balance = kl_balance
        self.free_bits = free_bits
        self.vicreg_weight = vicreg_weight
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight

        self.deterministic_size = deterministic_size
        self.stochastic_groups = stochastic_groups
        self.stochastic_classes = stochastic_classes
        self.z_dim = stochastic_groups * stochastic_classes

        # Action projection
        self.net_embedding = nn.Embedding(num_nets_max, net_embed_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(net_embed_dim + heatmap_latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.action_dim = 256
        self.action_proj_move = nn.Sequential(
            nn.Linear(10 + 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Recurrent cell (GRU)
        self.gru = nn.GRUCell(self.z_dim + self.action_dim, deterministic_size)
        self.gru_norm = nn.LayerNorm(deterministic_size)

        # Context embedding dim
        self.context_dim = 2 * vit_encoder.embed_dim

        # Prior/Posterior heads
        self.prior_head = nn.Sequential(
            nn.Linear(deterministic_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.z_dim)
        )

        self.posterior_head = nn.Sequential(
            nn.Linear(deterministic_size + self.context_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.z_dim)
        )

        # JEPA Predictor (predicts target next context embedding)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(deterministic_size + self.z_dim + self.action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.context_dim)
        )

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(deterministic_size + self.z_dim, reward_head_hidden),
            nn.LayerNorm(reward_head_hidden),
            nn.ReLU(),
            nn.Linear(reward_head_hidden, 1)
        )

        # Continue head
        self.continue_head = nn.Sequential(
            nn.Linear(deterministic_size + self.z_dim, continue_head_hidden),
            nn.LayerNorm(continue_head_hidden),
            nn.ReLU(),
            nn.Linear(continue_head_hidden, 1)
        )

        # Target encoders (EMA copies)
        self.target_vit = copy.deepcopy(vit_encoder)
        self.target_gnn = copy.deepcopy(gnn_encoder)
        self.target_fusion = copy.deepcopy(fusion)
        for p in list(self.target_vit.parameters()) + list(self.target_gnn.parameters()) + list(self.target_fusion.parameters()):
            p.requires_grad = False

    def update_target_weights(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_vit.parameters(), self.online_vit.parameters()):
                target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)
            for target_param, online_param in zip(self.target_gnn.parameters(), self.online_gnn.parameters()):
                target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)
            for target_param, online_param in zip(self.target_fusion.parameters(), self.online_fusion.parameters()):
                target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def initial_state(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deterministic_size, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def get_action_embedding(self, net_idx: torch.Tensor, heatmap_latent: torch.Tensor) -> torch.Tensor:
        net_emb = self.net_embedding(net_idx)
        act_concat = torch.cat([net_emb, heatmap_latent], dim=-1)
        return self.action_proj(act_concat)

    def get_action_embedding_move(self, move_action_onehot: torch.Tensor, cursor_delta: torch.Tensor) -> torch.Tensor:
        act_concat = torch.cat([move_action_onehot, cursor_delta], dim=-1)
        return self.action_proj_move(act_concat)

    def get_context_embedding(self, raster: torch.Tensor, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], use_target: bool = False) -> torch.Tensor:
        vit = self.target_vit if use_target else self.online_vit
        gnn = self.target_gnn if use_target else self.online_gnn
        fusion = self.target_fusion if use_target else self.online_fusion

        spatial_patches, cls_spatial = vit(raster)
        node_embs = gnn(x_dict, edge_index_dict)
        pad_embs = node_embs['pad'].unsqueeze(0)
        fused_pads, fused_spatial = fusion(pad_embs, spatial_patches)

        global_spatial = cls_spatial
        global_graph = fused_pads.mean(dim=1)
        context_emb = torch.cat([global_spatial, global_graph], dim=-1)
        
        # Apply LayerNorm to stabilize embedding scales and prevent collapse/saturation
        return F.layer_norm(context_emb, (context_emb.shape[-1],))

    def rssm_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        context_emb: torch.Tensor,
        action_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([z_prev, action_emb], dim=-1)
        h_t = self.gru(gru_in, h_prev)
        h_t = self.gru_norm(h_t)

        post_logits = self.posterior_head(torch.cat([h_t, context_emb], dim=-1))
        z_t_post, post_probs = straight_through_categorical(post_logits, self.stochastic_groups, self.stochastic_classes)

        prior_logits = self.prior_head(h_t)
        _, prior_probs = straight_through_categorical(prior_logits, self.stochastic_groups, self.stochastic_classes)

        return h_t, z_t_post, post_probs, prior_probs

    def predict_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([z_prev, action_emb], dim=-1)
        h_t = self.gru(gru_in, h_prev)
        h_t = self.gru_norm(h_t)

        prior_logits = self.prior_head(h_t)
        z_t_prior, _ = straight_through_categorical(prior_logits, self.stochastic_groups, self.stochastic_classes)

        return h_t, z_t_prior

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context_embs = batch['context_embeddings']
        net_actions = batch['net_actions']
        heatmap_actions = batch['heatmap_actions']
        rewards = batch['rewards']
        continues = batch['continues']
        masks = batch['masks']

        B, T, _ = context_embs.shape
        device = context_embs.device

        h, z = self.initial_state(B, device)

        kl_losses = []
        pred_losses = []
        reward_losses = []
        continue_losses = []
        variance_losses = []
        covariance_losses = []

        flat_net_actions = net_actions.reshape(-1)
        flat_heatmap_actions = heatmap_actions.reshape(-1, heatmap_actions.shape[-1])
        if heatmap_actions.shape[-1] == 3:
            move_action_onehot = F.one_hot(flat_net_actions.long(), num_classes=10).float()
            action_embs = self.get_action_embedding_move(move_action_onehot, flat_heatmap_actions).reshape(B, T, -1)
        else:
            action_embs = self.get_action_embedding(flat_net_actions, flat_heatmap_actions).reshape(B, T, -1)

        target_context_embs = batch.get('target_context_embeddings', context_embs)
        for t in range(T - 1):
            act_emb = action_embs[:, t]
            ctx_t = context_embs[:, t]
            ctx_next = target_context_embs[:, t + 1]
            mask_t = masks[:, t]

            ctx_next_pred = self.jepa_predictor(torch.cat([h, z, act_emb], dim=-1))
            mse = F.mse_loss(ctx_next_pred, ctx_next, reduction='none').mean(dim=-1)
            pred_losses.append(mse * mask_t)

            var_loss = self.compute_variance_loss(ctx_next_pred)
            cov_loss = self.compute_covariance_loss(ctx_next_pred)
            variance_losses.append(var_loss * mask_t)
            covariance_losses.append(cov_loss * mask_t)

            h, z, post_probs, prior_probs = self.rssm_step(h, z, ctx_next, act_emb)

            post_p = post_probs.reshape(B, self.stochastic_groups, self.stochastic_classes)
            prior_p = prior_probs.reshape(B, self.stochastic_groups, self.stochastic_classes)

            kl = post_p * (torch.log(post_p + 1e-8) - torch.log(prior_p + 1e-8))
            kl_sum = kl.sum(dim=-1).mean(dim=-1)

            kl_balanced = torch.max(kl_sum, torch.tensor(self.free_bits, device=device))
            kl_losses.append(kl_balanced * mask_t)

            pred_reward = self.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)
            target_reward = symlog(rewards[:, t])
            rew_loss = F.mse_loss(pred_reward, target_reward, reduction='none')
            reward_losses.append(rew_loss * mask_t)

            pred_continue = self.continue_head(torch.cat([h, z], dim=-1)).squeeze(-1)
            cont_loss = F.binary_cross_entropy_with_logits(pred_continue, continues[:, t], reduction='none')
            continue_losses.append(cont_loss * mask_t)

        denom = masks[:, :-1].sum() + 1e-8
        
        loss_kl = (torch.stack(kl_losses).sum() / denom) if kl_losses else torch.tensor(0.0, device=device)
        loss_pred = (torch.stack(pred_losses).sum() / denom) if pred_losses else torch.tensor(0.0, device=device)
        loss_reward = (torch.stack(reward_losses).sum() / denom) if reward_losses else torch.tensor(0.0, device=device)
        loss_continue = (torch.stack(continue_losses).sum() / denom) if continue_losses else torch.tensor(0.0, device=device)
        loss_var = (torch.stack(variance_losses).sum() / denom) if variance_losses else torch.tensor(0.0, device=device)
        loss_cov = (torch.stack(covariance_losses).sum() / denom) if covariance_losses else torch.tensor(0.0, device=device)

        return {
            'loss_kl': loss_kl,
            'loss_pred': loss_pred,
            'loss_reward': loss_reward,
            'loss_continue': loss_continue,
            'loss_variance': loss_var,
            'loss_covariance': loss_cov
        }

    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, grad_clip: float = 100.0) -> Dict[str, float]:
        self.train()
        losses = self.compute_loss(batch)
        
        total_loss = (
            self.invariance_weight * losses['loss_pred'] +
            self.variance_weight * losses['loss_variance'] +
            self.covariance_weight * losses['loss_covariance'] +
            losses['loss_kl'] +
            losses['loss_reward'] +
            losses['loss_continue']
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        return {
            'wm_total_loss': total_loss.item(),
            'wm_pred_loss': losses['loss_pred'].item(),
            'wm_kl_loss': losses['loss_kl'].item(),
            'wm_reward_loss': losses['loss_reward'].item(),
            'wm_continue_loss': losses['loss_continue'].item(),
            'wm_var_loss': losses['loss_variance'].item(),
            'wm_cov_loss': losses['loss_covariance'].item(),
            'wm_grad_norm': grad_norm.item()
        }

    def compute_variance_loss(self, x: torch.Tensor):
        B, C = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        loss = torch.mean(F.relu(1.0 - std))
        return loss

    def compute_covariance_loss(self, x: torch.Tensor):
        B, C = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)
        x_norm = x - x.mean(dim=0, keepdim=True)
        cov = (x_norm.T @ x_norm) / (B - 1)
        diag = torch.diag(cov)
        cov_off_diag = cov - torch.diag(diag)
        loss = torch.sum(cov_off_diag ** 2) / C
        return loss
