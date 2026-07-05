import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from pcb_router.models.route_step_policy import RouteStepPolicy

import copy

class DreamerActorCritic(nn.Module):
    def __init__(
        self,
        h_dim: int = 512,
        z_dim: int = 1024,
        embed_dim: int = 384,
        net_selector_dim: int = 256,
        heatmap_latent_dim: int = 256,
        value_hidden_dim: int = 256
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.heatmap_latent_dim = heatmap_latent_dim
        
        self.state_dim = h_dim + z_dim
        
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU()
        )
        
        # Step-by-step routing policy
        self.step_policy = RouteStepPolicy(embed_dim=embed_dim)
        
        self.net_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, net_selector_dim),
            nn.Tanh(),
            nn.Linear(net_selector_dim, 1)
        )
        
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(embed_dim + self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        self.heatmap_mean = nn.Linear(512, heatmap_latent_dim)
        self.heatmap_log_std = nn.Parameter(torch.zeros(1, heatmap_latent_dim))
        
        self.value_head = nn.Sequential(
            nn.Linear(self.state_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
        
        self.target_value_head = copy.deepcopy(self.value_head)
        for p in self.target_value_head.parameters():
            p.requires_grad = False

    def update_target_critic(self, ema_decay=0.98):
        with torch.no_grad():
            for target_param, critic_param in zip(self.target_value_head.parameters(), self.value_head.parameters()):
                target_param.data.mul_(ema_decay).add_(critic_param.data, alpha=1.0 - ema_decay)

    def select_net(self, net_embeddings, unrouted_mask, h, z, deterministic=False):
        B, num_nets, _ = net_embeddings.shape
        state = torch.cat([h, z], dim=-1)
        state_proj = self.state_proj(state).unsqueeze(1)
        state_proj = state_proj.expand(-1, num_nets, -1)
        
        x = torch.cat([net_embeddings, state_proj], dim=-1)
        scores = self.net_scorer(x).squeeze(-1)
        
        scores = scores.masked_fill(~unrouted_mask, -1e4)
        all_masked = (~unrouted_mask).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        probs = F.softmax(scores, dim=-1)
        
        dist = Categorical(probs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy()

    def get_heatmap_latent(self, net_embedding, h, z, deterministic=False):
        state = torch.cat([h, z], dim=-1)
        x = torch.cat([net_embedding, state], dim=-1)
        h_feat = self.heatmap_mlp(x)
        mean = self.heatmap_mean(h_feat)
        log_std = torch.clamp(self.heatmap_log_std, min=-20.0, max=2.0)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def get_value(self, h, z, use_target=False):
        state = torch.cat([h, z], dim=-1)
        value_head = self.target_value_head if use_target else self.value_head
        return value_head(state).squeeze(-1)

    def forward(self, net_embeddings, unrouted_mask, h, z, deterministic=False):
        B = net_embeddings.shape[0]
        
        net_idx, log_prob_net, ent_net = self.select_net(net_embeddings, unrouted_mask, h, z, deterministic)
        
        selected_net_emb = net_embeddings[torch.arange(B, device=net_embeddings.device), net_idx]
        
        heatmap_latent, log_prob_heatmap, ent_heatmap = self.get_heatmap_latent(
            selected_net_emb, h, z, deterministic
        )
        
        value = self.get_value(h, z)
        
        return net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, value

    def forward_step(self, fused_spatial, cursor_pos, target_pos, moves_remaining_frac, h, z):
        logits = self.step_policy(fused_spatial, cursor_pos, target_pos, moves_remaining_frac, h, z)
        value = self.get_value(h, z)
        return logits, value

    def forward_step_cropped(self, cropped_spatial, cursor_pos, target_pos, moves_remaining_frac, h, z):
        logits = self.step_policy.forward_cropped(cropped_spatial, cursor_pos, target_pos, moves_remaining_frac, h, z)
        value = self.get_value(h, z)
        return logits, value

    def act(self, net_embeddings, unrouted_mask, h, z, explore=True):
        with torch.no_grad():
            net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, _ = self.forward(
                net_embeddings, unrouted_mask, h, z, deterministic=not explore
            )
        return net_idx, heatmap_latent, log_prob_net, log_prob_heatmap

    def evaluate_actions(self, net_embeddings, unrouted_mask, h, z, net_idx, heatmap_latent):
        B = net_embeddings.shape[0]
        state = torch.cat([h, z], dim=-1)
        
        state_proj = self.state_proj(state).unsqueeze(1).expand(-1, net_embeddings.shape[1], -1)
        x = torch.cat([net_embeddings, state_proj], dim=-1)
        scores = self.net_scorer(x).squeeze(-1)
        scores = scores.masked_fill(~unrouted_mask, -1e4)
        all_masked = (~unrouted_mask).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        probs = F.softmax(scores, dim=-1)
        dist_net = Categorical(probs)
        
        log_prob_net = dist_net.log_prob(net_idx)
        entropy_net = dist_net.entropy()
        
        selected_net_emb = net_embeddings[torch.arange(B, device=net_embeddings.device), net_idx]
        x_heat = torch.cat([selected_net_emb, state], dim=-1)
        h_feat = self.heatmap_mlp(x_heat)
        mean = self.heatmap_mean(h_feat)
        log_std = torch.clamp(self.heatmap_log_std, min=-20.0, max=2.0)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist_heatmap = Normal(mean, std)
        
        log_prob_heatmap = dist_heatmap.log_prob(heatmap_latent).sum(dim=-1)
        entropy_heatmap = dist_heatmap.entropy().sum(dim=-1)
        
        value = self.get_value(h, z)
        
        return log_prob_net, log_prob_heatmap, value, entropy_net, entropy_heatmap

