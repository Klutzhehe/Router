import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class PPOPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        net_selector_dim: int = 256,
        heatmap_latent_dim: int = 256,
        value_hidden_dim: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heatmap_latent_dim = heatmap_latent_dim
        
        # Net selector: scores net embeddings (queries)
        self.net_scorer = nn.Sequential(
            nn.Linear(embed_dim, net_selector_dim),
            nn.Tanh(),
            nn.Linear(net_selector_dim, 1)
        )
        
        # Heatmap latent head: MLP outputting mean and log_std
        # Input is concatenated: selected_net_emb (384) + global_spatial (384) = 768
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        self.heatmap_mean = nn.Linear(512, heatmap_latent_dim)
        self.heatmap_log_std = nn.Parameter(torch.zeros(1, heatmap_latent_dim))
        
        # Value head
        # Input is concatenated: CLS token (384) + mean_graph (384) = 768
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )

    def select_net(self, net_embeddings, unrouted_mask, deterministic=False):
        """
        Args:
            net_embeddings: (B, num_nets, embed_dim)
            unrouted_mask: (B, num_nets) bool tensor (True for unrouted/valid, False for routed)
        """
        B, num_nets, _ = net_embeddings.shape
        scores = self.net_scorer(net_embeddings).squeeze(-1) # (B, num_nets)
        
        # Mask out routed/invalid nets
        scores = scores.masked_fill(~unrouted_mask, -1e9)
        # Prevent NaN if all nets are masked out
        all_masked = (~unrouted_mask).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        probs = F.softmax(scores, dim=-1)
        
        dist = Categorical(probs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy()

    def get_heatmap_latent(self, net_embedding, global_spatial, deterministic=False):
        """
        Args:
            net_embedding: (B, embed_dim)
            global_spatial: (B, embed_dim)
        """
        x = torch.cat((net_embedding, global_spatial), dim=-1) # (B, embed_dim * 2)
        h = self.heatmap_mlp(x)
        mean = self.heatmap_mean(h)
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

    def get_value(self, cls_spatial, mean_graph):
        x = torch.cat((cls_spatial, mean_graph), dim=-1)
        return self.value_head(x).squeeze(-1)

    def forward(self, net_embeddings, unrouted_mask, spatial_patches, cls_spatial, pad_to_net_map=None, deterministic=False):
        """
        Full forward pass for selecting actions during rollout collection
        Args:
            net_embeddings: (B, num_nets, embed_dim)
            unrouted_mask: (B, num_nets)
            spatial_patches: (B, N_patches, embed_dim)
            cls_spatial: (B, embed_dim)
        """
        B = net_embeddings.shape[0]
        
        # 1. Select Net
        net_idx, log_prob_net, ent_net = self.select_net(net_embeddings, unrouted_mask, deterministic)
        
        # 2. Extract selected net's embedding
        # Gather along net dimension
        # net_idx shape is (B,)
        selected_net_emb = net_embeddings[torch.arange(B, device=net_embeddings.device), net_idx] # (B, dim)
        
        # 3. Mean pool spatial patches for global spatial context
        global_spatial = spatial_patches.mean(dim=1) # (B, dim)
        
        # 4. Get Heatmap Latent
        heatmap_latent, log_prob_heatmap, ent_heatmap = self.get_heatmap_latent(
            selected_net_emb, global_spatial, deterministic
        )
        
        # 5. Get Value baseline estimation
        # We need a graph global representation. We'll use the mean net embedding
        mean_graph = net_embeddings.mean(dim=1)
        value = self.get_value(cls_spatial, mean_graph)
        
        return net_idx, heatmap_latent, log_prob_net, log_prob_heatmap, value

    def evaluate_actions(self, net_embeddings, unrouted_mask, spatial_patches, cls_spatial, net_idx, heatmap_latent):
        """
        Evaluates given actions for PPO update
        """
        B = net_embeddings.shape[0]
        
        # Net selection evaluations
        scores = self.net_scorer(net_embeddings).squeeze(-1)
        scores = scores.masked_fill(~unrouted_mask, -1e9)
        # Prevent NaN if all nets are masked out
        all_masked = (~unrouted_mask).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        probs = F.softmax(scores, dim=-1)
        dist_net = Categorical(probs)
        
        log_prob_net = dist_net.log_prob(net_idx)
        entropy_net = dist_net.entropy()
        
        # Heatmap latent evaluations
        selected_net_emb = net_embeddings[torch.arange(B, device=net_embeddings.device), net_idx]
        global_spatial = spatial_patches.mean(dim=1)
        
        x = torch.cat((selected_net_emb, global_spatial), dim=-1)
        h = self.heatmap_mlp(x)
        mean = self.heatmap_mean(h)
        log_std = torch.clamp(self.heatmap_log_std, min=-20.0, max=2.0)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist_heatmap = Normal(mean, std)
        
        log_prob_heatmap = dist_heatmap.log_prob(heatmap_latent).sum(dim=-1)
        entropy_heatmap = dist_heatmap.entropy().sum(dim=-1)
        
        # Value evaluation
        mean_graph = net_embeddings.mean(dim=1)
        value = self.get_value(cls_spatial, mean_graph)
        
        return log_prob_net, log_prob_heatmap, value, entropy_net, entropy_heatmap
