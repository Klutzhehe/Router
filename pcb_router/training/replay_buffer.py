import random
import collections
from typing import List, Dict, Deque, Tuple, Any
import torch

class Episode:
    def __init__(self):
        self.context_embeddings: List[torch.Tensor] = []  # List of (384,) tensors or similar
        self.actions: List[Tuple[torch.Tensor, torch.Tensor]] = []  # List of tuples: (net_idx, heatmap_latent)
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.length: int = 0
        self._finalized = False

    def append(self, context_embedding: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor], reward: float, done: bool):
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized episode.")
        self.context_embeddings.append(context_embedding)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length = len(self.context_embeddings)

    def finalize(self):
        """Convert list elements into contiguous PyTorch tensors on CPU to speed up sampling and save memory."""
        if self._finalized:
            return
        if self.length == 0:
            self._finalized = True
            return
            
        # Convert lists to contiguous CPU tensors
        self.context_embeddings_tensor = torch.stack(self.context_embeddings).cpu()
        
        net_act_list = [a[0] for a in self.actions]
        self.net_actions_tensor = torch.stack(net_act_list).cpu() if isinstance(net_act_list[0], torch.Tensor) else torch.tensor(net_act_list, dtype=torch.long).cpu()
        
        heat_act_list = [a[1] for a in self.actions]
        self.heatmap_actions_tensor = torch.stack(heat_act_list).cpu() if isinstance(heat_act_list[0], torch.Tensor) else torch.tensor(heat_act_list, dtype=torch.float32).cpu()
        
        self.rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32).cpu()
        self.dones_tensor = torch.tensor(self.dones, dtype=torch.bool).cpu()
        
        # Handle target_context_embeddings
        if hasattr(self, 'target_context_embeddings') and self.target_context_embeddings:
            self.target_context_embeddings_tensor = torch.stack(self.target_context_embeddings).cpu()
        else:
            self.target_context_embeddings_tensor = self.context_embeddings_tensor
            
        # Handle unrouted_masks
        if hasattr(self, 'unrouted_masks'):
            if isinstance(self.unrouted_masks, list):
                self.unrouted_masks = torch.stack(self.unrouted_masks).squeeze(1).cpu()

        # Clear lists to reclaim memory
        self.context_embeddings = []
        self.actions = []
        self.rewards = []
        self.dones = []
        if hasattr(self, 'target_context_embeddings'):
            self.target_context_embeddings = []
        if hasattr(self, 'net_embeddings_list'):
            self.net_embeddings_list = []
        if hasattr(self, 'unrouted_masks_list'):
            self.unrouted_masks_list = []
        self._finalized = True


class ReplayBuffer:
    def __init__(self, capacity_episodes: int, min_episode_len: int = 1):
        self.capacity_episodes = capacity_episodes
        self.min_episode_len = min_episode_len
        self.episodes: List[Episode] = []
        self.position = 0
        self.schema_version = "1.0.0"

        # Rolling cache of (h_t, z_t) latents computed during world model training.
        self._latent_cache_capacity = 10000
        self.latent_cache: Deque[Dict[str, torch.Tensor]] = collections.deque(maxlen=self._latent_cache_capacity)

    @property
    def latent_cache_capacity(self) -> int:
        return self._latent_cache_capacity

    @latent_cache_capacity.setter
    def latent_cache_capacity(self, capacity: int):
        self._latent_cache_capacity = capacity
        # Re-create deque keeping existing elements up to the new capacity
        if hasattr(self, 'latent_cache'):
            old_cache = self.latent_cache
            self.latent_cache = collections.deque(old_cache, maxlen=capacity)
        else:
            self.latent_cache = collections.deque(maxlen=capacity)

    def add_episode(self, episode: Episode):
        if episode.length < self.min_episode_len:
            return
        episode.finalize()
        if len(self.episodes) < self.capacity_episodes:
            self.episodes.append(episode)
        else:
            self.episodes[self.position] = episode
            self.position = (self.position + 1) % self.capacity_episodes

    def cache_latents(self, h: torch.Tensor, z: torch.Tensor):
        """
        Cache states (h, z) generated during world model training forward passes.
        Args:
            h: Detached deterministic state tensor of shape (B, T, h_dim) or (B * T, h_dim)
            z: Detached stochastic state tensor of shape (B, T, z_dim) or (B * T, z_dim)
        """
        h_flat = h.reshape(-1, h.shape[-1]).detach().cpu()
        z_flat = z.reshape(-1, z.shape[-1]).detach().cpu()
        
        for i in range(h_flat.shape[0]):
            self.latent_cache.append({'h': h_flat[i], 'z': z_flat[i]})

    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch_size sequences of length seq_len from the buffer.
        """
        if not self.episodes:
            raise ValueError("Buffer is empty.")

        weights = []
        valid_episodes = []
        for ep in self.episodes:
            w = max(ep.length - seq_len + 1, 1)
            weights.append(w)
            valid_episodes.append(ep)

        sampled_episodes = random.choices(valid_episodes, weights=weights, k=batch_size)

        batch_context_embeddings = []
        batch_net_actions = []
        batch_heatmap_actions = []
        batch_rewards = []
        batch_continues = []
        batch_masks = []

        for ep in sampled_episodes:
            if ep.length >= seq_len:
                start_idx = random.randint(0, ep.length - seq_len)
                end_idx = start_idx + seq_len
                pad_len = 0
            else:
                start_idx = 0
                end_idx = ep.length
                pad_len = seq_len - ep.length

            # Extract slices directly from contiguous tensors (avoid stack/tensor creation overhead)
            ctx_tensor = ep.context_embeddings_tensor[start_idx:end_idx]
            net_act_tensor = ep.net_actions_tensor[start_idx:end_idx]
            heat_act_tensor = ep.heatmap_actions_tensor[start_idx:end_idx]
            rew_tensor = ep.rewards_tensor[start_idx:end_idx]
            cont_tensor = 1.0 - ep.dones_tensor[start_idx:end_idx].to(torch.float32)
            mask_tensor = torch.ones(end_idx - start_idx, dtype=torch.float32)

            # Padding if needed
            if pad_len > 0:
                ctx_dim = ctx_tensor.shape[-1]
                ctx_pad = torch.zeros(pad_len, ctx_dim, dtype=ctx_tensor.dtype, device=ctx_tensor.device)
                ctx_tensor = torch.cat([ctx_tensor, ctx_pad], dim=0)

                net_act_pad = torch.zeros(pad_len, dtype=net_act_tensor.dtype, device=net_act_tensor.device)
                net_act_tensor = torch.cat([net_act_tensor, net_act_pad], dim=0)

                heat_dim = heat_act_tensor.shape[-1]
                heat_act_pad = torch.zeros(pad_len, heat_dim, dtype=heat_act_tensor.dtype, device=heat_act_tensor.device)
                heat_act_tensor = torch.cat([heat_act_tensor, heat_act_pad], dim=0)

                rew_pad = torch.zeros(pad_len, dtype=torch.float32)
                rew_tensor = torch.cat([rew_tensor, rew_pad], dim=0)

                cont_pad = torch.zeros(pad_len, dtype=torch.float32)
                cont_tensor = torch.cat([cont_tensor, cont_pad], dim=0)

                mask_pad = torch.zeros(pad_len, dtype=torch.float32)
                mask_tensor = torch.cat([mask_tensor, mask_pad], dim=0)

            batch_context_embeddings.append(ctx_tensor)
            batch_net_actions.append(net_act_tensor)
            batch_heatmap_actions.append(heat_act_tensor)
            batch_rewards.append(rew_tensor)
            batch_continues.append(cont_tensor)
            batch_masks.append(mask_tensor)

        return {
            'context_embeddings': torch.stack(batch_context_embeddings),
            'net_actions': torch.stack(batch_net_actions),
            'heatmap_actions': torch.stack(batch_heatmap_actions),
            'rewards': torch.stack(batch_rewards),
            'continues': torch.stack(batch_continues),
            'masks': torch.stack(batch_masks)
        }

    def sample_latents(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """
        Sample batch_size single (h_t, z_t) latents from the cache for seeding imagination.
        """
        if len(self.latent_cache) < batch_size:
            if len(self.latent_cache) > 0:
                sampled = random.choices(self.latent_cache, k=batch_size)
            else:
                return None
        else:
            sampled = random.sample(self.latent_cache, batch_size)

        h_list = [item['h'] for item in sampled]
        z_list = [item['z'] for item in sampled]

        return {
            'h': torch.stack(h_list).to(device),
            'z': torch.stack(z_list).to(device)
        }

    def __len__(self):
        return len(self.episodes)
