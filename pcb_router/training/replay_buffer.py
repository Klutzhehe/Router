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

    def append(self, context_embedding: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor], reward: float, done: bool):
        self.context_embeddings.append(context_embedding)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length = len(self.context_embeddings)


class ReplayBuffer:
    def __init__(self, capacity_episodes: int, min_episode_len: int = 1):
        self.capacity_episodes = capacity_episodes
        self.min_episode_len = min_episode_len
        self.episodes: List[Episode] = []
        self.position = 0
        self.schema_version = "1.0.0"

        # Rolling cache of (h_t, z_t) latents computed during world model training.
        # Use a deque with maxlen for O(1) eviction instead of list.pop(0) which is O(n).
        self.latent_cache_capacity = 10000
        self.latent_cache: Deque[Dict[str, torch.Tensor]] = collections.deque(maxlen=self.latent_cache_capacity)

    def add_episode(self, episode: Episode):
        if episode.length < self.min_episode_len:
            return
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
            # deque(maxlen=...) automatically discards the oldest item when full
            self.latent_cache.append({'h': h_flat[i], 'z': z_flat[i]})

    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch_size sequences of length seq_len from the buffer.
        Weights sampling by episode length to ensure uniform transition representation.
        Returns:
            Dict containing:
                - context_embeddings: (B, T, embed_dim)
                - net_actions: (B, T) discrete actions
                - heatmap_actions: (B, T, heatmap_dim)
                - rewards: (B, T)
                - continues: (B, T) - (1.0 - done)
                - masks: (B, T) mask indicating valid (non-padded) entries
        """
        if not self.episodes:
            raise ValueError("Buffer is empty.")

        # Compute sampling weights based on valid start indices in each episode
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

            # Extract slices
            ctx_slice = ep.context_embeddings[start_idx:end_idx]
            act_slice = ep.actions[start_idx:end_idx]
            rew_slice = ep.rewards[start_idx:end_idx]
            done_slice = ep.dones[start_idx:end_idx]

            # Separate actions
            net_act_slice = [a[0] for a in act_slice]
            heat_act_slice = [a[1] for a in act_slice]

            # Convert to tensors
            if len(ctx_slice) > 0:
                ctx_tensor = torch.stack(ctx_slice) if isinstance(ctx_slice[0], torch.Tensor) else torch.tensor(ctx_slice)
                net_act_tensor = torch.stack(net_act_slice) if isinstance(net_act_slice[0], torch.Tensor) else torch.tensor(net_act_slice)
                heat_act_tensor = torch.stack(heat_act_slice) if isinstance(heat_act_slice[0], torch.Tensor) else torch.tensor(heat_act_slice)
            else:
                ctx_tensor = torch.zeros(0, 384)
                net_act_tensor = torch.zeros(0, dtype=torch.long)
                heat_act_tensor = torch.zeros(0, 256)

            rew_tensor = torch.tensor(rew_slice, dtype=torch.float32)
            cont_tensor = 1.0 - torch.tensor(done_slice, dtype=torch.float32)
            mask_tensor = torch.ones(len(ctx_slice), dtype=torch.float32)

            # Padding if needed
            if pad_len > 0:
                # Pad context embeddings
                ctx_dim = ctx_tensor.shape[-1] if len(ctx_slice) > 0 else 384
                ctx_pad = torch.zeros(pad_len, ctx_dim, dtype=ctx_tensor.dtype, device=ctx_tensor.device)
                ctx_tensor = torch.cat([ctx_tensor, ctx_pad], dim=0)

                # Pad net actions with 0
                net_act_pad = torch.zeros(pad_len, dtype=net_act_tensor.dtype, device=net_act_tensor.device)
                net_act_tensor = torch.cat([net_act_tensor, net_act_pad], dim=0)

                # Pad heatmap actions
                heat_dim = heat_act_tensor.shape[-1] if len(ctx_slice) > 0 else 256
                heat_act_pad = torch.zeros(pad_len, heat_dim, dtype=heat_act_tensor.dtype, device=heat_act_tensor.device)
                heat_act_tensor = torch.cat([heat_act_tensor, heat_act_pad], dim=0)

                # Pad reward/continue/mask with 0
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
        If cache is not populated enough, fallback to initial zero states (caller can check if it returns None).
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
