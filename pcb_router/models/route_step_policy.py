import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouteStepPolicy(nn.Module):
    def __init__(self, embed_dim=384, cursor_embed_dim=32, hidden_dim=256, crop_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.crop_size = crop_size
        
        # Position projections
        self.cursor_proj = nn.Sequential(
            nn.Linear(3, cursor_embed_dim),
            nn.ReLU(),
            nn.Linear(cursor_embed_dim, cursor_embed_dim)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(3, cursor_embed_dim),
            nn.ReLU(),
            nn.Linear(cursor_embed_dim, cursor_embed_dim)
        )
        self.budget_proj = nn.Sequential(
            nn.Linear(1, cursor_embed_dim),
            nn.ReLU(),
            nn.Linear(cursor_embed_dim, cursor_embed_dim)
        )
        
        # Crop projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(embed_dim * crop_size * crop_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Fused MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3 * cursor_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Heads
        self.action_head = nn.Linear(hidden_dim, 10)
        self.value_head = nn.Linear(hidden_dim, 1)

    def crop_spatial(self, fused_spatial, cursor_pos):
        """
        Extracts a local crop_size x crop_size patch region around cursor_pos from fused_spatial.
        fused_spatial: (B, N_patches, C)
        cursor_pos: (B, 3) normalized
        """
        B, N, C = fused_spatial.shape
        grid_size = int(round(math.sqrt(N)))
        
        # Reshape to (B, C, grid_size, grid_size)
        x = fused_spatial.transpose(1, 2).view(B, C, grid_size, grid_size)
        
        # Unnormalize cursor position to patch coordinates
        cx_norm = cursor_pos[:, 0]
        cy_norm = cursor_pos[:, 1]
        
        px = torch.clamp((cx_norm * grid_size).long(), 0, grid_size - 1)
        py = torch.clamp((cy_norm * grid_size).long(), 0, grid_size - 1)
        
        # Padding size
        pad = self.crop_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
        
        H_pad, W_pad = grid_size + 2 * pad, grid_size + 2 * pad
        padded_flat = padded.view(B, C, H_pad * W_pad)
        
        # Generate 1D indices for the crop window around (py + pad, px + pad)
        device = fused_spatial.device
        offsets = torch.arange(-pad, pad + 1, device=device)
        offsets_y, offsets_x = torch.meshgrid(offsets, offsets, indexing='ij')
        offsets_y = offsets_y.flatten().view(1, -1)
        offsets_x = offsets_x.flatten().view(1, -1)
        
        cy = py + pad
        cx = px + pad
        
        crop_y = cy.view(-1, 1) + offsets_y
        crop_x = cx.view(-1, 1) + offsets_x
        
        flat_indices = crop_y * W_pad + crop_x
        flat_indices_expanded = flat_indices.unsqueeze(1).expand(-1, C, -1)
        
        cropped = torch.gather(padded_flat, 2, flat_indices_expanded)
        return cropped.transpose(1, 2).reshape(B, -1)

    def forward(self, fused_spatial, cursor_pos, target_pos, moves_remaining_frac):
        """
        fused_spatial: (B, N_patches, C)
        cursor_pos: (B, 3)
        target_pos: (B, 3)
        moves_remaining_frac: (B, 1)
        """
        cropped_spatial = self.crop_spatial(fused_spatial, cursor_pos)
        return self.forward_cropped(cropped_spatial, cursor_pos, target_pos, moves_remaining_frac)

    def forward_cropped(self, cropped_spatial, cursor_pos, target_pos, moves_remaining_frac):
        """
        cropped_spatial: (B, crop_size * crop_size * C)
        cursor_pos: (B, 3)
        target_pos: (B, 3)
        moves_remaining_frac: (B, 1)
        """
        spatial_emb = self.spatial_proj(cropped_spatial)
        cursor_emb = self.cursor_proj(cursor_pos)
        target_emb = self.target_proj(target_pos)
        budget_emb = self.budget_proj(moves_remaining_frac)
        
        x = torch.cat([spatial_emb, cursor_emb, target_emb, budget_emb], dim=-1)
        feat = self.mlp(x)
        
        logits = self.action_head(feat)
        value = self.value_head(feat)
        
        return logits, value
