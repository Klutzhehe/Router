import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 13,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_grid_size: int = 1024
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Patch projection
        self.patch_embed = nn.Conv2d(
            image_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Base positional embedding for max_grid_size / patch_size
        self.base_num_patches_side = max_grid_size // patch_size
        self.base_num_patches = self.base_num_patches_side ** 2
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.base_num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed like standard ViT
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        # Add a small helper to avoid floating point issues
        w0, h0 = int(w0), int(h0)
        
        # Interpolate spatial position embeddings
        # Reshape to (1, base_side, base_side, dim) -> (1, dim, base_side, base_side)
        patch_pos_embed = patch_pos_embed.reshape(1, self.base_num_patches_side, self.base_num_patches_side, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode='bicubic',
            align_corners=False,
        )
        # Reshape back to (1, w0*h0, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patchify and project
        x_patches = self.patch_embed(x) # (B, embed_dim, H/patch_size, W/patch_size)
        H_p, W_p = x_patches.shape[2], x_patches.shape[3]
        x_patches = x_patches.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1) # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        pos_embed = self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x + pos_embed.to(x.device))
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        cls_output = x[:, 0]
        patch_output = x[:, 1:]
        
        return patch_output, cls_output

    def get_spatial_features(self, x):
        """Returns patch features reshaped to (B, H_patches, W_patches, embed_dim)"""
        H, W = x.shape[2], x.shape[3]
        H_p, W_p = H // self.patch_size, W // self.patch_size
        patch_output, _ = self.forward(x)
        return patch_output.view(-1, H_p, W_p, self.embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x
