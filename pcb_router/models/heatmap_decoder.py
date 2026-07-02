import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        spatial_dim: int = 384,
        max_layers: int = 8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_layers = max_layers
        
        # 1. Project policy latent to a small 4x4 spatial feature map
        self.latent_proj = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 2. Project ViT spatial features from 384 to 256 channels
        self.spatial_proj = nn.Conv2d(spatial_dim, 256, kernel_size=1)
        
        # 3. Transposed Conv layers for upsampling (4 layers, each 2x upsampling)
        # 4 -> 8 -> 16 -> 32 -> 64 -> ... wait, we want to upsample from H_p, W_p to 16 * H_p, 16 * W_p
        # So we can upsample the (4, 4) latent map to (H_p, W_p) first,
        # add the projected spatial context, and then apply 4 transposed conv layers to upsample by 16.
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 2x
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 2x
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, max_layers + 1, kernel_size=4, stride=2, padding=1), # 2x -> outputs (max_layers + 1) channels
            nn.Sigmoid()
        )

    def forward(self, latent, spatial_context, target_h, target_w, active_layers_mask=None):
        """
        Args:
            latent: (B, 256) continuous policy action
            spatial_context: (B, N_patches, 384) from ViT
            target_h: Target height of board raster (int)
            target_w: Target width of board raster (int)
            active_layers_mask: (B, 8) tensor of active layers (1.0 active, 0.0 inactive)
        """
        B = latent.shape[0]
        H_p = target_h // 16
        W_p = target_w // 16
        
        # 1. Project and reshape latent to (B, 256, 4, 4)
        x_latent = self.latent_proj(latent).view(B, 256, 4, 4)
        
        # Upsample latent to match patch dimensions (H_p, W_p)
        x_latent = F.interpolate(x_latent, size=(H_p, W_p), mode='bilinear', align_corners=False)
        
        # 2. Reshape and project spatial context to (B, 256, H_p, W_p)
        # spatial_context shape is (B, H_p * W_p, 384)
        x_spatial = spatial_context.transpose(1, 2).view(B, 384, H_p, W_p)
        x_spatial = self.spatial_proj(x_spatial)
        
        # Fuse latent and spatial context
        x = x_latent + x_spatial
        
        # 3. Upsample by 16x using transposed convolutions
        x = self.up1(x) # (B, 128, 2 * H_p, 2 * W_p)
        x = self.up2(x) # (B, 64, 4 * H_p, 4 * W_p)
        x = self.up3(x) # (B, 32, 8 * H_p, 8 * W_p)
        x = self.up4(x) # (B, max_layers + 1, 16 * H_p, 16 * W_p) = (B, 9, target_h, target_w)
        
        # Interpolate in case target dimensions don't exactly match multiples of 16
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
        # Mask out inactive layer heatmaps
        if active_layers_mask is not None:
            # active_layers_mask is (B, 8), add channels dimension for broadcasting
            layer_mask = active_layers_mask.unsqueeze(-1).unsqueeze(-1) # (B, 8, 1, 1)
            # Create a mask for via probability too (via probability is always active)
            via_mask = torch.ones(B, 1, 1, 1, device=x.device)
            full_mask = torch.cat((layer_mask, via_mask), dim=1) # (B, 9, 1, 1)
            
            x = x * full_mask
            
        return x
