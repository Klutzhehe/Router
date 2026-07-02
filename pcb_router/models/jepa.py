import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pcb_router.models.vit_encoder import TransformerBlock, ViTEncoder

class SpatialJEPA(nn.Module):
    def __init__(
        self,
        vit_encoder: ViTEncoder,
        predictor_layers: int = 6,
        predictor_dim: int = 384,
        predictor_heads: int = 6,
        num_nets_max: int = 100,
        net_embed_dim: int = 128,
        heatmap_latent_dim: int = 256,
        ema_decay: float = 0.996,
        vicreg_weight: float = 0.1,
        variance_weight: float = 25.0,
        invariance_weight: float = 25.0,
        covariance_weight: float = 1.0
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.vicreg_weight = vicreg_weight
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight
        self.predictor_dim = predictor_dim
        
        # Action projection
        self.net_embedding = nn.Embedding(num_nets_max, net_embed_dim)
        self.action_proj = nn.Linear(net_embed_dim + heatmap_latent_dim, predictor_dim)
        
        # Predictor: processes z_t (B, N_patches, 384) + action_token (B, 1, 384)
        self.predictor_blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, predictor_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(predictor_layers)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        
        # Target Encoder (EMA copy of vit_encoder)
        self.target_encoder = copy.deepcopy(vit_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def update_target_encoder(self):
        """EMA update of target encoder weights from online vit_encoder"""
        # We need to find the online encoder. Since it's passed at init, we'll need to pass it or keep a reference
        # We assume the parent trainer has the online encoder, so we can update parameters from it
        pass

    def update_target_weights(self, online_encoder: nn.Module):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_encoder.parameters(), online_encoder.parameters()):
                target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def predict(self, z_t: torch.Tensor, action: tuple):
        """
        Predict z_{t+1} given z_t and action
        Args:
            z_t: State embedding (B, N_patches, dim)
            action: Tuple (net_index, heatmap_latent)
                - net_index: (B,) discrete tensor
                - heatmap_latent: (B, 256) continuous tensor
        """
        net_idx, heatmap_latent = action
        B = z_t.shape[0]
        
        # 1. Project action to 384-dim
        net_emb = self.net_embedding(net_idx) # (B, net_embed_dim)
        act_concat = torch.cat((net_emb, heatmap_latent), dim=-1) # (B, net_embed_dim + 256)
        action_token = self.action_proj(act_concat).unsqueeze(1) # (B, 1, dim)
        
        # 2. Prepend action token to state patches
        # Input to predictor: (B, N_patches + 1, dim)
        x = torch.cat((action_token, z_t), dim=1)
        
        for block in self.predictor_blocks:
            x = block(x)
            
        x = self.predictor_norm(x)
        
        # 3. Extract predicted z_{t+1} (discard the action token output)
        predicted_z = x[:, 1:]
        return predicted_z

    def compute_loss(self, current_raster: torch.Tensor, action: tuple, next_raster: torch.Tensor, online_encoder: nn.Module):
        """
        Compute JEPA prediction loss + VICReg regularization
        """
        # Encode current state with online encoder
        with torch.no_grad():
            z_t, _ = online_encoder(current_raster)
            
        # Encode next state with target (EMA) encoder
        with torch.no_grad():
            z_next_target, _ = self.target_encoder(next_raster)
            
        # Predict next state embedding
        z_next_pred = self.predict(z_t, action)
        
        # 1. Invariance Loss (MSE between predicted and target)
        loss_invariance = F.mse_loss(z_next_pred, z_next_target)
        
        # 2. VICReg losses
        loss_variance = self.compute_variance_loss(z_next_pred) + self.compute_variance_loss(z_next_target)
        loss_covariance = self.compute_covariance_loss(z_next_pred) + self.compute_covariance_loss(z_next_target)
        
        loss_total = (
            self.invariance_weight * loss_invariance +
            self.variance_weight * loss_variance +
            self.covariance_weight * loss_covariance
        )
        
        return loss_total, {
            'loss_jepa_total': loss_total.item(),
            'loss_invariance': loss_invariance.item(),
            'loss_variance': loss_variance.item(),
            'loss_covariance': loss_covariance.item()
        }

    def compute_variance_loss(self, x: torch.Tensor):
        # x is (B, N, C)
        # Calculate variance across batch dimension
        B, N, C = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Reshape to (B * N, C) or compute per patch
        std = torch.sqrt(x.var(dim=0) + 1e-4) # (N, C)
        loss = torch.mean(F.relu(1.0 - std))
        return loss

    def compute_covariance_loss(self, x: torch.Tensor):
        # x is (B, N, C)
        B, N, C = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Flatten patches into batch for cov calculation: (B * N, C)
        # Standardize x
        x_flat = x.reshape(-1, C)
        x_flat = x_flat - x_flat.mean(dim=0, keepdim=True)
        
        cov = (x_flat.T @ x_flat) / (B * N - 1) # (C, C)
        
        # Sum of square of off-diagonal elements
        diag = torch.diag(cov)
        cov_off_diag = cov - torch.diag(diag)
        loss = torch.sum(cov_off_diag ** 2) / C
        return loss
