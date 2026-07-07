"""
demand_predictor.py
===================
The learned "planner" for the completing router (Stage 3 of the overhaul).

It is a **supervised, high-resolution dense predictor**, not an RL policy. Given the current board
and the pins of the next K nets in the sequence, it predicts a **per-layer "future demand" heatmap**
— where those upcoming nets will route — so the current net can be routed to *leave space* for them
(A* cost = obstacles + λ·demand). See router-completing-router-overhaul.md §3.

Design choices (kept optimized):
  * **U-Net** dense predictor with skip connections → genuinely high-res output (sharp pad/obstacle
    detail preserved), unlike upsampling a coarse patch grid.
  * **Fully convolutional**: any board size works; input is padded internally to a multiple of
    2^depth and the output is cropped back. Batches are formed by padding to a common canvas.
  * Fixed **8-layer** I/O (MAX_LAYERS); inactive layers are masked out, so one model serves 1–8 layer
    boards.
  * GroupNorm (batch-size robust) + SiLU. Returns **logits** (use BCEWithLogits); `predict()` applies
    sigmoid and the active-layer mask.

Input channels (MAX_LAYERS * 3 = 24):
    [0:8]    per-layer occupancy the router sees (pads + obstacles + already-routed copper)
    [8:16]   current net's pins, PER-LAYER (disc-stamped on the pin's own layer; through-hole
             pins, layer == -1, are stamped on every active layer)
    [16:24]  FUTURE nets' pins, PER-LAYER (same convention). Trained with a variable horizon:
             usually the next k <= K nets, sometimes ALL remaining nets — so at inference this
             channel may legally hold anything from one net's pins up to every unrouted net's
             pins. The full-horizon mode is what makes the model usable as a whole-board demand
             predictor: warm-starting the rip-up router's congestion cost field and computing
             the routability/overflow score (predicted demand vs. capacity).
Output channels (8): per-layer demand in [0,1] after sigmoid.

Pin channels are per-layer (not a single flattened 2D map) so the model is actually told which
layer an upcoming pin sits on, instead of having to infer it indirectly — it needs this because the
label it's trained against (future route occupancy) is itself per-layer and via-crossings show up
as demand appearing in two different layer channels at the same (x, y).
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LAYERS = 8
IN_CHANNELS = MAX_LAYERS * 3
PIN_STAMP_RADIUS = 3


# ── model ────────────────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        g = groups if cout % groups == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DemandPredictor(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, out_layers: int = MAX_LAYERS,
                 base: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.out_layers = out_layers
        chs = [base * (2 ** i) for i in range(depth + 1)]  # e.g. [32, 64, 128, 256]

        self.inc = _ConvBlock(in_channels, chs[0])
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList([_ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.up_convs = nn.ModuleList([_ConvBlock(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_layers, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, H, W) -> (B, out_layers, H, W) LOGITS."""
        _, _, H, W = x.shape
        m = 2 ** self.depth
        ph, pw = (m - H % m) % m, (m - W % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))

        y = skips[-1]
        for i, (up, up_conv) in enumerate(zip(self.ups, self.up_convs)):
            y = up(y)
            y = up_conv(torch.cat([y, skips[-2 - i]], dim=1))

        y = self.head(y)
        return y[..., :H, :W]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, active_layers_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sigmoid probabilities with inactive layers zeroed. active_layers_mask: (B, MAX_LAYERS)."""
        p = torch.sigmoid(self.forward(x))
        if active_layers_mask is not None:
            p = p * active_layers_mask[:, :, None, None].to(p.dtype)
        return p


# ── encoding helpers (numpy; used for dataset gen and at inference) ────────────────
def _stamp_disc(grid: np.ndarray, cx: int, cy: int, radius: int, value: float = 1.0):
    H, W = grid.shape
    r = int(radius)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    np.maximum(grid[y0:y1, x0:x1], mask * value, out=grid[y0:y1, x0:x1])


def _pin_layers(pin_layer: int, num_layers: int):
    """Which of the board's active layers a pin's disc should be stamped onto. Through-hole pins
    (layer == -1) span every active layer; SMD pins stamp only their own layer. Mirrors
    train_demand_predictor.base_occupancy's layers_of() so pins and occupancy agree on convention."""
    if pin_layer == -1:
        return range(min(num_layers, MAX_LAYERS))
    return [pin_layer] if 0 <= pin_layer < MAX_LAYERS else []


def rasterize_pins(pins, H: int, W: int, num_layers: int, radius: int = PIN_STAMP_RADIUS) -> np.ndarray:
    """Per-layer pin map: (MAX_LAYERS, H, W), each pin disc-stamped onto its own layer (or every
    active layer if through-hole) instead of a single layer-agnostic 2D map."""
    grid = np.zeros((MAX_LAYERS, H, W), dtype=np.float32)
    for p in pins:
        for l in _pin_layers(p.layer, num_layers):
            _stamp_disc(grid[l], int(p.global_x), int(p.global_y), radius)
    return grid


def rasterize_segments_per_layer(segments, H: int, W: int, num_layers: int,
                                 line_radius: int = 1) -> np.ndarray:
    """Stamp trace-segment centrelines (dilated by line_radius) into a per-layer occupancy grid."""
    out = np.zeros((MAX_LAYERS, H, W), dtype=np.float32)
    for seg in segments:
        layer = int(getattr(seg, "layer", 0))
        if not (0 <= layer < min(num_layers, MAX_LAYERS)):
            continue
        x0, y0 = int(seg.start_x), int(seg.start_y)
        x1, y1 = int(seg.end_x), int(seg.end_y)
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        for x, y in zip(xs, ys):
            _stamp_disc(out[layer], x, y, line_radius)
    return out


def encode_input(occupancy_per_layer: np.ndarray, current_pins, future_pins,
                 H: int, W: int, num_layers: int) -> np.ndarray:
    """Build the (IN_CHANNELS, H, W) model input.

    occupancy_per_layer: (num_layers, H, W) occupancy the router sees for the current net
                         (pads + obstacles + already-routed copper). Padded to MAX_LAYERS here.
    """
    x = np.zeros((IN_CHANNELS, H, W), dtype=np.float32)
    n = min(occupancy_per_layer.shape[0], MAX_LAYERS)
    x[:n] = occupancy_per_layer[:n]
    x[MAX_LAYERS:MAX_LAYERS * 2] = rasterize_pins(current_pins, H, W, num_layers)
    x[MAX_LAYERS * 2:MAX_LAYERS * 3] = rasterize_pins(future_pins, H, W, num_layers)
    return x


def occupancy_stack(board_state, num_layers: int) -> np.ndarray:
    """Per-layer occupancy (pads + obstacles + routed copper) as seen right now, padded to MAX_LAYERS."""
    occ = np.zeros((MAX_LAYERS, board_state.height, board_state.width), dtype=np.float32)
    for l in range(min(num_layers, MAX_LAYERS)):
        occ[l] = board_state.get_occupancy(l).astype(np.float32)
    return occ


def active_layers_mask(num_layers: int) -> np.ndarray:
    m = np.zeros(MAX_LAYERS, dtype=np.float32)
    m[:min(num_layers, MAX_LAYERS)] = 1.0
    return m
