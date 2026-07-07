"""
train_demand_predictor.py
=========================
Supervised training for the DemandPredictor (Stage 3 of the completing-router overhaul).

Task: given the board + current net + the next-K nets' pins, predict the per-layer occupancy of
where those next-K nets actually route. Ground truth comes from boards completed by the Stage-1
RipUpRerouteRouter.

Optimizations:
  * **Compact storage, lazy rasterization.** We store only (board, net order, cell routes) per board
    and rasterize (input, label) tensors on the fly in the Dataset — no giant tensors on disk.
  * **Batch-max padding** (not a fixed canvas): a batch is padded to its own max H/W rounded to a
    multiple of 2^depth, with a spatial-valid mask so loss ignores padding. No wasted compute on
    small boards.
  * **Cheap incremental occupancy:** a net-agnostic base occupancy (pads/obstacles) is computed once
    per board; already-routed nets are added by stamping their route cells.
"""

import os
import sys
import math
import pickle
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _here not in sys.path:
    sys.path.insert(0, _here)

import yaml

from pcb_router.data.board_generator import BoardGenerator
# NOTE: RipUpRerouteRouter is imported lazily inside generate_dataset() so that training / the
# Dataset class do not depend on the routing package (and aren't blocked by any WIP there).
from pcb_router.models.demand_predictor import (
    DemandPredictor, MAX_LAYERS, IN_CHANNELS, PIN_STAMP_RADIUS,
    rasterize_pins, _stamp_disc,
)

PAD_STAMP_RADIUS = 4
LINE_STAMP_RADIUS = 1


# ── occupancy / label rasterization ───────────────────────────────────────────────
def base_occupancy(board) -> np.ndarray:
    """Net-agnostic static occupancy (pads + obstacles + keep-outs), per layer, padded to MAX_LAYERS.
    Computed once per board and reused for every net's input."""
    H, W, L = board.height, board.width, board.num_layers
    occ = np.zeros((MAX_LAYERS, H, W), dtype=np.float32)

    def layers_of(layer):
        return range(min(L, MAX_LAYERS)) if layer == -1 else ([layer] if 0 <= layer < MAX_LAYERS else [])

    for pin in board.pins.values():
        for l in layers_of(pin.layer):
            _stamp_disc(occ[l], int(pin.global_x), int(pin.global_y), PAD_STAMP_RADIUS)
    for obs in getattr(board, "obstacles", []):
        x0, y0 = max(0, int(obs.x)), max(0, int(obs.y))
        x1, y1 = min(W, int(obs.x + obs.width)), min(H, int(obs.y + obs.height))
        for l in layers_of(getattr(obs, "layer", -1)):
            occ[l, y0:y1, x0:x1] = 1.0
    for ko in getattr(board, "keep_out_zones", []):
        x0, y0 = max(0, int(ko.x)), max(0, int(ko.y))
        x1, y1 = min(W, int(ko.x + ko.width)), min(H, int(ko.y + ko.height))
        for l in layers_of(getattr(ko, "layer", -1)):
            occ[l, y0:y1, x0:x1] = 1.0
    return occ


def stamp_route(dst: np.ndarray, cells, H: int, W: int, radius: int = LINE_STAMP_RADIUS):
    """Stamp a route's cells (list of (x, y, layer)) into a (MAX_LAYERS, H, W) grid in place."""
    for (x, y, l) in cells:
        li = int(l)
        if 0 <= li < MAX_LAYERS:
            _stamp_disc(dst[li], int(x), int(y), radius)


# ── dataset generation (compact) ──────────────────────────────────────────────────
def generate_dataset(stage_names: List[str], boards_per_stage: int, out_path: str,
                     max_iters: int = 6, seed0: int = 0, save_every: int = 10):
    """Generate (and incrementally save) routed boards for the demand-predictor dataset.

    Resumable: if `out_path` already has boards saved (e.g. from a session that got
    disconnected), they're loaded and generation continues on top of them instead of
    restarting from zero. Saves every `save_every` new boards (atomically, via a temp file +
    rename) plus once at the end, so a Colab disconnect mid-run only costs the boards generated
    since the last checkpoint, not the whole run.
    """
    from pcb_router.routing.rip_up_router import RipUpRerouteRouter  # lazy: only needed to gen data
    cur = yaml.safe_load(open(os.path.join(_here, "configs/curriculum.yaml")))
    stages = {s["name"]: s for s in cur["stages"]}

    samples: List[dict] = []
    next_seed = seed0
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "samples" in loaded:
            samples = loaded["samples"]
            next_seed = loaded.get("next_seed", seed0 + len(samples))
        else:
            samples = loaded  # legacy bare-list pickle from before resumable saving existed
            next_seed = seed0 + len(samples)
        print(f"resuming: {len(samples)} boards already saved at {out_path}", flush=True)

    seed = next_seed

    def save():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"samples": samples, "next_seed": seed}, f)
        os.replace(tmp_path, out_path)

    for sname in stage_names:
        stage = stages[sname]
        made = sum(1 for s in samples if s.get("stage") == sname)
        while made < boards_per_stage:
            random.seed(seed)
            cfg = BoardGenerator.from_curriculum_stage(stage)
            cfg.seed = seed
            seed += 1
            board = BoardGenerator().generate(cfg)
            res = RipUpRerouteRouter(board, max_iterations=max_iters).route()
            # Keep only boards where every net routed, so labels are clean full routes.
            if res["completed"] != res["total"] or res["total"] < 2:
                continue
            order = [n.id for n in board.nets]
            routes = {nid: [(int(x), int(y), int(l)) for (x, y, l) in p]
                      for nid, p in res["routes"].items() if p}
            samples.append({"board": board, "order": order, "routes": routes, "stage": sname})
            made += 1
            print(f"  [{sname}] board {made}/{boards_per_stage} "
                  f"({board.width}x{board.height} L{board.num_layers} nets={res['total']})", flush=True)
            if len(samples) % save_every == 0:
                save()
                print(f"  [checkpoint] saved {len(samples)} boards -> {out_path}", flush=True)

    save()
    print(f"Saved {len(samples)} boards -> {out_path}", flush=True)
    return samples


# ── torch Dataset (lazy rasterization) ─────────────────────────────────────────────
class DemandDataset(Dataset):
    def __init__(self, samples, K: int = 3):
        # Tolerate loading the raw pickle straight off disk: generate_dataset saves
        # {'samples': [...], 'next_seed': int} so runs are resumable; unwrap it here so both
        # notebook cells (which just do pickle.load(...)) keep working unchanged.
        if isinstance(samples, dict) and "samples" in samples:
            samples = samples["samples"]
        self.K = K
        self.samples = samples
        # Precompute base occupancy per board (once) and flat index of (board, net position).
        self.base_occ = [base_occupancy(s["board"]) for s in samples]
        self.index: List[Tuple[int, int]] = []
        for bi, s in enumerate(samples):
            for i in range(len(s["order"]) - 1):  # need at least one future net
                self.index.append((bi, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        bi, i = self.index[idx]
        s = self.samples[bi]
        board, order, routes = s["board"], s["order"], s["routes"]
        H, W, L = board.height, board.width, board.num_layers
        pins = board.pins

        # Input occupancy = base (pads/obstacles) + already-routed nets (0..i-1).
        occ = self.base_occ[bi].copy()
        for nid in order[:i]:
            if routes.get(nid):
                stamp_route(occ, routes[nid], H, W)

        cur_net = next(n for n in board.nets if n.id == order[i])
        cur_pins = [pins[p] for p in cur_net.pin_ids]

        future_ids = order[i + 1: i + 1 + self.K]
        future_pins = [pins[p] for nid in future_ids
                       for p in next(n for n in board.nets if n.id == nid).pin_ids]

        x = np.zeros((IN_CHANNELS, H, W), dtype=np.float32)
        x[:MAX_LAYERS] = occ
        x[MAX_LAYERS:MAX_LAYERS * 2] = rasterize_pins(cur_pins, H, W, L, PIN_STAMP_RADIUS)
        x[MAX_LAYERS * 2:MAX_LAYERS * 3] = rasterize_pins(future_pins, H, W, L, PIN_STAMP_RADIUS)

        # Label = occupancy of the future nets' actual routes.
        y = np.zeros((MAX_LAYERS, H, W), dtype=np.float32)
        for nid in future_ids:
            if routes.get(nid):
                stamp_route(y, routes[nid], H, W)

        lmask = np.zeros(MAX_LAYERS, dtype=np.float32)
        lmask[:min(L, MAX_LAYERS)] = 1.0
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(lmask)


def collate_pad(batch, mult: int = 8):
    """Pad a batch to its max H/W (rounded to `mult`), with a spatial-valid mask."""
    maxH = max(b[0].shape[1] for b in batch)
    maxW = max(b[0].shape[2] for b in batch)
    maxH = math.ceil(maxH / mult) * mult
    maxW = math.ceil(maxW / mult) * mult
    xs, ys, lm, sm = [], [], [], []
    for x, y, mask in batch:
        _, h, w = x.shape
        pad = (0, maxW - w, 0, maxH - h)
        xs.append(F.pad(x, pad))
        ys.append(F.pad(y, pad))
        lm.append(mask)
        s = torch.zeros(1, maxH, maxW)
        s[:, :h, :w] = 1.0
        sm.append(s)
    return torch.stack(xs), torch.stack(ys), torch.stack(lm), torch.stack(sm)


# ── loss ───────────────────────────────────────────────────────────────────────────
def demand_loss(logits, target, lmask, smask, pos_weight: float = 20.0):
    """Weighted BCE over active layers + valid spatial region. Positives (route cells) are rare,
    so up-weight them."""
    w = torch.where(target > 0.5, torch.full_like(target, pos_weight), torch.ones_like(target))
    w = w * lmask[:, :, None, None] * smask  # zero out inactive layers and padded pixels
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="sum")
    denom = w.sum().clamp_min(1.0)
    return bce / denom


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernel" in str(type(get_ipython()))
    except Exception:
        return False


def _render_progress(sample, pred, losses, epoch, out_dir, show, layer=0):
    """Save (and, in a notebook, live-display) a loss curve + prediction panels for one sample.
    sample=(x, y) numpy tensors; pred=(MAX_LAYERS, H, W) predicted demand for the same sample."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = sample
    def npy(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
    x, y, pred = npy(x), npy(y), npy(pred)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.5])
    axl = fig.add_subplot(gs[0, :])
    axl.plot(range(1, len(losses) + 1), losses, "-o", ms=3, color="#3B82F6")
    axl.set_title(f"training loss (epoch {epoch}, latest {losses[-1]:.4f})")
    axl.set_xlabel("epoch"); axl.grid(alpha=0.3)

    panels = [(x[MAX_LAYERS], "current net pins"),
              (x[MAX_LAYERS + 1], "next-K nets pins"),
              (pred[layer], f"PREDICTED demand (L{layer})"),
              (y[layer], f"ACTUAL future routes (L{layer})")]
    for j, (im, title) in enumerate(panels):
        a = fig.add_subplot(gs[1, j])
        a.imshow(im, origin="lower", cmap="inferno" if j >= 2 else "viridis")
        a.set_title(title, fontsize=9); a.axis("off")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}.png"), dpi=90,
                bbox_inches="tight", facecolor="white")
    if show and _in_notebook():
        from IPython.display import clear_output, display
        clear_output(wait=True)
        display(fig)
    plt.close(fig)


def train(dataset_path: str, epochs: int = 20, batch_size: int = 8, lr: float = 2e-3,
          K: int = 3, device: str = "auto", ckpt: str = "checkpoints/demand_predictor.pt",
          viz_every: int = 5, viz_sample: int = 0, viz_dir: str = "checkpoints/viz",
          show: bool = True):
    device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                          (device if device != "auto" else "cpu"))
    with open(dataset_path, "rb") as f:
        samples = pickle.load(f)
    ds = DemandDataset(samples, K=K)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad,
                    num_workers=0, drop_last=False)
    model = DemandPredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f"train: {len(ds)} samples from {len(samples)} boards, device={device}", flush=True)

    # Fixed sample for the live visualizer (its input never changes, only the prediction does).
    viz_sample = min(viz_sample, len(ds) - 1)
    vx, vy, vlm = ds[viz_sample]
    vxb, _, vlmb, _ = collate_pad([ds[viz_sample]])

    losses = []
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y, lm, sm in dl:
            x, y, lm, sm = x.to(device), y.to(device), lm.to(device), sm.to(device)
            loss = demand_loss(model(x), y, lm, sm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        losses.append(tot / max(1, nb))
        print(f"epoch {ep+1:3d}  loss {losses[-1]:.4f}", flush=True)

        if viz_every and ((ep + 1) % viz_every == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                pred = model.predict(vxb.to(device), vlmb.to(device))[0]
            # crop prediction back to the un-padded sample size
            pred = pred[:, :vx.shape[1], :vx.shape[2]].cpu()
            _render_progress((vx, vy), pred, losses, ep + 1, viz_dir, show)

    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "K": K}, ckpt)
    print(f"saved -> {ckpt}  (progress PNGs in {viz_dir}/)", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true", help="generate dataset")
    ap.add_argument("--stages", nargs="+", default=["s10_via_plus_multi_net"])
    ap.add_argument("--boards", type=int, default=40)
    ap.add_argument("--data", default="data/demand_dataset.pkl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--viz_every", type=int, default=5)
    ap.add_argument("--no_show", action="store_true", help="save viz PNGs but don't display")
    args = ap.parse_args()
    if args.gen:
        generate_dataset(args.stages, args.boards, args.data)
    else:
        train(args.data, epochs=args.epochs, batch_size=args.batch, lr=args.lr, K=args.K,
              viz_every=args.viz_every, show=not args.no_show)
