# Router Overhaul — From RL Step-Policy to a Completing Router

**Status:** In progress. Stage 1 (rip-up-and-reroute engine) skeleton built and validated; open
problems identified. Everything below reflects decisions and findings as of this overhaul.

---

## 0. TL;DR

We are pivoting the project's goal and architecture:

- **Goal:** a system that **actually completes boards** (routes every net legally), not a demo of
  an RL agent learning to draw wires.
- **Core reframe:** *completion is guaranteed by classical search (A\* + rip-up-and-reroute); ML
  makes it smarter, not the other way around.*
- **New role for the "Dreamer"/world-model idea:** it is a **planner/predictor** that guides A\*,
  not a cell-by-cell drawing policy. Concretely: a **learned predictor of where the next few nets
  will route**, so the current net leaves space and avoids clashing.
- **Dropped:** the autoregressive per-cell step policy and its Dreamer actor-critic training loop.
- **Kept and promoted:** `AStarPathfinder`, `MeanderInserter`, the GNN/ViT board encoders (repurposed
  for prediction), diff-pair board generation.

---

## 1. Why the previous architecture was the wrong shape

The previous system used a **DreamerV3-style RL agent** whose policy moved a cursor **one cell at a
time** to draw each trace, trained on imagined latent rollouts.

Problems (diagnosed over an extended debugging session):

1. **A reactive step policy can't guarantee completion.** Perfect/complete routing needs global
   planning and backtracking (rip up a net, reroute others, retry). A policy that picks one cell at
   a time structurally can't do that.
2. **Legality was being asked of the wrong component.** A\* trivially guarantees legal, complete
   per-net paths; forcing a neural policy to learn legality is backwards.
3. **Training was slow, fragile, and signal-limited.** Reward shaping, endgame stalling,
   obstacle-blind imagination, exploration noise, curriculum thresholds, and a completion-metric
   bug all had to be fixed just to route **one** net ~85–90% of the time on an empty board.
4. **Perception was coarse.** The step policy saw a 3×3 crop of a 8×8 ViT patch grid — it literally
   could not resolve small pads it had to avoid.

Conclusion: keep the *world-model / prediction* idea (it's genuinely useful), drop the *cell-stepping
RL actor* (it's the wrong tool for completion).

---

## 2. Target architecture

```
            ┌─────────────────────────────────────────────────────────────┐
            │  INPUTS (per board)                                          │
            │  • Board: components, pins, obstacles, keep-outs, layers     │
            │  • Net sequence (ORDER GIVEN — random in training, LLM at     │
            │    inference; we do NOT learn ordering)                       │
            │  • Per-net constraints: trace width, diff-pair flag, length   │
            └─────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  FOR EACH net in the given order:                                          │
   │                                                                            │
   │   1. DEMAND PREDICTOR (learned, high-res)                                  │
   │      GNN/ViT + U-Net → per-layer "future demand" heatmap:                  │
   │      where will the NEXT K nets route? (all-at-once, v1)                    │
   │                                                                            │
   │   2. A* ROUTES THIS NET  (executor, guarantees legality)                   │
   │      cost = base_obstacles  +  λ · future_demand                           │
   │      → the net steers OUT of where upcoming nets need to go (leaves space) │
   │                                                                            │
   │   3. Apply constraints: trace width (clearance), diff-pair (coupled),      │
   │      length tuning (MeanderInserter, reserve slack)                        │
   └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  RIP-UP-AND-REROUTE  (negotiated congestion — the completion engine)       │
   │  Residual conflicts (shared cells) → raise congestion cost, rip up losers, │
   │  reroute, repeat until clean or budget. Guarantees completion where it     │
   │  exists. Catches whatever the predictor mispredicted.                      │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Division of labor:**

| Concern | Owner |
|---|---|
| Legal, complete per-net path | A\* (`AStarPathfinder.find_path`) |
| Completing the whole board / conflict resolution | Rip-up-and-reroute (negotiated congestion) |
| *Proactively* leaving room for future nets | Learned demand predictor (the "planner") |
| Net ordering | **External** (random in training / LLM at inference) — not learned |
| Length tuning | `MeanderInserter` (post-route) |
| Differential pairs | Coupled routing |
| Trace width | A\* clearance |

---

## 3. The learned planner: predict-then-route

### 3.1 The behavior we want
> "When I route this net, I know the next couple of nets will need space near me, so I leave room
> for them — and I can look a few nets ahead and estimate how they'll route to avoid clashing."

### 3.2 How it's realized
1. Look at the next **K** nets in the sequence (their pin locations are known).
2. **Predict a per-layer "future demand" heatmap** — where those K nets will likely route.
3. Route the current net with A\* on `cost = obstacles + λ · future_demand`. It naturally avoids the
   corridors future nets need → it leaves space.
4. Route the next net, re-predict, repeat.

### 3.3 Why this is the big unlock — it's SUPERVISED, not RL
The predictor's job is: *"given the board + this net + the next K nets' pins, predict where those K
nets will route."* That's a **prediction task with ground-truth labels**, not reward-chasing RL.

- **Data:** completed boards (from the Stage-1 rip-up-reroute router, or the existing BC dataset).
- **Label:** for net *i*, the actual occupancy of nets *i+1 … i+K*'s real routes.
- **Model:** GNN/ViT encode board + relevant pins → U-Net decodes a per-layer demand heatmap.
- **Loss:** predict the heatmap (weighted BCE). Stable, fast, no exploration.

This is what the old heatmap decoder was *reaching* for, but it never had a clean objective under RL.
Now the objective is crisp: **predict the future, route to avoid it.**

### 3.4 High-resolution demand map (U-Net)
The old heatmap looked "8×8" because its spatial input was the coarse ViT patch grid — upsampling a
blurry 8×8 doesn't create detail. For a genuinely sharp demand map we use a **U-Net dense predictor**:

- Input: full-resolution board raster (+ current/next-K pins as extra channels).
- Down/up-sampling with **skip connections** that carry sharp early-layer detail (pad edges,
  obstacle boundaries) to the full-resolution output.
- Output: **per-layer** demand map `(num_layers, H, W)` at (or near) routing resolution.
- The GNN net-connectivity embeddings condition the U-Net bottleneck.

Resolution knob: full-res on small boards; drop to half-res + 2× upsample only if memory bites at 8
layers.

### 3.5 All-at-once vs sequential (design decision: **all-at-once for v1**)
- **All-at-once:** one forward pass predicts a single blended demand blob for the next K nets. Simple,
  fast, stable. Fuzzy but enough to "leave space." **← v1.**
- **Sequential (rollout):** imagine net i+1, drop it on the board, imagine i+2 given i+1, … up to i+K.
  Sharper (models interactions/order), but K calls, more complex, errors compound. **← later upgrade;
  this is the true recurrent world-model imagination.**

---

## 4. Constraints (all are *given* per-net inputs; the router satisfies them)

- **Trace width** — net/net-class property. Changes A\* clearance (inflate obstacles by
  `width/2 + clearance`); demand guidance should leave wider corridors for wide nets.
- **Differential pairs** — route P/N **coupled** (route P, route N alongside with matched spacing;
  match intra-pair length via meander). Board generator already produces diff pairs.
- **Length tuning** — per-net target length; route with **slack**, then `MeanderInserter.insert_meanders`
  adds meanders to hit target within tolerance. Guidance should route length-target nets loosely.

### Multi-layer (6–8 layers)
- Already representable: board raster has **8 copper channels**; A\* routes in 3D `(x, y, layer)` with
  vias; curriculum scales `num_layers` up to 6.
- More layers = more routing resource = **easier** completion (via-escape around congestion).
- Demand prediction is **per-layer**. Via/layer assignment starts simple (A\*'s existing via cost),
  smarter assignment is a later refinement.
- **Out of scope (real-DDR signoff):** impedance control, reference planes, per-layer velocity
  compensation, fly-by topology, SI simulation.

---

## 5. Scope reality check — DDR

Real DDR (tight group length-match to ±2–5 mil, propagation-delay matching across layers, fly-by
topology, controlled impedance, SI signoff) is **out of reach in this timeframe and not what this
architecture targets.** What *is* achievable and honest:

> A completing router that keeps **diff pairs** coupled and **length-matches a bus group to a
> tolerance**, on synthetic DDR-like boards. Call it "DDR-style constrained routing," not
> signoff-grade DDR.

---

## 6. One-week build order

| Stage | Days | What | Reuses | New | Milestone |
|---|---|---|---|---|---|
| **1. Completion engine** | 1–2 | `RipUpRerouteRouter`: route all nets (given order) via A\* + negotiated congestion; rip up conflicts, reroute until clean | `AStarPathfinder.find_path`, `BoardState`, `TraceGenerator` | orchestrator + congestion cost | Completes a routable multi-net board, no ML |
| **2. Constraints** | 3 | Trace width, diff pairs (coupled), length tuning (meander + slack) | `MeanderInserter`, diff-pair board gen | width→A\* cost, coupled router | Routes a length-matched, diff-pair board |
| **3. Demand predictor** | 4–5 | High-res U-Net: board + current/next-K pins → per-layer demand. Supervised on completed boards | GNN/ViT, board raster | U-Net predictor + dataset gen | Predicts where future nets route |
| **4. Predict-then-route** | 6 | Feed demand map as A\* cost bias (all-at-once) → leaves space | Stages 1+3 | integration | Fewer rip-up iterations vs Stage-1 baseline |
| **5. Eval + polish** | 7 | Metrics (completion %, DRC, rip-up count, length error), 6–8 layer runs, demo | all | eval harness | Numbers + demo |

Principle: **each stage is independently a working thing.** Stage 1 alone gives an app that completes
boards; ML is added value measured against it.

---

## 7. Stage 1 — current status

**File:** `pcb_router/routing/rip_up_router.py` → `class RipUpRerouteRouter`.

### 7.1 How it works
- Routes nets in a given order using `AStarPathfinder.find_path`.
- **Pads / obstacles / keep-outs are HARD blocks; other nets' TRACES are SOFT** — achieved by routing
  against a **trace-free** `BoardState` (so `get_occupancy` only returns pads/obstacles) while tracking
  trace usage separately.
- Congestion is injected through A\*'s heatmap channel: `h_val = 1 / (1 + alpha·(history + usage))`
  (free cell → h_val≈1 → base cost; congested → h_val→0 → up to `(1+heatmap_weight)×` cost).
- Each iteration rips up and reroutes every net; cells shared by ≥2 nets get a growing `history`
  penalty so nets peel apart on the next pass. Stops when no cell is shared (converged) or budget hit.
- Materializes final routes into a `BoardState` via `TraceGenerator.generate_traces` +
  `add_routed_trace`.

### 7.2 Validated (works)
- 2-net board: **2/2**, converged 1 iter, 43 trace segments placed.
- 5-net board (seed with a routable layout): **5/5**, converged 1 iter.
- 4-net board (routable layout): **4/4**, converged.
- Negotiated-congestion loop, trace materialization, and stats all functioning.

### 7.3 Open problems found (the real work ahead)
1. **Single-layer pad-sealing (genuine, not a bug).** On congested single-layer boards, other nets'
   pad **clearance rings** can completely seal a target pin → that net is genuinely unroutable on one
   layer. Confirmed: a failing net returns `None` even with a **2,000,000-node** A\* budget in 0.2s
   (small sealed reachable region). Router correctly reports e.g. 3/4.
2. **Via-escape is NOT happening on multi-layer boards.** On 2-layer boards, completion did **not**
   improve and **`vias = 0`** — the router stays on one layer and fails instead of hopping up to route
   around the pad wall. Suspected causes (to investigate):
   - **Through-hole pads (layer −1)** block *all* layers → layer change doesn't escape them.
   - **Via cost too high** (`AStarPathfinder.base_via_cost = 15`) → failing looks "cheaper" than a
     2-via detour.
   - **No congestion gradient to the emptier layer** — pads are hard blocks, so nothing softly nudges
     routes onto layer 2; A\* only vias when the *target* demands it.
   **This is the key Stage-1 unlock**: make the router treat layers as a real routing resource.
3. **Speed.** Failing / large A\* searches on ~300×300 boards are slow (a 4-net board hit ~118s;
   a 6-net 2-layer board ~160s). Needs a pass: tighter iteration caps, better heuristic, coarser
   routing grid, or early-abort on unreachable targets.

### 7.4 Non-router issue noticed
- **Test reproducibility:** `BoardGenerator.from_curriculum_stage` samples board size from the global
  `random` state *before* `generate()` re-seeds, so a given `cfg.seed` does not fully determine the
  board. Seed the global RNG before `from_curriculum_stage` for deterministic tests.

---

## 8. Immediate next steps

1. **Solve via-escape (highest priority).** Check pad layer types on a failing board; lower/tune
   `base_via_cost`; consider making layer usage congestion-aware so routes spread across layers.
   Without this, multi-layer doesn't buy completion.
2. **Speed pass on A\*** — early-abort unreachable targets, tighter caps, or coarser grid.
3. Then proceed to **Stage 2 (constraints)** and **Stage 3 (demand predictor)**.

## 9. Open design questions
- Demand predictor input encoding for the next-K pins (extra raster channels vs GNN conditioning vs
  both).
- How to reserve length-tuning slack in the guidance (bias vs explicit budget).
- Diff-pair routing: extend A\* to route the pair jointly vs route-then-mirror.
- Rip-up-reroute ordering under failure (which nets to rip up first).

## 10. Decisions log
- ✅ Completion via classical search; ML is guidance, not the router.
- ✅ Drop the autoregressive step policy + Dreamer actor-critic loop.
- ✅ Net order is an external input (not learned).
- ✅ Demand predictor is **supervised** (predict future routes), high-res **U-Net**, **all-at-once** for v1.
- ✅ Constraints (width, diff-pair, length) are given per-net; router satisfies them.
- ✅ Support up to 8 layers; layers are a routing resource (via-escape).
- ✅ Target "DDR-style" constrained routing, not signoff-grade DDR.
