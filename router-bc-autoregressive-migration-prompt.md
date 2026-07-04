# Implementation Prompt: BC-Bootstrapped Autoregressive Routing Policy + Granular Curriculum (Router)

## Context

Repo: `github.com/Klutzhehe/Router`

Today, routing decisions are made entirely by `AStarPathfinder` (`pcb_router/routing/pathfinder.py`). The network's only contribution is a predicted cost heatmap + via-probability map (`heatmap_decoder.py`) that A* is handed and obeys — the policy never picks a cell, a direction, or a via. This is visible directly in `DreamerJEPATrainer._phase1_collect_real` (`pcb_router/training/trainer.py`, current default trainer per `notebooks/03_training.py`): the RSSM action embedding is `(net_idx, heatmap_latent)`, decoded into `heatmaps_via`, then handed to `self.env.step_with_heatmaps()`, which calls `pathfinder.find_path()` and returns whatever A* found. A* also runs single-threaded, pure-Python, once per net, serially inside `env.step()` — the CPU bottleneck discussed previously.

This prompt replaces the *inner* per-net routing decision with a learned autoregressive policy that emits one grid move at a time, trained first via **behavior cloning (BC)** against A*-generated expert trajectories, then fine-tuned with RL inside the existing Dreamer imagination loop. It does **not** touch the RSSM/world-model architecture, JEPA representation learning, or the outer "which net to route next" decision (`select_net` stays as-is) — only what happens once a net has been selected.

**A* is not being deleted.** It becomes (a) the BC teacher that generates training labels, and (b) the permanent regression baseline / fallback routing mode. Keep `step_with_heatmaps` fully runnable throughout this work.

---

## Guiding principle

**Never train the new per-cell policy inside the Dreamer/imagination loop before it can reliably imitate A* in isolation.** Build and validate BC completely outside the RSSM — no `h`/`z`, no imagination, no replay buffer — first. Only integrate into `DreamerJEPATrainer` once the standalone BC policy passes its own closed-loop acceptance bar (defined in Phase 2 below). This is the same ordering discipline as the prior Dreamer refactor (`router-training-separation-prompt.md`): debug one new system at a time, not two simultaneously.

---

## Phase 0 — Shared primitives (new/shared files)

### `pcb_router/routing/obstacle_maps.py` (new — extract, don't duplicate)

`pathfinder.py` currently builds `temp_obstacle_maps` and `via_blocked` inline inside `find_path()` (lines ~55-109). The new per-cell policy needs the *exact same* obstacle/via-clearance semantics for action masking — if this logic is copy-pasted instead of shared, the BC teacher (A*) and the student (per-cell policy) will silently disagree about what's a legal move whenever one gets updated and the other doesn't. Extract into:

```python
def build_obstacle_maps(board_state, active_layers, exempt_cells=None) -> dict[int, np.ndarray]: ...
def build_via_blocked_maps(board_state, obstacle_maps, active_layers) -> dict[int, np.ndarray]: ...
```

Refactor `pathfinder.py` to call these instead of its inline version (behavior must be bit-identical — add a unit test that diffs old vs. new obstacle maps on a handful of boards before deleting the inline code).

### Action space

Fixed 10-way discrete action set, matching A*'s own move set so BC labels map 1:1:

```
0-3: N, S, E, W        (orthogonal, cost 1.0)
4-7: NE, NW, SE, SW    (diagonal, cost sqrt(2))
8:   via_up   (layer - 1)
9:   via_down (layer + 1)
```

No explicit "stop/terminate" action — mirror A*'s own convention: the env auto-terminates the current net when the cursor's (x, y[, layer]) equals the target. Adding a learned terminate action is unnecessary complexity this doesn't need yet.

### `PCBRoutingEnv.step_move(action_id)` (new method, additive)

Add alongside — **not instead of** — `step_with_heatmaps`. Advances a per-net "cursor" state one cell (or one via) at a time:

- Input: discrete `action_id` (0-9)
- Validity check: reuse `build_obstacle_maps`/`build_via_blocked_maps` from Phase 0 exactly as `find_path` does today for its neighbor expansion (same bounds check, same `obstacle_threshold`, same via clearance radius)
- On invalid move (off-board, obstacle, blocked via clearance): stay in place, apply the invalid-move penalty from the new step reward (below), do not advance
- On reaching target cell/layer: mark net routed, call trace generation for the accumulated cell path (reuse `TraceGenerator`/`MeanderInserter` unchanged — they only care about the final cell list, not how it was produced)
- Incremental occupancy: `BoardState.add_routed_trace()` currently expects a whole net's finished `TraceSegment` list. Add a lighter incremental variant, e.g. `BoardState.rasterize_partial_move(x1, y1, l1, x2, y2, l2)`, calling the existing `_rasterize_trace`/`_rasterize_circle` primitives per single segment, so the raster the policy observes reflects the trace-so-far, cell by cell (this matters — without it, the policy never sees its own partial progress and can't learn to avoid re-crossing itself)
- Step budget: cap at `max_moves_per_net = ceil(manhattan_distance(source, target) * budget_multiplier)` (start `budget_multiplier: 4.0`, analogous in spirit to A*'s `max_iterations` cap) — truncate and penalize rather than allow unbounded wandering

### New observation fields

Add to the obs dict alongside `board_raster`/`layer_mask`:
- `cursor_pos`: normalized (x, y, layer) of current head
- `target_pos`: normalized (x, y, layer) of current target pin
- `moves_remaining_frac`: `(max_moves_per_net - moves_taken) / max_moves_per_net`

These need a small embedding/concat point into the fused representation before the new policy head — see Phase 2.

---

## Phase 1 — Expert trajectory dataset generation

### `scripts/generate_bc_dataset.py` (new)

For each sampled board (draw from curriculum stage configs — see the granular curriculum below, and initially bias sampling toward the *early* stages, since those are the skills BC needs to nail first):

1. Run the real env exactly as `_phase1_collect_real` does today up through `pathfinder.find_path()` — get A*'s full cell path for each net.
2. Walk consecutive path cells `(x_i, y_i, l_i) → (x_{i+1}, y_{i+1}, l_{i+1})`. Convert each transition to a discrete action label via `cell_delta_to_action(dx, dy, dl)` (inverse of the move table above).
3. **Re-render the raster incrementally per cell**, not once per net — this is a real difference from current data collection. At step `i` along the path, the observation must reflect the board with cells `0..i-1` already laid down (via `rasterize_partial_move`), not the pre-route board and not the finished board. Otherwise BC learns to imitate moves conditioned on information (the finished trace) it won't have at inference time.
4. Store `(raster_i, graph, cursor_i, target, action_label)` tuples, sharded by curriculum stage name.

**Validation (do this before training anything on the dataset):** replay a sampled sequence's recorded action labels through a from-scratch `step_move` loop and confirm it reconstructs A*'s exact original path. This is a cheap deterministic unit test that catches transcription/off-by-one bugs (e.g. via direction sign, diagonal cost indexing) before they silently corrupt BC training.

---

## Phase 2 — Standalone BC pretraining (isolated, no RSSM)

### `pcb_router/models/route_step_policy.py` (new)

```python
class RouteStepPolicy(nn.Module):
    """Small head: (local spatial context + cursor + target) -> 10-way categorical.
    Deliberately NOT wired to h/z here — this class must be trainable and testable
    with zero dependency on JEPAWorldModel or DreamerActorCritic."""
    def __init__(self, embed_dim=384, cursor_embed_dim=32, hidden_dim=256):
        ...
        # Scaffold a value head now (predict steps-to-completion), even though
        # pure BC doesn't need it — avoids a second head-shape/checkpoint change
        # when Phase 3 wires this into actor-critic RL fine-tuning.

    def forward(self, fused_spatial, cursor_pos, target_pos, moves_remaining_frac):
        # concat cursor/target/budget embeddings with a local crop of fused_spatial
        # around cursor_pos (don't just mean-pool globally — the move decision is
        # local; global pooling throws away exactly the information A* uses)
        ...
        return move_logits, value_estimate
```

### `scripts/train_bc_policy.py` (new)

- Pure supervised cross-entropy against the Phase 1 dataset. **No environment stepping in this script at all.**
- Standard train/val split by board seed (not by transition — splitting mid-path leaks information across the split).
- Reuse the existing `vit`/`gnn`/`fusion` encoders. Default to **frozen** encoders for the first pass (JEPA already learned reasonable spatial representations; don't let early noisy BC gradients disturb them), then optionally unfreeze for a short fine-tune pass once cross-entropy has plateaued. Call out both as config options; frozen-first is the default.
- Action masking during training: even though expert actions are always valid by construction, mask the logits for illegal moves (`scores.masked_fill(~valid_mask, -1e4)`, same pattern already used in `PPOPolicy.select_net`) before computing the loss — this teaches the policy the mask exists and prevents it from assigning meaningful probability mass to actions it will never be allowed to take at inference.

### Acceptance criteria for Phase 2 (do not proceed to Phase 3 until these pass)

1. Held-out per-action cross-entropy accuracy above some threshold (start with 90%+ as a sanity floor on the easiest curriculum stages — tune once you see real numbers).
2. **Closed-loop rollout completion rate**, not just token accuracy — actually run the trained policy greedily (with masking) inside `step_move` on held-out boards, end to end, and compare its completion rate and DRC violation rate against A*'s own completion rate on the *same* boards. This is the real signal: per-step accuracy can look excellent while compounding single-step errors still cause closed-loop failures (classic distribution-shift problem in imitation learning — the policy visits states A* never demonstrated recovery from). If closed-loop completion lags A* by a wide margin on easy stages, the fix is DAgger-style iteration (run the current BC policy, have A* re-label the states it actually visits, retrain) rather than just collecting more on-expert-distribution data — flag this as a likely follow-up, don't skip straight to Phase 3 if closed-loop numbers are weak.
3. Confirm via logging that no gradients flow into `vit`/`gnn`/`fusion` during the frozen-encoder pass (same style of guardrail as the bit-identical-parameters check in the Dreamer separation prompt — cheap to write, catches a real class of bug).

---

## Phase 3 — Integrate into `DreamerJEPATrainer` (only after Phase 2 passes)

- **Initialize from Phase 2 weights, never from scratch.** Load the pretrained `RouteStepPolicy` into whatever module replaces the heatmap-producing path in `DreamerActorCritic`. Training this from random init inside the Dreamer loop defeats the entire purpose of doing BC first.
- **RSSM action embedding.** `JEPAWorldModel.get_action_embedding` currently takes `(net_idx, heatmap_latent)` once per net. Add `get_action_embedding_move(move_action_onehot, cursor_delta)` for the per-cell case. This changes what one `rssm_step` corresponds to: previously one step = one whole net; now one step = one grid cell.
- **This reopens the episode-length problem from the earlier autoregressive discussion**, now concretely: `seq_len: 50` in `configs/training.yaml` covers a small fraction of a single net's route once a step is one cell. Two options, in order of how much they change the codebase:
  - **(a) Flat, larger `seq_len`** (e.g. 200-400) — simplest, but multiplies world-model training memory/compute roughly proportionally. Start here.
  - **(b) Hierarchical two-level rollout** (outer net-level macro-steps containing inner cell-level micro-sequences) — only build this if (a) doesn't fit in Colab's VRAM/compute budget after profiling. Don't build it preemptively.
- **Reward.** Add `RewardCalculator.calculate_step()` for dense per-cell reward, mirroring A*'s own cost terms so the learned policy's behavior stays interpretable relative to the baseline it was cloned from:
  - distance-to-target delta (progress reward/penalty)
  - obstacle-collision / invalid-move penalty
  - direction-change penalty, same magnitude as `direction_change_penalty` in `pathfinder.py`
  - via cost, same magnitude as `base_via_cost`
  - Keep the existing `calculate()` (net-level: completion, wirelength detour ratio, DRC violations, length-matching error) as a **terminal bonus layered on top** when the net completes, not replaced — it already encodes real design intent (detour ratio, length matching) that per-step reward shouldn't try to duplicate.
- **Keep both routing modes switchable.** Add `routing_mode: astar_guided | autoregressive` to `configs/training.yaml`. This is your A/B tool and regression baseline for the rest of the project — don't remove the ability to fall back to `astar_guided` at any point.

---

## Detailed curriculum: granular, single-skill-at-a-time ramp

This replaces the 6 broad stages in `configs/curriculum.yaml` with a longer sequence where each stage introduces **exactly one** new difficulty axis. This matters more than usual for this migration specifically: a from-scratch-feeling per-cell policy (even BC-initialized) generalizes worse across simultaneous new axes than A*+heatmap did, so isolate skills before combining them.

One real code gap to fix first: `CurriculumManager.should_advance()` (`pcb_router/training/curriculum.py`) only reads `min_episodes` from the global `progression` block, not per-stage — unlike `completion_threshold`/`drc_violation_threshold`, which already support per-stage overrides. Fix:

```python
min_episodes = stage_cfg.get('min_episodes', self.progression_cfg.get('min_episodes', 500))
```

Without this fix, every stage below — including the trivial single-net-empty-board one — is stuck requiring the same 500 episodes as the hardest stage, which wastes a lot of the granularity this curriculum is trying to buy you.

```yaml
progression:
  completion_threshold: 0.95
  drc_violation_threshold: 0.02
  eval_window: 100
  min_episodes: 500   # fallback default; stages below override it explicitly

stages:
  # --- Block A: pure single-net pathing, no other skills yet ---
  - name: "s00_single_net_empty_board"
    description: "1 net, tiny board, zero obstacles — pure move-toward-target"
    min_episodes: 100
    drc_violation_threshold: 0.50
    board_generator:
      num_nets: 1
      num_layers: 1
      board_size_range: [100, 150]
      num_components_range: [2, 2]
      obstacle_density: 0.0
      diff_pairs: false
      length_matching: false
    reward_weights: {completion: 1.5, wirelength: 0.05, drc_violations: 0.1, congestion: 0.0, length_error: 0.0, all_complete_bonus: 0.3}

  - name: "s01_single_net_sparse_obstacles"
    description: "1 net, small board, light obstacles — first real detours"
    min_episodes: 150
    drc_violation_threshold: 0.40
    board_generator:
      num_nets: 1
      num_layers: 1
      board_size_range: [150, 250]
      num_components_range: [2, 3]
      obstacle_density: 0.05
    reward_weights: {completion: 1.5, wirelength: 0.08, drc_violations: 0.15, congestion: 0.0, length_error: 0.0, all_complete_bonus: 0.3}

  - name: "s02_single_net_moderate_obstacles"
    description: "1 net, obstacle density high enough to force genuine routing choices"
    min_episodes: 200
    drc_violation_threshold: 0.30
    board_generator:
      num_nets: 1
      num_layers: 1
      board_size_range: [200, 300]
      num_components_range: [3, 5]
      obstacle_density: 0.15
    reward_weights: {completion: 1.3, wirelength: 0.1, drc_violations: 0.2, congestion: 0.0, length_error: 0.0, all_complete_bonus: 0.3}

  # --- Block B: net count ramps by exactly +1, everything else held fixed ---
  - name: "s03_two_nets"
    description: "2 nets, sequential — must not re-cross its own first trace"
    min_episodes: 250
    drc_violation_threshold: 0.25
    board_generator:
      num_nets: 2
      num_layers: 1
      board_size_range: [250, 300]
      num_components_range: [3, 5]
      obstacle_density: 0.1
    reward_weights: {completion: 1.2, wirelength: 0.1, drc_violations: 0.25, congestion: 0.05, length_error: 0.0, all_complete_bonus: 0.35}

  - name: "s04_three_nets"
    min_episodes: 250
    drc_violation_threshold: 0.22
    board_generator:
      num_nets: 3
      num_layers: 1
      board_size_range: [250, 320]
      num_components_range: [4, 6]
      obstacle_density: 0.1
    reward_weights: {completion: 1.1, wirelength: 0.1, drc_violations: 0.3, congestion: 0.1, length_error: 0.0, all_complete_bonus: 0.4}

  - name: "s05_four_nets"
    min_episodes: 300
    drc_violation_threshold: 0.20
    board_generator:
      num_nets: 4
      num_layers: 1
      board_size_range: [280, 350]
      num_components_range: [5, 7]
      obstacle_density: 0.1
    reward_weights: {completion: 1.1, wirelength: 0.1, drc_violations: 0.3, congestion: 0.15, length_error: 0.0, all_complete_bonus: 0.4}

  - name: "s06_five_nets_congestion"
    description: "First stage where net *ordering* starts to matter — board shrinks relative to net count"
    min_episodes: 350
    drc_violation_threshold: 0.18
    board_generator:
      num_nets: 5
      num_layers: 1
      board_size_range: [260, 320]
      num_components_range: [5, 8]
      obstacle_density: 0.15
    reward_weights: {completion: 1.0, wirelength: 0.1, drc_violations: 0.35, congestion: 0.2, length_error: 0.0, all_complete_bonus: 0.4}

  # --- Block C: differential pairs, isolated from congestion ---
  - name: "s07_diff_pair_intro"
    description: "One diff pair only, clean board, generous length tolerance — isolate the new skill"
    min_episodes: 300
    drc_violation_threshold: 0.25
    board_generator:
      num_nets: 2
      num_layers: 1
      board_size_range: [250, 300]
      num_components_range: [2, 3]
      obstacle_density: 0.05
      diff_pairs: true
      num_diff_pairs_range: [1, 1]
      length_matching: true
      length_tolerance_mm: 2.0
    reward_weights: {completion: 1.2, wirelength: 0.08, drc_violations: 0.2, congestion: 0.0, length_error: 0.15, all_complete_bonus: 0.4}

  - name: "s08_diff_pair_plus_singles"
    description: "1 diff pair + 1-2 single nets, tighter length tolerance"
    min_episodes: 350
    drc_violation_threshold: 0.20
    board_generator:
      num_nets_range: [3, 4]
      num_layers: 1
      board_size_range: [280, 350]
      num_components_range: [4, 6]
      obstacle_density: 0.1
      diff_pairs: true
      num_diff_pairs_range: [1, 1]
      length_matching: true
      length_tolerance_mm: 1.0
    reward_weights: {completion: 1.1, wirelength: 0.1, drc_violations: 0.3, congestion: 0.1, length_error: 0.25, all_complete_bonus: 0.4}

  # --- Block D: vias / multi-layer, isolated from diff pairs and congestion ---
  - name: "s09_via_intro"
    description: "1-2 nets, 2 layers, obstacle placed so a via is *required* to reach target — isolate via placement"
    min_episodes: 300
    drc_violation_threshold: 0.25
    board_generator:
      num_nets_range: [1, 2]
      num_layers: 2
      board_size_range: [250, 320]
      num_components_range: [3, 5]
      obstacle_density: 0.1
      diff_pairs: false
      length_matching: false
    reward_weights: {completion: 1.2, wirelength: 0.1, drc_violations: 0.2, congestion: 0.0, length_error: 0.0, all_complete_bonus: 0.4}

  - name: "s10_via_plus_multi_net"
    description: "Combine via usage with the net-count skill from Block B"
    min_episodes: 350
    drc_violation_threshold: 0.20
    board_generator:
      num_nets_range: [3, 6]
      num_layers: 2
      board_size_range: [300, 380]
      num_components_range: [5, 8]
      obstacle_density: 0.12
    reward_weights: {completion: 1.1, wirelength: 0.1, drc_violations: 0.3, congestion: 0.15, length_error: 0.0, all_complete_bonus: 0.45}

  # --- Block E: combine everything learned so far ---
  - name: "s11_multi_net_multi_layer_diffpairs"
    description: "All prior skills together for the first time: net count, vias, diff pairs"
    min_episodes: 400
    drc_violation_threshold: 0.15
    board_generator:
      num_nets_range: [4, 10]
      num_layers_range: [2, 4]
      board_size_range: [350, 450]
      num_components_range: [6, 12]
      obstacle_density: 0.12
      diff_pairs: true
      num_diff_pairs_range: [1, 3]
      length_matching: true
      length_tolerance_mm: 1.0
    reward_weights: {completion: 1.0, wirelength: 0.1, drc_violations: 0.4, congestion: 0.2, length_error: 0.2, all_complete_bonus: 0.5}

  - name: "s12_congested_multi_net"
    description: "Roughly the old stage-4 difficulty, reached gradually this time"
    min_episodes: 450
    drc_violation_threshold: 0.10
    board_generator:
      num_nets_range: [8, 15]
      num_layers_range: [2, 4]
      board_size_range: [400, 480]
      num_components_range: [8, 15]
      obstacle_density: 0.15
      diff_pairs: true
      num_diff_pairs_range: [2, 4]
      length_matching: true
      length_tolerance_mm: 0.5
    reward_weights: {completion: 1.0, wirelength: 0.1, drc_violations: 0.5, congestion: 0.2, length_error: 0.25, all_complete_bonus: 0.5}

  # --- Block F: unchanged from current curriculum.yaml ---
  - name: "s13_full_design_rules"
    description: "All constraints: keep-outs, width rules, impedance, copper pours"
    min_episodes: 500
    drc_violation_threshold: 0.05
    board_generator:
      num_nets_range: [15, 50]
      num_layers_range: [2, 6]
      board_size_range: [450, 512]
      num_components_range: [8, 25]
      obstacle_density: 0.15
      diff_pairs: true
      num_diff_pairs_range: [2, 6]
      length_matching: true
      length_tolerance_mm: 0.3
      keep_out_zones: true
      num_keep_out_zones_range: [1, 5]
      width_rules: true
      net_classes: ["signal", "power", "high_speed"]
    reward_weights: {completion: 1.0, wirelength: 0.1, drc_violations: 0.5, congestion: 0.2, length_error: 0.3, all_complete_bonus: 0.5}

  - name: "s14_real_world"
    description: "Route actual KiCad designs with all constraints"
    min_episodes: 500
    drc_violation_threshold: 0.02
    board_generator:
      source: "kicad_import"
    reward_weights: {completion: 1.0, wirelength: 0.1, drc_violations: 0.5, congestion: 0.2, length_error: 0.3, all_complete_bonus: 0.5}
```

### How this interacts with BC dataset generation (Phase 1)

Generate the initial BC dataset primarily from stages `s00`-`s06` (Blocks A/B — pure movement and net-count skills). Diff pairs (`s07`/`s08`) and vias (`s09`/`s10`) don't need separate "expert demonstration" logic beyond what A* already does per-net — A* already routes diff-pair members as two independent paths (length-matching is a `MeanderInserter` post-process, not a routing-time decision) and already places vias when a layer change is cheapest. So the same `generate_bc_dataset.py` script naturally produces via- and diff-pair-relevant transitions once you run it against boards from those stages too — just make sure the dataset sharding (Phase 1) keeps stage labels so you can check BC accuracy *per stage*, not just in aggregate. A policy that's 95% accurate overall but only 60% accurate on via transitions (because they're rare in the data) will fail exactly at `s09`, and you want to catch that in Phase 2's acceptance check, not discover it three curriculum stages later.

---

## Notes for the agent doing this work

- Build in order: Phase 0 (shared obstacle code + `step_move` + observation fields) → Phase 1 (dataset generation, validated against exact A* reconstruction) → Phase 2 (standalone BC, validated by closed-loop rollout comparison against A*, not just cross-entropy) → Phase 3 (Dreamer integration) → curriculum rollout. Don't start Phase 3 while Phase 2's closed-loop acceptance bar is still failing — you want to know a bug is in the BC policy itself, not tangled up with RSSM/imagination interactions.
- Fix `CurriculumManager.should_advance()`'s missing per-stage `min_episodes` support before turning on the granular curriculum above — otherwise the early trivial stages take exactly as long to clear as the hard ones, defeating the point.
- Keep `routing_mode: astar_guided` fully working at every point in this migration. It's your ground truth for "is the autoregressive policy actually as good as what we had," not just a fallback.
- If Phase 2's closed-loop completion rate stays meaningfully below A*'s even after retraining, the fix is DAgger-style iterative relabeling (let the BC policy roll out, have A* relabel the states it actually visits, retrain on the union) — not more on-expert-distribution data. Flag this explicitly rather than quietly burning compute on Phase 3 with a shaky Phase 2 base.
