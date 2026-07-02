# Refactor Prompt: PPO+JEPA-auxiliary → GNN → JEPA World Model → DreamerV3 Actor

## Context

Repo: `github.com/Klutzhehe/Router`

Current architecture (from README):

```
Board Raster (13ch) ──► ViT-Small Encoder ──┐
                                             ├──► Cross-Attention Fusion ──► PPO Policy ──► Net Selection + Heatmap
Graph (Pads/Nets) ───► Hetero-GAT Encoder ──┘                                          └──► Heatmap Decoder ──► A* Router

Spatial JEPA: ViT Online ──► Predictor ──► Predict z_{t+1}  (EMA Target Encoder)
```

Today, JEPA only predicts the *next spatial patch embedding* from the ViT encoder — it's a side self-supervised objective that helps the ViT learn good representations. The PPO actor-critic trains entirely on **real environment steps** (`num_rollout_steps` per update), and the actual trace geometry is realized by a classical A* router downstream of a heatmap decoder.

## Goal

Restructure so JEPA becomes the **world model** and the actor-critic trains primarily on **imagined latent rollouts**, DreamerV3-style:

```
Board Raster (13ch) ──► ViT-Small Encoder ──┐
                                             ├──► Cross-Attention Fusion ──► Context Encoder ──► JEPA World Model
Graph (Pads/Nets) ───► Hetero-GAT Encoder ──┘                                                          │
                                                                                                        ▼
                                                                          Recurrent Latent State (deterministic h_t + stochastic z_t)
                                                                                                        │
                                              ┌─────────────────────────────────────────────────────────┼────────────────────────┐
                                              ▼                                                          ▼                        ▼
                                     Predictor (action-conditioned, EMA target)              Reward Head            Continue (done) Head
                                              │
                                              ▼
                                  Actor-Critic (trains on IMAGINED rollouts, λ-returns)
                                              │
                                              ▼
                                  Net Selection + Heatmap ──► Heatmap Decoder ──► A* Router (unchanged, symbolic backstop)
```

Key shift: PPO's `num_rollout_steps` of *real* environment interaction per update becomes a small amount of real data collection (to train the world model + seed the replay buffer), plus a much larger number of *imagined* steps unrolled entirely through the JEPA predictor for actor-critic training. This is the core efficiency gain of the Dreamer approach and the reason we're doing this refactor.

## File-by-file changes

### `pcb_router/models/gnn_encoder.py`
No architectural change needed. Just confirm its output embedding dimension is documented and stable — it now feeds into the JEPA context encoder rather than directly into the policy.

### `pcb_router/models/fusion.py`
Keep the cross-attention fusion of ViT + GAT embeddings, but rename its output conceptually to `context_embedding` — this becomes the **input to the JEPA world model**, not a direct input to the policy anymore. The policy will no longer see this fused embedding directly; it only sees the recurrent latent state produced by JEPA.

### `pcb_router/models/jepa.py` — the core of this refactor
Currently: `ViT Online → Predictor → predict z_{t+1}`, with an EMA target encoder, operating only on spatial patches.

Change to a full recurrent world model:
1. **Recurrent latent state**: add a GRU or similar recurrent cell producing deterministic state `h_t`. Combine with a stochastic latent `z_t` (sampled from a small categorical or Gaussian head conditioned on `h_t` and the context embedding) — this mirrors DreamerV3's RSSM split between deterministic and stochastic components, and gives the world model the stochasticity it needs to represent routing uncertainty (which net to prioritize, which direction is ambiguous).
2. **Action-conditioned predictor**: the predictor must now take `(h_t, z_t, action_t)` and output `h_{t+1}` (prior), *not* just predict the next patch from the current patch. The action is the net-selection + heatmap-direction decision from the policy.
3. **Keep the EMA target encoder + stop-gradient setup** — this is your existing anti-collapse mechanism and it should carry over unchanged as the *posterior* target for `z_{t+1}` (encode the real next context embedding, EMA-updated, stop-gradient into the predictor's KL/regression loss).
4. **Add a reward head**: small MLP off `(h_t, z_t)` predicting the shaped reward (wirelength penalty, DRC risk, net-completion bonus) — needed so imagined rollouts can score themselves without touching the real environment.
5. **Add a continuation head**: small MLP off `(h_t, z_t)` predicting episode-continuation probability (1 - done), so imagined rollouts know when to stop.
6. **Loss terms to combine**: (a) latent prediction loss (posterior vs. prior, KL-balanced like DreamerV3, or JEPA-style regression + variance/covariance regularization if you keep it non-probabilistic), (b) reward head regression loss, (c) continuation head cross-entropy loss. Log each separately — if only the reward/continue losses go down while prediction loss stalls, that's your collapse signal (see Validation section).

### `pcb_router/models/policy.py`
Replace the PPO actor-critic with a Dreamer-style actor-critic:
- **Input**: recurrent latent state `(h_t, z_t)` from the world model, not the raw fused embedding.
- **Training data**: imagined trajectories only — unroll the *actor* through the world model's predictor for a fixed imagination horizon `H` (start with H=15, matching DreamerV3 defaults, tune down if routing episodes are short), scoring each imagined step with the reward head and continue head.
- **Critic target**: λ-returns computed over the imagined trajectory (bootstrapped with the critic's own value at the horizon), not GAE over real rollouts.
- **Actor loss**: reinforce/backprop-through-world-model on the λ-return, plus an entropy bonus over the net-selection action distribution to prevent premature convergence to always routing the same net order.
- **Output stays the same shape**: net selection logits + heatmap — no change needed to `heatmap_decoder.py` other than confirming its input now comes from the actor's output conditioned on imagined-or-real latent state (same interface either way, so downstream A* router is untouched).

### `pcb_router/training/trainer.py`
Rename/restructure `PPOJEPATrainer` → `DreamerJEPATrainer` (or keep the class name if you'd rather not break the Colab notebook's import — your call, just document it). New training loop per iteration:
1. **Collect real data**: run the current actor policy (grounded through the real world model, not imagined) in `pcb_env.py` for a small number of real steps (e.g. 64–100, much less than before), store `(context_embedding, action, reward, done)` transitions in a replay buffer.
2. **Train world model**: sample a batch of real sequences from the replay buffer, train the encoder+GNN+fusion+JEPA predictor+reward head+continue head jointly on real transitions (this is where the actual grounding happens — without this step the actor is training against a world model that's never been checked against reality).
3. **Imagine rollouts**: from real latent states sampled out of the replay buffer, unroll the actor through the *frozen-for-this-step* world model predictor for horizon `H`, collecting imagined `(latent, action, reward_pred, continue_pred)` sequences.
4. **Train actor-critic**: on the imagined rollouts only, using λ-returns as above.
5. Repeat. Track a `train_ratio` (imagined gradient steps per real environment step) as a config value — this is the main lever for real-vs-imagined training balance.

### `pcb_router/training/rewards.py`
No change to the reward *shaping* logic itself, but make sure the reward function is called and logged per real transition so it can supervise the new reward head (step 2 above needs ground-truth reward, not just for the actor).

### `pcb_router/env/pcb_env.py` and `pcb_router/env/drc_checker.py`
No architectural change. Just confirm the env exposes a clean `done`/`continue` flag per step (episode ends on full route completion, DRC failure past some threshold, or step limit) — the continuation head needs this as a supervised target.

### `configs/model.yaml`
Add:
- `world_model.deterministic_size`, `world_model.stochastic_size` (the `h_t` / `z_t` dims)
- `world_model.imagination_horizon` (default 15)
- `world_model.reward_head_hidden`, `world_model.continue_head_hidden`
- `world_model.kl_balance` if you go probabilistic, or `world_model.jepa_reg_weight` (variance/covariance regularization coefficient) if you keep it JEPA-regression-style rather than KL-based

### `configs/training.yaml`
Replace PPO-specific keys (`num_rollout_steps`, `num_epochs` as PPO update epochs) with:
- `real_steps_per_iteration` (replaces the old rollout collection size, much smaller now)
- `train_ratio` (imagined actor-critic gradient steps per real step collected)
- `replay_buffer_size`
- `world_model_lr`, `actor_lr`, `critic_lr` (Dreamer typically uses separate LRs for these)

### `configs/curriculum.yaml`
No structural change — curriculum staging by board complexity still applies, just note that early curriculum stages (small boards, few nets) are also where you should validate the world model's prediction accuracy before trusting imagined training (see below).

## Validation / acceptance criteria (do this before declaring the refactor "done")

1. **World model grounding check**: on a held-out set of real trajectories, compare the world model's predicted reward/continue/next-latent against ground truth. If prediction error doesn't clearly decrease over training, the actor-critic is about to train against a broken simulator — stop and fix the world model before touching the actor.
2. **Latent collapse check**: log the variance of `z_t` across a batch. If it collapses toward a constant while losses still look fine, the anti-collapse regularization (EMA target + stop-gradient, and/or variance/covariance terms) isn't sufficient — this is the most likely failure mode per our earlier discussion.
3. **Imagined vs. real reward correlation**: periodically run the actor in the *real* environment and compare episode return against what the imagined rollouts predicted for equivalent states. Large, growing divergence means the actor is exploiting world-model errors (classic Dreamer failure mode) — shrink `imagination_horizon` or increase `real_steps_per_iteration` if this happens.
4. **Regression test against current PPO baseline**: keep the existing PPO+JEPA-auxiliary trainer runnable (or checkpointed results saved) so you can compare sample efficiency and final routing quality (DRC-clean rate, via count, wirelength) directly against the new Dreamer-style trainer on the same curriculum stage.

## Notes for the agent doing this work

- Do this incrementally, one file group at a time, in the order listed above (jepa.py first, since everything else depends on its new interface) — don't attempt a single giant rewrite.
- Preserve the existing A* router and heatmap decoder interfaces exactly; this refactor is scoped to the representation/training loop, not the final path-realization step.
- Flag explicitly if `torch-geometric`/`torch-scatter` versions in `requirements.txt` need bumping to support any new layer types used in the recurrent latent state.
