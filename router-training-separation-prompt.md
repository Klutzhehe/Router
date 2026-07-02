# Implementation Prompt: Decouple World-Model and Actor-Critic Training (Router)

## Context

Repo: `github.com/Klutzhehe/Router`

This follows the prior refactor (`router-dreamerv3-refactor-prompt.md`) that restructures JEPA into a recurrent world model (RSSM-style: deterministic `h_t` + stochastic `z_t`) and replaces PPO with a Dreamer-style actor-critic trained on imagined rollouts.

That prompt specifies **what** the new components are (`jepa.py` as world model, `policy.py` as Dreamer actor-critic, three-phase `trainer.py` loop). This prompt specifies **how to build and validate the training separation itself**, in detail, plus concrete optimizations. Treat this as the execution plan for `trainer.py`, `replay_buffer.py`, and the associated config/logging work.

Current state: `PPOJEPATrainer` trains everything end-to-end in one step — real rollout collection, PPO update, and JEPA auxiliary loss all share one forward pass with no buffer between them. That coupling is the thing being removed.

## Guiding principle

**Never build the three-phase loop and the new world model at the same time.** Build and validate the world model in total isolation from any actor first. Only wire in the actor-critic once the world model passes its own validation suite standalone. This ordering is not optional — it is the difference between debugging one system at a time versus two simultaneously.

---

## Phase 0 — Replay buffer (new file)

### `pcb_router/training/replay_buffer.py`

This is the seam that separates "touches the environment" from "touches gradients." Nothing downstream of this file should ever call `pcb_env.py` directly except the real-data-collection routine.

**Data model**: store episodes, not flat transitions, since sequence sampling needs temporal continuity for the RSSM.

```python
class Episode:
    context_embeddings: List[Tensor]   # from fusion.py, pre-JEPA
    actions: List[Tensor]              # net-selection + heatmap-direction
    rewards: List[float]
    dones: List[bool]
    length: int

class ReplayBuffer:
    def __init__(self, capacity_episodes: int, min_episode_len: int = 1):
        ...

    def add_episode(self, episode: Episode): ...

    def sample_sequences(self, batch_size: int, seq_len: int) -> Batch:
        # Sample batch_size episodes weighted by length (longer episodes = more
        # valid start indices), then slice a random seq_len window from each.
        # Pad + mask if an episode is shorter than seq_len.
        ...

    def sample_latents(self, batch_size: int) -> Tensor:
        # For imagination seeding: sample single (h_t, z_t) states, not
        # sequences. These come from a *cached* pass of the world model
        # over stored episodes (see Phase 2 caching note below), not
        # recomputed from raw context embeddings every call.
        ...

    def __len__(self): ...
```

**Design details to get right:**
- **Eviction policy**: ring buffer (FIFO) over episodes, not transitions — evicting mid-episode breaks sequence sampling. Once `capacity_episodes` is hit, drop oldest episode.
- **Weighted sampling by length**: uniform episode sampling under-samples long episodes' interior transitions. Weight sampling probability by `max(episode.length - seq_len + 1, 1)`.
- **Store `context_embedding`, not raw board raster / graph**: recomputing ViT+GAT+fusion on every buffer read is wasteful. Cache the fused embedding at collection time. If you later change the encoder architecture, the buffer must be invalidated (log a buffer schema version to catch this).
- **On-disk overflow (optional, only if boards are large)**: if `context_embedding` tensors make `replay_buffer_size` too large for RAM, memory-map episodes to disk with `torch.save`/`np.memmap` keyed by episode id. Don't build this until you've actually measured you need it.

---

## Phase 1 — Standalone world model validation harness

Build this **before** touching `trainer.py`. It is throwaway/dev tooling, not part of the shipped training loop, but it is mandatory infrastructure — do not skip it to save time.

### `scripts/debug_world_model.py`

```python
def collect_random_episodes(env, buffer, n_episodes, policy=None):
    """policy=None → uniform random actions. Deliberately NOT the trained
    actor: we want broad state coverage, not the narrow distribution an
    undertrained actor would produce."""
    for _ in range(n_episodes):
        obs = env.reset()
        episode = Episode()
        done = False
        while not done:
            action = policy.act(obs) if policy else env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            episode.append(obs.context_embedding, action, reward, done)
            obs = next_obs
        buffer.add_episode(episode)

def train_world_model_standalone(cfg):
    env = PCBEnv(cfg.env)
    buffer = ReplayBuffer(cfg.training.replay_buffer_size)
    collect_random_episodes(env, buffer, n_episodes=cfg.debug.warmup_episodes)

    world_model = JEPAWorldModel(cfg.model.world_model)
    wm_opt = torch.optim.AdamW(
        world_model.parameters(),
        lr=cfg.training.world_model_lr,
        eps=1e-8,               # DreamerV3 uses larger eps than default 1e-8→1e-5-ish; see optimization notes
        weight_decay=cfg.training.wm_weight_decay,
    )

    for step in range(cfg.debug.train_steps):
        batch = buffer.sample_sequences(cfg.training.batch_size, cfg.training.seq_len)
        losses = world_model.compute_loss(batch)   # dict: pred, reward, continue, kl (if probabilistic)
        total = (
            cfg.loss_weights.pred * losses['pred']
            + cfg.loss_weights.reward * losses['reward']
            + cfg.loss_weights.continue_ * losses['continue']
        )
        wm_opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg.training.wm_grad_clip)
        wm_opt.step()

        log_scalar('wm/pred_loss', losses['pred'], step)
        log_scalar('wm/reward_loss', losses['reward'], step)
        log_scalar('wm/continue_loss', losses['continue'], step)
        log_scalar('wm/grad_norm', grad_norm, step)

        if step % cfg.debug.eval_every == 0:
            run_validation_suite(world_model, buffer, step)
```

### `run_validation_suite(world_model, buffer, step)` — implement all three checks from the original prompt, as actual code, not just log-reading:

1. **Grounding check**: hold out ~10% of episodes at buffer-creation time (`buffer_eval = ReplayBuffer(...)`, filled once, never trained on). Every `eval_every` steps, run the world model open-loop on held-out sequences (feed real actions, predict forward without ever re-injecting real observations after `t=0`) and compute:
   - Predicted-vs-real reward MSE at each imagined horizon step (1, 5, 10, 15 steps out)
   - Continue-head accuracy at each horizon step
   - Latent prediction error (posterior `z_t` from real next embedding vs. prior `z_t` from predictor)

   Plot error **as a function of horizon distance**, not just aggregate — this tells you how far you can trust imagination, which directly informs `imagination_horizon`.

2. **Latent collapse check**: every eval step, compute `z_t.var(dim=0).mean()` across a batch. Log it. Also log the **per-dimension** variance, not just the mean — a world model can look fine on average variance while a subset of latent dimensions have fully collapsed. Flag if any dimension's variance drops below a threshold (e.g. `1e-4`) for more than N consecutive eval steps.

3. **Reward/continue vs. prediction loss divergence check**: log all three losses on the same plot. If `reward_loss` and `continue_loss` keep decreasing while `pred_loss` plateaus or increases, that's the collapse signal called out in the original prompt — the model is learning to predict reward/continue from a degenerate or shortcut-y latent rather than one that actually models board dynamics. Add an automated check: if `pred_loss` hasn't improved by >X% over the last N eval windows while `reward_loss` has, print a warning banner in the logs.

**Do not proceed to Phase 2 until this harness shows stable, decreasing `pred_loss`, non-collapsing `z_t` variance, and no reward/continue-loss-only-improving pattern**, on at least one curriculum stage (start with the smallest-board stage from `curriculum.yaml`).

---

## Phase 2 — Three-phase `trainer.py` loop

Only after Phase 1 passes. Restructure `DreamerJEPATrainer`:

```python
class DreamerJEPATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = PCBEnv(cfg.env)
        self.world_model = JEPAWorldModel(cfg.model.world_model)
        self.target_encoder = copy.deepcopy(self.world_model.encoder_stack)  # EMA target
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.actor_critic = DreamerActorCritic(cfg.model)
        self.buffer = ReplayBuffer(cfg.training.replay_buffer_size)

        self.wm_opt = torch.optim.AdamW(self.world_model.parameters(), lr=cfg.training.world_model_lr,
                                          weight_decay=cfg.training.wm_weight_decay)
        self.actor_opt = torch.optim.AdamW(self.actor_critic.actor.parameters(), lr=cfg.training.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.actor_critic.critic.parameters(), lr=cfg.training.critic_lr)

        self.scaler_wm = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)
        self.scaler_ac = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)

        self.global_step = 0
        self.imagination_horizon = cfg.training.imagination_horizon_start  # see horizon scheduling below

    def train_iteration(self):
        self._phase1_collect_real()
        wm_metrics = self._phase2_train_world_model()
        ac_metrics = self._phase3_train_actor_critic()
        self._maybe_run_validation_suite()
        self._maybe_checkpoint()
        self.global_step += 1
        return {**wm_metrics, **ac_metrics}
```

### Phase 1 of the loop: real collection

```python
def _phase1_collect_real(self):
    n = self.cfg.training.real_steps_per_iteration
    steps_collected = 0
    while steps_collected < n:
        obs = self.env.reset()
        episode = Episode()
        h, z = self.world_model.initial_state(batch_size=1)
        done = False
        while not done and steps_collected < n:
            with torch.no_grad():
                context_emb = self.world_model.encode(obs)          # ViT+GAT+fusion, real obs
                action, _ = self.actor_critic.act(h, z, explore=True)  # exploration noise/epsilon, see below
                h, z = self.world_model.rssm_step(h, z, context_emb, action)
            next_obs, reward, done, info = self.env.step(action)
            episode.append(context_emb, action, reward, done)
            obs = next_obs
            steps_collected += 1
        self.buffer.add_episode(episode)
```

Note the actor here runs **grounded** through the real world model step-by-step (not imagined) — this is real interaction, so `h_t, z_t` come from real posterior updates, not the predictor's prior.

### Phase 2 of the loop: world model update

Identical to the standalone harness in Phase 1 above — reuse the same `compute_loss` call and loss-weighting/logging code, don't fork it into a second implementation. Extract it into a shared method so the standalone debug script and the trainer call the exact same code path (`world_model.train_step(batch, optimizer)` living in `jepa.py` itself, called from both places).

### Phase 3 of the loop: imagination + actor-critic

```python
def _phase3_train_actor_critic(self):
    self._freeze(self.world_model)
    metrics = defaultdict(list)
    for _ in range(self.cfg.training.train_ratio):
        init_latents = self.buffer.sample_latents(self.cfg.training.imagine_batch_size)
        h0, z0 = init_latents['h'].detach(), init_latents['z'].detach()

        traj = self._imagine(h0, z0, horizon=self.imagination_horizon)
        lambda_returns = compute_lambda_returns(
            rewards=traj.rewards, values=traj.values, continues=traj.continues,
            gamma=self.cfg.training.gamma, lam=self.cfg.training.lambda_,
        )

        critic_loss = self.actor_critic.critic_loss(traj.h, traj.z, lambda_returns.detach())
        self.critic_opt.zero_grad(set_to_none=True)
        self.scaler_ac.scale(critic_loss).backward()
        self.scaler_ac.unscale_(self.critic_opt)
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.cfg.training.critic_grad_clip)
        self.scaler_ac.step(self.critic_opt)

        actor_loss, entropy = self.actor_critic.actor_loss(
            traj.h, traj.z, traj.actions, lambda_returns.detach(),
            entropy_coef=self._current_entropy_coef(),
        )
        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler_ac.scale(actor_loss).backward()
        self.scaler_ac.unscale_(self.actor_opt)
        torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.cfg.training.actor_grad_clip)
        self.scaler_ac.step(self.actor_opt)
        self.scaler_ac.update()

        metrics['actor_loss'].append(actor_loss.item())
        metrics['critic_loss'].append(critic_loss.item())
        metrics['entropy'].append(entropy.item())
        metrics['imagined_return_mean'].append(lambda_returns.mean().item())

    self._unfreeze(self.world_model)
    return {k: sum(v) / len(v) for k, v in metrics.items()}

def _imagine(self, h0, z0, horizon):
    h, z = h0, z0
    traj = Trajectory()
    for t in range(horizon):
        action, log_prob = self.actor_critic.actor.sample(h, z)   # gradients flow, world model frozen not detached
        reward_pred = self.world_model.reward_head(h, z)
        continue_pred = self.world_model.continue_head(h, z)
        value = self.actor_critic.critic(h, z)
        traj.append(h, z, action, reward_pred, continue_pred, value, log_prob)
        h, z = self.world_model.predict_step(h, z, action)  # prior only, no real obs
    traj.bootstrap_value = self.actor_critic.critic(h, z)
    return traj

def _freeze(self, module):
    for p in module.parameters():
        p.requires_grad_(False)

def _unfreeze(self, module):
    for p in module.parameters():
        p.requires_grad_(True)
```

**Critical correctness details:**
- `init_latents` must be `.detach()`-ed before imagination starts — otherwise actor-critic gradients leak backward into whatever computation graph produced them during Phase 2, silently corrupting the world model's next update.
- `_freeze`/`_unfreeze` toggles `requires_grad`, it does **not** wrap imagination in `torch.no_grad()`. The actor loss needs gradients to flow through `predict_step` (backprop-through-model), it just shouldn't accumulate into the world model's own optimizer.
- Two separate `GradScaler`s if using AMP (see optimizations below) — sharing one scaler between world-model and actor-critic phases causes incorrect loss-scale adaptation since the two phases have very different gradient magnitude profiles.

---

## Config additions (extends `configs/training.yaml` and `configs/model.yaml`)

```yaml
training:
  real_steps_per_iteration: 64
  train_ratio: 100                    # imagined AC gradient steps per real step collected
  replay_buffer_size: 5000            # episodes, not transitions
  batch_size: 64                      # sequence batch for world model updates
  seq_len: 50                         # training sequence length for world model (>= imagination_horizon)
  imagine_batch_size: 512             # parallel imagined rollouts per AC update — cheap since no env calls
  imagination_horizon_start: 5        # see horizon scheduling
  imagination_horizon_end: 15
  imagination_horizon_ramp_iters: 20000

  world_model_lr: 3e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  wm_weight_decay: 1e-6
  wm_grad_clip: 100.0
  actor_grad_clip: 100.0
  critic_grad_clip: 100.0

  gamma: 0.997
  lambda_: 0.95

  entropy_coef_start: 3e-3
  entropy_coef_end: 3e-4
  entropy_coef_decay_iters: 50000

  use_amp: true
  compile_world_model: true           # torch.compile, see below

  critic_target_ema: 0.98             # slow-moving target critic, see optimizations
  wm_ema_decay: 0.995                 # EMA target encoder decay

loss_weights:
  pred: 1.0
  reward: 1.0
  continue_: 1.0
  kl_balance: 0.8                     # if probabilistic RSSM: weight on prior vs posterior KL term
  free_bits: 1.0                      # KL free-bits floor, prevents over-regularizing posterior to prior early

eval:
  eval_every_iters: 200
  held_out_episodes: 200
  log_horizon_breakdown: true         # per-horizon-step error, not just aggregate
```

---

## Optimizations to add while implementing this

These aren't separate work — bake them in during Phase 1/2 implementation, since retrofitting them later means re-touching the same code twice.

### World model training
- **Symlog transform on reward targets and value targets** (DreamerV3 trick): `symlog(x) = sign(x) * log(1 + |x|)`. PCB routing rewards likely have wide dynamic range (small step penalties vs. large net-completion bonuses); symlog compresses this so the reward head doesn't need to fit both scales directly. Apply `symlog` before the regression loss, `symexp` (inverse) when consuming predictions in `_imagine`.
- **Free bits on the KL term** (if going probabilistic RSSM): clamp the per-dimension KL loss to not push below a small floor (e.g. 1 nat), preventing the posterior from being over-regularized toward the prior before the prior has learned anything useful — a common early-training collapse mode.
- **KL balancing**: weight gradients into the prior vs. posterior asymmetrically (`kl_balance` config above, DreamerV3 default ~0.8 on the prior side) so the prior is pulled toward the posterior faster than the posterior is pulled toward the prior — keeps the posterior grounded in real data.
- **Discrete (categorical) stochastic latents over Gaussian**, if choosing probabilistic: DreamerV3 found categorical latents with straight-through gradients more stable for RSSM than Gaussian, especially against collapse. Worth defaulting to this rather than Gaussian given routing's inherently discrete decisions (which net, which direction).
- **Layer norm in the GRU/recurrent cell and MLPs**, not just at the output — stabilizes multi-hundred-step training runs.
- **Gradient clipping by global norm** (shown above, `wm_grad_clip`) — RSSM training is prone to occasional gradient spikes early on.

### Actor-critic training
- **Target critic with EMA** (`critic_target_ema`): bootstrap λ-returns using a slow-moving copy of the critic rather than the live critic being updated in the same step — reduces the moving-target problem, standard in Dreamer and most actor-critic setups.
- **Entropy coefficient decay schedule** rather than a fixed constant — start higher to encourage exploring net-ordering strategies early, decay once the policy has reasonable coverage. Linear or exponential decay both fine; linear shown in config above.
- **Imagination horizon warmup/ramp**: start `imagination_horizon` short (5) and ramp to the target (15) over the first N iterations. Early in training the world model's multi-step predictions are unreliable; imagining 15 steps into a bad model wastes compute and teaches the actor to exploit model error. Tie the ramp rate to the Phase-1-style grounding check if possible — only extend horizon once per-horizon-step prediction error at the current horizon is below a threshold.
- **Reward/value normalization via running percentile stats** (DreamerV3-style): scale returns by an exponentially-decayed running estimate of the 5th–95th percentile range, rather than raw magnitude, so actor-critic loss scale stays stable across curriculum stages with very different reward magnitudes (small vs. large boards).

### Systems / throughput
- **`torch.compile` on the world model's `predict_step`**: this function gets called `imagination_horizon × train_ratio` times per iteration (e.g. 15 × 100 = 1500 calls) with a fixed shape — an ideal `torch.compile` target. Don't compile the full world model if the encoder path has data-dependent control flow from the GNN; compile just the recurrent step function.
- **Mixed precision (AMP) with separate `GradScaler`s** for world-model and actor-critic phases, as noted above.
- **Vectorized imagination across `imagine_batch_size`**: this should already be implicit in the tensor shapes above (`h, z` carry a batch dimension), but confirm `_imagine` never loops over the batch dimension in Python — the entire point of imagined rollouts being cheap is that they parallelize on GPU with no environment/CPU round-trip.
- **Cache encoded latents for `sample_latents`**: don't re-run the encoder over stored `context_embedding`s every time you need seed states for imagination. Maintain a small rolling cache of `(h_t, z_t)` computed during Phase 2's world-model training batches (you're already running the encoder over these sequences — extract and cache the resulting states rather than recomputing in a separate pass).
- **Async real-data collection (optional, only if env stepping is a bottleneck)**: run `_phase1_collect_real` in a background worker/process writing into a thread-safe buffer while Phase 2/3 run on GPU, so environment stepping (likely CPU-bound, A* router involved) doesn't stall GPU-bound world-model/actor training. Only add this once profiling shows Phase 1 is the bottleneck — don't add complexity preemptively.

### Logging / observability
- Log all three phases' losses and timings **separately per iteration**, not aggregated — you need to see world-model loss and actor-critic loss on independent timelines to diagnose which phase is misbehaving.
- Log wall-clock time per phase (`phase1_collect_s`, `phase2_wm_s`, `phase3_ac_s`) — this tells you where to spend optimization effort and whether `train_ratio` is set sensibly relative to real collection cost.
- Log `imagination_horizon` and `entropy_coef` current values each iteration (they're scheduled, not constant — easy to lose track of where you are in the ramp).
- Dashboard/plot per-horizon-step prediction error (from the validation suite) over training time, not just a single scalar — this is your primary signal for whether it's safe to extend `imagination_horizon`.

---

## Validation / acceptance criteria for this specific piece of work

1. Phase 1 standalone harness runs to convergence on the smallest curriculum stage with decreasing `pred_loss`, stable non-collapsing `z_t` variance, and no reward/continue-only-improving divergence, **before** Phase 2/3 are implemented.
2. Once the full three-phase trainer is running: confirm via logging that `wm_opt.step()` is never called during Phase 3 and `actor_opt.step()`/`critic_opt.step()` are never called during Phase 2 (add an assertion or step-counter check in each optimizer's call site as a guardrail, not just a visual log check).
3. Confirm gradient isolation directly: after a Phase 3 update, check that the world model's parameters are bit-identical to before Phase 3 ran (a cheap `torch.allclose` sanity check you can run in a unit test, not just in production training).
4. Re-run the three validation checks from the original refactor prompt (grounding, collapse, imagined-vs-real reward correlation) on the full trainer, not just the standalone harness — the actor's exploration behavior changes the data distribution the world model sees in Phase 1, which can surface issues the random-policy harness didn't.
5. Confirm `imagination_horizon` ramp and `entropy_coef` decay are actually changing over training (log inspection) and not accidentally left at their start values due to a scheduling bug.

## Notes for the agent doing this work

- Build in the order listed: `replay_buffer.py` → `scripts/debug_world_model.py` validation harness → only then the three-phase `trainer.py` loop → then layer in the optimizations. Don't add symlog/KL-balance/torch.compile/etc. while the basic separation is still unvalidated — you want to know a bug is in the separation logic, not tangled up with an optimization you added at the same time.
- Share the `world_model.train_step(batch, optimizer)` implementation between the standalone debug script and `trainer.py`'s Phase 2 — don't fork it into two copies that can drift.
- The `_freeze`/`_unfreeze` + `.detach()` pattern is the single most bug-prone part of this work. Add the bit-identical-parameters unit test (validation criterion 3 above) early, not as an afterthought — it's cheap to write and catches the most common mistake in this kind of setup.
- Keep the old `PPOJEPATrainer` runnable throughout this work as the regression baseline, per the original refactor prompt's acceptance criteria.
