# Go1 velocity tracking with SuperMimic

This package adds two mjlab tasks that keep the stock Go1 velocity task but train
the policy with the SuperMimic model-based loop instead of PPO:

| Task id | Terrain | Actor observations |
|---|---|---|
| `Mjlab-Velocity-SuperMimic-Unitree-Go1` | rough (stock Go1) | proprioception + command, **no base linear velocity, no height scan** |
| `Mjlab-Velocity-SuperMimic-Flat-Unitree-Go1` | flat | same |

The critic group is unchanged, so the world model and the critic still receive
the privileged base linear velocity and the terrain height scan. This is the
asymmetric setup used by the paper: the deployed actor is blind, the learned
dynamics model is not.

## Algorithm

The implementation is a port of the reference reproduction in
`~/桌面/知识库/supermimic_vibed` to mjlab:

1. **Collect** transitions with the current deterministic policy plus
   `N(0, 0.1²)` action noise, into a per-environment replay buffer.
2. **Fit the world model** on `k=8`-step autoregressive windows. The state is
   `[base pose (relative to env origin), base velocity, 12 joints, 4 foot
   contacts, 4 air times, 4 foot heights]`; the context `xi` is the 187-D
   terrain height scan (rough task only); the push velocity is an explicit
   one-step exogenous input so the deterministic model can represent mjlab's
   interval pushes.
3. **Improve the policy** by BPTT through `H=32` imagined steps, with a critic
   that bootstraps the return beyond the horizon and a soft survival weight.
   The reward, observation and termination heads are learned from the
   simulator signals (the paper uses analytic heads because its tracking reward
   is a closed-form function of the state; the velocity reward depends on
   contact sensors and hidden timers, so learned heads are the pragmatic
   choice here).

## Run

```bash
# rough terrain, one GPU
uv run train Mjlab-Velocity-SuperMimic-Unitree-Go1 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5000 \
  --agent.logger wandb

# flat terrain
uv run train Mjlab-Velocity-SuperMimic-Flat-Unitree-Go1 \
  --env.scene.num-envs 1024
```

The first `agent.wm_only_iters` iterations (default 100) only fit the world
model; policy updates start afterwards. The default network widths and batch
sizes are sized for a single consumer GPU. On an 8 GB laptop GPU start with
`--env.scene.num-envs 256` and, if needed, reduce
`--agent.wm-batch 128 --agent.pi-batch 128`.

Play a checkpoint with the standard entry point:

```bash
uv run play Mjlab-Velocity-SuperMimic-Unitree-Go1 \
  --checkpoint-file logs/rsl_rl/go1_velocity_supermimic/<run>/model_5000.pt
```

## NoCrouch variant

`Mjlab-Velocity-SuperMimic-Unitree-Go1-NoCrouch` targets a failure mode seen on
the stock rough task: the policy sits or crouches, matches the small yaw
command, ignores the forward command, and never terminates because the rough
termination set only checks thigh contact and terrain bounds.  The variant adds:

* `low_base` termination when the root height drops below 0.15 m (training
  only; disabled in play because the random play terrain can dip);
* `trunk_contact` termination on `trunk_ground_touch`, active in training and
  play;
* forward-only commands (`lin_vel_x` positive, zero yaw, no heading controller)
  that widen to lateral/yaw commands at 4k and 8k vectorized environment steps;
* `action_clip=2.0` and `action_noise=0.3` in its runner config.
* an analytic `height_term_min=0.15` gate combined with the learned
  termination head during imagination, plus a 10x positive-class weight for the
  termination BCE.  The learned head alone underestimates the rare low-base
  events (the stock run showed `survival≈0.95` while almost every real episode
  ended in a hard termination), which lets the policy fall and believe it
  survives.

```bash
uv run train Mjlab-Velocity-SuperMimic-Unitree-Go1-NoCrouch \
  --env.scene.num-envs 2048

uv run play Mjlab-Velocity-SuperMimic-Unitree-Go1-NoCrouch \
  --checkpoint-file logs/rsl_rl/go1_velocity_supermimic_nocrouch/<run>/model_5000.pt
```

The play task samples straight walking commands (`lin_vel_x` 0.5-1.0, zero
yaw), which makes it easy to see whether the policy actually walks or just
stands still.


## GPU memory notes (8 GB laptop GPU)

Measured on an RTX 4060 Laptop (8 GB), rough task:

| `--env.scene.num-envs` | peak GPU memory | note |
|---|---|---|
| 128 | ~0.6 GB | comfortable |
| 256 | ~0.8 GB | comfortable |
| 512 | ~2.0 GB | tested end-to-end with the full runner |
| 2048 | ~3.1 GB | environment only, no runner buffer |
| 4096 | OOM | fails in `sim.forward` on the first step |

The replay buffer also scales with `num_envs * buffer_steps * xi_dim`; on the
rough task the 187-D height scan dominates. On 8 GB start with 512 (rough) or
1024 (flat), and reduce `buffer_steps` to 250 if needed. The paper's 4096
environments require a much larger GPU.

## Important deviations / caveats

* **Learned reward head.** Because the stock velocity reward contains contact
  terms (`air_time`, `foot_slip`, `soft_landing`) and terminations
  (`illegal_contact`) that are not a closed-form function of the reduced state,
  the reward and termination heads are fitted from data. They can be exploited
  by the policy if the world model is inaccurate; monitor `sim_rew` in
  `train_log.csv` / W&B, and increase `wm_only_iters`, `wm_batches` or the
  buffer size before trusting a long run. As a guard, policy updates only start
  once the normalized world-model error is below `wm_err_threshold` (default
  1.5) in addition to the `wm_only_iters` warm-up.
* **Behavior-cloning anchor.** A small MSE anchor to the logged action
  (`bc_coef`, default 0.1) keeps the policy near the replay distribution while
  the learned reward head is still improving. Set `bc_coef=0` for the paper's
  pure objective.
* **Terrain scan is held constant over the imagination horizon.** The 187-D
  scan is a per-transition context, not a predicted state; over `H=32` (0.64 s)
  this is a reasonable approximation but not exact on fast terrain changes.
* **Domain-randomization context is incomplete.** `xi` currently carries the
  terrain scan but not friction / encoder bias / CoM; the world model therefore
  fits the average dynamics across those randomizations. Adding them to `xi`
  (and to the observation head input) is the natural next step.
* **Push handling is an extension.** The paper removes root-velocity pushes
  because a deterministic model cannot represent the discontinuity. This port
  keeps them as an explicit one-step input `u_t`, sampled during imagination
  with the environment's push distribution.
* The paper-scale configuration (2048-wide MLP, 4096 imagined rollouts, 4096
  environments) needs an 80 GB GPU; the defaults here are deliberately smaller.
