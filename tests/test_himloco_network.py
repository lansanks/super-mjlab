"""Numerical tests for the HIM estimator and actor."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab.tasks.himloco.rl.network.actor import HimActorModel
from mjlab.tasks.himloco.rl.network.estimator import HIMEstimator, sinkhorn
from mjlab.viewer.viewer_config import ViewerConfig

NUM_ENVS = 4
OBS_DIM = 6
HISTORY = 3
STATE_DIM = 5
ACTION_DIM = 2
LATENT_DIM = 4
OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
  "him": ["him_input"],
}


def make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
  """Fake observations with the HIM groups."""
  return TensorDict(
    {
      "actor": torch.randn(num_envs, OBS_DIM),
      "critic": torch.randn(num_envs, OBS_DIM),
      "him_input": torch.randn(num_envs, HISTORY, OBS_DIM),
      "him_state": torch.randn(num_envs, STATE_DIM),
    },
    batch_size=[num_envs],
  )


def make_estimator() -> HIMEstimator:
  torch.manual_seed(0)
  return HIMEstimator(
    obs_dim=OBS_DIM,
    history_length=HISTORY,
    state_dim=STATE_DIM,
    latent_dim=LATENT_DIM,
    encoder_hidden_dims=(8,),
    target_hidden_dims=(8,),
    num_prototypes=3,
  )


def make_actor() -> HimActorModel:
  torch.manual_seed(0)
  return HimActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(8,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    him_cfg={
      "obs_group": "him_input",
      "state_group": "him_state",
      "latent_dim": LATENT_DIM,
      "encoder_hidden_dims": (8,),
      "target_hidden_dims": (8,),
      "num_prototypes": 3,
    },
  )


def test_estimator_shapes_and_normalization():
  estimator = make_estimator().eval()
  history = torch.randn(NUM_ENVS, HISTORY, OBS_DIM)
  state = torch.randn(NUM_ENVS, STATE_DIM)
  velocity, latent = estimator.encode_history(history)
  assert velocity.shape == (NUM_ENVS, HIMEstimator.velocity_dim)
  assert latent.shape == (NUM_ENVS, LATENT_DIM)
  assert torch.allclose(latent.norm(dim=-1), torch.ones(NUM_ENVS), atol=1e-5)
  target = estimator.encode_target(state)
  assert target.shape == (NUM_ENVS, LATENT_DIM)
  assert torch.allclose(target.norm(dim=-1), torch.ones(NUM_ENVS), atol=1e-5)
  # The flattened form must give the same result as the 3D form.
  flat_velocity, flat_latent = estimator.encode_history(history.flatten(1))
  assert torch.allclose(velocity, flat_velocity, atol=1e-6)
  assert torch.allclose(latent, flat_latent, atol=1e-6)


def test_actor_publishes_latent_and_viewer_config():
  actor = make_actor()
  env = SimpleNamespace(
    num_envs=NUM_ENVS,
    device="cpu",
    cfg=SimpleNamespace(viewer=ViewerConfig()),
  )
  actor.bind_debug_env(env)
  actor(make_obs())
  output = actor.last_him_output
  assert output is not None
  torch.testing.assert_close(env.him_latent_estimate, output.latent)
  assert env.cfg.viewer.latent_bar_attr == "him_latent_estimate"
  assert env.cfg.viewer.latent_bar_range == (-1.0, 1.0)


def test_history_is_converted_to_newest_first():
  estimator = make_estimator()
  history = torch.arange(NUM_ENVS * HISTORY * OBS_DIM, dtype=torch.float32)
  history = history.reshape(NUM_ENVS, HISTORY, OBS_DIM)
  frames = estimator._prepare_history(history)
  assert torch.equal(frames, history.flip(1))
  # A flattened history is reshaped before the flip.
  flat = estimator._prepare_history(history.flatten(1))
  assert torch.equal(flat, history.flip(1))


def cosine_scores(batch: int = 7, prototypes: int = 3, dim: int = 4) -> torch.Tensor:
  """Cosine similarities, the domain expected by :func:`sinkhorn`."""
  embeddings = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)
  prototypes_t = torch.nn.functional.normalize(torch.randn(prototypes, dim), dim=-1)
  return embeddings @ prototypes_t.T


def reference_sinkhorn(out: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
  """Independent transcription of the released Sinkhorn normalization."""
  plan = torch.exp(out / eps).T
  num_prototypes, batch = plan.shape[0], plan.shape[1]
  plan = plan / plan.sum()
  for _ in range(iters):
    plan = plan / plan.sum(dim=1, keepdim=True) / num_prototypes
    plan = plan / plan.sum(dim=0, keepdim=True) / batch
  return (plan * batch).T


def test_sinkhorn_matches_the_released_implementation():
  torch.manual_seed(0)
  scores = cosine_scores()
  q = sinkhorn(scores, eps=0.05, iters=3)
  assert q.shape == (7, 3)
  assert (q >= 0).all()
  assert torch.isfinite(q).all()
  assert torch.allclose(q, reference_sinkhorn(scores, eps=0.05, iters=3), atol=1e-6)
  # Each sample's assignment is a probability distribution over prototypes.
  assert torch.allclose(q.sum(dim=1), torch.ones(7), atol=1e-5)


def test_sinkhorn_balances_prototypes_when_converged():
  torch.manual_seed(0)
  scores = cosine_scores()
  # With the released hyperparameters (eps=0.05, 3 iterations) the plan is close
  # to a hard assignment and balance is only partially enforced; with a smooth
  # plan the alternating normalization converges to equal prototype masses.
  q = sinkhorn(scores, eps=1.0, iters=30)
  assert torch.allclose(q.sum(dim=1), torch.ones(7), atol=1e-5)
  assert torch.allclose(q.sum(dim=0), torch.full((3,), 7.0 / 3.0), atol=1e-4)


def test_losses_are_finite_and_normalize_prototypes():
  estimator = make_estimator()
  history = torch.randn(NUM_ENVS, HISTORY, OBS_DIM)
  state = torch.randn(NUM_ENVS, STATE_DIM)
  est_loss, swap_loss, diagnostics = estimator.losses(history, state)
  assert torch.isfinite(est_loss)
  assert torch.isfinite(swap_loss)
  norms = estimator.proto.weight.norm(dim=-1)
  assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
  assert 0.0 <= diagnostics["proto_entropy"] <= 1.0
  assert 0.0 <= diagnostics["proto_active"] <= estimator.num_prototypes
  assert 0.0 <= diagnostics["swap_accuracy"] <= 1.0


def test_velocity_head_learns_the_successor_velocity():
  torch.manual_seed(0)
  estimator = make_estimator()
  optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)
  history = torch.randn(32, HISTORY, OBS_DIM)
  state = torch.randn(32, STATE_DIM)
  first = None
  est_loss = None
  for _ in range(30):
    est_loss, swap_loss, _ = estimator.losses(history, state)
    optimizer.zero_grad()
    (est_loss + swap_loss).backward()
    optimizer.step()
    if first is None:
      first = float(est_loss.detach())
  assert first is not None
  assert est_loss is not None
  assert float(est_loss.detach()) < first


def test_actor_latent_dim_and_gradient_routing():
  actor = make_actor()
  obs = make_obs()
  assert actor.him_group == "him_input"
  assert actor.state_group == "him_state"
  assert actor._get_latent_dim() == OBS_DIM + HIMEstimator.velocity_dim + LATENT_DIM

  actions = actor(obs, stochastic_output=True)
  # The policy path is stop-gradient: a policy-only loss must not reach the
  # estimator, while the policy MLP receives gradients.
  actor.get_output_log_prob(actions).sum().backward()
  for parameter in actor.estimator.parameters():
    assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
  assert any(
    parameter.grad is not None and parameter.grad.abs().sum() > 0
    for parameter in actor.mlp.parameters()
  )

  # The estimator loss only drives the estimator.
  actor.zero_grad()
  estimator = actor.estimator
  est_loss, swap_loss, _ = estimator.losses(
    obs["him_input"].detach(), obs["him_state"].detach()
  )
  (est_loss + swap_loss).backward()
  for parameter in actor.mlp.parameters():
    assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
  assert any(
    parameter.grad is not None and parameter.grad.abs().sum() > 0
    for parameter in estimator.parameters()
  )


def test_onnx_wrapper_uses_the_history():
  actor = make_actor().eval()
  onnx_model = actor.as_onnx(verbose=False)
  history = torch.randn(2, HISTORY * OBS_DIM)
  actions = onnx_model(history)
  assert actions.shape == (2, ACTION_DIM)
  assert onnx_model.input_size == HISTORY * OBS_DIM
