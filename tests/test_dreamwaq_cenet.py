"""Numerical-safety tests for the DreamWaQ CENet."""

import torch

from mjlab.tasks.dreamwaq.rl.network.cenet import CENet


def test_posterior_log_variance_is_clamped():
  torch.manual_seed(0)
  cenet = CENet(
    obs_dim=4,
    history_length=2,
    latent_dim=2,
    encoder_hidden_dims=(4,),
    decoder_hidden_dims=(4,),
  )
  head = cenet.encoder[-1]
  assert isinstance(head, torch.nn.Linear)

  # Drive the log-variance head far outside the range in both directions.
  head.bias.data[cenet.velocity_dim + cenet.latent_dim :] = 100.0
  output = cenet(torch.randn(3, cenet.input_dim))
  assert torch.all(output.log_var == cenet.log_var_range[1])
  assert torch.isfinite(output.z).all()
  assert torch.isfinite(CENet.kl_divergence(output.mu, output.log_var)).all()

  head.bias.data[cenet.velocity_dim + cenet.latent_dim :] = -100.0
  output = cenet(torch.randn(3, cenet.input_dim))
  assert torch.all(output.log_var == cenet.log_var_range[0])
  assert torch.isfinite(output.z).all()
  assert torch.isfinite(CENet.kl_divergence(output.mu, output.log_var)).all()
