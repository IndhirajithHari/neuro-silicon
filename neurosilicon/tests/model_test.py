"""Tests for the PyTorch edge-GNN policy."""
import numpy as np
import torch

from neurosilicon import env as env_lib
from neurosilicon import model as model_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def _batch(obs, n=2):
  return {k: torch.stack([torch.as_tensor(obs[k]) for _ in range(n)])
          for k in obs}


def test_forward_shapes_and_masking():
  torch.manual_seed(0)
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC, init_fd=False)
  obs, _ = env.reset()
  policy = model_lib.CircuitPolicy()
  batch = _batch(obs)
  logits, value = policy(batch)
  assert logits.shape == (2, 128 * 128)
  assert value.shape == (2, 1)
  masked = policy.masked_logits(batch)
  illegal = (batch['mask'] == 0)
  assert torch.all(masked[illegal] < -1e8)
  # Masked sampling only picks legal cells.
  dist = torch.distributions.Categorical(logits=masked)
  for _ in range(20):
    a = int(dist.sample()[0])
    assert batch['mask'][0, a] == 1


def test_mask_floor_matches_alphachip():
  assert model_lib.MASK_LOGIT == -(2.0**32) + 1
  torch.manual_seed(0)
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC,
                                 init_fd=False)
  obs, _ = env.reset()
  policy = model_lib.CircuitPolicy()
  batch = _batch(obs, n=1)
  masked = policy.masked_logits(batch)
  illegal = (batch['mask'] == 0)
  assert torch.all(masked[illegal] == model_lib.MASK_LOGIT)


def test_glorot_init_biases_zero():
  torch.manual_seed(0)
  policy = model_lib.CircuitPolicy()
  n_lin = 0
  for m in policy.modules():
    if isinstance(m, torch.nn.Linear):
      n_lin += 1
      assert torch.all(m.bias == 0.0)
      assert float(m.weight.std()) > 0.0
  assert n_lin > 5


def test_dirichlet_noise_exact_math():
  torch.manual_seed(0)
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC,
                                 init_fd=False)
  obs, _ = env.reset()
  batch = _batch(obs, n=1)
  plain = model_lib.CircuitPolicy()
  logits, _ = plain(batch)
  # Default weight 0.0 is a no-op.
  assert torch.equal(plain.add_noise(logits), logits)
  noisy = model_lib.CircuitPolicy(policy_noise_weight=0.25)
  torch.manual_seed(7)
  out = noisy.add_noise(logits)
  torch.manual_seed(7)
  probs = torch.softmax(logits, dim=-1)
  expect = torch.log(
      0.75 * probs + 0.25 * torch.distributions.Dirichlet(
          torch.full_like(probs, 0.1)).sample() + model_lib.EPSILON)
  assert torch.allclose(out, expect)
  # Noise applies before masking.
  masked = noisy.masked_logits(batch, apply_noise=True)
  illegal = (batch['mask'] == 0)
  assert torch.all(masked[illegal] == model_lib.MASK_LOGIT)


def test_backward_flows():
  torch.manual_seed(1)
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC, init_fd=False)
  obs, _ = env.reset()
  policy = model_lib.CircuitPolicy()
  batch = _batch(obs, n=1)
  logits, value = policy(batch)
  loss = logits.sum() + value.sum()
  loss.backward()
  grads = [p.grad for p in policy.parameters() if p.requires_grad]
  assert grads and all(g is not None for g in grads)
