"""Tests for coordinate-descent fine-tuning."""
import numpy as np

from neurosilicon import env as env_lib
from neurosilicon import finetune as finetune_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_cd_does_not_regress_toy():
  env = env_lib.ChipPlacementEnv(
      TOY_NETLIST, init_placement=TOY_PLC, fd_steps=(2, 2, 2), init_fd=False)
  obs, _ = env.reset()
  done = False
  while not done:
    legal = np.flatnonzero(obs['mask'])
    obs, _, terminated, truncated, info = env.step(int(legal[0]))
    done = terminated or truncated
  before = info['proxy_cost']
  cost, parts = finetune_lib.run_cd(env.plc, epochs=1, k_distance_bound=2)
  assert np.isfinite(cost)
  assert cost <= before + 1e-9
  assert set(parts) == {'wirelength', 'congestion', 'density'}


def test_cd_defaults_match_alphachip():
  import inspect
  sig = inspect.signature(finetune_lib.run_cd)
  assert sig.parameters['use_stdcell_placer'].default is True
  assert sig.parameters['stdcell_placer'].default == 'fd'
  assert sig.parameters['epochs'].default == 2
  assert sig.parameters['k_distance_bound'].default == 5
