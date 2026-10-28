"""Tests for rollouts and baselines."""
import json
import os

from neurosilicon import env as env_lib
from neurosilicon import evaluate as evaluate_lib
from neurosilicon import model as model_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_random_baseline_finite(tmp_path):
  outdir = str(tmp_path / 'eval')
  results = evaluate_lib.run_baselines(
      TOY_NETLIST, TOY_PLC, episodes=2, fd_steps=(2, 2, 2), init_fd=False, outdir=outdir)
  assert len(results) == 4
  assert all(r['proxy_cost'] > 0 for r in results)
  assert os.path.isfile(os.path.join(outdir, 'results.json'))
  assert os.path.isfile(os.path.join(outdir, 'best_baseline.plc'))
  with open(os.path.join(outdir, 'results.json')) as f:
    assert len(json.load(f)) == 4


def test_env_reset_beyond_5000_nodes():
  """Designs over the 5000-node obs cap must reset (openc906: 16k).

  Regression: the extractor sized a scratch array by the cap, so
  node index 7445 crashed reset. Skips where the user-local
  imported-design registry is absent (CI checkouts).
  """
  import pytest
  from neurosilicon import project as project_lib
  entry = project_lib.load_registry().get('openc906')
  if not entry or not os.path.isfile(entry.get('netlist', '')):
    pytest.skip('openc906 not imported on this machine')
  env = env_lib.ChipPlacementEnv(
      entry['netlist'], init_placement=entry['plc'],
      fd_steps=(2, 2, 2), init_fd=False)
  obs, _ = env.reset()
  assert obs['mask'].shape[0] > 0
  assert len(env.placement_order) > 0


def test_policy_rollout_runs():
  env = env_lib.ChipPlacementEnv(
      TOY_NETLIST, init_placement=TOY_PLC, fd_steps=(2, 2, 2), init_fd=False)
  policy = model_lib.CircuitPolicy()
  reward, info = evaluate_lib.rollout_policy(env, policy, greedy=True)
  assert info['infeasible'] is False
  assert reward < 0
