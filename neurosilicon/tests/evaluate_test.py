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


def test_policy_rollout_runs():
  env = env_lib.ChipPlacementEnv(
      TOY_NETLIST, init_placement=TOY_PLC, fd_steps=(2, 2, 2), init_fd=False)
  policy = model_lib.CircuitPolicy()
  reward, info = evaluate_lib.rollout_policy(env, policy, greedy=True)
  assert info['infeasible'] is False
  assert reward < 0
