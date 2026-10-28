"""Smoke test: tiny PPO run completes and checkpoints."""
import csv
import os

import numpy as np

from neurosilicon import train as train_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_tiny_ppo_run(tmp_path):
  outdir = str(tmp_path / 'train')
  history, best = train_lib.train(
      [TOY_NETLIST, TOY_NETLIST],
      init_placements=[TOY_PLC, TOY_PLC],
      outdir=outdir,
      iterations=1,
      episodes_per_iter=2,
      eval_every=1,
      fd_steps=(2, 2, 2), init_fd=False)
  assert len(history) == 1
  assert np.isfinite(history[0][1])
  assert np.isfinite(best) and best > 0
  assert os.path.isfile(os.path.join(outdir, 'latest.pt'))
  assert os.path.isfile(os.path.join(outdir, 'best.pt'))
  assert os.path.isfile(os.path.join(outdir, 'best_policy_0.plc'))
  assert os.path.isfile(os.path.join(outdir, 'best_policy_1.plc'))


def test_cosine_lr_endpoints():
  assert train_lib.cosine_lr(0.004, 0, 10) == 0.004
  assert abs(train_lib.cosine_lr(0.004, 10, 10) - 0.0004) < 1e-12
  assert abs(train_lib.cosine_lr(0.004, 5, 10) - 0.0022) < 1e-12
  # Past the horizon clamps at the floor.
  assert train_lib.cosine_lr(0.004, 99, 10) == train_lib.cosine_lr(
      0.004, 10, 10)


def test_tiny_ppo_writes_history(tmp_path):
  outdir = str(tmp_path / 'train')
  history, _ = train_lib.train(
      [TOY_NETLIST], init_placements=[TOY_PLC], outdir=outdir,
      iterations=1, episodes_per_iter=1, eval_every=1,
      fd_steps=(2, 2, 2), init_fd=False)
  with open(os.path.join(outdir, 'history.csv')) as f:
    assert len(list(csv.reader(f))) == 2
  assert len(history) == 1
