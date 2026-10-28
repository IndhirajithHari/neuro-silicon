"""Tests for GNN simulation (graph extract + recorded rollout)."""
import os

import numpy as np
import torch

from neurosilicon import gnn_viz as gnn_lib
from neurosilicon import render as render_lib
from neurosilicon import setup as setup_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_circuit_edges_shape():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  edges = gnn_lib.circuit_edges(plc)
  assert isinstance(edges, list) and len(edges) > 0
  for a, b, c in edges:
    assert a != b and c >= 1


def test_simulate_records_steps(tmp_path):
  from neurosilicon import model as model_lib
  torch.manual_seed(0)
  ckpt = os.path.join(str(tmp_path), 'p.pt')
  torch.save(model_lib.CircuitPolicy().state_dict(), ckpt)
  sim = gnn_lib.simulate(TOY_NETLIST, TOY_PLC, ckpt, fd_steps=(2, 2, 2),
                         init_fd=False)
  assert len(sim['steps']) > 0
  for s in sim['steps']:
    assert abs(float(np.sum(s['probs'])) - 1.0) < 1e-4
    assert isinstance(s['value'], float)
    assert len(s['placed']) == s['node_idx'] + 1
  assert set(sim) >= {'steps', 'final', 'order', 'grid', 'canvas',
                      'edges', 'soft_xy'}


def test_simulate_missing_checkpoint(tmp_path):
  try:
    gnn_lib.simulate(TOY_NETLIST, TOY_PLC,
                     os.path.join(str(tmp_path), 'no.pt'))
  except FileNotFoundError:
    pass
  else:
    raise AssertionError('expected FileNotFoundError')


def test_plot_sim_step_png():
  import matplotlib.pyplot as plt
  fig = render_lib.plot_sim_step(
      (200.0, 100.0),
      [{'name': 'ML', 'x': 50.0, 'y': 50.0, 'w': 20.0, 'h': 10.0}],
      np.full((10, 20), 1.0 / 200.0), 'MR', (5, 10), -1.2, 1, 2,
      edges=[('ML', 'MR', 3)])
  assert render_lib.figure_png(fig)[:8] == b'\x89PNG\r\n\x1a\n'
  plt.close(fig)
