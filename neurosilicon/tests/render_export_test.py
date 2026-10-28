"""Tests for rendering and DEF export."""
import os

from neurosilicon import export_def as export_def_lib
from neurosilicon import render as render_lib
from neurosilicon import setup as setup_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_save_figures(tmp_path):
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  paths = render_lib.save_figures(plc, str(tmp_path))
  for p in paths:
    assert os.path.isfile(p) and os.path.getsize(p) > 1000


def test_thermal_and_charts(tmp_path):
  import matplotlib.pyplot as plt
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  power = render_lib.estimate_power(plc)
  assert abs(sum(power.values()) - 1.0) < 1e-6
  paths = render_lib.save_figures(plc, str(tmp_path), power=power)
  assert len(paths) == 3
  for p in paths:
    assert os.path.isfile(p) and os.path.getsize(p) > 1000
  rows = [{'method': 'random', 'episode': 0, 'proxy_cost': 1.2,
           'wirelength': 0.3, 'congestion': 0.8, 'density': 0.5}]
  png = render_lib.figure_png(render_lib.cost_breakdown_figure(rows))
  assert png[:8] == b'\x89PNG\r\n\x1a\n'
  hist = render_lib.history_figure(
      [{'iter': 0, 'mean_reward': -1.0, 'eval_cost': float('nan')},
       {'iter': 1, 'mean_reward': -0.9, 'eval_cost': 1.3}])
  assert render_lib.figure_png(hist)[:8] == b'\x89PNG\r\n\x1a\n'
  plt.close('all')


def test_floorplan_blocked_overlay():
  import matplotlib.pyplot as plt
  import numpy as np
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  cols, rows = plc.get_grid_num_columns_rows()
  blocked = np.zeros((rows, cols), dtype=bool)
  blocked[:, :cols // 2] = True
  fig = render_lib.plot_floorplan(plc, blocked=blocked)
  assert render_lib.figure_png(fig)[:8] == b'\x89PNG\r\n\x1a\n'
  plt.close(fig)


def test_export_def(tmp_path):
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  path = export_def_lib.export_def(plc, str(tmp_path / 'toy.def'))
  text = open(path).read()
  assert 'DIEAREA ( 0 0 ) ( 200000 100000 ) ;' in text
  assert 'COMPONENTS 14 ;' in text
  assert 'PINS 12 ;' in text
  assert text.strip().endswith('END DESIGN')
