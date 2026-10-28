"""Tests for the Yosys-JSON to proto converter (hermetic fixture)."""
import os

import numpy as np

from neurosilicon import env as env_lib
from neurosilicon import setup as setup_lib
from neurosilicon import synth_to_proto as converter_lib

JSON = 'neurosilicon/tests/testdata/synth_small.json'
LEF = 'neurosilicon/tests/testdata/mini.lef'


def test_convert_small_hierarchy(tmp_path):
  stats = converter_lib.Converter(JSON, LEF, 'mini').convert(str(tmp_path))
  assert stats['macros'] == 4
  assert stats['nets'] == 5
  assert stats['ports'] == 2
  assert stats['warnings'] == []
  assert os.path.isfile(stats['netlist'])
  assert os.path.isfile(stats['plc'])
  plc = setup_lib.create_plc(stats['netlist'],
                             init_placement=stats['plc'])
  assert plc.hard_macro_cnt == 4
  assert plc.get_cost() >= 0.0


def test_converted_env_episode(tmp_path):
  stats = converter_lib.Converter(JSON, LEF, 'mini').convert(str(tmp_path))
  env = env_lib.ChipPlacementEnv(stats['netlist'],
                                 init_placement=stats['plc'],
                                 fd_steps=(2, 2, 2), init_fd=False)
  obs, _ = env.reset()
  done = False
  while not done:
    legal = np.flatnonzero(obs['mask'])
    assert len(legal) > 0
    obs, reward, terminated, truncated, info = env.step(int(legal[0]))
    done = terminated or truncated
  assert info['infeasible'] is False
  assert np.isfinite(reward)


def _bare_converter():
  conv = converter_lib.Converter.__new__(converter_lib.Converter)
  conv.warnings = []
  conv.macros = {}
  conv.stdcells = {}
  return conv


def test_cluster_skips_high_fanout_nets():
  conv = _bare_converter()
  for i in range(600):
    conv.stdcells[f'c{i}'] = {'w': 1.0, 'h': 1.0}
  nets = [{'nodes': [('stdcell', f'c{i}') for i in range(600)]},
          {'nodes': [('stdcell', 'c0'), ('stdcell', 'c1')]}]
  assign = conv._cluster_stdcells(nets, target_size=25)
  # The 600-fanout net is ignored: only the 2-pin net groups c0/c1.
  assert assign.get('c0') == assign.get('c1')
  assert 'c2' not in assign


def test_cluster_area_cap_breaks_up_groups():
  conv = _bare_converter()
  for i in range(40):
    conv.stdcells[f'c{i}'] = {'w': 10.0, 'h': 10.0}
  chain = [{'nodes': [('stdcell', f'c{i}'), ('stdcell', f'c{i + 1}')]
            } for i in range(39)]
  assign = conv._cluster_stdcells(chain, target_size=25)
  # Total 4000 area -> cap 250 -> at most 2 cells of 100 area each.
  groups = {}
  for k, g in assign.items():
    groups.setdefault(g, []).append(k)
  assert len(groups) >= 10
  assert all(len(m) <= 2 for m in groups.values())


def test_convert_selects_grid_and_smooth5(tmp_path):
  stats = converter_lib.Converter(JSON, LEF, 'mini').convert(
      str(tmp_path), auto_grid=True, grid_sweep_step=8)
  assert stats['grid'][0] * stats['grid'][1] >= 100
  with open(stats['plc']) as f:
    header = f.read()
  assert 'Smoothing factor : 5' in header
