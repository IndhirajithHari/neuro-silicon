"""Tests for desktop upgrade backends: auto-discover, live placement,
and the MCP-style assistant (all Qt-free, CI-safe)."""
import os
import zipfile

import pytest

from neurosilicon import assistant as assistant_lib
from neurosilicon import discover as discover_lib
from neurosilicon import liveplace as liveplace_lib
from neurosilicon import settings as settings_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def _write(root, rel, text):
  p = os.path.join(root, rel)
  os.makedirs(os.path.dirname(p), exist_ok=True)
  with open(p, 'w') as f:
    f.write(text)
  return p


def test_discover_rtl_folder(tmp_path):
  root = str(tmp_path / 'chip')
  _write(root, 'soc.v', 'module soc; leaf u(); endmodule\n')
  _write(root, 'leaf.v', 'module leaf; endmodule\n')
  _write(root, 'tb_soc.v', 'module tb_soc; soc u(); endmodule\n')
  d = discover_lib.discover_design(root)
  assert d['kind'] == 'rtl_project'
  assert d['needs_synth'] is True
  assert d['suggested_top'] == 'soc'
  assert 'leaf.v' in d['rtl_files']
  assert any('testbench' in line.lower() for line in d['found'] + d['notes'])
  plan = discover_lib.import_plan(d)
  assert any('yosys' in step.lower() for step in plan)


def test_discover_ready_netlist_folder(tmp_path):
  root = str(tmp_path / 'ready')
  _write(root, 'netlist.pb.txt', 'netlist {}\n')
  _write(root, 'initial.plc',
         '# Width : 200  Height : 100\n# Columns : 8  Rows : 4\n')
  d = discover_lib.discover_design(root)
  assert d['kind'] == 'ready_netlist'
  assert d['needs_synth'] is False
  assert d['ready'] is True
  assert d['netlist'].endswith('netlist.pb.txt')
  assert d['plc'].endswith('initial.plc')


def test_discover_single_file_and_zip(tmp_path):
  f = _write(str(tmp_path), 'MyChip.v', 'module MyChip; endmodule\n')
  d = discover_lib.discover_design(f)
  assert d['kind'] in ('rtl_project', 'rtl_files')
  assert d['suggested_top'] == 'MyChip'
  zpath = str(tmp_path / 'design.zip')
  with zipfile.ZipFile(zpath, 'w') as z:
    z.writestr('core/top.v', 'module top; endmodule\n')
  out = discover_lib.ensure_unpacked(zpath, str(tmp_path / 'unzipped'))
  assert os.path.isfile(os.path.join(out, 'core', 'top.v'))
  d2 = discover_lib.discover_design(out)
  assert d2['suggested_top'] == 'top'


def test_discover_sim_driver_only_folder(tmp_path):
  root = str(tmp_path / 'rocketish')
  _write(root, 'src/TestDriver.v',
         'module TestDriver;\n'
         '  reg clock = 0;\n'
         '  always #(`CLOCK_PERIOD/2.0) clock = ~clock;\n'
         'endmodule\n')
  d = discover_lib.discover_design(root)
  assert d['kind'] == 'unknown'
  assert any('synthesizable' in m for m in d['missing'])
  assert any('generated' in n.lower() for n in d['notes'])


def test_discover_missing_and_empty(tmp_path):
  with pytest.raises(ValueError):
    discover_lib.discover_design(str(tmp_path / 'nope'))
  empty = str(tmp_path / 'empty')
  os.makedirs(empty)
  d = discover_lib.discover_design(empty)
  assert d['kind'] == 'unknown'
  assert d['missing']


def test_assistant_offline_answers():
  llm = settings_lib.defaults()['llm']
  ctx = {'design': 'toy (2 macros)', 'canvas': (200.0, 100.0),
         'results': [{'method': 'first_legal', 'episode': 0,
                      'proxy_cost': 1.05, 'wirelength': 0.21,
                      'congestion': 0.73, 'density': 0.47}],
         'best_proxy': 1.05}
  text, used_llm = assistant_lib.answer('Explain the best run', ctx, llm)
  assert used_llm is False
  assert 'toy' in text and '1.05' in text
  mode, rects, note = assistant_lib.suggest_constraint(
      'Keep a 20% routing channel open in the center',
      (200.0, 100.0), llm)
  assert mode == 'block' and len(rects) == 1 and 'fallback' in note


def test_connection_explanation_hides_key():
  llm = settings_lib.defaults()['llm']
  off_text = assistant_lib.connection_explanation(llm)
  assert 'off' in off_text and 'Connectors' in off_text
  llm['provider'] = 'api'
  llm['api_key'] = 'sk-test'
  text = assistant_lib.connection_explanation(llm)
  assert 'gpt-4o-mini' in text and 'get_run_results' in text
  assert 'sk-test' not in text
  reply, used = assistant_lib.answer('what is connected?', {}, llm)
  assert used is False and 'Connected:' in reply


def test_guide_and_connector_status():
  guide = assistant_lib.GUIDE_TEXT
  for word in ('OpenAI', 'Ollama', 'LM Studio', 'MCP', 'API key'):
    assert word in guide
  llm = settings_lib.defaults()['llm']
  label, ok, detail = assistant_lib.connector_status(llm)
  assert ok is False and 'off' in (label + detail).lower()


def test_live_rollout_snapshots():
  snaps = []
  info, frames = liveplace_lib.run_live_rollout(
      TOY_NETLIST, TOY_PLC, fd_steps=(2, 2, 2), init_fd=False,
      on_snapshot=snaps.append)
  assert info['infeasible'] is False
  assert frames and frames[-1]['macros']
  assert snaps == frames or len(snaps) == len(frames)
  last = frames[-1]
  assert last['canvas'][0] > 0 and last['step'] >= 1
  m = last['macros'][0]
  assert {'name', 'x', 'y', 'w', 'h', 'soft'} <= set(m)
  # Moving-heat data: first frame has nothing placed, last has heat.
  assert liveplace_lib.frame_power(frames[0]) == {}
  heat = liveplace_lib.frame_power(last)
  assert heat and all(v > 0 for v in heat.values())
