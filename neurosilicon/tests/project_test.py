"""Tests for chip-project folder import (scan/params/command/passthrough)."""
import os

import pytest

from neurosilicon import project as project_lib


def _write(root, rel, text='module m; endmodule\n'):
  p = os.path.join(root, rel)
  os.makedirs(os.path.dirname(p), exist_ok=True)
  with open(p, 'w') as f:
    f.write(text)
  return p


def test_scan_folder_lists_rtl_and_tops(tmp_path):
  root = str(tmp_path)
  _write(root, 'a.v', 'module top; endmodule\n')
  _write(root, 'sub/b.v', 'module leaf; endmodule\n')
  _write(root, 'c.sv', 'module svc; endmodule\n')
  _write(root, 'notes.txt', 'ignore\n')
  out = project_lib.scan_folder(root)
  assert out['verilog'] == ['a.v', os.path.join('sub', 'b.v')]
  assert out['systemverilog'] == ['c.sv']
  assert out['tops'] == ['leaf', 'svc', 'top']


def test_scan_folder_errors(tmp_path):
  with pytest.raises(ValueError):
    project_lib.scan_folder(os.path.join(str(tmp_path), 'nope'))
  empty = os.path.join(str(tmp_path), 'empty')
  os.makedirs(empty)
  with pytest.raises(ValueError):
    project_lib.scan_folder(empty)


def test_parse_params():
  assert project_lib.parse_params('A=1, B = 128') == {'A': '1', 'B': '128'}
  assert project_lib.parse_params('') == {}
  with pytest.raises(ValueError):
    project_lib.parse_params('NOEQUALS')


def test_run_synth_absolutizes_and_wraps_errors(tmp_path):
  import os
  from unittest import mock
  build = os.path.join(str(tmp_path), 'b')
  seen = {}

  def fake_run(argv, **kw):
    seen['argv0'] = argv[0]
    seen['cwd'] = kw.get('cwd')
    with open(os.path.join(build, 'design.json'), 'w') as f:
      f.write('{}')
    return mock.Mock(returncode=0, stdout='Number of cells: 5\n',
                     stderr='')

  with mock.patch('subprocess.run', side_effect=fake_run):
    out = project_lib.run_synth(['a.v'], 'top', build, '/x/x.lib',
                                'tools/yosys')
  assert os.path.isabs(seen['argv0']) and os.path.isabs(seen['cwd'])
  assert out['cells'] == 5
  with mock.patch('subprocess.run',
                  side_effect=FileNotFoundError('nope')):
    try:
      project_lib.run_synth(['a.v'], 'top', build, '/x/x.lib',
                            'tools/yosys')
    except RuntimeError as e:
      assert 'launch failed' in str(e)
    else:
      raise AssertionError('expected RuntimeError')


def test_resolve_tool_anchors_at_repo_root():
  rel = project_lib.resolve_tool('tools/yosys')
  assert os.path.isabs(rel)
  assert rel == os.path.join(project_lib.repo_root(), 'tools', 'yosys')
  assert project_lib.resolve_tool('/abs/path') == '/abs/path'


def test_run_synth_timeout_writes_partial_log(tmp_path):
  import subprocess
  from unittest import mock
  build = os.path.join(str(tmp_path), 'b')
  err = subprocess.TimeoutExpired(cmd='yosys', timeout=5,
                                  output='reading..\n', stderr='warn\n')
  with mock.patch('subprocess.run', side_effect=err):
    try:
      project_lib.run_synth(['a.v'], 'top', build, '/x/x.lib',
                            'tools/yosys', timeout=5)
    except RuntimeError as e:
      assert 'timed out after 5s' in str(e)
    else:
      raise AssertionError('expected RuntimeError')
  with open(os.path.join(build, 'synth.log')) as f:
    log = f.read()
  assert '[TIMEOUT after 5s]' in log and 'reading..' in log


def test_run_synth_missing_binary_is_runtime_error(tmp_path):
  try:
    project_lib.run_synth(['a.v'], 'top', str(tmp_path), '/x/x.lib',
                          'tools/definitely_not_a_tool')
  except RuntimeError as e:
    assert 'not found' in str(e)
  else:
    raise AssertionError('expected RuntimeError')


def test_import_project_passes_init_fd(tmp_path):
  from unittest import mock
  folder = os.path.join(str(tmp_path), 'chip')
  os.makedirs(folder)
  _write(folder, 'a.v', 'module top; endmodule\n')
  for f in ('x.lef', 'x.lib', 'x.tlef'):
    _write(str(tmp_path), f, 'stub\n')
  settings = {'pdk': {
      'lef': os.path.join(str(tmp_path), 'x.lef'),
      'liberty': os.path.join(str(tmp_path), 'x.lib'),
      'techlef': os.path.join(str(tmp_path), 'x.tlef'),
      'yosys': 'tools/yosys'}}
  seen = {}

  def fake_baselines(netlist, plc, **kw):
    seen.update(kw)
    return [{'proxy_cost': 1.0}]

  with mock.patch.object(
      project_lib, 'run_synth',
      return_value={'json': 'd.json', 'net': 'd.v', 'log': 's.log',
                    'cells': 5}), \
      mock.patch('neurosilicon.synth_to_proto.Converter') as conv, \
      mock.patch('neurosilicon.evaluate.run_baselines',
                 side_effect=fake_baselines):
    conv.return_value.convert.return_value = {'netlist': 'n', 'plc': 'p'}
    project_lib.import_project(
        folder, ['a.v'], 'top', os.path.join(str(tmp_path), 'out'),
        settings, episodes=1, init_fd=False, cluster_size=200,
        grid_sweep_step=8)
  assert seen['init_fd'] is False
  assert seen['fd_steps'] == (5, 5, 5)
  _, conv_kw = conv.return_value.convert.call_args
  assert conv_kw == {'cluster_size': 200, 'grid_sweep_step': 8}


def test_scan_accepts_single_file_and_uppercase(tmp_path):
  root = str(tmp_path)
  f = _write(root, 'MyChip.V', 'module MyChip; leaf u(); endmodule\n')
  _write(root, 'leaf.v', 'module leaf; endmodule\n')
  out = project_lib.scan_folder(f)
  assert out['verilog'] == ['MyChip.V']
  assert out['tops'] == ['MyChip']
  assert out['suggested_top'] == 'MyChip'


def test_scan_suggests_root_top_and_excludes_testbenches(tmp_path):
  root = str(tmp_path)
  _write(root, 'top.v',
         'module soc; leaf u(); endmodule\n')
  _write(root, 'leaf.v', 'module leaf; endmodule\n'
         'module aaa_stub; endmodule\n')
  _write(root, 'tb_soc.v', 'module tb_soc; soc u(); endmodule\n')
  out = project_lib.scan_folder(root)
  assert out['suggested_top'] == 'soc'
  assert out['testbenches'] == ['tb_soc.v']
  assert project_lib.default_files(out) == ['leaf.v', 'top.v']


def test_import_files_rejects_dupes_and_missing(tmp_path):
  a = _write(str(tmp_path / 'x'), 'same.v')
  b = _write(str(tmp_path / 'y'), 'same.v')
  settings = {'pdk': {}}
  with pytest.raises(ValueError, match='duplicate'):
    project_lib.import_files([a, b], 'm', str(tmp_path / 'o'), settings)
  with pytest.raises(ValueError, match='missing'):
    project_lib.import_files([str(tmp_path / 'nope.v')], 'm',
                             str(tmp_path / 'o'), settings)


def test_register_netlist_validates_pair(tmp_path):
  net = _write(str(tmp_path), 'n.pb.txt', 'netlist {}\n')
  plc = _write(str(tmp_path), 'i.plc',
               '# Width : 10  Height : 10\n# Columns : 4  Rows : 4\n')
  reg = os.path.join(str(tmp_path), 'reg.json')
  entry = project_lib.register_netlist('d', net, plc, path=reg)
  assert entry['netlist'] == os.path.abspath(net)
  with pytest.raises(ValueError, match='canvas'):
    project_lib.register_netlist('d2', net,
                                 _write(str(tmp_path), 'bad.plc', 'x\n'),
                                 path=reg)


def test_prune_to_top_drops_unreachable_variants():
  texts = {
      'top.v': 'module soc; leaf u(); endmodule\n',
      'leaf.v': 'module leaf; endmodule\n',
      'leaf_fpga.v': 'module leaf; endmodule\n',  # duplicate variant
      'tb/tb_soc.v': 'module tb_soc; soc u(); endmodule\n',
  }
  kept, dropped = project_lib.prune_to_top(texts, 'soc')
  assert kept == ['leaf.v', 'top.v']
  assert 'leaf_fpga.v' in dropped and 'tb/tb_soc.v' in dropped
  assert project_lib.is_testbench('hw/tb/dump.v')
  assert not project_lib.is_testbench('hw/amber23/a23_core.v')


def test_synth_command_shape():
  cmd = project_lib.synth_command(['a.v', 'b.sv'], 'top', 'build',
                                  '/x/x.lib', {'M': '8'})
  assert 'read_verilog a.v; read_verilog -sv b.sv;' in cmd
  assert 'chparam -set M 8 top' in cmd
  assert 'hierarchy -check -top top' in cmd
  assert 'autoname' in cmd
  assert 'write_json build/design.json' in cmd
  assert 'write_verilog -noattr build/design_net.v' in cmd


def test_synth_text_strips_simulation_only_code():
  src = (
      'module top;\n'
      'real period = 6000;\n'
      'integer n = 0;\n'
      'initial begin\n'
      '  n = $time;\n'
      'end\n'
      'initial begin\n'
      '  $readmemh("boot.hex", mem);\n'
      'end\n'
      'always @(posedge clk) q <= d;\n'
      'endmodule\n')
  out = project_lib._synth_text(src)
  assert 'integer period = 6000;' in out
  assert '$time' not in out
  assert '$readmemh' in out  # preloads survive staging
  assert 'always @(posedge clk) q <= d;' in out


def test_huge_memory_modules_detected():
  texts = {
      'mem.v': ('module main_mem #(parameter W = 32) (input clk);\n'
                'reg [127:0] ram [2**(MSB-2)-1:0];\nendmodule\n'),
      'cfg.v': 'module cfg; localparam MSB = 26; endmodule\n',
      'small.v': ('module small; reg [7:0] fifo [15:0]; endmodule\n'),
  }
  huge = project_lib.huge_memory_modules(texts)
  assert huge == {'main_mem': 128 * 2**24}
  stub = project_lib._module_stub(texts['mem.v'], 'main_mem')
  assert stub.startswith('(* blackbox *) module main_mem')
  assert 'reg [127:0]' not in stub and 'input clk' in stub


def test_sim_only_detects_test_driver():
  texts = {'TestDriver.v': (
      'module TestDriver;\n'
      '  reg clock = 0;\n'
      '  always #(`CLOCK_PERIOD/2.0) clock = ~clock;\n'
      '  initial verbose = $test$plusargs("verbose");\n'
      'endmodule\n')}
  assert project_lib.sim_only_modules(texts) == {'TestDriver'}


def test_sim_only_modules_detected():
  texts = {
      'clk.v': ('module clocks_resets (input p, output reg c);\n'
                'always @(posedge p) begin\n'
                '  # (1000 / 2)\n'
                '  c = 1;\n'
                'end\nendmodule\n'),
      'ok.v': ('module leaf #(parameter W = 2) (input clk);\n'
               'sub #(.W(W)) u (.clk(clk));\nendmodule\n'),
  }
  assert project_lib.sim_only_modules(texts) == {'clocks_resets'}


def test_blackbox_stubs_for_vendor_primitives():
  texts = {'x.v': 'module top; DSP48A1 #(.A1REG(0)) u_dsp '
                  '(.CLK(c), .P(p)); endmodule\n'}
  out = project_lib.blackbox_stubs(texts, {'top'})
  assert 'module DSP48A1' in out and 'CLK' in out and 'A1REG' in out
  assert project_lib.blackbox_stubs(texts, {'top', 'DSP48A1'}) == ''
