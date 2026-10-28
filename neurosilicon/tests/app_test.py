"""Tests for GUI helpers (no Streamlit runtime needed)."""
import numpy as np

from neurosilicon import app as app_lib


def test_build_mask_fn_kinds():
  fn, _ = app_lib.build_mask_fn('None', 20, 10, 200.0, 100.0)
  assert fn is None
  fn, _ = app_lib.build_mask_fn('Center keep-out channel', 20, 10, 200.0,
                                100.0)
  out = fn(0, np.ones(200, dtype=np.int32))
  assert int(np.sum(out)) < 200
  fn, _ = app_lib.build_mask_fn('Chat constraint', 20, 10, 200.0, 100.0,
                                'Keep center open')
  out = fn(0, np.ones(200, dtype=np.int32))
  assert int(np.sum(out)) < 200


def test_run_episodes_first_legal_and_fallback():
  rows, plc, cost, notes = app_lib.run_episodes(
      'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt',
      'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc',
      'first_legal', 1, 2, None, None, '', 0, 0)
  assert len(rows) == 1
  assert cost > 0 and plc is not None
  rows, plc, cost, notes = app_lib.run_episodes(
      'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt',
      'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc',
      'policy', 1, 2, None, None, '/nonexistent.pt', 0, 0)
  assert rows[0]['method'] == 'first_legal'
  assert any('fall back' in n or 'fell back' in n for n in notes)


def test_missing_api_fresh_tree():
  assert app_lib.missing_api() == []


def test_missing_api_detects_stale():
  import neurosilicon.render as render_lib
  fn = render_lib.estimate_power
  try:
    del render_lib.estimate_power
    assert ('neurosilicon.render.estimate_power'
            in app_lib.missing_api())
  finally:
    render_lib.estimate_power = fn
  assert app_lib.missing_api() == []


def test_reload_stale_roundtrip():
  done = app_lib.reload_stale(['neurosilicon.render.estimate_power',
                               'neurosilicon.render.plot_thermal'])
  assert done == ['neurosilicon.render']
  assert app_lib.missing_api() == []


def test_browse_folder_zenity_success():
  from unittest import mock
  proc = mock.Mock(returncode=0, stdout='/tmp/chip\n')
  with mock.patch('shutil.which', return_value='/usr/bin/zenity'), \
       mock.patch('subprocess.run', return_value=proc) as m:
    path, available = app_lib.browse_folder()
  assert (path, available) == ('/tmp/chip', True)
  assert '--directory' in m.call_args[0][0]


def test_browse_folder_no_backend():
  from unittest import mock
  with mock.patch('shutil.which', return_value=None):
    with mock.patch.dict('sys.modules', {'tkinter': None}):
      path, available = app_lib.browse_folder()
  assert (path, available) == (None, False)


def test_placement_details_toy():
  from neurosilicon import setup as setup_lib
  plc = setup_lib.create_plc(
      'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt',
      init_placement=('circuit_training/environment/test_data/'
                      'toy_macro_stdcell/initial.plc'))
  d = app_lib.placement_details(plc)
  assert d['counts']['hard_macros'] == 2
  assert {m['macro'] for m in d['macros']} == {'ML', 'MR'}
  assert set(d) == {'macros', 'counts', 'canvas', 'grid', 'utilization'}
  assert d['utilization'] >= 0.0
  assert all(set(m) == {'macro', 'x', 'y', 'w', 'h', 'area', 'orient'}
             for m in d['macros'])


def test_run_episodes_callback():
  fired = []

  def cb(ep, row, plc, secs):
    fired.append((ep, row['proxy_cost'], plc is not None, secs >= 0))

  rows, _, _, _ = app_lib.run_episodes(
      'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt',
      'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc',
      'first_legal', 2, 2, None, None, '', 0, 0, on_episode=cb)
  assert len(rows) == 2 and len(fired) == 2
  assert all('top10cong' in r and r['top10cong'] >= 0 for r in rows)
  assert [f[0] for f in fired] == [0, 1]
  assert all(f[3] for f in fired)


def test_checkpoints_cover_sky130_designs():
  for name in app_lib.CHECKPOINTS:
    assert name in app_lib.AVAILABLE_NETLISTS


def test_netlist_options_exist():
  import os
  for _, (netlist, plc) in app_lib.AVAILABLE_NETLISTS.items():
    assert os.path.isfile(netlist)
    assert os.path.isfile(plc)


def test_all_netlists_absolute():
  import os
  nets = app_lib.all_netlists()
  assert nets
  for _, (netlist, plc) in nets.items():
    assert os.path.isabs(netlist) and os.path.isabs(plc)
    assert os.path.isfile(netlist) and os.path.isfile(plc)


def test_required_api_all_exist():
  import importlib
  for modname, attrs in app_lib.REQUIRED_API.items():
    mod = importlib.import_module(modname)
    for attr in attrs:
      assert hasattr(mod, attr), f'{modname}.{attr}'


def test_required_api_covers_all_module_uses():
  import ast
  with open(app_lib.__file__) as f:
    tree = ast.parse(f.read())
  aliases = {}
  for node in ast.walk(tree):
    if (isinstance(node, ast.ImportFrom) and
        node.module == 'neurosilicon'):
      for a in node.names:
        if '.' not in a.name:
          aliases[a.asname or a.name] = 'neurosilicon.' + a.name
  assert aliases
  missing = set()
  for node in ast.walk(tree):
    if (isinstance(node, ast.Attribute) and
        isinstance(node.value, ast.Name) and
        node.value.id in aliases):
      mod = aliases[node.value.id]
      if node.attr not in app_lib.REQUIRED_API.get(mod, []):
        missing.add(f'{mod}.{node.attr}')
  assert not missing, sorted(missing)


def test_save_load_gui_run_roundtrip():
  import os
  import shutil
  run_id = 'test_tmp_roundtrip'
  run_dir = os.path.join('logs', 'gui_runs', 'runs', run_id)
  shutil.rmtree(run_dir, ignore_errors=True)
  try:
    rows, plc, cost, _ = app_lib.run_episodes(
        'circuit_training/environment/test_data/toy_macro_stdcell/'
        'netlist.pb.txt',
        'circuit_training/environment/test_data/toy_macro_stdcell/'
        'initial.plc',
        'first_legal', 1, 2, None, None, '', 0, 0)
    assert plc is not None
    fn, _ = app_lib.build_mask_fn('Center keep-out channel', 20, 10,
                                  200.0, 100.0)
    blocked = (fn(0, np.ones(200, dtype=np.int32)).reshape(10, 20)
               == 0) if fn else None
    run = {'choice': 'toy', 'run_id': run_id,
           'netlist': 'circuit_training/environment/test_data/'
                      'toy_macro_stdcell/netlist.pb.txt',
           'rows': rows, 'best_cost': cost, 'power': None,
           'advice': None, 'validated': None, 'blocked': blocked}
    saved = app_lib.save_gui_run(run_id, run, plc)
    assert os.path.isfile(os.path.join(saved, 'run.json'))
    assert os.path.isfile(os.path.join(saved, 'best.plc'))
    assert app_lib.patch_gui_run(run_id, advice='hi')
    assert any(m.get('run_id') == run_id
               for m in app_lib.list_gui_runs())
    run2, plc2 = app_lib.load_gui_run(saved)
    assert run2['rows'] == rows and run2['advice'] == 'hi'
    assert os.path.isabs(run2['netlist'])
    assert (run2['blocked'] == blocked).all()
    d1 = [(m['macro'], m['x'], m['y'])
          for m in app_lib.placement_details(plc)['macros']]
    d2 = [(m['macro'], m['x'], m['y'])
          for m in app_lib.placement_details(plc2)['macros']]
    assert d1 and d1 == d2
  finally:
    shutil.rmtree(run_dir, ignore_errors=True)
