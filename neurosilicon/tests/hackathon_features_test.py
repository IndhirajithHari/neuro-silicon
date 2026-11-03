"""Tests for the hackathon differentiators (Qt-free, CI-safe).

Covers the Auto-Optimize diagnosis/fallback chain, the PPA summary,
the Before/After compare picker, the critical-path/Fmax proxy, and
the RTL co-pilot templates. Desktop wiring is checked by exact UI
label strings so no Qt runtime is needed here.
"""
import os

from neurosilicon import app as app_lib
from neurosilicon import assistant as assistant_lib
from neurosilicon import liveplace as liveplace_lib
from neurosilicon import settings as settings_lib
from neurosilicon import setup as setup_lib

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


def _toy_plc():
  return setup_lib.create_plc(TOY_NETLIST, TOY_PLC)


def test_critical_path_is_simple_and_fmax_consistent():
  plc = _toy_plc()
  cp = liveplace_lib.critical_path(plc)
  assert cp['path'] and cp['delay_ns'] > 0
  assert len(set(cp['path'])) == len(cp['path'])  # simple path, no loops
  assert len(cp['points']) == len(cp['path'])
  assert len(cp['segments']) == len(cp['path']) - 1
  assert abs(cp['fmax_mhz'] - 1000.0 / cp['delay_ns']) < 0.01
  assert cp['wire_delay_ns'] >= 0 and cp['logic_delay_ns'] > 0
  assert {'ML', 'MR'} & set(cp['path'])  # reaches a hard macro


def test_ppa_metrics_and_summary_labels():
  plc = _toy_plc()
  ppa = liveplace_lib.ppa_metrics(plc)
  for key in ('proxy_cost', 'wirelength', 'congestion', 'density',
              'peak_power', 'peak_temp', 'peak_macro',
              'critical_delay_ns', 'fmax_mhz'):
    assert key in ppa
  assert ppa['peak_macro'] in ('ML', 'MR')
  same = liveplace_lib.format_ppa_summary(ppa, ppa)
  for label in ('PPA Improvement Summary', 'Δ Proxy', 'Δ Wirelength',
                'Δ Congestion', 'Δ Peak Temp', 'est. Fmax'):
    assert label in same
  better = dict(ppa, proxy_cost=ppa['proxy_cost'] * 0.9,
                wirelength=ppa['wirelength'] * 0.8,
                congestion=ppa['congestion'] * 0.7,
                peak_temp=ppa['peak_temp'] * 0.5)
  d = liveplace_lib.ppa_delta(ppa, better)
  assert d['proxy_pct'] < 0 and d['congestion_pct'] < 0
  assert '-30.0%' in liveplace_lib.format_ppa_summary(ppa, better)


def test_pick_compare_runs_needs_two_feasible():
  rows = [{'proxy_cost': 2.0}, {'proxy_cost': -1.0},
          {'proxy_cost': 1.5}, {'proxy_cost': 1.2}]
  base, best = liveplace_lib.pick_compare_runs(rows)
  assert base['proxy_cost'] == 2.0 and best['proxy_cost'] == 1.2
  base, best = liveplace_lib.pick_compare_runs([{'proxy_cost': 1.0}])
  assert base is not None and best is None
  assert liveplace_lib.pick_compare_runs([]) == (None, None)


def test_diagnose_offline_fallback_constraints():
  plc = _toy_plc()
  llm = settings_lib.defaults()['llm']
  assert llm['provider'] == 'off'
  diag = assistant_lib.diagnose_placement(plc, llm, 0.5, 0.0)
  assert diag['source'] == 'offline-fallback'
  assert 'spread_hot_macros' in diag['actions']
  assert 'keepout_center_channel' in diag['actions']
  assert diag['congestion_weight'] > 0.5  # increased
  assert diag['thermal_weight'] >= 0.5    # increased
  assert diag['rects']  # keep-out inside the 200x100 canvas
  x0, y0, x1, y1 = diag['rects'][0]
  assert 0 <= x0 < x1 <= 200.0 and 0 <= y0 < y1 <= 100.0


def test_diagnose_queries_muse_first_and_sanitizes(monkeypatch):
  plc = _toy_plc()
  seen = []

  def fake_chat(messages, settings):
    seen.append(settings.get('model'))
    return ('{"actions": ["spread_hot_macros"], '
            '"rects": [[-50, -50, 5000, 5000]], '
            '"congestion_weight": 0.1, "thermal_weight": 0.0, '
            '"explanation": "spread them"}')

  monkeypatch.setattr(assistant_lib.llm_lib, 'chat', fake_chat)
  llm = {'provider': 'api', 'api_key': 'k', 'base_url': 'https://x/v1',
         'model': 'muse-wonder-1.0', 'timeout': 5}
  diag = assistant_lib.diagnose_placement(plc, llm, 0.5, 0.0)
  assert seen[0] == 'muse-arcadia-1.0'  # Muse queried first
  assert diag['source'] == 'muse-arcadia-1.0'
  assert diag['congestion_weight'] >= 0.8  # never lowered
  x0, y0, x1, y1 = diag['rects'][0]
  assert 0 <= x0 < x1 <= 200.0 and (x1 - x0) * (y1 - y0) <= 0.6 * 200 * 100 + 1


def test_find_bottleneck_is_a_hard_macro():
  plc = _toy_plc()
  info = assistant_lib.find_bottleneck(plc)
  assert info['macro'] in ('ML', 'MR')  # never a soft cluster
  assert info['reason'] in ('routing congestion', 'thermal hotspot')


def test_rtl_templates_by_bottleneck_kind():
  llm = settings_lib.defaults()['llm']
  cache, used = assistant_lib.suggest_rtl_fix(
      'a25_dcache', 'module a25_dcache; endmodule', llm, 'a25_dcache.v')
  assert used is False and 'bank' in cache.lower()
  assert 'a25_dcache' in cache and '```verilog' in cache
  sram, _ = assistant_lib.suggest_rtl_fix('sram_256x128', '', llm)
  assert 'bank' in sram.lower() and 'sram_256x128' in sram
  bus, _ = assistant_lib.suggest_rtl_fix('wishbone_arbiter', '', llm)
  assert 'pipelin' in bus.lower()
  logic, _ = assistant_lib.suggest_rtl_fix('a25_alu', '', llm)
  assert 'clock gat' in logic.lower()
  for text in (cache, bus, logic):
    assert '\n-  ' in text and '\n+ ' in text.replace('\n+  ', '\n+ ')


def test_find_verilog_and_bottleneck_advice(tmp_path):
  root = tmp_path / 'amber25'
  root.mkdir()
  (root / 'a25_dcache.v').write_text(
      '// cache\nmodule a25_dcache #(parameter LINES=256);\n'
      'endmodule\n')
  (root / 'a25_icache.v').write_text('module a25_icache; endmodule\n')
  found = assistant_lib.find_verilog_for_macro('a25_dcache', [str(root)])
  assert found.endswith('a25_dcache.v')
  snippet = assistant_lib.read_verilog_snippet(found)
  assert 'module a25_dcache' in snippet
  llm = settings_lib.defaults()['llm']
  text, used, info = assistant_lib.bottleneck_rtl_advice(
      _toy_plc(), [str(root)], llm)
  assert used is False and 'RTL fix for bottleneck' in text
  assert info['macro'] in ('ML', 'MR')


def test_run_auto_optimize_toy_completes_loop():
  settings = settings_lib.load()
  result = app_lib.run_auto_optimize(
      TOY_NETLIST, TOY_PLC, settings_lib.defaults()['llm'],
      settings['engine'], episodes=1, fd=2, seed=0)
  assert 'error' not in result, result.get('error')
  assert result['base_rows'] and result['ai_rows']
  assert result['diagnosis']['source']
  for label in ('PPA Improvement Summary', 'Δ Proxy', 'Δ Peak Temp'):
    assert label in result['summary']


def test_render_critical_path_and_compare_figure():
  import matplotlib
  matplotlib.use('Agg')
  from neurosilicon import render as render_lib
  plc = _toy_plc()
  fig = render_lib.plot_floorplan(plc)
  cp = render_lib.draw_critical_path(fig.axes[0], plc)
  assert cp['path']
  both = render_lib.comparison_figure(plc, plc)
  assert len(both.axes) >= 2


def test_chat_suggestion_dropdown_asks_and_resets():
  """Suggestion dropdown + type-ahead completer (offscreen Qt).

  Skips where PySide6 is unavailable (CI installs no Qt); runs in the
  desktop dev environment. Picking a suggestion must post the prompt
  as a user message with an assistant reply and reset the dropdown to
  its placeholder; picking the placeholder must post nothing.
  """
  os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
  import pytest
  pytest.importorskip('PySide6.QtWidgets')
  from PySide6.QtWidgets import QApplication
  from neurosilicon import desktop as desktop_lib
  app = QApplication.instance() or QApplication([])
  win = desktop_lib.MainWindow()
  try:
    win.ai_provider.setCurrentText('off')  # deterministic offline reply
    assert win.chat_suggest.count() == 1 + len(win._chat_suggestions)
    assert win.chat_suggest.itemText(0).startswith('Suggestions')
    assert [win.chat_suggest.itemText(i)
            for i in range(1, win.chat_suggest.count())] == \
        win._chat_suggestions
    completer = win.chat_input.completer()
    assert completer is not None
    assert completer.model().stringList() == win._chat_suggestions
    before = win.chat_msgs.count()
    win._on_suggestion_picked(0)
    assert win.chat_msgs.count() == before
    win._on_suggestion_picked(1)
    assert win.chat_suggest.currentIndex() == 0
    assert win.chat_msgs.count() == before + 2  # You + Neuro AI
    tabs = [win.right_tabs.tabText(i)
            for i in range(win.right_tabs.count())]
    assert 'Neuro AI' in tabs and 'Assistant' not in tabs
    from PySide6.QtWidgets import QLabel
    ai_holder = win._bubble('Assistant', 'hello')
    ai_texts = [lb.text() for lb in ai_holder.findChildren(QLabel)]
    assert 'Neuro AI' in ai_texts
    assert any(':' in t and len(t) == 5 for t in ai_texts)  # HH:MM stamp
    you_holder = win._bubble('You', 'hi')
    you_texts = [lb.text() for lb in you_holder.findChildren(QLabel)]
    assert 'Neuro AI' not in you_texts and 'hi' in you_texts
  finally:
    win.close()
    app.processEvents()


def test_chat_import_routes_paths(tmp_path, monkeypatch):
  """Neuro AI chat import: typed/picked paths route to the Import tab.

  RTL projects open pre-filled for top/file confirmation (no synthesis
  started from chat); ready netlist pairs register immediately via
  do_import (stubbed here so the real registry is untouched); missing
  paths get a chat message, not a crash. Skips without PySide6.
  """
  os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
  import pytest
  pytest.importorskip('PySide6.QtWidgets')
  from PySide6.QtWidgets import QApplication
  from neurosilicon import desktop as desktop_lib
  app = QApplication.instance() or QApplication([])
  win = desktop_lib.MainWindow()
  try:
    win.ai_provider.setCurrentText('off')
    root = str(tmp_path / 'chip')
    _write(root, 'soc.v', 'module soc; leaf u(); endmodule\n')
    _write(root, 'leaf.v', 'module leaf; endmodule\n')
    before = win.chat_msgs.count()
    win._chat_import_path(root)
    assert win._disc['kind'] == 'rtl_project'
    assert win.imp_path.text() == root
    assert win.tabs.currentIndex() == 0  # Import tab
    assert win.chat_msgs.count() == before + 1  # guidance, no synth
    win.chat_input.setText(f'import {root}')
    before = win.chat_msgs.count()
    win._chat_send()
    assert win.chat_msgs.count() == before + 2  # You + Neuro AI
    # AI import opens a FOLDER picker (not a file picker) and
    # auto-fetches the RTL project files inside it.
    assert win.chat_import_btn.text() == 'Import project folder…'
    monkeypatch.setattr(
        desktop_lib.QFileDialog, 'getOpenFileName',
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('AI import must not open a file picker'))))
    monkeypatch.setattr(
        desktop_lib.QFileDialog, 'getExistingDirectory',
        staticmethod(lambda *a, **k: root))
    before = win.chat_msgs.count()
    win._chat_import_project()
    assert win._disc['kind'] == 'rtl_project'
    assert win.chat_msgs.count() == before + 1
    assert win.imp_top.currentText() == 'soc'
    assert set(win._checked_files()) == {'soc.v', 'leaf.v'}
    # A typed single RTL file also resolves to its folder project.
    win._disc = None
    win._chat_import_path(os.path.join(root, 'soc.v'))
    assert win._disc['kind'] == 'rtl_project'
    assert win.imp_path.text() == root
    assert set(win._checked_files()) == {'soc.v', 'leaf.v'}
    ready = str(tmp_path / 'ready')
    _write(ready, 'netlist.pb.txt', 'netlist {}\n')
    _write(ready, 'initial.plc',
           '# Width : 200  Height : 100\n# Columns : 8  Rows : 4\n')
    called = []
    monkeypatch.setattr(win, 'do_import', lambda: called.append(True))
    win._chat_import_path(ready)
    assert called == [True]
    before = win.chat_msgs.count()
    win._chat_import_path('/no/such/path')
    assert win.chat_msgs.count() == before + 1
  finally:
    win.close()
    app.processEvents()


def test_desktop_and_console_carry_exact_labels():
  root = os.path.dirname(os.path.dirname(
      os.path.dirname(os.path.abspath(__file__))))
  with open(os.path.join(root, 'neurosilicon', 'desktop.py')) as f:
    desktop = f.read()
  with open(os.path.join(root, 'neurosilicon', 'app.py')) as f:
    console = f.read()
  for src in (desktop, console):
    assert '⚡ Auto-Optimize (AI Loop)' in src
    assert 'Compare (Baseline vs. AI)' in src
    assert 'Suggest RTL Fix for Bottleneck' in src
    assert 'Show Critical Path' in src
  assert desktop.count('⚡ Auto-Optimize (AI Loop)') >= 2
  assert desktop.count('Suggest RTL Fix for Bottleneck') >= 2
  assert 'est. Fmax' in desktop and 'est. Fmax (MHz)' in console
  assert 'Run #1' in desktop and 'Best AI-Optimized' in desktop
