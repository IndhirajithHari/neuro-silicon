"""Tests for the EDA & visual-polish upgrades (Qt-free, CI-safe).

Hierarchy colors, pin-aware HPWL + orientation flips, SA polish with
pinning, technology-node presets, OpenROAD Tcl export, and the
3-agent design review. Desktop wiring is checked by exact labels.
"""
import os

from neurosilicon import assistant as assistant_lib
from neurosilicon import export_def as export_def_lib
from neurosilicon import liveplace as liveplace_lib
from neurosilicon import render as render_lib
from neurosilicon import settings as settings_lib
from neurosilicon import setup as setup_lib
from neurosilicon import tech as tech_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def _toy_plc():
  return setup_lib.create_plc(TOY_NETLIST, TOY_PLC)


def test_hierarchy_groups_and_snapshot_fields():
  assert liveplace_lib.hierarchy_group('a25_icache')[0] == 'icache'
  assert liveplace_lib.hierarchy_group('a25_dcache')[0] == 'dcache'
  assert liveplace_lib.hierarchy_group('i_frontend/i_icache/sram')[0] == 'icache'
  assert liveplace_lib.hierarchy_group('a25_alu')[0] == 'alu_exec'
  assert liveplace_lib.hierarchy_group('sram_256x128')[0] == 'sram_mem'
  assert liveplace_lib.hierarchy_group('ctrl')[0] == 'default'
  fills = {g[2] for g in liveplace_lib.HIERARCHY_GROUPS}
  assert len(fills) == len(liveplace_lib.HIERARCHY_GROUPS)
  frame = liveplace_lib.snapshot(_toy_plc(), 2, 2, '')
  for m in frame['macros']:
    assert {'group', 'fill', 'edge', 'orient', 'notch', 'pinned'} <= set(m)
    assert m['notch'] in ('N', 'S', 'E', 'W')
    assert m['orient'] in (0, 90, 180, 270)


def test_orientation_optimizer_is_readonly_and_improves():
  plc = _toy_plc()
  cost_before = plc.get_cost()
  best, before, after = liveplace_lib.optimize_orientations(plc)
  assert after <= before + 1e-9
  assert set(best.values()) <= {0, 90, 180, 270}
  # Read-only: the plc itself (label + pin-offset cost) is untouched.
  assert plc.get_cost() == cost_before
  assert liveplace_lib.net_hpwl(plc, best) == after


def test_sa_polish_improves_and_respects_pins():
  plc = _toy_plc()
  sa = liveplace_lib.simulated_annealing(plc, steps=200, seed=0)
  assert sa['hpwl_after'] <= sa['hpwl_before'] + 1e-9
  assert sa['steps'] == 200 and sa['accepted'] >= 0
  pinned = liveplace_lib.simulated_annealing(
      plc, steps=200, seed=0, pinned={'ML'})
  assert pinned['positions']['ML'] == plc.get_node_location(
      plc.get_node_index('ML'))


def test_write_pinned_placement_fixes_macro(tmp_path):
  out = str(tmp_path / 'pinned.plc')
  liveplace_lib.write_pinned_placement(
      TOY_NETLIST, TOY_PLC, {'ML': (60.0, 50.0)}, out)
  plc = setup_lib.create_plc(TOY_NETLIST, out)
  idx = plc.get_node_index('ML')
  assert plc.is_node_fixed(idx) is True
  x, y = plc.get_node_location(idx)
  assert abs(x - 60.0) < 1e-6 and abs(y - 50.0) < 1e-6
  # The pinned macro survives an env reset and placement stays legal.
  from neurosilicon import env as env_lib
  from neurosilicon import evaluate as evaluate_lib
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=out,
                                 fd_steps=(2, 2, 2), init_fd=False)
  _reward, info = evaluate_lib.rollout_first_legal(env)
  assert info['infeasible'] is False
  ex, ey = env.plc.get_node_location(env.plc.get_node_index('ML'))
  assert abs(ex - 60.0) < 1e-6 and abs(ey - 50.0) < 1e-6


def test_tech_presets_scale_power_and_congestion():
  assert set(tech_lib.TECH_NODES) == {'sky130', 'asap7', 'gf180mcu'}
  asap = tech_lib.preset('asap7')
  assert asap['voltage'] == 0.7 and asap['clock_mhz'] == 1000.0
  assert tech_lib.preset('gf180mcu')['voltage'] == 3.3
  assert tech_lib.preset('nope')['label'].startswith('sky130')
  plc = _toy_plc()
  hot = sum(tech_lib.corner_power(plc, 'asap7').values())
  ref = sum(tech_lib.corner_power(plc, 'sky130').values())
  assert hot > ref  # 1 GHz dynamic + 3.2x leakage dominates
  grid = [[1.0, 2.0]]
  scaled = tech_lib.scaled_congestion(grid, 'asap7')
  assert scaled[0][1] > grid[0][1]
  assert tech_lib.scaled_congestion(grid, 'sky130') == grid


def test_openroad_tcl_export():
  plc = _toy_plc()
  text = export_def_lib.openroad_tcl_text(plc, {'ML': 90})
  assert text.count('place_macro -macro_name') == 2
  assert 'place_macro -macro_name ML -location {' in text
  assert '-orientation R90' in text and '-orientation R0' in text
  path = export_def_lib.export_openroad_tcl(plc, '/tmp/ns_test.tcl')
  with open(path) as f:
    assert 'place_macro' in f.read()


def test_multi_agent_review_recommends_weights():
  ctx = {'design': 'toy (2 macros)', 'canvas': (200.0, 100.0),
         'results': [{'method': 'first_legal', 'episode': 0,
                      'proxy_cost': 1.05, 'wirelength': 0.21,
                      'congestion': 0.73, 'density': 0.47}],
         'best_proxy': 1.05, 'hottest': [('ML', 0.5), ('MR', 0.5)],
         'ppa': {'peak_macro': 'ML', 'fmax_mhz': 420.0,
                 'congestion': 0.73},
         'critical': {'fmax_mhz': 420.0, 'delay_ns': 2.3,
                      'path': ['PR2', 'MR']},
         'congestion_analysis': {'overall_congestion': 0.35,
                                 'peak_congestion': 0.62}}
  text, weights = assistant_lib.multi_agent_review(ctx)
  for who in ('Thermal Specialist', 'Routing & Timing Specialist',
              'Lead Physical Design Architect'):
    assert who in text
  assert set(weights) == {'wirelength_weight', 'congestion_weight',
                          'thermal_weight'}
  assert all(0.0 < v <= 2.0 for v in weights.values())
  assert 'ML' in text


def test_render_kpi_and_hierarchy_floorplan():
  plc = _toy_plc()
  fig = render_lib.kpi_bar_figure(
      [{'method': 'first_legal', 'episode': 0, 'proxy_cost': 1.0,
        'wirelength': 0.2, 'congestion': 0.7, 'density': 0.4}])
  assert fig.axes
  floor = render_lib.plot_floorplan(plc)
  assert floor.axes[0].get_legend() is not None  # hierarchy legend


def test_gui_source_labels():
  root = os.path.dirname(os.path.dirname(
      os.path.dirname(os.path.abspath(__file__))))
  with open(os.path.join(root, 'neurosilicon', 'desktop.py')) as f:
    desktop = f.read()
  with open(os.path.join(root, 'neurosilicon', 'app.py')) as f:
    console = f.read()
  for src in (desktop, console):
    assert 'Technology Node' in src
    assert 'Post-Placement Polish (200-step Simulated Annealing' in src
    assert 'Run 3-Agent Design Review' in src
  with open(os.path.join(root, 'neurosilicon', 'tech.py')) as f:
    tech_src = f.read()
  for needle in ('sky130 (130nm, 1.8V, 100 MHz)',
                 'asap7 (7nm FinFET, 0.7V, 1000 MHz',
                 'gf180mcu (180nm, 3.3V, 50 MHz)'):
    assert needle in tech_src, needle
  for needle in ('Download OpenROAD .tcl', 'Export Timelapse GIF',
                 'Show Critical Path', '🔒 Pinned'):
    assert needle in desktop, needle
  assert 'Download OpenROAD .tcl' in console
  assert settings_lib.defaults()  # settings schema untouched
