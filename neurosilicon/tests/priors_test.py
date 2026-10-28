"""Tests for priors, chat parsing, and extra env costs."""
import numpy as np

from neurosilicon import env as env_lib
from neurosilicon import priors as priors_lib
from neurosilicon import setup as setup_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_keepout_blocks_center_cells():
  fn = priors_lib.make_keepout_fn(20, 10, 200.0, 100.0,
                                  [(80.0, 0.0, 120.0, 100.0)])
  mask = np.ones(200, dtype=np.int32)
  out = fn(0, mask)
  assert int(np.sum(out)) < 200
  # A center cell (row 5, col 10) is blocked; a corner cell is open.
  assert out[5 * 20 + 10] == 0
  assert out[0] == 1


def test_chat_center_channel():
  mode, rects, explanation = priors_lib.chat_to_spec(
      'Keep a 20% routing channel open in the center', 200.0, 100.0)
  assert mode == 'block'
  assert len(rects) == 1
  x0, _, x1, _ = rects[0]
  assert abs((x1 - x0) - 40.0) < 1e-6
  assert 'center' in explanation


def test_chat_edge_allow():
  fn, explanation = priors_lib.chat_to_mask_fn(
      'Place all SRAM macros along the left and right edges', 20, 10,
      200.0, 100.0)
  assert fn is not None
  mask = np.ones(200, dtype=np.int32)
  out = fn(0, mask)
  assert int(np.sum(out)) < 200
  assert out[5 * 20 + 0] == 1
  assert out[5 * 20 + 10] == 0


def test_thermal_penalty_adjacent_vs_far():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  power = {'ML': 1.0, 'MR': 1.0}
  plc.place_node(0, 5 * 20 + 5)
  plc.place_node(1, 5 * 20 + 7)
  near = priors_lib.thermal_overlap_penalty(plc, power)
  plc.place_node(1, 5 * 20 + 18)
  far = priors_lib.thermal_overlap_penalty(plc, power)
  assert near > 0.0
  assert far == 0.0


def test_broadcast_penalty_spread_vs_colo():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  plc.place_node(0, 5 * 20 + 5)
  plc.place_node(1, 5 * 20 + 5)
  colo = priors_lib.broadcast_penalty(plc, fanout_threshold=1)
  plc.place_node(1, 5 * 20 + 18)
  spread = priors_lib.broadcast_penalty(plc, fanout_threshold=1)
  assert colo > 0.0
  assert spread > colo
  # Deterministic, and empty above the max toy fanout (6).
  assert priors_lib.broadcast_penalty(plc, fanout_threshold=1) == spread
  assert priors_lib.broadcast_penalty(plc, fanout_threshold=7) == 0.0
  assert priors_lib.broadcast_penalty(plc, fanout_threshold=10 ** 9) == 0.0


def test_flat_net_spans_and_penalty(tmp_path):
  import json
  import os

  from neurosilicon import synth_to_proto as converter_lib
  stats = converter_lib.Converter(
      'neurosilicon/tests/testdata/synth_small.json',
      'neurosilicon/tests/testdata/mini.lef', 'mini').convert(str(tmp_path))
  spans = converter_lib.Converter(
      'neurosilicon/tests/testdata/synth_small.json',
      'neurosilicon/tests/testdata/mini.lef',
      'mini').flat_net_spans()
  assert len(spans) == 5
  assert all(set(s) == {'fanout', 'nodes'} for s in spans)
  with open(os.path.join(str(tmp_path), 'clusters.json')) as f:
    clusters = json.load(f)
  plc = setup_lib.create_plc(stats['netlist'],
                             init_placement=stats['plc'])
  v1 = priors_lib.flat_broadcast_penalty(plc, spans, clusters,
                                         fanout_threshold=1)
  assert v1 >= 0.0
  assert priors_lib.flat_broadcast_penalty(plc, spans, clusters,
                                           fanout_threshold=1) == v1
  assert priors_lib.flat_broadcast_penalty(plc, spans, clusters,
                                           fanout_threshold=2) == 0.0


def test_thermal_corner_scaling():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  base = {'ML': 0.6, 'MR': 0.4}
  ref = priors_lib.thermal_corner(plc, base)
  assert abs(ref['total'] - 1.1) < 1e-6
  assert 0.0 < ref['balance'] <= 1.0
  assert ref['rows'][0]['hotspot'] == 100.0
  fast = priors_lib.thermal_corner(plc, base, clock_mhz=200.0)
  assert abs(fast['total'] - 2.1) < 1e-6
  idle = priors_lib.thermal_corner(plc, base, activity=0.0)
  assert abs(idle['total'] - 0.1) < 1e-6


def test_top_congestion_properties():
  import numpy as np
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  t10 = priors_lib.top_congestion(plc)
  assert t10 >= 0.0
  assert priors_lib.top_congestion(plc) == t10
  cols, rows = plc.get_grid_num_columns_rows()
  v = np.asarray(plc.V_routing_cong, dtype=float).reshape(rows, cols)
  h = np.asarray(plc.H_routing_cong, dtype=float).reshape(rows, cols)
  assert t10 >= float(np.mean(v + h)) - 1e-9
  assert priors_lib.top_congestion(plc, frac=1.0) == (
      priors_lib.top_congestion(plc, frac=1.0))


def test_env_extra_cost_in_info():
  env = env_lib.ChipPlacementEnv(
      TOY_NETLIST,
      init_placement=TOY_PLC,
      fd_steps=(2, 2, 2), init_fd=False,
      extra_costs=[('thermal', 0.5,
                    lambda plc: priors_lib.thermal_overlap_penalty(
                        plc, {'ML': 1.0, 'MR': 1.0}))])
  obs, _ = env.reset()
  done = False
  while not done:
    legal = np.flatnonzero(obs['mask'])
    obs, _, terminated, truncated, info = env.step(int(legal[0]))
    done = terminated or truncated
  assert 'extra_thermal' in info
  assert info['proxy_cost'] >= (
      info['wirelength'] + 0.5 * info['congestion'] + info['density'])
