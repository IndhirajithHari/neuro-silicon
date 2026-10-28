"""Focused tests for the vendored open-source PlacementCost core.

Covers: parsing of Google's test netlists (including raw stdcells),
node counts, and sane proxy-cost ranges.
"""
from neurosilicon.thirdparty.tilos.Plc_client import plc_client_os

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
ARIANE_NETLIST = (
    'circuit_training/environment/test_data/ariane/netlist.pb.txt')


def test_toy_parses_with_stdcells():
  plc = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  assert plc.hard_macro_cnt == 2
  # Raw stdcells are kept as (small) soft macros.
  assert plc.soft_macros_cnt == 12
  assert plc.ports_cnt == 12


def test_toy_costs_sane():
  plc = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  plc.set_canvas_size(200.0, 100.0)
  plc.set_placement_grid(20, 10)
  plc.set_routes_per_micron(10.0, 10.0)
  wl = plc.get_wirelength()
  cost = plc.get_cost()
  assert wl > 0.0
  assert 0.0 <= cost <= 1.0
  assert plc.get_congestion_cost() >= 0.0
  assert plc.get_density_cost() >= 0.0


def test_congestion_uses_top_10_percent():
  plc = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  plc.set_canvas_size(200.0, 100.0)
  plc.set_placement_grid(20, 10)
  plc.set_routes_per_micron(10.0, 10.0)
  got = plc.get_congestion_cost()
  expect = plc.abu(plc.V_routing_cong + plc.H_routing_cong, 0.10)
  assert got == expect


def test_blockage_blocks_mask_and_seeds_density():
  plc = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  plc.set_canvas_size(200.0, 100.0)
  plc.set_placement_grid(20, 10)
  macro = plc.hard_macro_indices[0]
  before = sum(plc.get_node_mask(macro))
  # Block the whole left half of the canvas.
  plc.create_blockage(0.0, 0.0, 100.0, 100.0, 1.0)
  after = sum(plc.get_node_mask(macro))
  assert 0 < after < before
  d0 = plc.get_density_cost()
  plc.FLAG_UPDATE_DENSITY = True
  assert plc.get_density_cost() >= d0
  # Rate-0 blockage still blocks hard-macro masks (density-only relief).
  plc2 = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  plc2.set_canvas_size(200.0, 100.0)
  plc2.set_placement_grid(20, 10)
  plc2.create_blockage(0.0, 0.0, 100.0, 100.0, 0.0)
  assert sum(plc2.get_node_mask(macro)) < before


def test_overlap_threshold_allows_slivers():
  plc = plc_client_os.PlacementCost(netlist_file=TOY_NETLIST)
  plc.set_canvas_size(200.0, 100.0)
  plc.set_placement_grid(20, 10)
  plc.set_overlap_threshold(0.0)
  m0, m1 = plc.hard_macro_indices[:2]
  plc.place_node(m0, 0)
  strict = sum(plc.get_node_mask(m1))
  plc.set_overlap_threshold(4e-3)
  plc.FLAG_UPDATE_CONGESTION = True
  relaxed = sum(plc.get_node_mask(m1))
  assert relaxed >= strict


def test_ariane_counts_match_plc_header():
  plc = plc_client_os.PlacementCost(netlist_file=ARIANE_NETLIST)
  assert plc.hard_macro_cnt == 133
  assert plc.soft_macros_cnt == 799
  assert plc.ports_cnt == 1231
  assert len(plc.port_indices + plc.hard_macro_indices +
             plc.soft_macro_indices) == 2163
