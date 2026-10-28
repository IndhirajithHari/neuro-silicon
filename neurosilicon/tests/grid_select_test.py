"""Tests for AlphaChip grid auto-selection port."""
from neurosilicon import grid_select as grid_select_lib
from neurosilicon import setup as setup_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_waste_ratio_bounds():
  w = grid_select_lib.get_waste_ratio([10.0, 20.0], 5.0)
  assert 0.0 <= w < 1.0
  # Exact fit wastes nothing extra beyond spanning.
  assert grid_select_lib.get_waste_ratio([10.0], 10.0) == 0.0


def test_select_prefers_fewer_cells_within_tolerance():
  mk = lambda km, n: grid_select_lib.GridChoice(  # noqa: E731
      key_metric=km, num_gcells=n)
  choices = {(64, 64): mk(2.0, 4096), (32, 32): mk(1.99, 1024),
             (16, 16): mk(1.0, 256)}
  assert grid_select_lib.select_from_grid_choices(
      choices, tolerance=0.05) == (32, 32)


def test_suggest_toy_returns_feasible_grid():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  m0 = plc.hard_macro_indices[0]
  x0, y0 = plc.get_node_location(m0)
  cols, rows = grid_select_lib.get_grid_suggestion(
      plc, min_num=10, max_num=24, min_num_grid_cells=100,
      max_num_grid_cells=600, sweep_step=4)
  assert cols is not None and rows is not None
  assert 10 <= cols <= 24 and 10 <= rows <= 24
  assert 100 <= cols * rows <= 600
  # Suggestion restores macro coordinates afterwards.
  assert plc.get_node_location(m0) == (x0, y0)
