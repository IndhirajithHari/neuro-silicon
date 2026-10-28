"""Placement-grid auto-selection, ported from AlphaChip.

Faithful port of circuit_training/grouping/grid_size_selection.py
(get_grid_suggestion / get_grid_choices / select_from_grid_choices)
with absl flags replaced by plain parameters (same defaults). Used
offline at conversion time: sweeps candidate columns/rows, keeps grids
where every hard macro places, scores waste/emptiness metrics, and
picks the smallest grid within tolerance of the best metric.
"""
import dataclasses
import math
from typing import Dict, List, Optional, Tuple

from neurosilicon.thirdparty.tilos.Plc_client import placement_util_os as placement_util


def get_span_and_extra_piece(seg_width, gcell_width):
  span = -1
  extra_space = -1.0
  i = 0
  while extra_space < 0.0:
    span = i * 2 + 1
    extra_space = (span * gcell_width - seg_width) / 2.0
    i += 1
  extra_piece = gcell_width - extra_space
  return span, extra_piece


def get_waste_ratio(segment_widths, gcell_width):
  tot_width = sum(segment_widths)
  assert tot_width > 0.0
  index = 0
  prev_extra_piece = 0.0
  for segw in segment_widths:
    spanned, extra_piece = get_span_and_extra_piece(segw, gcell_width)
    if extra_piece + prev_extra_piece <= gcell_width:
      index -= 1
    index += spanned
    prev_extra_piece = extra_piece
  span = (index + 1) * gcell_width
  return (span - tot_width) / span


def get_hard_macros(plc, include_fixed_macros):
  return [m for m in plc.get_macro_indices()
          if not plc.is_node_soft_macro(m)
          and (include_fixed_macros or not plc.is_node_fixed(m))]


def get_macro_widths_heights(plc, add_size, include_fixed_macros):
  widths, heights = [], []
  for m in get_hard_macros(plc, include_fixed_macros):
    w, h = plc.get_node_width_height(m)
    # TILOS core has no bloat API; AlphaChip default bloat is 0.
    bx = plc.get_macro_bloat_width() if hasattr(
        plc, 'get_macro_bloat_width') else 0.0
    by = plc.get_macro_bloat_height() if hasattr(
        plc, 'get_macro_bloat_height') else 0.0
    widths.append(w + bx + add_size)
    heights.append(h + by + add_size)
  return widths, heights


def get_empty_cells_ratio(plc):
  used_threshold = 1e-5
  cells = plc.get_grid_cells_density()
  return float(sum(d < used_threshold for d in cells)) / len(cells)


def try_placing(plc, hard_macros):
  for m in hard_macros:
    plc.unfix_node_coord(m)
  plc.unplace_all_nodes()
  for m in hard_macros:
    avail = [p for p, v in enumerate(plc.get_node_mask(m)) if v]
    if not avail:
      return False
    plc.place_node(m, avail[0])
  return True


@dataclasses.dataclass
class GridChoice:
  key_metric: float = 0.0
  empty_ratio: float = 0.0
  hor_waste: float = 0.0
  ver_waste: float = 0.0
  num_gcells: int = 0


def get_key_metric(d: GridChoice):
  return d.empty_ratio + (1.0 - d.hor_waste) + (1.0 - d.ver_waste)


def get_grid_choices(plc,
                     min_num=10,
                     max_num=128,
                     max_grid_size=128,
                     min_num_grid_cells=500,
                     max_num_grid_cells=4096,
                     max_aspect_ratio=1.5,
                     add_size=0.0,
                     include_fixed_macros=False,
                     sweep_step=1):
  """Sweeps (cols, rows); returns {(cols, rows): GridChoice} or None.

  sweep_step=1 is the exact AlphaChip sweep; larger steps sample the
  sweep on coarse grids (much faster, slightly different optima).
  """
  hard_macros = placement_util.get_ordered_node_indices(
      mode='descending_size_macro_first', plc=plc,
      exclude_fixed_nodes=not include_fixed_macros)
  hard_macros = [m for m in hard_macros
                 if not plc.is_node_soft_macro(m)][:len(
                     get_hard_macros(plc, include_fixed_macros))]
  if not hard_macros:
    return None
  choices: Dict[Tuple[int, int], GridChoice] = {}
  hor_waste, ver_waste = {}, {}
  macro_widths, macro_heights = get_macro_widths_heights(
      plc, add_size, include_fixed_macros)
  canvas_w, canvas_h = plc.get_canvas_width_height()
  for rows in range(min_num, max_num, sweep_step):
    for cols in range(min_num, max_num, sweep_step):
      if rows > max_grid_size or cols > max_grid_size:
        continue
      if not (min_num_grid_cells <= cols * rows <= max_num_grid_cells):
        continue
      gw, gh = canvas_w / cols, canvas_h / rows
      if max(gw, gh) / min(gw, gh) > max_aspect_ratio:
        continue
      plc.set_placement_grid(cols, rows)
      if not try_placing(plc, hard_macros):
        continue
      if cols not in hor_waste:
        hor_waste[cols] = get_waste_ratio(macro_widths, gw)
      if rows not in ver_waste:
        ver_waste[rows] = get_waste_ratio(macro_heights, gh)
      d = GridChoice(empty_ratio=get_empty_cells_ratio(plc),
                     hor_waste=hor_waste[cols],
                     ver_waste=ver_waste[rows],
                     num_gcells=cols * rows)
      d.key_metric = get_key_metric(d)
      choices[(cols, rows)] = d
  return choices


def select_from_grid_choices(choices, tolerance=0.05):
  """Picks smallest grid within tolerance of the best key metric."""
  ranked = sorted(choices.items(), key=lambda kv: kv[1].key_metric,
                  reverse=True)
  floor = ranked[0][1].key_metric * (1.0 - tolerance)
  qualified = [(k, v) for k, v in ranked if v.key_metric >= floor]
  return max(qualified, key=lambda kv: kv[1].key_metric / kv[1].num_gcells)[0]


def get_grid_suggestion(plc, tolerance=0.05, **sweep_kwargs):
  """Returns (cols, rows) suggestion, restoring macro positions after."""
  orig_xy = placement_util.get_node_xy_coordinates(plc)
  orig_ori = placement_util.get_macro_orientations(plc)
  # TILOS orientation restore only handles hard macros (soft-macro pin
  # maps are absent); filter rather than crashing on group restores.
  hard = set(get_hard_macros(plc, True))
  orig_ori = {m: o for m, o in orig_ori.items() if m in hard}
  try:
    choices = get_grid_choices(plc, **sweep_kwargs)
  finally:
    placement_util.restore_node_xy_coordinates(plc, orig_xy)
    placement_util.restore_macro_orientations(plc, orig_ori)
  if not choices:
    return None, None
  return select_from_grid_choices(choices, tolerance)


def suggest_for_proto(netlist_path, canvas_w, canvas_h,
                      **sweep_kwargs):
  """Builds a throwaway PLC and suggests a grid. Returns (cols, rows)."""
  from neurosilicon import setup as setup_lib
  plc = setup_lib.create_plc(netlist_path, canvas=(canvas_w, canvas_h))
  cols, rows = get_grid_suggestion(plc, **sweep_kwargs)
  return cols or 32, rows or 32
