"""Build a configured open-source PlacementCost for a netlist.

This is NeuroSilicon's tolerant replacement for the TF-stack setup path:
canvas/grid/tech parameters are resolved from explicit arguments first, then
from `.plc`/netlist file comments, then from defaults. Restoring node
positions from an initial `.plc` is validated *before* loading so a
degenerate file (e.g. header-only) warns instead of aborting.
"""
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from neurosilicon.thirdparty.tilos.Plc_client import plc_client_os
from neurosilicon.thirdparty.tilos.Plc_client import placement_util_os as placement_util

logger = logging.getLogger(__name__)

DEFAULT_ROUTES_PER_MICRON = (70.33, 74.51)
DEFAULT_MACRO_ROUTING_ALLOC = (51.79, 51.79)

# AlphaChip congestion-grid sizing (placement_util.py), recorded for the
# day the core grows set_congestion_grid. Currently the TILOS core sizes
# congestion on the placement grid, so this is documentation, not code:
#   size = 2 * routes_per_congestion_grid / (routes_h + routes_v)
#   cols = max(1, int(canvas_w / size)); rows = max(1, int(canvas_h / size))
DEFAULT_ROUTES_PER_CONGESTION_GRID = 1000


def parse_plc_header(plc_path: str) -> Dict[str, object]:
  """Extracts canvas/grid/tech metadata from `.plc` comment lines."""
  header: Dict[str, object] = {}
  with open(plc_path, 'r') as f:
    for line in f:
      if not line.startswith('#'):
        break
      m = re.search(r'Width : ([\d\.]+)  Height : ([\d\.]+)', line)
      if m:
        header['width'] = float(m.group(1))
        header['height'] = float(m.group(2))
      m = re.search(r'Columns : (\d+)  Rows : (\d+)', line)
      if m:
        header['columns'] = int(m.group(1))
        header['rows'] = int(m.group(2))
      m = re.search(
          r'Routes per micron, hor : ([\d\.]+)  ver : ([\d\.]+)', line)
      if m:
        header['routes_h'] = float(m.group(1))
        header['routes_v'] = float(m.group(2))
      m = re.search(
          r'Routes used by macros, hor : ([\d\.]+)  ver : ([\d\.]+)', line)
      if m:
        header['macro_alloc_h'] = float(m.group(1))
        header['macro_alloc_v'] = float(m.group(2))
      m = re.search(r'Smoothing factor : (\d+)', line)
      if m:
        header['smooth'] = int(m.group(1))
      m = re.search(r'Overlap threshold : ([\d\.e\-]+)', line)
      if m:
        header['overlap'] = float(m.group(1))
      m = re.search(r'Block : (\S+)', line)
      if m:
        header['block'] = m.group(1)
  return header


def count_plc_nodes(plc_path: str) -> List[int]:
  """Returns the node indices listed in the `.plc` body (non-comment lines)."""
  indices = []
  with open(plc_path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      try:
        indices.append(int(line.split()[0]))
      except (ValueError, IndexError):
        continue
  return indices


def default_grid(canvas_w: float, canvas_h: float,
                 target_cells: int = 32) -> Tuple[int, int]:
  """Aspect-preserving fallback grid when no grid metadata exists."""
  if canvas_w >= canvas_h:
    cols = target_cells
    rows = max(1, round(target_cells * canvas_h / max(canvas_w, 1e-9)))
  else:
    rows = target_cells
    cols = max(1, round(target_cells * canvas_w / max(canvas_h, 1e-9)))
  return cols, rows


def create_plc(netlist_file: str,
               init_placement: Optional[str] = None,
               canvas: Optional[Tuple[float, float]] = None,
               grid: Optional[Tuple[int, int]] = None,
               routes_per_micron: Optional[Tuple[float, float]] = None,
               macro_routing_alloc: Optional[Tuple[float, float]] = None,
               smooth: Optional[int] = None,
               overlap: Optional[float] = None,
               spacing: Tuple[float, float] = (0.1, 0.1),
               block_name: Optional[str] = None,
               restore_positions: bool = True,
               fix_ports: bool = True,
               boundary_check: bool = True):
  """Creates and configures a PlacementCost for `netlist_file`.

  Note: AlphaChip's create_placement_cost defaults boundary_check to
  False, but off-canvas macros break our DEF/OpenROAD flow, so strict
  canvas containment stays the default here (pass False for exact
  AlphaChip legality).
  """
  plc = plc_client_os.PlacementCost(netlist_file, spacing[0], spacing[1])
  plc.set_canvas_boundary_check(boundary_check)

  header: Dict[str, object] = {}
  if init_placement and os.path.isfile(init_placement):
    header = parse_plc_header(init_placement)
  net_w, net_h, net_cols, net_rows = (
      placement_util.extract_parameters_from_comments(netlist_file))

  width = canvas[0] if canvas else header.get('width', net_w)
  height = canvas[1] if canvas else header.get('height', net_h)
  cols = grid[0] if grid else header.get('columns', net_cols)
  rows = grid[1] if grid else header.get('rows', net_rows)
  if width and height and (not cols or not rows):
    cols, rows = default_grid(width, height)
  if width and height and cols and rows:
    plc.set_canvas_size(width, height)
    plc.set_placement_grid(cols, rows)
  else:
    logger.warning('Canvas/grid unknown; using PlacementCost defaults.')

  routes = routes_per_micron or (
      (header['routes_h'], header['routes_v'])
      if 'routes_h' in header else DEFAULT_ROUTES_PER_MICRON)
  plc.set_routes_per_micron(routes[0], routes[1])
  alloc = macro_routing_alloc or (
      (header['macro_alloc_h'], header['macro_alloc_v'])
      if 'macro_alloc_h' in header else DEFAULT_MACRO_ROUTING_ALLOC)
  plc.set_macro_routing_allocation(alloc[0], alloc[1])
  plc.set_congestion_smooth_range(
      smooth if smooth is not None else header.get('smooth', 5))
  plc.set_overlap_threshold(
      overlap if overlap is not None else header.get('overlap', 4e-3))
  plc.set_project_name('neurosilicon')
  plc.set_block_name(
      block_name or header.get('block')
      or os.path.basename(os.path.dirname(netlist_file.rstrip('/')))
      or 'unset_block')
  plc.make_soft_macros_square()

  blockages = (
      placement_util.get_blockages_from_comments(
          [p for p in (netlist_file, init_placement) if p]) or [])
  for blockage in blockages:
    plc.create_blockage(*blockage)

  if restore_positions and init_placement and os.path.isfile(init_placement):
    expected = sorted(plc.port_indices + plc.hard_macro_indices +
                      plc.soft_macro_indices)
    found = sorted(count_plc_nodes(init_placement))
    if found and found == expected:
      plc.restore_placement(init_placement)
    else:
      logger.warning(
          'Skipping restore of %s: covers %d nodes, netlist has %d '
          '(keeping netlist coordinates).',
          init_placement, len(found), len(expected))

  if fix_ports:
    placement_util.fix_port_coordinates(plc)
  return plc
