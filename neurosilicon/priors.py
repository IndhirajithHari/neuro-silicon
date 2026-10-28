"""Constraint priors: keep-outs, edge bias, chat parsing, thermal cost.

Keep-outs and edge bands compile to `extra_mask_fn` callables for
`ChipPlacementEnv` (hard boolean masks, as in the "chat-to-floorplan"
spec). The thermal penalty plugs into the env's `extra_costs` hook.
"""
import math
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1 in microns


def cells_in_rects(cols: int, rows: int, canvas_w: float, canvas_h: float,
                   rects: List[Rect]) -> np.ndarray:
  """Boolean [rows*cols] vector: True where the cell center is in `rects`."""
  cw, ch = canvas_w / cols, canvas_h / rows
  inside = np.zeros(rows * cols, dtype=bool)
  for r in range(rows):
    for c in range(cols):
      x, y = (c + 0.5) * cw, (r + 0.5) * ch
      for (x0, y0, x1, y1) in rects:
        if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
          inside[r * cols + c] = True
          break
  return inside


def make_keepout_fn(cols: int, rows: int, canvas_w: float, canvas_h: float,
                    rects: List[Rect]):
  """Returns an env `extra_mask_fn` blocking `rects` for every macro."""
  blocked = cells_in_rects(cols, rows, canvas_w, canvas_h, rects)

  def fn(node_idx: int, mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).reshape(-1).copy()
    m[blocked[:m.shape[0]]] = 0
    return m

  return fn


def make_allow_fn(cols: int, rows: int, canvas_w: float, canvas_h: float,
                  rects: List[Rect]):
  """Returns an env `extra_mask_fn` restricting macros to `rects`."""
  allowed = cells_in_rects(cols, rows, canvas_w, canvas_h, rects)

  def fn(node_idx: int, mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).reshape(-1).copy()
    m[~allowed[:m.shape[0]]] = 0
    return m

  return fn


def make_perimeter_fn(cols: int, rows: int, canvas_w: float, canvas_h: float,
                      margin_frac: float = 0.25):
  """Restricts macros to a perimeter band (warm-start style prior)."""
  mx, my = canvas_w * margin_frac, canvas_h * margin_frac
  bands = [(0, 0, canvas_w, my), (0, canvas_h - my, canvas_w, canvas_h),
           (0, 0, mx, canvas_h), (canvas_w - mx, 0, canvas_w, canvas_h)]
  return make_allow_fn(cols, rows, canvas_w, canvas_h, bands)


def _percent(text: str, default: float) -> float:
  m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
  return float(m.group(1)) / 100.0 if m else default


def chat_to_spec(text: str, canvas_w: float,
                 canvas_h: float) -> Tuple[str, List[Rect], str]:
  """Parses a natural-language constraint into (mode, rects, explanation).

  mode is 'block' (keep-out) or 'allow' (restrict to). This is a
  rule-based parser; see LLMPriorBackend for the LLM upgrade path.
  """
  t = text.lower()
  if any(k in t for k in ('center', 'middle', 'central')) and any(
      k in t for k in ('open', 'channel', 'free', 'clear', 'keep out',
                       'keep-out', 'reserve')):
    frac = _percent(t, 0.20)
    cx0 = canvas_w * (0.5 - frac / 2)
    cx1 = canvas_w * (0.5 + frac / 2)
    return ('block', [(cx0, 0.0, cx1, canvas_h)],
            f'Keep-out: center {frac * 100:.0f}% vertical channel.')
  edges = []
  if 'left' in t:
    edges.append('left')
  if 'right' in t:
    edges.append('right')
  if 'top' in t:
    edges.append('top')
  if 'bottom' in t:
    edges.append('bottom')
  if edges and any(k in t for k in ('edge', 'perimeter', 'side', 'along')):
    frac = _percent(t, 0.25)
    rects = []
    if 'left' in edges:
      rects.append((0, 0, canvas_w * frac, canvas_h))
    if 'right' in edges:
      rects.append((canvas_w * (1 - frac), 0, canvas_w, canvas_h))
    if 'bottom' in edges:
      rects.append((0, 0, canvas_w, canvas_h * frac))
    if 'top' in edges:
      rects.append((0, canvas_h * (1 - frac), canvas_w, canvas_h))
    if any(k in t for k in ('place', 'put', 'keep', 'along')):
      return ('allow', rects, f'Restrict to {"/".join(edges)} edge band(s).')
    return ('block', rects, f'Keep-out: {"/".join(edges)} edge band(s).')
  return ('block', [], 'No constraint recognized; no mask applied.')


def chat_to_mask_fn(text: str, cols: int, rows: int, canvas_w: float,
                    canvas_h: float):
  """Compiles chat text to an env mask fn. Returns (fn_or_None, explanation)."""
  return chat_spec_to_mask_fn(
      *chat_to_spec(text, canvas_w, canvas_h), cols, rows, canvas_w,
      canvas_h)


def chat_spec_to_mask_fn(mode: str, rects: List[Rect], explanation: str,
                         cols: int, rows: int, canvas_w: float,
                         canvas_h: float):
  """Compiles a parsed (mode, rects) spec to a mask fn.

  Shared by the rule-based parser and the LLM parser so both produce
  identical masks from identical specs.
  """
  if not rects or mode == 'none':
    return None, explanation
  if mode == 'allow':
    return make_allow_fn(cols, rows, canvas_w, canvas_h, rects), explanation
  return make_keepout_fn(cols, rows, canvas_w, canvas_h, rects), explanation


def thermal_overlap_penalty(plc, power: Dict[str, float],
                             radius_factor: float = 1.0) -> float:
  """Penalizes placing high-power macros adjacent to each other.

  Args:
    plc: PlacementCost with current positions.
    power: macro name -> power (watts or normalized). Macros absent from
      the dict are treated as zero power.
    radius_factor: scales the adjacency radius (mean macro half-perimeter).

  Returns:
    Normalized pair penalty >= 0 (0 when no hot pair is adjacent).
  """
  hots = []
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    p = power.get(plc.get_node_name(idx), 0.0)
    if p > 0:
      x, y = plc.get_node_location(idx)
      w, h = plc.get_node_width_height(idx)
      hots.append((p, x, y, (w + h) / 2.0))
  total = sum(p for p, _, _, _ in hots)
  if total <= 0 or len(hots) < 2:
    return 0.0
  penalty = 0.0
  for i in range(len(hots)):
    for j in range(i + 1, len(hots)):
      p1, x1, y1, r1 = hots[i]
      p2, x2, y2, r2 = hots[j]
      dist = math.hypot(x1 - x2, y1 - y2)
      if dist < (r1 + r2) * radius_factor:
        penalty += (p1 * p2) / (total * total)
  return penalty


def thermal_corner(plc, base_power: Dict[str, float],
                   clock_mhz: float = 100.0, voltage: float = 1.8,
                   activity: float = 0.2,
                   leakage_frac: float = 0.1) -> Dict:
  """Per-macro power at a clock/voltage/activity corner (relative units).

  Model (documented proxy, not signoff): dynamic power scales as
  C * V^2 * f * activity from the reference corner (100 MHz, 1.8 V,
  activity 0.2), where C is the normalized `base_power` share; leakage
  is `leakage_frac` of total base power spread by macro area.

  Returns {'rows': [{macro, power, density, hotspot}], 'total',
  'balance', 'peak_macro'} where density is power per unit area,
  hotspot is density normalized to 100 at the hottest macro, and
  balance is mean/max density (1.0 = perfectly uniform).
  """
  base_total = sum(max(v, 0.0) for v in base_power.values())
  hard = [i for i in plc.get_macro_indices()
          if not plc.is_node_soft_macro(i)]
  areas = {}
  for idx in hard:
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = max(w * h, 0.0)
  area_total = sum(areas.values())
  dyn_scale = ((voltage / 1.8) ** 2 * (clock_mhz / 100.0) *
               (activity / 0.2) if activity > 0 else 0.0)
  rows = []
  for name, area in areas.items():
    dyn = max(base_power.get(name, 0.0), 0.0) * dyn_scale
    leak = (leakage_frac * base_total * area / area_total
            if area_total > 0 else 0.0)
    total = dyn + leak
    rows.append({'macro': name, 'power': total,
                 'density': total / area if area > 0 else 0.0})
  peak = max([r['density'] for r in rows] + [0.0])
  for r in rows:
    r['hotspot'] = 100.0 * r['density'] / peak if peak > 0 else 0.0
  mean_d = sum(r['density'] for r in rows) / max(len(rows), 1)
  peak_row = max(rows, key=lambda r: r['density'], default=None)
  return {'rows': sorted(rows, key=lambda r: -r['density']),
          'total': sum(r['power'] for r in rows),
          'balance': mean_d / peak if peak > 0 else 1.0,
          'peak_macro': peak_row['macro'] if peak_row else ''}


def top_congestion(plc, frac: float = 0.10) -> float:
  """AlphaChip-style congestion: mean over the top-frac grid cells.

  Averages the combined V+H routing-congestion map over the most
  congested `frac` fraction of cells (AlphaChip uses 10%). Reported
  alongside the optimizer's own congestion number; lower is better.
  """
  import numpy as np
  plc.get_congestion_cost()  # populate V/H routing maps
  cols, rows = plc.get_grid_num_columns_rows()
  v = np.asarray(plc.V_routing_cong, dtype=float).reshape(rows, cols)
  h = np.asarray(plc.H_routing_cong, dtype=float).reshape(rows, cols)
  flat = np.sort((v + h).ravel())[::-1]
  k = max(int(len(flat) * frac), 1)
  return float(np.mean(flat[:k]))


def broadcast_penalty(plc, fanout_threshold: int = 16) -> float:
  """Total span load of high-fanout (broadcast/clock-like) nets.

  For each net with >= `fanout_threshold` sinks, adds
  (parent-node HPWL / canvas) * log2(fanout). Purely structural: no
  net-name matching — clocks and reset trees always exceed the threshold
  while ordinary signal nets do not. Pin spans are approximated by
  parent-node locations via public accessors only (the private pin
  position helpers exit the process on unexpected input).

  Within-design comparator (lower is better); 0 when no net qualifies.
  Plugs into the env `extra_costs` hook as ('broadcast', weight, fn).
  """
  canvas = max(plc.get_canvas_width_height())
  if canvas <= 0:
    return 0.0
  mods = plc.modules_w_pins
  total = 0.0
  for driver, sinks in plc.nets.items():
    try:
      fanout = len(sinks)
    except TypeError:
      continue
    if fanout < fanout_threshold:
      continue
    xs, ys = [], []
    try:
      pins = [driver] + list(sinks)
    except TypeError:
      continue
    for pin_name in pins:
      idx = plc.mod_name_to_indices.get(pin_name)
      if idx is None or idx < 0 or idx >= len(mods):
        continue
      ref = plc.get_ref_node_id(idx)
      loc_idx = ref if ref != -1 else idx
      if loc_idx < 0 or loc_idx >= len(mods):
        continue
      if mods[loc_idx].get_type() not in ('MACRO', 'STDCELL', 'PORT'):
        continue
      x, y = plc.get_node_location(loc_idx)
      xs.append(x)
      ys.append(y)
    if len(xs) < 2:
      continue
    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
    total += (hpwl / canvas) * math.log2(fanout)
  return total


def flat_broadcast_penalty(plc, net_spans, clusters,
                           fanout_threshold: int = 64,
                           fanout_power: float = 1.0) -> float:
  """Hierarchical broadcast load: flat fanout x cluster span.

  Args:
    plc: PlacementCost with current positions.
    net_spans: Converter.flat_net_spans() output for the same netlist.
    clusters: {stdcell key: cluster name} (clusters.json from conversion).
    fanout_threshold: only nets with flat fanout >= this count.
    fanout_power: fanout weight exponent (1.0 = linear sink load).

  For each qualifying net, adds (cluster-span HPWL / canvas) *
  fanout**fanout_power, where the span covers the plc nodes (clusters,
  hard macros, ports) the net's flat endpoints map to. Unlike
  broadcast_penalty (cluster-level fanout), this sees true flat fanout,
  so flop-RAM clocks and wide cpu<->RAM buses dominate as they do in
  the router. Within-design comparator (lower is better).
  """
  canvas = max(plc.get_canvas_width_height())
  if canvas <= 0:
    return 0.0
  mods = plc.modules_w_pins
  total = 0.0
  for net in net_spans:
    fanout = net['fanout']
    if fanout < fanout_threshold:
      continue
    names = set()
    for key in net['nodes']:
      if key[0] == 'stdcell':
        g = clusters.get(key[1])
        if g is not None:
          names.add(g)
      elif key[0] == 'pin':
        names.add(key[1])
      elif key[0] == 'port':
        names.add(key[1])
    xs, ys = [], []
    for name in names:
      idx = plc.mod_name_to_indices.get(name)
      if idx is None or idx < 0 or idx >= len(mods):
        continue
      if mods[idx].get_type() not in ('MACRO', 'STDCELL', 'PORT'):
        continue
      x, y = plc.get_node_location(idx)
      xs.append(x)
      ys.append(y)
    if len(xs) < 2:
      continue
    span = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / canvas
    total += span * (fanout ** fanout_power)
  return total


class LLMPriorBackend:
  """Hook for LLM-based constraint parsing (Llama/Muse upgrade path).

  Configure by subclassing and implementing `parse`, or point
  `NEUROSILICON_LLM_URL` at an OpenAI-compatible chat endpoint. Without
  a backend, `parse` falls back to the rule-based `chat_to_spec`.
  """

  def __init__(self, endpoint: Optional[str] = None):
    import os
    self.endpoint = endpoint or os.environ.get('NEUROSILICON_LLM_URL', '')

  def parse(self, text: str, canvas_w: float,
            canvas_h: float) -> Tuple[str, List[Rect], str]:
    if not self.endpoint:
      return chat_to_spec(text, canvas_w, canvas_h)
    raise NotImplementedError(
        'LLM backend calls are not implemented in this snapshot; '
        'subclass LLMPriorBackend.parse or unset NEUROSILICON_LLM_URL.')
