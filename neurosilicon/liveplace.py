"""AlphaChip-style sequential placement with live snapshots (Qt-free).

The engine mirrors AlphaChip's MDP: hard macros are placed one at a
time, largest first, onto the placement grid with a legality mask;
at the end the standard-cell clusters relax with the force-directed
placer and the episode is scored with the proxy cost (wirelength +
congestion + density). Each step yields a plain-data snapshot the
desktop canvas can paint in real time — no matplotlib round-trip.

Also provides the headless data behind the canvas views: net segments
for the wiring view and the scaled per-macro power behind the
thermal view.
"""
import numpy as np

from neurosilicon import env as env_lib


# -- RTL hierarchy color-coding -------------------------------------------
# (group, legend label, fill hex, edge hex). Prefix match on the RTL
# module hierarchy (names like i_frontend/i_icache/... or a25_dcache).
HIERARCHY_GROUPS = (
    ('icache', 'I-Cache', '#0e7490', '#22d3ee'),
    ('dcache', 'D-Cache', '#a21caf', '#e879f9'),
    ('cache', 'Cache', '#1d4ed8', '#60a5fa'),
    ('alu_exec', 'ALU / Exec', '#047857', '#34d399'),
    ('sram_mem', 'SRAM / Memory', '#b45309', '#fbbf24'),
    ('default', 'Logic / Other', '#78716c', '#a8a29e'),
)

_ORIENT_DEG = {'N': 0, 'FN': 0, 'E': 90, 'FE': 90,
               'S': 180, 'FS': 180, 'W': 270, 'FW': 270}
_EDGES = ('N', 'E', 'S', 'W')


def hierarchy_group(name):
  """(group, label, fill, edge) for a macro name's RTL hierarchy.

  Matches module-name tokens: icache -> cyan, dcache -> magenta,
  alu/exec -> emerald, sram/mem/ram -> amber, everything else slate.
  """
  low = (name or '').lower()
  if 'icache' in low:
    key = 'icache'
  elif 'dcache' in low:
    key = 'dcache'
  elif 'cache' in low:
    key = 'cache'
  elif any(k in low for k in ('alu', 'exec', 'execute', 'decode')):
    key = 'alu_exec'
  elif any(k in low for k in ('sram', 'mem', 'ram', 'regfile', 'rf_')):
    key = 'sram_mem'
  else:
    key = 'default'
  for group, label, fill, edge in HIERARCHY_GROUPS:
    if group == key:
      return group, label, fill, edge
  return HIERARCHY_GROUPS[-1]


def orientation_deg(plc, idx):
  """Macro orientation in degrees (0/90/180/270) from the plc label."""
  try:
    return _ORIENT_DEG.get(str(plc.get_macro_orientation(idx)), 0)
  except Exception:
    return 0


def pin_edge(plc, name):
  """Pin-facing edge ('N'/'S'/'E'/'W') from mean pin-offset direction."""
  pins = []
  try:
    pins = list((plc.hard_macros_to_inpins or {}).get(name) or [])
  except AttributeError:
    pins = []
  ox = oy = 0.0
  mods = getattr(plc, 'modules_w_pins', [])
  for pin in pins:
    pidx = plc.mod_name_to_indices.get(pin)
    if pidx is None or pidx >= len(mods):
      continue
    try:
      dx, dy = mods[pidx].get_offset()
    except (AttributeError, TypeError):
      continue
    ox += float(dx)
    oy += float(dy)
  if ox == 0.0 and oy == 0.0:
    return 'N'
  if abs(ox) >= abs(oy):
    return 'E' if ox > 0 else 'W'
  return 'N' if oy > 0 else 'S'


def rotated_edge(edge, deg):
  """Edge after rotating a macro by `deg` (clockwise steps of 90)."""
  i = _EDGES.index(edge) if edge in _EDGES else 0
  return _EDGES[(i + (int(deg) // 90)) % 4]


def macro_dict(plc, idx, placed=True):
  x, y = plc.get_node_location(idx)
  w, h = plc.get_node_width_height(idx)
  name = plc.get_node_name(idx)
  soft = bool(plc.is_node_soft_macro(idx))
  group, _label, fill, edge = hierarchy_group(name)
  try:
    pinned = bool(plc.is_node_fixed(idx))
  except Exception:
    pinned = False
  return {'name': name, 'x': float(x), 'y': float(y),
          'w': float(w), 'h': float(h), 'soft': soft,
          'placed': bool(placed), 'group': group, 'fill': fill,
          'edge': edge, 'orient': 0 if soft else orientation_deg(plc, idx),
          'notch': 'N' if soft else pin_edge(plc, name),
          'pinned': pinned}


def snapshot(plc, step, total, current='', placed=None):
  """Plain-data frame: canvas, grid, every macro, ports, blockages.

  `placed`: set of node indices already placed (hard macros not yet
  placed render as faint outlines on the canvas).
  """
  w, h = plc.get_canvas_width_height()
  cols, rows = plc.get_grid_num_columns_rows()
  return {
      'step': step, 'total': total, 'current': current,
      'canvas': (float(w), float(h)), 'grid': (int(cols), int(rows)),
      'macros': [macro_dict(plc, i, placed is None or i in placed)
                 for i in plc.get_macro_indices()],
      'ports': [{'x': float(x), 'y': float(y)}
                for x, y in (plc.get_node_location(i)
                             for i in plc.port_indices)],
      'blockages': [(float(a), float(b), float(c), float(d))
                    for a, b, c, d, *_ in plc.get_blockages()]}


def static_snapshot(netlist, plc_path):
  """One frame for an already-placed design (initial or best .plc)."""
  from neurosilicon import setup as setup_lib
  plc = setup_lib.create_plc(netlist, plc_path)
  hard = [i for i in plc.get_macro_indices()
          if not plc.is_node_soft_macro(i)]
  return plc, snapshot(plc, len(hard), len(hard), '')


def run_live_rollout(netlist, init_placement=None, fd_steps=(10, 10, 10),
                     method='first_legal', seed=0, init_fd=True,
                     extra_mask_fn=None, on_snapshot=None,
                     max_frames=160, on_live=None):
  """Sequential rollout emitting a snapshot as each macro lands.

  Returns (info, frames). `info` is the env terminal info (proxy cost
  and parts) or an infeasible marker. Frames are thinned to at most
  `max_frames` so large designs stay real-time; the final frame is
  always kept.
  """
  env = env_lib.ChipPlacementEnv(
      netlist, init_placement=init_placement, fd_steps=fd_steps,
      extra_mask_fn=extra_mask_fn, init_fd=init_fd, seed=seed)
  obs, _ = env.reset()
  plc = env.plc
  order = env.placement_order
  total = len(order)
  stride = max(1, -(-total // max(max_frames, 1)))
  rng = np.random.default_rng(seed)
  frames = []
  placed = set()

  def emit(step, current):
    frame = snapshot(plc, step, total, current, placed)
    frames.append(frame)
    if on_snapshot is not None:
      on_snapshot(frame)

  emit(0, plc.get_node_name(order[0]) if order else '')
  info = {'infeasible': True, 'proxy_cost': -1.0, 'wirelength': -1.0,
          'congestion': -1.0, 'density': -1.0}
  done = total == 0
  step = 0
  while not done:
    legal = np.flatnonzero(obs['mask'])
    if len(legal) == 0:
      break
    action = (int(legal[0]) if method != 'random'
              else int(rng.choice(legal)))
    node = order[step]
    obs, _, terminated, truncated, step_info = env.step(action)
    placed.add(node)
    step += 1
    if step_info:
      info = step_info
    done = terminated or truncated
    if step % stride == 0 or done:
      nxt = (plc.get_node_name(order[step]) if step < total else '')
      emit(step, nxt)
    if on_live is not None and (step % 5 == 0 or done):
      try:
        wl = float(plc.get_cost())
        dens = float(plc.get_density_cost())
        _, cong, _peak = rudy_grid(plc)
        on_live({'step': step, 'total': total,
                 'wirelength': wl, 'congestion': float(cong),
                 'density': dens,
                 'proxy_cost': wl + 0.5 * cong + dens,
                 'current': plc.get_node_name(node)})
      except Exception:
        pass
    if done and not step_info.get('infeasible', True):
      info = step_info
  if frames and frames[-1]['step'] != step:
    emit(step, '')
  return info, frames


def frame_power(frame):
  """Per-frame heat data: {macro name: area} for placed hard macros.

  Drives the moving thermal visualization: as the sequential placer
  lands more macros, the heat map grows with them. The canvas
  normalizes values per frame, and the desktop scales them by the
  clock/voltage/activity corner.
  """
  return {m['name']: float(m['w']) * float(m['h'])
          for m in frame.get('macros', [])
          if not m['soft'] and m.get('placed', True)}


def _centers(plc):
  centers = {}
  for idx in list(plc.get_macro_indices()) + list(plc.port_indices):
    centers[plc.get_node_name(idx)] = plc.get_node_location(idx)
  return centers


def rudy_grid(plc):
  """RUDY routing-demand heatmap over the placement grid.

  For every net, the bounding box of its placed endpoints accrues
  (bbox wirelength / bbox area) demand into each overlapped tile —
  the classic Rectangular Uniform Wire Density estimate. Returns
  (grid, overall, peak): grid is rows×cols nested lists, overall is
  the mean of the hottest 10% of tiles (AlphaChip-style overflow),
  peak is the single hottest tile.
  """
  import numpy as np
  cols, rows = plc.get_grid_num_columns_rows()
  W, H = plc.get_canvas_width_height()
  cw, ch = W / max(cols, 1), H / max(rows, 1)
  grid = np.zeros((rows, cols), dtype=float)
  centers = _centers(plc)
  for driver, sinks in plc.nets.items():
    pts = []
    for name in [driver] + list(sinks):
      p = centers.get(name.split('/')[0])
      if p is not None:
        pts.append(p)
    if len(pts) < 2:
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    area = max((x1 - x0) * (y1 - y0), cw * ch)
    dens = ((x1 - x0) + (y1 - y0)) / area
    c0 = max(0, min(cols - 1, int(x0 / cw)))
    c1 = max(0, min(cols - 1, int(x1 / cw)))
    r0 = max(0, min(rows - 1, int(y0 / ch)))
    r1 = max(0, min(rows - 1, int(y1 / ch)))
    grid[r0:r1 + 1, c0:c1 + 1] += dens
  flat = np.sort(grid, axis=None)
  top = flat[int(0.9 * len(flat)):] if len(flat) else flat
  overall = float(top.mean()) if len(top) else 0.0
  return grid.tolist(), overall, float(grid.max() if grid.size else 0.0)


def congestion_analysis(plc):
  """MCP-ready congestion payload: scores, peak tiles, hot macros."""
  grid, overall, peak = rudy_grid(plc)
  rows = len(grid)
  cols = len(grid[0]) if rows else 0
  W, H = plc.get_canvas_width_height()
  cw, ch = W / max(cols, 1), H / max(rows, 1)
  cells = sorted(
      ((grid[r][c], c, r) for r in range(rows) for c in range(cols)),
      reverse=True)[:5]
  tiles = [(c, r, float(v)) for v, c, r in cells if v > 0]
  net_count = {}
  for driver, sinks in plc.nets.items():
    for name in [driver] + list(sinks):
      base = name.split('/')[0]
      net_count[base] = net_count.get(base, 0) + 1
  macros = []
  for idx in plc.get_macro_indices():
    name = plc.get_node_name(idx)
    x, y = plc.get_node_location(idx)
    w, h = plc.get_node_width_height(idx)
    hit = 0.0
    for c, r, v in tiles:
      tx, ty = (c + 0.5) * cw, (r + 0.5) * ch
      if abs(tx - x) <= w / 2 + cw and abs(ty - y) <= h / 2 + ch:
        hit = max(hit, v)
    if hit > 0:
      macros.append({'name': name, 'nets': net_count.get(name, 0),
                     'peak_tile': hit,
                     'soft': bool(plc.is_node_soft_macro(idx))})
  macros.sort(key=lambda m: (-m['peak_tile'], -m['nets']))
  return {'overall_congestion': overall, 'peak_congestion': peak,
          'peak_congestion_tiles': tiles,
          'congested_macros': macros[:8], 'grid': grid}


def segment_hotspots(grid, top_frac=0.15):
  """SAM-style segmentation: connected clusters of the hottest tiles.

  Thresholds the top `top_frac` of non-zero tiles and flood-fills
  4-connected regions. Returns [{cells, bbox, peak}] with bbox in
  (col0, row0, col1, row1) tile coordinates, hottest first.
  """
  import numpy as np
  arr = np.asarray(grid, dtype=float)
  if arr.size == 0 or arr.max() <= 0:
    return []
  rows, cols = arr.shape
  nz = arr[arr > 0]
  thr = float(np.percentile(nz, 100.0 * (1.0 - top_frac))) \
      if nz.size else float(arr.max())
  hot = (arr >= thr) & (arr > 0)
  seen = np.zeros_like(hot)
  clusters = []
  for r in range(rows):
    for c in range(cols):
      if not hot[r, c] or seen[r, c]:
        continue
      stack, cells = [(r, c)], []
      seen[r, c] = True
      while stack:
        cr, cc = stack.pop()
        cells.append((cr, cc))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          nr, nc = cr + dr, cc + dc
          if 0 <= nr < rows and 0 <= nc < cols and \
                  hot[nr, nc] and not seen[nr, nc]:
            seen[nr, nc] = True
            stack.append((nr, nc))
      rs = [p[0] for p in cells]
      cs = [p[1] for p in cells]
      clusters.append({
          'cells': len(cells),
          'bbox': (min(cs), min(rs), max(cs), max(rs)),
          'peak': float(max(arr[p] for p in cells))})
  clusters.sort(key=lambda k: -k['peak'])
  return clusters[:8]


def power_grid(plc, power):
  """Rasterizes per-macro power onto the placement grid (centers)."""
  cols, rows = plc.get_grid_num_columns_rows()
  W, H = plc.get_canvas_width_height()
  grid = [[0.0] * cols for _ in range(rows)]
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    x, y = plc.get_node_location(idx)
    c = max(0, min(cols - 1, int(x / max(W, 1e-9) * cols)))
    r = max(0, min(rows - 1, int(y / max(H, 1e-9) * rows)))
    grid[r][c] += float(power.get(plc.get_node_name(idx), 0.0))
  return grid


def segments_for_macro(plc, name, limit=300):
  """Flylines (x1, y1, x2, y2) of nets touching macro `name`."""
  centers = _centers(plc)
  out = []
  for driver, sinks in plc.nets.items():
    ends = [driver] + list(sinks)
    bases = [e.split('/')[0] for e in ends]
    if name not in bases:
      continue
    src = centers.get(bases[0])
    if src is None:
      continue
    for base in bases[1:]:
      dst = centers.get(base)
      if dst is None or dst == src:
        continue
      out.append((float(src[0]), float(src[1]),
                  float(dst[0]), float(dst[1])))
      if len(out) >= limit:
        return out
  return out


def net_segments(plc, limit=240):
  """Wire segments (x1, y1, x2, y2) between connected block centers."""
  centers = {}
  for idx in list(plc.get_macro_indices()) + list(plc.port_indices):
    centers[plc.get_node_name(idx)] = plc.get_node_location(idx)
  out = []
  for driver, sinks in plc.nets.items():
    src = centers.get(driver.split('/')[0])
    if src is None:
      continue
    for sink in sinks:
      dst = centers.get(sink.split('/')[0])
      if dst is None or dst == src:
        continue
      out.append((float(src[0]), float(src[1]),
                  float(dst[0]), float(dst[1])))
      if len(out) >= limit:
        return out
  return out


# -- PPA metrics, critical timing path, compare helpers ------------------
# Documented proxies (not signoff timing/power): wire delay scales with
# Manhattan distance between block centers; logic delay grows mildly
# with macro size. Fmax = 1 / critical delay.
WIRE_DELAY_NS_PER_UM = 0.002
GLOBAL_NET_FANOUT = 32

def _block_info(plc):
  """{name: (x, y, w, h, is_hard, is_port)} for macros and ports.

  Soft (standard-cell cluster) macros are included: macro-to-macro
  signal paths route through them, so excluding them would sever
  every real path. They carry a small pass-through logic delay.
  """
  info = {}
  for idx in plc.get_macro_indices():
    x, y = plc.get_node_location(idx)
    w, h = plc.get_node_width_height(idx)
    soft = bool(plc.is_node_soft_macro(idx))
    info[plc.get_node_name(idx)] = (float(x), float(y), float(w),
                                    float(h), not soft, False)
  for idx in plc.port_indices:
    x, y = plc.get_node_location(idx)
    info[plc.get_node_name(idx)] = (float(x), float(y), 0.0, 0.0,
                                    False, True)
  return info


def _logic_delay_ns(info, name):
  _x, _y, w, h, hard, is_port = info[name]
  if is_port:
    return 0.02
  if not hard:
    return 0.08
  import math
  return 0.8 + 0.01 * math.sqrt(max(w * h, 0.0))


def critical_path(plc):
  """Longest macro-to-macro signal path (critical-delay proxy).

  Builds the block graph from `plc.nets` (driver block -> sink blocks,
  global nets with fanout > GLOBAL_NET_FANOUT skipped as clock/reset
  broadcast), weights each hop by Manhattan wire delay plus the sink
  block's logic delay, and returns the longest path in the levelized
  DAG (feedback loops are cut by levelization, as in static timing).

  Returns {path, points, segments, delay_ns, wire_delay_ns,
  logic_delay_ns, fmax_mhz, hops}. `points` are block centers in path
  order (microns); `segments` are consecutive center pairs. With no
  usable nets the path is the single largest hard macro.
  """
  info = _block_info(plc)
  if not info:
    return {'path': [], 'points': [], 'segments': [], 'delay_ns': 0.0,
            'wire_delay_ns': 0.0, 'logic_delay_ns': 0.0,
            'fmax_mhz': 0.0, 'hops': 0}
  edges = []
  seen = set()
  for driver, sinks in plc.nets.items():
    try:
      sink_list = list(sinks)
    except TypeError:
      continue
    if len(sink_list) > GLOBAL_NET_FANOUT:
      continue
    src = driver.split('/')[0]
    if src not in info:
      continue
    for sink in sink_list:
      dst = str(sink).split('/')[0]
      if dst not in info or dst == src or (src, dst) in seen:
        continue
      seen.add((src, dst))
      edges.append((src, dst))
  logic = {n: _logic_delay_ns(info, n) for n in info}
  if not edges:
    hard = [n for n, v in info.items() if v[4]] or list(info)
    name = max(hard, key=lambda n: info[n][2] * info[n][3])
    d = logic[name]
    return {'path': [name], 'points': [info[name][:2]], 'segments': [],
            'delay_ns': d, 'wire_delay_ns': 0.0, 'logic_delay_ns': d,
            'fmax_mhz': 1000.0 / d if d > 0 else 0.0, 'hops': 0}
  # Levelize (static-timing style): longest hop-count level per node,
  # then keep only level-increasing edges. The result is a DAG, so
  # feedback loops in the net graph cannot inflate the delay by
  # walking in circles — the path is a true simple path.
  level = {n: 0 for n in info}
  for _ in range(len(info)):
    changed = False
    for src, dst in edges:
      if level[src] + 1 > level[dst]:
        level[dst] = level[src] + 1
        changed = True
    if not changed:
      break
  dag = [(s, d) for s, d in edges if level[d] > level[s]]
  dist = dict(logic)
  pred = {n: None for n in info}
  wire_of = {}
  for src, dst in sorted(dag, key=lambda e: level[e[0]]):
    sx, sy = info[src][0], info[src][1]
    dx, dy = info[dst][0], info[dst][1]
    wire = (abs(dx - sx) + abs(dy - sy)) * WIRE_DELAY_NS_PER_UM
    cand = dist[src] + wire + logic[dst]
    if cand > dist[dst] + 1e-12:
      dist[dst] = cand
      pred[dst] = src
      wire_of[dst] = wire
  end = max(dist, key=lambda n: dist[n])
  path, node, guard = [], end, set()
  while node is not None and node not in guard:
    guard.add(node)
    path.append(node)
    node = pred[node]
  path.reverse()
  wire_total = sum(wire_of.get(n, 0.0) for n in path[1:])
  logic_total = sum(logic[n] for n in path)
  points = [info[n][:2] for n in path]
  delay = dist[end]
  return {'path': path, 'points': points,
          'segments': [(points[i][0], points[i][1],
                        points[i + 1][0], points[i + 1][1])
                       for i in range(len(points) - 1)],
          'delay_ns': float(delay), 'wire_delay_ns': float(wire_total),
          'logic_delay_ns': float(logic_total),
          'fmax_mhz': float(1000.0 / delay) if delay > 0 else 0.0,
          'hops': len(path) - 1}


def ppa_metrics(plc, power=None, weights=(1.0, 0.5, 1.0)):
  """Complete PPA snapshot for one placement (all documented proxies).

  Returns proxy/wirelength/congestion/density, RUDY overall/peak, peak
  power + peak power density (the peak-temperature proxy), the peak
  macro, and the critical-path delay / estimated Fmax.
  """
  wl = float(plc.get_cost())
  cong = float(plc.get_congestion_cost())
  dens = float(plc.get_density_cost())
  w_wl, w_cong, w_dens = weights
  if power is None:
    power = scaled_power(plc)
  areas = {}
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = max(float(w) * float(h), 0.0)
  peak_macro, peak_power, peak_density = '', 0.0, 0.0
  for name, p in power.items():
    p = float(p)
    d = p / areas[name] if areas.get(name, 0.0) > 0 else 0.0
    if d > peak_density:
      peak_macro, peak_power, peak_density = name, p, d
  _grid, rudy, rudy_peak = rudy_grid(plc)
  cp = critical_path(plc)
  return {'proxy_cost': float(w_wl * wl + w_cong * cong + w_dens * dens),
          'wirelength': wl, 'congestion': cong, 'density': dens,
          'rudy_congestion': float(rudy), 'rudy_peak': float(rudy_peak),
          'peak_power': float(peak_power),
          'peak_density': float(peak_density),
          'peak_temp': float(peak_density), 'peak_macro': peak_macro,
          'critical_delay_ns': cp['delay_ns'],
          'fmax_mhz': cp['fmax_mhz'], 'critical_path': cp['path']}


def _pct(before, after):
  if not before:
    return 0.0
  return (after - before) / before * 100.0


def ppa_delta(before, after):
  """Percent change per PPA metric (negative = improvement)."""
  return {'proxy_pct': _pct(before.get('proxy_cost', 0.0),
                            after.get('proxy_cost', 0.0)),
          'wirelength_pct': _pct(before.get('wirelength', 0.0),
                                 after.get('wirelength', 0.0)),
          'congestion_pct': _pct(before.get('congestion', 0.0),
                                 after.get('congestion', 0.0)),
          'peak_temp_pct': _pct(before.get('peak_temp', 0.0),
                                after.get('peak_temp', 0.0)),
          'peak_power_pct': _pct(before.get('peak_power', 0.0),
                                 after.get('peak_power', 0.0)),
          'fmax_pct': _pct(before.get('fmax_mhz', 0.0),
                           after.get('fmax_mhz', 0.0))}


def format_ppa_summary(before, after):
  """Structured PPA Improvement Summary (Baseline -> AI-Optimized)."""
  d = ppa_delta(before, after)
  return (
      'PPA Improvement Summary (Baseline → AI-Optimized)\n'
      f"  Δ Proxy %      : {d['proxy_pct']:+.1f}%  "
      f"({before.get('proxy_cost', 0.0):.4f} → "
      f"{after.get('proxy_cost', 0.0):.4f})\n"
      f"  Δ Wirelength % : {d['wirelength_pct']:+.1f}%  "
      f"({before.get('wirelength', 0.0):.4f} → "
      f"{after.get('wirelength', 0.0):.4f})\n"
      f"  Δ Congestion % : {d['congestion_pct']:+.1f}%  "
      f"({before.get('congestion', 0.0):.4f} → "
      f"{after.get('congestion', 0.0):.4f})\n"
      f"  Δ Peak Temp %  : {d['peak_temp_pct']:+.1f}%  "
      f"(peak {before.get('peak_macro', '?')} → "
      f"{after.get('peak_macro', '?')})\n"
      f"  Δ Peak Power % : {d['peak_power_pct']:+.1f}%  ·  "
      f"est. Fmax {before.get('fmax_mhz', 0.0):.0f} → "
      f"{after.get('fmax_mhz', 0.0):.0f} MHz "
      f"({d['fmax_pct']:+.1f}%)\n"
      '  (negative Δ = improvement; proxies, not signoff)')


def pick_compare_runs(results):
  """(baseline_row, best_later_row) for the Before/After compare view.

  Baseline is the first feasible episode; best-later is the lowest-
  proxy feasible episode after it. Returns (row, None) when fewer
  than two feasible episodes exist.
  """
  feas = [r for r in (results or []) if r.get('proxy_cost', -1) > 0]
  if not feas:
    return None, None
  base = feas[0]
  later = feas[1:]
  if not later:
    return base, None
  return base, min(later, key=lambda r: r['proxy_cost'])


def top_hotspot(plc):
  """Top SAM hotspot: (analysis, clusters, rect_microns_or_None).

  Segments the RUDY congestion grid (segment_hotspots) and converts
  the hottest cluster bbox to canvas microns.
  """
  analysis = congestion_analysis(plc)
  clusters = segment_hotspots(analysis['grid'])
  if not clusters:
    return analysis, clusters, None
  W, H = plc.get_canvas_width_height()
  cols, rows = plc.get_grid_num_columns_rows()
  cw, ch = W / max(cols, 1), H / max(rows, 1)
  c0, r0, c1, r1 = clusters[0]['bbox']
  rect = (float(c0 * cw), float(r0 * ch), float((c1 + 1) * cw),
          float((r1 + 1) * ch))
  return analysis, clusters, rect


# -- Pin-aware HPWL, orientation flips, SA polish, macro pinning --------
def _rotate_offset(dx, dy, deg):
  """Pin offset rotated by a macro orientation (plc E-step convention)."""
  deg = int(deg) % 360
  if deg == 90:
    return dy, -dx
  if deg == 180:
    return -dx, -dy
  if deg == 270:
    return -dy, dx
  return dx, dy


def _hpwl_model(plc):
  """Read-only HPWL model: centers, sizes, net endpoint pin offsets.

  Returns (centers, sizes, hard_names, nets) where nets is a list of
  [(block, pin_fullname_or_None)] endpoint lists. Pin positions are
  block center + orientation-rotated pin offset, mirroring the plc's
  own pin geometry without mutating it (plc orientation setters do
  not round-trip, so orientation search must stay read-only).
  """
  centers, sizes = {}, {}
  for idx in list(plc.get_macro_indices()) + list(plc.port_indices):
    name = plc.get_node_name(idx)
    centers[name] = plc.get_node_location(idx)
    sizes[name] = plc.get_node_width_height(idx)
  hard = [plc.get_node_name(i) for i in plc.get_macro_indices()
          if not plc.is_node_soft_macro(i)]
  offsets = {}
  mods = getattr(plc, 'modules_w_pins', [])
  for parent, pins in (getattr(plc, 'hard_macros_to_inpins', {})
                       or {}).items():
    for pin in pins or []:
      pidx = plc.mod_name_to_indices.get(pin)
      if pidx is None or pidx >= len(mods):
        continue
      try:
        offsets[pin] = (parent, mods[pidx].get_offset())
      except (AttributeError, TypeError):
        continue
  nets = []
  for driver, sinks in plc.nets.items():
    ends = []
    for full in [driver] + list(sinks):
      if full in offsets:
        ends.append((offsets[full][0], full))
      else:
        base = str(full).split('/')[0]
        if base in centers:
          ends.append((base, None))
    if len(ends) >= 2:
      nets.append(ends)
  return centers, sizes, hard, nets, offsets


def _hpwl_from(model, positions, orientations):
  centers, _sizes, _hard, nets, offsets = model
  total = 0.0
  for ends in nets:
    xs, ys = [], []
    for block, pin in ends:
      bx, by = positions.get(block, centers.get(block, (0.0, 0.0)))
      if pin is not None:
        dx, dy = _rotate_offset(*offsets[pin][1],
                                orientations.get(block, 0))
        xs.append(bx + dx)
        ys.append(by + dy)
      else:
        xs.append(bx)
        ys.append(by)
    total += (max(xs) - min(xs)) + (max(ys) - min(ys))
  return total


def net_hpwl(plc, orientations=None, positions=None):
  """Total pin-aware net HPWL in microns (orientation-aware proxy)."""
  model = _hpwl_model(plc)
  centers = {k: tuple(v) for k, v in model[0].items()}
  if positions:
    centers.update({k: tuple(v) for k, v in positions.items()})
  return _hpwl_from(model, centers, dict(orientations or {}))


def optimize_orientations(plc, orientations=None):
  """Greedy 0/90/180/270 flip search minimizing pin-aware HPWL.

  One pass over hard macros with pins; each macro keeps the
  orientation with the lowest total HPWL. Read-only on the plc.
  Returns (orientations {name: deg}, hpwl_before, hpwl_after).
  """
  model = _hpwl_model(plc)
  centers = {k: tuple(v) for k, v in model[0].items()}
  current = {}
  for idx in plc.get_macro_indices():
    if not plc.is_node_soft_macro(idx):
      current[plc.get_node_name(idx)] = orientation_deg(plc, idx)
  current.update({k: int(v) % 360
                  for k, v in (orientations or {}).items()})
  before = _hpwl_from(model, centers, current)
  best = dict(current)
  with_pins = {parent for parent, _off in model[4].values()}
  for name in model[2]:
    if name not in with_pins:
      continue
    chosen, chosen_cost = best[name], _hpwl_from(model, centers, best)
    for deg in (0, 90, 180, 270):
      if deg == best[name]:
        continue
      trial = dict(best)
      trial[name] = deg
      cost = _hpwl_from(model, centers, trial)
      if cost < chosen_cost - 1e-9:
        chosen, chosen_cost = deg, cost
    best[name] = chosen
  return best, before, _hpwl_from(model, centers, best)


def simulated_annealing(plc, steps=200, seed=0, pinned=(),
                        orientations=None, use_orientation_flips=True):
  """Post-placement SA polish: compatible-macro swaps + flips.

  Moves: swap two same-size (or transposed-size) hard macros, flip
  one macro's orientation, or nudge one macro. Pinned macros never
  move. Score is pin-aware HPWL; the best state seen is returned as
  {positions, orientations, hpwl_before, hpwl_after, accepted, steps}.
  Read-only on the plc — callers apply `positions` explicitly.
  """
  import random as _random
  rng = _random.Random(seed)
  model = _hpwl_model(plc)
  centers = {k: tuple(v) for k, v in model[0].items()}
  sizes = model[1]
  hard = [n for n in model[2] if n not in set(pinned or ())]
  pos = {n: centers[n] for n in model[2]}
  orient = {}
  for idx in plc.get_macro_indices():
    if not plc.is_node_soft_macro(idx):
      orient[plc.get_node_name(idx)] = orientation_deg(plc, idx)
  orient.update({k: int(v) % 360
                 for k, v in (orientations or {}).items()})
  canvas_w, canvas_h = plc.get_canvas_width_height()
  cur = _hpwl_from(model, {**centers, **pos}, orient)
  before = cur
  best_pos, best_orient, best_cost = dict(pos), dict(orient), cur
  pairs = [(a, b) for i, a in enumerate(hard) for b in hard[i + 1:]
           if (abs(sizes[a][0] - sizes[b][0]) < 1e-6
               and abs(sizes[a][1] - sizes[b][1]) < 1e-6)
           or (abs(sizes[a][0] - sizes[b][1]) < 1e-6
               and abs(sizes[a][1] - sizes[b][0]) < 1e-6)]
  temp0 = max(before * 0.02, 1.0)
  accepted = 0
  for step in range(max(int(steps), 0)):
    if not hard:
      break
    temp = temp0 * (0.02 ** (step / max(steps, 1)))
    trial_pos, trial_orient = dict(pos), dict(orient)
    roll = rng.random()
    if pairs and roll < 0.6:
      a, b = pairs[rng.randrange(len(pairs))]
      trial_pos[a], trial_pos[b] = trial_pos[b], trial_pos[a]
    elif use_orientation_flips and roll < 0.85:
      name = hard[rng.randrange(len(hard))]
      trial_orient[name] = (trial_orient.get(name, 0)
                            + rng.choice((90, 180, 270))) % 360
    else:
      name = hard[rng.randrange(len(hard))]
      w, h = sizes[name]
      x, y = trial_pos[name]
      nx = min(max(x + rng.uniform(-0.12, 0.12) * canvas_w, w / 2),
               max(canvas_w - w / 2, w / 2))
      ny = min(max(y + rng.uniform(-0.12, 0.12) * canvas_h, h / 2),
               max(canvas_h - h / 2, h / 2))
      trial_pos[name] = (nx, ny)
    cost = _hpwl_from(model, {**centers, **trial_pos}, trial_orient)
    if cost <= cur or rng.random() < _anneal_accept(cur, cost, temp):
      pos, orient, cur = trial_pos, trial_orient, cost
      accepted += 1
      if cost < best_cost:
        best_pos, best_orient, best_cost = dict(pos), dict(orient), cost
  return {'positions': best_pos, 'orientations': best_orient,
          'hpwl_before': before, 'hpwl_after': best_cost,
          'accepted': accepted, 'steps': int(steps)}


def _anneal_accept(cur, cost, temp):
  import math
  if temp <= 0:
    return 0.0
  return math.exp(-(cost - cur) / temp)


def apply_positions(plc, positions):
  """Writes {macro: (x, y)} centers into a plc (positions only)."""
  for name, (x, y) in (positions or {}).items():
    idx = plc.get_node_index(name)
    if idx is not None and idx >= 0:
      plc.update_node_coords(idx, float(x), float(y))
  return plc


def write_pinned_placement(netlist, base_plc_path, pins, out_path):
  """Init .plc with pinned macros fixed at their (x, y) centers.

  Fixed nodes survive env resets (unplace-all skips them and the
  placement order excludes them), so later runs place the remaining
  macros around the pins. Returns out_path.
  """
  from neurosilicon import setup as setup_lib
  from neurosilicon.thirdparty.tilos.Plc_client import (
      placement_util_os as placement_util)
  plc = setup_lib.create_plc(netlist, base_plc_path)
  canvas_w, canvas_h = plc.get_canvas_width_height()
  for name, (x, y) in (pins or {}).items():
    idx = plc.get_node_index(name)
    if idx is None or idx < 0:
      continue
    w, h = plc.get_node_width_height(idx)
    cx = min(max(float(x), w / 2), max(canvas_w - w / 2, w / 2))
    cy = min(max(float(y), h / 2), max(canvas_h - h / 2, h / 2))
    plc.update_node_coords(idx, cx, cy)
    plc.fix_node_coord(idx)
  placement_util.save_placement(plc, out_path)
  return out_path


def scaled_power(plc, clock_mhz=100.0, voltage=1.8, activity=0.2):
  """Per-hard-macro relative power at a clock/voltage/activity corner.

  Area-proportional estimate (documented proxy, not signoff power),
  scaled by (V/1.8)^2 x (f/100MHz) x (activity/0.2). Sums to the
  scale factor, so corners stay comparable.
  """
  areas = {}
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = float(w) * float(h)
  total = sum(areas.values())
  if total <= 0:
    return {k: 0.0 for k in areas}
  scale = ((voltage / 1.8) ** 2 * (clock_mhz / 100.0)
           * (activity / 0.2))
  return {k: (v / total) * scale for k, v in areas.items()}
