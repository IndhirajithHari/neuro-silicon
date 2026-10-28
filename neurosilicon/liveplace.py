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


def macro_dict(plc, idx, placed=True):
  x, y = plc.get_node_location(idx)
  w, h = plc.get_node_width_height(idx)
  return {'name': plc.get_node_name(idx), 'x': float(x), 'y': float(y),
          'w': float(w), 'h': float(h),
          'soft': bool(plc.is_node_soft_macro(idx)),
          'placed': bool(placed)}


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
