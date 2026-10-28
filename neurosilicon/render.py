"""Floorplan + congestion rendering for real placements (headless-safe)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_floorplan(plc, ax=None, show_ports=True, show_soft=True,
                   blocked=None):
  """Draws macros/ports/blockages from a PlacementCost onto `ax`.

  `blocked`: optional boolean (rows, cols) real-grid array; blocked
  cells render as a translucent red overlay (constraint mask view).
  """
  w, h = plc.get_canvas_width_height()
  own = ax is None
  if own:
    _, ax = plt.subplots(figsize=(8, 8 * h / max(w, 1e-9)))
  ax.set_xlim(0, w)
  ax.set_ylim(0, h)
  ax.set_aspect('equal')
  ax.set_facecolor('#0f172a')
  ax.set_title('NeuroSilicon floorplan')
  if blocked is not None:
    import numpy as np
    from matplotlib.colors import ListedColormap
    glow = np.where(np.asarray(blocked, dtype=bool), 1.0, np.nan)
    ax.imshow(glow, origin='lower', extent=[0, w, 0, h],
              cmap=ListedColormap(['#ef444466']), vmin=0, vmax=1)
  for (x0, y0, x1, y1, *_) in plc.get_blockages():
    ax.add_patch(
        patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                          facecolor='#ef444433', edgecolor='#ef4444',
                          linestyle='--', linewidth=1.0))
  from neurosilicon import liveplace as liveplace_lib
  seen_groups = []
  for idx in plc.get_macro_indices():
    x, y = plc.get_node_location(idx)
    mw, mh = plc.get_node_width_height(idx)
    if plc.is_node_soft_macro(idx):
      if not show_soft:
        continue
      fc, ec = '#64748b55', '#94a3b8'
    else:
      _g, label, fill, edge = liveplace_lib.hierarchy_group(
          plc.get_node_name(idx))
      fc, ec = fill + 'cc', edge
      if label not in seen_groups:
        seen_groups.append(label)
    ax.add_patch(
        patches.Rectangle((x - mw / 2, y - mh / 2), mw, mh,
                          facecolor=fc, edgecolor=ec, linewidth=1.0))
  if own and seen_groups:
    handles = []
    for _g, label, fill, edge in liveplace_lib.HIERARCHY_GROUPS:
      if label in seen_groups:
        handles.append(patches.Patch(facecolor=fill, edgecolor=edge,
                                     label=label))
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              title='RTL hierarchy', title_fontsize=7)
  if show_ports:
    xs, ys = [], []
    for idx in plc.port_indices:
      x, y = plc.get_node_location(idx)
      xs.append(x)
      ys.append(y)
    ax.scatter(xs, ys, s=4, c='#fbbf24', zorder=3)
  if own:
    return ax.figure
  return ax


def congestion_maps(plc):
  """Returns (vertical, horizontal) congestion grids (rows x cols)."""
  import numpy as np
  cols, rows = plc.get_grid_num_columns_rows()
  v = np.asarray(plc.V_routing_cong, dtype=float).reshape(rows, cols)
  h = np.asarray(plc.H_routing_cong, dtype=float).reshape(rows, cols)
  return v, h


def plot_congestion(plc, ax=None):
  """Plots the combined V+H congestion heatmap (computes costs first)."""
  import numpy as np
  plc.get_congestion_cost()
  v, h = congestion_maps(plc)
  w, h_w = plc.get_canvas_width_height()
  own = ax is None
  if own:
    _, ax = plt.subplots(figsize=(7, 7))
  im = ax.imshow((v + h).T, origin='lower', cmap='inferno',
                 extent=[0, w, 0, h_w])
  ax.set_title('Routing congestion (V+H)')
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  if own:
    return ax.figure
  return ax


def estimate_power(plc):
  """Area-proportional power estimate per hard macro, normalized to 1.

  Documented estimate (not signoff power): real power needs a liberty
  + activity flow. Keys are macro names; soft macros are skipped.
  """
  areas = {}
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = w * h
  total = sum(areas.values())
  if total <= 0:
    return {k: 0.0 for k in areas}
  return {k: v / total for k, v in areas.items()}


def plot_thermal(plc, power=None, ax=None):
  """Per-macro power heatmap. `power`: {macro name: watts-ish}."""
  power = power or estimate_power(plc)
  w, h = plc.get_canvas_width_height()
  own = ax is None
  if own:
    _, ax = plt.subplots(figsize=(8, 8 * h / max(w, 1e-9)))
  ax.set_xlim(0, w)
  ax.set_ylim(0, h)
  ax.set_aspect('equal')
  ax.set_facecolor('#0f172a')
  ax.set_title('Thermal map (relative macro power)')
  vals = [power.get(plc.get_node_name(i), 0.0)
          for i in plc.get_macro_indices()
          if not plc.is_node_soft_macro(i)]
  vmax = max(vals) if vals else 1.0
  cmap = plt.get_cmap('inferno')
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    x, y = plc.get_node_location(idx)
    mw, mh = plc.get_node_width_height(idx)
    p = power.get(plc.get_node_name(idx), 0.0) / max(vmax, 1e-9)
    ax.add_patch(
        patches.Rectangle((x - mw / 2, y - mh / 2), mw, mh,
                          facecolor=cmap(p), edgecolor='#e2e8f0',
                          linewidth=0.8))
  sm = plt.cm.ScalarMappable(cmap=cmap,
                             norm=plt.Normalize(0, max(vmax, 1e-9)))
  plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='power')
  if own:
    return ax.figure
  return ax


def cost_breakdown_figure(rows):
  """Grouped wl/cong/dens/top10 bars per episode from results rows."""
  import numpy as np
  labels = [f"{r.get('method', '?')}#{r.get('episode', '?')}" for r in rows]
  wl = [max(r.get('wirelength', -1), 0) for r in rows]
  cong = [max(r.get('congestion', -1), 0) for r in rows]
  dens = [max(r.get('density', -1), 0) for r in rows]
  top10 = [max(r.get('top10cong', -1), 0) for r in rows]
  x = np.arange(len(rows))
  fig, ax = plt.subplots(figsize=(max(7, len(rows)), 4))
  ax.bar(x - 0.375, wl, 0.2, label='wirelength')
  ax.bar(x - 0.125, cong, 0.2, label='congestion')
  ax.bar(x + 0.125, dens, 0.2, label='density')
  ax.bar(x + 0.375, top10, 0.2, label='top-10% cong')
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=30, ha='right')
  ax.set_ylabel('cost component')
  ax.set_title('Cost breakdown per episode')
  ax.legend()
  fig.tight_layout()
  return fig


def history_figure(rows):
  """Training curves (eval cost + mean reward) from history rows.

  Rows: list of {iter, mean_reward, eval_cost} dicts (NaN evals skipped
  in the eval curve).
  """
  import numpy as np
  it = [r['iter'] for r in rows]
  mr = [r['mean_reward'] for r in rows]
  ev = [(r['iter'], r['eval_cost']) for r in rows
        if r.get('eval_cost') == r.get('eval_cost')]
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(it, mr, label='mean reward', color='#0284c7')
  if ev:
    ax.plot([e[0] for e in ev], [e[1] for e in ev], 'o-',
            label='eval cost', color='#dc2626')
  ax.set_xlabel('iteration')
  ax.set_title('Training progress')
  ax.legend()
  fig.tight_layout()
  return fig


def plot_sim_step(canvas, placed, probs, current_macro, action_cell,
                  value, step, total, edges=(), soft_xy=None):
  """One GNN rollout step: heatmap + partial floorplan + graph edges.

  Args:
    canvas: (width, height) microns.
    placed: [{name, x, y, w, h}] macros placed so far.
    probs: 2D (rows, cols) policy distribution over the real grid.
    current_macro: name being placed this step.
    action_cell: (row, col) in 128-padded coords.
    value: value-head estimate V(s).
    edges: circuit_edges() list; pairs with both ends drawn are shown.
    soft_xy: {name: (x, y)} soft-macro dots (initial coords).
  """
  w, h = canvas
  rows, cols = probs.shape
  cw, ch = w / max(cols, 1), h / max(rows, 1)
  fig, ax = plt.subplots(figsize=(8, 8 * h / max(w, 1e-9)))
  ax.set_xlim(0, w)
  ax.set_ylim(0, h)
  ax.set_aspect('equal')
  ax.set_facecolor('#0f172a')
  im = ax.imshow(probs, origin='lower', cmap='inferno', alpha=0.85,
                 extent=[0, w, 0, h])
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(cell)')
  centers = dict(soft_xy or {})
  if centers:
    ax.scatter([p[0] for p in centers.values()],
               [p[1] for p in centers.values()],
               s=6, c='#94a3b8', alpha=0.7, zorder=2)
  for k, m in enumerate(placed):
    centers[m['name']] = (m['x'], m['y'])
    ax.add_patch(
        patches.Rectangle((m['x'] - m['w'] / 2, m['y'] - m['h'] / 2),
                          m['w'], m['h'], facecolor='#0284c788',
                          edgecolor='#38bdf8', linewidth=1.2))
    ax.text(m['x'], m['y'], m['name'], color='white', fontsize=7,
            ha='center', va='center')
    ax.text(m['x'] - m['w'] / 2, m['y'] + m['h'] / 2, f'{k + 1}',
            color='#fbbf24', fontsize=9, weight='bold', ha='left',
            va='top')
  ecmax = max([c for _, _, c in edges] + [1])
  for a, b, c in edges:
    if a in centers and b in centers:
      ax.plot([centers[a][0], centers[b][0]],
              [centers[a][1], centers[b][1]],
              color='white', alpha=0.10 + 0.35 * c / ecmax, linewidth=0.8)
  r, c = action_cell
  p = float(probs[r][c]) if 0 <= r < rows and 0 <= c < cols else 0.0
  ax.plot((c + 0.5) * cw, (r + 0.5) * ch, marker='*', color='#fbbf24',
          markersize=14, markeredgecolor='black')
  ax.set_title(f'Step {step + 1}/{total}: {current_macro} → '
               f'({r},{c}) p={p:.3f} V={value:.2f}')
  return fig


def draw_critical_path(ax, plc, cp=None):
  """Overlays the critical timing path (gold/magenta arrows) on `ax`.

  `cp` is a liveplace.critical_path() result; computed when omitted.
  Segments route Manhattan-style (horizontal then vertical elbow)
  between block centers, with an arrowhead at every hop. Returns cp.
  """
  from neurosilicon import liveplace as liveplace_lib
  cp = cp or liveplace_lib.critical_path(plc)
  pts = cp.get('points') or []
  for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
    ax.plot([x1, x2, x2], [y1, y1, y2], color='#ff2ea6', linewidth=4.0,
            alpha=0.55, zorder=5, solid_capstyle='round')
    ax.plot([x1, x2, x2], [y1, y1, y2], color='#ffd700', linewidth=1.8,
            zorder=6, solid_capstyle='round')
    ax.annotate('', xy=(x2, y2), xytext=(x2, y1),
                arrowprops={'arrowstyle': '-|>', 'color': '#ffd700',
                            'lw': 1.8}, zorder=6)
  if pts:
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26,
               c='#ffd700', edgecolors='#ff2ea6', zorder=7)
  return cp


def comparison_figure(plc_before, plc_after, before_label='Run #1 — Baseline',
                      after_label='Best AI-Optimized'):
  """Before vs. After split figure: two floorplans side by side."""
  fig, axes = plt.subplots(1, 2, figsize=(13, 6))
  for ax, plc, label in ((axes[0], plc_before, before_label),
                         (axes[1], plc_after, after_label)):
    plot_floorplan(plc, ax=ax)
    ax.set_title(label)
  fig.tight_layout()
  return fig


def kpi_bar_figure(rows):
  """Before vs. After KPI bars: proxy/wirelength/congestion per run.

  Feasible rows only, in run order — the Pareto-style trade-off view
  (lower is better on every bar; the best run per metric wins).
  """
  import numpy as np
  feas = [r for r in (rows or []) if r.get('proxy_cost', -1) > 0]
  fig, ax = plt.subplots(figsize=(max(7, len(feas) * 1.2), 4))
  if not feas:
    ax.set_title('KPI bars — no feasible runs yet')
    return fig
  labels = [f"{r.get('method', '?')[:18]}#{r.get('episode', '?')}"
            for r in feas]
  x = np.arange(len(feas))
  ax.bar(x - 0.25, [r['proxy_cost'] for r in feas], 0.25,
         label='Proxy Cost', color='#e8a33d')
  ax.bar(x, [r['wirelength'] for r in feas], 0.25,
         label='Wirelength', color='#38bdf8')
  ax.bar(x + 0.25, [r['congestion'] for r in feas], 0.25,
         label='Congestion', color='#e879f9')
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
  ax.set_ylabel('cost (lower is better)')
  ax.set_title('Before vs. After KPI — Proxy / Wirelength / Congestion '
               'per run (Pareto view)')
  ax.legend()
  fig.tight_layout()
  return fig


def figure_png(fig):
  """Renders a matplotlib figure to PNG bytes (for downloads)."""
  import io
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=110)
  return buf.getvalue()


def save_figures(plc, outdir, power=None):
  """Writes floorplan/congestion (+thermal when `power`) PNGs to `outdir`."""
  import os
  os.makedirs(outdir, exist_ok=True)
  paths = []
  for name, fig in (('floorplan.png', plot_floorplan(plc)),
                    ('congestion.png', plot_congestion(plc))):
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
  if power is not None:
    fig = plot_thermal(plc, power)
    p = os.path.join(outdir, 'thermal.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
  return paths
