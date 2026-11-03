"""NeuroSilicon desktop application (native window, no web server).

Industry-style EDA workstation on Qt (PySide6), dark theme:

* Left: design browser, design facts, toolchain + AI status.
* Center: a native real-time floorplan canvas (QPainter, no image
  round-trip) with AlphaChip-style playback — macros land one by one,
  largest first, with Play/Pause, step, scrub and speed controls and
  Floorplan / Congestion / Thermal / Wires views — over tabbed
  Import / Place / Results & Downloads / Thermal lab panels.
* Right: MCP-style AI chatbox, AI Connectors (cloud API + local LLM,
  optional, with Test connection), an in-app setup Guide, full
  Settings, and the live log.

Import is automatic: pick a whole chip design (folder, any file in
it, or a .zip — drag-and-drop works too) and `discover` fetches what
floorplanning needs (ready netlist pair, or RTL + auto-picked top for
yosys synthesis). Placement reuses the CLI engine (`project`,
`evaluate`, `liveplace`, `render`); downloads export the placed
`.plc`, viewer DEF, PNGs and a report ZIP. Nothing uses HTTP except
the optional LLM connectors.

Run from the repo root::

    .venv/bin/python -m neurosilicon.desktop
"""
import math
import os
import re
import sys
import time
import traceback
import zipfile

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')

from PySide6.QtCore import QObject, QPointF, QThread, QTimer, Signal, Qt
from PySide6.QtGui import (QAction, QColor, QFont, QPainter, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QCompleter, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox,
    QDoubleSpinBox, QSplitter, QTableWidget, QFrame, QTableWidgetItem,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from neurosilicon import assistant as assistant_lib
from neurosilicon import discover as discover_lib
from neurosilicon import liveplace as liveplace_lib
from neurosilicon import project as project_lib
from neurosilicon import settings as settings_lib
from neurosilicon import tech as tech_lib

BUILTIN_NETLISTS = {
    'toy (2 macros)': (
        'circuit_training/environment/test_data/toy_macro_stdcell/'
        'netlist.pb.txt',
        'circuit_training/environment/test_data/toy_macro_stdcell/'
        'initial.plc'),
    'ariane (133 macros)': (
        'circuit_training/environment/test_data/ariane/netlist.pb.txt',
        'circuit_training/environment/test_data/ariane/initial.plc'),
    'servant SoC (18 macros, sky130)': (
        'logs/servant_proto/netlist.pb.txt',
        'logs/servant_proto/initial.plc'),
    'picorv32 (2 macros + clusters, sky130)': (
        'logs/picorv32_proto/netlist.pb.txt',
        'logs/picorv32_proto/initial.plc'),
    'picosoc (6 macros + clusters, sky130)': (
        'logs/picosoc_proto/netlist.pb.txt',
        'logs/picosoc_proto/initial.plc'),
    'ultraembedded riscv_core (10 macros + clusters, sky130)': (
        'logs/ue_proto/netlist.pb.txt',
        'logs/ue_proto/initial.plc'),
}

DARK = """
QMainWindow, QWidget { background:#0d1117; color:#e6edf3; }
QLabel { color:#e6edf3; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
QTableWidget, QPlainTextEdit, QTextBrowser { background:#161b22;
border:1px solid #30363d; border-radius:4px; padding:3px; }
QPushButton { background:#21262d; border:1px solid #30363d;
border-radius:4px; padding:5px 12px; }
QPushButton:hover { background:#30363d; }
QPushButton:disabled { color:#6e7681; }
QPushButton#primary { background:#9e6a1e; border-color:#e8a33d;
font-weight:600; }
QPushButton#primary:hover { background:#b57a24; }
QHeaderView::section { background:#21262d; color:#e6edf3; padding:4px; }
QTabBar::tab { background:#161b22; padding:6px 12px; }
QTabBar::tab:selected { background:#21262d;
border-bottom:2px solid #e8a33d; }
QProgressBar { border:1px solid #30363d; border-radius:4px;
text-align:center; }
QProgressBar::chunk { background:#e8a33d; }
QLabel#card { background:#161b22; border:1px solid #30363d;
border-radius:6px; padding:6px 10px; }
QLabel#cardTitle { color:#8b949e; font-size:11px; }
"""


def _wrap_label(text):
  """QLabel with word wrap (long help text must never widen a panel)."""
  label = QLabel(text)
  label.setWordWrap(True)
  return label


def _abs(path):
  return path if os.path.isabs(path) else os.path.join(
      project_lib.repo_root(), path)


class Worker(QObject):
  """Runs one engine callable off the UI thread, streaming events."""
  log = Signal(str)
  row = Signal(dict)
  snap = Signal(dict)
  phase = Signal(str)
  live = Signal(dict)
  done = Signal(object)
  failed = Signal(str, str)

  def __init__(self, fn):
    super().__init__()
    self._fn = fn

  def run(self):
    try:
      self.done.emit(self._fn(self))
    except Exception:
      full = traceback.format_exc()
      short = full.strip().splitlines()[-1]
      if 'yosys failed' in full or 'ERROR:' in full:
        short += ('\n\nHint: check the Log — files unreachable from '
                  'the top are pruned automatically; if the failing '
                  'file IS in your design, its syntax needs a newer '
                  'yosys frontend.')
      self.failed.emit(short, full)


_HEAT_STOPS = ((0.0, (13, 8, 35)), (0.35, (122, 30, 105)),
               (0.65, (221, 81, 40)), (0.85, (247, 165, 32)),
               (1.0, (252, 255, 164)))


def _heat_color(t):
  """Inferno-like ramp: 0 (cold/dark) -> 1 (hot/bright)."""
  t = max(0.0, min(1.0, float(t)))
  for (t0, c0), (t1, c1) in zip(_HEAT_STOPS, _HEAT_STOPS[1:]):
    if t <= t1:
      f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
      return QColor(*(int(a + (b - a) * f) for a, b in zip(c0, c1)))
  return QColor(*_HEAT_STOPS[-1][1])


class ChipCanvas(QWidget):
  macro_clicked = Signal(dict)
  pin_toggled = Signal(dict)
  macro_moved = Signal(str, float, float)
  """Real-time floorplan canvas painted natively (no PNG round-trip).

  Holds one plain-data frame from `liveplace` plus view state. Views:
  Floorplan (hard macros blue, soft clusters grey, pending macros as
  faint outlines, the macro being placed outlined in amber),
  Congestion (routing-demand heat cells under the macros), Thermal
  (per-macro power heat), Wires (net segments between block centers),
  and combined assemblies Floorplan + Wires / Thermal + Wires (base
  view with the net segments overlaid). Chat keep-out rectangles
  and netlist blockages render in red.
  """

  VIEWS = ('Floorplan', 'Congestion', 'Thermal', 'Wires',
           'Floorplan + Wires', 'Thermal + Wires',
           'Congestion + Wires', 'Compare (Baseline vs. AI)')

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setMinimumSize(340, 240)
    self._frame = None
    self._view = 'Floorplan'
    self._power = {}
    self._segments = []
    self._congestion = None  # (rows x cols) array-like or None
    self._keepouts = []
    self._contour_boxes = []  # [(x0, y0, x1, y1, label)] microns
    self._selected = ''
    self._highlight_segments = []
    self._tf = None  # (ox, oy, scale, W, H) from last paint
    self._compare = None  # (left_snap, right_snap) for Compare view
    self._crit_points = []  # [(x, y)] critical-path centers, microns
    self._crit_visible = False
    self._pinned_names = set()
    self._orientations = {}  # {macro: degrees} optimizer overrides
    self._drag_name = None
    self._drag_start = None
    self._drag_xy = None
    self._dragging = False

  def set_frame(self, frame):
    self._frame = frame
    self.update()

  def set_view(self, view):
    if view in self.VIEWS:
      self._view = view
      self.update()

  def set_power(self, power):
    self._power = dict(power or {})
    self.update()

  def set_segments(self, segments):
    self._segments = list(segments or [])
    self.update()

  def set_congestion(self, grid):
    self._congestion = grid
    self.update()

  def set_keepouts(self, rects):
    self._keepouts = [tuple(r) for r in (rects or [])]
    self.update()

  def set_contours(self, boxes):
    """SAM bottleneck contours: [(x0, y0, x1, y1, label)] in microns."""
    self._contour_boxes = list(boxes or [])
    self.update()

  def set_compare(self, left, right):
    """Snapshots for Compare (Baseline vs. AI); None clears the pair.

    Each snapshot: {label, frame, power, grid, ppa, badges}. `badges`
    (right side only) is [(text, improved)] drawn as delta pills.
    """
    self._compare = (left, right) if left and right else None
    self.update()

  def set_pinned(self, names):
    """Macros rendered with the 🔒 pinned badge (amber outline)."""
    self._pinned_names = set(names or ())
    self.update()

  def set_orientations(self, orientations):
    """{macro: degrees} overrides rotating each pin-facing notch."""
    self._orientations = dict(orientations or {})
    self.update()

  def set_critical_path(self, points, visible=True):
    """Critical-path overlay polyline [(x, y)] in microns."""
    self._crit_points = [(float(x), float(y)) for x, y in (points or [])]
    self._crit_visible = bool(visible)
    self.update()

  def set_highlight(self, name, segments):
    """Selects a macro and shows its point-to-point flylines."""
    self._selected = name or ''
    self._highlight_segments = list(segments or [])
    self.update()

  def _macro_at_widget(self, px, py):
    """Macro dict under a widget pixel, or None (hard macros first)."""
    if not self._frame or not self._tf:
      return None
    ox, oy, s, W, H = self._tf
    if s <= 0:
      return None
    dx, dy = (px - ox) / s, H - (py - oy) / s
    for soft_pass in (False, True):
      for m in reversed(self._frame['macros']):
        if m['soft'] != soft_pass:
          continue
        if abs(dx - m['x']) <= m['w'] / 2 and \
                abs(dy - m['y']) <= m['h'] / 2:
          return m
    return None

  def _paint_compare(self, p):
    """Split-screen Before vs. After (baseline left, AI right)."""
    if not self._compare:
      p.setPen(QColor('#8b949e'))
      p.drawText(self.rect(), Qt.AlignCenter,
                 'Compare (Baseline vs. AI)\nRun ⚡ Auto-Optimize '
                 '(AI Loop) — or two scored runs — first.\nLeft will '
                 'show Run #1 (Baseline), right the best AI-Optimized '
                 'run with Δ badges.')
      return
    left, right = self._compare
    half = self.width() // 2
    self._paint_side(p, 4, 4, half - 8, self.height() - 8, left, False)
    self._paint_side(p, half + 4, 4, half - 8, self.height() - 8,
                     right, True)
    p.setPen(QPen(QColor('#e8a33d'), 2))
    p.drawLine(half, 6, half, self.height() - 6)

  def _paint_side(self, p, rx, ry, rw, rh, snap, is_after):
    frame = snap.get('frame') or {}
    W, H = frame.get('canvas', (0.0, 0.0))
    p.save()
    p.setClipRect(rx, ry, rw, rh)
    p.fillRect(rx, ry, rw, rh, QColor('#0f172a'))
    title_font = QFont()
    title_font.setPointSize(10)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(QColor('#fbbf24' if is_after else '#e6edf3'))
    p.drawText(rx + 8, ry + 18, snap.get('label', ''))
    ppa = snap.get('ppa') or {}
    small = QFont()
    small.setPointSize(8)
    p.setFont(small)
    p.setPen(QColor('#e6edf3'))
    p.drawText(rx + 8, ry + 34,
               f"proxy {ppa.get('proxy_cost', 0.0):.4f} · wl "
               f"{ppa.get('wirelength', 0.0):.4f} · cong "
               f"{ppa.get('congestion', 0.0):.4f} · Fmax "
               f"{ppa.get('fmax_mhz', 0.0):.0f} MHz")
    if W > 0 and H > 0:
      top, bottom = ry + 42, ry + rh - 10
      s = min((rw - 16) / W, max(bottom - top, 10) / H)
      ox, oy = rx + (rw - W * s) / 2, top + (bottom - top - H * s) / 2

      def X(x):
        return ox + x * s

      def Y(y):
        return oy + (H - y) * s

      grid = snap.get('grid')
      if grid:
        rows_n = len(grid)
        cols_n = len(grid[0]) if rows_n else 0
        vmax = max((max(r) for r in grid), default=1.0) or 1.0
        cw, ch = W / max(cols_n, 1), H / max(rows_n, 1)
        for r, row in enumerate(grid):
          for c, v in enumerate(row):
            if v > 0:
              col = _heat_color(v / vmax)
              col.setAlpha(190)
              p.fillRect(int(X(c * cw)), int(Y((r + 1) * ch)),
                         max(1, int(cw * s)) + 1,
                         max(1, int(ch * s)) + 1, col)
      power = snap.get('power') or {}
      pmax = max(power.values(), default=1.0) or 1.0
      for m in frame.get('macros', []):
        x, y = X(m['x'] - m['w'] / 2), Y(m['y'] + m['h'] / 2)
        w_px, h_px = m['w'] * s, m['h'] * s
        if m['soft']:
          p.fillRect(int(x), int(y), int(w_px), int(h_px),
                     QColor(100, 116, 139, 52))
        elif power:
          p.fillRect(int(x), int(y), int(w_px), int(h_px),
                     _heat_color((power.get(m['name'], 0.0) / pmax)
                                 ** 0.5))
          p.setPen(QPen(QColor('#e2e8f0'), 1))
          p.drawRect(int(x), int(y), int(w_px), int(h_px))
        else:
          p.fillRect(int(x), int(y), int(w_px), int(h_px),
                     QColor(2, 132, 199, 150))
          p.setPen(QPen(QColor('#38bdf8'), 1))
          p.drawRect(int(x), int(y), int(w_px), int(h_px))
      p.setPen(QPen(QColor('#e8a33d'), 2))
      p.drawRect(int(ox), int(oy), int(W * s), int(H * s))
    badges = snap.get('badges') or []
    by = ry + 48
    for text, improved in badges:
      bw = max(118, 8 + p.fontMetrics().horizontalAdvance(text))
      bg = QColor('#14532d') if improved else QColor('#450a0a')
      fg = QColor('#4ade80') if improved else QColor('#f87171')
      p.setPen(QPen(fg, 1))
      p.fillRect(rx + rw - bw - 8, by, bw, 20, bg)
      p.drawRect(rx + rw - bw - 8, by, bw, 20)
      p.setPen(fg)
      p.drawText(rx + rw - bw, by + 14, text)
      by += 26
    p.restore()

  def _draw_arrow_head(self, p, x_from, y_from, x_to, y_to, color):
    ang = math.atan2(y_to - y_from, x_to - x_from)
    size = 9.0
    pts = [QPointF(x_to, y_to),
           QPointF(x_to - size * math.cos(ang - 0.45),
                   y_to - size * math.sin(ang - 0.45)),
           QPointF(x_to - size * math.cos(ang + 0.45),
                   y_to - size * math.sin(ang + 0.45))]
    p.setPen(QPen(color, 1))
    p.setBrush(color)
    p.drawPolygon(QPolygonF(pts))
    p.setBrush(Qt.NoBrush)

  def mousePressEvent(self, event):
    m = self._macro_at_widget(event.position().x(),
                              event.position().y())
    if event.button() == Qt.RightButton:
      # Right-click (or double-click) toggles Click-to-Lock pinning.
      if m is not None and not m['soft']:
        self.pin_toggled.emit(m)
      return
    if m is not None:
      self._selected = m['name']
      if not m['soft']:
        self._drag_name = m['name']
        self._drag_start = (event.position().x(),
                            event.position().y())
        self._drag_xy = (m['x'], m['y'])
        self._dragging = False
      self.update()
      self.macro_clicked.emit(m)
    super().mousePressEvent(event)

  def mouseDoubleClickEvent(self, event):
    m = self._macro_at_widget(event.position().x(),
                              event.position().y())
    if m is not None and not m['soft']:
      self.pin_toggled.emit(m)
    super().mouseDoubleClickEvent(event)

  def mouseMoveEvent(self, event):
    if self._drag_name and self._drag_start and self._tf:
      px, py = event.position().x(), event.position().y()
      if abs(px - self._drag_start[0]) + abs(py - self._drag_start[1]) > 6:
        self._dragging = True
        ox, oy, s, W, H = self._tf
        if s > 0:
          self._drag_xy = ((px - ox) / s, H - (py - oy) / s)
          self.update()
    super().mouseMoveEvent(event)

  def mouseReleaseEvent(self, event):
    if self._dragging and self._drag_name and self._drag_xy:
      self.macro_moved.emit(self._drag_name, float(self._drag_xy[0]),
                            float(self._drag_xy[1]))
    self._drag_name = None
    self._drag_start = None
    self._dragging = False
    self.update()
    super().mouseReleaseEvent(event)

  def paintEvent(self, _event):
    p = QPainter(self)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(self.rect(), QColor('#0d1117'))
    if self._view == 'Compare (Baseline vs. AI)':
      self._paint_compare(p)
      return
    frame = self._frame
    if not frame:
      p.setPen(QColor('#8b949e'))
      p.drawText(self.rect(), Qt.AlignCenter,
                 'Floorplan canvas\nImport or pick a design, then '
                 'Place → Run floorplan.\nMacros will land here live, '
                 'largest first (AlphaChip-style).')
      return
    W, H = frame['canvas']
    if W <= 0 or H <= 0:
      return
    margin = 30
    base = self._view.split(' +')[0]
    # Reserve a strip BELOW the die for the hierarchy legend so it
    # never covers macros on the chip floor itself.
    legend_reserve = 46 if base == 'Floorplan' else 0
    avail_h = max(self.height() - 2 * margin - legend_reserve, 40)
    s = min((self.width() - 2 * margin) / W, avail_h / H)
    ox = (self.width() - W * s) / 2
    oy = margin + (avail_h - H * s) / 2
    self._tf = (ox, oy, s, W, H)

    def X(x):
      return ox + x * s

    def Y(y):
      return oy + (H - y) * s

    p.fillRect(int(ox), int(oy), int(W * s), int(H * s),
               QColor('#0f172a'))
    cols, rows = frame['grid']
    show_wires = 'Wires' in self._view
    if base == 'Congestion' and self._congestion is not None:
      grid = self._congestion
      vmax = max((max(r) for r in grid), default=1.0) or 1.0
      cw, ch = W / max(cols, 1), H / max(rows, 1)
      for r, row in enumerate(grid):
        for c, v in enumerate(row):
          if v > 0:
            col = _heat_color(v / vmax)
            col.setAlpha(210)
            p.fillRect(int(X(c * cw)), int(Y((r + 1) * ch)),
                       max(1, int(cw * s)) + 1, max(1, int(ch * s)) + 1,
                       col)
    p.setPen(QPen(QColor(48, 54, 61, 120), 1))
    step_c = max(1, cols // 40) if cols > 40 else 1
    step_r = max(1, rows // 40) if rows > 40 else 1
    for c in range(step_c, cols, step_c):
      p.drawLine(int(X(c * W / cols)), int(oy),
                 int(X(c * W / cols)), int(oy + H * s))
    for r in range(step_r, rows, step_r):
      p.drawLine(int(ox), int(Y(r * H / rows)),
                 int(ox + W * s), int(Y(r * H / rows)))
    if show_wires:
      pen = QPen(QColor(230, 237, 243, 46), 1)
      p.setPen(pen)
      for x1, y1, x2, y2 in self._segments:
        p.drawLine(int(X(x1)), int(Y(y1)), int(X(x2)), int(Y(y2)))
    red_fill, red_pen = QColor(239, 68, 68, 46), QPen(
        QColor('#ef4444'), 1, Qt.DashLine)
    for (x0, y0, x1, y1) in list(frame.get('blockages', [])) \
            + list(self._keepouts):
      x0, x1 = max(0.0, min(x0, x1)), min(W, max(x0, x1))
      y0, y1 = max(0.0, min(y0, y1)), min(H, max(y0, y1))
      if x1 <= x0 or y1 <= y0:
        continue
      p.setPen(red_pen)
      p.fillRect(int(X(min(x0, x1))), int(Y(max(y0, y1))),
                 int(abs(x1 - x0) * s), int(abs(y1 - y0) * s), red_fill)
      p.drawRect(int(X(min(x0, x1))), int(Y(max(y0, y1))),
                 int(abs(x1 - x0) * s), int(abs(y1 - y0) * s))
    power = self._power
    if base == 'Thermal' and not power:
      power = {m['name']: m['w'] * m['h'] for m in frame['macros']
               if not m['soft']}
    pmax = max(power.values(), default=1.0) or 1.0
    font = QFont()
    font.setPointSize(7)
    p.setFont(font)
    for m in frame['macros']:
      mx, my = m['x'], m['y']
      if self._dragging and m['name'] == self._drag_name \
              and self._drag_xy:
        mx, my = self._drag_xy
      x, y, w, h = (X(mx - m['w'] / 2), Y(my + m['h'] / 2),
                    m['w'] * s, m['h'] * s)
      if base == 'Thermal' and not m['soft']:
        p.fillRect(int(x), int(y), int(w), int(h),
                   _heat_color((power.get(m['name'], 0.0) / pmax)
                               ** 0.5))
        p.setPen(QPen(QColor('#e2e8f0'), 1))
        p.drawRect(int(x), int(y), int(w), int(h))
      elif m['soft']:
        p.fillRect(int(x), int(y), int(w), int(h),
                   QColor(100, 116, 139, 52))
        p.setPen(QPen(QColor(148, 163, 184, 160), 1))
        p.drawRect(int(x), int(y), int(w), int(h))
      elif not m.get('placed', True):
        p.setPen(QPen(QColor(56, 189, 248, 90), 1, Qt.DashLine))
        p.drawRect(int(x), int(y), int(w), int(h))
      else:
        # Hierarchy color-coding by RTL module prefix.
        fill = QColor(m.get('fill', '#0284c7'))
        fill.setAlpha(185)
        p.fillRect(int(x), int(y), int(w), int(h), fill)
        p.setPen(QPen(QColor(m.get('edge', '#38bdf8')), 1))
        p.drawRect(int(x), int(y), int(w), int(h))
      if not m['soft'] and w > 16 and h > 10:
        # Pin-orientation notch on the pin-facing edge (rotated by
        # the optimizer's chosen orientation for this macro).
        deg = self._orientations.get(m['name'], m.get('orient', 0))
        edge = liveplace_lib.rotated_edge(m.get('notch', 'N'), deg)
        cx, cy = x + w / 2, y + h / 2
        tick = max(5.0, min(11.0, (w if edge in 'NS' else h) * 0.32))
        if edge == 'N':
          pts = [(cx - tick / 2, y), (cx + tick / 2, y)]
        elif edge == 'S':
          pts = [(cx - tick / 2, y + h), (cx + tick / 2, y + h)]
        elif edge == 'E':
          pts = [(x + w, cy - tick / 2), (x + w, cy + tick / 2)]
        else:
          pts = [(x, cy - tick / 2), (x, cy + tick / 2)]
        p.setPen(QPen(QColor(13, 17, 23, 220), 4))
        p.drawLine(int(pts[0][0]), int(pts[0][1]),
                   int(pts[1][0]), int(pts[1][1]))
        p.setPen(QPen(QColor('#ffffff'), 2))
        p.drawLine(int(pts[0][0]), int(pts[0][1]),
                   int(pts[1][0]), int(pts[1][1]))
      if not m['soft'] and (m['name'] in self._pinned_names
                            or m.get('pinned')):
        p.setPen(QPen(QColor('#f59e0b'), 2))
        p.drawRect(int(x), int(y), int(w), int(h))
        p.fillRect(int(x) + 2, int(y) + 2, 13, 13, QColor('#f59e0b'))
        p.setPen(QColor('#111827'))
        bold7 = QFont()
        bold7.setPointSize(8)
        bold7.setBold(True)
        p.setFont(bold7)
        p.drawText(int(x) + 4, int(y) + 12, 'P')
        p.setFont(font)
      if m['name'] == frame.get('current') and not m['soft']:
        p.setPen(QPen(QColor('#fbbf24'), 3))
        p.drawRect(int(x), int(y), int(w), int(h))
      if w > 52 and h > 12 and not m['soft']:
        p.setPen(QColor('#e6edf3'))
        p.drawText(int(x) + 2, int(y) + 2, int(w) - 4, int(h) - 4,
                   Qt.AlignCenter | Qt.TextWordWrap, m['name'])
    if self._highlight_segments:
      p.setPen(QPen(QColor(251, 191, 36, 215), 1.5))
      for x1, y1, x2, y2 in self._highlight_segments:
        p.drawLine(int(X(x1)), int(Y(y1)), int(X(x2)), int(Y(y2)))
    if self._selected:
      for m in frame['macros']:
        if m['name'] == self._selected:
          p.setPen(QPen(QColor('#ffffff'), 2))
          p.drawRect(int(X(m['x'] - m['w'] / 2)),
                     int(Y(m['y'] + m['h'] / 2)),
                     int(m['w'] * s), int(m['h'] * s))
          break
    for (x0, y0, x1, y1, label) in self._contour_boxes:
      rx, ry = int(X(min(x0, x1))), int(Y(max(y0, y1)))
      rw, rh = int(abs(x1 - x0) * s), int(abs(y1 - y0) * s)
      p.setPen(QPen(QColor(250, 204, 21, 70), 5))
      p.drawRect(rx, ry, rw, rh)
      p.setPen(QPen(QColor('#facc15'), 1.5, Qt.DashLine))
      p.drawRect(rx, ry, rw, rh)
      p.setPen(QColor('#facc15'))
      p.drawText(rx + 2, max(12, ry - 4), label)
    if self._crit_visible and len(self._crit_points) >= 2:
      pts = [(X(x), Y(y)) for x, y in self._crit_points]
      for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        # Manhattan elbow: horizontal, then vertical into the sink.
        p.setPen(QPen(QColor(255, 46, 166, 150), 5))
        p.drawLine(int(x1), int(y1), int(x2), int(y1))
        p.drawLine(int(x2), int(y1), int(x2), int(y2))
        p.setPen(QPen(QColor('#ffd700'), 2))
        p.drawLine(int(x1), int(y1), int(x2), int(y1))
        p.drawLine(int(x2), int(y1), int(x2), int(y2))
        self._draw_arrow_head(p, x2, y1, x2, y2, QColor('#ffd700'))
      for x, y in pts:
        p.setPen(QPen(QColor('#ff2ea6'), 1))
        p.setBrush(QColor('#ffd700'))
        p.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
      p.setBrush(Qt.NoBrush)
    p.setPen(QColor('#fbbf24'))
    for port in frame.get('ports', []):
      p.fillRect(int(X(port['x'])) - 2, int(Y(port['y'])) - 2, 4, 4,
                 QColor('#fbbf24'))
    p.setPen(QPen(QColor('#e8a33d'), 2))
    p.drawRect(int(ox), int(oy), int(W * s), int(H * s))
    p.setPen(QColor('#e6edf3'))
    cur = frame.get('current') or 'done'
    p.drawText(int(ox) + 6, int(oy) + 16,
               f"{self._view} · step {frame['step']}/{frame['total']}"
               f" · placing: {cur}")
    if base == 'Floorplan':
      # Hierarchy color legend BELOW the die (outside the chip
      # floor): one horizontal strip of swatch + label per group.
      present = {}
      for m in frame['macros']:
        if not m['soft']:
          present[m.get('group', 'default')] = True
      rows = [g for g in liveplace_lib.HIERARCHY_GROUPS
              if g[0] in present]
      if rows:
        legend_font = QFont()
        legend_font.setPointSize(7)
        p.setFont(legend_font)
        lx = int(ox)
        ly = int(oy + H * s) + 9
        max_x = self.width() - 10
        for _key, label, fill, edge in rows:
          text_w = p.fontMetrics().horizontalAdvance(label)
          if lx + 16 + text_w > max_x and lx > int(ox):
            lx = int(ox)
            ly += 15
          p.fillRect(lx, ly, 10, 10, QColor(fill))
          p.setPen(QPen(QColor(edge), 1))
          p.drawRect(lx, ly, 10, 10)
          p.setPen(QColor('#e6edf3'))
          p.drawText(lx + 14, ly + 9, label)
          lx += 16 + text_w + 14


class ParetoWidget(QWidget):
  """Before vs. After KPI bar chart (Pareto view) for Results.

  One bar group per feasible run: Proxy Cost / Wirelength /
  Congestion, each normalized to its metric maximum (lower wins).
  """

  _METRICS = (('proxy_cost', 'Proxy', '#e8a33d'),
              ('wirelength', 'Wirelength', '#38bdf8'),
              ('congestion', 'Congestion', '#e879f9'))

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setMinimumHeight(170)
    self._rows = []

  def set_data(self, rows):
    self._rows = [dict(r) for r in (rows or [])
                  if r.get('proxy_cost', -1) > 0]
    self.update()

  def paintEvent(self, _event):
    p = QPainter(self)
    p.fillRect(self.rect(), QColor('#0d1117'))
    p.setPen(QPen(QColor('#30363d'), 1))
    p.drawRect(0, 0, self.width() - 1, self.height() - 1)
    title = QFont()
    title.setPointSize(9)
    title.setBold(True)
    p.setFont(title)
    p.setPen(QColor('#e6edf3'))
    p.drawText(8, 16, 'Before vs. After KPI — Pareto Chart '
                      '(Proxy / Wirelength / Congestion, lower is better)')
    if not self._rows:
      small = QFont()
      small.setPointSize(8)
      p.setFont(small)
      p.setPen(QColor('#8b949e'))
      p.drawText(8, 38, 'No feasible runs yet — Run floorplan or '
                        '⚡ Auto-Optimize to populate the trade-off chart.')
      return
    lx = self.width() - 250
    small = QFont()
    small.setPointSize(7)
    p.setFont(small)
    for i, (_key, label, color) in enumerate(self._METRICS):
      bx = lx + i * 82
      p.fillRect(bx, 7, 10, 10, QColor(color))
      p.setPen(QColor('#e6edf3'))
      p.drawText(bx + 13, 16, label)
    top, bottom = 26, self.height() - 30
    plot_h = max(bottom - top, 20)
    left = 44
    plot_w = max(self.width() - left - 10, 40)
    maxima = {k: max(r.get(k, 0.0) for r in self._rows) or 1.0
              for k, _l, _c in self._METRICS}
    n = len(self._rows)
    group_w = plot_w / max(n, 1)
    bar_w = max(4.0, group_w / 4.5)
    for i, row in enumerate(self._rows):
      gx = left + i * group_w
      for j, (key, _label, color) in enumerate(self._METRICS):
        v = float(row.get(key, 0.0))
        bh = plot_h * v / maxima[key]
        bx = gx + 4 + j * (bar_w + 2)
        p.fillRect(int(bx), int(bottom - bh), int(bar_w), int(bh),
                   QColor(color))
        p.setPen(QColor('#e6edf3'))
        if bh > 14:
          p.drawText(int(bx) - 2, int(bottom - bh) - 3,
                     f'{v:.2f}')
      p.setPen(QColor('#8b949e'))
      label = f"#{i + 1} {str(row.get('method', ''))[:14]}"
      p.drawText(int(gx) + 2, bottom + 12, label)
    p.setPen(QPen(QColor(48, 54, 61, 140), 1))
    p.drawLine(left, bottom, left + int(plot_w), bottom)


class MainWindow(QMainWindow):

  def __init__(self):
    super().__init__()
    self.setWindowTitle('NeuroSilicon — AI Floorplanner')
    # Fit the actual screen (minus the shell's top bar/dock) instead
    # of a fixed 1440x900 that overflows smaller work areas.
    self.setMinimumSize(1024, 640)
    screen = QApplication.primaryScreen()
    if screen is not None:
      avail = screen.availableGeometry()
      self.resize(max(1024, min(1440, avail.width() - 16)),
                  max(640, min(900, avail.height() - 36)))
      self.move(avail.x() + max(0, (avail.width() - self.width()) // 2),
                avail.y() + max(0, (avail.height() - self.height()) // 2))
    else:
      self.resize(1280, 800)
    self.setStyleSheet(DARK)
    self.setAcceptDrops(True)
    self._settings = settings_lib.load()
    self._thread = None
    self._worker = None
    self._designs = {}
    self._results = []
    self._frames = []
    self._frame_idx = 0
    self._plc_pair = None
    self._current_plc = None
    self._best_plc_path = ''
    self._outdir = os.path.join(project_lib.repo_root(), 'logs',
                                'desktop_runs')
    self._disc = None
    self._job_kind = ''
    self._job_t0 = 0.0
    self._last_frame_show = 0.0
    self._pending_load = None
    self._prev_best = None
    self._prev_design = ''
    self._thermal_fix = False
    self._keepouts = []       # [(x0,y0,x1,y1)] chat keep-out rects
    self._keepout_mode = 'block'
    self._compare_left = None   # baseline snapshot for Compare view
    self._compare_right = None  # best AI snapshot for Compare view
    self._run_snapshots = []    # scored-run snapshots, in run order
    self._crit_plc = None
    self._crit_result = None
    self._pinned = {}           # {macro: (x, y)} Click-to-Lock pins
    self._orientations = {}     # {macro: degrees} orientation flips
    self._tech_key = tech_lib.DEFAULT_NODE
    self._t0 = None
    self._build_ui()
    self.refresh_designs()
    self._refresh_tool_status()
    self._refresh_ai_status()
    self._on_tech_changed(self._tech_key, 'init')
    self._update_pin_label()
    self.log('Ready. Import a whole chip design (Import tab, or drop '
             'a folder/file/.zip here), pick a design, then Place → '
             'Run floorplan and watch the macros land live.')

  # -- layout ---------------------------------------------------------
  def _build_ui(self):
    menu = self.menuBar()
    file_menu = menu.addMenu('&File')
    for label, slot in (
        ('Import &chip design…', lambda: self.tabs.setCurrentIndex(0)),
        ('Download placed &.plc…', self._save_plc),
        ('Download &DEF…', self._save_def),
        ('Download report &ZIP…', self._save_zip),
        ('&Open output folder', self.open_outdir),
        ('E&xit', self.close)):
      act = QAction(label, self)
      act.triggered.connect(slot)
      file_menu.addAction(act)
    run_act = QAction('&Run floorplan', self)
    run_act.triggered.connect(self.run_floorplan)
    menu.addMenu('&Run').addAction(run_act)
    toggle_act = QAction('Toggle &designs panel', self)
    toggle_act.triggered.connect(lambda: self._toggle_left())
    menu.addMenu('&View').addAction(toggle_act)
    help_act = QAction('AI &setup guide', self)
    help_act.triggered.connect(
        lambda: self.right_tabs.setCurrentIndex(2))
    menu.addMenu('&Help').addAction(help_act)
    # Top button bar: panel toggles + internal log terminal.
    bar = self.addToolBar('Panels')
    bar.setMovable(False)
    for label, tip, slot in (
        ('Designs', 'Show/hide the designs panel', self._toggle_left),
        ('AI Panel', 'Show/hide the AI assistant panel',
         self._toggle_right),
        ('Log Terminal', 'Show the internal log terminal',
         self._show_log)):
      act = QAction(label, self)
      act.setToolTip(tip)
      act.triggered.connect(lambda _=0, s=slot: s())
      bar.addAction(act)

    central = QWidget()
    self.setCentralWidget(central)
    outer = QHBoxLayout(central)
    split = QSplitter(Qt.Horizontal)
    outer.addWidget(split)

    left = QWidget()
    left.setMinimumWidth(220)
    lv = QVBoxLayout(left)
    header = QHBoxLayout()
    header.addWidget(QLabel('Designs'))
    header.addStretch(1)
    self.panel_hide_btn = QPushButton('◀ Hide')
    self.panel_hide_btn.setToolTip('Hide the designs panel')
    self.panel_hide_btn.clicked.connect(
        lambda: self._set_left_visible(False))
    header.addWidget(self.panel_hide_btn)
    lv.addLayout(header)
    self.design_list = QListWidget()
    self.design_list.currentTextChanged.connect(self._design_picked)
    self.design_list.itemClicked.connect(self._design_clicked)
    lv.addWidget(self.design_list, 1)
    self.design_info = QLabel('No design selected.')
    self.design_info.setWordWrap(True)
    lv.addWidget(self.design_info)
    self.tool_status = QLabel('')
    self.tool_status.setWordWrap(True)
    lv.addWidget(self.tool_status)
    self.ai_status = QLabel('')
    self.ai_status.setWordWrap(True)
    lv.addWidget(self.ai_status)
    self.design_panel = left
    split.addWidget(left)

    center = QWidget()
    center.setMinimumWidth(560)
    cv = QVBoxLayout(center)
    cards = QHBoxLayout()
    self.card_proxy = QLabel('proxy\n—')
    self.card_wl = QLabel('wirelength\n—')
    self.card_cong = QLabel('congestion\n—')
    self.card_macros = QLabel('macros placed\n—')
    self.card_fmax = QLabel('est. Fmax\n—')
    self.card_fmax.setToolTip(
        'Estimated Fmax from the critical timing path: 1000 / '
        'critical delay (ns). Manhattan wire delay + macro logic '
        'delay proxy — not signoff STA.')
    for card in (self.card_proxy, self.card_wl, self.card_cong,
                 self.card_macros, self.card_fmax):
      card.setObjectName('card')
      card.setAlignment(Qt.AlignCenter)
      cards.addWidget(card, 1)
    cv.addLayout(cards)
    self.canvas = ChipCanvas()
    self.canvas.macro_clicked.connect(self._macro_clicked)
    self.canvas.pin_toggled.connect(self._toggle_pin)
    self.canvas.macro_moved.connect(self._move_macro)
    cv.addWidget(self.canvas, 3)
    prow = QHBoxLayout()
    prow.setSpacing(6)
    self.left_toggle = QPushButton('☰')
    self.left_toggle.setToolTip(
        'Show/hide the designs panel (auto-hides during runs)')
    self.left_toggle.clicked.connect(lambda: self._toggle_left())
    prow.addWidget(self.left_toggle)
    self.play_btn = QPushButton('▶ Play')
    self.play_btn.clicked.connect(self._toggle_play)
    self.step_btn = QPushButton('Step ›')
    self.step_btn.clicked.connect(
        lambda: self._show_frame(self._frame_idx + 1))
    self.scrub = QSlider(Qt.Horizontal)
    self.scrub.setRange(0, 0)
    self.scrub.valueChanged.connect(self._show_frame)
    self.speed_combo = QComboBox()
    self.speed_combo.addItems(['Slow', 'Normal', 'Fast'])
    self.speed_combo.setCurrentText('Normal')
    self.speed_combo.setMaximumWidth(92)
    self.speed_combo.setToolTip('Playback speed')
    self.speed_combo.currentTextChanged.connect(
        lambda _t: self._play_timer.setInterval(self._speed_ms()))
    self.view_combo = QComboBox()
    self.view_combo.addItems(list(ChipCanvas.VIEWS))
    self.view_combo.setMaximumWidth(148)
    self.view_combo.setToolTip('Canvas view')
    self.view_combo.currentTextChanged.connect(self._view_changed)
    self.frame_label = QLabel('—')
    self.frame_label.setMaximumWidth(170)
    self.sam_check = QCheckBox('SAM')
    self.sam_check.setToolTip(
        'Highlight Bottleneck Zones (SAM Contour): segment the '
        'top 15% hottest / most-congested regions on Thermal or '
        'Congestion views')
    self.sam_check.stateChanged.connect(
        lambda _s: self._update_contours())
    self.crit_check = QCheckBox('Show Critical Path')
    self.crit_check.setToolTip(
        'Draw the critical timing path (gold/magenta directed '
        'arrows, Manhattan-routed) across its macros. Delay = wire '
        'delay + macro logic delay; see the est. Fmax card.')
    self.crit_check.stateChanged.connect(
        lambda _s: self._update_timing())
    prow.addWidget(self.play_btn)
    prow.addWidget(self.step_btn)
    prow.addWidget(self.scrub, 1)
    prow.addWidget(self.speed_combo)
    prow.addWidget(self.view_combo)
    prow.addWidget(self.sam_check)
    prow.addWidget(self.crit_check)
    prow.addWidget(self.frame_label)
    cv.addLayout(prow)
    self.tabs = QTabWidget()
    cv.addWidget(self.tabs, 2)
    split.addWidget(center)

    right = QWidget()
    right.setMinimumWidth(400)
    rv = QVBoxLayout(right)
    rheader = QHBoxLayout()
    rheader.addWidget(QLabel('Run status'))
    rheader.addStretch(1)
    self.right_hide_btn = QPushButton('Hide ▶')
    self.right_hide_btn.setToolTip('Hide the AI / status panel')
    self.right_hide_btn.clicked.connect(
        lambda: self._set_right_visible(False))
    rheader.addWidget(self.right_hide_btn)
    rv.addLayout(rheader)
    self.progress = QProgressBar()
    self.progress.setRange(0, 100)
    rv.addWidget(self.progress)
    self.elapsed = QLabel('idle')
    rv.addWidget(self.elapsed)
    self.best_label = QLabel('best proxy: —')
    rv.addWidget(self.best_label)
    self.progress_label = QLabel('idle')
    self.progress_label.setWordWrap(True)
    # Fixed height keeps wrapped status text from resizing the
    # panel vertically while the splitter is dragged (wobble).
    self.progress_label.setMinimumHeight(30)
    rv.addWidget(self.progress_label)
    self.right_tabs = QTabWidget()
    rv.addWidget(self.right_tabs, 1)
    self.right_panel = right
    split.addWidget(right)
    # Stable splitter: only the center stretches, panels never
    # collapse while dragging (no snap-back / wobble at the limits).
    split.setChildrenCollapsible(False)
    split.setHandleWidth(6)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setStretchFactor(2, 0)
    split.setSizes([260, 760, 440])

    self.tabs.addTab(self._import_tab(), 'Import')
    self.tabs.addTab(self._place_tab(), 'Place')
    self.tabs.addTab(self._results_tab(), 'Results')
    self.tabs.addTab(self._thermal_tab(), 'Thermal')
    self.tabs.currentChanged.connect(self._tab_changed)
    self.right_tabs.addTab(self._chat_tab(), 'Neuro AI')
    self.right_tabs.addTab(self._connectors_tab(), 'Connectors')
    self.right_tabs.addTab(self._guide_tab(), 'Guide')
    self.right_tabs.addTab(self._settings_tab(), 'Settings')
    for tw in (self.tabs, self.right_tabs):
      tw.tabBar().setElideMode(Qt.ElideRight)
      # Scroll buttons keep tab titles readable ("Assistant",
      # "Connectors", ...) instead of eliding to "As… Co…" when
      # the panel is narrow.
      tw.tabBar().setUsesScrollButtons(True)
    logw = QWidget()
    logv = QVBoxLayout(logw)
    self.log_view = QPlainTextEdit()
    self.log_view.setReadOnly(True)
    logv.addWidget(self.log_view)
    self.right_tabs.addTab(logw, 'Log')
    self._timer = QTimer(self)
    self._timer.timeout.connect(self._tick)
    self._snap_timer = QTimer(self)
    self._snap_timer.setSingleShot(True)
    self._snap_timer.timeout.connect(self._flush_snap)
    self._play_timer = QTimer(self)
    self._play_timer.setInterval(self._speed_ms())
    self._play_timer.timeout.connect(
        lambda: self._show_frame(self._frame_idx + 1, from_timer=True))
    self.statusBar().showMessage('Ready')

  def _speed_ms(self):
    return {'Slow': 450, 'Normal': 140, 'Fast': 45}.get(
        self.speed_combo.currentText() if hasattr(self, 'speed_combo')
        else 'Normal', 140)

  # -- tabs -----------------------------------------------------------
  def _import_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    desc = QLabel(
        'Select the WHOLE chip design — a project folder, any file '
        'inside it (.v/.sv, netlist.pb.txt, .plc), or a .zip — or '
        'drag-and-drop it anywhere on this window. NeuroSilicon '
        'auto-fetches what floorplanning needs: a ready netlist pair '
        'places directly; RTL gets an auto-picked top and yosys '
        'synthesis. No mode to choose.')
    desc.setWordWrap(True)
    v.addWidget(desc)
    self.imp_path = QLineEdit()
    self.imp_path.setPlaceholderText(
        '/path/to/chip-folder  (or a file inside it, or design.zip)')
    self.imp_path.setMinimumWidth(80)
    v.addWidget(self.imp_path)
    row = QHBoxLayout()
    for label, slot in (('Folder…', lambda: self._browse_pick('folder')),
                        ('File…', lambda: self._browse_pick('file')),
                        ('ZIP…', lambda: self._browse_pick('zip')),
                        ('Auto-detect', self.detect_import)):
      b = QPushButton(label)
      b.clicked.connect(slot)
      row.addWidget(b)
    row.addStretch(1)
    v.addLayout(row)
    self.disc_view = QPlainTextEdit()
    self.disc_view.setReadOnly(True)
    self.disc_view.setPlaceholderText(
        'Auto-detect results: found files, missing pieces, import plan.')
    v.addWidget(self.disc_view, 1)
    form = QFormLayout()
    self.imp_top = QComboBox()
    self.imp_top.setEditable(True)
    form.addRow('Top module (auto-picked)', self.imp_top)
    self.imp_name = QLineEdit()
    form.addRow('Project name', self.imp_name)
    self.imp_params = QLineEdit()
    self.imp_params.setPlaceholderText('e.g. MEM_WORDS=128,ENABLE_DIV=1')
    form.addRow('Parameters (optional)', self.imp_params)
    self.imp_defines = QLineEdit()
    self.imp_defines.setPlaceholderText('e.g. XILINX_FPGA')
    form.addRow('Verilog defines (optional)', self.imp_defines)
    for edit in (self.imp_name, self.imp_params, self.imp_defines):
      edit.setMinimumWidth(80)
    self.imp_episodes = QSpinBox()
    self.imp_episodes.setRange(1, 5)
    self.imp_episodes.setValue(1)
    form.addRow('Baseline episodes after import', self.imp_episodes)
    v.addLayout(form)
    v.addWidget(QLabel('RTL files to synthesize (uncheck to exclude)'))
    self.file_list = QListWidget()
    v.addWidget(self.file_list, 1)
    brow = QHBoxLayout()
    self.imp_go_btn = QPushButton('Import design')
    self.imp_go_btn.setObjectName('primary')
    self.imp_go_btn.clicked.connect(self.do_import)
    brow.addWidget(self.imp_go_btn)
    brow.addWidget(_wrap_label('Ready netlists register instantly; RTL '
                          'synthesizes, then baselines run.'))
    v.addLayout(brow)
    self.rtl_btn_import = QPushButton('Suggest RTL Fix for Bottleneck')
    self.rtl_btn_import.setToolTip(
        'Muse co-pilot: find the macro/module causing the worst '
        'thermal or routing congestion (e.g. a25_dcache), read its .v '
        'snippet, and propose an RTL fix (banking / pipelining / '
        'clock gating) in the Assistant pane. Works offline.')
    self.rtl_btn_import.clicked.connect(self._suggest_rtl_fix)
    v.addWidget(self.rtl_btn_import)
    # Scrollable vertically only: no sideways overflow, ever.
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(w)
    return scroll

  def _place_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(_wrap_label(
        'AlphaChip-style engine: hard macros place sequentially, '
        'largest first, each on a legal grid cell; standard cells then '
        'relax (force-directed) and the episode is scored on '
        'wirelength + congestion + density. Watch it live on the '
        'canvas, then replay with Play / scrub.'))
    form = QFormLayout()
    self.run_animate = QCheckBox('Animate placement live (real-time)')
    self.run_animate.setChecked(True)
    form.addRow('Simulation', self.run_animate)
    self.run_method = QComboBox()
    self.run_method.addItems(['first_legal', 'random'])
    form.addRow('Live rollout policy', self.run_method)
    self.run_episodes = QSpinBox()
    self.run_episodes.setRange(1, 10)
    self.run_episodes.setValue(1)
    form.addRow('Baseline episodes per method', self.run_episodes)
    self.run_fd = QSpinBox()
    self.run_fd.setRange(0, 100)
    self.run_fd.setValue(10)
    form.addRow('FD relax steps per stage', self.run_fd)
    self.run_seed = QSpinBox()
    self.run_seed.setRange(0, 99999)
    form.addRow('Random seed', self.run_seed)
    self.run_finetune = QSpinBox()
    self.run_finetune.setRange(0, 50)
    form.addRow('Finetune epochs (coordinate descent)', self.run_finetune)
    self.run_polish = QCheckBox(
        'Post-Placement Polish (200-step Simulated Annealing macro swap)')
    self.run_polish.setChecked(True)
    self.run_polish.setToolTip(
        'After placement, 200 SA steps swap compatible (same-size) '
        'macros, flip orientations, and nudge positions to cut '
        'pin-aware HPWL and congestion. Pinned macros never move. '
        'The measured HPWL gain is logged per run.')
    form.addRow('SA polish', self.run_polish)
    self.run_orient = QCheckBox(
        'Optimize macro orientation flips (0°/90°/180°/270°, HPWL)')
    self.run_orient.setChecked(True)
    self.run_orient.setToolTip(
        'Tests every hard macro at 0/90/180/270 degrees against '
        'pin-aware HPWL and keeps the best flip per macro. The '
        'pin-facing notch on canvas rotates with the choice; the '
        'choice exports as R0/R90/R180/R270 in OpenROAD Tcl.')
    form.addRow('Orientation', self.run_orient)
    v.addLayout(form)
    self.pin_label = QLabel('🔒 Pinned macros: none — right-click or '
                            'double-click a macro to lock it, or drag '
                            'it to a grid tile.')
    self.pin_label.setWordWrap(True)
    v.addWidget(self.pin_label)
    clear_pins = QPushButton('Clear pinned macros')
    clear_pins.clicked.connect(self._clear_pins)
    v.addWidget(clear_pins)
    self.constraint_label = QLabel('Chat constraint: none')
    self.constraint_label.setWordWrap(True)
    v.addWidget(self.constraint_label)
    clear = QPushButton('Clear chat constraint')
    clear.clicked.connect(self._clear_constraint)
    v.addWidget(clear)
    self.auto_btn_place = QPushButton('⚡ Auto-Optimize (AI Loop)')
    self.auto_btn_place.setStyleSheet(
        'background:#9e6a1e; border:1px solid #facc15; '
        'font-weight:700; padding:7px 12px;')
    self.auto_btn_place.setToolTip(
        'Autonomous 3-iteration PPA loop, no typing needed: '
        '(1) baseline placement + proxy/wirelength/RUDY/thermal, '
        '(2) SAM hotspot + muse-arcadia-1.0 diagnosis into JSON '
        'constraints, (3) AI-guided re-placement. Both runs land in '
        'Results; the PPA Improvement Summary posts in Assistant.')
    self.auto_btn_place.clicked.connect(self.auto_optimize)
    v.addWidget(self.auto_btn_place)
    self.run_btn = QPushButton('Run floorplan')
    self.run_btn.setObjectName('primary')
    self.run_btn.clicked.connect(self.run_floorplan)
    v.addWidget(self.run_btn)
    v.addStretch(1)
    return w

  def _results_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    self.details_view = QPlainTextEdit()
    self.details_view.setReadOnly(True)
    self.details_view.setMaximumHeight(150)
    self.details_view.setPlaceholderText(
        'Design & run details: canvas, macros, wirelength, '
        'congestion, density, thermal hotspots…')
    v.addWidget(self.details_view)
    self.result_table = QTableWidget(0, 7)
    self.result_table.setHorizontalHeaderLabels(
        ['method', 'ep', 'proxy', 'wirelength', 'congestion',
         'density', 'seconds'])
    v.addWidget(self.result_table, 1)
    self.delta_label = QLabel('')
    self.delta_label.setWordWrap(True)
    v.addWidget(self.delta_label)
    self.pareto = ParetoWidget()
    v.addWidget(self.pareto)
    grid = QHBoxLayout()
    for label, slot in (
        ('Download placed .plc…', self._save_plc),
        ('Download DEF…', self._save_def),
        ('Download OpenROAD .tcl', self._save_tcl),
        ('Export Timelapse GIF', self._save_gif),
        ('Floorplan PNG…', lambda: self._save_png('floorplan')),
        ('Thermal PNG…', lambda: self._save_png('thermal')),
        ('Congestion PNG…', lambda: self._save_png('congestion')),
        ('Report ZIP…', self._save_zip),
        ('Open output folder', self.open_outdir)):
      b = QPushButton(label)
      b.clicked.connect(slot)
      grid.addWidget(b)
    v.addLayout(grid)
    v.addWidget(_wrap_label('Placed .plc opens in other EDA tools; DEF is '
                       'viewer-grade for KLayout / OpenROAD. The ZIP '
                       'bundles the run folder (results, placements, '
                       'figures).'))
    return w

  def _thermal_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(_wrap_label(
        'Thermal imaging — pick a design, drag the corner sliders, '
        'and the canvas Thermal view repaints instantly. Model: '
        'P ∝ macro area × (V/1.8)² × (f/100MHz) × (activity/0.2). '
        'Documented area-proportional proxy, not signoff power.'))
    form = QFormLayout()
    self.tech_combo = QComboBox()
    for key, node in tech_lib.TECH_NODES.items():
      self.tech_combo.addItem(node['label'], key)
    self.tech_combo.setCurrentIndex(0)
    self.tech_combo.setToolTip(
        'Technology Node preset: rescales voltage, default clock, '
        'static leakage / temperature, and the routing-pitch '
        'congestion multiplier. Thermal and Congestion views update '
        'instantly. Documented proxies, not signoff PDK data.')
    self.tech_combo.currentIndexChanged.connect(
        lambda _i: self._on_tech_changed(
            self.tech_combo.currentData(), 'thermal'))
    form.addRow('Technology Node', self.tech_combo)
    self.tech_label = QLabel('')
    self.tech_label.setWordWrap(True)
    form.addRow('', self.tech_label)
    self.th_clock = QSlider(Qt.Horizontal)
    self.th_clock.setRange(10, 1200)
    self.th_clock.setValue(100)
    self.th_volt = QSlider(Qt.Horizontal)
    self.th_volt.setRange(50, 350)
    self.th_volt.setValue(180)
    self.th_act = QSlider(Qt.Horizontal)
    self.th_act.setRange(5, 100)
    self.th_act.setValue(20)
    for slider, label in ((self.th_clock, 'Clock (MHz)'),
                          (self.th_volt, 'Voltage (×0.01 V)'),
                          (self.th_act, 'Activity (%)')):
      slider.valueChanged.connect(self._thermal_update)
      form.addRow(label, slider)
    v.addLayout(form)
    thermal_view_btn = QPushButton('Show Thermal view on canvas')
    thermal_view_btn.clicked.connect(
        lambda: self.view_combo.setCurrentText('Thermal'))
    v.addWidget(thermal_view_btn)
    self.th_label = QLabel('Pick a design to see its heat map.')
    self.th_label.setWordWrap(True)
    v.addWidget(self.th_label)
    v.addStretch(1)
    return w

  def _chat_tab(self):
    """Modular chat: header (status + Clear) / bubbles / quick asks /
    composer. Messages are individual bubble widgets in a scroll
    column — user right, assistant left — not one raw text dump."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setSpacing(6)
    header = QHBoxLayout()
    title = QLabel('Neuro AI')
    title.setStyleSheet('font-weight:600; font-size:13px;')
    header.addWidget(title)
    self.chat_status = QLabel('AI: off')
    self.chat_status.setStyleSheet(
        'background:#21262d; border:1px solid #30363d; '
        'border-radius:8px; padding:2px 8px; color:#8b949e;')
    header.addWidget(self.chat_status)
    header.addStretch(1)
    clear = QPushButton('Clear chat')
    clear.setToolTip('Remove all messages and start fresh')
    clear.clicked.connect(self._chat_clear)
    header.addWidget(clear)
    v.addLayout(header)
    v.addWidget(_wrap_label(
        'Reads the current floorplan through MCP-style tools '
        '(design info, run results, thermal, constraints). Works '
        'offline; enable AI Connectors for LLM answers.'))
    self.chat_scroll = QScrollArea()
    self.chat_scroll.setWidgetResizable(True)
    self.chat_scroll.setHorizontalScrollBarPolicy(
        Qt.ScrollBarAlwaysOff)
    self.chat_area = QWidget()
    self.chat_area.setStyleSheet('background:#0b141a;')
    self.chat_scroll.setStyleSheet('background:#0b141a; border:none;')
    self.chat_msgs = QVBoxLayout(self.chat_area)
    self.chat_msgs.setSpacing(8)
    self.chat_msgs.addStretch(1)
    self.chat_scroll.setWidget(self.chat_area)
    v.addWidget(self.chat_scroll, 1)
    self._chat_append(
        'Assistant',
        'Pick a design and run Place, then ask me to explain the '
        'result — or type a constraint like "Keep a 20% routing '
        'channel open in the center". Ask "what is connected?" '
        'anytime to see your AI setup.')
    self._chat_suggestions = [
        'Explain the best run',
        'Why is congestion high?',
        'Which macros run hottest?',
        'What is connected?',
        'Keep a 20% routing channel open in the center',
        'Spread the hot macros apart',
        'Keep a routing channel at the edge',
        'What should I try next to improve placement?',
    ]
    self.chat_suggest = QComboBox()
    self.chat_suggest.setToolTip(
        'Pick a suggested question or placement constraint — it is '
        'asked immediately. You can also just type in the box below: '
        'suggestions appear as you type.')
    self.chat_suggest.setStyleSheet(
        'QComboBox { background:#21262d; border:1px solid #30363d; '
        'border-radius:8px; padding:4px 8px; color:#e6edf3; }')
    self.chat_suggest.addItem(
        'Suggestions — pick a question or constraint…')
    self.chat_suggest.addItems(self._chat_suggestions)
    self.chat_suggest.activated.connect(self._on_suggestion_picked)
    v.addWidget(self.chat_suggest)
    self.chat_import_btn = QPushButton('Import project folder…')
    self.chat_import_btn.setToolTip(
        'Import a chip project from here: pick the project folder — '
        'NeuroSilicon auto-fetches the RTL (.v/.sv) project files '
        'inside it, auto-picks the top module, and opens them in '
        'the Import tab. Ready netlists register instantly. You '
        'can also type: import <folder>')
    self.chat_import_btn.clicked.connect(self._chat_import_project)
    v.addWidget(self.chat_import_btn)
    self.apply_fix_btn = QPushButton('Apply AI Fix & Re-Place')
    self.apply_fix_btn.setObjectName('primary')
    self.apply_fix_btn.setToolTip(
        'Re-run placement with the active chat constraint + '
        'thermal/congestion weights; new row lands in Results with '
        'Δ% vs the previous run')
    self.apply_fix_btn.clicked.connect(self._apply_ai_fix)
    v.addWidget(self.apply_fix_btn)
    self.auto_btn_chat = QPushButton('⚡ Auto-Optimize (AI Loop)')
    self.auto_btn_chat.setStyleSheet(
        'background:#9e6a1e; border:1px solid #facc15; '
        'font-weight:700; padding:7px 12px;')
    self.auto_btn_chat.setToolTip(
        'Autonomous 3-iteration PPA loop: baseline → SAM + '
        'muse-arcadia-1.0 diagnosis → AI-guided re-placement. '
        'The PPA Improvement Summary posts here in chat.')
    self.auto_btn_chat.clicked.connect(self.auto_optimize)
    v.addWidget(self.auto_btn_chat)
    self.rtl_btn_chat = QPushButton('Suggest RTL Fix for Bottleneck')
    self.rtl_btn_chat.setToolTip(
        'Muse co-pilot: propose an RTL fix (memory banking, bus '
        'pipelining, or clock gating) for the hottest / most '
        'congested macro, from its imported .v source.')
    self.rtl_btn_chat.clicked.connect(self._suggest_rtl_fix)
    v.addWidget(self.rtl_btn_chat)
    self.review_btn = QPushButton('Run 3-Agent Design Review')
    self.review_btn.setToolTip(
        'Thermal Specialist + Routing & Timing Specialist + Lead '
        'Physical Design Architect review the current floorplan and '
        'recommend Wirelength / Congestion / Thermal weights '
        '(applied to Settings automatically). Works offline.')
    self.review_btn.clicked.connect(self._run_agent_review)
    v.addWidget(self.review_btn)
    row = QHBoxLayout()
    row.setSpacing(6)
    self.chat_input = QLineEdit()
    self.chat_input.setPlaceholderText(
        'Ask about this floorplan, or state a constraint…')
    completer = QCompleter(self._chat_suggestions, self.chat_input)
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setFilterMode(Qt.MatchContains)
    completer.setCompletionMode(QCompleter.PopupCompletion)
    self.chat_input.setCompleter(completer)
    self.chat_input.setStyleSheet(
        'QLineEdit { background:#202c33; border:1px solid #202c33; '
        'border-radius:14px; padding:6px 12px; color:#e9edef; }')
    self.chat_input.returnPressed.connect(
        lambda: self._chat_send())
    send = QPushButton('Send')
    send.setObjectName('primary')
    send.setStyleSheet(
        'QPushButton { background:#00a884; border:1px solid #00a884; '
        'border-radius:14px; padding:6px 14px; color:#ffffff; '
        'font-weight:600; }')
    send.clicked.connect(lambda: self._chat_send())
    row.addWidget(self.chat_input, 1)
    row.addWidget(send)
    v.addLayout(row)
    return w

  def _bubble(self, who, text):
    """One WhatsApp-style bubble (user right-green / Neuro AI left)."""
    is_user = who == 'You'
    bubble = QFrame()
    bubble.setStyleSheet(
        'QFrame { background:#005c4b; border:1px solid #005c4b; '
        'border-radius:12px; }' if is_user else
        'QFrame { background:#202c33; border:1px solid #202c33; '
        'border-radius:12px; }')
    bl = QVBoxLayout(bubble)
    bl.setContentsMargins(10, 7, 10, 5)
    if not is_user:
      name = QLabel('Neuro AI')
      name.setStyleSheet(
          'background:transparent; color:#25d366; font-size:10px; '
          'font-weight:600;')
      bl.addWidget(name)
    body = QLabel(text)
    body.setWordWrap(True)
    body.setTextInteractionFlags(Qt.TextSelectableByMouse)
    body.setMaximumWidth(460)
    body.setStyleSheet('background:transparent; color:#e9edef;')
    bl.addWidget(body)
    stamp = QLabel(time.strftime('%H:%M'))
    stamp.setAlignment(Qt.AlignRight)
    stamp.setStyleSheet(
        'background:transparent; color:#8696a0; font-size:9px;')
    bl.addWidget(stamp)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    if is_user:
      row.addStretch(1)
      row.addWidget(bubble)
    else:
      row.addWidget(bubble)
      row.addStretch(1)
    holder = QWidget()
    holder.setLayout(row)
    return holder

  def _chat_clear(self):
    while self.chat_msgs.count() > 1:
      child = self.chat_msgs.takeAt(0)
      if child.widget():
        child.widget().deleteLater()
    self._chat_append('Assistant', 'Chat cleared. Ask me about this '
                                 'floorplan, thermals, congestion, '
                                 'or a placement constraint.')

  def _connectors_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(_wrap_label(
        'Optional AI connectors. Off by default; keys stay in '
        '~/.config/neurosilicon/settings.json, never in the repo. '
        'See the Guide tab for step-by-step setup.'))
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    llm = self._settings['llm']
    self.ai_provider = QComboBox()
    self.ai_provider.addItems(['off', 'api', 'local'])
    self.ai_provider.setCurrentText(llm.get('provider', 'off'))
    form.addRow('Provider', self.ai_provider)
    self.ai_key = QLineEdit(llm.get('api_key', ''))
    self.ai_key.setEchoMode(QLineEdit.Password)
    self.ai_key.setPlaceholderText('API key (cloud API only)')
    form.addRow('API key', self.ai_key)
    self.ai_base = QLineEdit(llm.get('base_url', ''))
    form.addRow('API base URL', self.ai_base)
    self.ai_model = QLineEdit(llm.get('model', ''))
    form.addRow('API model', self.ai_model)
    self.ai_local_url = QLineEdit(llm.get('local_url', ''))
    form.addRow('Local server URL', self.ai_local_url)
    self.ai_local_model = QLineEdit(llm.get('local_model', ''))
    form.addRow('Local model', self.ai_local_model)
    self.ai_timeout = QSpinBox()
    self.ai_timeout.setRange(5, 600)
    self.ai_timeout.setValue(int(llm.get('timeout', 60)))
    form.addRow('Request timeout (s)', self.ai_timeout)
    v.addLayout(form)
    self.ai_conn_status = QLabel('')
    self.ai_conn_status.setWordWrap(True)
    v.addWidget(self.ai_conn_status)
    row = QHBoxLayout()
    save = QPushButton('Save connectors')
    save.clicked.connect(self._save_connectors)
    test = QPushButton('Test connection')
    test.clicked.connect(self._test_connection)
    models_btn = QPushButton('List models')
    models_btn.setToolTip(
        'Ask the endpoint which exact model IDs your key can use')
    models_btn.clicked.connect(self._list_models)
    row.addWidget(save)
    row.addWidget(test)
    row.addWidget(models_btn)
    v.addLayout(row)
    self.models_scroll = QScrollArea()
    self.models_scroll.setWidgetResizable(True)
    self.models_scroll.setMaximumHeight(150)
    self.models_scroll.setVisible(False)
    self.models_widget = QWidget()
    self.models_layout = QVBoxLayout(self.models_widget)
    self.models_layout.addWidget(
        QLabel('Models from the endpoint — click one to use it:'))
    self.models_scroll.setWidget(self.models_widget)
    v.addWidget(self.models_scroll)
    v.addStretch(1)
    return w

  def _guide_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    browser = QTextBrowser()
    browser.setHtml(assistant_lib.GUIDE_HTML)
    v.addWidget(browser, 1)
    return w

  def _settings_tab(self):
    w = QWidget()
    v = QVBoxLayout(w)
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    self.set_fields = {}
    pdk = self._settings['pdk']
    for key, label in (('yosys', 'Yosys binary'),
                       ('openroad', 'OpenROAD binary'),
                       ('lef', 'Cell LEF (blank = sky130 auto)'),
                       ('liberty', 'Liberty .lib (blank = auto)'),
                       ('techlef', 'Tech LEF (blank = auto)')):
      edit = QLineEdit(pdk.get(key, ''))
      edit.setMinimumWidth(120)
      browse = QPushButton('Browse…')
      browse.setFixedWidth(78)
      browse.clicked.connect(lambda _=0, e=edit: self._browse_file(e))
      row = QHBoxLayout()
      row.addWidget(edit, 1)
      row.addWidget(browse)
      form.addRow(label, row)
      self.set_fields[key] = edit
    eng = self._settings['engine']
    self.eng_spins = {}
    for key, label in (('wirelength_weight', 'Wirelength weight'),
                       ('congestion_weight', 'Congestion weight'),
                       ('density_weight', 'Density weight'),
                       ('thermal_weight', 'Thermal weight')):
      spin = QDoubleSpinBox()
      spin.setRange(0.0, 10.0)
      spin.setSingleStep(0.1)
      spin.setValue(float(eng.get(key, 0.0)))
      form.addRow(label, spin)
      self.eng_spins[key] = spin
    self.set_checkpoint = QLineEdit(
        self._settings['model'].get('checkpoint', ''))
    form.addRow('PPO checkpoint (.pt, optional)', self.set_checkpoint)
    self.tech_combo_settings = QComboBox()
    for key, node in tech_lib.TECH_NODES.items():
      self.tech_combo_settings.addItem(node['label'], key)
    self.tech_combo_settings.setToolTip(
        'Same Technology Node preset as the Thermal tab (kept in '
        'sync): voltage, clock, leakage/temperature, congestion '
        'multiplier.')
    self.tech_combo_settings.currentIndexChanged.connect(
        lambda _i: self._on_tech_changed(
            self.tech_combo_settings.currentData(), 'settings'))
    form.addRow('Technology Node', self.tech_combo_settings)
    v.addLayout(form)
    save = QPushButton('Save settings')
    save.clicked.connect(self._save_settings)
    v.addWidget(save)
    self.set_status = QLabel('')
    self.set_status.setWordWrap(True)
    v.addWidget(self.set_status)
    v.addStretch(1)
    return w

  # -- settings / connectors ------------------------------------------
  def _browse_file(self, edit):
    path, _ = QFileDialog.getOpenFileName(self, 'Pick file')
    if path:
      edit.setText(path)

  def _save_settings(self):
    for key, edit in self.set_fields.items():
      self._settings['pdk'][key] = edit.text().strip()
    for key, spin in self.eng_spins.items():
      self._settings['engine'][key] = float(spin.value())
    self._settings['model']['checkpoint'] = \
        self.set_checkpoint.text().strip()
    settings_lib.save(self._settings)
    self._settings = settings_lib.load()
    self._refresh_tool_status()
    self.set_status.setText('Saved to ~/.config/neurosilicon/settings.json')
    self.log('Settings saved.')

  def _llm_form_settings(self):
    s = dict(self._settings['llm'])
    s['provider'] = self.ai_provider.currentText()
    s['api_key'] = self.ai_key.text().strip()
    s['base_url'] = self.ai_base.text().strip()
    s['model'] = self.ai_model.text().strip()
    s['local_url'] = self.ai_local_url.text().strip()
    s['local_model'] = self.ai_local_model.text().strip()
    s['timeout'] = self.ai_timeout.value()
    return s

  def _save_connectors(self):
    self._settings['llm'] = self._llm_form_settings()
    settings_lib.save(self._settings)
    self._settings = settings_lib.load()
    self._refresh_ai_status()
    self.ai_conn_status.setText('Connectors saved locally.')
    self.log('AI connectors saved (keys stay in your local settings).')
    # The chat panel explains what just got connected and how to use it.
    self._set_right_visible(True)
    self._chat_append(
        'Assistant',
        assistant_lib.connection_explanation(self._settings['llm']))

  def _refresh_ai_status(self):
    label, _ok, detail = assistant_lib.connector_status(
        self._settings['llm'])
    self.ai_status.setText(f'AI\n{label}\n{detail}')
    if hasattr(self, 'ai_conn_status'):
      self.ai_conn_status.setText(f'{label} — {detail}')
    if hasattr(self, 'chat_status'):
      self.chat_status.setText(label)
      self.chat_status.setToolTip(f'{label} — {detail}')

  def _test_connection(self):
    from neurosilicon import llm as llm_lib
    llm_settings = self._llm_form_settings()
    self.ai_conn_status.setText('Testing…')

    def job(_w):
      return llm_lib.test_connection(llm_settings)

    def done(result):
      ok, msg, secs = result
      text = f"{'Connected' if ok else 'Failed'} ({secs}s): {msg}"
      if not ok and ('model_not_found' in msg or ' 404' in msg):
        text += ('\nHint: the server answered, so the URL and key '
                 'reach it — the API model name is not an ID this '
                 'endpoint offers. Press List models and copy one '
                 'exact ID (assistant/app names like "Muse Wonder" '
                 'are not API model IDs).')
      elif not ok and (' 401' in msg or ' 403' in msg):
        text += ('\nHint: authentication failed — check the API key '
                 '(and that it belongs to this base URL).')
      elif not ok:
        text += ('\nHint: check the base URL is reachable and the '
                 'server is running (Guide tab has per-provider '
                 'steps).')
      self.ai_conn_status.setText(text)
      self.log(f'LLM test: {msg}')
      if ok:
        self._chat_append(
            'Assistant',
            'Connection test succeeded.\n' + assistant_lib
            .connection_explanation(llm_settings))
    self._start(job, done)

  def _list_models(self):
    from neurosilicon import llm as llm_lib
    llm_settings = self._llm_form_settings()
    self.ai_conn_status.setText('Asking the endpoint for models…')

    def job(_w):
      try:
        return True, llm_lib.list_models(llm_settings)
      except llm_lib.LLMError as e:
        return False, str(e)

    def done(result):
      ok, payload = result
      if not ok:
        self.ai_conn_status.setText(f'List models failed: {payload}')
        return
      ids = payload
      current = (self.ai_model.text().strip() if llm_settings.get(
          'provider') == 'api' else self.ai_local_model.text().strip())
      note = ('' if current in ids else
              f'\nYour current model "{current}" is NOT in this '
              'list — copy one exact ID below into the model field, '
              'then Save + Test.')
      shown = ', '.join(ids[:20]) + ('…' if len(ids) > 20 else '')
      self.ai_conn_status.setText(
          f'Models your key can use ({len(ids)}): {shown}{note}')
      # Selectable model buttons: one click fills + saves + tests.
      while self.models_layout.count() > 1:
        child = self.models_layout.takeAt(1)
        if child.widget():
          child.widget().deleteLater()
      for mid in ids:
        b = QPushButton(mid)
        b.setToolTip(f'Use model {mid} (fills, saves, tests)')
        b.clicked.connect(
            lambda _=0, m=mid: self._apply_model(m))
        self.models_layout.addWidget(b)
      self.models_layout.addStretch(1)
      self.models_scroll.setVisible(True)
      self._chat_append(
          'Assistant',
          f'Models available at this endpoint ({len(ids)}): '
          + ', '.join(ids[:30])
          + ('\nCopy one exact ID into AI Connectors → API model '
             '(or Local model), Save, then Test connection.'
             if current not in ids else
             '\nYour current model is in the list — Test away.'))
      self.log(f'Listed {len(ids)} models from the endpoint.')
    self._start(job, done)

  def _apply_model(self, model_id):
    """One-click model select: fill the field, save, test."""
    if self.ai_provider.currentText() == 'local':
      self.ai_local_model.setText(model_id)
    else:
      self.ai_model.setText(model_id)
    self.ai_conn_status.setText(
        f'Selected model: {model_id} — saved, testing connection…')
    self.log(f'Model selected: {model_id}')
    self._save_connectors()
    self._test_connection()

  # -- status / designs -------------------------------------------------
  def log(self, msg):
    self.log_view.appendPlainText(str(msg))
    self.statusBar().showMessage(str(msg)[:120])

  def _tick(self):
    if self._t0:
      self.elapsed.setText(f'running… {time.time() - self._t0:.0f}s')

  def _refresh_tool_status(self):
    pdk = self._settings['pdk']
    lines = []
    for label, key in (('yosys', 'yosys'), ('OpenROAD', 'openroad')):
      path = project_lib.resolve_tool(pdk.get(key, ''))
      ok = os.path.isfile(path) and os.access(path, os.X_OK)
      lines.append(f"{'●' if ok else '○'} {label}: "
                   f"{'ready' if ok else 'missing'}")
    try:
      project_lib.resolve_pdk(pdk)
      lines.append('● sky130 PDK: ready')
    except ValueError:
      lines.append('○ sky130 PDK: not found')
    self.tool_status.setText('Toolchain\n' + '\n'.join(lines))

  def refresh_designs(self):
    self._designs = {}
    for name, (netlist, plc) in BUILTIN_NETLISTS.items():
      if os.path.isfile(_abs(netlist)):
        self._designs[name] = (_abs(netlist), _abs(plc))
    for name, entry in project_lib.load_registry().items():
      if os.path.isfile(entry.get('netlist', '')):
        self._designs[f'imported: {name}'] = (
            entry['netlist'], entry.get('plc', ''))
    self.design_list.clear()
    self.design_list.addItems(sorted(self._designs))
    # Startup shows an EMPTY canvas: a design's floorplan image and
    # details appear only when the user selects it (or imports one).
    self.design_list.setCurrentRow(-1)
    self.design_list.clearSelection()

  def _select_design(self, text):
    """Selects a design row by exact text (loads its floorplan)."""
    for i in range(self.design_list.count()):
      if self.design_list.item(i).text() == text:
        self.design_list.setCurrentRow(i)
        return True
    return False

  def _design_picked(self, name):
    pair = self._designs.get(name)
    if not pair:
      return
    from neurosilicon import setup as setup_lib
    self._plc_pair = pair
    try:
      header = setup_lib.parse_plc_header(pair[1])
      nodes = len(setup_lib.count_plc_nodes(pair[1]))
      info = (f'{name}\nnodes: {nodes} · canvas: '
              f"{header.get('width', '?')}×{header.get('height', '?')} · "
              f"grid: {header.get('columns', '?')}×"
              f"{header.get('rows', '?')}")
    except OSError:
      info = name
    self.design_info.setText(info)
    # Pins/orientations are per-design state.
    self._pinned = {}
    self._orientations = {}
    self.canvas.set_pinned(set())
    self.canvas.set_orientations({})
    self._update_pin_label()
    if self._thread is not None:
      # Never parse a big netlist on the UI thread mid-job.
      self._pending_load = (pair, name)
      self.log(f'{name}: queued — floorplan loads when the '
               'current job finishes.')
      return
    self._load_design_async(pair, name)

  def _load_design_async(self, pair, name):
    """Loads a design's floorplan OFF the UI thread (no freeze)."""
    self._job_kind = 'load'
    self.progress_label.setText(f'Loading {name} floorplan…')

    def job(w):
      plc, frame = liveplace_lib.static_snapshot(pair[0], pair[1])
      return {'plc': plc, 'frame': frame,
              'segments': liveplace_lib.net_segments(plc)}

    def done(res):
      self._apply_canvas_plc(res['plc'], res['frame'], res['segments'])
      self.progress_label.setText(
          f'{name} loaded — click it again for the live heat-map '
          'simulation, or Place → Run floorplan.')
    self._start(job, done)

  def _apply_canvas_plc(self, plc, frame, segments=None,
                        keep_frames=False):
    self._current_plc = plc
    self._congestion_ready = False
    self._congestion_plc = None
    self._congestion_analysis = None
    if segments is None:
      try:
        segments = liveplace_lib.net_segments(plc)
      except Exception:
        segments = []
    self.canvas.set_segments(segments)
    self.canvas.set_keepouts(self._keepouts)
    self.canvas.set_pinned(set(self._pinned))
    self.canvas.set_orientations(self._orientations)
    self._update_timing()
    if keep_frames and self._frames:
      self._frames.append(frame)
      self.scrub.setRange(0, len(self._frames) - 1)
      self._show_frame(len(self._frames) - 1)
    else:
      self._set_frames([frame])
    self._thermal_update()

  # -- designs panel / demo simulation -----------------------------------
  def _set_left_visible(self, visible):
    self.design_panel.setVisible(visible)
    # Unhide must be obvious: the control-bar button becomes a
    # labeled, amber-highlighted "▶ Designs" while the panel hides.
    self.left_toggle.setText('☰' if visible else '▶ Designs')
    self.left_toggle.setStyleSheet(
        '' if visible else
        'background:#9e6a1e; border-color:#e8a33d; font-weight:600;')
    self.left_toggle.setToolTip(
        'Hide designs panel' if visible else 'Show designs panel')

  def _toggle_left(self):
    self._set_left_visible(not self.design_panel.isVisible())

  def _set_right_visible(self, visible):
    self.right_panel.setVisible(visible)

  def _toggle_right(self):
    self._set_right_visible(not self.right_panel.isVisible())

  def _show_log(self):
    """Top-button entry to the internal log terminal (Log tab)."""
    self._set_right_visible(True)
    for i in range(self.right_tabs.count()):
      if self.right_tabs.tabText(i) == 'Log':
        self.right_tabs.setCurrentIndex(i)
        break
    self.log_view.ensureCursorVisible()

  def _design_clicked(self, item):
    """Clicking a design plays the moving heat-map simulation.

    The static floorplan image loads via _design_picked; the click
    additionally streams a sequential placement so the user sees
    macros landing with a live (Thermal) heat map. Large designs
    take a few seconds to build before the first frame lands.
    """
    if self._thread is not None:
      self.log('Busy — simulation already running.')
      return
    pair = self._designs.get(item.text())
    if not pair:
      return
    self.view_combo.setCurrentText('Thermal')
    self._start_demo(pair)

  def _start_demo(self, pair):
    netlist, plc_path = pair
    self._job_kind = 'demo'
    self._job_t0 = time.time()
    self._frames = []
    self.scrub.setRange(0, 0)
    self.canvas.set_congestion(None)
    self._set_left_visible(False)
    self.progress_label.setText(
        'Preparing simulation (building netlist)… frames will '
        'stream as each macro lands.')

    def job(w):
      w.log.emit('Demo simulation: sequential placement, largest '
                 'first, with live heat map (Thermal view).')
      info, _frames = liveplace_lib.run_live_rollout(
          netlist, plc_path, fd_steps=(5, 5, 5), method='first_legal',
          seed=0, init_fd=False,
          on_snapshot=lambda f: w.snap.emit(f),
          on_live=lambda i: w.live.emit(i))
      return info

    def done(info):
      self.progress_label.setText(
          f"Demo simulation finished — proxy "
          f"{info.get('proxy_cost', -1):.4f}. Use Play/scrub to "
          'replay, or Run floorplan for scored baselines.')
      self.log(f"Demo done (proxy {info.get('proxy_cost', -1):.4f}). "
               'Scrub the slider to replay the heat map; switch View '
               'anytime (including Floorplan + Wires / Thermal + '
               'Wires combined).')
      self._live_info(info)
      if info.get('proxy_cost', -1) > 0:
        row = {'method': 'live (first_legal)', 'episode': 0,
               'proxy_cost': float(info['proxy_cost']),
               'wirelength': float(info['wirelength']),
               'congestion': float(info['congestion']),
               'density': float(info['density']),
               'seconds': time.time() - self._job_t0}
        self._live_row(row, count_progress=False)
        self._note_run_delta(row)
      self._refresh_details()
    self._start(job, done)

  def _refresh_details(self):
    """Full design + run details (wirelength, congestion, thermal…)."""
    if not hasattr(self, 'details_view'):
      return
    item = self.design_list.currentItem()
    lines = [f"Design: {item.text() if item else '—'}"]
    plc = self._current_plc
    if plc is not None:
      w, h = plc.get_canvas_width_height()
      cols, rows = plc.get_grid_num_columns_rows()
      hard = sum(1 for i in plc.get_macro_indices()
                 if not plc.is_node_soft_macro(i))
      soft = sum(1 for i in plc.get_macro_indices()
                 if plc.is_node_soft_macro(i))
      lines.append(
          f'Die: {w:.1f}×{h:.1f} µm · grid {cols}×{rows} · hard '
          f'macros {hard} · soft clusters {soft} · ports '
          f'{len(plc.port_indices)} · nets {len(plc.nets)}')
    ok = [r for r in self._results if r['proxy_cost'] > 0]
    if ok:
      best = min(ok, key=lambda r: r['proxy_cost'])
      lines.append(
          f"Best placement: proxy {best['proxy_cost']:.4f} "
          f"= wirelength {best['wirelength']:.4f} + congestion "
          f"{best['congestion']:.4f} + density {best['density']:.4f} "
          f"({best['method']} episode {best['episode']}, "
          f"{best['seconds']:.1f}s) — lower is better on every term.")
      lines.append(f'Episodes finished: {len(self._results)} '
                   f'(table below has every wirelength/congestion/'
                   'density row)')
    else:
      lines.append('No scored run yet — Run floorplan fills in '
                   'wirelength / congestion / density per episode.')
    power = getattr(self, '_power', {}) or {}
    if power:
      hot = sorted(power.items(), key=lambda kv: -kv[1])[:5]
      lines.append('Thermal hotspots (relative power): ' + ', '.join(
          f'{n} {p:.3f}' for n, p in hot))
    if self._keepouts:
      lines.append(f'Chat constraint active ({self._keepout_mode}): '
                   f'{len(self._keepouts)} rect(s) — applied to runs.')
    node = tech_lib.preset(self._tech_key)
    lines.append(f"Technology Node: {node['label']} · congestion ×"
                 f"{node['congestion_multiplier']} · leakage ×"
                 f"{node['leakage_factor']}")
    if self._pinned:
      lines.append('🔒 Pinned: ' + ', '.join(
          f'{n} @ ({x:.0f}, {y:.0f})'
          for n, (x, y) in sorted(self._pinned.items())))
    flips = {n: d for n, d in self._orientations.items() if d}
    if flips:
      lines.append('Orientation flips (HPWL-optimized): '
                   + ', '.join(f'{n}={d}°'
                               for n, d in sorted(flips.items())[:8]))
    if self._best_plc_path:
      lines.append(f'Placed file: {self._best_plc_path} · output '
                   f'folder: {self._outdir}')
    self.details_view.setPlainText('\n'.join(lines))
    if hasattr(self, 'pareto'):
      self.pareto.set_data(self._results)

  # -- worker / playback / canvas ---------------------------------------
  def _busy(self, busy):
    buttons = [self.imp_go_btn, self.run_btn]
    for name in ('auto_btn_place', 'auto_btn_chat', 'rtl_btn_import',
                 'rtl_btn_chat', 'apply_fix_btn', 'review_btn',
                 'chat_import_btn'):
      btn = getattr(self, name, None)
      if btn is not None:
        buttons.append(btn)
    for b in buttons:
      b.setEnabled(not busy)
    if busy:
      self._t0 = time.time()
      self._timer.start(500)
      self.progress.setValue(0)
      self.progress.setFormat('%p%')
      self.progress_label.setText('Working… (progress details here)')
    else:
      self._timer.stop()
      self._t0 = None
      self.elapsed.setText('idle')
      self.progress.setValue(100)
      self.progress.setFormat('%p%')
      if self.progress_label.text().startswith('Working'):
        self.progress_label.setText('Done.')

  def _start(self, fn, on_done):
    if self._thread is not None:
      self.log('Busy — wait for the current job to finish.')
      return
    self._busy(True)
    self._thread = QThread(self)
    self._worker = Worker(fn)
    self._worker.moveToThread(self._thread)
    self._thread.started.connect(self._worker.run)
    self._worker.log.connect(self.log)
    self._worker.row.connect(self._live_row)
    self._worker.snap.connect(self._live_snap)
    self._worker.phase.connect(self.progress_label.setText)
    self._worker.live.connect(self._live_info)
    # Bound-method slot: queued to the UI thread. A bare lambda here
    # would run on_done on the WORKER thread (widget updates and
    # QThread.wait from the wrong thread — the old "simulation does
    # nothing / freezes" bug).
    self._pending_done = on_done
    self._worker.done.connect(self._dispatch_done)
    self._worker.failed.connect(self._failed)
    self._thread.start()

  def _dispatch_done(self, result):
    fn, self._pending_done = getattr(self, '_pending_done', None), None
    try:
      if fn is not None:
        fn(result)
    finally:
      self._finish()

  def _finish(self):
    if hasattr(self, '_snap_timer'):
      self._snap_timer.stop()
    if self._frames:
      self._show_frame(len(self._frames) - 1)
    self._busy(False)
    if self._thread:
      self._thread.quit()
      self._thread.wait(5000)
      self._thread = None
      self._worker = None
    if getattr(self, '_pending_load', None):
      pair, name = self._pending_load
      self._pending_load = None
      self._load_design_async(pair, name)

  def _failure_details(self, short, full):
    """Detailed, actionable failure report (also used for imports)."""
    lines = [f'FAILED ({self._job_kind or "job"}): {short}', '']
    errs = re.findall(r'^[^\n]*ERROR:[^\n]*$', full, re.MULTILINE)
    if errs:
      lines.append('Errors reported by the tools:')
      lines += [f'  {e.strip()}' for e in errs[:5]]
      lines.append('')
    m = re.search(r'see ([^\s)]*synth\.log)', full)
    if m and os.path.isfile(m.group(1)):
      try:
        with open(m.group(1), errors='replace') as f:
          tail = f.read().strip().splitlines()[-12:]
        lines.append(f'Synthesis log tail ({m.group(1)}):')
        lines += [f'  {t}' for t in tail]
        lines.append('')
      except OSError:
        pass
    if self._job_kind == 'import' and self._disc:
      d = self._disc
      lines.append(
          f"What was being imported: kind={d['kind']} root={d['root']} "
          f"top={d['suggested_top'] or '—'} rtl_files="
          f"{len(d['rtl_files'])}")
      for note in d['found'] + d['notes']:
        lines.append(f'  · {note}')
      for miss in d['missing']:
        lines.append(f'  ✗ missing: {miss}')
      lines.append('')
    lines.append('What to try:')
    if 'undefined macro' in full or 'compiler directive' in full:
      lines.append('  · A file uses a makefile-defined macro (e.g. '
                   '`CLOCK_PERIOD) that only the simulation build '
                   'defines. Simulation test drivers cannot be '
                   'placed — exclude them (uncheck in Import), set '
                   'the macro under Verilog defines if the file is '
                   'real RTL, or import generated RTL instead.')
    if 'TOK_REAL' in full or 'syntax error' in full:
      lines.append('  · A source uses simulation-only syntax the '
                   'bundled yosys cannot parse. Staging already '
                   'neutralizes real/$time/#delay code and blackboxes '
                   'behavioral memories; if this persists, note the '
                   'file:line above and try the design\'s core as '
                   'top instead of the full SoC.')
    if 'bad_alloc' in full or 'out of memory' in full.lower():
      lines.append('  · Synthesis ran out of memory: huge behavioral '
                   'memories are auto-blackboxed; for very large SoCs '
                   'pick the core module as top, or close other apps.')
    if 'not found' in full or 'missing' in full:
      lines.append('  · Check Settings → PDK & Toolchain paths '
                   '(yosys/OpenROAD/LEF/liberty), then retry.')
    lines.append('  · Full traceback is in the Log tab; the synthesis '
                 'log path above keeps the complete yosys output.')
    return '\n'.join(lines)

  def _failed(self, short, full):
    self._finish()
    details = self._failure_details(short, full)
    self.log(full)
    self.log(details)
    self.progress_label.setText(f'Failed: {short[:90]}')
    if self._job_kind == 'import':
      self.disc_view.setPlainText(details)
      self.tabs.setCurrentIndex(0)
    self.details_view.setPlainText(details)
    QMessageBox.critical(
        self, 'NeuroSilicon — failed',
        short + '\n\nFull details: Results tab (details box), '
        'Import tab, and the Log tab.')

  def _live_row(self, row, count_progress=True):
    self._results.append(dict(row))
    r = self.result_table.rowCount()
    self.result_table.insertRow(r)
    vals = [row['method'], row['episode'], f"{row['proxy_cost']:.4f}",
            f"{row['wirelength']:.4f}", f"{row['congestion']:.4f}",
            f"{row['density']:.4f}", f"{row['seconds']:.1f}"]
    for c, val in enumerate(vals):
      self.result_table.setItem(r, c, QTableWidgetItem(str(val)))
    ok = [x for x in self._results if x['proxy_cost'] > 0]
    if ok:
      best = min(ok, key=lambda x: x['proxy_cost'])
      self.best_label.setText(f"best proxy: {best['proxy_cost']:.4f}")
      self.card_proxy.setText(f"proxy\n{best['proxy_cost']:.4f}")
      self.card_wl.setText(f"wirelength\n{best['wirelength']:.4f}")
      self.card_cong.setText(f"congestion\n{best['congestion']:.4f}")
    if count_progress:
      total = max(2 * max(self.run_episodes.value(), 1), 1)
      base = 60 if self._job_kind in ('run', 'import') else 0
      self.progress.setValue(base + int((100 - base) * (r + 1) / total))
      self.progress.setFormat(f"episode {r + 1}/{total} · %p%")
      self.progress_label.setText(
          f"Scored episode {r + 1}/{total}: {row['method']} "
          f"ep {row['episode']} — proxy {row['proxy_cost']:.4f}, "
          f"wirelength {row['wirelength']:.4f}, congestion "
          f"{row['congestion']:.4f}, density {row['density']:.4f}")
    self._refresh_details()

  def _note_run_delta(self, best_row):
    """Δ% of this run's best vs the previous run on the same design."""
    if not best_row or best_row.get('proxy_cost', -1) <= 0:
      return
    item = self.design_list.currentItem()
    design = item.text() if item else ''
    prev = self._prev_best
    if prev and self._prev_design == design:
      def _d(key):
        return (best_row[key] - prev[key]) / prev[key] * 100 \
            if prev.get(key) else 0.0
      self.delta_label.setText(
          f"Δ vs previous run ({design}): proxy {_d('proxy_cost'):+.1f}%"
          f" · wirelength {_d('wirelength'):+.1f}% · congestion "
          f"{_d('congestion'):+.1f}%  (negative = better)")
    elif hasattr(self, 'delta_label'):
      self.delta_label.setText(
          'First scored run on this design — run again (e.g. after '
          'Apply AI Fix & Re-Place) to see Δ improvements here.')
    self._prev_best = dict(best_row)
    self._prev_design = design

  def _live_info(self, info):
    """Live-rollout score fills the metric cards immediately.

    (Baseline cards otherwise stay blank until the first scored
    episode finishes — minutes on large designs, which reads as
    "no data / frozen" even though placement already scored.)
    """
    if info.get('proxy_cost', -1) > 0:
      self.card_proxy.setText(f"proxy\n{info['proxy_cost']:.4f}")
      self.card_wl.setText(f"wirelength\n{info['wirelength']:.4f}")
      self.card_cong.setText(f"congestion\n{info['congestion']:.4f}")
      self._live_best = dict(info)

  def _live_snap(self, frame):
    self._frames.append(frame)
    self.scrub.setRange(0, max(0, len(self._frames) - 1))
    total = max(frame.get('total', 1), 1)
    pct = int(60 * frame.get('step', 0) / total)
    self.progress.setValue(pct)
    self.progress.setFormat(f"placing {frame.get('step', 0)}/{total} · %p%")
    self.progress_label.setText(
        f"Live placement: macro {frame.get('step', 0)}/{total} "
        f"({pct}%) — placing {frame.get('current') or '…'} · "
        'heat map updates every macro')
    # Coalesce canvas redraws to ~20 FPS: frames still all record
    # for scrub/playback, but painting every macro on a 133-macro
    # design starves the UI event loop (the "Not Responding" freeze).
    now = time.monotonic()
    if now - self._last_frame_show >= 0.05:
      self._last_frame_show = now
      self._show_frame(len(self._frames) - 1)
    elif not self._snap_timer.isActive():
      self._snap_timer.start(50)

  def _flush_snap(self):
    if self._frames:
      self._last_frame_show = time.monotonic()
      self._show_frame(len(self._frames) - 1)

  def _set_frames(self, frames):
    self._play_timer.stop()
    self.play_btn.setText('▶ Play')
    self._frames = list(frames)
    self.scrub.setRange(0, max(0, len(self._frames) - 1))
    self._show_frame(len(self._frames) - 1 if self._frames else 0)

  def _show_frame(self, idx, from_timer=False):
    if not self._frames:
      return
    idx = max(0, min(int(idx), len(self._frames) - 1))
    if from_timer and idx == self._frame_idx and \
            idx >= len(self._frames) - 1:
      self._play_timer.stop()
      self.play_btn.setText('▶ Play')
      return
    self._frame_idx = idx
    frame = self._frames[idx]
    self.scrub.blockSignals(True)
    self.scrub.setValue(idx)
    self.scrub.blockSignals(False)
    self.canvas.set_frame(frame)
    # Real-time heat while scrubbing/playing: colors follow THIS
    # frame's placed macros (scaled by the thermal corner sliders).
    self._thermal_update()
    placed = sum(1 for m in frame['macros']
                 if not m['soft'] and m.get('placed', True))
    hard = sum(1 for m in frame['macros'] if not m['soft'])
    self.card_macros.setText(f'macros placed\n{placed}/{hard}')
    cur = frame.get('current') or 'done'
    self.frame_label.setText(
        f"{idx + 1}/{len(self._frames)} · {frame['step']}/"
        f"{frame['total']}")
    self.frame_label.setToolTip(
        f"Frame {idx + 1} of {len(self._frames)} · step "
        f"{frame['step']}/{frame['total']} · placing: {cur}")

  def _toggle_play(self):
    if not self._frames:
      return
    if self._play_timer.isActive():
      self._play_timer.stop()
      self.play_btn.setText('▶ Play')
      return
    if self._frame_idx >= len(self._frames) - 1:
      self._show_frame(0)
    self._play_timer.start()
    self.play_btn.setText('⏸ Pause')

  def _tab_changed(self, idx):
    """Opening the Thermal tab puts the thermal map on the canvas."""
    if 0 <= idx < self.tabs.count() and \
            self.tabs.tabText(idx) == 'Thermal':
      self.view_combo.setCurrentText('Thermal')

  def _update_timing(self):
    """est. Fmax card + critical-path overlay for the current plc."""
    plc = self._current_plc
    if plc is None:
      self.card_fmax.setText('est. Fmax\n—')
      self.canvas.set_critical_path([], False)
      return
    if self._crit_plc is not plc:
      try:
        self._crit_result = liveplace_lib.critical_path(plc)
        self._crit_plc = plc
      except Exception as e:
        self.log(f'critical path unavailable: {e}')
        return
    cp = self._crit_result
    self.card_fmax.setText(f"est. Fmax\n{cp['fmax_mhz']:.0f} MHz")
    self.card_fmax.setToolTip(
        f"crit. delay {cp['delay_ns']:.2f} ns (wire "
        f"{cp['wire_delay_ns']:.2f} + logic {cp['logic_delay_ns']:.2f})"
        f" · path: {' → '.join(cp['path'][:8])}"
        f"{'…' if len(cp['path']) > 8 else ''} · Manhattan wire + "
        'logic delay proxy, not signoff STA.')
    visible = bool(hasattr(self, 'crit_check')
                   and self.crit_check.isChecked())
    self.canvas.set_critical_path(cp['points'], visible)

  def _snapshot_for(self, plc, row, label):
    """Compare-ready snapshot: frame, heat data, PPA for one run."""
    hard = sum(1 for i in plc.get_macro_indices()
               if not plc.is_node_soft_macro(i))
    frame = liveplace_lib.snapshot(plc, hard, hard, '')
    power = liveplace_lib.scaled_power(plc)
    try:
      grid = liveplace_lib.congestion_analysis(plc)['grid']
    except Exception:
      grid = None
    return {'label': label, 'frame': frame, 'power': power,
            'grid': grid, 'row': dict(row or {}),
            'ppa': liveplace_lib.ppa_metrics(plc, power)}

  def _record_snapshot(self, snap):
    """Appends a scored run and refreshes the Compare pair."""
    self._run_snapshots.append(snap)
    if self._compare_left is None:
      self._compare_left = snap
    later = self._run_snapshots[1:]
    if later:
      self._compare_right = min(
          later, key=lambda s: s['ppa']['proxy_cost'])
    if hasattr(self, 'view_combo') and \
            self.view_combo.currentText() == 'Compare (Baseline vs. AI)':
      self._refresh_compare()

  def _refresh_compare(self):
    """Pushes the baseline/best-AI pair (with Δ badges) to canvas."""
    left, right = self._compare_left, self._compare_right
    if left is None and len(self._run_snapshots) >= 2:
      left = self._run_snapshots[0]
      right = min(self._run_snapshots[1:],
                  key=lambda s: s['ppa']['proxy_cost'])
      self._compare_left, self._compare_right = left, right
    if not left or not right:
      self.canvas.set_compare(None, None)
      return
    d = liveplace_lib.ppa_delta(left['ppa'], right['ppa'])
    badges = [(f"{d['congestion_pct']:+.0f}% Congestion",
               d['congestion_pct'] < 0),
              (f"{d['wirelength_pct']:+.0f}% Wirelength",
               d['wirelength_pct'] < 0),
              (f"{d['peak_power_pct']:+.0f}% Peak mW",
               d['peak_power_pct'] < 0)]
    self.canvas.set_compare(
        dict(left), {**right, 'badges': badges})

  def _view_changed(self, view):
    if view == 'Compare (Baseline vs. AI)':
      self.canvas.set_view(view)
      self._refresh_compare()
      if not self._compare_left or not self._compare_right:
        self.log('Compare (Baseline vs. AI): needs 2 scored runs — '
                 'press ⚡ Auto-Optimize (AI Loop) to create the '
                 'Baseline + AI pair.')
      return
    self.canvas.set_view(view)
    if 'Congestion' in view:
      self._ensure_congestion()
    if 'Thermal' in view:
      self._thermal_update()

  def _load_plc_to_canvas(self, netlist, plc_path, keep_frames=False):
    plc, frame = liveplace_lib.static_snapshot(netlist, plc_path)
    self._apply_canvas_plc(plc, frame, keep_frames=keep_frames)

  def _ensure_congestion(self):
    """RUDY congestion grid for the CURRENT placement (cached per plc).

    (Recomputed whenever the plc object changes — a stale-cache bug
    here is what used to render Congestion as an empty black grid
    after a run.)
    """
    plc = self._current_plc
    if plc is None:
      return
    if getattr(self, '_congestion_ready', False) and \
            getattr(self, '_congestion_plc', None) is plc:
      return
    try:
      self._congestion_analysis = \
          liveplace_lib.congestion_analysis(plc)
      self._congestion_plc = plc
      self.canvas.set_congestion(tech_lib.scaled_congestion(
          self._congestion_analysis['grid'], self._tech_key))
      self._congestion_ready = True
      self._update_contours()
    except Exception as e:
      self.log(f'congestion view unavailable: {e}')

  def _update_contours(self):
    """SAM bottleneck segmentation over the active heat grid."""
    if not hasattr(self, 'sam_check') or not self.sam_check.isChecked():
      if hasattr(self, 'canvas'):
        self.canvas.set_contours([])
      return
    view = self.view_combo.currentText()
    base = view.split(' +')[0]
    plc = self._current_plc
    if plc is None or base not in ('Thermal', 'Congestion'):
      self.canvas.set_contours([])
      return
    try:
      if base == 'Thermal':
        grid = liveplace_lib.power_grid(
            plc, getattr(self, '_power', {}) or {})
        fmt = 'Peak: {:.3f} rel.'
      else:
        self._ensure_congestion()
        grid = tech_lib.scaled_congestion(
            self._congestion_analysis['grid'], self._tech_key)
        fmt = 'RUDY: {:.2f}x'
      W, H = plc.get_canvas_width_height()
      cols, rows = plc.get_grid_num_columns_rows()
      cw, ch = W / max(cols, 1), H / max(rows, 1)
      boxes = []
      for cl in liveplace_lib.segment_hotspots(grid, 0.15):
        c0, r0, c1, r1 = cl['bbox']
        boxes.append((c0 * cw, r0 * ch, (c1 + 1) * cw,
                      (r1 + 1) * ch, fmt.format(cl['peak'])))
      self.canvas.set_contours(boxes)
    except Exception as e:
      self.log(f'SAM contours unavailable: {e}')

  def _macro_clicked(self, m):
    """Canvas click: flylines + macro facts in the status bar."""
    plc = self._current_plc
    if plc is None:
      return
    try:
      segs = liveplace_lib.segments_for_macro(plc, m['name'])
    except Exception:
      segs = []
    self.canvas.set_highlight(m['name'], segs)
    power = (getattr(self, '_power', {}) or {}).get(m['name'])
    local = ''
    ca = getattr(self, '_congestion_analysis', None)
    if ca and ca.get('grid'):
      grid = ca['grid']
      rows = len(grid)
      cols = len(grid[0]) if rows else 0
      W, H = plc.get_canvas_width_height()
      c = max(0, min(cols - 1, int(m['x'] / max(W, 1e-9) * cols)))
      r = max(0, min(rows - 1, int(m['y'] / max(H, 1e-9) * rows)))
      local = f" · local RUDY {grid[r][c]:.3f}"
    self.statusBar().showMessage(
        f"{m['name']} · ({m['x']:.1f}, {m['y']:.1f}) µm · "
        f"{m['w']:.1f}×{m['h']:.1f} µm · "
        f"power {power:.3f} rel.{local} · {len(segs)} flylines"
        if power is not None else
        f"{m['name']} · ({m['x']:.1f}, {m['y']:.1f}) µm · "
        f"{m['w']:.1f}×{m['h']:.1f} µm{local} · {len(segs)} flylines")
    self.log(f"Selected macro {m['name']} at ({m['x']:.1f}, "
             f"{m['y']:.1f}) — {len(segs)} flylines highlighted.")

  # -- pinning / technology node ---------------------------------------
  def _all_hard_pinned(self):
    """True when every hard macro on canvas is 🔒 (nothing to place)."""
    if not self._pinned or not self._frames:
      return False
    hard = {m['name'] for m in self._frames[-1]['macros']
            if not m['soft']}
    return bool(hard) and hard <= set(self._pinned)

  def _effective_init_plc(self, netlist, plc_path):
    """Init .plc honoring 🔒 pins (original path when no pins)."""
    if not self._pinned:
      return plc_path
    out = os.path.join(project_lib.repo_root(), 'logs',
                       'desktop_runs', 'pinned_init.plc')
    try:
      liveplace_lib.write_pinned_placement(netlist, plc_path,
                                           self._pinned, out)
      self.log(f'🔒 {len(self._pinned)} pinned macro(s) held via '
               f'pinned init placement: {sorted(self._pinned)}')
      return out
    except Exception as e:
      self.log(f'pinned placement unavailable ({e}); pins ignored '
               'for this run.')
      return plc_path

  def _update_pin_label(self):
    if not hasattr(self, 'pin_label'):
      return
    if self._pinned:
      items = ', '.join(f'{n} @ ({x:.0f}, {y:.0f})'
                        for n, (x, y) in sorted(self._pinned.items()))
      self.pin_label.setText(f'🔒 Pinned macros: {items} — held fixed '
                             'in the next placement runs.')
    else:
      self.pin_label.setText('🔒 Pinned macros: none — right-click or '
                             'double-click a macro to lock it, or drag '
                             'it to a grid tile.')

  def _toggle_pin(self, m):
    name = m['name']
    plc = self._current_plc
    if name in self._pinned:
      del self._pinned[name]
      if plc is not None:
        try:
          idx = plc.get_node_index(name)
          if idx is not None and idx >= 0:
            plc.unfix_node_coord(idx)
        except Exception:
          pass
      self.log(f'🔓 Unpinned {name}.')
    else:
      self._pinned[name] = (float(m['x']), float(m['y']))
      if plc is not None:
        try:
          idx = plc.get_node_index(name)
          if idx is not None and idx >= 0:
            plc.fix_node_coord(idx)
        except Exception:
          pass
      self.log(f"🔒 Pinned {name} @ ({m['x']:.1f}, {m['y']:.1f}) — "
               'fixed for subsequent placement runs; drag it to move '
               'the pin to another grid tile.')
    self.canvas.set_pinned(set(self._pinned))
    self._update_pin_label()

  def _clear_pins(self):
    if self._current_plc is not None:
      for name in list(self._pinned):
        try:
          idx = self._current_plc.get_node_index(name)
          if idx is not None and idx >= 0:
            self._current_plc.unfix_node_coord(idx)
        except Exception:
          pass
    self._pinned = {}
    self.canvas.set_pinned(set())
    self._update_pin_label()
    self.log('🔓 All pinned macros cleared.')

  def _move_macro(self, name, x, y):
    if self._thread is not None:
      self.log('Busy — drag-to-move waits for the current job.')
      if self._frames:
        self.canvas.set_frame(
            self._frames[min(self._frame_idx, len(self._frames) - 1)])
      return
    plc = self._current_plc
    if plc is None:
      return
    # Snap to the nearest placement-grid tile center, clamped on-die.
    canvas_w, canvas_h = plc.get_canvas_width_height()
    cols, rows = plc.get_grid_num_columns_rows()
    cw, ch = canvas_w / max(cols, 1), canvas_h / max(rows, 1)
    macro = next((mm for f in self._frames
                  for mm in f['macros'] if mm['name'] == name), None)
    mw = macro['w'] if macro else 0.0
    mh = macro['h'] if macro else 0.0
    sx = min(max((int(x / cw) + 0.5) * cw, mw / 2),
             max(canvas_w - mw / 2, mw / 2))
    sy = min(max((int(y / ch) + 0.5) * ch, mh / 2),
             max(canvas_h - mh / 2, mh / 2))
    self._pinned[name] = (float(sx), float(sy))
    try:
      idx = plc.get_node_index(name)
      if idx is not None and idx >= 0:
        plc.update_node_coords(idx, float(sx), float(sy))
        plc.fix_node_coord(idx)
    except Exception as e:
      self.log(f'drag-to-move failed for {name}: {e}')
      return
    for frame in self._frames:
      for mm in frame['macros']:
        if mm['name'] == name:
          mm['x'], mm['y'] = float(sx), float(sy)
    self._crit_plc = None
    self.canvas.set_pinned(set(self._pinned))
    self.canvas.update()
    self._update_timing()
    self._thermal_update()
    self._update_pin_label()
    self.log(f'🔒 {name} moved + pinned @ tile ({sx:.1f}, {sy:.1f}) '
             'µm — remaining macros will place around it.')

  def _on_tech_changed(self, key, _origin=''):
    if not key:
      return
    self._tech_key = key
    node = tech_lib.preset(key)
    for combo_name in ('tech_combo', 'tech_combo_settings'):
      combo = getattr(self, combo_name, None)
      if combo is not None:
        idx = combo.findData(key)
        if idx >= 0 and combo.currentIndex() != idx:
          combo.blockSignals(True)
          combo.setCurrentIndex(idx)
          combo.blockSignals(False)
    if hasattr(self, 'th_clock'):
      self.th_clock.setValue(int(node['clock_mhz']))
    if hasattr(self, 'th_volt'):
      self.th_volt.setValue(int(float(node['voltage']) * 100))
    if hasattr(self, 'tech_label'):
      self.tech_label.setText(
          f"{node['label']}: V={node['voltage']} V · f="
          f"{node['clock_mhz']:.0f} MHz · leakage ×"
          f"{node['leakage_factor']} (ΔT {node['temp_offset_c']:+.0f}°C "
          f"proxy) · congestion ×{node['congestion_multiplier']} · "
          f"{node['note']}")
    self._congestion_ready = False
    if hasattr(self, 'view_combo') and \
            'Congestion' in self.view_combo.currentText():
      self._ensure_congestion()
    self._update_contours()
    self._thermal_update()
    self.log(f"Technology Node → {node['label']} (congestion ×"
             f"{node['congestion_multiplier']}, leakage ×"
             f"{node['leakage_factor']}).")

  def _thermal_update(self):
    plc = self._current_plc
    if plc is None:
      return
    try:
      power = tech_lib.corner_power(
          plc, self._tech_key,
          clock_mhz=float(self.th_clock.value()),
          voltage=self.th_volt.value() / 100.0,
          activity=self.th_act.value() / 100.0)
    except Exception as e:
      self.th_label.setText(f'thermal view unavailable: {e}')
      return
    self._power = power
    if self._frames:
      # Heat follows the displayed frame (live / scrub / playback).
      frame = self._frames[min(self._frame_idx, len(self._frames) - 1)]
      self.canvas.set_power(tech_lib.frame_corner_power(
          frame, self._tech_key,
          clock_mhz=float(self.th_clock.value()),
          voltage=self.th_volt.value() / 100.0,
          activity=self.th_act.value() / 100.0))
    else:
      self.canvas.set_power(power)
    hot = max(power, key=power.get) if power else '?'
    node = tech_lib.preset(self._tech_key)
    self.th_label.setText(
        f"node {self._tech_key} · clock {self.th_clock.value()} MHz · "
        f'{self.th_volt.value() / 100:.2f} V · activity '
        f'{self.th_act.value()}% · leakage ×{node["leakage_factor"]} '
        f'(ΔT {node["temp_offset_c"]:+.0f}°C proxy) · hottest macro: '
        f'{hot} ({power.get(hot, 0):.3f} rel.) Switch View → Thermal '
        'to see the heat map.')
    self._refresh_details()

  # -- import (auto-discover) -------------------------------------------
  def _browse_pick(self, kind):
    if kind == 'folder':
      path = QFileDialog.getExistingDirectory(self, 'Chip design folder')
    elif kind == 'zip':
      path, _ = QFileDialog.getOpenFileName(
          self, 'Chip design ZIP', '', 'ZIP (*.zip);;All files (*)')
    else:
      path, _ = QFileDialog.getOpenFileName(
          self, 'Any file inside the chip design', '',
          'Design files (*.v *.sv *.pb.txt *.plc *.zip);;All files (*)')
    if path:
      self.imp_path.setText(path)
      self.detect_import()

  def dragEnterEvent(self, event):
    if event.mimeData().hasUrls():
      event.acceptProposedAction()

  def dropEvent(self, event):
    urls = event.mimeData().urls()
    if urls:
      self.imp_path.setText(urls[0].toLocalFile())
      self.tabs.setCurrentIndex(0)
      self.detect_import()

  def detect_import(self):
    path = self.imp_path.text().strip()
    if not path:
      QMessageBox.warning(self, 'No path', 'Pick a folder, file, or ZIP.')
      return
    try:
      d = discover_lib.discover_design(path)
    except ValueError as e:
      QMessageBox.warning(self, 'Cannot read design', str(e))
      self.log(f'Auto-detect failed: {e}')
      return
    self._disc = d
    lines = [f"Kind: {d['kind']}  ·  root: {d['root']}"]
    lines += [f'✓ {x}' for x in d['found']]
    lines += [f'• {x}' for x in d['notes']]
    lines += [f'✗ missing: {x}' for x in d['missing']]
    lines.append('Import plan:')
    lines += [f'  {i + 1}. {s}' for i, s in
              enumerate(discover_lib.import_plan(d))]
    self.disc_view.setPlainText('\n'.join(lines))
    self.imp_top.clear()
    if d['tops']:
      self.imp_top.addItems(d['tops'])
    if d['suggested_top']:
      self.imp_top.setCurrentText(d['suggested_top'])
    if not self.imp_name.text():
      stem = os.path.basename(os.path.normpath(d['root'])) or 'chip'
      self.imp_name.setText(stem)
    checked = set(d['rtl_files'])
    all_rtl = list(d['rtl_files'])
    if d['kind'] == 'rtl_project':
      try:
        scan = project_lib.scan_folder(d['root'])
        all_rtl = scan['verilog'] + scan['systemverilog']
      except ValueError:
        pass
    self._fill_files(all_rtl, checked)
    self.log(f"Auto-detected {d['kind']}: "
             f"{d['found'][0] if d['found'] else d['root']}")

  def _fill_files(self, rel_files, checked):
    self.file_list.clear()
    for rel in rel_files:
      item = QListWidgetItem(rel)
      item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
      item.setCheckState(Qt.Checked if rel in checked else Qt.Unchecked)
      self.file_list.addItem(item)

  def _checked_files(self):
    return [self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked]

  def do_import(self):
    d = self._disc
    if d is None:
      self.detect_import()
      d = self._disc
      if d is None:
        return
    name = self.imp_name.text().strip() or 'chip'
    try:
      params = project_lib.parse_params(self.imp_params.text())
      defines = [x.strip() for x in
                 self.imp_defines.text().replace(',', ' ').split()
                 if x.strip()] or None
      if d['kind'] == 'ready_netlist':
        entry = project_lib.register_netlist(name, d['netlist'], d['plc'])
        self.log(f"Registered ready netlist: {entry['netlist']}")
        self.refresh_designs()
        self._select_design(f'imported: {name}')
        self.tabs.setCurrentIndex(1)
        return
      if d['kind'] == 'unknown':
        raise ValueError('nothing importable found — see Auto-detect')
      top = self.imp_top.currentText().strip()
      if not top:
        raise ValueError('pick a top module (Auto-detect first)')
      checked = self._checked_files()
      if not checked:
        raise ValueError('no RTL files checked')
    except (ValueError, OSError) as e:
      QMessageBox.warning(self, 'Import failed', str(e))
      return
    outdir = os.path.join(project_lib.repo_root(), 'logs',
                          'desktop_imports', name)
    episodes = self.imp_episodes.value()
    settings = self._settings
    root = d['root']
    fn = lambda w: project_lib.import_project(  # noqa: E731
        root, checked, top, outdir, settings, params=params,
        defines=defines, episodes=episodes,
        progress=lambda s, m: w.log.emit(f'[{s}] {m}'))

    def done(summary):
      project_lib.add_to_registry(
          name, summary['proto']['netlist'], summary['proto']['plc'],
          extra={'top': summary['top'], 'outdir': outdir})
      self.log(f"Import complete: {summary['cells']} cells, best "
               f"proxy {summary['best_proxy']:.4f} -> {outdir}")
      self.refresh_designs()
      self._select_design(f'imported: {name}')
      self.tabs.setCurrentIndex(1)
    self._job_kind = 'import'
    self._start(fn, done)

  # -- placement --------------------------------------------------------
  def _mask_fn_for_rects(self, netlist, plc_path, rects, mode='block'):
    """Env mask fn for explicit keep-out/allow rects (or None)."""
    if not rects:
      return None
    from neurosilicon import priors as priors_lib
    from neurosilicon import setup as setup_lib
    plc = setup_lib.create_plc(netlist, plc_path)
    w, h = plc.get_canvas_width_height()
    cols, rows = plc.get_grid_num_columns_rows()
    if mode == 'allow':
      return priors_lib.make_allow_fn(cols, rows, w, h, list(rects))
    return priors_lib.make_keepout_fn(cols, rows, w, h, list(rects))

  def _mask_fn(self, netlist, plc_path):
    """Env mask fn for the chat keep-out/allow rects (or None)."""
    return self._mask_fn_for_rects(netlist, plc_path, self._keepouts,
                                   self._keepout_mode)

  def run_floorplan(self):
    item = self.design_list.currentItem()
    if not item:
      QMessageBox.warning(self, 'No design', 'Pick a design first.')
      return
    netlist, plc_path = self._designs[item.text()]
    outdir = os.path.join(project_lib.repo_root(), 'logs',
                          'desktop_runs', 'last')
    self._outdir = outdir
    episodes = self.run_episodes.value()
    fd, seed = self.run_fd.value(), self.run_seed.value()
    finetune = self.run_finetune.value()
    animate = self.run_animate.isChecked()
    method = self.run_method.currentText()
    keepouts = list(self._keepouts)
    keepout_mode = self._keepout_mode
    self._results = []
    self.result_table.setRowCount(0)
    self.best_label.setText('best proxy: —')
    for card in (self.card_proxy, self.card_wl, self.card_cong):
      card.setText(card.text().split('\n')[0] + '\n—')
    self._frames = []
    self.scrub.setRange(0, 0)
    self.canvas.set_congestion(None)
    self._job_kind = 'run'
    self._job_t0 = time.time()
    self._run_rows_start = 0
    if hasattr(self, 'eng_spins'):
      wl_w = float(self.eng_spins['wirelength_weight'].value())
      cg_w = float(self.eng_spins['congestion_weight'].value())
      dn_w = float(self.eng_spins['density_weight'].value())
      th_w = float(self.eng_spins['thermal_weight'].value())
    else:
      eng = self._settings['engine']
      wl_w = float(eng.get('wirelength_weight', 1.0))
      cg_w = float(eng.get('congestion_weight', 0.5))
      dn_w = float(eng.get('density_weight', 1.0))
      th_w = float(eng.get('thermal_weight', 0.0))
    thermal_fix = self._thermal_fix
    if self._all_hard_pinned():
      QMessageBox.warning(
          self, 'Nothing left to place',
          'Every hard macro is 🔒 pinned. Unpin at least one macro '
          '(right-click it) so the placer has something to place.')
      return
    init_path = self._effective_init_plc(netlist, plc_path)
    do_polish = bool(hasattr(self, 'run_polish')
                     and self.run_polish.isChecked())
    do_orient = bool(hasattr(self, 'run_orient')
                     and self.run_orient.isChecked())
    pinned_names = set(self._pinned)
    self._set_left_visible(False)
    self.progress_label.setText(
        'Starting live placement (0–60%), then scored baseline '
        'episodes (60–100%). Details stream into Results.')
    self.tabs.setCurrentIndex(2)

    def job(w):
      from neurosilicon import evaluate as evaluate_lib
      from neurosilicon import priors as priors_lib
      from neurosilicon import setup as setup_lib
      extra_costs = None
      if thermal_fix or th_w > 0:
        try:
          power0 = liveplace_lib.scaled_power(
              setup_lib.create_plc(netlist, init_path))
          extra_costs = [(
              'thermal', max(th_w, 0.5),
              lambda plc, _p=power0:
              priors_lib.thermal_overlap_penalty(plc, _p))]
          w.log.emit('AI thermal fix: hot-macro overlap penalty '
                     f'active (weight {max(th_w, 0.5)}).')
        except Exception as e:
          w.log.emit(f'thermal fix unavailable: {e}')
      mask_fn = None
      if keepouts:
        self._keepout_mode = keepout_mode
        self._keepouts = keepouts
        mask_fn = self._mask_fn(netlist, init_path)
        w.log.emit(f'Chat constraint active ({keepout_mode}): '
                   f'{len(keepouts)} rect(s)')
      if animate:
        w.log.emit(f'Live engine ({method}): placing macros '
                   'sequentially, largest first…')
        info, _frames = liveplace_lib.run_live_rollout(
            netlist, init_path, fd_steps=(fd,) * 3, method=method,
            seed=seed, extra_mask_fn=mask_fn, init_fd=False,
            on_snapshot=lambda f: w.snap.emit(f),
            on_live=lambda i: w.live.emit(i))
        w.log.emit(f"live rollout proxy: "
                   f"{info.get('proxy_cost', -1):.4f}")
        w.live.emit(dict(info))
        w.phase.emit(
            f'Live placement scored (cards above). Now running '
            f'{2 * episodes} scored baseline episodes (60–100%) — '
            'each rebuilds the design and can take minutes on large '
            'chips like ariane; rows land in Results as they finish.')
        w.log.emit('Baselines: fast desktop mode (construction FD '
                   'skipped; terminal relaxation still applied).')
      results = evaluate_lib.run_baselines(
          netlist, init_path, episodes=episodes, fd_steps=(fd,) * 3,
          outdir=outdir, seed=seed, finetune_epochs=finetune,
          extra_mask_fn=mask_fn, init_fd=False,
          wirelength_weight=wl_w, congestion_weight=cg_w,
          density_weight=dn_w, extra_costs=extra_costs,
          progress=lambda r: w.row.emit(dict(r)))
      polish = None
      if (do_polish or do_orient) and \
              any(r['proxy_cost'] > 0 for r in results):
        try:
          from neurosilicon.thirdparty.tilos.Plc_client import (
              placement_util_os as placement_util)
          plc_b = setup_lib.create_plc(
              netlist, os.path.join(outdir, 'best_baseline.plc'))
          orient, ob, oa = liveplace_lib.optimize_orientations(plc_b)
          if not do_orient:
            orient, oa = {}, ob
          polish = {'hpwl_before': ob, 'hpwl_after': oa,
                    'orientations': orient, 'accepted': 0,
                    'path': None}
          if do_polish:
            sa = liveplace_lib.simulated_annealing(
                plc_b, steps=200, seed=seed, pinned=pinned_names,
                orientations=orient)
            liveplace_lib.apply_positions(plc_b, sa['positions'])
            polished = os.path.join(outdir, 'best_polished.plc')
            placement_util.save_placement(plc_b, polished)
            polish.update({'hpwl_after': sa['hpwl_after'],
                           'orientations': sa['orientations'],
                           'accepted': sa['accepted'],
                           'path': polished})
          gain = (polish['hpwl_before'] - polish['hpwl_after']) \
              / max(polish['hpwl_before'], 1e-9) * 100
          w.log.emit(
              f"SA polish + orientation flips: pin-HPWL "
              f"{polish['hpwl_before']:.0f} → "
              f"{polish['hpwl_after']:.0f} µm ({gain:+.1f}%, "
              f"{polish['accepted']} accepted moves, pinned held).")
        except Exception as e:
          w.log.emit(f'SA polish unavailable: {e}')
      return {'results': results, 'polish': polish}

    def done(result):
      results = result['results']
      polish = result.get('polish')
      best_plc = os.path.join(outdir, 'best_baseline.plc')
      if polish:
        self._orientations = dict(polish.get('orientations') or {})
        self.canvas.set_orientations(self._orientations)
        if polish.get('path'):
          best_plc = polish['path']
        gain = (polish['hpwl_before'] - polish['hpwl_after']) \
            / max(polish['hpwl_before'], 1e-9) * 100
        self.log(f"Polish applied: HPWL {polish['hpwl_before']:.0f} "
                 f"→ {polish['hpwl_after']:.0f} µm ({gain:+.1f}%). "
                 'Orientation flips feed the OpenROAD Tcl export.')
      self._best_plc_path = best_plc
      try:
        from neurosilicon import render as render_lib
        from neurosilicon import setup as setup_lib
        render_lib.save_figures(
            setup_lib.create_plc(netlist, best_plc), outdir,
            power=getattr(self, '_power', None) or None)
      except Exception as e:
        self.log(f'figure export failed: {e}')
      try:
        self._load_plc_to_canvas(netlist, best_plc, keep_frames=True)
      except Exception as e:
        self.log(f'best-placement draw failed: {e}')
      try:
        ok_rows = [r for r in results if r['proxy_cost'] > 0]
        if ok_rows and self._current_plc is not None:
          n = len(self._run_snapshots) + 1
          self._record_snapshot(self._snapshot_for(
              self._current_plc,
              min(ok_rows, key=lambda r: r['proxy_cost']),
              f'Run #{n}'))
      except Exception as e:
        self.log(f'compare snapshot unavailable: {e}')
      live = getattr(self, '_live_best', None)
      if live and live.get('proxy_cost', -1) > 0:
        self._live_row(
            {'method': 'live (first_legal)', 'episode': 0,
             'proxy_cost': float(live['proxy_cost']),
             'wirelength': float(live['wirelength']),
             'congestion': float(live['congestion']),
             'density': float(live['density']),
             'seconds': time.time() - self._job_t0},
            count_progress=False)
      new_ok = [r for r in self._results if r['proxy_cost'] > 0]
      if new_ok:
        self._note_run_delta(
            min(new_ok, key=lambda r: r['proxy_cost']))
      best = min((r['proxy_cost'] for r in results
                  if r['proxy_cost'] > 0), default=-1)
      ok = [r for r in results if r['proxy_cost'] > 0]
      if ok:
        b = min(ok, key=lambda r: r['proxy_cost'])
        self.progress_label.setText(
            f"Done — best proxy {b['proxy_cost']:.4f} (wirelength "
            f"{b['wirelength']:.4f}, congestion {b['congestion']:.4f}, "
            f"density {b['density']:.4f}). Full details in Results.")
      self._refresh_details()
      self.log(f'Done. Best proxy {best:.4f}. Files in {outdir} — '
               'download the placed .plc / DEF from Results.')
      self.tabs.setCurrentIndex(2)
    self._start(job, done)

  # -- downloads ----------------------------------------------------------
  def _require_best(self):
    if self._best_plc_path and os.path.isfile(self._best_plc_path):
      return self._best_plc_path
    QMessageBox.warning(
        self, 'Nothing placed yet',
        'Run Place → Run floorplan first; downloads export the best '
        'placement of that run.')
    return ''

  def _save_plc(self):
    src = self._require_best()
    if not src:
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, 'Download placed file (.plc)', 'placed.plc',
        'Placement (*.plc)')
    if dst:
      import shutil
      shutil.copyfile(src, dst)
      self.log(f'Placed file saved: {dst}')

  def _save_def(self):
    src = self._require_best()
    if not src or not self._plc_pair:
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, 'Download DEF', 'placed.def', 'DEF (*.def)')
    if not dst:
      return
    from neurosilicon import export_def as export_def_lib
    from neurosilicon import setup as setup_lib
    plc = setup_lib.create_plc(self._plc_pair[0], src)
    export_def_lib.export_def(plc, dst)
    self.log(f'DEF saved: {dst}')

  def _save_tcl(self):
    src = self._require_best()
    if not src or not self._plc_pair:
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, 'Download OpenROAD .tcl', 'placed_openroad.tcl',
        'OpenROAD Tcl (*.tcl)')
    if not dst:
      return
    from neurosilicon import export_def as export_def_lib
    from neurosilicon import setup as setup_lib
    plc = setup_lib.create_plc(self._plc_pair[0], src)
    export_def_lib.export_openroad_tcl(plc, dst, self._orientations)
    def_path = os.path.join(os.path.dirname(os.path.abspath(dst)),
                            'placed.def')
    export_def_lib.export_def(plc, def_path)
    self.log(f'OpenROAD Tcl saved: {dst} (+ placed.def alongside: '
             f'{def_path}) — place_macro lines carry the optimizer '
             'orientation flips as R0/R90/R180/R270.')

  def _save_gif(self):
    if not self._frames:
      QMessageBox.warning(
          self, 'No timelapse yet',
          'Run Place → Run floorplan (or click a design for the live '
          'simulation) first; the playback frames become the GIF.')
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, 'Export Timelapse GIF', 'placement_timelapse.gif',
        'GIF (*.gif)')
    if not dst:
      return
    try:
      import tempfile

      from PIL import Image
      tmp = tempfile.mkdtemp(prefix='ns_gif_')
      canvas = ChipCanvas()
      canvas.resize(760, 540)
      canvas.set_view('Floorplan')
      paths = []
      for i, frame in enumerate(self._frames):
        canvas.set_frame(frame)
        canvas.set_power(liveplace_lib.frame_power(frame))
        pix = QPixmap(760, 540)
        canvas.render(pix)
        path = os.path.join(tmp, f'frame_{i:04d}.png')
        pix.save(path)
        paths.append(path)
      images = [Image.open(p).convert('RGB') for p in paths]
      images[0].save(dst, save_all=True, append_images=images[1:],
                     duration=90, loop=0)
      self.log(f'Timelapse GIF saved: {dst} ({len(images)} frames '
               'from the placement playback history).')
    except Exception as e:
      QMessageBox.warning(self, 'GIF export failed', str(e))
      self.log(f'GIF export failed: {e}')

  def _save_png(self, kind):
    src = self._require_best()
    if not src or not self._plc_pair:
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, f'Download {kind} PNG', f'{kind}.png', 'PNG (*.png)')
    if not dst:
      return
    import matplotlib.pyplot as plt
    from neurosilicon import render as render_lib
    from neurosilicon import setup as setup_lib
    plc = setup_lib.create_plc(self._plc_pair[0], src)
    fig = {'floorplan': lambda: render_lib.plot_floorplan(plc),
           'thermal': lambda: render_lib.plot_thermal(
               plc, getattr(self, '_power', None)),
           'congestion': lambda: render_lib.plot_congestion(plc)}[kind]()
    with open(dst, 'wb') as f:
      f.write(render_lib.figure_png(fig))
    plt.close(fig)
    self.log(f'{kind} PNG saved: {dst}')

  def _save_zip(self):
    if not os.path.isdir(self._outdir):
      QMessageBox.warning(self, 'No run yet', 'Run a floorplan first.')
      return
    dst, _ = QFileDialog.getSaveFileName(
        self, 'Download report ZIP', 'neurosilicon_report.zip',
        'ZIP (*.zip)')
    if not dst:
      return
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
      for dirpath, _, files in os.walk(self._outdir):
        for fn in files:
          full = os.path.join(dirpath, fn)
          z.write(full, os.path.relpath(full, self._outdir))
    self.log(f'Report ZIP saved: {dst}')

  def open_outdir(self):
    import subprocess
    os.makedirs(self._outdir, exist_ok=True)
    subprocess.Popen(['xdg-open', self._outdir])

  # -- AI chatbox ---------------------------------------------------------
  def _chat_context(self):
    canvas = (0.0, 0.0)
    if self._frames:
      canvas = self._frames[-1]['canvas']
    elif self._plc_pair:
      from neurosilicon import setup as setup_lib
      try:
        hdr = setup_lib.parse_plc_header(self._plc_pair[1])
        canvas = (float(hdr.get('width', 0)), float(hdr.get('height', 0)))
      except OSError:
        pass
    power = getattr(self, '_power', {}) or {}
    hottest = sorted(power.items(), key=lambda kv: -kv[1])[:5]
    item = self.design_list.currentItem()
    ca = getattr(self, '_congestion_analysis', None)
    if ca is None and self._current_plc is not None:
      try:
        self._ensure_congestion()
        ca = getattr(self, '_congestion_analysis', None)
      except Exception:
        ca = None
    if ca is not None:
      ca = {k: v for k, v in ca.items() if k != 'grid'}
    return {'design': item.text() if item else 'no design selected',
            'canvas': canvas, 'results': list(self._results),
            'best_proxy': min((r['proxy_cost'] for r in self._results
                               if r['proxy_cost'] > 0), default=None),
            'hottest': hottest, 'congestion_analysis': ca}

  def _maybe_constraint(self, text):
    """Binds chat text into Place state: mask rects and/or weights."""
    from neurosilicon import priors as priors_lib
    ctx = self._chat_context()
    w, h = ctx['canvas']
    if w <= 0:
      return
    low = text.lower()
    if any(k in low for k in ('hot', 'thermal', 'spread', 'heat',
                              'temperature', 'cool')):
      # Thermal fix: raise thermal + congestion pressure for re-place.
      self._thermal_fix = True
      if hasattr(self, 'eng_spins'):
        self.eng_spins['thermal_weight'].setValue(
            max(float(self.eng_spins['thermal_weight'].value()), 0.5))
        self.eng_spins['congestion_weight'].setValue(
            max(float(self.eng_spins['congestion_weight'].value()),
                0.8))
      self.constraint_label.setText(
          'Chat constraints: thermal spread fix ACTIVE (hot-macro '
          'overlap penalty + congestion weight raised) — press '
          'Apply AI Fix & Re-Place.')
      self.log('Chat thermal fix bound: thermal/congestion weights '
               'raised for the next placement.')
    if not any(k in low for k in ('keep', 'reserve', 'channel', 'edge',
                                  'perimeter', 'center', 'centre',
                                  'constrain', 'block', 'allow')):
      return
    mode, rects, note = priors_lib.chat_to_spec(text, w, h)
    if mode == 'none' or not rects:
      return
    self._keepouts = [tuple(r) for r in rects]
    self._keepout_mode = mode
    self.canvas.set_keepouts(self._keepouts)
    suffix = (' + thermal spread fix' if self._thermal_fix else '')
    self.constraint_label.setText(
        f'Chat constraint ({mode}): {note}{suffix} — applies to the '
        'next run / Apply AI Fix & Re-Place.')
    self.log(f'Chat constraint ({mode}): {note}')

  def _clear_constraint(self):
    self._keepouts = []
    self._thermal_fix = False
    self.canvas.set_keepouts([])
    self.constraint_label.setText('Chat constraint: none')
    self.log('Chat constraint cleared.')

  def _apply_ai_fix(self):
    """Closed loop: active AI constraint + weights -> re-place now."""
    if not self.design_list.currentItem():
      self._chat_append('Assistant', 'Pick a design first, then '
                                     'Apply AI Fix & Re-Place.')
      return
    if not self._keepouts and not self._thermal_fix:
      self._chat_append(
          'Assistant',
          'No active AI constraint yet. Type one first — e.g. '
          '"Keep a 20% routing channel in the center" or "Spread '
          'hot macros apart" — then press Apply AI Fix & Re-Place.')
      return
    parts = []
    if self._keepouts:
      parts.append(f'{len(self._keepouts)} keep-out rect(s) '
                   f'({self._keepout_mode})')
    if self._thermal_fix:
      parts.append('thermal-spread penalty + raised congestion '
                   'weight')
    self._chat_append(
        'Assistant',
        'Applying AI fix: ' + ' + '.join(parts) + '. Re-placing '
        'now — the new run lands in Results below the previous '
        'one, with Δ% vs the last run.')
    self.log('Apply AI Fix & Re-Place: ' + ' + '.join(parts))
    self.run_floorplan()

  def auto_optimize(self):
    """⚡ Auto-Optimize (AI Loop): baseline → diagnosis → re-place.

    One background job runs all 3 iterations without user typing.
    Every episode row streams into Results; the PPA Improvement
    Summary and per-iteration notes post in the Assistant chat.
    """
    item = self.design_list.currentItem()
    if not item:
      self._chat_append('Assistant', '⚡ Auto-Optimize (AI Loop): '
                                     'pick a design first, then press '
                                     '⚡ again.')
      return
    if self._thread is not None:
      self.log('Busy — Auto-Optimize waits for the current job.')
      return
    netlist, plc_path = self._designs[item.text()]
    episodes = max(1, self.run_episodes.value())
    fd, seed = self.run_fd.value(), self.run_seed.value()
    finetune = self.run_finetune.value()
    if hasattr(self, 'eng_spins'):
      wl_w = float(self.eng_spins['wirelength_weight'].value())
      cg_w = float(self.eng_spins['congestion_weight'].value())
      dn_w = float(self.eng_spins['density_weight'].value())
      th_w = float(self.eng_spins['thermal_weight'].value())
    else:
      eng = self._settings['engine']
      wl_w, cg_w, dn_w, th_w = (float(eng.get('wirelength_weight', 1.0)),
                                float(eng.get('congestion_weight', 0.5)),
                                float(eng.get('density_weight', 1.0)),
                                float(eng.get('thermal_weight', 0.0)))
    llm_settings = (self._llm_form_settings()
                    if hasattr(self, 'ai_provider')
                    else dict(self._settings['llm']))
    if self._all_hard_pinned():
      self._chat_append('Assistant', '⚡ Auto-Optimize stopped: every '
                                     'hard macro is 🔒 pinned. Unpin '
                                     'at least one macro first.')
      return
    init_path = self._effective_init_plc(netlist, plc_path)
    do_polish = bool(hasattr(self, 'run_polish')
                     and self.run_polish.isChecked())
    do_orient = bool(hasattr(self, 'run_orient')
                     and self.run_orient.isChecked())
    pinned_names = set(self._pinned)
    base_out = os.path.join(project_lib.repo_root(), 'logs',
                            'desktop_runs', 'auto_baseline')
    ai_out = os.path.join(project_lib.repo_root(), 'logs',
                          'desktop_runs', 'auto_ai')
    self._job_kind = 'auto'
    self._job_t0 = time.time()
    self._set_right_visible(True)
    self.right_tabs.setCurrentIndex(0)
    self._chat_append('Assistant', '⚡ Auto-Optimize (AI Loop) started '
                                   '— Iter 1/3: baseline placement…')
    self.progress_label.setText('⚡ Auto-Optimize Iter 1/3: baseline '
                                'placement…')

    def job(w):
      from neurosilicon import evaluate as evaluate_lib
      from neurosilicon import priors as priors_lib
      from neurosilicon import setup as setup_lib

      def tagged(row, prefix):
        out = dict(row)
        out['method'] = f"{prefix} · {row['method']}"
        return out

      def snap(plc, rows, label):
        hard = sum(1 for i in plc.get_macro_indices()
                   if not plc.is_node_soft_macro(i))
        power = liveplace_lib.scaled_power(plc)
        try:
          grid = liveplace_lib.congestion_analysis(plc)['grid']
        except Exception:
          grid = None
        best = min((r for r in rows if r['proxy_cost'] > 0),
                   key=lambda r: r['proxy_cost'], default=None)
        return {'label': label,
                'frame': liveplace_lib.snapshot(plc, hard, hard, ''),
                'power': power, 'grid': grid, 'row': dict(best or {}),
                'ppa': liveplace_lib.ppa_metrics(plc, power)}, best

      w.phase.emit('⚡ Iter 1/3 — Baseline placement (proxy, '
                   'wirelength, RUDY congestion, thermal map)…')
      base_rows = evaluate_lib.run_baselines(
          netlist, init_path, episodes=episodes, fd_steps=(fd,) * 3,
          outdir=base_out, seed=seed, finetune_epochs=finetune,
          init_fd=False, wirelength_weight=wl_w,
          congestion_weight=cg_w, density_weight=dn_w,
          progress=lambda r: w.row.emit(tagged(r, 'Baseline')))
      if not any(r['proxy_cost'] > 0 for r in base_rows):
        return {'error': 'Baseline placement was infeasible — loosen '
                         'constraints and retry Auto-Optimize.'}
      base_plc = setup_lib.create_plc(
          netlist, os.path.join(base_out, 'best_baseline.plc'))
      base_snap, base_best = snap(base_plc, base_rows, 'Run #1 — Baseline')
      w.phase.emit('⚡ Iter 2/3 — SAM hotspot segmentation + '
                   'muse-arcadia-1.0 diagnosis…')
      diag = assistant_lib.diagnose_placement(base_plc, llm_settings,
                                              cg_w, th_w)
      w.log.emit(f"Auto-Optimize diagnosis ({diag['source']}): "
                 f"{diag['actions']} congestion_weight="
                 f"{diag['congestion_weight']} thermal_weight="
                 f"{diag['thermal_weight']} — {diag['explanation']}")
      template = setup_lib.create_plc(netlist, init_path)
      W, H = template.get_canvas_width_height()
      cols, rows_g = template.get_grid_num_columns_rows()
      mask_fn = None
      if diag.get('rects'):
        mask_fn = (priors_lib.make_allow_fn
                   if diag.get('mode') == 'allow'
                   else priors_lib.make_keepout_fn)(
                       cols, rows_g, W, H, [tuple(r)
                                            for r in diag['rects']])
      power0 = liveplace_lib.scaled_power(template)
      extra = [('thermal', float(diag['thermal_weight']),
                lambda plc, _p=power0:
                priors_lib.thermal_overlap_penalty(plc, _p))] \
          if float(diag.get('thermal_weight', 0.0)) > 0 else None
      w.phase.emit('⚡ Iter 3/3 — AI-guided re-placement with the '
                   'diagnosed constraints + weights…')
      ai_rows = evaluate_lib.run_baselines(
          netlist, init_path, episodes=episodes, fd_steps=(fd,) * 3,
          outdir=ai_out, seed=seed, finetune_epochs=finetune,
          init_fd=False, extra_mask_fn=mask_fn,
          wirelength_weight=wl_w,
          congestion_weight=float(diag['congestion_weight']),
          density_weight=dn_w, extra_costs=extra,
          progress=lambda r: w.row.emit(tagged(r, 'AI')))
      retried = False
      if not any(r['proxy_cost'] > 0 for r in ai_rows) and mask_fn is not None:
        retried = True
        w.log.emit('AI keep-out made placement infeasible; retrying '
                   'weights-only (thermal/congestion pressure kept).')
        ai_rows = evaluate_lib.run_baselines(
            netlist, init_path, episodes=episodes, fd_steps=(fd,) * 3,
            outdir=ai_out + '_retry', seed=seed,
            finetune_epochs=finetune, init_fd=False,
            wirelength_weight=wl_w,
            congestion_weight=float(diag['congestion_weight']),
            density_weight=dn_w, extra_costs=extra,
            progress=lambda r: w.row.emit(tagged(r, 'AI')))
        ai_dir = ai_out + '_retry'
      else:
        ai_dir = ai_out
      if not any(r['proxy_cost'] > 0 for r in ai_rows):
        return {'error': 'AI-guided re-placement was infeasible even '
                         'after the weights-only retry.',
                'diagnosis': diag}
      ai_plc = setup_lib.create_plc(
          netlist, os.path.join(ai_dir, 'best_baseline.plc'))
      ai_path = os.path.join(ai_dir, 'best_baseline.plc')
      polish = None
      if do_polish or do_orient:
        try:
          from neurosilicon.thirdparty.tilos.Plc_client import (
              placement_util_os as placement_util)
          orient, ob, oa = liveplace_lib.optimize_orientations(ai_plc)
          if not do_orient:
            orient, oa = {}, ob
          polish = {'hpwl_before': ob, 'hpwl_after': oa,
                    'orientations': orient, 'accepted': 0}
          if do_polish:
            sa = liveplace_lib.simulated_annealing(
                ai_plc, steps=200, seed=seed, pinned=pinned_names,
                orientations=orient)
            liveplace_lib.apply_positions(ai_plc, sa['positions'])
            ai_path = os.path.join(ai_dir, 'best_polished.plc')
            placement_util.save_placement(ai_plc, ai_path)
            polish.update({'hpwl_after': sa['hpwl_after'],
                           'orientations': sa['orientations'],
                           'accepted': sa['accepted']})
          w.log.emit(
              f"Auto-Optimize polish: pin-HPWL {ob:.0f} → "
              f"{polish['hpwl_after']:.0f} µm (SA swaps + flips).")
        except Exception as e:
          w.log.emit(f'Auto-Optimize polish unavailable: {e}')
          polish = None
      ai_snap, ai_best = snap(ai_plc, ai_rows, 'Best AI-Optimized')
      return {'baseline': base_snap, 'optimized': ai_snap,
              'diagnosis': diag, 'retried': retried, 'polish': polish,
              'summary': liveplace_lib.format_ppa_summary(
                  base_snap['ppa'], ai_snap['ppa']),
              'ai_plc': ai_path,
              'base_plc': os.path.join(base_out, 'best_baseline.plc')}

    def done(result):
      if result.get('error'):
        self._chat_append('Assistant', '⚡ Auto-Optimize stopped: '
                                       + result['error'])
        self.log('Auto-Optimize: ' + result['error'])
        return
      diag = result['diagnosis']
      b, a = result['baseline']['ppa'], result['optimized']['ppa']
      self._compare_left = result['baseline']
      self._compare_right = result['optimized']
      self._run_snapshots.append(result['baseline'])
      self._run_snapshots.append(result['optimized'])
      best_plc = (result['ai_plc']
                  if a['proxy_cost'] <= b['proxy_cost']
                  else result['base_plc'])
      self._best_plc_path = best_plc
      self._outdir = os.path.dirname(best_plc)
      try:
        self._load_plc_to_canvas(netlist, best_plc, keep_frames=True)
      except Exception as e:
        self.log(f'Auto-Optimize canvas draw failed: {e}')
      if result.get('polish'):
        self._orientations = dict(
            result['polish'].get('orientations') or {})
        self.canvas.set_orientations(self._orientations)
        self.log(f"Polish orientations applied "
                 f"({len([d for d in self._orientations.values() if d])} "
                 'flips) — notches rotated, Tcl exports R90/R180/R270.')
      if hasattr(self, 'eng_spins'):
        self.eng_spins['congestion_weight'].setValue(
            float(diag['congestion_weight']))
        self.eng_spins['thermal_weight'].setValue(
            float(diag['thermal_weight']))
      self._keepouts = [tuple(r) for r in diag.get('rects', [])]
      self._keepout_mode = diag.get('mode', 'block')
      self._thermal_fix = True
      self.canvas.set_keepouts(self._keepouts)
      self.constraint_label.setText(
          f"AI diagnosis ({diag['source']}): {diag['explanation']} "
          f"[congestion {diag['congestion_weight']}, thermal "
          f"{diag['thermal_weight']}]")
      hot = diag.get('hotspot') or {}
      self._chat_append(
          'Assistant',
          '⚡ Auto-Optimize (AI Loop) complete.\n'
          f"Iter 1 Baseline: proxy {b['proxy_cost']:.4f}, wirelength "
          f"{b['wirelength']:.4f}, congestion {b['congestion']:.4f}, "
          f"peak macro {b['peak_macro']}, est. Fmax "
          f"{b['fmax_mhz']:.0f} MHz.\n"
          f"Iter 2 Diagnosis ({diag['source']}): SAM hotspot RUDY "
          f"peak {hot.get('peak_congestion', 0.0):.3f} → "
          f"{', '.join(diag['actions'])}; congestion_weight="
          f"{diag['congestion_weight']}, thermal_weight="
          f"{diag['thermal_weight']}. {diag['explanation']}\n"
          f"Iter 3 AI-guided: proxy {a['proxy_cost']:.4f}, "
          f"wirelength {a['wirelength']:.4f}, congestion "
          f"{a['congestion']:.4f}, est. Fmax {a['fmax_mhz']:.0f} MHz"
          f"{' (weights-only retry)' if result.get('retried') else ''}"
          '.\n' + result['summary'] +
          '\nView → Compare (Baseline vs. AI) shows the split-screen '
          'Before/After with Δ badges. Both runs are logged in Results.')
      self.log('Auto-Optimize complete. ' + result['summary'].splitlines()[1].strip())
      self.tabs.setCurrentIndex(2)
      self._refresh_details()
    self._start(job, done)

  def _suggest_rtl_fix(self):
    """Suggest RTL Fix for Bottleneck (Muse co-pilot)."""
    if self._thread is not None:
      self.log('Busy — RTL suggestion waits for the current job.')
      return
    roots = []
    if self._disc and self._disc.get('root'):
      roots.append(self._disc['root'])
    imp = self.imp_path.text().strip() if hasattr(self, 'imp_path') else ''
    if imp:
      roots.append(imp if os.path.isdir(imp) else os.path.dirname(imp))
    for rel in self._checked_files():
      for root in list(roots):
        full = os.path.join(root, rel)
        if os.path.isfile(full):
          roots.append(full)
    plc = self._current_plc
    if plc is None and not roots:
      self._set_right_visible(True)
      self.right_tabs.setCurrentIndex(0)
      self._chat_append('Assistant', 'Suggest RTL Fix needs a '
                                     'placement or imported RTL: run '
                                     'Place first, or Import an RTL '
                                     'project (e.g. Amber with '
                                     'a25_dcache.v) and press the '
                                     'button again.')
      return
    llm_settings = (self._llm_form_settings()
                    if hasattr(self, 'ai_provider')
                    else dict(self._settings['llm']))
    power = getattr(self, '_power', None) or None
    self._set_right_visible(True)
    self.right_tabs.setCurrentIndex(0)
    self._chat_append('You', 'Suggest RTL Fix for Bottleneck')

    def job(_w):
      return assistant_lib.bottleneck_rtl_advice(plc, roots,
                                                 llm_settings, power)

    def done(result):
      text, used, info = result
      self._chat_append('Assistant', text)
      self.log(f"RTL fix for {info.get('macro', '?')} "
               f"({info.get('reason', '')}, file "
               f"{info.get('file') or '—'}, "
               f"{'LLM' if used else 'offline template'}).")
    self._job_kind = 'rtl'
    self._start(job, done)

  def _run_agent_review(self):
    """Run 3-Agent Design Review (Thermal/Routing/Lead Architect)."""
    if self._thread is not None:
      self.log('Busy — 3-Agent Review waits for the current job.')
      return
    self._set_right_visible(True)
    self.right_tabs.setCurrentIndex(0)
    self._chat_append('You', 'Run 3-Agent Design Review')
    ctx = self._chat_context()
    plc = self._current_plc
    crit = dict(self._crit_result) if self._crit_result else {}

    def job(_w):
      ppa = {}
      if plc is not None:
        try:
          ppa = liveplace_lib.ppa_metrics(plc)
        except Exception:
          ppa = {}
      full = dict(ctx)
      full['ppa'] = ppa
      full['critical'] = crit
      return assistant_lib.multi_agent_review(full)

    def done(result):
      text, weights = result
      self._chat_append('Assistant', text)
      if hasattr(self, 'eng_spins'):
        for key, value in weights.items():
          if key in self.eng_spins:
            self.eng_spins[key].setValue(float(value))
      self.log('3-Agent Review: recommended weights applied to '
               'Settings (wirelength/congestion/thermal spins).')
    self._job_kind = 'review'
    self._start(job, done)

  def _chat_append(self, who, text):
    holder = self._bubble(who, text)
    self.chat_msgs.insertWidget(self.chat_msgs.count() - 1, holder)
    QTimer.singleShot(60, self._chat_to_bottom)

  def _chat_to_bottom(self):
    bar = self.chat_scroll.verticalScrollBar()
    bar.setValue(bar.maximum())

  def _chat_import_project(self):
    """Chat import entry: folder picker -> shared Import-tab flow.

    Opens a folder (never a single file); discovery then auto-fetches
    every RTL project file inside it and auto-picks the top module.
    """
    path = QFileDialog.getExistingDirectory(
        self, 'Import chip project folder')
    if path:
      self._chat_import_path(path)

  def _chat_import_path(self, path):
    """Detect a chat-picked/typed project folder and route the import.

    A single file (typed `import <file>` or a legacy pick) resolves to
    its containing folder, so the whole RTL project is fetched, not
    just that file. Ready netlist pairs register immediately
    (do_import); RTL projects open pre-filled in the Import tab so the
    user confirms the top module and file list before synthesis.
    """
    path = path.strip().strip('"').strip("'")
    if self._thread is not None:
      self._chat_append('Neuro AI', 'Busy — import waits for the '
                                    'current job to finish.')
      return
    if not os.path.exists(path):
      self._chat_append('Neuro AI', f'Import: path not found: {path}')
      return
    if os.path.isfile(path) and not path.lower().endswith('.zip'):
      path = os.path.dirname(os.path.abspath(path))
    self.imp_path.setText(path)
    self._disc = None
    self.tabs.setCurrentIndex(0)
    self.detect_import()
    d = self._disc
    if d is None:
      self._chat_append('Neuro AI', 'Import: could not read that '
                                    'path — see the Import tab for '
                                    'details.')
      return
    found = d['found'][0] if d['found'] else d['kind']
    if d['kind'] == 'ready_netlist':
      self._chat_append('Neuro AI', f'Import: {found} Registering '
                                    'it now…')
      self.do_import()
    elif d['kind'] == 'unknown':
      missing = '; '.join(d['missing']) or 'nothing importable found'
      self._chat_append('Neuro AI', f'Import: {missing}. Pick the '
                                    'folder holding the RTL or the '
                                    'netlist pair instead.')
    else:
      self._chat_append(
          'Neuro AI',
          f'Import: {found} Auto-fetched {len(d["rtl_files"])} RTL '
          f'project file(s) from the folder and opened them in the '
          f"Import tab — check the top module "
          f"({d['suggested_top'] or 'pick one'}) and the checked "
          'files, then press Import design to synthesize.')

  def _on_suggestion_picked(self, index):
    """Suggestion dropdown: ask the picked prompt, then reset."""
    if index <= 0:
      return
    prompt = self.chat_suggest.itemText(index)
    self.chat_suggest.setCurrentIndex(0)
    self._chat_send(prompt)

  def _chat_send(self, preset=None):
    text = (preset or self.chat_input.text()).strip()
    if not text:
      return
    self.chat_input.clear()
    if preset is None and text.lower().startswith('import '):
      candidate = text[7:].strip().strip('"').strip("'")
      if os.path.exists(candidate):
        self._chat_append('You', text)
        self._chat_import_path(candidate)
        return
    self._chat_append('You', text)
    self._maybe_constraint(text)
    self._settings['llm'] = self._llm_form_settings() \
        if hasattr(self, 'ai_provider') else self._settings['llm']
    ctx = self._chat_context()
    llm_settings = dict(self._settings['llm'])
    if llm_settings.get('provider', 'off') == 'off' or self._thread:
      reply, _used = assistant_lib.answer(text, ctx, llm_settings)
      self._chat_append('Assistant', reply)
      return

    def job(_w):
      return assistant_lib.answer(text, ctx, llm_settings)

    def done(result):
      reply, used = result
      self._chat_append('Assistant', reply)
      self.log('Assistant answered via LLM.' if used
               else 'Assistant answered offline.')
    self._start(job, done)


def main():
  if '--auto-record-all' in sys.argv:
    argv = [a for a in sys.argv if a != '--auto-record-all']
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'recordings_phase1')
    if '--record-dir' in argv:
      i = argv.index('--record-dir')
      outdir = argv[i + 1]
      del argv[i:i + 2]
    sys.argv = argv
    os.makedirs(outdir, exist_ok=True)
    from neurosilicon import auto_record
    sys.exit(auto_record.run_desktop_auto_record(outdir))
  app = QApplication(sys.argv)
  app.setApplicationName('NeuroSilicon')
  win = MainWindow()
  win.show()
  sys.exit(app.exec())


if __name__ == '__main__':
  main()
