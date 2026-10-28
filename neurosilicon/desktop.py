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

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Qt
from PySide6.QtGui import (QAction, QColor, QFont, QPainter, QPen)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QDoubleSpinBox, QSplitter, QTableWidget,
    QFrame, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout,
    QWidget)

from neurosilicon import assistant as assistant_lib
from neurosilicon import discover as discover_lib
from neurosilicon import liveplace as liveplace_lib
from neurosilicon import project as project_lib
from neurosilicon import settings as settings_lib

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
           'Floorplan + Wires', 'Thermal + Wires')

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setMinimumSize(340, 240)
    self._frame = None
    self._view = 'Floorplan'
    self._power = {}
    self._segments = []
    self._congestion = None  # (rows x cols) array-like or None
    self._keepouts = []

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

  def paintEvent(self, _event):
    p = QPainter(self)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(self.rect(), QColor('#0d1117'))
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
    s = min((self.width() - 2 * margin) / W,
            (self.height() - 2 * margin) / H)
    ox, oy = (self.width() - W * s) / 2, (self.height() - H * s) / 2

    def X(x):
      return ox + x * s

    def Y(y):
      return oy + (H - y) * s

    p.fillRect(int(ox), int(oy), int(W * s), int(H * s),
               QColor('#0f172a'))
    cols, rows = frame['grid']
    base = self._view.split(' +')[0]
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
      x, y, w, h = (X(m['x'] - m['w'] / 2), Y(m['y'] + m['h'] / 2),
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
        p.fillRect(int(x), int(y), int(w), int(h),
                   QColor(2, 132, 199, 150))
        p.setPen(QPen(QColor('#38bdf8'), 1))
        p.drawRect(int(x), int(y), int(w), int(h))
      if m['name'] == frame.get('current') and not m['soft']:
        p.setPen(QPen(QColor('#fbbf24'), 3))
        p.drawRect(int(x), int(y), int(w), int(h))
      if w > 52 and h > 12 and not m['soft']:
        p.setPen(QColor('#e6edf3'))
        p.drawText(int(x) + 2, int(y) + 2, int(w) - 4, int(h) - 4,
                   Qt.AlignCenter | Qt.TextWordWrap, m['name'])
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
    self._keepouts = []       # [(x0,y0,x1,y1)] chat keep-out rects
    self._keepout_mode = 'block'
    self._t0 = None
    self._build_ui()
    self.refresh_designs()
    self._refresh_tool_status()
    self._refresh_ai_status()
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
    left.setMinimumWidth(185)
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
    cv = QVBoxLayout(center)
    cards = QHBoxLayout()
    self.card_proxy = QLabel('proxy\n—')
    self.card_wl = QLabel('wirelength\n—')
    self.card_cong = QLabel('congestion\n—')
    self.card_macros = QLabel('macros placed\n—')
    for card in (self.card_proxy, self.card_wl, self.card_cong,
                 self.card_macros):
      card.setObjectName('card')
      card.setAlignment(Qt.AlignCenter)
      cards.addWidget(card, 1)
    cv.addLayout(cards)
    self.canvas = ChipCanvas()
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
    prow.addWidget(self.play_btn)
    prow.addWidget(self.step_btn)
    prow.addWidget(self.scrub, 1)
    prow.addWidget(self.speed_combo)
    prow.addWidget(self.view_combo)
    prow.addWidget(self.frame_label)
    cv.addLayout(prow)
    self.tabs = QTabWidget()
    cv.addWidget(self.tabs, 2)
    split.addWidget(center)

    right = QWidget()
    right.setMinimumWidth(300)
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
    rv.addWidget(self.progress_label)
    self.right_tabs = QTabWidget()
    rv.addWidget(self.right_tabs, 1)
    self.right_panel = right
    split.addWidget(right)
    split.setSizes([250, 720, 420])

    self.tabs.addTab(self._import_tab(), 'Import')
    self.tabs.addTab(self._place_tab(), 'Place')
    self.tabs.addTab(self._results_tab(), 'Results')
    self.tabs.addTab(self._thermal_tab(), 'Thermal')
    self.tabs.currentChanged.connect(self._tab_changed)
    self.right_tabs.addTab(self._chat_tab(), 'Assistant')
    self.right_tabs.addTab(self._connectors_tab(), 'Connectors')
    self.right_tabs.addTab(self._guide_tab(), 'Guide')
    self.right_tabs.addTab(self._settings_tab(), 'Settings')
    for tw in (self.tabs, self.right_tabs):
      tw.tabBar().setElideMode(Qt.ElideRight)
      tw.tabBar().setUsesScrollButtons(False)
    logw = QWidget()
    logv = QVBoxLayout(logw)
    self.log_view = QPlainTextEdit()
    self.log_view.setReadOnly(True)
    logv.addWidget(self.log_view)
    self.right_tabs.addTab(logw, 'Log')
    self._timer = QTimer(self)
    self._timer.timeout.connect(self._tick)
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
    v.addLayout(form)
    self.constraint_label = QLabel('Chat constraint: none')
    self.constraint_label.setWordWrap(True)
    v.addWidget(self.constraint_label)
    clear = QPushButton('Clear chat constraint')
    clear.clicked.connect(self._clear_constraint)
    v.addWidget(clear)
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
    grid = QHBoxLayout()
    for label, slot in (
        ('Download placed .plc…', self._save_plc),
        ('Download DEF…', self._save_def),
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
    self.th_clock = QSlider(Qt.Horizontal)
    self.th_clock.setRange(10, 500)
    self.th_clock.setValue(100)
    self.th_volt = QSlider(Qt.Horizontal)
    self.th_volt.setRange(60, 200)
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
    title = QLabel('Assistant')
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
    quick = QHBoxLayout()
    quick.setSpacing(6)
    for prompt in ('Explain the best run', 'Why is congestion high?',
                   'Which macros run hottest?',
                   'Keep a 20% routing channel open in the center'):
      b = QPushButton(prompt.split('?')[0][:24])
      b.setToolTip(prompt)
      b.clicked.connect(lambda _=0, t=prompt: self._chat_send(t))
      quick.addWidget(b)
    v.addLayout(quick)
    row = QHBoxLayout()
    row.setSpacing(6)
    self.chat_input = QLineEdit()
    self.chat_input.setPlaceholderText(
        'Ask about this floorplan, or state a constraint…')
    self.chat_input.returnPressed.connect(
        lambda: self._chat_send())
    send = QPushButton('Send')
    send.setObjectName('primary')
    send.clicked.connect(lambda: self._chat_send())
    row.addWidget(self.chat_input, 1)
    row.addWidget(send)
    v.addLayout(row)
    return w

  def _bubble(self, who, text):
    """One chat message bubble (user right / assistant left)."""
    is_user = who == 'You'
    bubble = QFrame()
    bubble.setStyleSheet(
        'QFrame { background:#9e6a1e; border:1px solid #e8a33d; '
        'border-radius:10px; }' if is_user else
        'QFrame { background:#161b22; border:1px solid #30363d; '
        'border-radius:10px; }')
    bl = QVBoxLayout(bubble)
    bl.setContentsMargins(10, 7, 10, 7)
    name = QLabel('You' if is_user else 'Assistant')
    name.setStyleSheet(
        'background:transparent; color:#ffd9a0; font-size:10px; '
        'font-weight:600;' if is_user else
        'background:transparent; color:#e8a33d; font-size:10px; '
        'font-weight:600;')
    bl.addWidget(name)
    body = QLabel(text)
    body.setWordWrap(True)
    body.setTextInteractionFlags(Qt.TextSelectableByMouse)
    body.setMaximumWidth(460)
    body.setStyleSheet('background:transparent; color:#e6edf3;')
    bl.addWidget(body)
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
    self.set_fields = {}
    pdk = self._settings['pdk']
    for key, label in (('yosys', 'Yosys binary'),
                       ('openroad', 'OpenROAD binary'),
                       ('lef', 'Cell LEF (blank = sky130 auto)'),
                       ('liberty', 'Liberty .lib (blank = auto)'),
                       ('techlef', 'Tech LEF (blank = auto)')):
      edit = QLineEdit(pdk.get(key, ''))
      browse = QPushButton('Browse…')
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
    try:
      self._load_plc_to_canvas(pair[0], pair[1])
    except Exception as e:
      self.log(f'canvas load failed: {e}')
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
          on_snapshot=lambda f: w.snap.emit(f))
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
    if self._best_plc_path:
      lines.append(f'Placed file: {self._best_plc_path} · output '
                   f'folder: {self._outdir}')
    self.details_view.setPlainText('\n'.join(lines))

  # -- worker / playback / canvas ---------------------------------------
  def _busy(self, busy):
    for b in (self.imp_go_btn, self.run_btn):
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
    self._busy(False)
    if self._thread:
      self._thread.quit()
      self._thread.wait(5000)
      self._thread = None
      self._worker = None

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

  def _live_row(self, row):
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

  def _view_changed(self, view):
    self.canvas.set_view(view)
    if 'Congestion' in view:
      self._ensure_congestion()
    if 'Thermal' in view:
      self._thermal_update()

  def _load_plc_to_canvas(self, netlist, plc_path, keep_frames=False):
    plc, frame = liveplace_lib.static_snapshot(netlist, plc_path)
    self._current_plc = plc
    self._congestion_ready = False
    self.canvas.set_segments(liveplace_lib.net_segments(plc))
    self.canvas.set_keepouts(self._keepouts)
    if keep_frames and self._frames:
      # Keep the live placement movie; the best placement becomes its
      # final frame so Play/scrub replays the whole AlphaChip-style run.
      self._frames.append(frame)
      self.scrub.setRange(0, len(self._frames) - 1)
      self._show_frame(len(self._frames) - 1)
    else:
      self._set_frames([frame])
    self._thermal_update()

  def _ensure_congestion(self):
    if getattr(self, '_congestion_ready', False) or \
            self._current_plc is None:
      return
    try:
      from neurosilicon import render as render_lib
      v, h = render_lib.congestion_maps(self._current_plc)
      self.canvas.set_congestion((v + h).tolist())
      self._congestion_ready = True
    except Exception as e:
      self.log(f'congestion view unavailable: {e}')

  def _thermal_update(self):
    plc = self._current_plc
    if plc is None:
      return
    try:
      power = liveplace_lib.scaled_power(
          plc, clock_mhz=float(self.th_clock.value()),
          voltage=self.th_volt.value() / 100.0,
          activity=self.th_act.value() / 100.0)
    except Exception as e:
      self.th_label.setText(f'thermal view unavailable: {e}')
      return
    self._power = power
    if self._frames:
      # Heat follows the displayed frame (live / scrub / playback).
      frame = self._frames[min(self._frame_idx, len(self._frames) - 1)]
      volts = self.th_volt.value() / 100.0
      scale = ((volts / 1.8) ** 2 * (self.th_clock.value() / 100.0)
               * ((self.th_act.value() / 100.0) / 0.2))
      self.canvas.set_power(
          {k: v * scale
           for k, v in liveplace_lib.frame_power(frame).items()})
    else:
      self.canvas.set_power(power)
    hot = max(power, key=power.get) if power else '?'
    self.th_label.setText(
        f'clock {self.th_clock.value()} MHz · '
        f'{self.th_volt.value() / 100:.2f} V · activity '
        f'{self.th_act.value()}% · hottest macro: {hot} '
        f'({power.get(hot, 0):.3f} rel.) Switch View → Thermal '
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
  def _mask_fn(self, netlist, plc_path):
    """Env mask fn for the chat keep-out/allow rects (or None)."""
    if not self._keepouts:
      return None
    from neurosilicon import priors as priors_lib
    from neurosilicon import setup as setup_lib
    plc = setup_lib.create_plc(netlist, plc_path)
    w, h = plc.get_canvas_width_height()
    cols, rows = plc.get_grid_num_columns_rows()
    if self._keepout_mode == 'allow':
      return priors_lib.make_allow_fn(cols, rows, w, h, self._keepouts)
    return priors_lib.make_keepout_fn(cols, rows, w, h, self._keepouts)

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
    self._set_left_visible(False)
    self.progress_label.setText(
        'Starting live placement (0–60%), then scored baseline '
        'episodes (60–100%). Details stream into Results.')
    self.tabs.setCurrentIndex(2)

    def job(w):
      from neurosilicon import evaluate as evaluate_lib
      mask_fn = None
      if keepouts:
        self._keepout_mode = keepout_mode
        self._keepouts = keepouts
        mask_fn = self._mask_fn(netlist, plc_path)
        w.log.emit(f'Chat constraint active ({keepout_mode}): '
                   f'{len(keepouts)} rect(s)')
      if animate:
        w.log.emit(f'Live engine ({method}): placing macros '
                   'sequentially, largest first…')
        info, _frames = liveplace_lib.run_live_rollout(
            netlist, plc_path, fd_steps=(fd,) * 3, method=method,
            seed=seed, extra_mask_fn=mask_fn, init_fd=False,
            on_snapshot=lambda f: w.snap.emit(f))
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
      return evaluate_lib.run_baselines(
          netlist, plc_path, episodes=episodes, fd_steps=(fd,) * 3,
          outdir=outdir, seed=seed, finetune_epochs=finetune,
          extra_mask_fn=mask_fn, init_fd=False,
          progress=lambda r: w.row.emit(dict(r)))

    def done(results):
      best_plc = os.path.join(outdir, 'best_baseline.plc')
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
    return {'design': item.text() if item else 'no design selected',
            'canvas': canvas, 'results': list(self._results),
            'best_proxy': min((r['proxy_cost'] for r in self._results
                               if r['proxy_cost'] > 0), default=None),
            'hottest': hottest}

  def _maybe_constraint(self, text):
    """Rule-parses a chat constraint into the canvas mask (instant)."""
    from neurosilicon import priors as priors_lib
    ctx = self._chat_context()
    w, h = ctx['canvas']
    if w <= 0:
      return
    low = text.lower()
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
    self.constraint_label.setText(
        f'Chat constraint ({mode}): {note} — applies to the next run.')
    self.log(f'Chat constraint ({mode}): {note}')

  def _clear_constraint(self):
    self._keepouts = []
    self.canvas.set_keepouts([])
    self.constraint_label.setText('Chat constraint: none')
    self.log('Chat constraint cleared.')

  def _chat_append(self, who, text):
    holder = self._bubble(who, text)
    self.chat_msgs.insertWidget(self.chat_msgs.count() - 1, holder)
    QTimer.singleShot(60, self._chat_to_bottom)

  def _chat_to_bottom(self):
    bar = self.chat_scroll.verticalScrollBar()
    bar.setValue(bar.maximum())

  def _chat_send(self, preset=None):
    text = (preset or self.chat_input.text()).strip()
    if not text:
      return
    self.chat_input.clear()
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
  app = QApplication(sys.argv)
  app.setApplicationName('NeuroSilicon')
  win = MainWindow()
  win.show()
  sys.exit(app.exec())


if __name__ == '__main__':
  main()
