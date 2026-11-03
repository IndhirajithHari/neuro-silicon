"""Deterministic --auto-record-all GUI walkthrough + frame recorder.

Drives the real desktop widgets (clicks, combo selections, sliders,
canvas mouse events) from a non-blocking QTimer step engine so the UI
thread never freezes. Frames are grabbed from the live window at
30 fps and piped to ffmpeg (rawvideo -> H.264 .mp4). On Wayland,
x11grab captures only the black XWayland root (verified on this
machine), so frames are captured from the window's own rendered
pixmap instead -- pixel-identical app rendering, never black.

Scene boundaries, active-panel bounding boxes and per-scene checks
are logged to recording_manifest.json for Phase 2 zoom/highlights.
"""
import json
import os
import subprocess
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt, QTimer
from PySide6.QtGui import (QColor, QImage, QMouseEvent, QPainter, QPen,
                           QPolygonF)
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton

W, H, FPS = 1920, 1080, 30

SCENES = [
    ('01_designs_and_live_placement', 'Designs + live placement (toy)'),
    ('02_playback_and_canvas_views', 'Playback bar + canvas views'),
    ('03_sam_critical_path_and_pinning', 'SAM + critical path + pin/drag'),
    ('04_thermal_lab_and_pdk_nodes', 'Thermal lab + PDK nodes'),
    ('05_rtl_project_import', 'RTL import: amber (Yosys)'),
    ('06_ai_connectors_meta_muse', 'Connectors: Meta Muse API'),
    ('07_neuro_ai_chat_and_constraints', 'Neuro AI chat + constraint fix'),
    ('08_auto_optimize_and_compare_view', 'Auto-Optimize + Compare view'),
    ('09_rtl_copilot_and_3_agent_review', 'RTL copilot + 3-agent review'),
    ('10_results_pareto_and_exports', 'Results, Pareto + exports'),
]


class Recorder:
  """Grabs window frames at 30 fps into an ffmpeg rawvideo pipe."""

  def __init__(self, win, out_path, cursor):
    self.win = win
    self.cursor = cursor  # dict: pos QPointF, ring QRect|None, ring_until
    self.frames = 0
    self.t0 = time.time()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-an', '-c:v', 'libopenh264', '-b:v', '12M',
           '-pix_fmt', 'yuv420p', '-r', str(FPS), out_path]
    self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    self.timer = QTimer(win)
    self.timer.timeout.connect(self._grab)
    self.timer.start(1000 // FPS)

  def now(self):
    return self.frames / FPS

  def _grab(self):
    pix = self.win.grab()
    if pix.width() != W or pix.height() != H:
      pix = pix.scaled(W, H)
    p = QPainter(pix)
    cur = self.cursor
    if cur['ring'] is not None and time.time() < cur['ring_until']:
      p.setPen(QPen(QColor('#facc15'), 3))
      p.setBrush(Qt.NoBrush)
      p.drawRoundedRect(cur['ring'], 6, 6)
    x, y = cur['pos'].x(), cur['pos'].y()
    arrow = QPolygonF([QPointF(x, y), QPointF(x, y + 22),
                       QPointF(x + 6, y + 17), QPointF(x + 10, y + 26),
                       QPointF(x + 13, y + 24), QPointF(x + 9, y + 15),
                       QPointF(x + 17, y + 15)])
    p.setPen(QPen(QColor('#000000'), 1))
    p.setBrush(QColor('#ffffff'))
    p.drawPolygon(arrow)
    p.end()
    img = pix.toImage().convertToFormat(QImage.Format_RGB888)
    self.proc.stdin.write(bytes(img.bits()[:W * H * 3]))
    self.frames += 1

  def stop(self):
    self.timer.stop()
    try:
      self.proc.stdin.close()
      self.proc.wait(timeout=120)
    except Exception:
      self.proc.kill()


def _find_button(win, text):
  for b in win.findChildren(QPushButton):
    if text in b.text() and b.isVisible():
      return b
  return None


class Runner:
  """Generator-step engine: yields ('wait', s) / ('until', fn, timeout,
  label) / ('move', QPointF) / ('do', fn). All QTimer-driven."""

  def __init__(self, win, outdir):
    self.win = win
    self.outdir = outdir
    self.exports = os.path.join(outdir, 'exports')
    os.makedirs(self.exports, exist_ok=True)
    self.cursor = {'pos': QPointF(W / 2, H / 2), 'ring': None,
                   'ring_until': 0.0}
    self.manifest = {'width': W, 'height': H, 'fps': FPS,
                     'encoder': 'libopenh264 -b:v 12M (no libx264 on '
                                'this ffmpeg; x11grab black on Wayland)',
                     'scenes': [], 'failures': []}
    self.rec = None
    self.gen = None
    self._until = None
    self._move = None
    self.save_path = os.path.join(self.exports, 'placed.plc')
    self._patch_dialogs()

  # -- dialog patching (runner process only): exports auto-save ----
  def _patch_dialogs(self):
    runner = self

    def fake_save(_parent, _caption, default, *_a, **_k):
      path = runner.save_path or default
      if path and not os.path.isabs(path):
        path = os.path.join(runner.exports, os.path.basename(path))
      return path, ''
    QFileDialog.getSaveFileName = staticmethod(fake_save)

  # -- helpers available to the script ------------------------------
  def fail(self, msg):
    self.manifest['failures'].append(msg)
    self.win.log(f'[auto-record] FAIL: {msg}')

  def bbox(self, widget):
    if widget is None:
      return [0, 0, 0, 0]
    g = widget.geometry()
    top = widget.mapTo(self.win, QPoint(0, 0))
    return [top.x(), top.y(), g.width(), g.height()]

  def panels(self):
    win = self.win
    return {
        'designs_panel': self.bbox(win.design_panel),
        'canvas': self.bbox(win.canvas),
        'center_tabs': self.bbox(win.tabs),
        'right_panel': self.bbox(win.right_panel),
        'kpi_cards': self.bbox(win.card_proxy),
        'playback_bar': self.bbox(win.play_btn),
        'progress': self.bbox(win.progress),
    }

  def highlight(self, widget):
    if widget is None:
      return
    top = widget.mapTo(self.win, QPoint(0, 0))
    self.cursor['ring'] = QRect(top.x() - 4, top.y() - 4,
                                widget.width() + 8, widget.height() + 8)
    self.cursor['ring_until'] = time.time() + 0.8

  def center_of(self, widget):
    top = widget.mapTo(self.win, QPoint(0, 0))
    return QPointF(top.x() + widget.width() / 2,
                   top.y() + widget.height() / 2)

  # -- engine ---------------------------------------------------------
  def start(self):
    master = os.path.join(self.outdir, 'phase1_full_walkthrough.mp4')
    self.rec = Recorder(self.win, master, self.cursor)
    self.gen = script(self)
    self._timer = QTimer(self.win)
    self._timer.timeout.connect(self._pump)
    self._timer.start(30)
    self._pump()

  def _pump(self):
    if self._move is not None:
      (x0, y0), (x1, y1), t_start, dur = self._move
      t = min(1.0, (time.time() - t_start) / dur)
      ease = 1 - (1 - t) ** 2
      self.cursor['pos'] = QPointF(x0 + (x1 - x0) * ease,
                                   y0 + (y1 - y0) * ease)
      if t < 1.0:
        return
      self._move = None
    if self._until is not None:
      pred, deadline, label = self._until
      try:
        ok = bool(pred())
      except Exception:
        ok = False
      if ok:
        self._until = None
      elif time.time() > deadline:
        self.fail(f'timeout waiting: {label}')
        self._until = None
      else:
        return
    try:
      step = next(self.gen)
    except StopIteration:
      self._finish()
      return
    except Exception as e:
      self.fail(f'script crashed: {type(e).__name__}: {e}')
      self._finish()
      return
    kind = step[0]
    if kind == 'wait':
      self._until = (lambda: False, time.time() + step[1], 'wait')
      # reuse until with a pure-time deadline
      self._until = (lambda: time.time() >= self._until[1], self._until[1], 'wait')
    elif kind == 'until':
      self._until = (step[1], time.time() + step[2], step[3])
    elif kind == 'move':
      cur = self.cursor['pos']
      self._move = ((cur.x(), cur.y()), (step[1].x(), step[1].y()),
                    time.time(), 0.45)
    elif kind == 'do':
      try:
        step[1]()
      except Exception as e:  # keep recording; log as failure
        self.fail(f'{type(e).__name__}: {e}')

  def _finish(self):
    self._timer.stop()
    self.manifest['duration_s'] = round(self.rec.now(), 2)
    self.rec.stop()
    with open(os.path.join(self.outdir, 'recording_manifest.json'),
              'w') as f:
      json.dump(self.manifest, f, indent=1)
    print(f'[auto-record] done: {self.rec.frames} frames, '
          f'failures={self.manifest["failures"]}', flush=True)
    QApplication.quit()


# -- script primitives (generator helpers) -----------------------------

def move_to(r, widget):
  yield ('move', r.center_of(widget))
  r.highlight(widget)


def click(r, widget, label=''):
  yield from move_to(r, widget)
  yield ('wait', 0.25)
  yield ('do', lambda: widget.click())
  yield ('wait', 0.4)


def not_busy(r):
  return lambda: r.win._thread is None


def macros_hard(r):
  frame = r.win.canvas._frame
  return [m for m in frame['macros'] if not m['soft']] if frame else []


def macro_point(r, m):
  ox, oy, s, Wc, Hc = r.win.canvas._tf
  top = r.win.canvas.mapTo(r.win, QPoint(0, 0))
  return QPointF(top.x() + ox + m['x'] * s, top.y() + oy + (Hc - m['y']) * s)


def mouse(r, pos, button, ev_type, buttons=None):
  canvas = r.win.canvas
  local = canvas.mapFrom(r.win, pos.toPoint())
  ev = QMouseEvent(ev_type, QPointF(local), button,
                   buttons or button, Qt.NoModifier)
  QApplication.sendEvent(canvas, ev)


def script(r):
  win = r.win
  repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  def scene(idx):
    sid, title = SCENES[idx]
    entry = {'id': sid, 'title': title, 'start_s': round(r.rec.now(), 2),
             'panels': r.panels(), 'checks': []}
    r.manifest['scenes'].append(entry)
    r.current = entry
    win.log(f'[auto-record] Scene {sid}: {title}')
    return entry

  def end_scene():
    r.current['end_s'] = round(r.rec.now(), 2)

  def check(name, fn):
    try:
      ok = bool(fn())
    except Exception:
      ok = False
    r.current['checks'].append({'check': name, 'ok': ok})
    if not ok:
      r.fail(f"{r.current['id']}: check failed: {name}")

  busy = not_busy(r)

  # ============ Scene 1: designs + live placement ==================
  sc = scene(0)
  yield ('wait', 2.5)
  toy_row = next((i for i in range(win.design_list.count())
                  if win.design_list.item(i).text().startswith('toy')), -1)
  check('designs list has toy/ariane/servant/imported amber',
        lambda: win.design_list.count() >= 4)
  item_rect_widget = win.design_list
  yield from move_to(r, item_rect_widget)
  yield ('do', lambda: win.design_list.setCurrentRow(toy_row))
  yield ('until', busy, 120, 'toy design load')
  yield ('wait', 1.5)
  yield from move_to(r, win.tabs.tabBar())
  yield ('do', lambda: win.tabs.setCurrentIndex(1))
  yield ('wait', 0.8)
  for chk in (win.run_animate, win.run_polish, win.run_orient):
    if not chk.isChecked():
      yield from click(r, chk)
  check('animate/polish/orient enabled',
        lambda: win.run_animate.isChecked() and win.run_polish.isChecked()
        and win.run_orient.isChecked())
  yield from click(r, win.run_btn, 'Run floorplan')
  yield ('until', busy, 900, 'toy Run floorplan completes')
  yield ('wait', 2.0)
  import re as _re

  def _kpi_ok(card):
    nums = _re.findall(r'-?\d+\.?\d*', card.text().split('\n')[-1])
    return bool(nums) and float(nums[0]) > 0
  for card, name in ((win.card_proxy, 'proxy'), (win.card_wl, 'wirelength'),
                     (win.card_cong, 'congestion'),
                     (win.card_macros, 'macros placed'),
                     (win.card_fmax, 'est. Fmax')):
    check(f'KPI {name} populated positive', lambda c=card: _kpi_ok(c))
  check('progress 100%', lambda: win.progress.value() == 100)
  yield ('wait', 2.0)
  end_scene()

  # ============ Scene 2: playback + canvas views ===================
  sc = scene(1)
  yield from click(r, win.play_btn, 'Play')
  yield ('wait', 2.5)
  yield from click(r, win.play_btn, 'Pause')
  yield ('wait', 0.8)
  for _ in range(3):
    yield from click(r, win.step_btn, 'Step')
  yield ('do', lambda: win.scrub.setValue(win.scrub.maximum() // 2))
  yield ('wait', 1.0)
  yield ('do', lambda: win.scrub.setValue(0))
  yield ('wait', 0.8)
  yield ('do', lambda: win.scrub.setValue(win.scrub.maximum()))
  yield ('wait', 0.8)
  for speed in ('Slow', 'Normal', 'Fast'):
    yield from move_to(r, win.speed_combo)
    yield ('do', lambda s=speed: win.speed_combo.setCurrentText(s))
    yield ('wait', 0.7)
  yield ('do', lambda: win.speed_combo.setCurrentText('Normal'))
  for view in ('Floorplan', 'Congestion', 'Thermal', 'Wires',
               'Floorplan + Wires', 'Thermal + Wires',
               'Congestion + Wires'):
    yield from move_to(r, win.view_combo)
    yield ('do', lambda v=view: win.view_combo.setCurrentText(v))
    yield ('wait', 2.5)
    check(f'view {view} active',
          lambda v=view: win.canvas._view == v)
  yield ('do', lambda: win.view_combo.setCurrentText('Floorplan'))
  yield ('wait', 1.0)
  end_scene()
  if os.environ.get('NS_AUTO_RECORD_STOP_AFTER') == '1':
    return

  # ============ Scene 3: SAM + critical path + pin/drag ============
  sc = scene(2)
  ari_row = next((i for i in range(win.design_list.count())
                  if win.design_list.item(i).text().startswith('ariane')), -1)
  yield ('do', lambda: win.design_list.setCurrentRow(ari_row))
  yield ('until', busy, 300, 'ariane design load')
  yield ('wait', 1.5)
  if not win.sam_check.isChecked():
    yield from click(r, win.sam_check, 'SAM on')
  yield ('do', lambda: win.view_combo.setCurrentText('Thermal'))
  yield ('wait', 1.5)
  if not win.crit_check.isChecked():
    yield from click(r, win.crit_check, 'Critical Path on')
  yield ('do', lambda: win.view_combo.setCurrentText('Floorplan'))
  yield ('wait', 3.0)
  check('SAM contours visible', lambda: len(win.canvas._contour_boxes) > 0)
  check('critical path visible',
        lambda: win.canvas._crit_visible and len(win.canvas._crit_points) > 1)
  hard = macros_hard(r)
  check('ariane has >=4 hard macros', lambda: len(hard) >= 4)
  for m in hard[:3]:
    pt = macro_point(r, m)
    yield ('move', pt)
    yield ('wait', 0.3)
    yield ('do', lambda p=pt: mouse(r, p, Qt.RightButton,
                                    QEvent.Type.MouseButtonPress,
                                    Qt.RightButton))
    yield ('do', lambda p=pt: mouse(r, p, Qt.RightButton,
                                    QEvent.Type.MouseButtonRelease,
                                    Qt.NoButton))
    yield ('wait', 0.7)
  check('3 macros pinned (P badge)', lambda: len(win._pinned) >= 3)
  if len(hard) > 3:
    m = hard[3]
    src = macro_point(r, m)
    dst = QPointF(src.x() + 60, src.y() + 45)
    yield ('move', src)
    yield ('do', lambda: mouse(r, src, Qt.LeftButton,
                                QEvent.Type.MouseButtonPress, Qt.LeftButton))
    for frac in (0.33, 0.66, 1.0):
      mid = QPointF(src.x() + (dst.x() - src.x()) * frac,
                    src.y() + (dst.y() - src.y()) * frac)
      yield ('move', mid)
      yield ('do', lambda p=mid: mouse(r, p, Qt.NoButton,
                                        QEvent.Type.MouseMove, Qt.LeftButton))
      yield ('wait', 0.25)
    yield ('do', lambda: mouse(r, dst, Qt.LeftButton,
                                QEvent.Type.MouseButtonRelease, Qt.NoButton))
    yield ('wait', 1.5)
    check('dragged macro moved+pinned',
          lambda: m['name'] in win._pinned)
  yield ('wait', 2.0)
  end_scene()

  # ============ Scene 4: thermal lab + PDK nodes ===================
  sc = scene(3)
  yield ('do', lambda: win.tabs.setCurrentIndex(3))
  yield ('wait', 1.0)
  for key in ('sky130', 'asap7', 'gf180mcu'):
    idx = win.tech_combo.findData(key)
    yield from move_to(r, win.tech_combo)
    yield ('do', lambda i=idx: win.tech_combo.setCurrentIndex(i))
    yield ('wait', 2.2)
    check(f'tech node {key}', lambda k=key: win._tech_key == k)
  for slider, val in ((win.th_clock, 450), (win.th_volt, 120),
                      (win.th_act, 65)):
    yield from move_to(r, slider)
    yield ('do', lambda s=slider, v=val: s.setValue(v))
    yield ('wait', 1.2)
  yield ('do', lambda: win.view_combo.setCurrentText('Thermal'))
  hard = macros_hard(r)
  if hard:
    hot = max(hard, key=lambda m: m['w'] * m['h'])
    pt = macro_point(r, hot)
    yield ('move', pt)
    yield ('do', lambda: mouse(r, pt, Qt.LeftButton,
                                QEvent.Type.MouseButtonPress, Qt.LeftButton))
    yield ('do', lambda: mouse(r, pt, Qt.LeftButton,
                                QEvent.Type.MouseButtonRelease, Qt.NoButton))
    yield ('wait', 2.0)
  check('thermal label names hottest macro',
        lambda: 'hottest macro:' in win.th_label.text())
  yield ('wait', 2.0)
  end_scene()

  # ============ Scene 5: RTL project import (amber) ================
  sc = scene(4)
  yield ('do', lambda: win.tabs.setCurrentIndex(0))
  yield ('wait', 1.0)
  amber_dir = os.path.join(repo, 'logs', 'imports', 'amber', 'build')
  yield from move_to(r, win.imp_path)
  yield ('do', lambda: win.imp_path.setText(amber_dir))
  yield ('do', lambda: win.imp_name.setText('amber'))
  detect_btn = _find_button(win, 'Auto-detect')
  yield from click(r, detect_btn, 'Auto-detect')
  yield ('wait', 2.0)
  check('auto-detect found amber RTL + top',
        lambda: win._disc is not None and win.imp_top.currentText() != '')

  def _amber_file_filter():
    # Mirror the file set of the previously successful amber import
    # (excludes the synthesized design_net.v, which duplicates modules).
    try:
      with open(os.path.join(repo, 'logs', 'imports', 'amber',
                             'import.json')) as f:
        wanted = set(json.load(f)['files'])
    except Exception:
      return
    from PySide6.QtCore import Qt as _Qt
    for i in range(win.file_list.count()):
      it = win.file_list.item(i)
      it.setCheckState(_Qt.Checked if it.text() in wanted
                       else _Qt.Unchecked)
  yield ('do', _amber_file_filter)
  yield ('wait', 1.0)
  yield from click(r, win.imp_go_btn, 'Import design')
  yield ('until', busy, 1500, 'amber Yosys import completes')
  yield ('wait', 2.0)
  check('imported: amber registered',
        lambda: any(win.design_list.item(i).text() == 'imported: amber'
                    for i in range(win.design_list.count())))
  yield ('wait', 2.0)
  end_scene()

  # ============ Scene 6: connectors (Meta Muse) ====================
  sc = scene(5)
  yield ('do', lambda: win.right_tabs.setCurrentIndex(1))
  yield ('wait', 1.5)
  check('API base URL is Meta',
        lambda: 'api.meta.ai' in win.ai_base.text())
  check('API key masked (password echo)',
        lambda: int(win.ai_key.echoMode()) == 1)
  models_btn = _find_button(win, 'List models')
  yield from click(r, models_btn, 'List models')
  yield ('until', busy, 300, 'List models returns')
  yield ('wait', 2.0)
  check('models listed from endpoint',
        lambda: 'Models your key can use' in win.ai_conn_status.text())
  test_btn = _find_button(win, 'Test connection')
  yield from click(r, test_btn, 'Test connection')
  yield ('until', busy, 300, 'Test connection returns')
  check('connection status Connected',
        lambda: win.ai_conn_status.text().startswith('Connected'))
  yield ('wait', 3.0)  # hold "Connected" 3 full seconds
  end_scene()

  # ============ Scene 7: Neuro AI chat + constraints ===============
  sc = scene(6)
  yield ('do', lambda: win.right_tabs.setCurrentIndex(0))
  yield ('wait', 1.0)
  n0 = win.chat_msgs.count()
  for prompt in ('Which macros run hottest?', 'Explain the best run'):
    idx = next((i for i in range(win.chat_suggest.count())
                if win.chat_suggest.itemText(i) == prompt), -1)
    check(f'suggestion exists: {prompt}', lambda i=idx: i > 0)
    yield from move_to(r, win.chat_suggest)
    yield ('do', lambda i=idx: win._on_suggestion_picked(i))
    yield ('until', busy, 300, f'LLM answer: {prompt}')
    yield ('until', lambda: win.chat_msgs.count() > n0, 60, 'chat reply')
    yield ('wait', 3.0)
  # Constraint + re-place on the fast toy design.
  yield ('do', lambda: win.design_list.setCurrentRow(toy_row))
  yield ('until', busy, 120, 'toy reload for constraint fix')
  cidx = next((i for i in range(win.chat_suggest.count())
               if '20% routing channel' in win.chat_suggest.itemText(i)), -1)
  yield from move_to(r, win.chat_suggest)
  yield ('do', lambda: win._on_suggestion_picked(cidx))
  yield ('wait', 2.0)
  check('center-channel constraint active',
        lambda: len(win._keepouts) > 0)
  yield from click(r, win.apply_fix_btn, 'Apply AI Fix: Re-Place')
  yield ('until', busy, 900, 'AI Fix re-placement completes')
  yield ('wait', 2.5)
  end_scene()

  # ============ Scene 8: Auto-Optimize + Compare ===================
  sc = scene(7)
  yield ('do', lambda: win.design_list.setCurrentRow(toy_row))
  yield ('until', busy, 120, 'toy selected for Auto-Optimize')
  yield from click(r, win.auto_btn_chat, 'Auto-Optimize (AI Loop)')
  yield ('until', busy, 1200, 'Auto-Optimize 3 iterations complete')
  yield ('wait', 2.0)
  check('PPA improvement summary in chat',
        lambda: win._compare_left is not None
        and win._compare_right is not None)
  yield from move_to(r, win.view_combo)
  yield ('do', lambda: win.view_combo.setCurrentText(
      'Compare (Baseline vs. AI)'))
  yield ('wait', 5.0)  # split screen + delta badges, fullscreen hold
  check('compare pair on canvas',
        lambda: win.canvas._compare is not None)
  yield ('wait', 1.0)
  end_scene()

  # ============ Scene 9: RTL copilot + 3-agent review ==============
  sc = scene(8)
  yield from click(r, win.rtl_btn_chat, 'Suggest RTL Fix')
  yield ('until', busy, 600, 'RTL fix suggestion returns')
  yield ('wait', 3.0)
  yield from click(r, win.review_btn, 'Run 3-Agent Design Review')
  yield ('until', busy, 600, '3-agent review returns')
  yield ('wait', 2.0)
  bar = win.chat_scroll.verticalScrollBar()
  top = bar.maximum()
  for frac in (0.25, 0.5, 0.75, 1.0):
    yield ('do', lambda f=frac: bar.setValue(int(top * f)))
    yield ('wait', 1.2)
  yield ('wait', 2.0)
  end_scene()

  # ============ Scene 10: results, Pareto, exports =================
  sc = scene(9)
  yield ('do', lambda: win.tabs.setCurrentIndex(2))
  yield ('do', lambda: win.view_combo.setCurrentText('Floorplan'))
  yield ('wait', 2.5)
  check('results table has scored rows',
        lambda: win.result_table.rowCount() >= 2)
  check('pareto chart has rows', lambda: len(win.pareto._rows) >= 2)
  yield ('wait', 2.0)
  exports = [
      ('Download placed .plc', 'placed.plc'),
      ('Download DEF', 'placed.def'),
      ('Download OpenROAD .tcl', 'placed_openroad.tcl'),
      ('Export Timelapse GIF', 'placement_timelapse.gif'),
  ]
  for label, fname in exports:
    r.save_path = os.path.join(r.exports, fname)
    btn = _find_button(win, label)
    check(f'export button present: {label}', lambda b=btn: b is not None)
    if btn is not None:
      yield from click(r, btn, label)
      yield ('until', busy, 300, f'export {fname}')
      yield ('wait', 1.5)
      check(f'export file written: {fname}',
            lambda f=fname: os.path.isfile(os.path.join(r.exports, f))
            and os.path.getsize(os.path.join(r.exports, f)) > 0)
  yield ('wait', 3.0)
  end_scene()
  yield ('wait', 1.0)


def run_desktop_auto_record(outdir):
  """Entry point used by neurosilicon.desktop --auto-record-all."""
  from neurosilicon.desktop import MainWindow
  app = QApplication.instance() or QApplication([])
  app.setApplicationName('NeuroSilicon')
  win = MainWindow()
  win.resize(W, H)
  win.show()
  runner = Runner(win, outdir)
  QTimer.singleShot(1200, runner.start)
  return app.exec()

