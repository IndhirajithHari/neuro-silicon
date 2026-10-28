"""Tests for console visual primitives (no Streamlit runtime needed)."""
from neurosilicon import ui as ui_lib


class FakeSt:

  def __init__(self):
    self.calls = []

  def markdown(self, body, unsafe_allow_html=False):
    self.calls.append((body, unsafe_allow_html))


def test_status_pill_states():
  assert 'ns-ok' in ui_lib.status_pill('yosys', 'ok')
  assert 'ns-warn' in ui_lib.status_pill('x', 'warn')
  assert 'ns-bad' in ui_lib.status_pill('lef', 'bad')
  assert 'ns-muted' in ui_lib.status_pill('x', 'bogus')
  assert '<script>' not in ui_lib.status_pill('<script>', 'ok')


def test_pills_row():
  row = ui_lib.pills_row([('a', 'ok'), ('b', 'warn')])
  assert 'ns-ok' in row and 'ns-warn' in row


def test_workflow_steps_marks_current():
  html = ui_lib.workflow_steps_html(1)
  assert html.count('ns-step done') == 1
  assert html.count('ns-step now') == 1
  assert 'Place' in html and 'Report' in html
  assert 'ns-step now' in ui_lib.workflow_steps_html(99)
  st = FakeSt()
  ui_lib.workflow_steps(st, 0)
  assert 'ns-steps' in st.calls[0][0]


def test_header_footer_render():
  st = FakeSt()
  ui_lib.page_header(st, 'Place', 'Run', 'sub')
  ui_lib.page_footer(st)
  assert len(st.calls) == 4
  assert all(unsafe for _, unsafe in st.calls)
  assert ui_lib.version() in st.calls[-1][0]


def test_inject_css():
  st = FakeSt()
  ui_lib.inject_css(st)
  assert '<style>' in st.calls[0][0]


def test_version_format():
  import re
  assert re.fullmatch(r'\d+\.\d+\.\d+', ui_lib.version())
