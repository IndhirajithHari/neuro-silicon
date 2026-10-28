"""Console visual system: theme CSS, headers, pills, workflow stepper.

Presentational only — no placement compute, no Streamlit imports.
Page code calls inject_css() once, then page_header()/page_footer()
around its content; status_pill() and workflow_steps_html() are pure
HTML builders (unit-tested) rendered with st.markdown(..., unsafe_allow_html=True).
"""
import html

from neurosilicon._version import __version__

_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.8rem; max-width: 1240px;}
.ns-eyebrow {font-size: 0.72rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: #8B949E; margin-bottom: 0;}
.ns-title {font-size: 1.55rem; font-weight: 700; margin: 0.1rem 0 0.15rem 0;}
.ns-sub {color: #8B949E; font-size: 0.9rem; margin-bottom: 0.8rem;}
.ns-pill {display: inline-block; padding: 0.08rem 0.55rem;
  border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  margin: 0 0.25rem 0.25rem 0;}
.ns-ok {background: #14301F; color: #3FB950;}
.ns-warn {background: #38300F; color: #D29922;}
.ns-bad {background: #3D1D1D; color: #F85149;}
.ns-muted {background: #21262D; color: #8B949E;}
.ns-steps {display: flex; gap: 0.4rem; margin: 0.4rem 0 1rem 0;}
.ns-step {flex: 1; padding: 0.45rem 0.6rem; border-radius: 6px;
  background: #161B22; color: #8B949E; font-size: 0.8rem;
  border-top: 2px solid #30363D;}
.ns-step.done {color: #3FB950; border-top-color: #3FB950;}
.ns-step.now {color: #E8A33D; border-top-color: #E8A33D;
  font-weight: 600;}
.ns-footer {color: #6E7681; font-size: 0.75rem; margin-top: 2rem;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
table {font-size: 0.85rem;}
</style>
"""

_PILL_CLASS = {'ok': 'ns-ok', 'warn': 'ns-warn', 'bad': 'ns-bad',
               'muted': 'ns-muted'}

_STEPS = ['Place', 'Inspect', 'Validate', 'Report']


def version():
  """Package version string (footer, manifests, diagnostics)."""
  return __version__


def inject_css(st):
  """Applies the console stylesheet. Call once per page load."""
  st.markdown(_CSS, unsafe_allow_html=True)


def page_header(st, eyebrow, title, subtitle=''):
  """Consistent page head: eyebrow, title, optional subtitle."""
  st.markdown(f'<div class="ns-eyebrow">{html.escape(eyebrow)}</div>',
              unsafe_allow_html=True)
  st.markdown(f'<div class="ns-title">{html.escape(title)}</div>',
              unsafe_allow_html=True)
  if subtitle:
    st.markdown(f'<div class="ns-sub">{html.escape(subtitle)}</div>',
                unsafe_allow_html=True)


def page_footer(st, extra=''):
  """Version + support footer. Call after page content."""
  st.markdown(
      f'<div class="ns-footer">NeuroSilicon console v{html.escape(version())}'
      f'{(" · " + html.escape(extra)) if extra else ""}</div>',
      unsafe_allow_html=True)


def status_pill(label, state='muted'):
  """Inline status pill HTML. State: ok, warn, bad, muted."""
  cls = _PILL_CLASS.get(state, 'ns-muted')
  return (f'<span class="ns-pill {cls}">{html.escape(label)}</span>')


def pills_row(items):
  """Concatenates [(label, state), ...] into one HTML row."""
  return ' '.join(status_pill(label, state) for label, state in items)


def workflow_steps_html(current):
  """Place → Inspect → Validate → Report stepper HTML.

  `current` is the active step index (0-based); earlier steps render
  done. Out-of-range values clamp.
  """
  current = max(0, min(current, len(_STEPS) - 1))
  parts = ['<div class="ns-steps">']
  for i, name in enumerate(_STEPS):
    cls = 'done' if i < current else ('now' if i == current else '')
    parts.append(f'<div class="ns-step {cls}">{i + 1} · {name}</div>')
  parts.append('</div>')
  return ''.join(parts)


def workflow_steps(st, current):
  """Renders the workflow stepper for the calling page."""
  st.markdown(workflow_steps_html(current), unsafe_allow_html=True)
