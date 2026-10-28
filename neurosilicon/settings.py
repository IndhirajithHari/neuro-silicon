"""Persistent industrial-style settings with plain-language explanations.

Settings live outside the repo (API keys must never be committed)::

    ~/.config/neurosilicon/settings.json

Use defaults() for the full schema, load() to read (merged over
defaults), save() to persist, and field_help() for UI explanations.
"""
import copy
import json
import os

SETTINGS_DIR = os.path.join(
    os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
    'neurosilicon')
SETTINGS_PATH = os.path.join(SETTINGS_DIR, 'settings.json')

# Ordered sections: id -> {title, blurb, fields}. Each field: key, label,
# type (text/secret/int/float/bool/select), default, help, options?.
SECTIONS = [
    {
        'id': 'run',
        'title': 'Run Defaults',
        'blurb': ('Everyday run controls. These pre-fill the Run page; '
                  'changing them here changes the defaults everywhere.'),
        'fields': [
            {'key': 'mode', 'label': 'Placement mode', 'type': 'select',
             'default': 'heuristic',
             'options': ['heuristic', 'neurosilicon', 'llm_assisted'],
             'help': ('heuristic: first_legal/random baselines, no model '
                      'needed. neurosilicon: our trained PPO policy from a '
                      'checkpoint. llm_assisted: heuristic placement plus '
                      'LLM review and advice. Our model is optional — the '
                      'flow works fully without it.')},
            {'key': 'episodes', 'label': 'Episodes per run', 'type': 'int',
             'default': 3,
             'help': 'Independent placement attempts per run; the best '
                     'is kept. More episodes cost linear time.'},
            {'key': 'fd_steps', 'label': 'FD steps per stage',
             'type': 'int', 'default': 10,
             'help': ('Force-directed relaxation steps (x3 stages) applied '
                      'to standard cells at the end of each episode. '
                      'Higher is smoother but slower.')},
            {'key': 'seed', 'label': 'Random seed', 'type': 'int',
             'default': 0,
             'help': 'Seed for random rollouts. Same seed + same netlist '
                     '= reproducible result.'},
            {'key': 'outdir', 'label': 'Run output folder', 'type': 'text',
             'default': 'logs/gui_runs',
             'help': 'Where GUI runs write placements, figures, and '
                     'reports, relative to the repo root.'},
        ],
    },
    {
        'id': 'engine',
        'title': 'Placement Engine',
        'blurb': ('Proxy-cost weights and solver behavior. The proxy cost '
                  'is wl*w_wl + cong*w_cong + dens*w_dens at episode end.'),
        'fields': [
            {'key': 'wirelength_weight', 'label': 'Wirelength weight',
             'type': 'float', 'default': 1.0,
             'help': 'Weight of HPWL wirelength in the proxy cost.'},
            {'key': 'congestion_weight', 'label': 'Congestion weight',
             'type': 'float', 'default': 0.5,
             'help': 'Weight of RUDY-style routing congestion.'},
            {'key': 'density_weight', 'label': 'Density weight',
             'type': 'float', 'default': 1.0,
             'help': 'Weight of cell-density overflow.'},
            {'key': 'broadcast_weight', 'label': 'Broadcast weight',
             'type': 'float', 'default': 0.0,
             'help': ('Extra cost on high-fanout net span (clock-like '
                      'nets). 0 disables. Experimental.')},
            {'key': 'broadcast_threshold',
             'label': 'Broadcast fanout threshold', 'type': 'int',
             'default': 16,
             'help': 'Nets with at least this many sinks count as '
                     'broadcast nets.'},
            {'key': 'thermal_weight', 'label': 'Thermal weight',
             'type': 'float', 'default': 0.0,
             'help': ('Extra cost penalizing adjacent high-power macros. '
                      '0 disables.')},
            {'key': 'infeasible_penalty', 'label': 'Infeasible penalty',
             'type': 'float', 'default': 4.0,
             'help': 'Penalty reward when a rollout dead-ends (no legal '
                     'cell). Trains the policy to stay placeable.'},
        ],
    },
    {
        'id': 'model',
        'title': 'NeuroSilicon Model',
        'blurb': ('Our trained policy. Optional: heuristic and LLM-assisted '
                  'modes work without any checkpoint.'),
        'fields': [
            {'key': 'checkpoint', 'label': 'Checkpoint (.pt)',
             'type': 'text', 'default': 'logs/joint4_train/best.pt',
             'help': ('Policy weights for neurosilicon mode, relative to '
                      'the repo root. Missing file falls back to heuristic '
                      'with a warning.')},
        ],
    },
    {
        'id': 'llm',
        'title': 'AI Providers',
        'blurb': ('External API or local LLM for constraint parsing and '
                  'result advice. Keys stay in your local settings file, '
                  'never in the repo. When provider is "off", rule-based '
                  'parsing is used instead.'),
        'fields': [
            {'key': 'provider', 'label': 'Provider', 'type': 'select',
             'default': 'off', 'options': ['off', 'api', 'local'],
             'help': ('off: no LLM, rule-based parsing. api: OpenAI-'
                      'compatible chat API (cloud or gateway). local: '
                      'local server such as Ollama or LM Studio.')},
            {'key': 'api_key', 'label': 'API key', 'type': 'secret',
             'default': '',
             'help': 'Bearer key for the API provider. Stored locally '
                     'only; never printed or logged.'},
            {'key': 'base_url', 'label': 'API base URL', 'type': 'text',
             'default': 'https://api.openai.com/v1',
             'help': 'Chat-completions endpoint root, e.g. a gateway or '
                     'compatible provider. /chat/completions is appended.'},
            {'key': 'model', 'label': 'API model', 'type': 'text',
             'default': 'gpt-4o-mini',
             'help': 'Model name sent to the API provider.'},
            {'key': 'local_url', 'label': 'Local server URL', 'type': 'text',
             'default': 'http://localhost:11434',
             'help': 'Local LLM server root (Ollama default shown; LM '
                     'Studio uses http://localhost:1234/v1).'},
            {'key': 'local_model', 'label': 'Local model', 'type': 'text',
             'default': 'llama3.1',
             'help': 'Model name served locally (ollama list to see).'},
            {'key': 'timeout', 'label': 'Request timeout (s)', 'type': 'int',
             'default': 60,
             'help': 'HTTP timeout per LLM request. Slow local models '
                     'may need more.'},
        ],
    },
    {
        'id': 'pdk',
        'title': 'PDK & Toolchain',
        'blurb': ('Silicon libraries and EDA binaries for synthesis and '
                  'routed validation. Blank PDK fields auto-detect the '
                  'ciel sky130 install.'),
        'fields': [
            {'key': 'lef', 'label': 'Cell LEF', 'type': 'text',
             'default': '',
             'help': 'Standard-cell LEF (cell sizes/pins). Blank = '
                     'auto-detect sky130_fd_sc_hd.'},
            {'key': 'liberty', 'label': 'Liberty (.lib)', 'type': 'text',
             'default': '',
             'help': 'Timing library for synthesis mapping. Blank = '
                     'auto-detect the sky130 TT corner.'},
            {'key': 'techlef', 'label': 'Tech LEF', 'type': 'text',
             'default': '',
             'help': 'Routing-layer tech file for OpenROAD validation. '
                     'Blank = auto-detect.'},
            {'key': 'yosys', 'label': 'Yosys binary', 'type': 'text',
             'default': 'tools/yosys',
             'help': 'Yosys used for project import synthesis.'},
            {'key': 'openroad', 'label': 'OpenROAD binary', 'type': 'text',
             'default': 'tools/openroad',
             'help': 'OpenROAD used for routed validation.'},
            {'key': 'utilization', 'label': 'Macro utilization',
             'type': 'float', 'default': 0.8,
             'help': 'Target cell density inside hard macros during '
                     'project import.'},
            {'key': 'canvas_util', 'label': 'Canvas utilization',
             'type': 'float', 'default': 0.35,
             'help': 'Die-size target: smaller canvas = denser, harder '
                     'to place.'},
        ],
    },
]


def _section(sid):
  for s in SECTIONS:
    if s['id'] == sid:
      return s
  raise KeyError(f'unknown settings section: {sid}')


def defaults():
  """Full default settings as {section: {key: value}}."""
  out = {}
  for s in SECTIONS:
    out[s['id']] = {f['key']: copy.deepcopy(f['default'])
                    for f in s['fields']}
  return out


def load(path=None):
  """Load settings merged over defaults. Missing file -> defaults."""
  merged = defaults()
  path = path or SETTINGS_PATH
  if not os.path.isfile(path):
    return merged
  with open(path) as f:
    saved = json.load(f)
  for sid, fields in saved.items():
    if sid not in merged or not isinstance(fields, dict):
      continue
    for k, v in fields.items():
      if k in merged[sid]:
        merged[sid][k] = v
  return merged


def save(settings, path=None):
  """Persist settings (only known keys). Creates the directory."""
  path = path or SETTINGS_PATH
  os.makedirs(os.path.dirname(path), exist_ok=True)
  clean = defaults()
  for sid, fields in settings.items():
    if sid not in clean or not isinstance(fields, dict):
      continue
    for k, v in fields.items():
      if k in clean[sid]:
        clean[sid][k] = v
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(clean, f, indent=2)
  os.replace(tmp, path)
  return path


def field_help(sid, key):
  """Explanation string for one field (for UI help text)."""
  for f in _section(sid)['fields']:
    if f['key'] == key:
      return f['help']
  raise KeyError(f'unknown field: {sid}.{key}')


def is_secret(sid, key):
  for f in _section(sid)['fields']:
    if f['key'] == key:
      return f['type'] == 'secret'
  raise KeyError(f'unknown field: {sid}.{key}')


def redacted(settings):
  """Copy with secret values replaced by '(set)'/'(empty)' markers."""
  out = copy.deepcopy(settings)
  for s in SECTIONS:
    for f in s['fields']:
      if f['type'] == 'secret':
        v = out.get(s['id'], {}).get(f['key'], '')
        out[s['id']][f['key']] = '(set)' if v else '(empty)'
  return out
