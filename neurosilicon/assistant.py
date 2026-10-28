"""MCP-style floorplan assistant + AI-connector guide (Qt-free).

The desktop chatbox talks to the chip through a small tool surface,
in the spirit of an MCP (Model Context Protocol) integration:

  get_design_info    canvas, grid, macro/port counts, top module
  get_run_results    episode table + best proxy cost breakdown
  get_thermal_info   hottest macros at the current corner
  propose_constraint turn chat text into a keep-out/allow mask

With an AI provider enabled (Settings > AI Connectors) the questions
go to that LLM with the tool results as context. With no provider,
a rule-based fallback answers from the same context, so the chatbox
works fully offline. Keys never leave the local settings file.
"""
import json
import os
import re

from neurosilicon import llm as llm_lib

MUSE_MODEL = 'muse-arcadia-1.0'

MCP_TOOLS = [
    {'name': 'get_design_info',
     'description': 'Canvas size, placement grid, macro/port counts.'},
    {'name': 'get_run_results',
     'description': 'Episode rows and best proxy/wirelength/congestion/'
                    'density.'},
    {'name': 'get_thermal_info',
     'description': 'Hottest macros at the current clock/voltage/'
                    'activity corner (area-proportional proxy).'},
    {'name': 'propose_constraint',
     'description': 'Parse chat text into a keep-out/allow rectangle '
                    'mask for the next placement run.'},
]

GUIDE_TEXT = """AI Connectors & MCP Chatbox — Setup Guide

Everything here is OPTIONAL. With no provider enabled, NeuroSilicon
places chips with its built-in engine and the chatbox answers with a
rule-based fallback. Enable a provider only if you want natural-
language constraint parsing and written run reviews.

1) Cloud API (OpenAI-compatible)
   - Get an API key from your provider (for OpenAI: platform.openai.com
     > API keys). Keep it secret; NeuroSilicon stores it only in
     ~/.config/neurosilicon/settings.json, never in the repo or logs.
   - In AI Connectors set Provider = api, paste the API key, set the
     base URL (OpenAI: https://api.openai.com/v1 — any compatible
     gateway works the same way) and the model name (e.g. gpt-4o-mini).
   - Press Test connection. "Connected" means one tiny chat round-trip
     succeeded; failures show the HTTP reason (401 = bad key,
     404 = wrong base URL or model).

2) Local LLM (Ollama)
   - Install Ollama, then:  ollama pull llama3.1   and   ollama serve
   - Provider = local, Local URL = http://localhost:11434,
     Local model = llama3.1 (check names with: ollama list).
   - No API key is needed and nothing leaves your machine.

3) Local LLM (LM Studio / OpenAI-compatible servers)
   - Start the LM Studio local server (default http://localhost:1234/v1)
     and load a model.
   - Provider = local, Local URL = http://localhost:1234/v1,
     Local model = the loaded model id.

4) MCP Chatbox (integrated assistant)
   - The chatbox on the right reads the current floorplan through
     MCP-style tools: get_design_info, get_run_results,
     get_thermal_info, propose_constraint.
   - Try: "Explain the best run", "Why is congestion high?",
     "Keep a 20% routing channel open in the center",
     "Which macros run hottest?"
   - A constraint reply shows the parsed keep-out rectangles; press
     Place > Run floorplan and the live engine respects the mask.
   - MCP here means the assistant can only see tool results (design
     facts and run numbers) — it never edits files or runs tools
     directly. Placement always runs in the local engine.

Troubleshooting
   - "LLM provider is off": pick api or local in AI Connectors.
   - Timeout on a slow local model: raise Request timeout (s).
   - Wrong-model 404s: copy the model id exactly (ollama list /
     LM Studio loaded-model name).
"""

GUIDE_HTML = (
    '<h3>AI Connectors &amp; MCP Chatbox — Setup Guide</h3>'
    '<p>Everything is <b>optional</b>. With no provider enabled the '
    'chatbox answers with a rule-based fallback and placement runs '
    'fully offline.</p>'
    '<h4>1) Cloud API (OpenAI-compatible)</h4><ol>'
    '<li>Get an API key from your provider (OpenAI: platform.openai.com '
    '&gt; API keys). It is stored only in '
    '<code>~/.config/neurosilicon/settings.json</code>.</li>'
    '<li>Provider = <b>api</b>, paste the API key, base URL '
    '<code>https://api.openai.com/v1</code>, model e.g. '
    '<code>gpt-4o-mini</code>.</li>'
    '<li>Press <b>Test connection</b> (401 = bad key, 404 = wrong '
    'base URL or model).</li></ol>'
    '<h4>2) Local LLM — Ollama</h4><ol>'
    '<li><code>ollama pull llama3.1</code> then <code>ollama serve</code></li>'
    '<li>Provider = <b>local</b>, URL <code>http://localhost:11434</code>, '
    'model <code>llama3.1</code> (<code>ollama list</code> shows names). '
    'No API key needed.</li></ol>'
    '<h4>3) Local LLM — LM Studio</h4><ol>'
    '<li>Start the LM Studio server (default '
    '<code>http://localhost:1234/v1</code>) and load a model.</li>'
    '<li>Provider = <b>local</b>, URL as above, model = loaded id.</li></ol>'
    '<h4>4) MCP Chatbox</h4>'
    '<p>The assistant reads the floorplan only through MCP-style tools: '
    '<code>get_design_info</code>, <code>get_run_results</code>, '
    '<code>get_thermal_info</code>, <code>propose_constraint</code>. '
    'Try: <i>Explain the best run</i> · <i>Keep a 20% routing channel '
    'open in the center</i> · <i>Which macros run hottest?</i> '
    'Placement always runs in the local engine; the assistant never '
    'edits files directly.</p>')


def connector_status(llm_settings):
  """(label, configured, detail) for the connectors panel. No network."""
  provider = llm_settings.get('provider', 'off')
  if provider == 'api':
    if not llm_settings.get('api_key'):
      return 'API: key missing', False, 'Paste an API key, then Test.'
    if not llm_settings.get('model'):
      return 'API: model missing', False, 'Set the API model name.'
    return (f"API: {llm_settings['model']}", True,
            f"via {llm_settings.get('base_url', '')} — press Test "
            'connection to verify.')
  if provider == 'local':
    if not llm_settings.get('local_model'):
      return 'Local: model missing', False, 'Set the local model name.'
    return (f"Local: {llm_settings['local_model']}", True,
            f"via {llm_settings.get('local_url', '')} — press Test "
            'connection to verify.')
  return ('AI: off', False,
          'Rule-based chat fallback. Pick api or local to enable an '
          'LLM (see Guide).')


def connection_explanation(llm_settings):
  """What is connected, how it works, and how to use it (chat text).

  Posted to the chat panel when connectors are saved or a connection
  test succeeds, and returned when the user asks what is connected.
  Never includes the API key.
  """
  provider = llm_settings.get('provider', 'off')
  label, _configured, detail = connector_status(llm_settings)
  tools = ', '.join(t['name'] for t in MCP_TOOLS)
  if provider == 'off':
    return (
        'AI connection: off. I currently answer with the built-in '
        'rule-based fallback (design facts and run numbers only), '
        'and placement constraints are parsed by pattern matching.\n'
        'To connect an LLM: open the AI Connectors tab (right), '
        'choose Provider = api (cloud, needs an API key + model) or '
        'local (Ollama / LM Studio, no key), Save, then Test '
        'connection. The Guide tab (right) walks through each '
        'provider step by step.\n'
        f'Once connected I read the floorplan only through MCP-style '
        f'tools ({tools}) and explain runs in natural language.')
  if not _configured:
    return (
        f'AI connection: {label}. {detail}\n'
        'To finish connecting: fill the missing field in the AI '
        'Connectors tab (right), Save, then Test connection. The '
        f'Guide tab walks through it. Once connected I read the '
        f'floorplan through MCP-style tools ({tools}).')
  endpoint = (llm_settings.get('base_url', '') if provider == 'api'
              else llm_settings.get('local_url', ''))
  model = (llm_settings.get('model', '') if provider == 'api'
           else llm_settings.get('local_model', ''))
  kind = 'cloud API (OpenAI-compatible)' if provider == 'api' \
      else 'local LLM server'
  return (
      f'Connected: {kind} · model {model} · endpoint {endpoint}. '
      f'{detail}\n'
      'How it works: when you ask a question, I send the LLM only '
      'MCP-style tool results (design info, run results, thermal '
      f'hotspots — tools: {tools}) plus your question. Your API key '
      'stays in the local settings file and is never sent anywhere '
      'except as the login header to that endpoint. Placement '
      'always runs in the local engine; I never edit files.\n'
      'What to try now: "Explain the best run" · "Why is congestion '
      'high?" · "Which macros run hottest?" · "Keep a 20% routing '
      'channel open in the center" (I will turn that into a keep-out '
      'mask for the next run).')


def suggest_constraint(text, canvas, llm_settings):
  """Chat text -> (mode, rects, explanation) via llm.parse_constraints."""
  w, h = canvas if canvas else (0.0, 0.0)
  return llm_lib.parse_constraints(text, float(w), float(h), llm_settings)


def _best_row(ctx):
  rows = [r for r in ctx.get('results', []) if r.get('proxy_cost', -1) > 0]
  return min(rows, key=lambda r: r['proxy_cost']) if rows else None


def _offline_answer(question, ctx):
  q = question.lower()
  design = ctx.get('design', 'no design selected')
  best = _best_row(ctx)
  lines = []
  if any(k in q for k in ('constrain', 'keep', 'reserve', 'channel',
                          'edge', 'perimeter', 'center', 'centre')):
    canvas = ctx.get('canvas') or (0.0, 0.0)
    if canvas[0] > 0:
      mode, rects, note = llm_lib.parse_constraints(
          question, canvas[0], canvas[1],
          {'provider': 'off'})
      lines.append(f'Parsed constraint ({mode}): {note}')
      for r in rects:
        lines.append(f'  rect ({r[0]:.1f}, {r[1]:.1f}) – '
                     f'({r[2]:.1f}, {r[3]:.1f}) microns')
      lines.append('Run Place to apply this mask to the live engine.')
    else:
      lines.append('Pick a design first so I can size the constraint '
                   'to its canvas.')
  elif any(k in q for k in ('thermal', 'hot', 'heat', 'temperature')):
    hot = ctx.get('hottest') or []
    if hot:
      lines.append('Hottest macros at the current corner '
                   '(area-proportional proxy, not signoff):')
      lines += [f'  {n}: {p:.3f} rel.' for n, p in hot[:5]]
    else:
      lines.append('Open the Thermal view after a run and ask again — '
                   'I rank macros by relative power there.')
    lines.append('Spread hot macros apart or lower clock/voltage to '
                 'cool the die.')
  elif any(k in q for k in ('congest', 'overflow', 'routing')):
    ca = ctx.get('congestion_analysis')
    if ca:
      lines.append(
          f"RUDY congestion from the live floorplan: overall "
          f"{ca['overall_congestion']:.3f}, peak tile "
          f"{ca['peak_congestion']:.3f}.")
      if ca.get('peak_congestion_tiles'):
        lines.append('Peak tiles (grid_x, grid_y, density): '
                     + '; '.join(
                         f'({c}, {r}) {v:.3f}'
                         for c, r, v in ca['peak_congestion_tiles'][:5])
                     + '.')
      if ca.get('congested_macros'):
        lines.append('Macros on/bordering the hot tiles: '
                     + ', '.join(
                         f"{m['name']} ({m['nets']} nets)"
                         for m in ca['congested_macros'][:6]) + '.')
      lines.append('To cut it: open a center channel, move those '
                   'macros apart, or press Apply AI Fix & Re-Place '
                   'after stating a constraint.')
    elif best:
      lines.append(f"Best run congestion is {best['congestion']:.3f} "
                   f"(wirelength {best['wirelength']:.3f}, density "
                   f"{best['density']:.3f}). Congestion is the "
                   'RUDY-style routing demand of the placed netlist.')
    else:
      lines.append('No finished run yet — run Place first, then ask '
                   'about congestion.')
    lines.append('To cut it: open a center channel, spread macros, or '
                 'raise the congestion weight in Settings.')
  elif best:
    lines.append(f'Design: {design}. Best proxy cost '
                 f"{best['proxy_cost']:.4f} "
                 f"({best.get('method', '?')} ep "
                 f"{best.get('episode', '?')}): wirelength "
                 f"{best['wirelength']:.4f}, congestion "
                 f"{best['congestion']:.4f}, density "
                 f"{best['density']:.4f}. Lower proxy is better; "
                 'wirelength tracks delay/capacitance, congestion '
                 'tracks routability, density tracks cell spreading.')
    lines.append('Next: try more episodes, a keep-out channel via '
                 'chat, or check the Thermal view for hotspots.')
  else:
    lines.append(f'Design: {design}. No finished placement yet. '
                 'Press Place > Run floorplan and watch the macros '
                 'land live — then ask me to explain the result, '
                 'check thermals, or propose a constraint.')
  lines.append('(offline rule-based answer — enable an AI connector '
               'for LLM reviews)')
  return '\n'.join(lines)


_SYSTEM = (
    'You are the NeuroSilicon floorplan assistant inside a desktop EDA '
    'app. You can inspect the design only through MCP-style tool '
    'results supplied in the context (design info, run results, '
    'thermal info). Answer concisely with concrete numbers from the '
    'context. If the user states a placement constraint, restate it '
    'as keep-out/allow rectangles in microns. Never claim to have run '
    'placement yourself; the local engine does that.')


# -- Auto-Optimize diagnosis + RTL co-pilot (Muse, offline fallback) -----
def _muse_chat(messages, llm_settings):
  """Chats via muse-arcadia-1.0 first, then the configured model.

  Returns (text, source) where source is 'muse-arcadia-1.0' or the
  configured model id. Raises LLMError when no provider can answer
  (provider off included) so callers fall back instantly.
  """
  provider = llm_settings.get('provider', 'off')
  if provider == 'off':
    raise llm_lib.LLMError('LLM provider is off.')
  muse = dict(llm_settings)
  if provider == 'api':
    muse['model'] = MUSE_MODEL
  else:
    muse['local_model'] = MUSE_MODEL
  try:
    return llm_lib.chat(messages, muse), MUSE_MODEL
  except llm_lib.LLMError:
    configured = (llm_settings.get('model') if provider == 'api'
                  else llm_settings.get('local_model')) or ''
    if configured and configured != MUSE_MODEL:
      return llm_lib.chat(messages, llm_settings), configured
    raise


def offline_diagnosis(canvas, hotspot_rect=None, congested=(),
                      congestion_weight=0.5, thermal_weight=0.0):
  """Deterministic JSON placement constraints (instant fallback).

  Always spreads hot macros (thermal weight) and opens a 20% center
  routing channel (keep-out), and never lowers either weight.
  """
  w, h = float(canvas[0]), float(canvas[1])
  rects = [(w * 0.4, 0.0, w * 0.6, h)] if w > 0 and h > 0 else []
  names = ', '.join(list(congested)[:4]) or 'the hotspot cluster'
  return {'actions': ['spread_hot_macros', 'keepout_center_channel'],
          'mode': 'block', 'rects': rects,
          'congestion_weight': round(min(2.0, max(
              float(congestion_weight) + 0.3, 0.8)), 3),
          'thermal_weight': round(min(2.0, max(
              float(thermal_weight), 0.5)), 3),
          'explanation': (f'Spread {names} apart and keep a 20% '
                          'center routing channel open.'),
          'source': 'offline-fallback',
          'hotspot_rect': hotspot_rect,
          'congested_macros': list(congested)}


def _sanitize_diagnosis(spec, canvas, congestion_weight, thermal_weight,
                        source, hotspot_rect, congested):
  w, h = float(canvas[0]), float(canvas[1])
  rects = []
  for r in spec.get('rects', []) or []:
    try:
      x0, y0, x1, y1 = (float(v) for v in r)
    except (TypeError, ValueError):
      continue
    x0, x1 = sorted((max(0.0, min(x0, w)), max(0.0, min(x1, w))))
    y0, y1 = sorted((max(0.0, min(y0, h)), max(0.0, min(y1, h))))
    if x1 - x0 > 1.0 and y1 - y0 > 1.0 and \
            (x1 - x0) * (y1 - y0) <= 0.6 * max(w * h, 1.0):
      rects.append((x0, y0, x1, y1))
  if not rects and w > 0 and h > 0:
    rects = [(w * 0.4, 0.0, w * 0.6, h)]
  try:
    cg = float(spec.get('congestion_weight', 0.0))
  except (TypeError, ValueError):
    cg = 0.0
  try:
    th = float(spec.get('thermal_weight', 0.0))
  except (TypeError, ValueError):
    th = 0.0
  actions = [a for a in (spec.get('actions') or [])
             if isinstance(a, str)] or ['spread_hot_macros',
                                        'keepout_center_channel']
  return {'actions': actions, 'mode': 'block', 'rects': rects,
          'congestion_weight': round(min(2.0, max(
              cg, float(congestion_weight) + 0.1, 0.8)), 3),
          'thermal_weight': round(min(2.0, max(
              th, float(thermal_weight), 0.5)), 3),
          'explanation': str(spec.get('explanation')
                             or 'AI-guided re-placement.'),
          'source': source, 'hotspot_rect': hotspot_rect,
          'congested_macros': list(congested)}


_DIAGNOSE_SYSTEM = (
    'You are muse-arcadia-1.0, a chip-floorplanning agent. Reply with '
    'ONE JSON object only, no prose: {"actions": ["spread_hot_macros", '
    '"keepout_center_channel"], "rects": [[x0,y0,x1,y1]], '
    '"congestion_weight": 0.8, "thermal_weight": 0.5, '
    '"explanation": "..."}. Coordinates are microns on the canvas '
    'given by the user. Keep rects under 60% of the die area and never '
    'lower the current weights.')


def diagnose_placement(plc, llm_settings, congestion_weight=0.5,
                       thermal_weight=0.0):
  """SAM + Muse diagnosis for the hottest congestion/thermal cluster.

  Segments the top RUDY hotspot (liveplace.top_hotspot), then asks
  muse-arcadia-1.0 for JSON placement constraints. Any LLM failure
  (offline, provider off, bad JSON) falls back instantly to
  offline_diagnosis. Never raises. Returns the sanitized dict from
  _sanitize_diagnosis / offline_diagnosis plus 'hotspot' facts.
  """
  from neurosilicon import liveplace as liveplace_lib
  canvas = plc.get_canvas_width_height()
  try:
    analysis, clusters, rect = liveplace_lib.top_hotspot(plc)
    congested = [m['name'] for m in analysis.get('congested_macros', [])]
    hotspot = {'overall_congestion': analysis['overall_congestion'],
               'peak_congestion': analysis['peak_congestion'],
               'cluster_cells': clusters[0]['cells'] if clusters else 0,
               'rect': rect}
  except Exception:
    congested, hotspot, rect = [], {}, None
  fallback = offline_diagnosis(canvas, rect, congested,
                               congestion_weight, thermal_weight)
  fallback['hotspot'] = hotspot
  try:
    raw, source = _muse_chat(
        [{'role': 'system', 'content': _DIAGNOSE_SYSTEM},
         {'role': 'user',
          'content': f'Canvas W={canvas[0]:.1f} H={canvas[1]:.1f} '
                     f'microns. RUDY overall '
                     f"{hotspot.get('overall_congestion', 0.0):.3f}, "
                     f"peak {hotspot.get('peak_congestion', 0.0):.3f}. "
                     f'Top SAM hotspot rect (microns): {rect}. '
                     f'Congested macros: {congested[:6]}. Current '
                     f'weights: congestion={congestion_weight}, '
                     f'thermal={thermal_weight}. Diagnose the '
                     'bottleneck and propose constraints.'}],
        llm_settings)
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
      raise llm_lib.LLMError('no JSON object in reply')
    out = _sanitize_diagnosis(json.loads(raw[start:end + 1]), canvas,
                              congestion_weight, thermal_weight, source,
                              rect, congested)
    out['hotspot'] = hotspot
    return out
  except (llm_lib.LLMError, ValueError, TypeError):
    return fallback


def find_bottleneck(plc, power=None):
  """Macro causing the worst thermal / routing-congestion bottleneck.

  Prefers the top RUDY-congested macro; otherwise the highest power-
  density (hottest) macro. Returns {macro, reason, peak_tile, nets,
  peak_power, peak_density, hottest}.
  """
  from neurosilicon import liveplace as liveplace_lib
  if power is None:
    power = liveplace_lib.scaled_power(plc)
  areas = {}
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = max(float(w) * float(h), 1e-9)
  hottest = max(power, key=lambda n: float(power[n]) / areas.get(n, 1e-9),
                default='')
  hot_density = (float(power.get(hottest, 0.0)) / areas[hottest]
                 if hottest else 0.0)
  try:
    ca = liveplace_lib.congestion_analysis(plc)
    # Soft clusters have no RTL module to fix; bottlenecks must be
    # hard macros (the names that map to .v files / placed blocks).
    cong = [m for m in (ca.get('congested_macros') or [])
            if m['name'] in areas]
  except Exception:
    cong = []
  if cong:
    top = cong[0]
    return {'macro': top['name'], 'reason': 'routing congestion',
            'peak_tile': float(top.get('peak_tile', 0.0)),
            'nets': int(top.get('nets', 0)),
            'peak_power': float(power.get(top['name'], 0.0)),
            'peak_density': float(power.get(top['name'], 0.0))
            / areas.get(top['name'], 1e-9), 'hottest': hottest}
  return {'macro': hottest, 'reason': 'thermal hotspot', 'peak_tile': 0.0,
          'nets': 0, 'peak_power': float(power.get(hottest, 0.0)),
          'peak_density': hot_density, 'hottest': hottest}


_MODULE_DEF_RE = re.compile(r'^\s*module\s+([A-Za-z_]\w*)', re.MULTILINE)


def find_verilog_for_macro(macro, roots):
  """Finds the .v/.sv file defining (or named for) `macro`, or None.

  `roots`: a folder path or an iterable of file paths / folders.
  Match order: exact basename, `module <macro>` in file text, file
  stem suffix of the macro name. Walks are bounded.
  """
  if isinstance(roots, (str, os.PathLike)):
    roots = [roots]
  files = []
  for root in roots or []:
    root = str(root or '')
    if os.path.isfile(root) and root.lower().endswith(('.v', '.sv')):
      files.append(root)
    elif os.path.isdir(root):
      for dirpath, _dirs, names in os.walk(root):
        for fn in sorted(names):
          if fn.lower().endswith(('.v', '.sv')):
            files.append(os.path.join(dirpath, fn))
        if len(files) > 400:
          break
  if not macro or not files:
    return None
  stem = re.sub(r'\W', '_', macro)
  for path in files:
    if os.path.splitext(os.path.basename(path))[0] == stem:
      return path
  for path in files:
    try:
      with open(path, errors='replace') as f:
        if macro in _MODULE_DEF_RE.findall(f.read()):
          return path
    except OSError:
      continue
  for path in files:
    base = os.path.splitext(os.path.basename(path))[0]
    if base and (stem.endswith(base) or base.endswith(stem)):
      return path
  return None


def read_verilog_snippet(path, max_chars=2600):
  """First `max_chars` around the module header of a Verilog file."""
  try:
    with open(path, errors='replace') as f:
      text = f.read()
  except OSError:
    return ''
  m = re.search(r'^\s*module\s', text, re.MULTILINE)
  start = max(0, m.start() - 200) if m else 0
  return text[start:start + max_chars].strip()


def offline_rtl_fix(macro, snippet='', filename=''):
  """Instant offline Verilog-template RTL fix (memory/bus/clock-gate).

  Returns Assistant-ready markdown with a unified-style Verilog diff.
  The template is picked from the bottleneck name/file: caches and
  SRAMs get memory banking, buses/arbiters get pipelining, anything
  else gets clock gating plus a pipeline register.
  """
  mod = re.sub(r'\W', '_', macro or 'bottleneck') or 'bottleneck'
  low = f'{macro} {filename}'.lower()
  if any(k in low for k in ('cache', 'sram', 'ram', 'mem', 'bank')):
    title, body = 'Memory banking (split the array, gate the idle bank)', (
        f'```verilog\n'
        f'--- a/{filename or (mod + ".v")}\n'
        f'+++ b/{filename or (mod + ".v")}\n'
        f'@@ module {mod} @@\n'
        f'-  // Single 256-line array: every access toggles the full\n'
        f'-  // array (peak power + routing hotspot at the macro).\n'
        f'+  // Banked: 2 banks x 128 lines. bank_sel = addr[MSB];\n'
        f'+  // only the selected bank clocks/decodes per access.\n'
        f'+  wire bank_sel = req_addr[ADDR_W-1];\n'
        f'+  {mod}_bank #(.LINES(128)) u_bank0 (.clk(clk & ~bank_sel), .*);\n'
        f'+  {mod}_bank #(.LINES(128)) u_bank1 (.clk(clk &  bank_sel), .*);\n'
        f'```\n'
        'Why: banking halves the toggled capacitance per access and '
        'splits the routing hotspot into two smaller macros the placer '
        'can separate. Expect lower peak power density (thermal) and '
        'lower RUDY congestion around the array; verify hit rate and '
        'timing after re-synthesis.')
  elif any(k in low for k in ('wishbone', 'bus', 'arbiter', 'mux',
                              'interconnect', 'icache', 'dcache')):
    title, body = 'Bus pipelining (register slice on the hot interface)', (
        f'```verilog\n'
        f'--- a/{filename or (mod + ".v")}\n'
        f'+++ b/{filename or (mod + ".v")}\n'
        f'@@ module {mod} @@\n'
        f'-  assign wb_dat_o = mem_dat;  // long combinational bus\n'
        f'+  // Register slice: cuts the critical bus path and lets\n'
        f'+  // the placer separate master/slave macros.\n'
        f'+  reg [31:0] dat_q; reg ack_q;\n'
        f'+  always @(posedge clk) begin\n'
        f'+    dat_q <= mem_dat; ack_q <= mem_ack & ~stall;\n'
        f'+  end\n'
        f'+  assign wb_dat_o = dat_q;\n'
        f'```\n'
        'Why: the register slice breaks the longest bus wire (better '
        'est. Fmax), removes the combinational hotspot, and gives the '
        'placer freedom to spread the endpoints. Costs one cycle of '
        'latency; keep the stall/ack handshake intact.')
  else:
    title, body = 'Clock gating + output pipelining (thermal first)', (
        f'```verilog\n'
        f'--- a/{filename or (mod + ".v")}\n'
        f'+++ b/{filename or (mod + ".v")}\n'
        f'@@ module {mod} @@\n'
        f'-  always @(posedge clk) begin ... end  // always toggles\n'
        f'+  wire clk_en = active_cycle | state_change;\n'
        f'+  wire gclk;  // latch-based integrated clock gate\n'
        f'+  assign gclk = clk & clk_en_q;\n'
        f'+  always @(posedge gclk) begin ... end\n'
        f'+  // + register the hottest output cone one stage earlier\n'
        f'```\n'
        'Why: gating idle cycles cuts dynamic power at the source '
        '(lower peak temperature), and the extra pipeline register '
        'shortens the critical path (higher est. Fmax). Confirm the '
        'enable covers every state the block can wake in.')
  src = f' (from {filename})' if filename else ''
  snip = ('\nCurrent RTL snippet:\n```verilog\n'
          + snippet[:900] + '\n```\n') if snippet else ''
  return (f'RTL fix for bottleneck `{macro}`{src} — {title}\n'
          + snip + body +
          '\n(offline Verilog template fallback — enable muse-arcadia-1.0 '
          'in AI Connectors for a model-written diff; re-import + '
          're-place to measure the PPA delta)')


_RTL_SYSTEM = (
    'You are muse-arcadia-1.0 acting as a hardware co-pilot (Muse '
    'Code style). Given a bottleneck macro and its Verilog snippet, '
    'propose ONE micro-architectural RTL optimization (memory banking, '
    'bus pipelining, or clock gating) that reduces thermal hotspots / '
    'routing congestion. Reply in markdown: 1-line summary, a fenced '
    'verilog unified diff (---/+++ and -/+ lines), then 2-3 lines on '
    'expected PPA impact. Keep the diff small and synthesizable.')


def suggest_rtl_fix(macro, snippet, llm_settings, filename=''):
  """RTL co-pilot suggestion. Returns (markdown_text, used_llm).

  Queries muse-arcadia-1.0 (then the configured model) with the
  bottleneck snippet; any failure returns the instant offline
  Verilog template. Never raises.
  """
  try:
    text, source = _muse_chat(
        [{'role': 'system', 'content': _RTL_SYSTEM},
         {'role': 'user',
          'content': f'Bottleneck macro/module: {macro} '
                     f'(file: {filename or "unknown"}). It causes the '
                     f'highest thermal / routing congestion in the '
                     f'current floorplan.\n\nVerilog snippet:\n'
                     f'```verilog\n{snippet[:2200]}\n```'}],
        llm_settings)
    return (f'RTL fix for bottleneck `{macro}` — via {source} '
            f'(Muse co-pilot)\n\n{text}', True)
  except (llm_lib.LLMError, ValueError, TypeError):
    return offline_rtl_fix(macro, snippet, filename), False


def bottleneck_rtl_advice(plc, roots, llm_settings, power=None):
  """End-to-end co-pilot: bottleneck -> .v snippet -> suggestion.

  Returns (text, used_llm, info) where info has macro/reason/file.
  With no placement, macro falls back to the first RTL file stem so
  the Import tab can still demo the flow.
  """
  info = {'macro': '', 'reason': 'no placement yet', 'file': ''}
  snippet = ''
  if plc is not None:
    info.update(find_bottleneck(plc, power))
  macro = info.get('macro', '')
  path = find_verilog_for_macro(macro, roots) if macro else None
  if path is None and roots:
    first = find_verilog_for_macro('', roots)
    # find_verilog needs a name; scan for any RTL file instead.
    path = first
    if path is None:
      stack = [str(r) for r in
               (roots if isinstance(roots, (list, tuple)) else [roots])]
      while stack and path is None:
        cur = stack.pop()
        if os.path.isfile(cur) and cur.lower().endswith(('.v', '.sv')):
          path = cur
        elif os.path.isdir(cur):
          for fn in sorted(os.listdir(cur)):
            full = os.path.join(cur, fn)
            if os.path.isfile(full) and \
                    fn.lower().endswith(('.v', '.sv')):
              path = full
              break
            if os.path.isdir(full):
              stack.append(full)
    if path and not macro:
      macro = os.path.splitext(os.path.basename(path))[0]
      info['macro'] = macro
      info['reason'] = 'largest/first imported RTL module (no run yet)'
  if path:
    info['file'] = os.path.basename(path)
    # Only quote the file when it actually defines the bottleneck;
    # an unrelated first-found file must not pose as its source.
    if not macro or find_verilog_for_macro(macro, [path]) == path:
      snippet = read_verilog_snippet(path)
  text, used = suggest_rtl_fix(macro or 'bottleneck', snippet,
                               llm_settings, info.get('file', ''))
  return text, used, info


def multi_agent_review(ctx, llm_settings=None):
  """Multi-agent silicon design review (3 specialists, deterministic).

  Reads the same MCP-style context as answer() — run rows, thermal
  hotspots, RUDY congestion, PPA snapshot, critical path — and returns
  (markdown, weights) where weights is the Lead Architect's
  recommended {wirelength_weight, congestion_weight, thermal_weight}.
  Rule-based so it works fully offline; numbers come from ctx only.
  """
  best = _best_row(ctx)
  ppa = ctx.get('ppa') or {}
  cp = ctx.get('critical') or {}
  hot = ctx.get('hottest') or []
  ca = ctx.get('congestion_analysis') or {}
  design = ctx.get('design', 'no design selected')
  peak_macro = (ppa.get('peak_macro')
                or (hot[0][0] if hot else '—'))
  rudy = ca.get('overall_congestion', ppa.get('rudy_congestion', 0.0))
  cong = (best or {}).get('congestion', ppa.get('congestion', 0.0))
  fmax = cp.get('fmax_mhz', ppa.get('fmax_mhz', 0.0))
  # Thermal Specialist: hotter peak -> stronger thermal pressure.
  thermal_w = 0.9 if hot and hot[0][1] >= 0.25 else (
      0.6 if hot else 0.5)
  # Routing & Timing: congested or slow -> raise congestion/wirelength.
  congestion_w = 1.2 if max(rudy, cong) > 1.0 else (
      0.8 if max(rudy, cong) > 0.7 else 0.5)
  wirelength_w = 1.3 if 0 < fmax < 150 else 1.0
  weights = {'wirelength_weight': round(wirelength_w, 2),
             'congestion_weight': round(congestion_w, 2),
             'thermal_weight': round(thermal_w, 2)}
  prox = (best or {}).get('proxy_cost', ppa.get('proxy_cost', 0.0))
  lines = [
      'Multi-Agent Silicon Design Review',
      f'Design: {design} · best proxy {prox:.4f} · est. Fmax '
      f'{fmax:.0f} MHz (proxy, not signoff)',
      '',
      '🌡 Thermal Specialist',
      (f'Peak macro: {peak_macro}. Hottest now: '
       + (', '.join(f'{n} {p:.3f} rel.' for n, p in hot[:3])
          if hot else 'run Place + Thermal first for live data')
       + f'. Recommends thermal_weight = {thermal_w:.1f} and spreading '
         'the peak macro away from its neighbours (Auto-Optimize '
         'Iter 2 does this automatically).'),
      '',
      '🧭 Routing & Timing Specialist',
      (f'RUDY overall congestion {rudy:.3f}, placed congestion '
       f'{cong:.3f}, critical delay '
       f"{cp.get('delay_ns', ppa.get('critical_delay_ns', 0.0)):.2f} ns "
       f"via {' → '.join((cp.get('path') or ppa.get('critical_path') or ['—'])[:5])}. "
       f'Recommends congestion_weight = {congestion_w:.1f}, '
       f'wirelength_weight = {wirelength_w:.1f}; keep the SAM center '
       'channel open and check Show Critical Path before sign-off.'),
      '',
      '🏛 Lead Physical Design Architect',
      (f'Verdict: proxy {prox:.4f} is the number to beat. Adopt '
       f"wirelength_weight={weights['wirelength_weight']:.2f}, "
       f"congestion_weight={weights['congestion_weight']:.2f}, "
       f"thermal_weight={weights['thermal_weight']:.2f} "
       '(applied to Settings), then run ⚡ Auto-Optimize (AI Loop) '
       'with SA polish + orientation flips and confirm in Compare '
       '(Baseline vs. AI). Recommended weights: '
       f"wl={weights['wirelength_weight']:.2f} · "
       f"cong={weights['congestion_weight']:.2f} · "
       f"thermal={weights['thermal_weight']:.2f}."),
  ]
  return '\n'.join(lines), weights


def answer(question, ctx, llm_settings):
  """Chatbox reply. Returns (text, used_llm). Never raises."""
  q = question.lower()
  if any(k in q for k in ('connect', 'status', 'setup', 'implement',
                          'how do i use', 'how to use', 'what can you',
                          'who are you', 'what are you')):
    return connection_explanation(llm_settings), False
  provider = llm_settings.get('provider', 'off')
  if provider != 'off':
    best = _best_row(ctx)
    facts = [f"design: {ctx.get('design', '?')}",
             f"canvas: {ctx.get('canvas', '?')}"]
    if best:
      facts.append(
          f"best: proxy {best['proxy_cost']:.4f} wl "
          f"{best['wirelength']:.4f} cong {best['congestion']:.4f} "
          f"dens {best['density']:.4f}")
    if ctx.get('hottest'):
      facts.append('hottest: ' + ', '.join(
          f'{n} {p:.3f}' for n, p in ctx['hottest'][:5]))
    ca = ctx.get('congestion_analysis')
    if ca:
      facts.append(
          f"RUDY congestion: overall "
          f"{ca['overall_congestion']:.3f}, peak "
          f"{ca['peak_congestion']:.3f}, peak tiles "
          f"{ca.get('peak_congestion_tiles', [])[:5]}, congested "
          f"macros {[m['name'] for m in ca.get('congested_macros', [])[:6]]}")
    try:
      text = llm_lib.chat(
          [{'role': 'system', 'content': _SYSTEM},
           {'role': 'user',
            'content': 'Context (MCP tool results):\n' + '\n'.join(facts)
                       + f'\n\nQuestion: {question}'}],
          llm_settings)
      return text, True
    except llm_lib.LLMError as e:
      # Clean badge in chat; the technical error goes to the log only.
      import logging
      logging.getLogger('neurosilicon.llm').warning('LLM failed: %s', e)
      return ('[Offline Mode — Answered via local EDA analyzer]\n'
              + _offline_answer(question, ctx)
              + f'\n(cloud LLM unreachable after retries — {e}; '
                'details in the Log tab)'), False
  return _offline_answer(question, ctx), False
