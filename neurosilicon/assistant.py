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
from neurosilicon import llm as llm_lib

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
    if best:
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
    try:
      text = llm_lib.chat(
          [{'role': 'system', 'content': _SYSTEM},
           {'role': 'user',
            'content': 'Context (MCP tool results):\n' + '\n'.join(facts)
                       + f'\n\nQuestion: {question}'}],
          llm_settings)
      return text, True
    except llm_lib.LLMError as e:
      return (f'(LLM unavailable: {e})\n'
              + _offline_answer(question, ctx)), False
  return _offline_answer(question, ctx), False
