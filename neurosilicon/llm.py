"""External AI API + local LLM providers (standard library only).

Two provider shapes, both driven by the ``llm`` settings section:

* ``api``: OpenAI-compatible ``POST {base_url}/chat/completions`` with a
  bearer key (cloud APIs, gateways, or OpenAI-compatible local servers
  such as LM Studio at ``http://localhost:1234/v1``).
* ``local``: native Ollama ``POST {local_url}/api/chat`` (no key).

Used for chat-to-floorplan constraint parsing (with rule-based fallback)
and for placement-result advice. All failures raise LLMError with a
plain message; nothing here prints keys.
"""
import http.client
import json
import ssl
import time
import urllib.error
import urllib.request


class LLMError(Exception):
  """LLM request or configuration failure (safe to show in UI)."""


def _clean_err(e):
  """Short, user-safe error text (never raw _ssl.c internals)."""
  s = str(e)
  if isinstance(e, (ssl.SSLError, EOFError)) or 'EOF' in s:
    return ('SSL EOF: the server closed the secure connection '
            'unexpectedly')
  return f'{type(e).__name__}: {s[:140]}'


_RETRYABLE = (ssl.SSLError, EOFError, ConnectionError, TimeoutError,
              http.client.HTTPException, urllib.error.URLError)


def _request_json(make_req, url, timeout):
  """Runs one JSON request with retry (Meta AI drops keep-alives).

  `Connection: close` prevents broken keep-alive socket reuse;
  transient SSL EOF / connection drops retry 3 times with
  exponential backoff (0.5s, 1s, 2s). HTTP errors are authoritative
  (no retry except 502/503/504). Final failure raises a clean
  LLMError — raw ssl traces stay out of the UI.
  """
  last = None
  for attempt in range(3):
    try:
      with urllib.request.urlopen(make_req(), timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
      body = e.read().decode('utf-8', 'replace')[:300]
      if e.code in (502, 503, 504) and attempt < 2:
        last = e
        time.sleep(0.5 * (2 ** attempt))
        continue
      raise LLMError(f'LLM HTTP {e.code}: {body}')
    except _RETRYABLE as e:
      last = e
      if attempt < 2:
        time.sleep(0.5 * (2 ** attempt))
        continue
      raise LLMError(
          f'LLM connection failed after 3 attempts ({_clean_err(e)}) '
          f'— {url}')
    except json.JSONDecodeError as e:
      raise LLMError(f'LLM bad response ({url}): {e}')
  raise LLMError(f'LLM connection failed ({_clean_err(last)}) — {url}')


def _headers(api_key=''):
  h = {'Content-Type': 'application/json', 'Connection': 'close'}
  if api_key:
    h['Authorization'] = f'Bearer {api_key}'
  return h


def _post_json(url, payload, timeout, api_key=''):
  data = json.dumps(payload).encode('utf-8')
  return _request_json(
      lambda: urllib.request.Request(url, data=data,
                                     headers=_headers(api_key)),
      url, timeout)


def chat(messages, llm_settings):
  """One chat round. Returns the assistant text.

  Args:
    messages: [{role, content}] list.
    llm_settings: the ``llm`` section dict from settings.load().

  Raises:
    LLMError: provider off, misconfigured, or request failed.
  """
  provider = llm_settings.get('provider', 'off')
  timeout = int(llm_settings.get('timeout', 60))
  if provider == 'api':
    base = llm_settings.get('base_url', '').rstrip('/')
    model = llm_settings.get('model', '')
    key = llm_settings.get('api_key', '')
    if not base or not model:
      raise LLMError('API provider needs a base URL and model name.')
    if not key:
      raise LLMError('API provider needs an API key.')
    body = _post_json(f'{base}/chat/completions',
                      {'model': model, 'messages': messages,
                       'temperature': 0.2},
                      timeout, api_key=key)
    try:
      return body['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
      raise LLMError('API returned an unexpected chat payload.')
  if provider == 'local':
    base = llm_settings.get('local_url', '').rstrip('/')
    model = llm_settings.get('local_model', '')
    if not base or not model:
      raise LLMError('Local provider needs a server URL and model name.')
    if base.endswith('/v1'):
      body = _post_json(f'{base}/chat/completions',
                        {'model': model, 'messages': messages,
                         'temperature': 0.2},
                        timeout)
      try:
        return body['choices'][0]['message']['content']
      except (KeyError, IndexError, TypeError):
        raise LLMError('Local server returned an unexpected payload.')
    body = _post_json(f'{base}/api/chat',
                      {'model': model, 'messages': messages,
                       'stream': False,
                       'options': {'temperature': 0.2}},
                      timeout)
    try:
      return body['message']['content']
    except (KeyError, TypeError):
      raise LLMError('Ollama returned an unexpected payload.')
  raise LLMError('LLM provider is off (Settings > AI Providers).')


def _get_json(url, timeout, api_key=''):
  req = urllib.request.Request(
      url, headers={'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'} if api_key
      else {'Content-Type': 'application/json'})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return json.loads(resp.read().decode('utf-8'))
  except urllib.error.HTTPError as e:
    body = e.read().decode('utf-8', 'replace')[:300]
    raise LLMError(f'LLM HTTP {e.code}: {body}')
  except urllib.error.URLError as e:
    raise LLMError(f'LLM unreachable ({url}): {e.reason}')
  except (TimeoutError, json.JSONDecodeError) as e:
    raise LLMError(f'LLM bad response ({url}): {e}')


def list_models(llm_settings):
  """Exact model IDs the configured endpoint offers. [str]

  api / OpenAI-compatible local: GET {base}/models. Ollama native:
  GET {base}/api/tags. Raises LLMError with the server's message —
  a model_not_found on Test almost always means the typed name is
  not one of these IDs (assistant/app names are not model IDs).
  """
  provider = llm_settings.get('provider', 'off')
  timeout = int(llm_settings.get('timeout', 60))
  if provider == 'api':
    base = llm_settings.get('base_url', '').rstrip('/')
    key = llm_settings.get('api_key', '')
    if not base or not key:
      raise LLMError('API provider needs a base URL and API key.')
    body = _get_json(f'{base}/models', timeout, api_key=key)
    return [m['id'] for m in body.get('data', []) if m.get('id')]
  if provider == 'local':
    base = llm_settings.get('local_url', '').rstrip('/')
    if not base:
      raise LLMError('Local provider needs a server URL.')
    if base.endswith('/v1'):
      body = _get_json(f'{base}/models', timeout)
      return [m['id'] for m in body.get('data', []) if m.get('id')]
    body = _get_json(f'{base}/api/tags', timeout)
    return [m['name'] for m in body.get('models', []) if m.get('name')]
  raise LLMError('LLM provider is off (Settings > AI Providers).')


def test_connection(llm_settings):
  """Minimal round-trip check. Returns (ok, message, seconds)."""
  t0 = time.time()
  try:
    chat([{'role': 'user',
           'content': 'Reply with exactly: floorplan link ok'}],
         llm_settings)
  except LLMError as e:
    return False, str(e), round(time.time() - t0, 1)
  return True, 'Connected.', round(time.time() - t0, 1)


_CONSTRAINT_SYSTEM = (
    'You translate chip-floorplan constraints into JSON. Reply with ONE '
    'JSON object only, no prose: {"mode": "block"|"allow"|"none", '
    '"rects": [[x0,y0,x1,y1], ...], "explanation": "..."}. Coordinates '
    'are microns on a canvas (0,0) to (W,H) given by the user. mode=block '
    'marks keep-out rectangles, mode=allow restricts macros to the '
    'rectangles, mode=none with [] means no constraint.')


def parse_constraints(text, canvas_w, canvas_h, llm_settings):
  """LLM constraint parse -> (mode, rects, explanation).

  Falls back to the rule-based parser on any LLM failure, and notes
  the fallback in the explanation. Never raises for LLM outages.
  """
  from neurosilicon import priors as priors_lib
  try:
    raw = chat(
        [{'role': 'system', 'content': _CONSTRAINT_SYSTEM},
         {'role': 'user',
          'content': f'Canvas W={canvas_w:.1f} H={canvas_h:.1f} microns. '
                     f'Constraint: {text}'}],
        llm_settings)
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
      raise LLMError('no JSON object in reply')
    spec = json.loads(raw[start:end + 1])
    mode = spec.get('mode', 'none')
    rects = [(float(x0), float(y0), float(x1), float(y1))
             for x0, y0, x1, y1 in spec.get('rects', [])]
    if mode not in ('block', 'allow', 'none'):
      raise LLMError(f'bad mode {mode!r}')
    if mode == 'none':
      rects = []
    return mode, rects, str(spec.get('explanation', 'LLM constraint.'))
  except (LLMError, ValueError, KeyError, TypeError) as e:
    mode, rects, explanation = priors_lib.chat_to_spec(
        text, canvas_w, canvas_h)
    return mode, rects, f'{explanation} (rule-based fallback: {e})'


def advise(results, llm_settings, context=''):
  """Ask the LLM to review run results. Returns markdown text.

  Args:
    results: list of {episode, proxy_cost, wirelength, congestion,
      density} dicts.
    context: optional extra facts (netlist, best cost, overflow).

  Raises:
    LLMError: provider off or request failed (caller shows it).
  """
  lines = ['episode proxy wl cong dens']
  for r in results:
    lines.append(
        f"{r.get('episode')} {r.get('proxy_cost', -1):.4f} "
        f"{r.get('wirelength', -1):.4f} {r.get('congestion', -1):.4f} "
        f"{r.get('density', -1):.4f}")
  return chat(
      [{'role': 'system',
        'content': ('You are a chip-floorplanning assistant. Review the '
                    'placement runs and reply in short markdown: verdict '
                    '(1 line), what the numbers say (bullets), and next '
                    'actions (bullets). Be concrete; cite episode ids.')},
       {'role': 'user',
        'content': f'{context}\n\n' + '\n'.join(lines)}],
      llm_settings)
