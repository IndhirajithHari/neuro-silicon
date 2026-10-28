"""Tests for persistent settings and LLM providers (mocked HTTP)."""
import io
import json
import os
from unittest import mock

from neurosilicon import llm as llm_lib
from neurosilicon import settings as settings_lib


def test_settings_defaults_have_help():
  d = settings_lib.defaults()
  assert set(d) == {'run', 'engine', 'model', 'llm', 'pdk'}
  for s in settings_lib.SECTIONS:
    for f in s['fields']:
      assert settings_lib.field_help(s['id'], f['key'])
  assert settings_lib.is_secret('llm', 'api_key')
  assert not settings_lib.is_secret('llm', 'model')


def test_settings_save_load_roundtrip(tmp_path):
  p = os.path.join(str(tmp_path), 's.json')
  assert settings_lib.load(p)['run']['episodes'] == 3
  s = settings_lib.defaults()
  s['run']['episodes'] = 7
  s['llm']['api_key'] = 'sk-test'
  settings_lib.save(s, p)
  back = settings_lib.load(p)
  assert back['run']['episodes'] == 7
  assert back['llm']['api_key'] == 'sk-test'
  assert back['engine']['congestion_weight'] == 0.5


def test_settings_redacted():
  s = settings_lib.defaults()
  s['llm']['api_key'] = 'sk-test'
  r = settings_lib.redacted(s)
  assert r['llm']['api_key'] == '(set)'
  assert 'sk-test' not in json.dumps(r)
  assert s['llm']['api_key'] == 'sk-test'


class _FakeResp:
  def __init__(self, payload):
    self._payload = json.dumps(payload).encode()

  def read(self):
    return self._payload

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False


def _llm(provider='api'):
  s = settings_lib.defaults()['llm']
  s['provider'] = provider
  s['api_key'] = 'sk-test'
  return s


def test_llm_api_chat_payload():
  body = {'choices': [{'message': {'content': 'hello chip'}}]}
  with mock.patch.object(llm_lib.urllib.request, 'urlopen',
                         return_value=_FakeResp(body)) as m:
    out = llm_lib.chat([{'role': 'user', 'content': 'hi'}], _llm('api'))
  assert out == 'hello chip'
  req = m.call_args[0][0]
  assert req.full_url == 'https://api.openai.com/v1/chat/completions'
  sent = json.loads(req.data.decode())
  assert sent['model'] == 'gpt-4o-mini'
  assert req.get_header('Authorization') == 'Bearer sk-test'


def test_llm_ollama_native_shape():
  body = {'message': {'content': 'local ok'}}
  with mock.patch.object(llm_lib.urllib.request, 'urlopen',
                         return_value=_FakeResp(body)) as m:
    out = llm_lib.chat([{'role': 'user', 'content': 'hi'}], _llm('local'))
  assert out == 'local ok'
  req = m.call_args[0][0]
  assert req.full_url == 'http://localhost:11434/api/chat'
  assert req.get_header('Authorization') is None


def test_list_models_api_and_ollama():
  body = {'data': [{'id': 'Llama-3.3-70B'}, {'id': 'Llama-4-Scout'}]}
  with mock.patch.object(llm_lib.urllib.request, 'urlopen',
                         return_value=_FakeResp(body)) as m:
    out = llm_lib.list_models(_llm('api'))
  assert out == ['Llama-3.3-70B', 'Llama-4-Scout']
  assert m.call_args[0][0].full_url == \
      'https://api.openai.com/v1/models'
  body = {'models': [{'name': 'llama3.1'}, {'name': 'qwen2.5'}]}
  with mock.patch.object(llm_lib.urllib.request, 'urlopen',
                         return_value=_FakeResp(body)):
    assert llm_lib.list_models(_llm('local')) == ['llama3.1', 'qwen2.5']


def test_llm_off_raises_clean_error():
  try:
    llm_lib.chat([], _llm('off'))
  except llm_lib.LLMError as e:
    assert 'off' in str(e)
  else:
    raise AssertionError('expected LLMError')


def test_llm_constraint_fallback_without_provider():
  mode, rects, explanation = llm_lib.parse_constraints(
      'Keep a 20% routing channel open in the center', 200.0, 100.0,
      _llm('off'))
  assert mode == 'block'
  assert len(rects) == 1
  assert 'fallback' in explanation


def test_llm_constraint_json_path():
  payload = ('{"mode": "block", "rects": [[80, 0, 120, 100]], '
             '"explanation": "center channel"}')
  body = {'choices': [{'message': {'content': payload}}]}
  with mock.patch.object(llm_lib.urllib.request, 'urlopen',
                         return_value=_FakeResp(body)):
    mode, rects, explanation = llm_lib.parse_constraints(
        'center channel please', 200.0, 100.0, _llm('api'))
  assert mode == 'block'
  assert rects == [(80.0, 0.0, 120.0, 100.0)]
  assert 'fallback' not in explanation
