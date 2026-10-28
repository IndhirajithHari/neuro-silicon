"""Tests for structured logging and run manifests."""
import json
import os

from neurosilicon import runlog as runlog_lib


def test_write_manifest(tmp_path):
  path = runlog_lib.write_manifest(str(tmp_path), 'train', {'a': 1})
  with open(path) as f:
    m = json.load(f)
  assert m['tool'] == 'train' and m['args'] == {'a': 1}
  assert m['version'] and m['git'] and m['started_at']
  assert m['argv']


def test_get_logger_writes_file(tmp_path):
  log = runlog_lib.get_logger('test_unit', log_dir=str(tmp_path))
  log.info('hello %s', 'world')
  for h in log.handlers:
    h.flush()
  with open(os.path.join(str(tmp_path), 'run.log')) as f:
    text = f.read()
  assert 'hello world' in text
  assert 'neurosilicon.test_unit' in text


def test_git_hash_never_raises():
  assert isinstance(runlog_lib.git_hash(), str)
