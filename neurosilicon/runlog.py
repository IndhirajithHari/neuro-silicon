"""Structured logging and run manifests for CLI tools and jobs.

Every long-running entry point (train, evaluate, import) gets:

- a ``logging`` logger writing timestamped lines to stderr **and**
  ``<outdir>/run.log`` (``get_logger``), and
- a ``<outdir>/manifest.json`` recording tool, package version, git
  hash, start time, and resolved arguments (``write_manifest``),

so any output directory is self-describing: what produced it, from
which code, with which inputs.
"""
import datetime
import json
import logging
import os
import subprocess
import sys

from neurosilicon._version import __version__


def git_hash():
  """Short HEAD hash of the repo, or 'unknown' (never raises)."""
  try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
        capture_output=True, text=True, timeout=10)
  except (OSError, subprocess.TimeoutExpired):
    return 'unknown'
  return out.stdout.strip() if out.returncode == 0 else 'unknown'


def get_logger(name, log_dir=None, filename='run.log'):
  """Returns a `neurosilicon.<name>` logger (stderr + file handlers).

  Idempotent: repeated calls with the same `name` return the same
  logger without duplicating handlers.
  """
  logger = logging.getLogger(f'neurosilicon.{name}')
  fmt = logging.Formatter(
      '%(asctime)s %(levelname)-7s %(name)s | %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S')
  if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    logger._ns_log_dirs = set()
  if log_dir:
    log_dir = os.path.abspath(log_dir)
    if log_dir not in logger._ns_log_dirs:
      os.makedirs(log_dir, exist_ok=True)
      fh = logging.FileHandler(os.path.join(log_dir, filename))
      fh.setFormatter(fmt)
      logger.addHandler(fh)
      logger._ns_log_dirs.add(log_dir)
  return logger


def write_manifest(outdir, tool, args):
  """Writes `<outdir>/manifest.json`; returns its path."""
  os.makedirs(outdir, exist_ok=True)
  manifest = {
      'tool': tool,
      'version': __version__,
      'git': git_hash(),
      'started_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      'argv': list(sys.argv),
      'args': {k: v for k, v in dict(args).items()},
  }
  path = os.path.join(outdir, 'manifest.json')
  with open(path, 'w') as f:
    json.dump(manifest, f, indent=2)
  return path
