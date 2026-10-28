"""Import a chip-project folder: scan RTL, synthesize, convert, baseline.

End-to-end flow used by the GUI Import page and reusable from scripts::

    stats = import_project(folder, files, top, outdir, settings)

Yosys runs as a subprocess (``tools/yosys`` by default); conversion and
baselines call the in-process Converter/evaluate APIs. Verilog (.v) is
fully supported; SystemVerilog (.sv) is attempted with ``read_verilog
-sv`` and reports the yosys error verbatim when the old parser chokes.
"""
import glob
import json
import os
import re
import shutil
import subprocess

_MODULE_RE = re.compile(r'^\s*module\s+(\w+)', re.MULTILINE)


def repo_root():
  """Absolute repo root (parent of the `neurosilicon` package)."""
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_tool(path):
  """Absolutizes a tool path against the repo root (not the cwd).

  Streamlit may be launched from any directory, so bare relative
  defaults like 'tools/yosys' must anchor at the repo, never the
  process working directory. Absolute paths pass through; ~ expands.
  """
  path = os.path.expanduser(path or '')
  if os.path.isabs(path):
    return path
  return os.path.join(repo_root(), path)


REGISTRY_PATH = os.path.join(repo_root(), 'logs', 'gui_runs',
                             'registry.json')


def load_registry(path=None):
  """Imported-project registry: {name: entry dict}. Missing -> {}.

  Entry: {netlist, plc, json, net, top, outdir}. Legacy [netlist, plc]
  lists are upgraded on load.
  """
  path = path or REGISTRY_PATH
  if not os.path.isfile(path):
    return {}
  with open(path) as f:
    data = json.load(f)
  if not isinstance(data, dict):
    return {}
  for name, entry in data.items():
    if isinstance(entry, list):
      data[name] = {'netlist': entry[0] if len(entry) > 0 else '',
                    'plc': entry[1] if len(entry) > 1 else '',
                    'json': '', 'net': '', 'top': '', 'outdir': ''}
  return data


def save_registry(registry, path=None):
  path = path or REGISTRY_PATH
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump(registry, f, indent=2)
  return path


def add_to_registry(name, netlist, plc, path=None, extra=None):
  reg = load_registry(path)
  entry = {'netlist': netlist, 'plc': plc, 'json': '', 'net': '',
           'top': '', 'outdir': ''}
  entry.update(extra or {})
  reg[name] = entry
  save_registry(reg, path)
  return reg


_RTL_EXTS = ('.v', '.sv')
_TESTBENCH_RE = re.compile(r'(^|[/_.\-])(tb|testbench|test_bench)([/_.\-]|$)',
                           re.IGNORECASE)
_INST_RE = re.compile(r'(?<![\w.])(\w+)\s+(?:#\s*\(|[A-Za-z_]\w*\s*\()')
_COND_RE = re.compile(r'^\s*`(ifdef|ifndef|else|endif)\b\s*(\w*)')


def _active_text(text, defined=frozenset()):
  """Strips `ifdef/`ifndef/`else/`endif regions (no defines set).

  Variant modules selected by `else branches (e.g. an FPGA barrel
  shifter under `ifndef ALTERA_FPGA) must not count as instantiated,
  or closure pruning keeps the very file it exists to drop.
  """
  out, stack = [], []  # stack: [parent_active, branch_taken]
  active = True
  for line in text.splitlines():
    m = _COND_RE.match(line)
    if m:
      kind, _name = m.group(1), m.group(2)
      if kind in ('ifdef', 'ifndef'):
        taken = (_name in defined) if kind == 'ifdef' else (_name not in defined)
        stack.append([active, taken])
        active = active and taken
      elif kind == 'else' and stack:
        parent, taken = stack[-1]
        stack[-1][1] = True
        active = parent and not taken
      elif kind == 'endif' and stack:
        active = stack.pop()[0]
      continue
    if active:
      out.append(line)
  return '\n'.join(out)
_KEYWORDS = frozenset(['module', 'endmodule', 'if', 'else', 'for', 'while',
                       'assign', 'always', 'initial', 'begin', 'end', 'case',
                       'generate', 'wire', 'reg', 'logic', 'input', 'output'])


def is_testbench(rel_path):
  """True for testbench-style files or folders (tb/, sim/, *_tb.v...)."""
  parts = re.split(r'[/\\]', rel_path)
  return any(_TESTBENCH_RE.search(p) for p in parts)


def prune_to_top(file_texts, top, defines=None):
  """Keeps only files in the instantiation closure of `top`.

  `file_texts`: {rel_path: source text}. Real projects ship files the
  top never instantiates — testbenches, FPGA/ASIC variants with
  duplicate modules, vendor stubs — and yosys parses every file it is
  given, so one unrelated file with old-frontend syntax (e.g.
  `33'h_1_0000...`) kills the whole import. Returns (kept, dropped)
  sorted rel-path lists. Files defining no module are kept (they may
  hold `defines/parameters other files need).
  """
  defined = frozenset(defines or ())
  defines = {}   # module -> file
  uses = {}      # file -> {instantiated modules}
  for rel in sorted(file_texts):
    text = _active_text(_synth_text(file_texts[rel]), defined)
    for m in _MODULE_RE.finditer(text):
      defines.setdefault(m.group(1), rel)
    uses[rel] = {m.group(1) for m in _INST_RE.finditer(text)
                 if m.group(1) not in _KEYWORDS}
  if top not in defines:
    return sorted(file_texts), []
  keep, queue = set(), [top]
  while queue:
    mod = queue.pop()
    rel = defines.get(mod)
    if rel is None or rel in keep:
      continue
    keep.add(rel)
    queue.extend(sorted(uses.get(rel, set()) - set(queue)))
  for rel in file_texts:
    if not any(m.group(1) for m in _MODULE_RE.finditer(file_texts[rel])):
      keep.add(rel)
  return sorted(keep), sorted(set(file_texts) - keep)


def suggest_top(tops, texts, hint=''):
  """Picks the most likely top module from scanned candidates.

  Preference: module matching `hint` (folder/file name), then a root
  module (defined but never instantiated), then sorted-first. This
  replaces the old alphabetical-first default, which silently picked
  unrelated modules (e.g. 'Blend') on real projects.
  """
  if not tops:
    return ''
  norm = re.sub(r'[^a-z0-9]', '', (hint or '').lower())
  for t in tops:
    if re.sub(r'[^a-z0-9]', '', t.lower()) == norm and norm:
      return t
  defined = set(tops)
  instantiated = set()
  for text in texts.values():
    for m in _INST_RE.finditer(_active_text(_synth_text(text))):
      if m.group(1) in defined and m.group(1) not in _KEYWORDS:
        instantiated.add(m.group(1))
  roots = sorted(t for t in tops if t not in instantiated)
  if roots:
    # The real top instantiates other modules; a stray stub does not.
    def children_of(t):
      own = ''.join(tx for tx in texts.values()
                    if re.search(r'^\s*module\s+' + re.escape(t) + r'\b',
                                 tx, re.MULTILINE))
      return sum(1 for m in _INST_RE.finditer(own)
                 if m.group(1) in defined and m.group(1) != t)
    return max(roots, key=lambda t: (children_of(t), t))
  return sorted(tops)[0]


def default_files(scan):
  """RTL files to synthesize by default: everything except testbenches."""
  files = [f for f in scan.get('verilog', []) + scan.get('systemverilog', [])
           if not is_testbench(f)]
  return files or list(scan.get('verilog', [])) + list(
      scan.get('systemverilog', []))


def scan_folder(path):
  """Lists RTL sources and candidate top modules under `path`.

  `path` may be a project folder OR a single .v/.sv file (the natural
  thing a file picker returns); a file scans as itself. Extensions
  match case-insensitively (.V/.SV count). Returns {'verilog': [...],
  'systemverilog': [...], 'headers': [...], 'tops': [...],
  'suggested_top': str, 'testbenches': [...]} with paths relative to
  the scanned folder, sorted. Raises ValueError when the path does
  not exist or holds no RTL.
  """
  if os.path.isfile(path):
    if not path.lower().endswith(_RTL_EXTS):
      raise ValueError(f'not an RTL file (.v/.sv): {path}')
    root, single = os.path.dirname(os.path.abspath(path)), path
  elif os.path.isdir(path):
    root, single = path, None
  else:
    raise ValueError(f'not a folder or RTL file: {path}')
  verilog, sv, headers = [], [], []
  if single:
    rel = os.path.basename(single)
    (sv if rel.lower().endswith('.sv') else verilog).append(rel)
  else:
    for dirpath, _, files in os.walk(root):
      for fn in sorted(files):
        low = fn.lower()
        rel = os.path.relpath(os.path.join(dirpath, fn), root)
        if low.endswith('.sv'):
          sv.append(rel)
        elif low.endswith('.v'):
          verilog.append(rel)
        elif low.endswith('.vh') or low.endswith('.h'):
          headers.append(rel)
  if not verilog and not sv:
    raise ValueError(f'no .v/.sv files under {path}')
  tops, texts = [], {}
  for rel in verilog + sv:
    try:
      with open(os.path.join(root, rel), errors='replace') as f:
        text = f.read()
    except OSError:
      continue
    texts[rel] = text
    for m in _MODULE_RE.finditer(text):
      if m.group(1) not in tops:
        tops.append(m.group(1))
  hint = os.path.splitext(os.path.basename(single))[0] if single \
      else os.path.basename(os.path.normpath(root))
  all_rtl = sorted(verilog) + sorted(sv)
  # Testbenches instantiate the design, so they must not vote on the
  # top: suggest only among modules defined in non-testbench files.
  design_texts = {f: t for f, t in texts.items() if not is_testbench(f)}
  design_tops = sorted({m.group(1) for t in design_texts.values()
                        for m in _MODULE_RE.finditer(t)})
  return {'verilog': sorted(verilog), 'systemverilog': sorted(sv),
          'headers': sorted(headers), 'tops': sorted(tops),
          'root': os.path.abspath(root),
          'testbenches': [f for f in all_rtl if is_testbench(f)],
          'suggested_top': suggest_top(design_tops or sorted(tops),
                                       design_texts or texts, hint)}


def parse_params(text):
  """Parses 'K=V, K=V' chparam text into an ordered dict."""
  out = {}
  for chunk in (text or '').split(','):
    chunk = chunk.strip()
    if not chunk:
      continue
    if '=' not in chunk:
      raise ValueError(f'bad param {chunk!r}, want NAME=value')
    k, v = chunk.split('=', 1)
    k, v = k.strip(), v.strip()
    if not k or not v:
      raise ValueError(f'bad param {chunk!r}, want NAME=value')
    out[k] = v
  return out


def resolve_pdk(pdk_settings):
  """Fills blank PDK fields from the ciel sky130 install."""
  out = dict(pdk_settings)
  blanks = [k for k in ('lef', 'liberty', 'techlef') if not out.get(k)]
  if blanks:
    roots = sorted(glob.glob(
        os.path.expanduser('~/.ciel/ciel/sky130/versions/*/sky130A')))
    if not roots:
      raise ValueError(
          'PDK fields blank and no ciel sky130 install found; set them '
          'in Settings > PDK & Toolchain.')
    base = roots[-1] + '/libs.ref/sky130_fd_sc_hd/'
    out.setdefault('lef', '')
    out.setdefault('liberty', '')
    out.setdefault('techlef', '')
    if not out['lef']:
      out['lef'] = base + 'lef/sky130_fd_sc_hd.lef'
    if not out['liberty']:
      out['liberty'] = (base + 'lib/sky130_fd_sc_hd__tt_100C_1v80.lib')
    if not out['techlef']:
      out['techlef'] = base + 'techlef/sky130_fd_sc_hd__nom.tlef'
  for k in ('lef', 'liberty', 'techlef'):
    if not os.path.isfile(out[k]):
      raise ValueError(f'PDK file missing: {k}={out[k]}')
  return out


def synth_command(files, top, outdir, liberty, params=None, defines=None):
  """Builds the yosys -p script (single string) for the standard flow."""
  verilog = [fn for fn in files if not fn.lower().endswith('.sv')]
  sverilog = [fn for fn in files if fn.lower().endswith('.sv')]
  reads = []
  if defines:
    reads.append('verilog_defines '
                 + ' '.join(f'-D{d}' for d in defines))
  if verilog:
    reads.append('read_verilog ' + ' '.join(verilog))
  if sverilog:
    reads.append('read_verilog -sv ' + ' '.join(sverilog))
  chparams = ' '.join(f'chparam -set {k} {v} {top};'
                      for k, v in (params or {}).items())
  return ('; '.join(reads) +
          f'; {chparams}hierarchy -check -top {top}; synth -top {top}; '
          f'abc -liberty {liberty}; dfflibmap -liberty {liberty}; '
          f'abc -liberty {liberty}; opt; autoname; stat; '
          f'write_json {outdir}/design.json; '
          f'write_verilog -noattr {outdir}/design_net.v')


def run_synth(files, top, build_dir, liberty, yosys_bin, params=None,
              timeout=3600, defines=None):
  """Runs yosys in `build_dir`. Returns {json, net, log, cells}.

  All paths are absolutized: yosys runs with cwd=build_dir, so relative
  tool/output paths would resolve inside it. Raises RuntimeError with
  the yosys log tail on failure (including a missing binary). Giant
  designs may need a larger `timeout` (several hours for 3M+ cells).
  On timeout the partial yosys output is still written to synth.log.
  """
  build_dir = os.path.abspath(build_dir)
  yosys_bin = resolve_tool(yosys_bin)
  if not (os.path.isfile(yosys_bin) and os.access(yosys_bin, os.X_OK)):
    raise RuntimeError(
        f'yosys binary not found or not executable: {yosys_bin} '
        '(set Settings > PDK & Toolchain > yosys, absolute path).')
  os.makedirs(build_dir, exist_ok=True)
  script = synth_command(files, top, build_dir,
                         os.path.abspath(liberty), params, defines)
  log_path = os.path.join(build_dir, 'synth.log')
  try:
    proc = subprocess.run(
        [yosys_bin, '-p', script], cwd=build_dir, capture_output=True,
        text=True, timeout=timeout)
  except subprocess.TimeoutExpired as e:
    partial = (f'$ {yosys_bin} -p {script!r}\n[TIMEOUT after {timeout}s]\n'
               f'{e.stdout or ""}{e.stderr or ""}')
    with open(log_path, 'w') as f:
      f.write(partial)
    raise RuntimeError(
        f'yosys timed out after {timeout}s (partial log: {log_path}); '
        f'retry with a larger timeout')
  except OSError as e:
    raise RuntimeError(f'yosys launch failed ({yosys_bin}): {e}')
  log = f'$ {yosys_bin} -p {script!r}\n' + proc.stdout + proc.stderr
  with open(log_path, 'w') as f:
    f.write(log)
  json_path = os.path.join(build_dir, 'design.json')
  net_path = os.path.join(build_dir, 'design_net.v')
  if proc.returncode != 0 or not os.path.isfile(json_path):
    tail = '\n'.join(log.strip().splitlines()[-15:])
    raise RuntimeError(f'yosys failed (see {log_path}):\n{tail}')
  cells = None
  for line in proc.stdout.splitlines():
    m = re.search(r'Number of cells:\s+(\d+)', line)
    if m:
      cells = int(m.group(1))
  return {'json': json_path, 'net': net_path, 'log': log_path,
          'cells': cells}


def import_project(folder, files, top, outdir, settings, params=None,
                   episodes=2, fd_steps=5, progress=None, init_fd=True,
                   cluster_size=25, grid_sweep_step=1, defines=None):
  """Full folder -> synth -> proto -> baselines pipeline.

  Args:
    folder: chip-project folder (scanned by scan_folder).
    files: RTL paths relative to `folder` to synthesize.
    top: top module name.
    outdir: run folder (gets build/, proto/, baseline/ subdirs).
    settings: settings.load() dict (uses pdk/run sections).
    params: {chparam: value} applied to `top`.
    episodes: baseline episodes per method.
    fd_steps: FD steps per stage for baselines.
    progress: optional callback(stage, message).
    init_fd: run construction-time mixed-size FD before each baseline
      episode (exact, slow); False skips it for time-boxed runs.
    cluster_size: target stdcells per soft-macro cluster. Raise for
      giant (>500k-cell) designs so episodes stay tractable.
    grid_sweep_step: grid auto-select sweep stride (1 = exact AlphaChip
      sweep; larger is much faster, slightly different optima).

  Returns stats dict with build/proto/baseline artifacts and costs.
  """
  from neurosilicon import evaluate as evaluate_lib
  from neurosilicon import runlog as runlog_lib
  from neurosilicon import synth_to_proto as converter_lib

  def note(stage, message):
    if progress is not None:
      progress(stage, message)

  outdir = (outdir if os.path.isabs(outdir)
            else os.path.join(repo_root(), outdir))
  folder = os.path.abspath(folder)
  runlog_lib.write_manifest(outdir, 'import', {
      'folder': folder, 'files': list(files), 'top': top,
      'params': dict(params or {}), 'episodes': episodes,
      'fd_steps': fd_steps, 'init_fd': init_fd,
      'cluster_size': cluster_size, 'grid_sweep_step': grid_sweep_step})
  pdk = resolve_pdk(settings['pdk'])
  yosys_bin = settings['pdk'].get('yosys', 'tools/yosys')
  build_dir = os.path.join(outdir, 'build')
  os.makedirs(build_dir, exist_ok=True)
  # Prune to the top's instantiation closure: unrelated files
  # (testbenches, duplicate-module variants) otherwise break yosys.
  texts = {}
  for rel in files:
    try:
      with open(os.path.join(folder, rel), errors='replace') as f:
        texts[rel] = f.read()
    except OSError:
      continue
  kept, dropped = prune_to_top(texts, top, defines)
  if dropped:
    note('synth', f'pruned {len(dropped)} file(s) unreachable from '
                  f'{top}: {dropped[:6]}'
                  f'{"..." if len(dropped) > 6 else ""}')
    files = [f for f in files if f in kept]
  # Copy sources locally: yosys argument parsing breaks on spaces, and
  # the build dir keeps a record of exactly what was synthesized.
  # Headers (.vh/.h) ride along so `include directives resolve; on a
  # basename collision the first copy wins and the clash is recorded.
  local = []
  seen = set()
  for rel in files:
    src = os.path.join(folder, rel)
    if not os.path.isfile(src):
      raise ValueError(f'source missing: {src}')
    dst = os.path.join(build_dir, os.path.basename(rel))
    if os.path.basename(rel) in seen:
      note('synth', f'basename clash, keeping first: {rel}')
      continue
    seen.add(os.path.basename(rel))
    _stage_copy(src, dst)
    local.append(os.path.basename(rel))
  try:
    headers = scan_folder(folder).get('headers', [])
  except ValueError:
    headers = []
  for rel in headers:
    base = os.path.basename(rel)
    if base in seen:
      continue
    seen.add(base)
    _stage_copy(os.path.join(folder, rel),
                    os.path.join(build_dir, base))
  _stage_includes(folder, build_dir,
                  [texts[f] for f in files if f in texts], note)
  return _run_pipeline(local, top, outdir, settings, pdk, yosys_bin,
                       params, episodes, fd_steps, note, init_fd,
                       cluster_size, grid_sweep_step, defines)


_INCLUDE_RE = re.compile(r'`include\s+"([^"]+)"')
_TRANSLATE_OFF_RE = re.compile(
    r'(//|/\*)\s*(synopsys|synthesis)\s+translate_off', re.IGNORECASE)
_TRANSLATE_ON_RE = re.compile(
    r'(//|/\*)\s*(synopsys|synthesis)\s+translate_on', re.IGNORECASE)


_REAL_DECL_RE = re.compile(r'^(\s*)(?:real|realtime)\s+(?=[A-Za-z_])')
_REAL_PARAM_RE = re.compile(r'\bparameter\s+(?:real|realtime)\s+')
_INITIAL_RE = re.compile(r'^\s*initial\b')
_BEGIN_END_RE = re.compile(r'\b(begin|end)\b')


def _strip_sim_initials(lines):
  """Drops `initial` blocks holding simulation-only constructs.

  Behavioral clock generators (Amber's `clocks_resets.v`: `real`
  periods measured with `$time`, `#` delays) live in `initial` blocks
  the bundled Verilog-2005 frontend cannot parse (TOK_REAL). Blocks
  with `#` delays or `$time` are simulation-only and never affect
  the netlist structure a floorplan needs, so the staged copy drops
  them; plain initialization blocks (including $readmemh preloads)
  are kept.
  """
  out, block, depth, started = [], None, 0, False
  for line in lines:
    if block is None:
      if _INITIAL_RE.match(line):
        block, depth, started = [line], 0, False
        for tok in _BEGIN_END_RE.findall(line):
          started = True
          depth += 1 if tok == 'begin' else -1
        if started and depth <= 0 or ';' in line and not started:
          if '#' not in line and '$time' not in line:
            out.extend(block)
          block = None
        continue
      out.append(line)
      continue
    block.append(line)
    for tok in _BEGIN_END_RE.findall(line):
      started = True
      depth += 1 if tok == 'begin' else -1
    if (started and depth <= 0) or (not started and ';' in line):
      text = '\n'.join(block)
      if '#' not in text and '$time' not in text:
        out.extend(block)
      block = None
  if block is not None:
    out.extend(block)
  return out


def _synth_text(text):
  """Prepares a source file for the bundled yosys frontend (staging).

  Real synthesis tools skip simulation-only code; the staged copy
  does the same so one behavioral block cannot kill a floorplan
  import. Sources are never touched. Transformations:

  * synopsys/synthesis translate_off regions are stripped (debug-only
    blocks such as Amber's decompiler with $fwrite / TB macros).
  * `initial` blocks containing `#` delays or `$time` are dropped
    (behavioral clock measurement/generation; see _strip_sim_initials).
  * `real`/`realtime` declarations become `integer`: the Verilog-2005
    frontend has no TOK_REAL, and the remaining uses are delay
    arithmetic in behavioral code where integer precision is fine
    for floorplanning.
  """
  out, off = [], False
  for line in text.splitlines():
    if _TRANSLATE_OFF_RE.search(line):
      off = True
      continue
    if _TRANSLATE_ON_RE.search(line):
      off = False
      continue
    if not off:
      out.append(line)
  out = _strip_sim_initials(out)
  out = [_REAL_DECL_RE.sub(r'\1integer ', _REAL_PARAM_RE.sub(
      'parameter ', line)) for line in out]
  return '\n'.join(out) + '\n'


def _stage_copy(src, dst):
  with open(src, errors='replace') as f:
    text = _synth_text(f.read())
  with open(dst, 'w') as f:
    f.write(text)


def _stage_includes(search_root, build_dir, seed_texts, note):
  """Copies `include targets of staged files into `build_dir`.

  Amber's a23_cache.v includes global_defines.v, which lives in the
  testbench folder — excluded from synthesis, but yosys still needs
  the file present or it aborts. Basenames are resolved anywhere
  under `search_root`; already-staged names win.
  """
  index = {}
  for dirpath, _, files in os.walk(search_root):
    for fn in files:
      index.setdefault(fn, os.path.join(dirpath, fn))
  queue = list(seed_texts)
  staged = set(os.listdir(build_dir))
  while queue:
    text = queue.pop()
    for inc in _INCLUDE_RE.findall(text):
      base = os.path.basename(inc)
      if base in staged:
        continue
      src = index.get(base)
      if src is None:
        # Generated files (e.g. Amber's boot-loader memparams, made
        # by sw/tools scripts from a compiled ELF) are often absent.
        # Stub them empty and say so: ROM contents don't affect the
        # floorplan, but never use this import for a bitstream.
        with open(os.path.join(build_dir, base), 'w') as f:
          f.write(f'// NeuroSilicon stub: {inc} not found in project '
                  '(generated file?); empty for floorplan synthesis.\n')
        note('synth', f'WARNING stubbed missing include: {inc}')
        staged.add(base)
        continue
      _stage_copy(src, os.path.join(build_dir, base))
      staged.add(base)
      note('synth', f'staged include: {base}')
      with open(src, errors='replace') as f:
        queue.append(f.read())


_INST_FULL_RE = re.compile(
    r'(?<![\w.$])([A-Za-z_]\w*)\s*'
    r'(?:#\s*\(((?:[^()]|\([^()]*\))*)\)\s*)?'
    r'[A-Za-z_]\w*\s*\((.*?)\)\s*;', re.DOTALL)
_PORT_RE = re.compile(r'\.(\w+)\s*\(')


def _strip_comments(text):
  text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
  return re.sub(r'//[^\n]*', ' ', text)


def blackbox_stubs(file_texts, defined, define_names=None):
  """Generates blackbox stubs for instantiated-but-undefined modules.

  Vendor primitives (Xilinx DSP48A1, PLL_ADV…) have no source in the
  project; real flows supply vendor libraries. For floorplanning a
  port-compatible blackbox is enough: ports/params are collected
  from the instantiation sites. Returns stub source ('' if none).
  """
  ports, params = {}, {}
  for text in file_texts.values():
    active = _strip_comments(
        _active_text(_synth_text(text), frozenset(define_names or ())))
    for m in _INST_FULL_RE.finditer(active):
      mod = m.group(1)
      if mod in defined or mod in _KEYWORDS:
        continue
      found_ports = _PORT_RE.findall(m.group(3))
      found_params = _PORT_RE.findall(m.group(2)) if m.group(2) else []
      if not found_ports and not found_params:
        continue  # function call / control flow, not an instantiation
      ports.setdefault(mod, set()).update(found_ports)
      if found_params:
        params.setdefault(mod, set()).update(found_params)
  out = []
  for mod in sorted(ports):
    ps = ''.join(f'parameter {p} = 0, ' for p in sorted(params.get(mod, ())))
    io = ', '.join(f'inout wire [0:0] {p}' for p in sorted(ports[mod]))
    out.append(f'(* blackbox *) module {mod} #({ps.rstrip(", ")}) '
               f'({io}); endmodule')
  return '\n'.join(out) + ('\n' if out else '')


_MEM_DECL_RE = re.compile(
    r'\breg\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?'
    r'[A-Za-z_]\w*\s*\[\s*([^\]]+)\]')
_PARAM_VAL_RE = re.compile(
    r'\b(?:localparam|parameter)\s+(?:\[[^\]]*\]\s*)?(\w+)\s*=\s*(\d+)\s*;')
_MODULE_SPAN_RE = re.compile(r'^\s*module\s+(\w+)\b.*?^\s*endmodule\b',
                             re.DOTALL | re.MULTILINE)


def _eval_int_expr(expr, params):
  """Evaluates a small integer constant expression, or None."""
  expr = expr.strip()
  if not re.fullmatch(r'[0-9A-Za-z_+\-*/%() ]+', expr):
    return None
  for name, val in params.items():
    expr = re.sub(r'\b' + re.escape(name) + r'\b', str(val), expr)
  if re.search(r'[A-Za-z_]', expr):
    return None
  try:
    return int(eval(expr, {'__builtins__': {}}, {}))  # noqa: S307
  except Exception:
    return None


def huge_memory_modules(file_texts, threshold_bits=1 << 20):
  """{module: bits} for modules declaring >= `threshold_bits` of RAM.

  Behavioral memory models (Amber's 128-bit x 16M-word `main_mem`)
  are simulation models of external memory; synthesizing them into
  gates exhausts memory in the bundled yosys. They must enter the
  floorplan as black-box hard macros instead.
  """
  params = {}
  for text in file_texts.values():
    for name, val in _PARAM_VAL_RE.findall(text):
      params.setdefault(name, int(val))
  huge = {}
  for text in file_texts.values():
    for span in _MODULE_SPAN_RE.finditer(text):
      body = span.group(0)
      for hi_w, lo_w, depth in _MEM_DECL_RE.findall(body):
        width = abs(int(hi_w) - int(lo_w)) + 1 if hi_w else 1
        parts = depth.split(':')
        hi = _eval_int_expr(parts[0], params)
        lo = _eval_int_expr(parts[1], params) if len(parts) > 1 else 0
        if hi is None or lo is None:
          continue
        bits = width * (abs(hi - lo) + 1)
        if bits >= threshold_bits:
          huge[span.group(1)] = max(bits, huge.get(span.group(1), 0))
  return huge


def _module_stub(module_text, name):
  """Blackbox stub keeping the module header (params + ports).

  The header ends at the `;` after the port list closes; parens are
  depth-counted on a comment-blanked copy so `#( ... )` parameter
  blocks, nested port parens, and comment text cannot cut it short.
  """
  m = re.search(r'^\s*module\s+' + re.escape(name) + r'\b', module_text,
                re.MULTILINE)
  head = module_text[m.start():] if m else module_text
  blanked = re.sub(r'//[^\n]*', lambda c: ' ' * len(c.group(0)), head)
  depth, cut = 0, len(head)
  for i, ch in enumerate(blanked):
    if ch == '(':
      depth += 1
    elif ch == ')':
      depth -= 1
    elif ch == ';' and depth <= 0:
      cut = i + 1
      break
  header = head[:cut].strip()
  return f'(* blackbox *) {header}\nendmodule\n'


def _blackbox_huge_memories(local, build_dir, note, threshold_bits=1 << 20):
  """Replaces huge behavioral memories in staged files with stubs."""
  texts = {}
  for base in local:
    try:
      with open(os.path.join(build_dir, base), errors='replace') as f:
        texts[base] = f.read()
    except OSError:
      continue
  huge = huge_memory_modules(texts, threshold_bits)
  if not huge:
    return local
  for base, text in texts.items():
    def _sub(span):
      name = span.group(1)
      if name not in huge:
        return span.group(0)
      return _module_stub(span.group(0), name)
    new = _MODULE_SPAN_RE.sub(_sub, text)
    if new != text:
      with open(os.path.join(build_dir, base), 'w') as f:
        f.write(new)
  for name, bits in sorted(huge.items()):
    note('synth', f'blackboxed huge behavioral memory {name} '
                  f'({bits / 8 / 1e6:.0f} MB model) as a hard macro; '
                  'its placed size is nominal, not signoff')
  return local


_DELAY_STMT_RE = re.compile(r'^\s*#(\s+\(|\d)')
_ALWAYS_DELAY_RE = re.compile(r'\balways\s*(?:@\s*\([^)]*\)\s*)?#')
_PLUSARGS_RE = re.compile(r'\$(?:value|test)\$plusargs|\$plusargs')


def sim_only_modules(file_texts):
  """Modules whose bodies are behavioral simulation models.

  Delay statements (`# (period/2)` clock generators), `$time`
  measurement, `always #…` delay clocks, and `$plusargs` testbench
  plumbing have no gate-level meaning; left in, they elaborate to
  degenerate networks that crash ABC — or fail parsing outright on
  makefile-defined macros (`CLOCK_PERIOD in rocket-chip's
  TestDriver). Amber's `clocks_resets` (non-FPGA branch) is the
  canonical case. Parameter overrides (`inst #(.W(2)) u (...)`)
  do not match: delay statements start the line, and overrides are
  never preceded by `always`.
  """
  out = set()
  for text in file_texts.values():
    for span in _MODULE_SPAN_RE.finditer(text):
      body = span.group(0)
      if ('$time' in body or _ALWAYS_DELAY_RE.search(body)
          or _PLUSARGS_RE.search(body) or any(
              _DELAY_STMT_RE.match(line)
              for line in body.splitlines())):
        out.add(span.group(1))
  return out


def _blackbox_sim_only(local, build_dir, note):
  """Replaces simulation-only modules in staged files with stubs."""
  texts = {}
  for base in local:
    try:
      with open(os.path.join(build_dir, base), errors='replace') as f:
        texts[base] = f.read()
    except OSError:
      continue
  sim_only = sim_only_modules(texts)
  if not sim_only:
    return local
  for base, text in texts.items():
    def _sub(span):
      if span.group(1) not in sim_only:
        return span.group(0)
      return _module_stub(span.group(0), span.group(1))
    new = _MODULE_SPAN_RE.sub(_sub, text)
    if new != text:
      with open(os.path.join(build_dir, base), 'w') as f:
        f.write(new)
  for name in sorted(sim_only):
    note('synth', f'blackboxed simulation-only module {name} '
                  '(behavioral delays/$time) as a hard macro')
  return local


def _with_blackboxes(local, build_dir, note, define_names=None):
  """Appends a generated blackbox file for vendor primitives."""
  texts, defined = {}, set()
  for base in local:
    try:
      with open(os.path.join(build_dir, base), errors='replace') as f:
        texts[base] = f.read()
    except OSError:
      continue
    defined.update(m.group(1) for m in _MODULE_RE.finditer(texts[base]))
  stubs = blackbox_stubs(texts, defined, define_names)
  if not stubs:
    return local
  name = 'neurosilicon_blackboxes.v'
  with open(os.path.join(build_dir, name), 'w') as f:
    f.write('// Auto-generated vendor-primitive blackboxes.\n' + stubs)
  note('synth', f'blackboxed vendor primitives: '
                f'{stubs.count("module")} (see {name})')
  return list(local) + [name]


def _run_pipeline(local, top, outdir, settings, pdk, yosys_bin, params,
                  episodes, fd_steps, note, init_fd, cluster_size,
                  grid_sweep_step, defines=None):
  """Synth -> proto -> baselines for staged basenames in build/."""
  from neurosilicon import evaluate as evaluate_lib
  from neurosilicon import synth_to_proto as converter_lib
  build_dir = os.path.join(outdir, 'build')
  local = _blackbox_huge_memories(local, build_dir, note)
  local = _blackbox_sim_only(local, build_dir, note)
  local = _with_blackboxes(local, build_dir, note, defines)
  note('synth', f'synthesizing {top} ({len(local)} files)...')
  synth = run_synth(local, top, build_dir, pdk['liberty'], yosys_bin,
                    params, defines=defines)
  note('proto', f"converting ({synth['cells']} cells)...")
  proto_dir = os.path.join(outdir, 'proto')
  stats = converter_lib.Converter(
      synth['json'], pdk['lef'], top,
      utilization=settings['pdk'].get('utilization', 0.8),
      canvas_utilization=settings['pdk'].get('canvas_util', 0.35),
  ).convert(proto_dir, cluster_size=cluster_size,
            grid_sweep_step=grid_sweep_step)
  note('baseline', 'running baselines...')
  base_dir = os.path.join(outdir, 'baseline')
  results = evaluate_lib.run_baselines(
      stats['netlist'], stats['plc'], episodes=episodes,
      fd_steps=(fd_steps,) * 3, outdir=base_dir, init_fd=init_fd)
  best = min((r['proxy_cost'] for r in results
              if r['proxy_cost'] > 0), default=-1)
  summary = {'top': top, 'cells': synth['cells'], 'files': local,
             'build_dir': build_dir, 'proto': stats,
             'baseline_dir': base_dir, 'results': results,
             'best_proxy': best}
  with open(os.path.join(outdir, 'import.json'), 'w') as f:
    json.dump({k: v for k, v in summary.items() if k != 'results'}, f,
              indent=2)
  note('done', f'best proxy {best:.4f}')
  return summary


def import_files(paths, top, outdir, settings, params=None, episodes=2,
                 fd_steps=5, progress=None, init_fd=True, cluster_size=25,
                 grid_sweep_step=1, defines=None):
  """Synthesizes explicit RTL file paths (what a file picker returns).

  Each path must exist and end in .v/.sv (any case). Files are staged
  into `outdir/build/` by basename; a basename collision is an error
  (rename a file) rather than a silent drop. Returns the same summary
  dict as import_project.
  """
  from neurosilicon import runlog as runlog_lib

  def note(stage, message):
    if progress is not None:
      progress(stage, message)

  paths = [os.path.abspath(p) for p in paths]
  if not paths:
    raise ValueError('no RTL files picked')
  for p in paths:
    if not os.path.isfile(p):
      raise ValueError(f'source missing: {p}')
    if not p.lower().endswith(_RTL_EXTS):
      raise ValueError(f'not an RTL file (.v/.sv): {p}')
  names = [os.path.basename(p) for p in paths]
  dupes = sorted({n for n in names if names.count(n) > 1})
  if dupes:
    raise ValueError(f'duplicate file names picked: {dupes}; '
                     'rename one so each name is unique')
  outdir = (outdir if os.path.isabs(outdir)
            else os.path.join(repo_root(), outdir))
  runlog_lib.write_manifest(outdir, 'import', {
      'files': paths, 'top': top, 'params': dict(params or {}),
      'episodes': episodes, 'fd_steps': fd_steps, 'init_fd': init_fd,
      'cluster_size': cluster_size, 'grid_sweep_step': grid_sweep_step})
  pdk = resolve_pdk(settings['pdk'])
  yosys_bin = settings['pdk'].get('yosys', 'tools/yosys')
  build_dir = os.path.join(outdir, 'build')
  os.makedirs(build_dir, exist_ok=True)
  texts = {}
  for src, base in zip(paths, names):
    with open(src, errors='replace') as f:
      texts[base] = f.read()
  kept, dropped = prune_to_top(texts, top, defines)
  if dropped:
    note('synth', f'pruned {len(dropped)} file(s) unreachable from {top}')
    paths = [p for p, b in zip(paths, names) if b in kept]
    names = kept
  for src, base in zip(paths, names):
    _stage_copy(src, os.path.join(build_dir, base))
  _stage_includes(os.path.commonpath([os.path.dirname(p) for p in paths]),
                  build_dir, list(texts.values()), note)
  return _run_pipeline(names, top, outdir, settings, pdk, yosys_bin,
                       params, episodes, fd_steps, note, init_fd,
                       cluster_size, grid_sweep_step, defines)


def register_netlist(name, netlist, plc, path=None, extra=None):
  """Registers a ready proto netlist pair (no synthesis).

  Validates both files exist, the netlist is non-empty, and the .plc
  header parses with canvas/grid; raises ValueError otherwise so a
  bad pick fails loudly at import instead of at placement time.
  """
  from neurosilicon import setup as setup_lib
  netlist, plc = os.path.abspath(netlist), os.path.abspath(plc)
  if not os.path.isfile(netlist) or os.path.getsize(netlist) == 0:
    raise ValueError(f'netlist missing or empty: {netlist}')
  if not os.path.isfile(plc):
    raise ValueError(f'placement missing: {plc}')
  header = setup_lib.parse_plc_header(plc)
  if 'width' not in header:
    raise ValueError(f'placement has no canvas header: {plc}')
  entry = {'netlist': netlist, 'plc': plc, 'json': '', 'net': '',
           'top': '', 'outdir': os.path.dirname(netlist)}
  entry.update(extra or {})
  reg = load_registry(path)
  reg[name] = entry
  save_registry(reg, path)
  return entry
