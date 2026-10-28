"""Auto-discover a whole chip design from any path the user picks.

The desktop Import flow used to make the user choose a mode first
(folder / files / ready netlist). `discover_design` removes that step:
give it a project folder, any single file inside the design, or a
`.zip` of the design, and it works out what floorplanning needs —
RTL + top module (synthesize), or a ready netlist + `.plc` pair (place
directly) — plus LEF/DEF/liberty files found along the way.

Qt-free so it is unit-testable in CI. Synthesis itself still happens
in `neurosilicon.project`; this module only inspects and plans.
"""
import os
import re
import zipfile

from neurosilicon import project as project_lib

_RTL_EXTS = ('.v', '.sv')
_MAX_WALK_FILES = 4000


def ensure_unpacked(path, dest):
  """Extracts a `.zip` design into `dest`; other paths pass through."""
  path = os.path.abspath(path)
  if not path.lower().endswith('.zip'):
    return path
  dest = os.path.abspath(dest)
  os.makedirs(dest, exist_ok=True)
  with zipfile.ZipFile(path) as z:
    for name in z.namelist():
      target = os.path.abspath(os.path.join(dest, name))
      if target != dest and not target.startswith(dest + os.sep):
        raise ValueError(f'unsafe path inside zip: {name}')
    z.extractall(dest)
  return dest


def _walk_files(root):
  out = []
  for dirpath, dirnames, files in os.walk(root):
    dirnames.sort()
    for fn in sorted(files):
      out.append(os.path.relpath(os.path.join(dirpath, fn), root))
      if len(out) >= _MAX_WALK_FILES:
        return out
  return out


def _find_ready_pair(root, rel_files):
  """(netlist, plc) absolute pair in one folder, or (None, None)."""
  by_dir = {}
  for rel in rel_files:
    by_dir.setdefault(os.path.dirname(rel), []).append(rel)
  for folder in sorted(by_dir):
    names = by_dir[folder]
    nets = [n for n in names if n.lower().endswith('.pb.txt')]
    plcs = [n for n in names if n.lower().endswith('.plc')]
    if not nets or not plcs:
      continue
    net = next((n for n in nets
                if os.path.basename(n) == 'netlist.pb.txt'), nets[0])
    plc = next((n for n in plcs
                if os.path.basename(n) == 'initial.plc'), plcs[0])
    return (os.path.join(root, net), os.path.join(root, plc))
  return None, None


def _base(path, root, kind):
  return {'path': path, 'root': root, 'kind': kind, 'ready': False,
          'needs_synth': False, 'netlist': '', 'plc': '',
          'rtl_files': [], 'tops': [], 'suggested_top': '', 'headers': [],
          'lef_files': [], 'def_files': [], 'liberty_files': [],
          'found': [], 'notes': [], 'missing': []}


def discover_design(path):
  """Inspects `path` (folder, single file, or .zip) for floorplanning.

  Returns a dict: kind ('ready_netlist' | 'rtl_project' | 'rtl_files' |
  'unknown'), ready, needs_synth, netlist/plc (ready pair), rtl_files
  (relative), tops, suggested_top, lef/def/liberty file lists, and
  human-readable found/notes/missing lists for the UI checklist.

  Raises:
    ValueError: the path does not exist.
  """
  path = os.path.abspath(os.path.expanduser(path or ''))
  if not os.path.exists(path):
    raise ValueError(f'path does not exist: {path}')
  if os.path.isfile(path) and path.lower().endswith('.zip'):
    path = ensure_unpacked(
        path, os.path.join(os.path.dirname(path),
                           os.path.splitext(os.path.basename(path))[0]
                           + '_unpacked'))
  if os.path.isfile(path):
    return _discover_file(path)
  return _discover_folder(path)


def _discover_file(path):
  root = os.path.dirname(path)
  low = path.lower()
  d = _base(path, root, 'unknown')
  if low.endswith('.pb.txt'):
    d['netlist'] = path
    sibs = sorted(f for f in os.listdir(root) if f.lower().endswith('.plc'))
    if sibs:
      pick = 'initial.plc' if 'initial.plc' in sibs else sibs[0]
      d.update(kind='ready_netlist', ready=True,
               plc=os.path.join(root, pick))
      d['found'].append(f'Ready netlist pair: {os.path.basename(path)} '
                        f'+ {pick} — no synthesis needed.')
    else:
      d['missing'].append('initial placement (.plc) next to the netlist')
      d['notes'].append('Netlist found; pick its .plc to place directly, '
                        'or import the RTL project instead.')
    return d
  if low.endswith('.plc'):
    sibs = sorted(f for f in os.listdir(root)
                  if f.lower().endswith('.pb.txt'))
    if sibs:
      d.update(kind='ready_netlist', ready=True, plc=path,
               netlist=os.path.join(root, sibs[0]))
      d['found'].append('Ready netlist pair found — no synthesis needed.')
    else:
      d['missing'].append('netlist proto (.pb.txt) next to the .plc')
    return d
  if low.endswith(_RTL_EXTS):
    scan = project_lib.scan_folder(path)
    d.update(kind='rtl_files', needs_synth=True, root=scan['root'],
             rtl_files=[os.path.basename(path)], tops=scan['tops'],
             suggested_top=scan['suggested_top'])
    d['found'].append(f"RTL file {os.path.basename(path)}; top module "
                      f"auto-picked: {scan['suggested_top']}.")
    d['notes'].append('Synthesis with yosys runs on Import; sibling '
                      'files in the same folder are scanned for the '
                      'instantiation closure.')
    return d
  d['missing'].append('RTL (.v/.sv) or a netlist pair (.pb.txt + .plc)')
  d['notes'].append(f'{os.path.basename(path)} alone cannot be placed; '
                    'select the design folder instead.')
  return d


def _discover_folder(root):
  d = _base(root, root, 'unknown')
  rel_files = _walk_files(root)
  d['lef_files'] = [r for r in rel_files if r.lower().endswith('.lef')]
  d['def_files'] = [r for r in rel_files if r.lower().endswith('.def')]
  d['liberty_files'] = [r for r in rel_files
                        if r.lower().endswith(('.lib', '.lib.gz'))]
  netlist, plc = _find_ready_pair(root, rel_files)
  try:
    scan = project_lib.scan_folder(root)
  except ValueError:
    scan = None
  if netlist:
    d.update(kind='ready_netlist', ready=True, netlist=netlist, plc=plc)
    d['found'].append('Ready netlist pair found — floorplanning can '
                      'start immediately, no synthesis needed.')
    if scan:
      d['notes'].append('RTL is also present; the ready netlist pair '
                        'wins. Delete/ignore it to re-synthesize.')
  elif scan:
    all_rtl = scan['verilog'] + scan['systemverilog']
    texts = {}
    for rel in all_rtl:
      try:
        with open(os.path.join(root, rel), errors='replace') as f:
          texts[rel] = f.read()
      except OSError:
        continue
    sim_only = project_lib.sim_only_modules(texts)
    non_sim_tops = [t for t in scan['tops'] if t not in sim_only]
    suggested = scan['suggested_top']
    if suggested in sim_only:
      defined = set(scan['tops'])
      inst = set()
      for rel, t in texts.items():
        if re.search(r'^\s*module\s+' + re.escape(suggested) + r'\b',
                     t, re.MULTILINE):
          inst |= {m.group(1)
                   for m in project_lib._INST_RE.finditer(t)
                   if m.group(1) not in project_lib._KEYWORDS}
      dut = sorted(m for m in inst
                   if m in defined and m not in sim_only)
      undefined = sorted(m for m in inst
                         if m not in defined and m != suggested)
      if dut:
        d['notes'].append(
            f'Auto-picked top {suggested} is a simulation test '
            f'driver; using its design {dut[0]} instead — verify '
            'in the Top field before Import.')
        suggested = dut[0]
      elif undefined or not non_sim_tops:
        missing_dut = (f' (the driver instantiates '
                       f'{undefined[0]}, which is not in this '
                       'folder)' if undefined else '')
        d['missing'].append(
            'synthesizable RTL — the Verilog found is simulation '
            'support code (test drivers, sim models) and the '
            f'design itself is generated elsewhere{missing_dut}')
        d['notes'].append(
            'This folder holds no generated RTL. For Chisel/Scala '
            'projects (e.g. rocket-chip), first generate Verilog '
            'from the generator, then import the generated .v '
            'file or its folder here.')
        return d
      else:
        d['notes'].append(
            f'Auto-picked top {suggested} is a simulation test '
            f'driver; using {non_sim_tops[0]} instead — verify '
            'in the Top field before Import.')
        suggested = non_sim_tops[0]
    d.update(kind='rtl_project', needs_synth=True,
             rtl_files=project_lib.default_files(scan),
             tops=scan['tops'], suggested_top=suggested,
             headers=scan['headers'])
    d['found'].append(
        f"RTL project: {len(scan['verilog'])} Verilog + "
        f"{len(scan['systemverilog'])} SystemVerilog file(s); "
        f"top auto-picked: {suggested}.")
    if scan['testbenches']:
      d['notes'].append(
          f"Testbench files excluded automatically: "
          f"{len(scan['testbenches'])}.")
    d['notes'].append('On Import, files unreachable from the top are '
                      'pruned so unrelated variants cannot break yosys.')
  else:
    d['missing'].append('RTL sources (.v/.sv) or netlist.pb.txt + .plc')
  for label, files in (('LEF', d['lef_files']), ('DEF', d['def_files']),
                       ('Liberty', d['liberty_files'])):
    if files:
      d['found'].append(f'{label}: {len(files)} file(s) found '
                        f'(e.g. {files[0]}).')
  return d


def import_plan(d):
  """Ordered human-readable steps Import will run for discovery `d`."""
  if d['kind'] == 'ready_netlist':
    return ['Use the ready netlist + placement pair (no synthesis).',
            'Register the design and enable Place.',
            'Run the AlphaChip-style sequential placer with live '
            'visualization.']
  if d['kind'] in ('rtl_project', 'rtl_files'):
    return [f"Scan RTL and prune to top '{d['suggested_top']}'.",
            'Synthesize with yosys (sky130 liberty from Settings).',
            'Convert the netlist to floorplan format (macros + '
            'standard-cell clusters).',
            'Run baseline episodes; the best placement is kept and '
            'can be downloaded as .plc / DEF.']
  return ['Nothing importable was found — see the missing list, then '
          'select the folder that holds the RTL or netlist pair.']
