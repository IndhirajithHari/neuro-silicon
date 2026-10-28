"""NeuroSilicon console: floorplan, import, validate, and report.

Run from the repo root:

  source .venv/bin/activate
  streamlit run neurosilicon/app.py

Pages (sidebar): Dashboard, Run floorplan (with project import),
Results & downloads, Thermal lab, GNN simulation, AI assistant,
Settings. All compute is real: placements come from the
proxy-cost engine, validation from OpenROAD, advice from the configured
LLM (or honest fallback text when none is configured).
"""
import io
import os
import sys

# Plain `streamlit run neurosilicon/app.py` puts the script's own folder on
# sys.path, hiding the top-level package; ensure the repo root is importable.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AVAILABLE_NETLISTS = {
    'toy (2 macros)': (
        'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt',
        'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc'),
    'ariane (133 macros)': (
        'circuit_training/environment/test_data/ariane/netlist.pb.txt',
        'circuit_training/environment/test_data/ariane/initial.plc'),
    'servant SoC (18 macros, sky130)': (
        'logs/servant_proto/netlist.pb.txt',
        'logs/servant_proto/initial.plc'),
    'picorv32 (2 macros + clusters, sky130)': (
        'logs/picorv32_proto/netlist.pb.txt',
        'logs/picorv32_proto/initial.plc'),
    'picosoc (6 macros + clusters, sky130)': (
        'logs/picosoc_proto/netlist.pb.txt',
        'logs/picosoc_proto/initial.plc'),
    'ultraembedded riscv_core (10 macros + clusters, sky130)': (
        'logs/ue_proto/netlist.pb.txt',
        'logs/ue_proto/initial.plc'),
}

# Built-ins with consistent (json, netlist.v) pairs for flat DEF.
# validation=False where yosys auto-names differ between the proto JSON
# and any re-synthesized Verilog (picorv32 v1 has no autoname JSON).
BUILTIN_SYNTH = {
    'servant SoC (18 macros, sky130)': {
        'json': 'logs/servant_build/servant_fmt.json',
        'net': 'logs/servant_build/servant_net.v', 'top': 'servant',
        'validation': True},
    'picosoc (6 macros + clusters, sky130)': {
        'json': 'logs/picosoc_build/picosoc.json',
        'net': 'logs/picosoc_build/picosoc_net.v', 'top': 'picosoc',
        'validation': True},
    'ultraembedded riscv_core (10 macros + clusters, sky130)': {
        'json': 'logs/ue_build/ue.json', 'net': 'logs/ue_build/ue_net.v',
        'top': 'riscv_core', 'validation': True},
}

# Public API the console needs. Streamlit's auto-reloader sometimes keeps
# pre-update modules in memory after a code change (new app.py running
# against an old render.py); missing_api() detects that stale-server
# state so main() can show remediation instead of a traceback.
REQUIRED_API = {
    'neurosilicon.env': ['ChipPlacementEnv'],
    'neurosilicon.evaluate': ['rollout_first_legal', 'rollout_policy',
                              'rollout_random', 'run_baselines'],
    'neurosilicon.export_def': ['export_def'],
    'neurosilicon.finetune': ['run_cd'],
    'neurosilicon.flatten_def': ['flatten'],
    'neurosilicon.gnn_viz': ['simulate'],
    'neurosilicon.llm': ['LLMError', 'advise', 'chat',
                         'parse_constraints', 'test_connection'],
    'neurosilicon.model': ['CircuitPolicy'],
    'neurosilicon.priors': ['broadcast_penalty', 'chat_spec_to_mask_fn',
                            'chat_to_mask_fn', 'make_keepout_fn',
                            'make_perimeter_fn', 'thermal_corner',
                            'thermal_overlap_penalty', 'top_congestion'],
    'neurosilicon.project': ['add_to_registry', 'default_files',
                             'import_files', 'import_project',
                             'load_registry', 'parse_params',
                             'register_netlist', 'repo_root',
                             'resolve_pdk', 'resolve_tool',
                             'scan_folder'],
    'neurosilicon.render': ['cost_breakdown_figure', 'estimate_power',
                            'figure_png', 'history_figure',
                            'plot_congestion', 'plot_floorplan',
                            'plot_sim_step', 'plot_thermal'],
    'neurosilicon.settings': ['SECTIONS', 'SETTINGS_PATH', 'defaults',
                              'load', 'redacted', 'save'],
    'neurosilicon.setup': ['create_plc'],
    'neurosilicon.synth_to_proto': ['Converter'],
    'neurosilicon.ui': ['inject_css', 'page_footer', 'page_header',
                        'pills_row', 'version', 'workflow_steps'],
}


def browse_folder():
  """Native folder picker. Returns (path_or_None, picker_available).

  Tries zenity, then tkinter; both need the Streamlit server to run on
  the local machine with a display. (None, True) means the user
  cancelled; (None, False) means no picker backend exists.
  """
  import shutil
  import subprocess
  if shutil.which('zenity'):
    try:
      proc = subprocess.run(
          ['zenity', '--file-selection', '--directory',
           '--title=Select chip project folder'],
          capture_output=True, text=True, timeout=300)
      if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip(), True
      return None, True
    except (OSError, subprocess.TimeoutExpired):
      pass
  try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
      root.attributes('-topmost', True)
    except Exception:
      pass
    path = filedialog.askdirectory(title='Select chip project folder')
    root.destroy()
    return (path or None), True
  except Exception:
    return None, False


def missing_api():
  """Lists 'module.attr' entries absent from the imported modules."""
  import importlib
  missing = []
  for modname, attrs in REQUIRED_API.items():
    try:
      mod = importlib.import_module(modname)
    except ImportError:
      missing.extend(f'{modname} (unimportable)' for _ in attrs)
      continue
    for attr in attrs:
      if not hasattr(mod, attr):
        missing.append(f'{modname}.{attr}')
  return missing


def reload_stale(stale):
  """Hot-reloads stale modules in-process. Returns reloaded module names.

  Safe here because every console access goes through module attributes
  (render_lib.X), never from-imported names, and nothing is instantiated
  before the guard runs.
  """
  import importlib
  done = []
  for entry in stale:
    if entry.endswith('(unimportable)'):
      modname = entry.split(' ')[0]
    else:
      modname = entry.rsplit('.', 1)[0]
    if modname in done:
      continue
    try:
      if modname in sys.modules:
        importlib.reload(sys.modules[modname])
      else:
        importlib.import_module(modname)
    except Exception:
      continue
    done.append(modname)
  return done


# Verified-good policy checkpoint per design (greedy-feasible in testing).
# Servant uses its specialist; the joint4 policy dead-ends greedy there.
CHECKPOINTS = {
    'servant SoC (18 macros, sky130)': 'logs/servant_train_long/best.pt',
    'picorv32 (2 macros + clusters, sky130)': 'logs/joint4_train/best.pt',
    'picosoc (6 macros + clusters, sky130)': 'logs/joint4_train/best.pt',
    'ultraembedded riscv_core (10 macros + clusters, sky130)':
        'logs/joint4_train/best.pt',
}

MODE_HELP = {
    'heuristic': 'Baselines only (random / first_legal + CD). No model '
                 'or API key needed.',
    'neurosilicon': 'Our trained PPO policy from a checkpoint file. Falls '
                    'back to heuristic with a warning when missing.',
    'llm_assisted': 'Heuristic placement plus LLM review and next-step '
                    'advice. Needs an AI provider in Settings.',
}


def build_mask_fn(kind, cols, rows, canvas_w, canvas_h, chat_text='',
                  llm_settings=None):
  """Compiles a mask selection to (fn, explanation).

  Chat constraints use the LLM parser when `llm_settings` enables a
  provider, else the rule-based parser (the LLM path falls back to
  rules on any outage and says so).
  """
  from neurosilicon import priors as priors_lib
  if kind == 'Center keep-out channel':
    cx0 = canvas_w * 0.4
    cx1 = canvas_w * 0.6
    return (priors_lib.make_keepout_fn(cols, rows, canvas_w, canvas_h,
                                       [(cx0, 0.0, cx1, canvas_h)]),
            'Keep-out: center 20% vertical channel.')
  if kind == 'Perimeter band only':
    return (priors_lib.make_perimeter_fn(cols, rows, canvas_w, canvas_h),
            'Restrict to perimeter band.')
  if kind == 'Chat constraint':
    if llm_settings and llm_settings.get('provider', 'off') != 'off':
      from neurosilicon import llm as llm_lib
      mode, rects, explanation = llm_lib.parse_constraints(
          chat_text, canvas_w, canvas_h, llm_settings)
      return priors_lib.chat_spec_to_mask_fn(
          mode, rects, cols, rows, canvas_w, canvas_h, explanation)
    return priors_lib.chat_to_mask_fn(chat_text, cols, rows, canvas_w,
                                      canvas_h)
  return None, 'No constraint.'


def all_netlists():
  """Built-ins plus imported registry entries with existing files.

  All paths are absolutized against the repo root so the console
  works no matter which directory Streamlit was launched from.
  """
  from neurosilicon import project as project_lib
  root = project_lib.repo_root()
  def abs2(p):
    return p if os.path.isabs(p) else os.path.join(root, p)
  out = {k: (abs2(nl), abs2(plc))
         for k, (nl, plc) in AVAILABLE_NETLISTS.items()}
  for name, entry in project_lib.load_registry().items():
    nl, plc = entry.get('netlist', ''), entry.get('plc', '')
    if nl and plc:
      out[f'imported: {name}'] = (abs2(nl), abs2(plc))
  return {k: v for k, v in out.items()
          if os.path.isfile(v[0]) and os.path.isfile(v[1])}


def synth_info_for(choice):
  """Flat-DEF/validation metadata for a netlist choice, or None."""
  from neurosilicon import project as project_lib
  root = project_lib.repo_root()
  def abs1(p):
    return p if not p or os.path.isabs(p) else os.path.join(root, p)
  if choice in BUILTIN_SYNTH:
    info = BUILTIN_SYNTH[choice]
    if os.path.isfile(abs1(info['json'])):
      return {**info, 'json': abs1(info['json']),
              'net': abs1(info['net'])}
    return None
  if choice.startswith('imported: '):
    entry = project_lib.load_registry().get(choice[len('imported: '):],
                                            {})
    if entry.get('json') and os.path.isfile(abs1(entry['json'])):
      net = abs1(entry.get('net', ''))
      return {'json': abs1(entry['json']), 'net': net,
              'top': entry.get('top', ''),
              'validation': bool(net and os.path.isfile(net))}
  return None


def placement_details(plc):
  """Full chip snapshot: macro rows, counts, canvas (no Streamlit)."""
  macros = []
  n_soft = 0
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      n_soft += 1
      continue
    x, y = plc.get_node_location(idx)
    w, h = plc.get_node_width_height(idx)
    try:
      orient = plc.modules_w_pins[idx].get_orientation()
    except (AttributeError, IndexError):
      orient = '?'
    macros.append({'macro': plc.get_node_name(idx),
                   'x': round(x, 2), 'y': round(y, 2),
                   'w': round(w, 2), 'h': round(h, 2),
                   'area': round(w * h, 1), 'orient': orient or '?'})
  cw, ch = plc.get_canvas_width_height()
  cols, rows = plc.get_grid_num_columns_rows()
  macro_area = sum(m['area'] for m in macros)
  return {'macros': macros,
          'counts': {'hard_macros': len(macros), 'soft_macros': n_soft,
                     'ports': len(plc.port_indices),
                     'nets': len(plc.nets)},
          'canvas': (round(cw, 2), round(ch, 2)),
          'grid': (cols, rows),
          'utilization': round(macro_area / (cw * ch), 4)
                         if cw * ch > 0 else 0.0}


def run_episodes(netlist, init_plc, method, episodes, fd, mask_fn,
                 extra_costs, ckpt_path, finetune_epochs, seed,
                 broadcast_kwargs=None, on_episode=None):
  """Runs episodes, returns (rows, best_plc, best_cost, notes).

  Pure compute (no Streamlit): methods random / first_legal / policy.
  `policy` falls back to first_legal with a note when the checkpoint
  is missing. CD finetune (with broadcast weights) applies per episode.
  `on_episode(ep, row, plc_or_None, seconds)` fires after each episode
  for live UIs (plc is None when infeasible).
  """
  import time

  import torch

  from neurosilicon import env as env_lib
  from neurosilicon import evaluate as evaluate_lib
  from neurosilicon import finetune as finetune_lib
  from neurosilicon import model as model_lib

  notes = []
  policy = None
  if method == 'policy':
    if ckpt_path and os.path.isfile(ckpt_path):
      policy = model_lib.CircuitPolicy()
      policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
      policy.eval()
    else:
      notes.append(f'Checkpoint missing ({ckpt_path}); fell back to '
                   'first_legal heuristic.')
      method = 'first_legal'

  def make_env():
    return env_lib.ChipPlacementEnv(
        netlist, init_placement=init_plc, fd_steps=(fd, fd, fd),
        extra_mask_fn=mask_fn, extra_costs=extra_costs or None)

  rows, best_plc, best_cost = [], None, float('inf')
  for ep in range(episodes):
    t0 = time.time()
    env = make_env()
    if method == 'random':
      _, info = evaluate_lib.rollout_random(env, seed=seed + ep)
    elif method == 'policy':
      _, info = evaluate_lib.rollout_policy(env, policy, greedy=False,
                                            seed=seed + ep)
    else:
      _, info = evaluate_lib.rollout_first_legal(env)
    if finetune_epochs > 0 and not info.get('infeasible', True):
      cost, parts = finetune_lib.run_cd(
          env.plc, epochs=finetune_epochs, **(broadcast_kwargs or {}))
      info = {'infeasible': False, 'proxy_cost': cost, **parts}
    secs = round(time.time() - t0, 1)
    top10 = -1.0
    if not info.get('infeasible', True):
      from neurosilicon import priors as priors_lib
      # getattr fallback: a stale server missing top_congestion must not
      # kill a long run — the metric degrades to -1 (unknown) instead.
      fn = getattr(priors_lib, 'top_congestion', None)
      if fn is not None:
        try:
          top10 = round(fn(env.plc), 4)
        except (ValueError, TypeError, AttributeError):
          top10 = -1.0
    row = {'method': method, 'episode': ep,
           'proxy_cost': round(info.get('proxy_cost', -1), 4),
           'wirelength': round(info.get('wirelength', -1), 4),
           'congestion': round(info.get('congestion', -1), 4),
           'density': round(info.get('density', -1), 4),
           'top10cong': top10,
           'seconds': secs}
    rows.append(row)
    cost = info.get('proxy_cost', -1)
    feasible_plc = None
    if 0 < cost < best_cost:
      best_cost, best_plc = cost, env.plc
    if not info.get('infeasible', True):
      feasible_plc = env.plc
    if on_episode is not None:
      on_episode(ep, row, feasible_plc, secs)
  return rows, best_plc, best_cost, notes


def _get_plc():
  from neurosilicon.thirdparty.tilos.Plc_client import (
      placement_util_os as placement_util)
  return placement_util


def _plc_to_text(plc):
  import tempfile
  placement_util = _get_plc()
  with tempfile.NamedTemporaryFile('w', suffix='.plc',
                                   delete=False) as f:
    path = f.name
  placement_util.save_placement(plc, path)
  with open(path) as f:
    return f.read()


GUI_RUNS_DIRNAME = os.path.join('logs', 'gui_runs', 'runs')


def save_gui_run(run_id, run, plc):
  """Persists a completed run to disk. Returns the run directory.

  Writes run.json (metadata + rows, JSON-safe), best.plc (placement),
  and blocked.npy (constraint mask, when present). Pure function of
  its arguments — no Streamlit — so switching tabs, refreshing the
  browser, or restarting the server never loses a finished run.
  """
  import json
  from neurosilicon import project as project_lib
  run_dir = os.path.join(project_lib.repo_root(), GUI_RUNS_DIRNAME,
                         run_id)
  os.makedirs(run_dir, exist_ok=True)
  meta = {k: v for k, v in run.items() if k != 'blocked'}
  with open(os.path.join(run_dir, 'run.json'), 'w') as f:
    json.dump(meta, f, indent=2)
  _get_plc().save_placement(plc, os.path.join(run_dir, 'best.plc'))
  if run.get('blocked') is not None:
    import numpy as np
    np.save(os.path.join(run_dir, 'blocked.npy'),
            np.asarray(run['blocked'], dtype=bool))
  return run_dir


def patch_gui_run(run_id, **patch):
  """Merges keys (advice, validated) into a saved run's run.json."""
  import json
  from neurosilicon import project as project_lib
  path = os.path.join(project_lib.repo_root(), GUI_RUNS_DIRNAME,
                      run_id, 'run.json')
  if not os.path.isfile(path):
    return False
  with open(path) as f:
    meta = json.load(f)
  meta.update(patch)
  with open(path, 'w') as f:
    json.dump(meta, f, indent=2)
  return True


def list_gui_runs():
  """Saved-run metadata, newest first. Each has '_dir' set."""
  import glob
  import json
  from neurosilicon import project as project_lib
  out = []
  pattern = os.path.join(project_lib.repo_root(), GUI_RUNS_DIRNAME,
                         '*', 'run.json')
  for path in sorted(glob.glob(pattern), reverse=True):
    try:
      with open(path) as f:
        meta = json.load(f)
    except (OSError, ValueError):
      continue
    meta['_dir'] = os.path.dirname(path)
    out.append(meta)
  return out


def load_gui_run(run_dir):
  """Restores (run_dict, plc) from a save_gui_run directory."""
  import json
  from neurosilicon import project as project_lib
  from neurosilicon import setup as setup_lib
  with open(os.path.join(run_dir, 'run.json')) as f:
    run = json.load(f)
  netlist = run.get('netlist', '')
  if netlist and not os.path.isabs(netlist):
    netlist = os.path.join(project_lib.repo_root(), netlist)
    run['netlist'] = netlist
  plc = setup_lib.create_plc(
      netlist, init_placement=os.path.join(run_dir, 'best.plc'))
  blocked_path = os.path.join(run_dir, 'blocked.npy')
  if os.path.isfile(blocked_path):
    import numpy as np
    run['blocked'] = np.load(blocked_path)
  else:
    run['blocked'] = None
  return run, plc


def _current_run_plc(st):
  """Session (run, plc), auto-restoring the newest saved run if empty.

  Returns (None, None) only when no run exists anywhere. Corrupt
  save dirs are skipped, never fatal.
  """
  run = st.session_state.get('last_run')
  plc = st.session_state.get('last_plc')
  if run and plc is not None:
    return run, plc
  for meta in list_gui_runs():
    try:
      run, plc = load_gui_run(meta['_dir'])
    except Exception:
      continue
    st.session_state['last_run'] = run
    st.session_state['last_plc'] = plc
    st.caption(f"Restored run {run.get('run_id', '?')} from disk — "
               'results survive tab switches, refreshes, and restarts.')
    return run, plc
  return None, None


def page_dashboard(st, settings):
  from neurosilicon import project as project_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Overview', 'Dashboard',
                     'System status, designs, and recent runs at a glance.')
  pills = []
  for label, path, need_exec in (
      ('yosys', settings['pdk'].get('yosys', 'tools/yosys'), True),
      ('openroad', settings['pdk'].get('openroad', 'tools/openroad'),
       True),
      ('validate_route.tcl', 'tools/validate_route.tcl', False)):
    full = project_lib.resolve_tool(path)
    ok = os.path.isfile(full) and (
        os.access(full, os.X_OK) if need_exec else True)
    pills.append((label, 'ok' if ok else 'bad'))
  try:
    pdk = project_lib.resolve_pdk(settings['pdk'])
    for key in ('lef', 'liberty', 'techlef'):
      pills.append(
          (key, 'ok' if os.path.isfile(pdk[key]) else 'bad'))
  except ValueError:
    pills += [('lef', 'bad'), ('liberty', 'bad'), ('techlef', 'bad')]
  provider = settings['llm'].get('provider', 'off')
  pills.append((f'LLM: {provider}',
                'ok' if provider != 'off' else 'muted'))
  st.subheader('Toolchain')
  st.markdown(ui_lib.pills_row(pills), unsafe_allow_html=True)
  st.caption('Red pills name a missing tool — fix the path in '
             'Settings > PDK & Toolchain.')

  st.subheader('Designs')
  nets = all_netlists()
  rows = []
  for name in nets:
    info = synth_info_for(name)
    ckpt = CHECKPOINTS.get(name, '')
    rows.append({
        'design': name,
        'source': ('imported' if name.startswith('imported: ')
                   else 'built-in'),
        'flat DEF': 'yes' if info and info.get('json') else '—',
        'route check': ('yes' if info and info.get('validation')
                        else '—'),
        'checkpoint': ('yes' if ckpt and os.path.isfile(ckpt)
                       else '—')})
  st.table(rows if rows else [{'design': 'none — import a project'}])

  st.subheader('Recent runs')
  runs = list_gui_runs()[:8]
  if runs:
    st.table([{'run': m.get('run_id', '?'),
               'design': (m.get('choice', '?') or '?')[:40],
               'best': m.get('best_cost', '?'),
               'started': m.get('started_at', '?')} for m in runs])
    st.caption('Open Results & downloads to inspect any run.')
  else:
    st.caption('No finished runs yet — place one from Run floorplan.')

  st.subheader('Quick start')
  st.markdown('1. **Run floorplan** — pick a design (or import RTL at '
              'the top of the page) and press Run.\n'
              '2. **Results & downloads** — inspect figures, the macro '
              'Inspector, and files.\n'
              '3. **Validation tab** — global-route the best placement '
              'with OpenROAD.\n'
              '4. **Report bundle** — download the ZIP with PNGs, DEFs, '
              'and settings.')


def page_run(st, settings):
  from neurosilicon import render as render_lib
  from neurosilicon import setup as setup_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Place', 'Run floorplan',
                     'Place macros with the proxy-cost engine. Every '
                     'number below comes from a real rollout — no mock '
                     'data.')
  ui_lib.workflow_steps(st, 0)
  _import_section(st, settings)
  st.divider()

  nets = all_netlists()
  if not nets:
    st.error('No netlists found. Import a chip project above first.')
    return
  left, right = st.columns([1, 2])
  with left:
    choice = st.selectbox(
        'Netlist', list(nets),
        help='Built-in training designs plus your imported projects.')
    if choice.startswith('ariane'):
      st.warning('Ariane has 133 macros (~9 min/episode on CPU). Use 1 '
                 'episode — progress streams below as each lands.')
    mode = st.selectbox(
        'Placement mode', ['heuristic', 'neurosilicon', 'llm_assisted'],
        index=['heuristic', 'neurosilicon',
               'llm_assisted'].index(settings['run']['mode']),
        help='How to place. Details per mode shown below.')
    st.caption(MODE_HELP[mode])
    method = 'first_legal'
    if mode == 'heuristic':
      method = st.selectbox(
          'Baseline', ['first_legal', 'random'],
          help='first_legal: deterministic reference. random: stochastic, '
               'sometimes better, sometimes infeasible.')
    ckpt = CHECKPOINTS.get(choice, settings['model']['checkpoint'])
    if mode == 'neurosilicon':
      ckpt = st.text_input('Checkpoint (.pt)', ckpt,
                           help='Trained policy weights, pre-filled with '
                                'the verified-good checkpoint for this '
                                'design. Missing file falls back to '
                                'heuristic with a warning.')
    episodes = st.number_input(
        'Episodes', 1, 20, settings['run']['episodes'],
        help='Independent attempts; the best placement is kept.')
    fd = st.number_input(
        'FD steps per stage', 1, 100, settings['run']['fd_steps'],
        help='Standard-cell relaxation at episode end. Higher is '
             'smoother but slower.')
    cd_epochs = st.number_input(
        'CD finetune epochs', 0, 10, 2,
        help='Coordinate-descent passes over macro positions after '
             'each episode. 0 disables.')
  with right:
    mask_kind = st.selectbox(
        'Constraint mask',
        ['None', 'Center keep-out channel', 'Perimeter band only',
         'Chat constraint'],
        help='Hard placement mask. Chat constraints parse via LLM '
             'when a provider is set, else rule-based.')
    chat_text = st.text_input(
        'Chat constraint', 'Keep a 20% routing channel open in the center',
        help='Plain-English constraint, e.g. "keepout on the left edge" '
             'or "place macros along the bottom".')
    eng = settings['engine']
    st.caption(f"Weights — wl {eng['wirelength_weight']}, cong "
               f"{eng['congestion_weight']}, dens "
               f"{eng['density_weight']}, broadcast "
               f"{eng['broadcast_weight']}, thermal {eng['thermal_weight']} "
               f"(edit in Settings).")
    power_file = st.file_uploader(
        'Macro power JSON (optional)',
        help='{"macro": watts} for the thermal map and penalty. '
             'Without it, power is estimated from macro area.')
    go = st.button('Run floorplan', type='primary',
                    help='Runs the configured episodes now. Results land '
                         'on Results & downloads.')

  if not go:
    st.info('Configure the run, then press **Run floorplan**.')
    return

  import datetime
  run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  started_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  st.caption(f'Run ID {run_id} · started {started_at} · seed '
             f"{settings['run']['seed']} · first_legal is deterministic "
             '(identical every run by design); random/policy vary by seed.')
  live_rows = st.empty()
  live_best = st.empty()
  live_state = {'rows': [], 'best': float('inf'), 'blocked': None}

  def _on_episode(ep, row, plc, secs):
    import matplotlib.pyplot as plt
    live_state['rows'].append(row)
    live_rows.table(live_state['rows'])
    cost = row['proxy_cost']
    if plc is not None and 0 < cost < live_state['best']:
      live_state['best'] = cost
      fig = render_lib.plot_floorplan(plc,
                                      blocked=live_state['blocked'])
      live_best.pyplot(fig)
      plt.close(fig)

  netlist, init_plc = nets[choice]
  with st.status('Placing macros...', expanded=True) as status:
    st.write(f'Loading {choice}...')
    template = setup_lib.create_plc(netlist, init_placement=init_plc)
    cols, rows = template.get_grid_num_columns_rows()
    canvas_w, canvas_h = template.get_canvas_width_height()
    mask_fn, explanation = build_mask_fn(
        mask_kind, cols, rows, canvas_w, canvas_h, chat_text,
        settings['llm'])
    st.write(f'Mask: {explanation}')
    if mask_fn is not None:
      import numpy as np
      live_state['blocked'] = (
          mask_fn(0, np.ones(rows * cols, dtype=np.int32)).reshape(
              rows, cols) == 0)
    power = None
    if power_file is not None:
      import json
      try:
        power = {k: float(v)
                 for k, v in json.load(power_file).items()}
        st.write(f'Loaded power for {len(power)} macros.')
      except (ValueError, AttributeError) as e:
        st.warning(f'Ignoring power file: {e}')
    if power is None and eng['thermal_weight'] > 0:
      power = render_lib.estimate_power(template)
      st.write('No power file: using area-proportional estimate.')
    extra_costs = []
    if eng['thermal_weight'] > 0 and power:
      from neurosilicon import priors as priors_lib
      extra_costs.append(('thermal', eng['thermal_weight'],
                          lambda plc, p=power: priors_lib.thermal_overlap_penalty(plc, p)))
    if eng['broadcast_weight'] > 0:
      from neurosilicon import priors as priors_lib
      th = eng['broadcast_threshold']
      extra_costs.append(('broadcast', eng['broadcast_weight'],
                          lambda plc: priors_lib.broadcast_penalty(
                              plc, fanout_threshold=th)))
    run_method = 'policy' if mode == 'neurosilicon' else method
    st.write(f'Running {episodes} {run_method} episode(s)...')
    rows_out, best_plc, best_cost, notes = run_episodes(
        netlist, init_plc, run_method, episodes, fd, mask_fn,
        extra_costs, ckpt, cd_epochs, settings['run']['seed'],
        {'broadcast_weight': eng['broadcast_weight'],
         'broadcast_threshold': eng['broadcast_threshold']},
        on_episode=_on_episode)
    for n in notes:
      st.warning(n)
    done_label = ('Run complete: best '
                  f'{best_cost:.4f}.' if best_plc is not None else
                  'Run complete: all infeasible.')
    status.update(label=done_label, state='complete')

  if best_plc is None:
    st.error('All episodes were infeasible. Loosen the constraint mask, '
             'or switch mode to heuristic / first_legal.')
    return
  st.session_state['last_run'] = {
      'choice': choice, 'netlist': netlist, 'rows': rows_out,
      'best_cost': best_cost, 'power': power,
      'advice': None, 'validated': None, 'run_id': run_id,
      'started_at': started_at, 'mode': mode, 'method': run_method,
      'ckpt': ckpt, 'episodes': episodes, 'fd': fd,
      'cd_epochs': cd_epochs, 'seed': settings['run']['seed'],
      'mask': explanation, 'blocked': live_state['blocked'],
  }
  st.session_state['last_plc'] = best_plc
  if mode == 'llm_assisted':
    from neurosilicon import llm as llm_lib
    with st.spinner('LLM reviewing results...'):
      try:
        st.session_state['last_run']['advice'] = llm_lib.advise(
            rows_out, settings['llm'],
            context=f'Netlist: {choice}. Best proxy: {best_cost:.4f}.')
      except llm_lib.LLMError as e:
        st.session_state['last_run']['advice'] = (
            f'LLM advice unavailable: {e}\n\nResults above are unaffected — '
            'configure Settings > AI Providers to enable advice.')
  try:
    saved = save_gui_run(run_id, st.session_state['last_run'], best_plc)
    st.caption(f'Run saved to {saved} — it survives tab switches, '
               'refreshes, and server restarts.')
  except (OSError, ValueError) as e:
    st.warning(f'Could not save run to disk: {e} (session copy kept).')
  st.success(f'Best proxy cost: {best_cost:.4f} — see Results & downloads.')


def _import_section(st, settings):
  from neurosilicon import project as project_lib

  st.subheader('Import chip project')
  st.caption('Point at any folder with RTL, pick files and a top module, '
             'and run synthesis → proto → baselines. The result registers '
             'as a netlist in the picker below.')
  notice = st.session_state.pop('import_notice', '')
  if notice:
    st.success(notice)
  def _on_browse():
    path, available = browse_folder()
    if path:
      st.session_state['import_folder'] = path
    st.session_state['folder_available'] = available

  fcol1, fcol2 = st.columns([4, 1])
  with fcol1:
    st.text_input(
        'Project folder (absolute path)', key='import_folder',
        help='Any folder containing .v files (.sv attempted, may fail on '
             'the old yosys parser). Example: a picorv32 checkout. Type, '
             'paste, or use Browse for a native picker.')
  with fcol2:
    st.write('')
    st.button('Browse…', on_click=_on_browse,
              help='Opens a native folder picker on the machine running '
                   'Streamlit (needs zenity or tkinter plus a display).')
  if st.session_state.get('folder_available') is False:
    st.info('No native picker backend found (needs zenity or python '
            'tkinter plus a display); type or paste the folder path.')
  folder = st.session_state.get('import_folder', '')
  if not folder:
    st.info('Enter a folder path to scan it.')
    return
  try:
    scan = project_lib.scan_folder(folder)
  except ValueError as e:
    st.error(str(e))
    return
  st.write(f"Found {len(scan['verilog'])} Verilog, "
           f"{len(scan['systemverilog'])} SystemVerilog files, "
           f"{len(scan['tops'])} modules.")
  files = st.multiselect(
      'RTL files to synthesize', scan['verilog'] + scan['systemverilog'],
      default=project_lib.default_files(scan),
      help='Files passed to yosys read_verilog (in order). Testbench '
           'files are excluded by default.')
  _tops = scan['tops'] if scan['tops'] else ['(none found)']
  top = st.selectbox(
      'Top module', _tops,
      index=_tops.index(scan['suggested_top'])
      if scan.get('suggested_top') in _tops else 0,
      help='synth -top target. Pre-picked as the likely top (root '
           'module); change if wrong.')
  params = st.text_input(
      'Verilog parameters (optional)', '',
      help='chparam overrides, e.g. MEM_WORDS=128,ENABLE_DIV=1.')
  name = st.text_input(
      'Project name', os.path.basename(os.path.normpath(folder)) or 'chip',
      help='Display name in the netlist picker.')
  episodes = st.number_input(
      'Baseline episodes per method', 1, 5, 2,
      help='Random + first_legal attempts each; the best registers with '
           'the project.')
  go = st.button('Import and baseline', type='primary',
                 help='Synthesizes the RTL, converts to a floorplan '
                      'netlist, baselines it, and registers the project.')
  if not go:
    return
  if not files or not top or top == '(none found)':
    st.error('Pick at least one file and a top module.')
    return
  try:
    param_dict = project_lib.parse_params(params)
  except ValueError as e:
    st.error(str(e))
    return
  outdir = project_lib.resolve_tool(
      os.path.join(settings['run']['outdir'], 'imports', name))
  with st.status(f'Importing {name}...', expanded=True) as status:
    def progress(stage, message):
      st.write(f'[{stage}] {message}')
    try:
      summary = project_lib.import_project(
          folder, files, top, outdir, settings, params=param_dict,
          episodes=episodes, progress=progress)
    except (ValueError, RuntimeError) as e:
      status.update(label='Import failed.', state='error')
      st.error(str(e))
      return
    project_lib.add_to_registry(
        name, summary['proto']['netlist'], summary['proto']['plc'],
        extra={'json': os.path.join(outdir, 'build', 'design.json'),
               'net': os.path.join(outdir, 'build', 'design_net.v'),
               'top': top, 'outdir': outdir})
    status.update(label='Import complete.', state='complete')
  st.session_state['import_notice'] = (
      f"{summary['cells']} cells, best baseline proxy "
      f"{summary['best_proxy']:.4f}. Registered as 'imported: {name}'.")
  st.rerun()


def _figure_downloads(st, figs):
  import matplotlib.pyplot as plt
  from neurosilicon import render as render_lib
  cols = st.columns(len(figs))
  for (label, fname, fig), col in zip(figs, cols):
    col.caption(label)
    col.pyplot(fig)
    col.download_button(f'Download {label}', render_lib.figure_png(fig),
                        file_name=fname, mime='image/png',
                        help=f'Downloads the {label} figure above as PNG.')
    plt.close(fig)


def _best_plc_path():
  import tempfile
  placement_util = _get_plc()
  import streamlit as st
  path = os.path.join(tempfile.gettempdir(), 'neurosilicon_best.plc')
  placement_util.save_placement(st.session_state['last_plc'], path)
  return path


def page_results(st, settings):
  from neurosilicon import export_def as export_def_lib
  from neurosilicon import flatten_def as flatten_lib
  from neurosilicon import render as render_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Inspect', 'Results & downloads',
                     'Figures, macro inspector, files, routed validation, '
                     'and training curves for the selected run.')
  ui_lib.workflow_steps(st, 1)
  run, plc = _current_run_plc(st)
  if not run or plc is None:
    st.info('No run yet. Go to Run floorplan first.')
    _history_browser(st)
    return
  saved_runs = list_gui_runs()
  if len(saved_runs) > 1:
    labels = {m.get('run_id', os.path.basename(m['_dir'])): m
              for m in saved_runs}
    current = run.get('run_id', '')
    pick = st.selectbox(
        'Run', sorted(labels),
        index=sorted(labels).index(current) if current in labels else 0,
        format_func=lambda rid: (
            f"{rid} · {labels[rid].get('choice', '?')} · "
            f"best {labels[rid].get('best_cost', '?')}"),
        help='Every finished run is saved to disk — switch between them '
             'without losing anything.')
    if pick != current:
      run, plc = load_gui_run(labels[pick]['_dir'])
      st.session_state['last_run'] = run
      st.session_state['last_plc'] = plc
      st.rerun()
  rows = run['rows']
  power = run.get('power') or render_lib.estimate_power(plc)
  info = synth_info_for(run['choice'])
  details = placement_details(plc)
  tab_fig, tab_insp, tab_files, tab_valid, tab_curves = st.tabs(
      ['Figures', 'Inspector', 'Files', 'Validation', 'Curves'])
  with tab_fig:
    c1, c2, c3 = st.columns(3)
    c1.metric('Best proxy cost', f"{run['best_cost']:.4f}",
              help='Lowest wl + 0.5·cong + dens over episodes (plus any '
                   'thermal/broadcast extras).')
    c2.metric('Episodes', str(len(rows)),
              help='Completed episodes this run (infeasible ones are '
                   'excluded from best).')
    c3.metric('Netlist', run['choice'][:32],
              help='Design that was placed.')
    st.caption('Per-episode cost components (-1 marks infeasible).')
    st.table(rows)
    if run.get('advice'):
      st.subheader('LLM review')
      st.markdown(run['advice'])
    if run.get('power') is None:
      st.caption('Thermal uses area-proportional power estimates (upload '
                 'a power JSON on the Run page for measured values).')
    if run.get('blocked') is not None:
      st.caption('Red wash on the floorplan = constraint-blocked cells.')
    figs = [('floorplan', 'floorplan.png',
             render_lib.plot_floorplan(plc,
                                       blocked=run.get('blocked'))),
            ('congestion', 'congestion.png',
             render_lib.plot_congestion(plc)),
            ('thermal', 'thermal.png',
             render_lib.plot_thermal(plc, power)),
            ('cost breakdown', 'cost_breakdown.png',
             render_lib.cost_breakdown_figure(rows))]
    _figure_downloads(st, figs)
  with tab_insp:
    st.subheader('Run provenance')
    st.caption('Exactly what produced this placement — fresh every run.')
    st.json({k: run.get(k, '—')
             for k in ('run_id', 'started_at', 'mode', 'method', 'ckpt',
                       'episodes', 'fd', 'cd_epochs', 'seed', 'mask')})
    st.subheader('Chip contents')
    cc = details['counts']
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric('Hard macros', str(cc['hard_macros']),
              help='Movable blocks placed by the engine.')
    i2.metric('Soft macros', str(cc['soft_macros']),
              help='Standard-cell clusters placed by force-directed '
                   'relaxation.')
    i3.metric('Ports / nets', f"{cc['ports']} / {cc['nets']}",
              help='Top-level I/O ports and signal nets in play.')
    i4.metric('Canvas / grid',
              f"{details['canvas'][0]:.0f}² · "
              f"{details['grid'][0]}x{details['grid'][1]}",
              help='Die size in microns and placement grid.')
    i5.metric('Utilization', f"{details['utilization'] * 100:.1f}%",
              help='Hard-macro area over die area. Can exceed 100% '
                   'when macros overlap.')
    st.caption('Every hard macro: position, size, orientation.')
    st.table(details['macros'])
  with tab_files:
    st.subheader('Placement files')
    plc_text = _plc_to_text(plc)
    d1, d2, d3 = st.columns(3)
    d1.download_button('Best placement (.plc)', plc_text,
                       file_name='best.plc',
                       help='Macro positions/orientations for reload or '
                            'downstream tools.')
    import tempfile
    viewer = os.path.join(tempfile.gettempdir(), 'neurosilicon_best.def')
    export_def_lib.export_def(plc, viewer)
    with open(viewer) as f:
      d2.download_button('Viewer DEF (.def)', f.read(),
                         file_name='best_viewer.def',
                         help='Macro-level DEF for viewing in any DEF '
                              'viewer.')
    if info and info.get('json'):
      try:
        from neurosilicon import project as project_lib
        lef = project_lib.resolve_pdk(settings['pdk'])['lef']
        flat_path = os.path.join(tempfile.gettempdir(),
                                 'neurosilicon_flat.def')
        flatten_lib.flatten(run['netlist'], _best_plc_path(),
                            info['json'], lef, info['top'], flat_path)
        with open(flat_path) as f:
          d3.download_button('Flat DEF, cell-grade (.def)', f.read(),
                             file_name='best_flat.def',
                             help='Every standard cell placed by tiling '
                                  'inside its macro — feeds OpenROAD.')
      except (ValueError, RuntimeError, TypeError, KeyError) as e:
        d3.caption(f'Flat DEF unavailable: {e}')
    else:
      d3.caption('Flat DEF needs a synth JSON (imported or sky130 '
                 'designs).')
    st.subheader('Report bundle')
    st.download_button(
        'Download full report (.zip)',
        _report_bundle(run, rows, power),
        file_name='neurosilicon_report.zip', mime='application/zip',
        help='plc, DEFs, all figures, results.json, advice, settings.')
  with tab_valid:
    st.subheader('OpenROAD validation')
    if info and info.get('validation'):
      if st.button('Validate best placement (global route)',
                      help='Runs OpenROAD global_route on the flat DEF and '
                           'reports overflow hotspots (minutes).'):
        with st.spinner('Running OpenROAD global_route (minutes)...'):
          result = _run_validation(run, info, settings)
        st.session_state['last_run']['validated'] = result
        if run.get('run_id'):
          try:
            patch_gui_run(run['run_id'], validated=result)
          except (OSError, ValueError):
            pass
        if result['ok']:
          st.success(f"Routed: overflow {result['overflow']} over "
                     f"{result['hotspots']} hotspots.")
        else:
          st.error(result['message'])
    else:
      st.caption('Routed validation needs a consistent (json, netlist.v) '
                 'pair: available for servant, picosoc, ultraembedded, and '
                 'imported projects.')
    validated = st.session_state['last_run'].get('validated')
    if validated and validated['ok']:
      st.download_button('Congestion report (.rpt)', validated['report'],
                         file_name='gr_congestion.rpt',
                         help='Per-hotspot overflow report from the '
                              'validation run above.')
  with tab_curves:
    _history_browser(st)


def _run_validation(run, info, settings):
  """Runs tools/validate_route.tcl on the session best placement."""
  import re
  import subprocess
  import tempfile
  from neurosilicon import flatten_def as flatten_lib
  from neurosilicon import project as project_lib
  try:
    pdk = project_lib.resolve_pdk(settings['pdk'])
  except ValueError as e:
    return {'ok': False, 'message': str(e)}
  workdir = tempfile.mkdtemp(prefix='ns_validate_')
  flat = os.path.join(workdir, 'flat.def')
  try:
    flatten_lib.flatten(run['netlist'], _best_plc_path(), info['json'],
                        pdk['lef'], info['top'], flat)
  except (ValueError, RuntimeError, TypeError, KeyError) as e:
    return {'ok': False, 'message': f'Flatten failed: {e}'}
  env = dict(os.environ, NS_LIB=pdk['liberty'], NS_TLEF=pdk['techlef'],
             NS_LEF=pdk['lef'], NS_VER=os.path.abspath(info['net']),
             NS_DEF=flat + '.nopins.def',
             NS_PLACE=flat + '.place.tcl',
             NS_TOP=info['top'],
             NS_RPT=os.path.join(workdir, 'gr.rpt'))
  openroad = project_lib.resolve_tool(
      settings['pdk'].get('openroad', 'tools/openroad'))
  if not (os.path.isfile(openroad) and os.access(openroad, os.X_OK)):
    return {'ok': False,
            'message': f'OpenROAD binary missing: {openroad} (set '
                       'Settings > PDK & Toolchain > openroad).'}
  try:
    proc = subprocess.run(
        [openroad, '-no_splash', '-exit',
         project_lib.resolve_tool('tools/validate_route.tcl')],
        capture_output=True, text=True, timeout=1800, env=env)
  except subprocess.TimeoutExpired:
    return {'ok': False, 'message': 'OpenROAD timed out (30 min).'}
  except OSError as e:
    return {'ok': False, 'message': f'OpenROAD launch failed: {e}'}
  log = proc.stdout + proc.stderr
  if 'GR_DONE' not in log:
    tail = '\n'.join(log.strip().splitlines()[-8:])
    return {'ok': False, 'message': f'OpenROAD failed:\n{tail}'}
  try:
    with open(os.path.join(workdir, 'gr.rpt')) as f:
      report = f.read()
  except OSError as e:
    return {'ok': False, 'message': f'No congestion report: {e}'}
  overflow = sum(int(m.group(1))
                 for m in re.finditer(r'overflow:(\d+)', report))
  hotspots = report.count('violation type:')
  return {'ok': True, 'overflow': overflow, 'hotspots': hotspots,
          'report': report}


def _report_bundle(run, rows, power):
  import json
  import tempfile
  import zipfile
  from neurosilicon import render as render_lib
  from neurosilicon import settings as settings_lib
  import streamlit as st
  plc = st.session_state['last_plc']
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('best.plc', _plc_to_text(plc))
    viewer = os.path.join(tempfile.gettempdir(), 'ns_report_viewer.def')
    from neurosilicon import export_def as export_def_lib
    export_def_lib.export_def(plc, viewer)
    z.write(viewer, 'best_viewer.def')
    import matplotlib.pyplot as plt
    for name, fig in (
        ('floorplan.png', render_lib.plot_floorplan(
            plc, blocked=run.get('blocked'))),
        ('congestion.png', render_lib.plot_congestion(plc)),
        ('thermal.png', render_lib.plot_thermal(plc, power)),
        ('cost_breakdown.png', render_lib.cost_breakdown_figure(rows))):
      z.writestr(name, render_lib.figure_png(fig))
      plt.close(fig)
    z.writestr('results.json', json.dumps(rows, indent=2))
    z.writestr('inspector.json', json.dumps(placement_details(plc),
                                            indent=2))
    z.writestr('advice.md', run.get('advice') or 'No LLM advice recorded.')
    z.writestr('settings.json', json.dumps(
        settings_lib.redacted(settings_lib.load()), indent=2))
    z.writestr('README.txt',
               f"NeuroSilicon report for {run['choice']}. Best proxy "
               f"{run['best_cost']:.4f}. Thermal uses uploaded power or "
               'area-proportional estimates (see README).')
  return buf.getvalue()


def _history_browser(st):
  import csv
  import glob
  st.caption('Eval cost + mean reward per iteration.')
  options = sorted(glob.glob('logs/*/history.csv'))
  if not options:
    st.caption('No logs/*/history.csv found.')
    return
  choice = st.selectbox(
      'History file', options,
      help='Training history CSV to plot (eval cost + mean reward).')
  rows = []
  with open(choice, newline='') as f:
    for r in csv.DictReader(f):
      try:
        rows.append({'iter': int(r['iter']),
                     'mean_reward': float(r['mean_reward']),
                     'eval_cost': float(r['eval_cost'])})
      except (ValueError, KeyError):
        continue
  if not rows:
    st.caption('Empty history.')
    return
  import matplotlib.pyplot as plt
  from neurosilicon import render as render_lib
  fig = render_lib.history_figure(rows)
  st.pyplot(fig)
  st.download_button('Download curve (.png)', render_lib.figure_png(fig),
                     file_name='training_curve.png', mime='image/png',
                     help='Downloads exactly the curve shown above.')
  plt.close(fig)


def page_thermal(st, settings):
  import time
  # Live clock sweep: each rerun consumes one step, so the map, metrics,
  # and table animate through corners until the list is exhausted.
  if st.session_state.get('thermal_sweep'):
    st.session_state['thermal_clock'] = (
        st.session_state['thermal_sweep'].pop(0))
    if not st.session_state['thermal_sweep']:
      del st.session_state['thermal_sweep']
    time.sleep(0.5)
    st.rerun()

  from neurosilicon import priors as priors_lib
  from neurosilicon import render as render_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Analyze', 'Thermal lab',
                     'Live thermal balance monitor. Move the corner '
                     'sliders and the map, metrics, and hotspot table '
                     'update instantly.')
  run, plc = _current_run_plc(st)
  if not run or plc is None:
    st.info('No placement yet. Go to Run floorplan first.')
    return
  st.caption(f"Placement: {run['choice']} "
             f"(best proxy {run['best_cost']:.4f}).")

  base = run.get('power')
  uploaded = st.file_uploader(
      'Base power JSON (optional)',
      help='{"macro": watts-ish} reference shares. Without it, uses the '
           'run power, else area-proportional estimates.')
  if uploaded is not None:
    import json
    try:
      base = {k: float(v) for k, v in json.load(uploaded).items()}
    except (ValueError, AttributeError) as e:
      st.warning(f'Ignoring power file: {e}')
  if base is None:
    base = render_lib.estimate_power(plc)
    st.caption('Base power: area-proportional estimate.')
  else:
    st.caption('Base power: uploaded / run values.')

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    clock = st.slider(
        'Clock (MHz)', 10, 1000, 100, key='thermal_clock',
        help='Operating frequency. Dynamic power scales linearly '
             'with it.')
  with c2:
    voltage = st.slider(
        'Supply (V)', 1.0, 1.8, 1.8, step=0.05,
        help='Supply voltage (sky130 nominal 1.8 V). Dynamic power '
             'scales with V squared.')
  with c3:
    activity = st.slider(
        'Activity', 0.05, 1.0, 0.2, step=0.05,
        help='Average switching activity (fraction of nodes toggling '
             'per cycle).')
  with c4:
    leakage = st.slider(
        'Leakage share', 0.0, 0.3, 0.1, step=0.01,
        help='Static power as a fraction of base total, spread by '
             'macro area.')
  if st.button('Sweep clock 50 → 800 MHz',
               help='Animates the clock slider through corners so you '
                    'can watch hotspots move in real time.'):
    st.session_state['thermal_sweep'] = [50, 100, 150, 200, 300, 400,
                                         600, 800]
    st.rerun()

  corner = priors_lib.thermal_corner(
      plc, base, clock_mhz=clock, voltage=voltage, activity=activity,
      leakage_frac=leakage)
  m1, m2, m3 = st.columns(3)
  m1.metric('Total power (rel. units)', f"{corner['total']:.3f}",
            help='Sum over macros at this corner. 1.1 at the reference '
                 'corner (100 MHz, 1.8 V, activity 0.2, leakage 0.1).')
  m2.metric('Balance score', f"{corner['balance']:.3f}",
            help='Mean / peak power density. 1.0 is perfectly uniform; '
                 'lower means worse hotspots.')
  m3.metric('Peak macro', corner['peak_macro'] or '—',
            help='Macro with the highest power density right now.')
  st.caption('Uniformity (balance score: mean / peak power density)')
  st.progress(min(max(corner['balance'], 0.0), 1.0))

  import matplotlib.pyplot as plt
  fig = render_lib.plot_thermal(
      plc, {r['macro']: r['power'] for r in corner['rows']})
  st.pyplot(fig)
  st.download_button(
      f"Thermal @ {clock}MHz (.png)", render_lib.figure_png(fig),
      file_name=f'thermal_{clock}MHz.png', mime='image/png',
      help='Downloads exactly the map shown above.')
  plt.close(fig)

  st.subheader('Hotspots')
  st.caption('Top macros by power density at this corner.')
  st.table([{k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in row.items()}
            for row in corner['rows'][:10]])
  st.caption('Model: P = C·(V/1.8)²·(f/100)·(α/0.2) + leakage·area share, '
             'a documented proxy — not signoff power/thermal analysis.')


def page_gnn(st, settings):
  from neurosilicon import gnn_viz as gnn_lib
  from neurosilicon import render as render_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Analyze', 'GNN simulation',
                     'Step through a real policy rollout macro by macro: '
                     'placement heatmap, value estimate, and circuit '
                     'graph. Heatmaps are the policy\u2019s own softmax; '
                     'final cost includes FD relaxation.')
  nets = all_netlists()
  if not nets:
    st.error('No netlists found. Import a chip project first.')
    return
  left, right = st.columns([1, 2])
  with left:
    choice = st.selectbox(
        'Netlist', list(nets),
        help='Design to simulate the policy on.')
    ckpt = st.text_input(
        'Checkpoint (.pt)',
        CHECKPOINTS.get(choice, settings['model']['checkpoint']),
        help='Policy weights to simulate, pre-filled with the '
             'verified-good checkpoint for this design. Must exist — '
             'no fallback here, this page is about the model itself. '
             'A dead-ending rollout means this policy cannot place '
             'this design greedily; try another checkpoint.')
    greedy = st.checkbox(
        'Greedy (argmax)', value=True,
        help='On: always the top-probability cell (deterministic). Off: '
             'samples from the distribution with the seed below.')
    seed = st.number_input(
        'Seed', 0, 9999, 0,
        help='Sampling seed when greedy is off.')
    go = st.button('Simulate rollout', type='primary',
                   help='Runs one full episode and records every step '
                        'for scrubbing (FD once at the end; big netlists '
                        'take minutes).')
  with right:
    st.caption('How to read it: the heatmap is where the GNN wants the '
               'current macro (brighter = higher probability), the gold '
               'star is the chosen cell, teal blocks are placed macros, '
               'grey dots are standard cells, white lines are nets '
               '(brighter = more parallel nets).')
  if go:
    netlist, init_plc = nets[choice]
    with st.status(f'Simulating {choice}...', expanded=True) as status:
      try:
        sim = gnn_lib.simulate(netlist, init_plc, ckpt, greedy=greedy,
                               seed=seed)
      except FileNotFoundError as e:
        status.update(label='Simulation failed.', state='error')
        st.error(str(e))
        return
      st.session_state['gnn_sim'] = sim
      st.session_state['gnn_choice'] = choice
      status.update(label='Simulation complete.', state='complete')
  sim = st.session_state.get('gnn_sim')
  if not sim or not sim['steps']:
    st.info('Press **Simulate rollout** to record an episode.')
    return
  final = sim['final']
  m1, m2, m3 = st.columns(3)
  m1.metric('Final proxy cost',
            f"{final.get('proxy_cost', -1):.4f}"
            if not final.get('infeasible', True) else 'infeasible',
            help='Terminal proxy after FD (or infeasible if the rollout '
                 'dead-ended).')
  m2.metric('Steps recorded', str(len(sim['steps'])),
            help='One step per placed macro.')
  m3.metric('Design', st.session_state.get('gnn_choice', '')[:32],
            help='Simulated netlist.')
  if final.get('infeasible', True):
    st.warning('Rollout dead-ended; scrub the partial steps below.')
  st.caption('Placement order: ' + ' → '.join(sim['order']))
  import matplotlib.pyplot as plt
  step = st.slider(
      'Rollout step', 0, len(sim['steps']) - 1, len(sim['steps']) - 1,
      help='Scrub through placement steps (starts at the final step).')
  s = sim['steps'][step]
  fig = render_lib.plot_sim_step(
      sim['canvas'], s['placed'], s['probs'], s['macro'],
      s['action_cell'], s['value'], step, len(sim['steps']),
      edges=sim['edges'], soft_xy=sim['soft_xy'])
  st.pyplot(fig)
  st.download_button(
      f'Download step {step} (.png)', render_lib.figure_png(fig),
      file_name=f'gnn_step_{step}.png', mime='image/png',
      help='Downloads exactly the figure shown above.')
  plt.close(fig)
  st.caption('Value-head estimate V(s) per step.')
  st.line_chart([round(x['value'], 4) for x in sim['steps']])


def page_assistant(st, settings):
  from neurosilicon import llm as llm_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Advise', 'AI assistant',
                     'Constraint parsing and result review via your '
                     'configured provider — or the built-in rules when '
                     'none is set.')
  provider = settings['llm']['provider']
  if provider == 'off':
    st.info('LLM provider is off. Chat constraints use the rule-based '
            'parser; advice is unavailable. Set one up in Settings > '
            'AI Providers (external API key or a local Ollama server).')
  else:
    what = (f"API ({settings['llm']['model']})"
            if provider == 'api'
            else f"local ({settings['llm']['local_model']})")
    st.success(f'Provider: {what}')
    if st.button('Test connection',
                    help='Sends one tiny chat round-trip and reports '
                         'latency.'):
      with st.spinner('Pinging provider...'):
        ok, message, secs = llm_lib.test_connection(settings['llm'])
      (st.success if ok else st.error)(f'{message} ({secs}s)')
  st.subheader('Ask about the last run')
  run, _ = _current_run_plc(st)
  if not run:
    st.caption('Run a floorplan first — the assistant reviews real runs.')
    return
  st.write(f"Netlist: {run['choice']}. Best proxy: {run['best_cost']:.4f}.")
  question = st.text_area(
      'Question',
      'Why did the best episode win, and what should I try next?',
      help='Free-form question; the assistant answers from the last '
           'run rows.')
  if st.button('Ask', help='Sends the question plus last-run data to '
                           'the configured provider.'):
    with st.spinner('Thinking...'):
      try:
        answer = llm_lib.chat(
            [{'role': 'system',
              'content': ('You are a chip-floorplanning assistant. Answer '
                          'from the run data; be concrete and cite '
                          'episodes.')},
             {'role': 'user',
              'content': f'Netlist: {run["choice"]}. '
                         f'Rows: {run["rows"]}. Question: {question}'}],
            settings['llm'])
      except llm_lib.LLMError as e:
        st.error(str(e))
        return
    st.markdown(answer)
    st.session_state['last_run']['advice'] = answer
    if run.get('run_id'):
      try:
        patch_gui_run(run['run_id'], advice=answer)
      except (OSError, ValueError):
        pass


def page_settings(st, settings):
  from neurosilicon import settings as settings_lib
  from neurosilicon import ui as ui_lib

  ui_lib.page_header(st, 'Configure', 'Settings',
                     f'Saved to {settings_lib.SETTINGS_PATH} — outside '
                     'the repo, so API keys are never committed.')
  edited = {}
  for section in settings_lib.SECTIONS:
    st.subheader(section['title'])
    st.caption(section['blurb'])
    edited[section['id']] = {}
    for f in section['fields']:
      key = f"{section['id']}.{f['key']}"
      current = settings[section['id']][f['key']]
      if f['type'] == 'bool':
        edited[section['id']][f['key']] = st.checkbox(
            f['label'], value=bool(current), help=f['help'], key=key)
      elif f['type'] == 'int':
        edited[section['id']][f['key']] = st.number_input(
            f['label'], value=int(current), help=f['help'], key=key)
      elif f['type'] == 'float':
        edited[section['id']][f['key']] = st.number_input(
            f['label'], value=float(current), help=f['help'], key=key)
      elif f['type'] == 'select':
        opts = f['options']
        edited[section['id']][f['key']] = st.selectbox(
            f['label'], opts,
            index=opts.index(current) if current in opts else 0,
            help=f['help'], key=key)
      elif f['type'] == 'secret':
        entered = st.text_input(
            f['label'], value='', type='password',
            help=f['help'] + ' (Blank keeps the saved value.)', key=key)
        edited[section['id']][f['key']] = entered or current
      else:
        edited[section['id']][f['key']] = st.text_input(
            f['label'], value=str(current), help=f['help'], key=key)
  c1, c2 = st.columns(2)
  if c1.button('Save settings', type='primary'):
    settings_lib.save(edited)
    st.success(f"Saved to {settings_lib.SETTINGS_PATH}.")
  if c2.button('Restore defaults'):
    settings_lib.save(settings_lib.defaults())
    st.success('Defaults restored.')
  with st.expander('Current saved values (secrets hidden)'):
    st.json(settings_lib.redacted(settings_lib.load()))


def main():
  import streamlit as st

  from neurosilicon import settings as settings_lib
  from neurosilicon import ui as ui_lib

  st.set_page_config(page_title='NeuroSilicon console', layout='wide',
                     page_icon='▦')
  stale = missing_api()
  if stale:
    st.error('This Streamlit server loaded old code before an update '
             '(stale: ' + ', '.join(stale) + ').')
    if st.button('Reload updated code & retry',
                 help='Hot-reloads the updated modules into this running '
                      'server — no terminal restart needed in most cases.'):
      reloaded = reload_stale(stale)
      if reloaded:
        st.rerun()
    st.caption('Still stuck? Stop the server with Ctrl+C in its terminal '
               'and re-run `streamlit run neurosilicon/app.py` from the '
               'repo root.')
    return
  ui_lib.inject_css(st)
  st.sidebar.title('NeuroSilicon console')
  st.sidebar.caption(f'AI floorplanning · v{ui_lib.version()}')
  page = st.sidebar.radio(
      'Section',
      ['Dashboard', 'Run floorplan', 'Results & downloads',
       'Thermal lab', 'GNN simulation', 'AI assistant', 'Settings'],
      help='Console section. Dashboard is system status, Run places '
           'macros (with project import on top), Results shows figures '
           'and files, Thermal lab sweeps power corners, GNN simulation '
           'steps through the policy, Assistant advises, Settings '
           'persists defaults.')
  settings = settings_lib.load()
  try:
    if page == 'Dashboard':
      page_dashboard(st, settings)
    elif page == 'Run floorplan':
      page_run(st, settings)
    elif page == 'Results & downloads':
      page_results(st, settings)
    elif page == 'Thermal lab':
      page_thermal(st, settings)
    elif page == 'GNN simulation':
      page_gnn(st, settings)
    elif page == 'AI assistant':
      page_assistant(st, settings)
    else:
      page_settings(st, settings)
  except Exception as e:  # Friendly boundary: log trace, show summary.
    import traceback
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs', 'gui_runs', 'console_errors.log')
    try:
      os.makedirs(os.path.dirname(log_path), exist_ok=True)
      with open(log_path, 'a') as f:
        f.write(f'--- {page} ---\n{traceback.format_exc()}\n')
    except OSError:
      log_path = '(log unwritable)'
    st.error(f'{page} hit an unexpected error: {e}\n\nFull traceback '
             f'appended to {log_path}. Completed runs on disk are '
             'unaffected — reopen Results & downloads.')
    return
  ui_lib.page_footer(st)


if __name__ == '__main__':
  main()
