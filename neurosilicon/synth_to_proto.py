"""Converts a Yosys JSON netlist + LEF sizes to circuit-training proto format.

Hierarchical leaf modules become hard macros; top-level leaf cells become
stdcells; boundary nets become weighted macro-pin nets. Emits
`netlist.pb.txt` + a matching `initial.plc`.
"""
import collections
import gc
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple


class DSU:

  def __init__(self):
    self.p = {}

  def find(self, x):
    # Iterative with path halving: recursion depth grows with hierarchy
    # on giant netlists and blows the stack / frame memory.
    p = self.p
    root = p.setdefault(x, x)
    while p[root] != root:
      root = p[root]
    while p[x] != root:
      p[x], x = root, p[x]
    return root

  def union(self, a, b):
    self.p[self.find(a)] = self.find(b)


POWER_PINS = frozenset(['VPWR', 'VGND', 'VPB', 'VNB'])


def parse_lef(lef_path: str):
  """Returns ({cell: (w, h)}, {cell: {pin: direction}}) from a LEF file."""
  sizes, pindirs = {}, {}
  cur, pin = None, None
  with open(lef_path) as f:
    for line in f:
      m = re.match(r'\s*MACRO\s+(\S+)', line)
      if m:
        cur, pin = m.group(1), None
        pindirs.setdefault(cur, {})
        continue
      if cur is None:
        continue
      m = re.match(r'\s*SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', line)
      if m and cur not in sizes:
        sizes[cur] = (float(m.group(1)), float(m.group(2)))
        continue
      m = re.match(r'\s*PIN\s+(\S+)', line)
      if m:
        pin = m.group(1)
        continue
      if pin:
        m = re.match(r'\s*DIRECTION\s+(\S+)\s*;', line)
        if m:
          pindirs[cur][pin] = m.group(1).lower()
          pin = None
  return sizes, pindirs


def parse_lef_sizes(lef_path: str) -> Dict[str, Tuple[float, float]]:
  """Returns {cell_type: (width, height)} in microns from a LEF file."""
  sizes, _ = parse_lef(lef_path)
  return sizes


def _trim_module(mod: dict) -> dict:
  """Keeps only the module fields the converter reads.

  Yosys also emits attributes / parameter defaults / netnames, which
  dominate memory on giant designs and are never read here.
  """
  cells = {}
  for name, info in mod.get('cells', {}).items():
    cell = {'type': info['type']}
    if 'connections' in info:
      cell['connections'] = info['connections']
    if 'port_directions' in info:
      cell['port_directions'] = info['port_directions']
    cells[name] = cell
  ports = {}
  for name, info in mod.get('ports', {}).items():
    ports[name] = {'bits': info.get('bits', []),
                   'direction': info.get('direction', 'input')}
  return {'cells': cells, 'ports': ports}


def load_modules(json_path: str):
  """Loads trimmed Yosys modules without holding the whole JSON in RAM.

  Files over 32 MB stream module-by-module via ijson when available
  (peak is the largest single module, not the 5-10x object graph of
  json.load); smaller files, or a missing ijson, use json.load.
  """
  try:
    big = os.path.getsize(json_path) > 32 * 1024 * 1024
  except OSError:
    big = False
  if big:
    try:
      import ijson
    except ImportError:
      big = False
  if not big:
    with open(json_path) as f:
      db = json.load(f)
    return {name: _trim_module(m) for name, m in db['modules'].items()}
  modules = {}
  with open(json_path, 'rb') as f:
    for name, mod in ijson.kvitems(f, 'modules'):
      modules[name] = _trim_module(mod)
  return modules


class Converter:
  """Yosys JSON -> proto netlist + initial placement."""

  def __init__(self, json_path: str, lef_path: str, top: str,
               utilization: float = 0.6,
               canvas_utilization: float = 0.4):
    self.modules = load_modules(json_path)
    self.sizes, self.pindirs = parse_lef(lef_path)
    self.top = top
    self.util = utilization
    self.canvas_util = canvas_utilization
    self.dsu = DSU()
    # (ctx_path, bit) -> list of endpoints
    self.endpoints: Dict[tuple, list] = {}
    # macro short name -> info
    self.macros: Dict[str, dict] = {}
    # leaf stdcells at dissolved levels: key -> info
    self.stdcells: Dict[str, dict] = {}
    self.top_ports: Dict[str, dict] = {}
    self.warnings: List[str] = []

  # -- hierarchy walk ----------------------------------------------------
  def _is_leaf_module(self, mod: str) -> bool:
    return all(info['type'] not in self.modules
               for info in self.modules[mod]['cells'].values())

  def _walk(self, mod: str, ctx: tuple, macro: Optional[str]):
    """Walks hierarchy; assigns leaf cells to a macro (or stdcell)."""
    for cell, info in self.modules[mod]['cells'].items():
      typ = info['type']
      conns = info.get('connections', {})
      dirs = info.get('port_directions', {})
      if typ in self.modules:
        child_ctx = ctx + ((cell, typ),)
        if macro is None and self._is_leaf_module(typ):
          name = cell
          self.macros[name] = {'ctx': child_ctx, 'mod': typ,
                               'cells': [], 'area': 0.0, 'ports': {}}
          self._record_instance_pins(ctx, cell, typ, conns, dirs, name)
          self._walk(typ, child_ctx, name)
        elif macro is None:
          # Dissolved level: recurse, keep macro=None.
          self._record_hier_pass(ctx, cell, typ, conns)
          self._walk(typ, child_ctx, None)
        else:
          self.warn(f'nested instance {cell} inside macro {macro}')
          self._record_hier_pass(ctx, cell, typ, conns)
          self._walk(typ, child_ctx, macro)
      else:
        w, h = self.sizes.get(typ, (None, None))
        if w is None:
          self.warn(f'unknown size for leaf type {typ}')
          w, h = 1.0, 1.0
        # Mapped cells lack port directions in JSON; use LEF data.
        dirs = dirs or self.pindirs.get(typ, {})
        key = '/'.join([c for c, _ in ctx] + [cell])
        # Slim record: conns/dirs/ctx are only needed to record the
        # endpoints below; keeping them per cell doubles memory on
        # million-cell designs.
        rec = {'type': typ, 'w': w, 'h': h, 'macro': macro}
        if macro is None:
          self.stdcells[key] = rec
        else:
          self.macros[macro]['cells'].append(rec)
          self.macros[macro]['area'] += w * h
        self._record_leaf_endpoints(ctx, key, conns, dirs, macro)

  def _seg(self, ctx: tuple, bit) -> Optional[tuple]:
    if isinstance(bit, str):
      return None
    return (ctx, bit)

  def _add_endpoint(self, ctx, bit, ep):
    seg = self._seg(ctx, bit)
    if seg is None:
      return
    self.dsu.find(seg)
    self.endpoints.setdefault(seg, []).append(ep)

  def _record_leaf_endpoints(self, ctx, key, conns, dirs, macro):
    for port, bits in conns.items():
      if port in POWER_PINS:
        continue
      for bit in bits:
        self._add_endpoint(ctx, bit, ('cell', key, port,
                                      dirs.get(port, 'input'), macro))

  def _record_instance_pins(self, ctx, cell, typ, conns, dirs, macro):
    ports = self.modules[typ]['ports']
    for port, bits in conns.items():
      cbits = ports[port]['bits']
      for k, bit in enumerate(bits):
        a = self._seg_or_const(ctx, bit)
        b = self._seg_or_const(ctx + ((cell, typ),), cbits[k])
        if a is not None and b is not None:
          self.dsu.union(a, b)
        self._add_endpoint(ctx, bit,
                            ('pin', macro, port, dirs.get(port, 'input')))

  def _record_hier_pass(self, ctx, cell, typ, conns):
    ports = self.modules[typ]['ports']
    for port, bits in conns.items():
      cbits = ports[port]['bits']
      for k, bit in enumerate(bits):
        a = self._seg_or_const(ctx, bit)
        b = self._seg_or_const(ctx + ((cell, typ),), cbits[k])
        if a is not None and b is not None:
          self.dsu.union(a, b)

  def _seg_or_const(self, ctx, bit):
    return None if isinstance(bit, str) else (ctx, bit)

  def warn(self, msg):
    seen = getattr(self, '_warn_set', None)
    if seen is None:
      seen = self._warn_set = set(self.warnings)
    if msg not in seen:
      seen.add(msg)
      self.warnings.append(msg)

  # -- nets ---------------------------------------------------------------
  def flat_net_spans(self):
    """Walk hierarchy; return per-net flat fanout + touched node keys.

    Returns a list of {'fanout': int, 'nodes': set} where fanout counts
    flat sink endpoints (cells + boundary pins + ports) and nodes holds
    _ep_node keys: ('stdcell', key), ('pin', macro, port), ('port', name).
    Only nets touching >= 2 distinct nodes with >= 1 sink are returned.
    Power pins and constant bits are excluded by the endpoint recorders.
    Resets walk state first, so it is safe to call on a fresh Converter
    (do not mix with convert() on the same instance).
    """
    self.dsu = DSU()
    self.endpoints = {}
    self.macros = {}
    self.stdcells = {}
    self._walk(self.top, (), None)
    for pname, pinfo in self.modules[self.top]['ports'].items():
      for bit in pinfo['bits']:
        self._add_endpoint((), bit, ('port', pname, pinfo['direction']))
    groups: Dict[tuple, list] = {}
    for seg, eps in self.endpoints.items():
      groups.setdefault(self.dsu.find(seg), []).extend(eps)
    out = []
    for root in sorted(groups, key=repr):
      eps = [e for e in groups[root]
             if not (e[0] == 'cell' and e[4])]
      nodes = {self._ep_node(e) for e in eps}
      sinks = [e for e in eps if self._ep_dir(e) != 'output']
      if len(nodes) < 2 or not sinks:
        continue
      out.append({'fanout': len(sinks), 'nodes': nodes})
    return out

  def _collect_nets(self):
    """Groups endpoints by DSU set; resolves driver/sinks per net.

    Returns a list of dicts with stable ids: id, driver (node key),
    sinks ([node keys]), dirs ({node key: dir}), counts ({node: endpoints}).
    """
    groups: Dict[tuple, list] = {}
    for seg, eps in self.endpoints.items():
      groups.setdefault(self.dsu.find(seg), []).extend(eps)
    nets = []
    for root in sorted(groups, key=repr):
      eps = groups[root]
      # Cells hidden inside macros don't appear; their boundary pins do.
      eps = [e for e in eps
             if not (e[0] == 'cell' and e[4])]
      # Collapse duplicate endpoints on the same node.
      by_node: Dict[tuple, list] = {}
      for ep in eps:
        by_node.setdefault(self._ep_node(ep), []).append(ep)
      if len(by_node) < 2:
        continue
      reps = {k: v[0] for k, v in by_node.items()}
      counts = {k: len(v) for k, v in by_node.items()}
      drivers = [k for k in by_node if self._ep_dir(reps[k]) == 'output']
      sinks = [k for k in by_node if self._ep_dir(reps[k]) != 'output']
      if not drivers or not sinks:
        continue
      if len(drivers) > 1:
        # Feedthrough: an output pin fed by an input pin of the same
        # macro on this net is not a true driver.
        in_macros = {reps[k][1] for k in sinks if reps[k][0] == 'pin'}
        real = [d for d in drivers
                if not (reps[d][0] == 'pin' and reps[d][1] in in_macros)]
        drivers = real or drivers
      if len(drivers) > 1:
        # Same-macro outputs share a source; cross-node conflicts are real.
        nodes = {reps[d][1] if reps[d][0] == 'pin' else d
                 for d in drivers}
        if len(nodes) > 1:
          self.warn(f'multi-driver net ({len(drivers)} drivers)')
      nets.append({'id': len(nets), 'driver': drivers[0], 'sinks': sinks,
                   'counts': counts, 'reps': reps,
                   'nodes': list(by_node)})
    return nets

  # AlphaChip grouping rules (meta_netlist_util.py, grouper.py):
  # ignore nets above this fanout when clustering, and cap cluster
  # area so no group spans more than ~1/4 of the canvas per side.
  MAX_CLUSTER_FANOUT = 500

  def _cell_area(self, cellkey):
    rec = self.stdcells.get(cellkey, {})
    return rec.get('w', 0.0) * rec.get('h', 0.0)

  def _cluster_stdcells(self, nets, target_size: int = 25):
    """Greedy connectivity clustering of dissolved-level stdcells.

    Returns {cell_key: Grp_N}. Cells without nets are left unassigned.
    Nets with fanout above MAX_CLUSTER_FANOUT are skipped (AlphaChip
    max_allowed_fanouts), and clusters stop growing past 1/16 of the
    total cell area (AlphaChip breakup: sqrt(canvas_area/16) per side).
    """
    adj: Dict[str, set] = {}
    for net in nets:
      cells = [k[1] for k in net['nodes'] if k[0] == 'stdcell']
      if len(cells) > self.MAX_CLUSTER_FANOUT:
        continue
      for k in cells:
        adj.setdefault(k, set())
      for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
          adj[cells[i]].add(cells[j])
          adj[cells[j]].add(cells[i])
    total_area = sum(r['w'] * r['h'] for r in self.stdcells.values())
    total_area += sum(m.get('area', 0.0) for m in self.macros.values())
    area_cap = total_area / 16.0 if total_area > 0 else float('inf')
    assign: Dict[str, str] = {}
    order = sorted(adj, key=lambda k: (-len(adj[k]), k))
    groups: List[List[str]] = []
    for seed in order:
      if seed in assign:
        continue
      members, queue = [seed], collections.deque([seed])
      assign[seed] = ''  # placeholder
      member_area = self._cell_area(seed)
      while queue and len(members) < target_size:
        cur = queue.popleft()
        for nb in sorted(adj[cur]):
          if (nb not in assign and len(members) < target_size and
              member_area + self._cell_area(nb) <= area_cap):
            assign[nb] = ''
            members.append(nb)
            member_area += self._cell_area(nb)
            queue.append(nb)
      groups.append(sorted(members))
    groups.sort(key=lambda g: g[0])
    for i, g in enumerate(groups):
      for k in g:
        assign[k] = f'Grp_{i}'
    return assign

  def _ep_node(self, ep):
    kind = ep[0]
    if kind == 'pin':
      return ('pin', ep[1], ep[2])
    if kind == 'cell':
      return ('stdcell', ep[1])
    return ('port', ep[1])  # ('port', name)

  def _ep_dir(self, ep):
    kind = ep[0]
    if kind == 'port':
      # Top-level input port drives; output port sinks.
      return 'output' if ep[2] == 'input' else 'input'
    if kind == 'cell':
      return ep[3]
    return ep[-1]

  # -- emit -----------------------------------------------------------------
  def convert(self, outdir: str, block: Optional[str] = None,
              cluster_size: int = 25, auto_grid: bool = True,
              grid_sweep_step: int = 1):
    os.makedirs(outdir, exist_ok=True)
    block = block or self.top
    self._walk(self.top, (), None)
    # Top-level ports.
    for pname, pinfo in self.modules[self.top]['ports'].items():
      self.top_ports[pname] = pinfo
      for bit in pinfo['bits']:
        self._add_endpoint((), bit, ('port', pname, pinfo['direction']))
    nets = self._collect_nets()
    # Walk state is dead weight after net collection; free it before
    # clustering/emit so giant designs peak once, not twice.
    self.endpoints = {}
    self.dsu = DSU()
    gc.collect()

    # Cluster dissolved-level stdcells into soft macros; rewrite their
    # net endpoints to cluster pins (one pin per cluster per net).
    assign = self._cluster_stdcells(nets, target_size=cluster_size)

    def rewrite(net, key):
      if key[0] == 'stdcell' and key[1] in assign:
        return ('pin', assign[key[1]], f'np{net["id"]}')
      return key

    pin_weight: Dict[tuple, float] = {}
    for net in nets:
      net['driver'] = rewrite(net, net['driver'])
      net['sinks'] = [rewrite(net, s) for s in net['sinks']]
      for key, cnt in net['counts'].items():
        wkey = rewrite(net, key)
        if wkey[0] == 'pin':
          pin_weight[(wkey[1], wkey[2])] = (
              pin_weight.get((wkey[1], wkey[2]), 0.0) + cnt)
    clusters: Dict[str, dict] = {}
    for cellkey, grp in assign.items():
      rec = self.stdcells[cellkey]
      c = clusters.setdefault(grp, {'cells': [], 'area': 0.0})
      c['cells'].append(cellkey)
      c['area'] += rec['w'] * rec['h']
    for grp, c in clusters.items():
      side = math.sqrt(max(c['area'], 1.0) / self.util)
      c['w'] = c['h'] = side

    # Macro geometry (square @ utilization).
    for name, m in self.macros.items():
      side = math.sqrt(max(m['area'], 1.0) / self.util)
      m['w'] = m['h'] = side
    total_area = (sum(m['area'] for m in self.macros.values()) +
                  sum(c['w'] * c['h'] for c in self.stdcells.values()))
    canvas = math.sqrt(max(total_area, 1.0) / self.canvas_util)
    self.canvas_w = self.canvas_h = canvas

    # Pin placement around macro perimeters (cluster pins stay centered).
    pin_names: Dict[str, List[str]] = (
        {m: [] for m in self.macros} |
        {g: [] for g in clusters})
    for net in nets:
      for key in [net['driver']] + net['sinks']:
        if key[0] == 'pin' and key[2] not in pin_names[key[1]]:
          pin_names[key[1]].append(key[2])
    pin_pos = {}
    for m, names in pin_names.items():
      names.sort()
      if m in clusters:
        for pn in names:
          pin_pos[(m, pn)] = (0.0, 0.0)
        continue
      w, h = self.macros[m]['w'], self.macros[m]['h']
      n = max(len(names), 1)
      for i, pn in enumerate(names):
        frac = (i + 0.5) / n * 4.0
        side = int(frac) % 4
        t = frac - math.floor(frac) - 0.5
        if side == 0:
          dx, dy = t * w, h / 2
        elif side == 1:
          dx, dy = w / 2, -t * h
        elif side == 2:
          dx, dy = -t * w, -h / 2
        else:
          dx, dy = -w / 2, t * h
        pin_pos[(m, pn)] = (dx, dy)

    # Node order: macros, soft clusters, leftover stdcells, ports, pins.
    macro_list = sorted(self.macros)
    cluster_list = sorted(clusters)
    std_list = sorted(s for s in self.stdcells if s not in assign)
    port_list = sorted(self.top_ports)
    pin_list = sorted([(m, pn) for m, names in pin_names.items()
                       for pn in names])
    order = ([('macro', m) for m in macro_list] +
             [('soft', g) for g in cluster_list] +
             [('stdcell', s) for s in std_list] +
             [('port', p) for p in port_list] +
             [('pin', m, pn) for m, pn in pin_list])
    index = {o: i for i, o in enumerate(order)}

    # Net driver/sink proto names.
    std_proto = {}
    for s in std_list:
      std_proto[s] = 'C' + str(abs(hash(s)) % 10**8)

    def name_of(key):
      if key[0] == 'pin':
        return f'{key[1]}/{key[2]}'
      if key[0] == 'stdcell':
        return std_proto[key[1]]
      return key[1]

    net_inputs: Dict[str, List[str]] = {}
    for net in nets:
      d = name_of(net['driver'])
      net_inputs.setdefault(d, [])
      for s in net['sinks']:
        sn = name_of(s)
        if sn != d and sn not in net_inputs[d]:
          net_inputs[d].append(sn)

    # Initial spread positions (row-major over canvas).
    n_macro = len(macro_list)
    ncol = math.ceil(math.sqrt(max(n_macro, 1)))
    init_xy = {}
    for i, m in enumerate(macro_list):
      r, c = divmod(i, ncol)
      init_xy[('macro', m)] = (
          canvas * (c + 0.5) / ncol, canvas * (r + 0.5) / ncol)
    for i, p in enumerate(port_list):
      init_xy[('port', p)] = (
          canvas * (i + 0.5) / max(len(port_list), 1), 0.5)
    for s in std_list:
      init_xy[('stdcell', s)] = (canvas / 2, canvas / 2)
    for g in cluster_list:
      init_xy[('soft', g)] = (canvas / 2, canvas / 2)

    # Write netlist.
    nl_path = os.path.join(outdir, 'netlist.pb.txt')
    with open(nl_path, 'w') as f:
      f.write(f'# proto-file: converted from {self.top} via yosys JSON\n')
      f.write(f'# FP bbox: {{0.0 0.0}} {{{canvas:.3f} {canvas:.3f}}}\n')
      for m in macro_list:
        mm = self.macros[m]
        x, y = init_xy[('macro', m)]
        self._node(f, m, [], {
            'type': 'MACRO', 'width': mm['w'], 'height': mm['h'],
            'x': x, 'y': y, 'orientation': 'N'})
      for g in cluster_list:
        c = clusters[g]
        x, y = init_xy[('soft', g)]
        self._node(f, g, [], {
            'type': 'macro', 'width': c['w'], 'height': c['h'],
            'x': x, 'y': y})
      for s in std_list:
        c = self.stdcells[s]
        x, y = init_xy[('stdcell', s)]
        self._node(f, std_proto[s], net_inputs.get(std_proto[s], []), {
            'type': 'STDCELL', 'width': c['w'], 'height': c['h'],
            'x': x, 'y': y})
      for p in port_list:
        x, y = init_xy[('port', p)]
        self._node(f, p, net_inputs.get(p, []), {
            'type': 'PORT', 'x': x, 'y': y})
      for m, pn in pin_list:
        w = pin_weight.get((m, pn), 1.0)
        if m in clusters:
          x, y = init_xy[('soft', m)]
          self._node(f, f'{m}/{pn}', net_inputs.get(f'{m}/{pn}', []), {
              'type': 'macro_pin', 'macro_name': m,
              'x': x, 'y': y, 'weight': w})
        else:
          dx, dy = pin_pos[(m, pn)]
          self._node(f, f'{m}/{pn}', net_inputs.get(f'{m}/{pn}', []), {
              'type': 'MACRO_PIN', 'macro_name': m,
              'x_offset': dx, 'y_offset': dy, 'weight': w})

    # Placement grid: AlphaChip waste-ratio auto-selection, else 32x32.
    grid_cols, grid_rows = 32, 32
    if auto_grid:
      from neurosilicon import grid_select as grid_select_lib
      try:
        grid_cols, grid_rows = grid_select_lib.suggest_for_proto(
            nl_path, canvas, canvas, sweep_step=grid_sweep_step)
      except Exception as e:  # selection is best-effort; 32x32 fallback
        self.warn(f'grid auto-select failed ({e}); using 32x32')
        grid_cols, grid_rows = 32, 32

    # Write initial placement (macros + stdcells + ports).
    plc_path = os.path.join(outdir, 'initial.plc')
    with open(plc_path, 'w') as f:
      f.write('# Placement file for Circuit Training (converted)\n')
      f.write(f'# Columns : {grid_cols}  Rows : {grid_rows}\n')
      f.write(f'# Width : {canvas:.3f}  Height : {canvas:.3f}\n')
      f.write(f'# Block : {block}\n')
      f.write('# Routes per micron, hor : 70.33  ver : 74.51\n')
      f.write('# Routes used by macros, hor : 51.79  ver : 51.79\n')
      f.write('# Smoothing factor : 5\n')
      f.write('# Overlap threshold : 0.004\n')
      f.write('#\n# node_index x y orientation fixed\n')
      for o in order:
        if o[0] == 'pin':
          continue
        x, y = init_xy[o]
        fixed = 1 if o[0] == 'port' else 0
        f.write(f'{index[o]} {x:.3f} {y:.3f} - {fixed}\n')
    import json as jsonlib
    with open(os.path.join(outdir, 'clusters.json'), 'w') as f:
      jsonlib.dump(assign, f)
    return {'netlist': nl_path, 'plc': plc_path,
            'macros': len(macro_list), 'clusters': len(cluster_list),
            'stdcells': len(std_list),
            'ports': len(port_list), 'pins': len(pin_list),
            'nets': len(nets), 'canvas': canvas,
            'grid': (grid_cols, grid_rows),
            'warnings': self.warnings}

  @staticmethod
  def _node(f, name, inputs, attrs):
    f.write('node {\n  name: "%s"\n' % name)
    for i in inputs:
      f.write('  input: "%s"\n' % i)
    for k in sorted(attrs):
      v = attrs[k]
      f.write('  attr {\n    key: "%s"\n    value {\n' % k)
      if isinstance(v, str):
        f.write('      placeholder: "%s"\n' % v)
      else:
        f.write('      f: %s\n' % v)
      f.write('    }\n  }\n')
    f.write('}\n')


def main():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('--json', required=True)
  p.add_argument('--lef', required=True)
  p.add_argument('--top', required=True)
  p.add_argument('--outdir', required=True)
  p.add_argument('--utilization', type=float, default=0.6)
  p.add_argument('--canvas_util', type=float, default=0.4)
  args = p.parse_args()
  stats = Converter(
      args.json, args.lef, args.top, utilization=args.utilization,
      canvas_utilization=args.canvas_util).convert(args.outdir)
  print({k: v for k, v in stats.items() if k != 'warnings'})
  for w in stats['warnings'][:10]:
    print('WARN:', w)


if __name__ == '__main__':
  main()
