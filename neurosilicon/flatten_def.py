"""Flattens a macro placement to a detailed DEF for OpenROAD validation.

Leaf cells are tiled in rows inside their macro's placed bounding box;
top-level ports use their placed coordinates. Connectivity comes from
the gate-level Verilog (no NETS section), so instance names must match
the Verilog hierarchy (`parent/child/cell`).
"""
import math
import os
import re


def read_proto_order(netlist_path):
  """Returns [node names] in netlist.pb.txt order."""
  names = []
  with open(netlist_path) as f:
    for line in f:
      m = re.match(r'\s*name:\s*"([^"]+)"', line)
      if m:
        names.append(m.group(1))
  return names


def read_plc_coords(plc_path, order):
  """Returns {node name: (x, y)} from a .plc file."""
  coords = {}
  with open(plc_path) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      parts = line.split()
      coords[order[int(parts[0])]] = (float(parts[1]), float(parts[2]))
  return coords


def tile_cells(cells, x0, y0, x1, y1):
  """Row-tiles [(key, w, h)] inside a box. Returns {key: (cx, cy)}."""
  cells = sorted(cells, key=lambda c: -c[2])
  out = {}
  y = y0
  i = 0
  while i < len(cells):
    row_h = cells[i][2]
    x = x0
    j = i
    while j < len(cells) and x + cells[j][1] <= x1 + 1e-9:
      out[cells[j][0]] = (x + cells[j][1] / 2, y + row_h / 2)
      x += cells[j][1]
      j += 1
    if j == i:  # Single cell wider than box: center it, warn via size.
      out[cells[i][0]] = ((x0 + x1) / 2, y + row_h / 2)
      j = i + 1
    y += row_h
    i = j
  return out


def flatten(netlist_path, plc_path, json_path, lef_path, top, out_path):
  """Writes a detailed DEF. Returns stats dict."""
  import json as jsonlib
  from neurosilicon.synth_to_proto import Converter, parse_lef
  sizes, _ = parse_lef(lef_path)
  with open(json_path) as f:
    db = jsonlib.load(f)
  modules = db['modules']

  order = read_proto_order(netlist_path)
  coords = read_plc_coords(plc_path, order)

  # Collect leaf cells per macro (macros = hierarchical leaf modules,
  # mirroring Converter._walk selection).
  conv = Converter.__new__(Converter)
  conv.modules = modules
  conv.warnings = []
  macro_cells = {}  # macro name -> [(hierkey, type)]
  orphan_cells = []

  def walk(mod, path, macro):
    for cell, info in modules[mod]['cells'].items():
      typ = info['type']
      if typ in modules:
        if macro is None and conv._is_leaf_module(typ):
          macro_cells.setdefault(cell, [])
          walk(typ, path + (cell,), cell)
        else:
          walk(typ, path + (cell,), macro)
      else:
        key = '/'.join(path + (cell,))
        if macro is None:
          orphan_cells.append((key, typ))
        else:
          macro_cells[macro].append((key, typ))

  walk(top, (), None)

  placed = {}  # hierkey -> (x, y, type, orient)
  overflow = []
  canvas_w = canvas_h = 0.0
  with open(netlist_path) as f:
    for line in f:
      m = re.search(r'FP bbox: \{[\d.]+ [\d.]+\} \{([\d.]+) ([\d.]+)\}', line)
      if m:
        canvas_w, canvas_h = float(m.group(1)), float(m.group(2))
  clusters_path = os.path.join(os.path.dirname(netlist_path),
                               'clusters.json')
  assign = {}
  if os.path.isfile(clusters_path):
    import json as jsonlib
    with open(clusters_path) as f:
      assign = jsonlib.load(f)
  for macro, members in macro_cells.items():
    if macro not in coords:
      continue
    # Macro size: re-derive from proto would need parsing; use plc
    # neighbors is wrong -- read from netlist instead (below).
    placed.setdefault('__macros__', {})[macro] = members
  # Macro w/h from netlist attrs.
  macro_wh = {}
  cur, last_key = None, None
  with open(netlist_path) as f:
    for line in f:
      m = re.match(r'\s*name:\s*"([^"]+)"', line)
      if m:
        cur, last_key = m.group(1), None
      m = re.match(r'\s*key:\s*"(\w+)"', line)
      if m:
        last_key = m.group(1)
      m = re.match(r'\s*f:\s*([\d.eE+-]+)', line)
      if m and cur and last_key in ('width', 'height'):
        macro_wh.setdefault(cur, {})[last_key] = float(m.group(1))
  for macro, members in placed.pop('__macros__').items():
    cx, cy = coords[macro]
    w = macro_wh.get(macro, {}).get('width', 1.0)
    h = macro_wh.get(macro, {}).get('height', 1.0)
    cells = [(k,) + sizes.get(t, (1.0, 1.0)) for k, t in members]
    pos = tile_cells(cells, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    for (k, t), (x, y) in zip(members, [pos[c[0]] for c in cells]):
      cw, ch = sizes.get(t, (1.0, 1.0))
      cx2 = min(max(x, cw / 2), canvas_w - cw / 2)
      cy2 = min(max(y, ch / 2), canvas_h - ch / 2)
      if abs(cx2 - x) + abs(cy2 - y) > 1e-9:
        overflow.append(k)
      placed[k] = (cx2, cy2, t, 'N')
  # Orphan cells: clustered ones tile inside their cluster's placed
  # box; leftovers tile across the full canvas.
  by_cluster = {}
  leftovers = []
  for k, t in orphan_cells:
    if k in assign:
      by_cluster.setdefault(assign[k], []).append((k, t))
    else:
      leftovers.append((k, t))
  def clamp(x, y, w, h):
    return (min(max(x, w / 2), canvas_w - w / 2),
            min(max(y, h / 2), canvas_h - h / 2))

  for grp, members in by_cluster.items():
    if grp not in coords:
      leftovers.extend(members)
      continue
    cx, cy = coords[grp]
    w = macro_wh.get(grp, {}).get('width', 1.0)
    h = macro_wh.get(grp, {}).get('height', 1.0)
    cells = [(k,) + sizes.get(t, (1.0, 1.0)) for k, t in members]
    pos = tile_cells(cells, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    for k, t in members:
      x, y = pos[k]
      cw, ch = sizes.get(t, (1.0, 1.0))
      placed[k] = clamp(x, y, cw, ch) + (t, 'N')
  if leftovers:
    ocells = [(k,) + sizes.get(t, (1.0, 1.0)) for k, t in leftovers]
    pos = tile_cells(ocells, 0, 0, canvas_w, canvas_h)
    for k, t in leftovers:
      x, y = pos[k]
      cw, ch = sizes.get(t, (1.0, 1.0))
      placed[k] = clamp(x, y, cw, ch) + (t, 'N')
  # Ports: node names whose proto type is PORT.
  port_names = set()
  cur, cur_type = None, None
  with open(netlist_path) as f:
    for line in f:
      m = re.match(r'\s*name:\s*"([^"]+)"', line)
      if m:
        if cur_type == 'PORT':
          port_names.add(cur)
        cur, cur_type = m.group(1), None
      m = re.search(r'placeholder:\s*"(\w+)"', line)
      if m:
        cur_type = m.group(1)
    if cur_type == 'PORT':
      port_names.add(cur)
  ports = {n: coords[n] for n in port_names if n in coords}
  port_dirs = {p: info.get('direction', 'input').upper()
               for p, info in modules[top].get('ports', {}).items()}

  def d(v):
    return int(round(v * 1000))

  mg = 2000  # die margin (dbu) so cell power rails stay inside DIEAREA
  dw, dh = d(canvas_w) + 2 * mg, d(canvas_h) + 2 * mg
  items = sorted(placed.items())
  pins_start = pins_end = None
  lines = ['VERSION 5.8 ;', 'DIVIDERCHAR "/" ;', 'BUSBITCHARS "[]" ;',
           f'DESIGN {top} ;', 'UNITS DISTANCE MICRONS 1000 ;',
           f'DIEAREA ( 0 0 ) ( {dw} {dh} ) ;']
  rh, sw = 2720, 460  # unithd row height / site width in dbu
  nrows, nsteps = int(d(canvas_h) // rh), int(d(canvas_w) // sw)
  for r in range(nrows):
    ori = 'N' if r % 2 == 0 else 'FS'
    lines.append(f'ROW _ROW_{r} unithd {mg} {mg + r * rh} {ori} '
                 f'DO {nsteps} BY 1 STEP {sw} 0 ;')
  lines.append(f'COMPONENTS {len(items)} ;')
  for k, (x, y, t, o) in items:
    lines.append(f'- {k} {t} + PLACED ( {d(x) + mg} {d(y) + mg} ) {o} ;')
  lines.append('END COMPONENTS')
  pins_start = len(lines)
  lines.append(f'PINS {len(ports)} ;')
  for name, (x, y) in sorted(ports.items()):
    lines.append(f'- {name} + NET {name} '
                 f'+ DIRECTION {port_dirs.get(name, "INPUT")} + USE SIGNAL')
    lines.append('  + PORT')
    lines.append('  + LAYER met1 ( -1000 -1000 ) ( 1000 1000 )')
    lines.append(f'  + PLACED ( {d(x) + mg} {d(y) + mg} ) N ;')
  lines.append('END PINS')
  pins_end = len(lines)
  lines.append('END DESIGN')
  with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  nopins_path = out_path + '.nopins.def'
  with open(nopins_path, 'w') as f:
    f.write('\n'.join(lines[:pins_start] + lines[pins_end:]) + '\n')
  tcl_path = out_path + '.place.tcl'
  with open(tcl_path, 'w') as f:
    f.write('set __blk [ord::get_db_block]\n')
    for k, (x, y, t, o) in items:
      esc = k.replace('[', '\\[').replace(']', '\\]')
      f.write('set __i [$__blk findInst {%s}]\n' % esc)
      f.write('if {$__i != "" && $__i != "NULL"} {\n')
      f.write('$__i setLocation %d %d\n' % (d(x) + mg, d(y) + mg))
      f.write('$__i setPlacementStatus PLACED\n}\n')
  return {'components': len(items), 'pins': len(ports),
          'overflow': len(overflow), 'path': out_path,
          'nopins_def': nopins_path, 'placements_tcl': tcl_path}


def main():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('--netlist', required=True)
  p.add_argument('--plc', required=True)
  p.add_argument('--json', required=True)
  p.add_argument('--lef', required=True)
  p.add_argument('--top', required=True)
  p.add_argument('--out', required=True)
  args = p.parse_args()
  print(flatten(args.netlist, args.plc, args.json, args.lef, args.top,
                args.out))


if __name__ == '__main__':
  main()
