"""Viewer-grade DEF export for KLayout / OpenROAD inspection.

Emits DIEAREA + COMPONENTS (+ PORT pins) with placed coordinates. NETS are
emitted on a best-effort basis; this file is meant for visual checking and
downstream experimentation, not signoff.
"""
import os


def _d(val):
  return int(round(val * 1000))


def export_def(plc, path, design=None):
  """Writes a minimal DEF 5.8 file for the current placement."""
  w, h = plc.get_canvas_width_height()
  design = design or plc.get_block_name() or 'neurosilicon'
  comps = []
  for idx in plc.get_macro_indices():
    name = plc.get_node_name(idx)
    x, y = plc.get_node_location(idx)
    orient = 'N'
    try:
      o = plc.get_macro_orientation(idx)
      if o:
        orient = o
    except Exception:
      pass
    master = ('SOFTMACRO' if plc.is_node_soft_macro(idx)
              else 'HARDMACRO')
    comps.append(
        f'- {name} {master} + PLACED ( {_d(x)} {_d(y)} ) {orient} ;')
  pins = []
  for idx in plc.port_indices:
    name = plc.get_node_name(idx)
    x, y = plc.get_node_location(idx)
    pins.append(f'- {name} + NET {name}_net + DIRECTION INPUT '
                f'+ PLACED ( {_d(x)} {_d(y)} ) N ;')
  nets = []
  for driver, sinks in plc.nets.items():
    terms = []
    ok = True
    for fullname in [driver] + list(sinks):
      if '/' in fullname:
        comp, pin = fullname.rsplit('/', 1)
        terms.append(f'( {comp} {pin} )')
      else:
        # Pin-less endpoint (port / raw stdcell): tie to its own name.
        terms.append(f'( {fullname} {fullname} )')
    nets.append(f'- {driver}_net ' + ' '.join(terms) + ' ;')
  with open(path, 'w') as f:
    f.write('VERSION 5.8 ;\n')
    f.write('DIVIDERCHAR "/" ;\n')
    f.write('BUSBITCHARS "[]" ;\n')
    f.write(f'DESIGN {design} ;\n')
    f.write('UNITS DISTANCE MICRONS 1000 ;\n')
    f.write(f'DIEAREA ( 0 0 ) ( {_d(w)} {_d(h)} ) ;\n')
    f.write(f'COMPONENTS {len(comps)} ;\n')
    for c in comps:
      f.write(c + '\n')
    f.write('END COMPONENTS\n')
    f.write(f'PINS {len(pins)} ;\n')
    for p in pins:
      f.write(p + '\n')
    f.write('END PINS\n')
    f.write(f'NETS {len(nets)} ;\n')
    for n in nets:
      f.write(n + '\n')
    f.write('END NETS\n')
    f.write('END DESIGN\n')
  return path
