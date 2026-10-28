"""Viewer-grade DEF export for KLayout / OpenROAD inspection.

Emits DIEAREA + COMPONENTS (+ PORT pins) with placed coordinates. NETS are
emitted on a best-effort basis; this file is meant for visual checking and
downstream experimentation, not signoff.
"""
import os


def _d(val):
  return int(round(val * 1000))


_ORIENT_R = {0: 'R0', 90: 'R90', 180: 'R180', 270: 'R270'}


def openroad_tcl_text(plc, orientations=None, design=None):
  """OpenROAD Tcl macro-placement script for the current placement.

  One `place_macro` per hard macro with its lower-left location in
  microns (converted from the stored centers, rotation-aware) and
  R0/R90/R180/R270 orientation. `orientations`: {macro: degrees}
  overrides from the orientation optimizer win over the plc label.
  """
  design = design or plc.get_block_name() or 'neurosilicon'
  w, h = plc.get_canvas_width_height()
  lines = ['# NeuroSilicon — OpenROAD macro placement',
           f'# Design: {design}  Die: {w:.2f} x {h:.2f} um',
           '# Locations are macro lower-left corners in microns.',
           '# Load your LEF/Liberty first, e.g.:',
           '#   read_lef <tech.lef>; read_liberty <liberty>; '
           'read_def placed.def',
           '']
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    name = plc.get_node_name(idx)
    x, y = plc.get_node_location(idx)
    mw, mh = plc.get_node_width_height(idx)
    deg = (orientations or {}).get(name)
    if deg is None:
      try:
        deg = {'N': 0, 'E': 90, 'S': 180, 'W': 270}.get(
            str(plc.get_macro_orientation(idx)).lstrip('F'), 0)
      except Exception:
        deg = 0
    deg = int(deg) % 360
    ew, eh = (mh, mw) if deg in (90, 270) else (mw, mh)
    lines.append(
        f'place_macro -macro_name {name} -location '
        f'{{{x - ew / 2:.2f} {y - eh / 2:.2f}}} '
        f'-orientation {_ORIENT_R[deg]}')
  lines.append('')
  lines.append('# End of NeuroSilicon macro placement')
  return '\n'.join(lines) + '\n'


def export_openroad_tcl(plc, path, orientations=None, design=None):
  """Writes openroad_tcl_text() to `path`. Returns the path."""
  with open(path, 'w') as f:
    f.write(openroad_tcl_text(plc, orientations, design))
  return path


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
