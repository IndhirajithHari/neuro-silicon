"""Technology-node (PDK) presets: sky130 vs. asap7 vs. gf180mcu.

Preset switching rescales the thermal corner (voltage, default clock,
static-leakage share / die temperature offset) and the congestion
display (routing-pitch multiplier). All values are documented proxies
for interactive exploration — not signoff PDK data.
"""
from neurosilicon import liveplace as liveplace_lib

TECH_NODES = {
    'sky130': {
        'label': 'sky130 (130nm, 1.8V, 100 MHz)', 'voltage': 1.8,
        'clock_mhz': 100.0, 'leakage_factor': 1.0,
        'congestion_multiplier': 1.0, 'temp_offset_c': 0.0,
        'note': 'Reference open PDK corner.'},
    'asap7': {
        'label': 'asap7 (7nm FinFET, 0.7V, 1000 MHz — High Leakage/Density)',
        'voltage': 0.7, 'clock_mhz': 1000.0, 'leakage_factor': 3.2,
        'congestion_multiplier': 1.35, 'temp_offset_c': 18.0,
        'note': 'FinFET density: fast but hot, leakage-heavy, tight pitch.'},
    'gf180mcu': {
        'label': 'gf180mcu (180nm, 3.3V, 50 MHz)', 'voltage': 3.3,
        'clock_mhz': 50.0, 'leakage_factor': 0.6,
        'congestion_multiplier': 0.85, 'temp_offset_c': -5.0,
        'note': 'Mature MCU node: slow, cool, roomy routing.'},
}

DEFAULT_NODE = 'sky130'


def preset(node_key):
  """Preset dict for `node_key` (unknown keys fall back to sky130)."""
  return TECH_NODES.get(node_key or DEFAULT_NODE,
                        TECH_NODES[DEFAULT_NODE])


def corner_power(plc, node_key=DEFAULT_NODE, clock_mhz=None,
                 voltage=None, activity=0.2):
  """Per-macro relative power at a technology-node corner.

  Dynamic power follows the usual (V/1.8)^2 x (f/100 MHz) x activity
  scaling around the preset defaults (explicit clock/voltage win);
  static leakage adds an area-proportional share scaled by the
  node's leakage factor. Returns {macro: relative power}.
  """
  node = preset(node_key)
  clock = float(node['clock_mhz'] if clock_mhz is None else clock_mhz)
  volt = float(node['voltage'] if voltage is None else voltage)
  dyn = liveplace_lib.scaled_power(plc, clock_mhz=clock, voltage=volt,
                                   activity=activity)
  areas = {}
  for idx in plc.get_macro_indices():
    if plc.is_node_soft_macro(idx):
      continue
    w, h = plc.get_node_width_height(idx)
    areas[plc.get_node_name(idx)] = max(float(w) * float(h), 0.0)
  total = sum(areas.values())
  if total <= 0:
    return dyn
  leak_total = 0.30 * float(node['leakage_factor'])
  return {name: dyn.get(name, 0.0) + leak_total * area / total
          for name, area in areas.items()}


def frame_corner_power(frame, node_key=DEFAULT_NODE, clock_mhz=None,
                       voltage=None, activity=0.2):
  """corner_power for a liveplace frame dict (placed macros only)."""
  node = preset(node_key)
  clock = float(node['clock_mhz'] if clock_mhz is None else clock_mhz)
  volt = float(node['voltage'] if voltage is None else voltage)
  areas = liveplace_lib.frame_power(frame)
  total = sum(areas.values())
  if total <= 0:
    return {}
  dyn_scale = ((volt / 1.8) ** 2 * (clock / 100.0) * (activity / 0.2))
  leak_total = 0.30 * float(node['leakage_factor'])
  return {name: (area / total) * dyn_scale + leak_total * area / total
          for name, area in areas.items()}


def scaled_congestion(grid, node_key=DEFAULT_NODE):
  """Congestion grid scaled by the node's routing-pitch multiplier."""
  mult = float(preset(node_key)['congestion_multiplier'])
  if mult == 1.0:
    return grid
  return [[float(v) * mult for v in row] for row in (grid or [])]
