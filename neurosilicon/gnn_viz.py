"""GNN simulation: step-through policy rollouts with real readouts.

simulate() runs one CircuitPolicy episode and records, per placement
step, the macro being placed, the policy's action distribution over the
real grid, the value estimate, and the positions placed so far. The GUI
scrubs through these steps; nothing is mocked — heatmaps are the
policy's own softmax, costs come from the terminal FD + proxy.
"""
import numpy as np
import torch

from neurosilicon import env as env_lib
from neurosilicon import model as model_lib


MAX_EDGES = 2000


def circuit_edges(plc):
  """Placeable-node connectivity: [(name_a, name_b, net_count)].

  Maps net endpoints to parent placeable nodes (hard + soft macros)
  via get_ref_node_id; parallel nets between a pair are counted,
  self-loops dropped. Capped to the MAX_EDGES strongest pairs so big
  netlists stay drawable.
  """
  mods = plc.modules_w_pins
  placeable = set(plc.get_macro_indices())
  counts = {}
  for driver, sinks in plc.nets.items():
    group = set()
    for pin_name in [driver] + list(sinks):
      idx = plc.mod_name_to_indices.get(pin_name)
      if idx is None or idx < 0 or idx >= len(mods):
        continue
      ref = plc.get_ref_node_id(idx)
      node = ref if ref != -1 else idx
      if node not in placeable:
        continue
      group.add(mods[node].get_name())
    group = sorted(group)
    for i in range(len(group)):
      for j in range(i + 1, len(group)):
        key = (group[i], group[j])
        counts[key] = counts.get(key, 0) + 1
  ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
  return [(a, b, c) for (a, b), c in ranked[:MAX_EDGES]]


@torch.no_grad()
def simulate(netlist, init_plc, ckpt_path, fd_steps=(5, 5, 5),
             greedy=True, seed=0, init_fd=True):
  """Records a full policy episode. Returns sim dict (see module docs).

  Raises:
    FileNotFoundError: checkpoint missing.
  """
  import os
  if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
  policy = model_lib.CircuitPolicy()
  policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
  policy.eval()
  rng = np.random.default_rng(seed)
  env = env_lib.ChipPlacementEnv(
      netlist, init_placement=init_plc, fd_steps=fd_steps, init_fd=init_fd)
  obs, _ = env.reset()
  rows, cols = env._rows, env._cols
  up_pad, right_pad = env.pad_offsets
  order = list(env._order)
  names = [env._plc.get_node_name(i) for i in order]
  canvas = env._plc.get_canvas_width_height()
  soft_xy = {}
  for i in env._plc.get_macro_indices():
    if env._plc.is_node_soft_macro(i):
      x, y = env._plc.get_node_location(i)
      soft_xy[env._plc.get_node_name(i)] = (x, y)

  steps, done = [], False
  info = {'infeasible': True, 'proxy_cost': -1.0}
  while not done:
    if int(np.sum(obs['mask'])) == 0:
      break
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    logits, value = policy.forward(batch)
    masked = policy.masked_logits(batch)[0]
    probs = torch.softmax(masked, dim=-1).numpy().reshape(128, 128)
    probs = probs[up_pad:up_pad + rows, right_pad:right_pad + cols]
    if greedy:
      action = int(torch.argmax(masked))
    else:
      flat = torch.softmax(masked, dim=-1).numpy()
      flat /= flat.sum()
      action = int(rng.choice(len(flat), p=flat))
    step_idx = env._step_idx
    obs, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    r, c = divmod(action, 128)
    r, c = r - up_pad, c - right_pad
    placed = []
    for k in range(min(step_idx + 1, len(order))):
      ni = order[k]
      x, y = env._plc.get_node_location(ni)
      w, h = env._plc.get_node_width_height(ni)
      placed.append({'name': names[k], 'x': x, 'y': y, 'w': w, 'h': h})
    steps.append({'macro': names[step_idx], 'node_idx': step_idx,
                  'probs': probs, 'value': float(value[0][0]),
                  'action_cell': (r, c), 'placed': placed})
  return {'steps': steps, 'final': info, 'order': names,
          'grid': (cols, rows), 'canvas': canvas,
          'edges': circuit_edges(env._plc), 'soft_xy': soft_xy}
