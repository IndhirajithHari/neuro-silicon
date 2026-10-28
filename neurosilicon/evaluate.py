"""Rollouts, baselines, and results tables for NeuroSilicon placements."""
import json
import os
import time

import numpy as np
import torch

from neurosilicon import env as env_lib
from neurosilicon.thirdparty.tilos.Plc_client import placement_util_os as placement_util


def rollout_random(env, seed=None):
  """Uniform-random rollout over legal cells. Returns (reward, info)."""
  rng = np.random.default_rng(seed)
  obs, _ = env.reset()
  done = False
  reward, info = -4.0, {'infeasible': True, 'proxy_cost': -1.0,
                        'wirelength': -1.0, 'congestion': -1.0,
                        'density': -1.0}
  while not done:
    legal = np.flatnonzero(obs['mask'])
    if len(legal) == 0:
      break
    obs, reward, terminated, truncated, info = env.step(
        int(rng.choice(legal)))
    done = terminated or truncated
  return reward, info


def rollout_first_legal(env):
  """Deterministic first-legal-cell rollout. Returns (reward, info)."""
  obs, _ = env.reset()
  done = False
  reward, info = -4.0, {'infeasible': True, 'proxy_cost': -1.0,
                        'wirelength': -1.0, 'congestion': -1.0,
                        'density': -1.0}
  while not done:
    legal = np.flatnonzero(obs['mask'])
    if len(legal) == 0:
      break
    obs, reward, terminated, truncated, info = env.step(int(legal[0]))
    done = terminated or truncated
  return reward, info


@torch.no_grad()
def rollout_policy(env, policy, greedy=False, seed=None):
  """Rolls out a CircuitPolicy. Returns (reward, info)."""
  policy.eval()
  rng = np.random.default_rng(seed)
  obs, _ = env.reset()
  done = False
  reward, info = -4.0, {'infeasible': True, 'proxy_cost': -1.0,
                        'wirelength': -1.0, 'congestion': -1.0,
                        'density': -1.0}
  while not done:
    if int(np.sum(obs['mask'])) == 0:
      break
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    masked = policy.masked_logits(batch)[0]
    if greedy:
      action = int(torch.argmax(masked))
    else:
      probs = torch.softmax(masked, dim=-1).numpy()
      probs /= probs.sum()
      action = int(rng.choice(len(probs), p=probs))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
  return reward, info


def _new_env(netlist_file, init_placement, fd_steps, extra_costs=None,
             init_fd=True, extra_mask_fn=None):
  return env_lib.ChipPlacementEnv(
      netlist_file, init_placement=init_placement, fd_steps=fd_steps,
      extra_costs=extra_costs, init_fd=init_fd,
      extra_mask_fn=extra_mask_fn)


def _broadcast_costs(weight, threshold):
  if weight <= 0:
    return None
  from neurosilicon import priors as priors_lib
  return [('broadcast', weight,
           lambda plc: priors_lib.broadcast_penalty(
               plc, fanout_threshold=threshold))]


def run_baselines(netlist_file,
                  init_placement=None,
                  episodes=5,
                  fd_steps=(100, 100, 100),
                  outdir='logs/eval',
                  seed=0,
                  finetune_epochs=0,
                  broadcast_weight=0.0,
                  broadcast_threshold=16,
                  init_fd=True,
                  progress=None,
                  extra_mask_fn=None):
  """Runs random + first-legal baselines; saves best .plc and results.json.

  `progress`, when given, is called with each finished episode row
  (GUI live tables) before the final results.json is written.
  """
  from neurosilicon import runlog as runlog_lib
  os.makedirs(outdir, exist_ok=True)
  runlog_lib.get_logger('evaluate', log_dir=outdir)
  runlog_lib.write_manifest(outdir, 'evaluate', {
      'netlist': netlist_file, 'init_placement': init_placement,
      'episodes': episodes, 'fd_steps': list(fd_steps), 'seed': seed,
      'finetune_epochs': finetune_epochs,
      'broadcast_weight': broadcast_weight,
      'broadcast_threshold': broadcast_threshold, 'init_fd': init_fd})
  results = []
  best_cost = float('inf')
  extra_costs = _broadcast_costs(broadcast_weight, broadcast_threshold)
  for name, fn in (('random', rollout_random),
                   ('first_legal', rollout_first_legal)):
    for ep in range(episodes):
      env = _new_env(netlist_file, init_placement, fd_steps, extra_costs,
                     init_fd, extra_mask_fn)
      t0 = time.time()
      reward, info = fn(env, seed=seed + ep) if name == 'random' else fn(env)
      if finetune_epochs > 0 and not info.get('infeasible', True):
        from neurosilicon import finetune as finetune_lib
        cost, parts = finetune_lib.run_cd(
            env.plc, epochs=finetune_epochs,
            broadcast_weight=broadcast_weight,
            broadcast_threshold=broadcast_threshold)
        reward = -cost
        info = {'infeasible': False, 'proxy_cost': cost, **parts}
      secs = time.time() - t0
      row = {
          'method': name,
          'episode': ep,
          'reward': float(reward),
          'proxy_cost': float(info.get('proxy_cost', -1)),
          'wirelength': float(info.get('wirelength', -1)),
          'congestion': float(info.get('congestion', -1)),
          'density': float(info.get('density', -1)),
          'seconds': round(secs, 2),
      }
      results.append(row)
      if progress is not None:
        progress(row)
      cost = row['proxy_cost']
      if 0 < cost < best_cost:
        best_cost = cost
        placement_util.save_placement(
            env.plc,
            os.path.join(outdir, 'best_baseline.plc'),
            user_comments=f'Best baseline: {name} ep={ep} cost={cost:.4f}')
  with open(os.path.join(outdir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
  return results


def print_table(results):
  print(f"{'method':<12}{'ep':<4}{'proxy_cost':<12}{'wl':<10}"
        f"{'cong':<10}{'dens':<10}{'secs':<7}")
  for r in results:
    print(f"{r['method']:<12}{r['episode']:<4}{r['proxy_cost']:<12.4f}"
          f"{r['wirelength']:<10.4f}{r['congestion']:<10.4f}"
          f"{r['density']:<10.4f}{r['seconds']:<7}")


def main():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('--netlist', required=True)
  p.add_argument('--init_placement', default=None)
  p.add_argument('--episodes', type=int, default=5)
  p.add_argument('--outdir', default='logs/eval')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--finetune_epochs', type=int, default=0)
  p.add_argument('--fd_steps', type=int, default=100)
  p.add_argument('--no_init_fd', action='store_true',
                 help='Skip construction-time mixed-size FD (faster).')
  p.add_argument('--broadcast_weight', type=float, default=0.0)
  p.add_argument('--broadcast_threshold', type=int, default=16)
  args = p.parse_args()
  results = run_baselines(args.netlist, args.init_placement, args.episodes,
                          outdir=args.outdir, seed=args.seed,
                          finetune_epochs=args.finetune_epochs,
                          fd_steps=(args.fd_steps,) * 3,
                          broadcast_weight=args.broadcast_weight,
                          broadcast_threshold=args.broadcast_threshold,
                          init_fd=not args.no_init_fd)
  print_table(results)


if __name__ == '__main__':
  main()
