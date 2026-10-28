"""Single-process masked PPO trainer for the placement policy."""
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from neurosilicon import env as env_lib
from neurosilicon import evaluate as evaluate_lib
from neurosilicon import model as model_lib
from neurosilicon import runlog as runlog_lib
from neurosilicon.thirdparty.tilos.Plc_client import placement_util_os as placement_util


def collate(obs_list):
  """Stacks a list of obs dicts into a batched torch dict."""
  keys = obs_list[0].keys()
  return {
      k: torch.as_tensor(np.stack([o[k] for o in obs_list]))
      for k in keys
  }


@torch.no_grad()
def collect_episode(env, policy):
  """Runs one episode, recording (obs, action, logprob, value, reward)."""
  policy.eval()
  noise = getattr(policy, 'policy_noise_weight', 0.0) > 0.0
  obs, _ = env.reset()
  steps = []
  done = False
  info = {'infeasible': True}
  while not done:
    if int(np.sum(obs['mask'])) == 0:
      # Dead end: charge the penalty to the placement that caused it.
      if steps:
        steps[-1]['reward'] = -4.0
      break
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    masked = policy.masked_logits(batch, apply_noise=noise)[0]
    dist = torch.distributions.Categorical(logits=masked)
    action = int(dist.sample())
    _, value = policy(batch)
    obs2, reward, terminated, truncated, info = env.step(action)
    steps.append({
        'obs': obs,
        'action': action,
        'logprob': float(dist.log_prob(torch.tensor(action))),
        'value': float(value[0, 0]),
        'reward': float(reward),
    })
    obs = obs2
    done = terminated or truncated
  return steps, info


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
  """Advantages + returns for one episode (bootstrap value 0)."""
  adv = np.zeros(len(rewards), dtype=np.float32)
  last = 0.0
  for t in reversed(range(len(rewards))):
    v_next = values[t + 1] if t + 1 < len(values) else 0.0
    delta = rewards[t] + gamma * v_next - values[t]
    last = delta + gamma * lam * last
    adv[t] = last
  return adv, adv + np.asarray(values, dtype=np.float32)


def ppo_update(policy, optimizer, flat_steps, advantages, returns,
               epochs=4, batch_size=128, clip=0.2, vf_coef=0.5,
               ent_coef=0.01, max_grad_norm=1.0):
  """Runs PPO epochs over flattened steps. Returns mean losses.

  Minibatch size 128, grad-clip 1.0, clip 0.2, vf 0.5, entropy 0.01
  match AlphaChip (train_ppo_lib.py, agent.py).
  """
  obs_batch = collate([s['obs'] for s in flat_steps])
  actions = torch.as_tensor([s['action'] for s in flat_steps])
  old_logp = torch.as_tensor([s['logprob'] for s in flat_steps])
  adv = torch.as_tensor(advantages)
  adv = (adv - adv.mean()) / (adv.std() + 1e-8)
  ret = torch.as_tensor(returns)
  n = len(flat_steps)
  idx = np.arange(n)
  tot_pg = tot_vf = tot_ent = 0.0
  count = 0
  policy.train()
  num_chunks = max(1, (n + batch_size - 1) // batch_size)
  for _ in range(epochs):
    np.random.shuffle(idx)
    for chunk in np.array_split(idx, num_chunks):
      b = {k: v[torch.as_tensor(chunk)] for k, v in obs_batch.items()}
      logits = policy.masked_logits(b)
      dist = torch.distributions.Categorical(logits=logits)
      new_logp = dist.log_prob(actions[torch.as_tensor(chunk)])
      ratio = torch.exp(new_logp - old_logp[torch.as_tensor(chunk)])
      a = adv[torch.as_tensor(chunk)]
      pg_loss = -torch.mean(
          torch.min(ratio * a,
                    torch.clamp(ratio, 1 - clip, 1 + clip) * a))
      _, value = policy(b)
      value = value.squeeze(-1)
      r = ret[torch.as_tensor(chunk)]
      vf_loss = F.mse_loss(value, r)
      ent = dist.entropy().mean()
      loss = pg_loss + vf_coef * vf_loss - ent_coef * ent
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
      optimizer.step()
      tot_pg += float(pg_loss)
      tot_vf += float(vf_loss)
      tot_ent += float(ent)
      count += 1
  return tot_pg / count, tot_vf / count, tot_ent / count


def cosine_lr(base_lr, step, decay_steps, alpha=0.1):
  """TF CosineDecay math: alpha + (1-alpha)*0.5*(1+cos(pi*step/decay))."""
  import math
  step = min(step, decay_steps)
  return float(alpha + (1.0 - alpha) * 0.5 *
               (1.0 + math.cos(math.pi * step / max(decay_steps, 1)))) * base_lr


def train(netlist_files,
          init_placements=None,
          outdir='logs/train',
          iterations=10,
          episodes_per_iter=4,
          eval_every=2,
          fd_steps=(100, 100, 100),
          lr=0.004,
          seed=0,
          resume=None,
          broadcast_weight=0.0,
          broadcast_threshold=16,
          dirichlet_weight=0.0,
          dirichlet_alpha=0.1,
          init_fd=True):
  """Trains and checkpoints. Returns (history rows, best eval cost).

  `netlist_files` may be one path or a list; multiple netlists are
  trained jointly (round-robin episodes, shared policy). LR schedule
  (0.004 cosine-decayed to 10%), Adam epsilon 1e-5, and dirichlet
  defaults match AlphaChip (train_ppo_lib.py, model_lib.py).
  """
  if isinstance(netlist_files, str):
    netlist_files = [netlist_files]
  if init_placements is None:
    init_placements = [None] * len(netlist_files)
  elif isinstance(init_placements, str):
    init_placements = [init_placements] * len(netlist_files)
  assert len(init_placements) == len(netlist_files)

  extra_costs = None
  if broadcast_weight > 0:
    from neurosilicon import priors as priors_lib
    extra_costs = [('broadcast', broadcast_weight,
                    lambda plc: priors_lib.broadcast_penalty(
                        plc, fanout_threshold=broadcast_threshold))]

  def make_env(i):
    return env_lib.ChipPlacementEnv(
        netlist_files[i % len(netlist_files)],
        init_placement=init_placements[i % len(netlist_files)],
        fd_steps=fd_steps,
        extra_costs=extra_costs,
        init_fd=init_fd)

  os.makedirs(outdir, exist_ok=True)
  log = runlog_lib.get_logger('train', log_dir=outdir)
  runlog_lib.write_manifest(outdir, 'train', {
      'netlists': list(netlist_files), 'init_placements': init_placements,
      'iterations': iterations, 'episodes_per_iter': episodes_per_iter,
      'eval_every': eval_every, 'fd_steps': list(fd_steps), 'lr': lr,
      'dirichlet_weight': dirichlet_weight,
      'dirichlet_alpha': dirichlet_alpha, 'init_fd': init_fd,
      'seed': seed, 'resume': resume,
      'broadcast_weight': broadcast_weight,
      'broadcast_threshold': broadcast_threshold})
  torch.manual_seed(seed)
  np.random.seed(seed)
  policy = model_lib.CircuitPolicy(
      dirichlet_alpha=dirichlet_alpha,
      policy_noise_weight=dirichlet_weight)
  if resume and os.path.isfile(resume):
    policy.load_state_dict(torch.load(resume, map_location='cpu'))
    log.info('resumed from %s', resume)
  optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
  history = []
  best_cost = float('inf')
  csv_path = os.path.join(outdir, 'history.csv')
  with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(
        ['iter', 'mean_reward', 'eval_cost', 'pg_loss', 'vf_loss',
         'entropy', 'seconds'])
  for it in range(iterations):
    t0 = time.time()
    for group in optimizer.param_groups:
      group['lr'] = cosine_lr(lr, it, iterations)
    flat, advs, rets, rewards = [], [], [], []
    for ep in range(episodes_per_iter):
      env = make_env(ep)
      log.info('iter=%d ep=%d net=%d start', it, ep,
               ep % len(netlist_files))
      steps, info = collect_episode(env, policy)
      log.info('iter=%d ep=%d done steps=%d', it, ep, len(steps))
      r = [s['reward'] for s in steps]
      v = [s['value'] for s in steps]
      adv, ret = compute_gae(r, v)
      if not steps:
        continue
      flat += steps
      advs.append(adv)
      rets.append(ret)
      rewards.append(sum(r))
    if not flat:
      log.warning('iter=%d no steps collected; skipping update.', it)
      continue
    pg_loss, vf_loss, ent = ppo_update(
        policy, optimizer, flat,
        np.concatenate(advs), np.concatenate(rets))
    eval_cost = float('nan')
    if (it + 1) % eval_every == 0 or it == iterations - 1:
      costs = []
      for i in range(len(netlist_files)):
        env = make_env(i)
        _, info = evaluate_lib.rollout_policy(env, policy, greedy=True)
        c = float(info.get('proxy_cost', float('nan')))
        if c and c > 0:
          costs.append(c)
          placement_util.save_placement(
              env.plc, os.path.join(outdir, f'best_policy_{i}.plc'),
              user_comments=f'iter={it} cost={c:.4f}')
      if costs:
        eval_cost = float(np.mean(costs))
      if eval_cost and eval_cost < best_cost:
        best_cost = eval_cost
        torch.save(policy.state_dict(), os.path.join(outdir, 'best.pt'))
    torch.save(policy.state_dict(), os.path.join(outdir, 'latest.pt'))
    row = [it, float(np.mean(rewards)), eval_cost, pg_loss, vf_loss, ent,
           round(time.time() - t0, 1)]
    history.append(row)
    with open(csv_path, 'a', newline='') as f:
      csv.writer(f).writerow(row)
    log.info('iter=%d mean_reward=%.4f eval_cost=%.4f pg=%.4f '
               'vf=%.4f ent=%.4f (%ss)', it, row[1], eval_cost,
               pg_loss, vf_loss, ent, row[6])
  return history, best_cost


def main():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('--netlist', required=True, action='append')
  p.add_argument('--init_placement', default=None, action='append')
  p.add_argument('--outdir', default='logs/train')
  p.add_argument('--iterations', type=int, default=10)
  p.add_argument('--episodes_per_iter', type=int, default=4)
  p.add_argument('--eval_every', type=int, default=2)
  p.add_argument('--fd_steps', type=int, default=100)
  p.add_argument('--no_init_fd', action='store_true',
                 help='Skip construction-time mixed-size FD (faster).')
  p.add_argument('--lr', type=float, default=0.004)
  p.add_argument('--dirichlet_weight', type=float, default=0.0)
  p.add_argument('--resume', default=None)
  p.add_argument('--broadcast_weight', type=float, default=0.0)
  p.add_argument('--broadcast_threshold', type=int, default=16)
  args = p.parse_args()
  train(args.netlist, args.init_placement, args.outdir, args.iterations,
        args.episodes_per_iter, args.eval_every,
        fd_steps=(args.fd_steps,) * 3, init_fd=not args.no_init_fd,
        lr=args.lr,
        dirichlet_weight=args.dirichlet_weight, resume=args.resume,
        broadcast_weight=args.broadcast_weight,
        broadcast_threshold=args.broadcast_threshold)


if __name__ == '__main__':
  main()
