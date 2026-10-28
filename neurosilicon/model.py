"""PyTorch port of AlphaChip's edge-centric GCN policy + value network.

Mirrors `circuit_training/model/model_lib.py:CircuitTrainingModel`:
node/edge embeddings -> edge-FC message passing with residual adds ->
global edge statistics + current-node attention -> deconv policy head
over the 128x128 canvas + MLP value head. Action masking is applied to
the logits by the caller (or via `masked_logits`).
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurosilicon.thirdparty.tilos.Plc_client import observation_config as obs_config_lib

EPSILON = 1e-6
# Exact AlphaChip illegal-action logit: -(2**32)+1
# (circuit_training/model/model.py CircuitTrainingModel.call).
MASK_LOGIT = -(2.0**32) + 1
# Exact AlphaChip dirichlet defaults (model_lib.py): concentration 0.1,
# noise weight 0.0 (exploration noise off unless training enables it).
DIRICHLET_ALPHA = 0.1
POLICY_NOISE_WEIGHT = 0.0


class EdgeFC(nn.Module):
  """Edge mixing MLP: [h_i, h_j, w] -> h."""

  def __init__(self, node_dim: int, num_layers: int = 1):
    super().__init__()
    layers = []
    in_dim = 2 * node_dim + 1
    for _ in range(num_layers):
      layers += [nn.Linear(in_dim, node_dim), nn.ReLU()]
      in_dim = node_dim
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class CircuitPolicy(nn.Module):
  """Edge-GNN actor-critic over padded graph observations."""

  def __init__(self,
               node_dim: int = 8,
               num_gcn_layers: int = 3,
               edge_fc_layers: int = 1,
               include_min_max_var: bool = True,
               dirichlet_alpha: float = DIRICHLET_ALPHA,
               policy_noise_weight: float = POLICY_NOISE_WEIGHT):
    super().__init__()
    self.node_dim = node_dim
    self.num_gcn_layers = num_gcn_layers
    self.include_min_max_var = include_min_max_var
    self.dirichlet_alpha = dirichlet_alpha
    self.policy_noise_weight = policy_noise_weight
    num_metadata = len(obs_config_lib.NETLIST_METADATA)

    self.metadata_encoder = nn.Sequential(
        nn.Linear(num_metadata, node_dim), nn.ReLU())
    self.feature_encoder = nn.Sequential(
        nn.Linear(8, node_dim), nn.ReLU())
    self.edge_fcs = nn.ModuleList(
        [EdgeFC(node_dim, edge_fc_layers) for _ in range(num_gcn_layers)])
    self.attn_q = nn.Linear(node_dim, node_dim)
    self.attn_k = nn.Linear(node_dim, node_dim)
    self.attn_v = nn.Linear(node_dim, node_dim)

    latent_dim = node_dim * (7 if include_min_max_var else 4)
    self.value_head = nn.Sequential(
        nn.Linear(latent_dim, 32), nn.ReLU(),
        nn.Linear(32, 8), nn.ReLU(),
        nn.Linear(8, 1))
    self.policy_head = nn.Sequential(
        nn.Linear(latent_dim, 8 * 8 * 32), nn.ReLU(),
        _Reshape(-1, 32, 8, 8),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                           padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                           padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2,
                           padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2,
                           padding=1, output_padding=1), nn.ReLU(),
        nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
    )
    self.apply(_glorot_init)

  def _gather_to_edges(self, h_nodes, adj_i, adj_j, adj_w):
    """Returns (i->j, j->i) edge embeddings, zero where weight is 0."""
    def pick(idx):
      return h_nodes.gather(
          1, idx.unsqueeze(-1).expand(-1, -1, h_nodes.shape[-1]))

    h_i, h_j = pick(adj_i), pick(adj_j)
    w = adj_w.unsqueeze(-1)
    e_ij = torch.cat([h_i, h_j, w], dim=-1)
    e_ji = torch.cat([h_j, h_i, w], dim=-1)
    keep = (adj_w != 0).unsqueeze(-1).expand_as(e_ij)
    return (torch.where(keep, e_ij, torch.zeros_like(e_ij)),
            torch.where(keep, e_ji, torch.zeros_like(e_ji)))

  def _scatter_to_nodes(self, h_edges, adj_i, adj_j, num_nodes):
    """Mean-aggregates edge embeddings onto endpoint nodes."""
    batch, num_edges, dim = h_edges.shape
    out = torch.zeros(batch, num_nodes, dim,
                      dtype=h_edges.dtype, device=h_edges.device)
    cnt = torch.zeros(batch, num_nodes, dim,
                      dtype=h_edges.dtype, device=h_edges.device)
    ones = torch.ones_like(h_edges)
    for b in range(batch):
      out[b].index_add_(0, adj_i[b], h_edges[b])
      out[b].index_add_(0, adj_j[b], h_edges[b])
      cnt[b].index_add_(0, adj_i[b], ones[b])
      cnt[b].index_add_(0, adj_j[b], ones[b])
    return out / (cnt + EPSILON)

  def _encode(self, obs: Dict[str, torch.Tensor]):
    meta = torch.cat([obs[k].float() for k in obs_config_lib.NETLIST_METADATA],
                     dim=1)
    h_meta = self.metadata_encoder(meta)

    node_types = obs['node_types']
    h_nodes = torch.stack([
        obs['locations_x'].float(),
        obs['locations_y'].float(),
        obs['macros_w'].float(),
        obs['macros_h'].float(),
        (node_types == obs_config_lib.HARD_MACRO).float(),
        (node_types == obs_config_lib.SOFT_MACRO).float(),
        (node_types == obs_config_lib.PORT_CLUSTER).float(),
        obs['is_node_placed'].float(),
    ], dim=2)
    h_nodes = self.feature_encoder(h_nodes)

    adj_i = obs['sparse_adj_i'].long()
    adj_j = obs['sparse_adj_j'].long()
    adj_w = obs['sparse_adj_weight'].float()
    num_nodes = h_nodes.shape[1]
    h_edges = torch.zeros_like(h_nodes[:, :adj_i.shape[1], :])
    for layer in self.edge_fcs:
      e_ij, e_ji = self._gather_to_edges(h_nodes, adj_i, adj_j, adj_w)
      h_edges = (layer(e_ij) + layer(e_ji)) / 2.0
      h_nodes = self._scatter_to_nodes(h_edges, adj_i, adj_j, num_nodes) + h_nodes

    parts = [h_meta, h_edges.mean(dim=1)]
    if self.include_min_max_var:
      parts += [
          h_edges.var(dim=1, unbiased=False),
          h_edges.amax(dim=1),
          h_edges.amin(dim=1),
      ]
    cur = obs['current_node'].long().view(-1, 1)
    h_cur = h_nodes.gather(1, cur.unsqueeze(-1).expand(-1, -1, self.node_dim))
    q = self.attn_q(h_cur)
    k = self.attn_k(h_nodes)
    v = self.attn_v(h_nodes)
    attn = torch.softmax(q @ k.transpose(1, 2), dim=-1) @ v
    parts += [attn.squeeze(1), h_cur.squeeze(1)]
    return torch.cat(parts, dim=1)

  def forward(self, obs: Dict[str, torch.Tensor]):
    """Returns (location_logits [B, 16384], value [B, 1])."""
    h = self._encode(obs)
    return self.policy_head(h), self.value_head(h)

  def add_noise(self, logits: torch.Tensor) -> torch.Tensor:
    """Dirichlet exploration noise, exact AlphaChip math (model_lib.py).

    noised = log((1-w)*softmax(logits) + w*Dirichlet(alpha) + eps).
    With the default weight 0.0 this is a no-op.
    """
    if self.policy_noise_weight == 0.0:
      return logits
    probs = torch.softmax(logits, dim=-1)
    alphas = torch.full_like(probs, self.dirichlet_alpha)
    noise = torch.distributions.Dirichlet(alphas).sample()
    noised = ((1.0 - self.policy_noise_weight) * probs +
              self.policy_noise_weight * noise)
    return torch.log(noised + EPSILON)

  def masked_logits(self, obs: Dict[str, torch.Tensor],
                    apply_noise: bool = False) -> torch.Tensor:
    logits, _ = self.forward(obs)
    if apply_noise:
      logits = self.add_noise(logits)
    mask = obs['mask'].float()
    return torch.where(mask.bool(), logits,
                       torch.full_like(logits, MASK_LOGIT))


def _glorot_init(module):
  """Glorot-normal kernel init, exact AlphaChip default (model_lib.py).

  Applied to every Linear/ConvTranspose2d; biases zeroed. Only affects
  freshly constructed policies, never loaded checkpoints.
  """
  if isinstance(module, (nn.Linear, nn.ConvTranspose2d)):
    nn.init.xavier_normal_(module.weight)
    if module.bias is not None:
      nn.init.zeros_(module.bias)


class _Reshape(nn.Module):

  def __init__(self, *shape):
    super().__init__()
    self.shape = shape

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.view(*self.shape)
