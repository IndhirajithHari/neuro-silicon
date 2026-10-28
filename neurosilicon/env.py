"""Gymnasium RL environment for sequential macro placement.

Episode = place each (non-fixed) hard macro once, in size order, onto a
placement grid. The policy observes graph + canvas features and outputs a
grid-cell action on a padded 128x128 canvas (actual grid centered, as in
AlphaChip's observation_extractor.py). At the terminal step, soft macros /
stdcells are relaxed with the force-directed placer and the reward is the
negative multi-objective proxy cost, mirroring the AlphaChip MDP:

  intermediate steps : reward 0
  infeasible action  : terminate with -infeasible_penalty
  terminal step      : reward -(w_wl*wl + w_cong*cong + w_dens*dens)
"""
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from neurosilicon import setup as setup_lib
from neurosilicon.thirdparty.tilos.Plc_client import observation_config as obs_config_lib
from neurosilicon.thirdparty.tilos.Plc_client import observation_extractor_os as obs_extractor_lib
from neurosilicon.thirdparty.tilos.Plc_client import placement_util_os as placement_util

MAX_GRID_SIZE = 128


class ChipPlacementEnv(gym.Env):
  """Sequential macro placer. Rebuilds the PLC each reset for isolation."""

  metadata = {'render_modes': ['rgb_array']}

  def __init__(self,
               netlist_file: str,
               init_placement: Optional[str] = None,
               order_mode: str = 'descending_size_macro_first',
               wirelength_weight: float = 1.0,
               congestion_weight: float = 0.5,
               density_weight: float = 1.0,
               infeasible_penalty: float = 4.0,
               canvas: Optional[Tuple[float, float]] = None,
               grid: Optional[Tuple[int, int]] = None,
               routes_per_micron: Optional[Tuple[float, float]] = None,
               fd_steps: Tuple[int, ...] = (100, 100, 100),
               extra_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
               extra_costs: Optional[list] = None,
               seed: Optional[int] = None,
               init_fd: bool = True):
    super().__init__()
    self._netlist_file = netlist_file
    self._init_placement = init_placement
    self._order_mode = order_mode
    self._w_wl = wirelength_weight
    self._w_cong = congestion_weight
    self._w_dens = density_weight
    self._infeasible_penalty = infeasible_penalty
    self._canvas = canvas
    self._grid = grid
    self._routes = routes_per_micron
    self._fd_steps = fd_steps
    self._extra_mask_fn = extra_mask_fn
    self._extra_costs = list(extra_costs) if extra_costs else []
    self._rng = np.random.default_rng(seed)
    self._init_fd = init_fd

    self._obs_config = obs_config_lib.ObservationConfig(
        max_grid_size=MAX_GRID_SIZE)
    self.observation_space = self._obs_config.observation_space
    self.action_space = gym.spaces.Discrete(MAX_GRID_SIZE * MAX_GRID_SIZE)

    self._plc = None
    self._extractor = None
    self._order: List[int] = []
    self._step_idx = 0
    self._cols = 0
    self._rows = 0

  # -- episode management -------------------------------------------------
  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    self._plc = setup_lib.create_plc(
        self._netlist_file,
        init_placement=self._init_placement,
        canvas=self._canvas,
        grid=self._grid,
        routes_per_micron=self._routes)
    self._plc.unplace_all_nodes()
    self._cols, self._rows = self._plc.get_grid_num_columns_rows()
    if self._cols > MAX_GRID_SIZE or self._rows > MAX_GRID_SIZE:
      raise ValueError(f'Grid {self._cols}x{self._rows} exceeds 128.')
    if self._init_fd:
      # Exact AlphaChip construction-time mixed-size FD (environment.py):
      # gives the observation extractor a sane initial stdcell placement.
      placement_util.fd_placement_schedule(
          self._plc,
          num_steps=(100, 100, 100),
          move_distance_factors=(1.0, 2.0, 2.0),
          attract_factor=(1.0e2, 1.0e-3, 1.0e-5),
          repel_factor=(0.0, 1.0e5, 1.0e6),
          use_current_loc=False,
          move_macros=True)
    self._order = placement_util.get_ordered_node_indices(
        self._order_mode, self._plc, exclude_fixed_nodes=True)
    # RL places hard macros; soft macros go to the FD placer at the end.
    self._order = [
        n for n in self._order if not self._plc.is_node_soft_macro(n)
    ]
    self._step_idx = 0
    self._extractor = obs_extractor_lib.ObservationExtractor(
        self._plc, self._obs_config)
    obs = self._current_obs(previous_node_index=-1)
    return obs, {}

  @property
  def plc(self):
    return self._plc

  @property
  def placement_order(self) -> List[int]:
    return list(self._order)

  # -- masking / actions ---------------------------------------------------
  def _raw_mask(self, node_idx: int) -> np.ndarray:
    mask = np.asarray(self._plc.get_node_mask(node_idx), dtype=np.int32)
    if self._extra_mask_fn is not None:
      mask = np.asarray(
          self._extra_mask_fn(node_idx, mask), dtype=np.int32)
    return mask

  @property
  def pad_offsets(self) -> Tuple[int, int]:
    """(up_pad, right_pad): centered 128x128 offsets (AlphaChip exact).

    Matches observation_extractor.py: rows_pad//2 and cols_pad//2, with
    the remainder on the low/left sides.
    """
    return ((MAX_GRID_SIZE - self._rows) // 2,
            (MAX_GRID_SIZE - self._cols) // 2)

  def _padded_mask(self, node_idx: int) -> np.ndarray:
    """Pads the RxC mask to 128x128, actual grid centered."""
    raw = self._raw_mask(node_idx).reshape(self._rows, self._cols)
    padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int32)
    up, right = self.pad_offsets
    padded[up:up + self._rows, right:right + self._cols] = raw
    return padded.reshape(-1)

  def _action_to_cell(self, action: int) -> Optional[int]:
    up, right = self.pad_offsets
    r, c = divmod(int(action), MAX_GRID_SIZE)
    r, c = r - up, c - right
    if not (0 <= r < self._rows and 0 <= c < self._cols):
      return None
    return r * self._cols + c

  # -- observations ----------------------------------------------------------
  def _current_obs(self,
                   previous_node_index: int,
                   node_position: Optional[int] = None) -> Dict[str, np.ndarray]:
    pos = self._step_idx if node_position is None else node_position
    node_idx = self._order[pos]
    mask = self._padded_mask(node_idx)
    macro_pos = self._plc.get_macro_indices().index(node_idx)
    return self._extractor.get_all_features(
        previous_node_index=(self._plc.get_macro_indices().index(
            self._order[self._step_idx - 1])
                             if previous_node_index >= 0 else -1),
        current_node_index=macro_pos,
        mask=mask)

  # -- stepping ---------------------------------------------------------------
  def step(self, action):
    node_idx = self._order[self._step_idx]
    cell = self._action_to_cell(int(action))
    legal = (cell is not None and
             self._raw_mask(node_idx).reshape(-1)[cell] == 1)
    if not legal:
      obs = self._current_obs(previous_node_index=self._step_idx - 1)
      return obs, -self._infeasible_penalty, True, False, {
          'infeasible': True,
          'wirelength': -1.0,
          'congestion': -1.0,
          'density': -1.0,
      }
    self._plc.place_node(node_idx, cell)
    self._step_idx += 1
    done = self._step_idx >= len(self._order)
    if not done:
      obs = self._current_obs(previous_node_index=self._step_idx - 1)
      if int(np.sum(obs['mask'])) == 0:
        # No valid cells remain: immediate termination with penalty.
        return obs, -self._infeasible_penalty, True, False, {
            'infeasible': True,
            'wirelength': -1.0,
            'congestion': -1.0,
            'density': -1.0,
        }
      return obs, 0.0, False, False, {}
    # Terminal: relax soft macros, then score.
    placement_util.fd_placement_schedule(
        self._plc, num_steps=self._fd_steps, use_current_loc=True)
    wl = self._plc.get_cost()
    cong = self._plc.get_congestion_cost()
    dens = self._plc.get_density_cost()
    cost = self._w_wl * wl + self._w_cong * cong + self._w_dens * dens
    info = {
        'infeasible': False,
        'wirelength': wl,
        'congestion': cong,
        'density': dens,
    }
    for name, weight, fn in self._extra_costs:
      value = float(fn(self._plc))
      cost += weight * value
      info[f'extra_{name}'] = value
    info['proxy_cost'] = cost
    obs = self._current_obs(
        previous_node_index=self._step_idx - 1,
        node_position=len(self._order) - 1)
    return obs, -cost, True, False, info

  def render(self):
    raise NotImplementedError('Use neurosilicon.render for floorplan plots.')
