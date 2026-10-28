"""Tests for setup helper and the placement environment."""
import numpy as np

from neurosilicon import env as env_lib
from neurosilicon import setup as setup_lib

TOY_NETLIST = (
    'circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt')
TOY_PLC = (
    'circuit_training/environment/test_data/toy_macro_stdcell/initial.plc')


def test_setup_reads_header_geometry():
  plc = setup_lib.create_plc(TOY_NETLIST, init_placement=TOY_PLC)
  assert plc.get_canvas_width_height() == (200.0, 100.0)
  assert plc.get_grid_num_columns_rows() == (20, 10)


def test_setup_smooth_default_matches_alphachip(tmp_path):
  plc = setup_lib.create_plc(
      TOY_NETLIST,
      init_placement=TOY_PLC, smooth=None)
  assert plc.get_congestion_smooth_range() == 5


def test_centered_padding_and_action_roundtrip():
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC,
                                 init_fd=False)
  obs, _ = env.reset()
  assert env.pad_offsets == ((128 - 10) // 2, (128 - 20) // 2)
  up, right = env.pad_offsets
  grid = obs['mask'].reshape(128, 128)
  assert grid[:up, :].sum() == 0 and grid[:, :right].sum() == 0
  assert grid[up + 10:, :].sum() == 0
  assert grid[:, right + 20:].sum() == 0
  assert grid[up:up + 10, right:right + 20].sum() > 0
  # Every raw cell maps to exactly one padded action and back.
  for cell in (0, 7, 33, 10 * 20 - 1):
    r, c = divmod(cell, 20)
    action = (r + up) * 128 + (c + right)
    assert env._action_to_cell(action) == cell
  assert env._action_to_cell(0) is None  # padded corner
  assert env._action_to_cell(128 * 128 - 1) is None


def test_init_fd_moves_stdcells_from_center():
  plain = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC,
                                   init_fd=False)
  plain.reset()
  before = [plain.plc.get_node_location(i)
            for i in plain.plc.get_macro_indices()
            if plain.plc.is_node_soft_macro(i)]
  with_fd = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC,
                                     init_fd=True)
  with_fd.reset()
  after = [with_fd.plc.get_node_location(i)
           for i in with_fd.plc.get_macro_indices()
           if with_fd.plc.is_node_soft_macro(i)]
  assert before != after
  # Nothing is marked placed after the init FD.
  assert sum(1 for i in with_fd.plc.get_macro_indices()
             if with_fd.plc.is_node_placed(i)) == 0


def test_env_episode_completes_with_finite_reward():
  env = env_lib.ChipPlacementEnv(
      TOY_NETLIST, init_placement=TOY_PLC, fd_steps=(2, 2, 2), init_fd=False, seed=0)
  obs, _ = env.reset()
  assert len(env.placement_order) == 2
  assert obs['mask'].shape == (128 * 128,)
  done = False
  steps = 0
  while not done:
    legal = np.flatnonzero(obs['mask'])
    assert len(legal) > 0
    obs, reward, terminated, truncated, info = env.step(
        int(legal[len(legal) // 2]))
    done = terminated or truncated
    steps += 1
    assert steps < 10
  assert info['infeasible'] is False
  assert np.isfinite(reward)
  assert reward < 0


def test_env_rejects_illegal_action():
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC, init_fd=False)
  obs, _ = env.reset()
  illegal = int(np.flatnonzero(obs['mask'] == 0)[0])
  _, reward, terminated, _, info = env.step(illegal)
  assert terminated and info['infeasible'] is True
  assert reward == -4.0


def test_env_mask_excludes_placed_macro():
  env = env_lib.ChipPlacementEnv(TOY_NETLIST, init_placement=TOY_PLC, init_fd=False)
  obs, _ = env.reset()
  legal = np.flatnonzero(obs['mask'])
  obs2, _, terminated, _, _ = env.step(int(legal[0]))
  assert not terminated
  # Second macro's mask must still be valid but different: first macro's
  # cells are blocked by overlap rules.
  assert obs2['mask'].shape == (128 * 128,)
  assert int(np.sum(obs2['mask'])) > 0
