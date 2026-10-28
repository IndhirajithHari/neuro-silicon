"""Coordinate-descent fine-tuning of a finished placement."""
from typing import Dict, Tuple

from neurosilicon.thirdparty.tilos.Plc_client import coordinate_descent_placer as cd_lib


def proxy_cost_fn(wirelength_weight=1.0, congestion_weight=0.5,
                  density_weight=1.0, broadcast_weight=0.0,
                  broadcast_threshold=16):
  """Builds a (cost, info) callable over a PlacementCost."""

  def fn(plc):
    wl = plc.get_cost()
    cong = plc.get_congestion_cost()
    dens = plc.get_density_cost()
    cost = (wirelength_weight * wl + congestion_weight * cong +
            density_weight * dens)
    info = {'wirelength': wl, 'congestion': cong, 'density': dens}
    if broadcast_weight > 0:
      from neurosilicon import priors as priors_lib
      b = priors_lib.broadcast_penalty(plc,
                                       fanout_threshold=broadcast_threshold)
      cost += broadcast_weight * b
      info['broadcast'] = b
    return cost, info

  return fn


def run_cd(plc,
           epochs: int = 2,
           k_distance_bound: int = 5,
           optimize_only_orientation: bool = False,
           use_stdcell_placer: bool = True,
           stdcell_placer: str = 'fd',
           **cost_weights) -> Tuple[float, Dict[str, float]]:
  """Greedy local search over macro positions/orientations in place.

  use_stdcell_placer=True matches AlphaChip's CD default (fd schedule
  with use_current_loc between passes; dreamplace is unavailable in
  this native stack so fd is the placer).
  """
  placer = cd_lib.CoordinateDescentPlacer(
      plc,
      cost_fn=proxy_cost_fn(**cost_weights),
      epochs=epochs,
      use_stdcell_placer=use_stdcell_placer,
      stdcell_placer=stdcell_placer,
      k_distance_bound=k_distance_bound,
      optimize_only_orientation=optimize_only_orientation)
  placer.place()
  return proxy_cost_fn(**cost_weights)(plc)
