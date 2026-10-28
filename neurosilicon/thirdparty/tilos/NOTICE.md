# Vendored TILOS-AI-Institute/MacroPlacement code (BSD 3-Clause)

The `Plc_client/*.py` files in this directory are vendored from
[TILOS-AI-Institute/MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement)
(`CodeElements/Plc_client/`), Copyright (c) 2018-2021 The Regents of the
University of California, BSD 3-Clause License (see `LICENSE` in this
directory). It is a clean-room open-source reimplementation of Google's
`plc_client` placement-cost API and is used here because Google's
`plc_wrapper_main` binary is no longer publicly downloadable.

NeuroSilicon modifications (minimal, marked `NeuroSilicon:` in comments):
- `from Plc_client import ...` rewritten to absolute
  `neurosilicon.thirdparty.tilos.Plc_client` imports.
- Google `plc_client` imports aliased to the open-source `plc_client_os`
  (same `PlacementCost` API surface).
- `observation_config.py`: `gym` -> `gymnasium`, TensorFlow dependency
  removed (`TensorType = np.ndarray`).
