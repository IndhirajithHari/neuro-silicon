# Changelog

All notable changes to the runnable NeuroSilicon system (`neurosilicon/`).
The restored AlphaChip reference in `circuit_training/` is versioned
upstream and unchanged.

## [Unreleased] — desktop workstation upgrade

- Desktop (PySide6): native real-time floorplan canvas (QPainter)
  replacing PNG frames — AlphaChip-style sequential playback with
  Play/Pause, step, scrub and speed controls; Floorplan / Congestion /
  Thermal / Wires views; live metric cards (proxy, wirelength,
  congestion, macros placed).
- Auto-import: select a whole chip design (folder, any file inside
  it, or a `.zip`; drag-and-drop supported) and `neurosilicon/discover`
  fetches what floorplanning needs — ready netlist pair or RTL with
  auto-picked top — with a found/missing checklist and import plan.
- AI Connectors tab (optional cloud API + local Ollama/LM Studio with
  Test connection), an in-app setup Guide, and an MCP-style chatbox
  (`neurosilicon/assistant`) that reads the floorplan through tool
  results, works offline, and turns chat constraints into engine
  keep-out masks.
- Downloads: placed `.plc`, viewer DEF, floorplan/thermal/congestion
  PNGs, and a report ZIP from Results & Downloads.
- New headless engines: `neurosilicon/liveplace` (sequential rollout
  snapshots, net segments, scaled thermal power) and an
  `extra_mask_fn` hook through `evaluate.run_baselines`.

## [1.0.0] — industrial console release

- Dark industrial console theme with a Dashboard landing page (system
  status, designs, recent runs), consistent headers/footers, workflow
  stepper, and status pills.
- Every finished GUI run persists to `logs/gui_runs/runs/<run_id>/`
  (run.json + best.plc + mask) with auto-restore and a run picker.
- AlphaChip-parity diagnostics: top-10% congestion metric, placed
  macro utilization, keep-out mask overlay, GNN placement-order trace.
- Structured logging (`neurosilicon/runlog.py`) and `manifest.json`
  (tool, version, git hash, args) in every train/evaluate/import
  output directory.
- Repo-root-anchored tool paths (yosys/OpenROAD/TCL/registry) so the
  console runs from any working directory.
- External API + local-LLM providers with secrets outside the repo,
  per-design verified checkpoints, OpenROAD global-route validation,
  thermal corner lab, and full-report ZIP bundles.
- CI: pytest suite on every push (`.github/workflows/ci.yml`).
