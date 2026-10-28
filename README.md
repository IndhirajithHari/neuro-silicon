# NeuroSilicon — AI Floorplanner

NeuroSilicon is an AI chip floorplanner: it places macros sequentially
with an edge-GNN policy trained by PPO, scored by an open-source
proxy-cost engine (wirelength + congestion + density) with
force-directed standard-cell relaxation. It runs as a native desktop
GUI (primary) or a Streamlit web console, plus CLI tools for baselines
and training. No TensorFlow, no proprietary binaries, CPU-friendly.

The runnable system is the `neurosilicon/` package. The restored
upstream AlphaChip reference implementation is retained under
`circuit_training/` for provenance (see [Upstream](#upstream-alphachip)
below); day-to-day work uses `neurosilicon/` only.

---

## Requirements

| Need | Details |
| --- | --- |
| OS | Linux (developed/tested on Fedora/Ubuntu-class desktops) |
| Python | 3.12 or newer (CI uses 3.12; the project `.venv` uses 3.14) |
| Display | X11 or Wayland session for the desktop GUI |
| GPU | Not required — CPU PyTorch build is enough |
| Optional tools | `yosys` (RTL synthesis on Import) and `openroad` (routed validation); paths are configurable in Settings |
| AI provider | Optional. Off by default; any OpenAI-compatible API or a local Ollama / LM Studio server |

Everything else — placement engine, SAM hotspot segmentation, thermal
and congestion maps, timing/Fmax estimates — is computed locally.
Every number is a documented proxy for interactive exploration, not
signoff timing, power, or routing.

## Setup

Run everything from the repository root. The environment belongs to
the project: create it as `.venv` inside the repo (never in `/tmp`
and never by installing into the system interpreter).

```bash
cd neuro-silicon

# 1. Create the project environment
python3 -m venv .venv

# 2. Install dependencies into it
.venv/bin/pip install numpy scipy matplotlib pytest gymnasium absl-py \
    gin-config streamlit PySide6 Pillow
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch

# 3. Matplotlib cache (avoids first-run cache warnings in sandboxes)
export MPLCONFIGDIR=/tmp/mpl
```

Notes:

- If `pip` refuses with `externally-managed-environment`, you are
  installing outside the venv — use `.venv/bin/pip` as above instead
  of working around it.
- If a `.venv` already exists in the repo, reuse it; do not create a
  second environment.
- Settings (including optional AI keys) live in
  `~/.config/neurosilicon/settings.json` — never in the repository.

### Verify the setup

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q
```

Expected: the suite passes (135 tests). One test
(`test_env_reset_beyond_5000_nodes`) skips itself when the openc906
design has not been imported on that machine — that skip is by design,
re-import openc906 to enable it.

## Run the desktop GUI (primary)

```bash
.venv/bin/python -m neurosilicon.desktop
```

A native Qt window opens — no web server, no browser. First run:

1. **Pick a design** in the left panel (for example
   `servant SoC (18 macros, sky130)` or `toy (2 macros)`), or use the
   **Import** tab to bring your own (folder, any file inside it, or a
   `.zip`; drag-and-drop onto the window also works). A ready
   `netlist.pb.txt` + `initial.plc` pair places directly; RTL gets an
   auto-picked top module and yosys synthesis.
2. **Place → Run floorplan** and watch macros land live on the canvas
   (Play/Pause, step, scrub, speed; Floorplan / Congestion / Thermal /
   Wires views). Or press **⚡ Auto-Optimize (AI Loop)** for the
   autonomous baseline → SAM + Muse diagnosis → AI-guided
   re-placement loop with a PPA Improvement Summary in the Assistant.
3. **Results** for the episode table, Pareto KPI chart, and downloads:
   placed `.plc`, viewer/flat DEF, floorplan/thermal/congestion PNGs,
   a report ZIP, **Download OpenROAD .tcl**, and
   **Export Timelapse GIF**.
4. **Assistant** (right panel) for chat answers over the live
   floorplan, **Suggest RTL Fix for Bottleneck**, and
   **Run 3-Agent Design Review**.

Things to try on the canvas:

- **Right-click / double-click** a macro to 🔒 pin it; **drag** it to
  move the pin to another grid tile. Pinned macros stay fixed in later
  runs while the rest place around them.
- Tick **Show Critical Path** for the gold/magenta timing path and the
  `est. Fmax` header card; tick **SAM** for hotspot contours.
- Switch **View → Compare (Baseline vs. AI)** after Auto-Optimize for
  the before/after split screen with delta badges.
- Switch the **Technology Node** (`sky130` / `asap7` / `gf180mcu`) in
  Thermal or Settings to rescale voltage, clock, leakage, and
  congestion instantly.
- Keep **Post-Placement Polish (SA)** and **orientation flips** on in
  Place for the extra HPWL reduction; flips export as
  `R0/R90/R180/R270` in the OpenROAD script.

Headless check (no display, CI, SSH):

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m neurosilicon.desktop
```

## Run the web console (Streamlit)

```bash
.venv/bin/python -m streamlit run neurosilicon/app.py --server.port 8765
```

Then open http://localhost:8765 in your browser. (Omit
`--server.port` to use the Streamlit default, 8501.)

Sections (left sidebar): **Dashboard** (toolchain/design/run status),
**Run floorplan** (place, with project import on top, chat constraints,
SA polish + orientation options, and **⚡ Auto-Optimize (AI Loop)**),
**Results & downloads** (figures with the **Compare (Baseline vs. AI)**
view, `est. Fmax` / critical-path overlay, KPI Pareto chart, OpenROAD
`.tcl`, DEF/`.plc` downloads), **Thermal lab** (power corners and the
Technology Node switcher), **GNN simulation**, **AI assistant**
(including **Run 3-Agent Design Review** and
**Suggest RTL Fix for Bottleneck**), and **Settings**.

If the console banner says it loaded old code after an update, use its
**Reload updated code & retry** button; if it persists, stop the
server with Ctrl+C and re-run the command above.

## Run from the CLI

Baselines (toy takes seconds; large designs take minutes on CPU):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.evaluate \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --episodes 5 --outdir logs/toy_baseline
```

Train the PPO policy (toy smoke run; scale `--iterations` up for real
training):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.train \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --outdir logs/toy_train --iterations 1 --episodes_per_iter 2
```

Both commands write `results.json`-style tables and placements under
the chosen `logs/…` output directory.

## Importing your own design

- **Ready netlists**: a folder (or `.zip`) containing
  `netlist.pb.txt` + `initial.plc` registers instantly, no synthesis.
- **RTL projects**: Verilog/SystemVerilog folders are discovered
  automatically; pick the top module, choose which files take part,
  and Import synthesizes with yosys, converts to a floorplan netlist,
  and baselines it. Sample RTL ships under `Chip models/` (for example
  the Amber core with `a25_dcache.v` / `a25_icache.v`).
- Imported designs persist in the local registry and reappear in the
  design list on the next launch. Generated import/build/run output
  lives under `logs/` and can be deleted freely — re-importing
  recreates it.

## AI connectors (optional)

Off by default and fully usable offline (every AI feature has a
deterministic local fallback). To enable, open **Connectors** in the
desktop right panel (or the console Settings):

- **api**: any OpenAI-compatible endpoint — base URL, API key, model.
  The optimization loop queries `muse-arcadia-1.0` first, then your
  configured model, then falls back offline.
- **local**: an Ollama / LM Studio server URL + local model name.

Keys are stored only in `~/.config/neurosilicon/settings.json`.
Never commit that file or paste keys into issues or docs.

## Settings reference

Settings persist in `~/.config/neurosilicon/settings.json`:

| Section | What it controls |
| --- | --- |
| `run` | Default mode, episodes, FD steps, seed, GUI output dir (`logs/gui_runs`) |
| `engine` | Wirelength / congestion / density / thermal weights, broadcast-net terms, infeasible penalty |
| `model` | Optional PPO checkpoint (`.pt`); missing file falls back to heuristic placement with a warning |
| `llm` | Provider (`off` / `api` / `local`), endpoint, model, timeout |
| `pdk` | yosys / OpenROAD binaries and LEF/Liberty/tech-LEF paths (blank = sky130 auto-detect) |

The same values are editable in the desktop **Settings** tab and the
console **Settings** page (`Save settings` / `Save connectors`).

## Outputs

| Location | Contents |
| --- | --- |
| `logs/gui_runs/` | Saved console runs (reloadable in Results) |
| `logs/desktop_runs/` | Desktop run outputs, pinned init placements, polished `.plc` files |
| `logs/<design>_proto/` | Converted floorplan netlists for imported/sample designs |
| `logs/joint*_train/`, `logs/*_train/` | PPO training history + `best.pt` checkpoints |
| Downloads (GUI) | Placed `.plc`, viewer/flat DEF, `placed.def` + OpenROAD `.tcl`, PNG figures, report ZIP, timelapse GIF |

`logs/` is generated output: deleting import/build/run subfolders is
safe (they regenerate), but keep `*_proto/` netlists and `*_train/`
checkpoints you still use — the default checkpoint path is
`logs/joint4_train/best.pt`.

## Project layout

| Path | Role |
| --- | --- |
| `neurosilicon/desktop.py` | Native Qt GUI (canvas, pins, compare, exports) |
| `neurosilicon/app.py` | Streamlit web console |
| `neurosilicon/env.py` | Gymnasium placement environment |
| `neurosilicon/model.py`, `train.py` | Edge-GNN policy + masked PPO trainer |
| `neurosilicon/evaluate.py` | Baselines and result tables (CLI) |
| `neurosilicon/liveplace.py` | Live rollouts, RUDY congestion, SAM, critical path/Fmax, PPA, SA polish, pinning |
| `neurosilicon/assistant.py`, `llm.py` | Assistant, Muse diagnosis, RTL co-pilot, 3-agent review |
| `neurosilicon/tech.py` | Technology-node presets (sky130 / asap7 / gf180mcu) |
| `neurosilicon/render.py`, `export_def.py` | Matplotlib figures, DEF + OpenROAD Tcl export |
| `neurosilicon/settings.py`, `project.py`, `discover.py` | Persisted settings, import registry, design discovery |
| `neurosilicon/thirdparty/` | Vendored open-source PLC core (BSD-3; see its notices) |
| `neurosilicon/tests/` | Pytest suite (`*_test.py`) |
| `circuit_training/` | Upstream AlphaChip reference (provenance; not needed to run) |
| `docs/`, `NEUROSILICON_HOW_IT_WORKS.md` | Netlist format, placement cost, system deep-dive |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `externally-managed-environment` from pip | Install with `.venv/bin/pip` (see Setup); do not bypass the system-interpreter guard |
| Desktop window does not appear | You need an X11/Wayland session; for headless checks use `QT_QPA_PLATFORM=offscreen` |
| Streamlit “loaded old code” banner | Use its reload button, else Ctrl+C and re-run the `streamlit run` command |
| Import fails mentioning yosys | Install yosys or set its path in Settings → PDK & Toolchain; the Dashboard pills name missing tools |
| Matplotlib cache warnings | `export MPLCONFIGDIR=/tmp/mpl` before running |
| One test skips (`openc906 not imported`) | Expected without that import; re-import openc906 in the GUI to enable it |
| Checkpoint warning on Run | The configured `.pt` is missing (for example its `logs/` folder was cleaned); placement falls back to heuristic — re-train or clear the path in Settings |

## Upstream AlphaChip

NeuroSilicon builds on the open-source AlphaChip / Circuit Training
framework (Mirhoseini et al., *Nature* 2021), retained under
`circuit_training/` with its own documentation in that tree. The
original TensorFlow/TF-Agents stack (Python ≤ 3.11 plus a retired
`plc_wrapper_main` binary) is replaced here by a Python-native runtime
over TILOS's open-source PLC reimplementation (vendored, BSD-3).
See `LICENSE`, `PRINCIPLES.md`, and `docs/` for licensing, principles,
and format references.

## License

Apache License 2.0 — see `LICENSE`. Vendored third-party code keeps
its own licenses (see `neurosilicon/thirdparty/` notices).

