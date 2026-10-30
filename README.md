```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗     ███████╗██╗██╗     ██╗ ██████╗ ██████╗ ███╗   ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗    ██╔════╝██║██║     ██║██╔════╝██╔═══██╗████╗  ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║    ███████╗██║██║     ██║██║     ██║   ██║██╔██╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║    ╚════██║██║██║     ██║██║     ██║   ██║██║╚██╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝    ███████║██║███████╗██║╚██████╗╚██████╔╝██║ ╚████║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ╚══════╝╚═╝╚══════╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝
```

# NeuroSilicon — AI Floorplanner

**An open-source, AI-assisted chip floorplanner that places macros the way a
learning agent plays a game: one macro at a time, scored on wirelength,
congestion, and density — live on your desktop.**

`version 1.0.0` · `Python 3.12+` · `Apache-2.0` · `136 tests` · `CPU-friendly, no GPU required` · `No TensorFlow, no proprietary binaries`

NeuroSilicon is for the moment in chip design *before* the expensive tools
come out: exploring whether a floorplan is worth pursuing. It takes RTL or a
ready netlist, synthesizes it with open-source tools, places its macros with a
graph-neural-network policy trained by reinforcement learning (or fast
heuristics when no model is loaded), scores every attempt with transparent
proxy metrics, and lets a built-in AI assistant — **Neuro AI** — explain the
result, propose constraints in plain English, and autonomously re-place the
design to beat its own baseline.

Everything runs locally on a laptop-class Linux machine. The numbers are
honest, documented proxies for exploration — not signoff timing, power, or
routing — and the code never pretends otherwise.

---

## Contents

- [What is NeuroSilicon?](#what-is-neurosilicon)
- [Why it matters — its role in the silicon industry](#why-it-matters--its-role-in-the-silicon-industry)
- [Key features](#key-features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run it](#run-it)
- [Importing your own design](#importing-your-own-design)
- [AI connectors (optional)](#ai-connectors-optional)
- [Settings, outputs, and project layout](#settings-outputs-and-project-layout)
- [Testing](#testing)
- [History — what each commit did](#history--what-each-commit-did)
- [Upstream AlphaChip](#upstream-alphachip)
- [Contributing and license](#contributing-and-license)

---

## What is NeuroSilicon?

Chip floorplanning decides where the big blocks (memories, caches, cores —
the *macros*) sit on the die before detailed placement and routing. It is one
of the most consequential early decisions in chip design: a bad floorplan
means long wires, routing congestion, hot spots, and missed clock targets
that no later step can fully repair. Traditionally it is done by experts
driving commercial EDA tools through many manual iterations.

NeuroSilicon turns that step into an open, interactive, learning problem:

1. **Sequential placement as a game.** Hard macros are placed one at a time,
   largest first, each on a legal grid cell. Standard cells then relax with a
   force-directed solver. The episode is scored at the end, exactly like a
   game score.
2. **A learned policy.** An edge-centric graph neural network (GNN) looks at
   the netlist graph and the partially built floorplan and chooses the next
   cell. It is trained with masked PPO. Heuristic players (`first_legal`,
   `random`) are always available, so the system works with no model at all.
3. **Transparent proxy scoring.** Every finished placement reports
   wirelength, RUDY-style routing congestion, cell density, an estimated
   maximum clock frequency, thermal hotspots, and a single combined *proxy
   cost* (lower is better). You can watch each term, not just the total.
4. **An AI loop on top.** The Neuro AI assistant reads the live floorplan
   through a small tool surface, explains results, turns chat sentences into
   placement constraints, diagnoses congestion hotspots, suggests RTL fixes
   for the bottleneck block, and runs a fully autonomous three-iteration
   optimization loop (baseline → diagnosis → AI-guided re-placement) with a
   before/after PPA summary.

The runnable system is the `neurosilicon/` package. It runs as a native Qt
desktop application (primary), a Streamlit web console, and command-line
tools for baselines and training.

### How it works in one paragraph

A Gymnasium environment exposes the floorplan as a padded 128×128 grid with
16,384 possible actions. The GNN policy encodes macro metadata and netlist
edges, proposes a cell for the macro currently being placed, and receives a
terminal reward equal to the negative proxy cost (`1.0·wirelength +
0.5·congestion + 1.0·density` by default). Illegal cells are masked out;
dead-end episodes receive a fixed penalty. After the macros land, simulated
annealing and orientation flips polish wirelength, and coordinate descent can
finetune further. The full technical deep-dive lives in
[NEUROSILICON_HOW_IT_WORKS.md](NEUROSILICON_HOW_IT_WORKS.md), and every
feature is catalogued in [docs/FEATURES.md](docs/FEATURES.md).

---

## Why it matters — its role in the silicon industry

Floorplanning sits at the front of the physical-design pipeline, and it has
historically been gated behind three walls: expensive commercial tool
licenses, scarce expert time, and slow iteration cycles. Work like Google's
AlphaChip (Mirhoseini et al., *Nature* 2021) showed that reinforcement
learning can produce competitive macro placements — but the original stack
(TensorFlow, TF-Agents, distributed Reverb servers, a closed placement
binary) put reproduction out of reach for most students, researchers, and
small teams.

That gap is exactly where NeuroSilicon fits. Its role in the broader silicon
ecosystem — the world of startups, university labs, open-silicon efforts
(sky130, OpenROAD, OpenLane), and the engineers who prototype the next chip
before a company exists — is:

- **Democratization.** A complete, readable, Python-native re-implementation
  of the AlphaChip idea that runs on a CPU laptop with no licenses. Anyone
  can read the MDP, change the reward, and see the effect the same afternoon.
- **Education and research platform.** The environment, policy, and proxy
  costs are small, documented, and test-covered — a practical testbed for
  new placement ideas, reward shaping, and GNN architectures.
- **Early design exploration.** Import real RTL (open cores ship as samples),
  get a floorplan and PPA proxies in minutes, and decide whether an idea
  deserves commercial-tool time. Routed OpenROAD validation is available as
  a reality check on the proxies.
- **AI-assistance research.** An assistant that can only see tool results —
  never edit files, never place anything itself — is a deliberately safe
  pattern for putting LLMs next to EDA state, with deterministic offline
  fallbacks for every AI feature.
- **Open-silicon momentum.** By building only on yosys, OpenROAD, and the
  sky130 PDK ecosystem, it complements the open-hardware movement rather than
  competing with commercial signoff tools.

NeuroSilicon does **not** replace Cadence/Synopsys signoff flows, and it does
not claim tapeout-ready results. Its metrics are proxies, its DEF export is
viewer-grade, and its routed check is a global-route pass. The honest
framing is: *a fast, open, AI-assisted sketchpad for the most creative step
in chip design* — the step where human and machine intuition are worth the
most.

---

## Key features

| Area | Highlights |
| --- | --- |
| Placement engine | Sequential largest-first macro placement on a masked 128×128 grid; force-directed standard-cell relaxation; random / first-legal / learned-policy rollouts |
| Learned policy | Edge-centric GCN actor-critic (PyTorch), masked PPO training, multi-netlist joint training, checkpoints with heuristic fallback |
| Scoring & proxies | Wirelength + congestion + density proxy cost; RUDY congestion maps; SAM hotspot segmentation; critical-path and estimated-Fmax proxy; thermal/power corners |
| Polish | 200-step simulated-annealing macro swaps, 0°/90°/180°/270° orientation flips (exported as `R0/R90/R180/R270`), coordinate-descent finetuning |
| Desktop GUI | Live canvas (Floorplan / Congestion / Thermal / Wires / Compare), Play/Pause/step/scrub playback, macro pinning by right-click or drag, KPI cards, episode table with Δ% and Pareto chart |
| Neuro AI chat | WhatsApp-style chat renamed **Neuro AI**: suggestion dropdown + type-ahead autocomplete, project import from chat, offline rule-based answers when no provider is configured |
| Auto-Optimize (AI Loop) | One click: baseline placement → SAM hotspot + AI diagnosis → constrained re-placement → PPA Improvement Summary and Baseline-vs-AI compare view |
| RTL co-pilot | Finds the thermal/congestion bottleneck macro, locates its Verilog, and proposes a synthesizable fix (memory banking, bus pipelining, clock gating) |
| 3-Agent Design Review | Thermal, Routing & Timing, and Lead Architect agents review the floorplan and recommend engine weights — deterministic, works offline |
| Import | Folder / single file / `.zip` / drag-and-drop; ready netlists register instantly; RTL is auto-discovered, synthesized with yosys, converted, and baselined |
| Export & validation | Placed `.plc`, viewer DEF, flat DEF, OpenROAD `.tcl`, floorplan/thermal/congestion PNGs, timelapse GIF, report ZIP; OpenROAD global-route validation |
| Designs included | `toy` (2 macros), `ariane` (133), `servant SoC` (sky130), `picorv32`, `picosoc`, `ultraembedded riscv_core` — plus whatever you import and register |

---

## Requirements

| Need | Details |
| --- | --- |
| OS | Linux (developed and tested on Fedora/Ubuntu-class desktops) |
| Python | 3.12 or newer (CI uses 3.12; the project `.venv` uses 3.14) |
| Display | X11 or Wayland session for the desktop GUI (headless check below works without one) |
| GPU | Not required — the CPU PyTorch build is enough |
| Memory/disk | Laptop-class is fine for included designs; very large SoCs (hundreds of thousands of cells) want several GB free for synthesis |
| Optional tools | `yosys` (RTL synthesis on Import) and `openroad` (routed validation); paths are configurable in Settings, blank PDK fields auto-detect a Ciel sky130 install |
| AI provider | Optional and off by default; any OpenAI-compatible API, or a local Ollama / LM Studio server |

Everything else — placement engine, SAM hotspot segmentation, thermal and
congestion maps, timing/Fmax estimates — is computed locally. Every number
is a documented proxy for interactive exploration, not signoff.

---

## Setup

Run everything from the repository root. The environment belongs to the
project: create it as `.venv` inside the repo (never in `/tmp`, and never by
installing into the system interpreter).

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

- If `pip` refuses with `externally-managed-environment`, you are installing
  outside the venv — use `.venv/bin/pip` as above instead of working around it.
- If a `.venv` already exists in the repo, reuse it; do not create a second one.
- Settings (including optional AI keys) live in
  `~/.config/neurosilicon/settings.json` — never in the repository.

### Verify the setup

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q
```

Expected: **136 passed, 1 skipped**. The skip
(`test_env_reset_beyond_5000_nodes`, reported as *openc906 not imported*) is
by design — it enables itself once that design has been imported on the
machine. CI runs the same suite on every push (`.github/workflows/ci.yml`);
CI installs no Qt, so the two offscreen desktop-UI tests skip there and run
in a local desktop environment.

---

## Run it

### Desktop GUI (primary)

```bash
.venv/bin/python -m neurosilicon.desktop
```

A native Qt window opens — no web server, no browser. First run:

1. **Pick a design** in the left panel (for example `servant SoC (18 macros,
   sky130)` or `toy (2 macros)`), or import your own (next section).
2. **Place → Run floorplan** and watch macros land live on the canvas
   (Play/Pause, step, scrub, speed; Floorplan / Congestion / Thermal / Wires
   views). Or press **⚡ Auto-Optimize (AI Loop)** for the autonomous
   baseline → SAM + AI diagnosis → AI-guided re-placement loop with a PPA
   Improvement Summary posted in Neuro AI.
3. **Results** for the episode table, Pareto KPI chart, and downloads:
   placed `.plc`, viewer/flat DEF, floorplan/thermal/congestion PNGs, a
   report ZIP, **Download OpenROAD .tcl**, and **Export Timelapse GIF**.
4. **Neuro AI** (right panel) for chat answers over the live floorplan,
   **Suggest RTL Fix for Bottleneck**, and **Run 3-Agent Design Review**.

Things to try on the canvas:

- **Right-click / double-click** a macro to pin it; **drag** it to move the
  pin to another grid tile. Pinned macros stay fixed in later runs while the
  rest place around them.
- Tick **Show Critical Path** for the gold/magenta timing path and the
  `est. Fmax` header card; tick **SAM** for hotspot contours.
- Switch **View → Compare (Baseline vs. AI)** after Auto-Optimize for the
  before/after split screen with delta badges.
- Switch the **Technology Node** (`sky130` / `asap7` / `gf180mcu`) in
  Thermal or Settings to rescale voltage, clock, leakage, and congestion
  instantly.
- Keep **Post-Placement Polish (SA)** and **orientation flips** on in Place
  for the extra wirelength reduction; flips export as `R0/R90/R180/R270` in
  the OpenROAD script.

Headless check (no display, CI, SSH):

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m neurosilicon.desktop
```

### Web console (Streamlit)

```bash
.venv/bin/python -m streamlit run neurosilicon/app.py --server.port 8765
```

Then open http://localhost:8765 in your browser. (Omit `--server.port` to
use the Streamlit default, 8501.)

Sections (left sidebar): **Dashboard** (toolchain/design/run status),
**Run floorplan** (place, with project import on top, chat constraints, SA
polish + orientation options, and **⚡ Auto-Optimize (AI Loop)**),
**Results & downloads** (figures with the **Compare (Baseline vs. AI)** view,
`est. Fmax` / critical-path overlay, KPI Pareto chart, OpenROAD `.tcl`,
DEF/`.plc` downloads), **Thermal lab** (power corners and the Technology
Node switcher), **GNN simulation**, **AI assistant** (including
**Run 3-Agent Design Review** and **Suggest RTL Fix for Bottleneck**), and
**Settings**.

If the console banner says it loaded old code after an update, use its
**Reload updated code & retry** button; if it persists, stop the server with
Ctrl+C and re-run the command above.

### Command line

Baselines (toy takes seconds; large designs take minutes on CPU):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.evaluate \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --episodes 5 --outdir logs/toy_baseline
```

Train the PPO policy (toy smoke run; scale `--iterations` up for real training):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.train \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --outdir logs/toy_train --iterations 1 --episodes_per_iter 2
```

Both write `manifest.json`, `run.log`, result tables, and placements under
the chosen `logs/…` output directory, so every run is self-describing.

---

## Importing your own design

Four doors, same room — every path ends with a registered design you can
place, compare, and export:

| Door | How |
| --- | --- |
| Neuro AI chat button | Right panel → **Neuro AI** → **Import project file…** → pick any file inside the project |
| Neuro AI chat text | Type `import /path/to/project` (folder, single RTL file, or `.zip`) in the chat box |
| Import tab | Centre panel → **Import** → **Folder… / File… / ZIP…** → **Auto-detect** → **Import design** |
| Drag-and-drop | Drop a folder, file, or `.zip` anywhere onto the desktop window |

What happens next:

- **Ready netlists**: a folder (or `.zip`) containing `netlist.pb.txt` +
  `initial.plc` registers instantly, no synthesis. From chat, registration
  happens immediately; from the Import tab, press Import design.
- **RTL projects**: Verilog/SystemVerilog is discovered automatically, the
  top module is auto-picked (verify it), unrelated files are pruned, and
  Import synthesizes with yosys, converts to a floorplan netlist, and runs
  baselines. From chat, RTL opens pre-filled in the Import tab for that
  confirmation — the long synthesis never starts silently.
- **Sample RTL**: open-source cores under `Chip models/` work the same way.
  Example that is known-good end to end: the Amber SoC top at
  `Chip models/amber/hw/vlog/system/system.v` (converted and registered as
  `imported: amber`). Note the bundled yosys speaks Verilog-2005 plus a
  subset of SystemVerilog; VHDL cores and modern-SystemVerilog designs may
  fail at synthesis with the yosys error shown verbatim.
- Imported designs persist in the local registry (`logs/gui_runs/registry.json`)
  and reappear in the design list on the next launch. Generated output lives
  under `logs/` and can be deleted freely — re-importing recreates it.

---

## AI connectors (optional)

Off by default and fully usable offline — every AI feature has a
deterministic local fallback, and Neuro AI answers from live tool results
(design info, run results, thermal info, constraints) either way. To enable
an LLM, open **Connectors** in the desktop right panel (or the console
Settings):

- **api**: any OpenAI-compatible endpoint — base URL, API key, model. The
  optimization loop and RTL co-pilot query the Muse model first, then your
  configured model, then fall back offline. Use **List models** to copy an
  exact model ID your key can access, then **Test connection**.
- **local**: an Ollama (`http://localhost:11434`) or LM Studio
  (`http://localhost:1234/v1`) server URL plus the local model name. No key
  leaves your machine.

Keys are stored only in `~/.config/neurosilicon/settings.json`. Never commit
that file or paste keys into issues or docs. The assistant can only *see*
tool results — it never edits files and never runs placement itself; the
local engine always does the placing.

---

## Settings, outputs, and project layout

### Settings reference

Settings persist in `~/.config/neurosilicon/settings.json`:

| Section | What it controls |
| --- | --- |
| `run` | Default mode, episodes, FD steps, seed, GUI output dir (`logs/gui_runs`) |
| `engine` | Wirelength / congestion / density / thermal weights, broadcast-net terms, infeasible penalty |
| `model` | Optional PPO checkpoint (`.pt`); a missing file falls back to heuristic placement with a warning |
| `llm` | Provider (`off` / `api` / `local`), endpoint, model, timeout |
| `pdk` | yosys / OpenROAD binaries and LEF/Liberty/tech-LEF paths (blank = sky130 auto-detect) |

The same values are editable in the desktop **Settings** tab and the console
**Settings** page (`Save settings` / `Save connectors`).

### Outputs

| Location | Contents |
| --- | --- |
| `logs/gui_runs/` | Saved console runs (reloadable in Results) and the imported-design registry |
| `logs/desktop_runs/` | Desktop run outputs, pinned init placements, polished `.plc` files |
| `logs/imports/<design>/` | Chat/Import conversions (`build/` synthesis, `proto/` netlist, `baseline/` scores, `import.json`) |
| `logs/<design>_proto/` | Converted floorplan netlists for earlier imported/sample designs |
| `logs/joint*_train/`, `logs/*_train/` | PPO training history + `best.pt` checkpoints |
| Downloads (GUI) | Placed `.plc`, viewer/flat DEF, `placed.def` + OpenROAD `.tcl`, PNG figures, report ZIP, timelapse GIF |

`logs/` is generated output: deleting import/build/run subfolders is safe
(they regenerate), but keep `*_proto/` and `logs/imports/` netlists and
`*_train/` checkpoints you still use — the default checkpoint path is
`logs/joint4_train/best.pt`.

### Project layout

| Path | Role |
| --- | --- |
| `neurosilicon/desktop.py` | Native Qt GUI (canvas, pins, compare, Neuro AI chat, exports) |
| `neurosilicon/app.py` | Streamlit web console |
| `neurosilicon/env.py` | Gymnasium placement environment |
| `neurosilicon/model.py`, `train.py` | Edge-GNN policy + masked PPO trainer |
| `neurosilicon/evaluate.py` | Baselines and result tables (CLI) |
| `neurosilicon/liveplace.py` | Live rollouts, RUDY congestion, SAM hotspots, critical path/Fmax, PPA, SA polish, orientations |
| `neurosilicon/assistant.py`, `llm.py` | Neuro AI assistant, diagnosis, RTL co-pilot, 3-agent review, LLM transport |
| `neurosilicon/priors.py`, `finetune.py`, `grid_select.py`, `setup.py`, `tech.py` | Constraint masks, coordinate descent, grid auto-selection, PLC construction, technology nodes |
| `neurosilicon/render.py`, `export_def.py`, `flatten_def.py` | Matplotlib figures, viewer/flat DEF, OpenROAD Tcl export |
| `neurosilicon/settings.py`, `project.py`, `discover.py`, `synth_to_proto.py`, `runlog.py` | Persisted settings, import pipeline + registry, design discovery, netlist conversion, run logging |
| `neurosilicon/thirdparty/` | Vendored open-source PLC core (BSD-3; see its notices) |
| `neurosilicon/tests/` | Pytest suite (19 files, `*_test.py`) |
| `circuit_training/` | Upstream AlphaChip reference (provenance; not needed to run) |
| `docs/`, `NEUROSILICON_HOW_IT_WORKS.md` | Feature reference, netlist format, placement cost, system deep-dive |

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `externally-managed-environment` from pip | Install with `.venv/bin/pip` (see Setup); do not bypass the system-interpreter guard |
| Desktop window does not appear | You need an X11/Wayland session; for headless checks use `QT_QPA_PLATFORM=offscreen` |
| Streamlit "loaded old code" banner | Use its reload button, else Ctrl+C and re-run the `streamlit run` command |
| Import fails mentioning yosys | Install yosys or set its path in Settings → PDK & Toolchain; the Dashboard pills name missing tools |
| LLM test shows 404 / model error | The endpoint answered but the model name is not an offered ID — use **List models** in Connectors and copy one exact ID |
| Matplotlib cache warnings | `export MPLCONFIGDIR=/tmp/mpl` before running |
| One test skips (`openc906 not imported`) | Expected without that import; import openc906 or ignore the skip |
| Checkpoint warning on Run | The configured `.pt` is missing (for example its `logs/` folder was cleaned); placement falls back to heuristic — re-train or clear the path in Settings |

---

## Testing

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q
```

136 tests across 19 files cover the placement environment and masking, the
GNN and PPO trainer, proxy costs, constraint parsing, import/synthesis,
DEF/PNG export, the desktop backends, Auto-Optimize, the RTL co-pilot
templates, and the Neuro AI chat widgets (suggestion dropdown, import
routing, Neuro AI naming — offscreen, no display needed). One test skips
by design until openc906 is imported locally; two Qt widget tests skip in
CI, which installs no PySide6, and run in a desktop environment.

---

## History — what each commit did

This repository has 294 commits in two eras. The first 291 are the restored
upstream AlphaChip / Circuit Training history (Google Research, 2021–2026),
retained for provenance under `circuit_training/` and summarized as a group
below. The last three are NeuroSilicon itself:

### NeuroSilicon commits

| Commit | Date | What it did |
| --- | --- | --- |
| `54dcd45` | 2026-10-28 | **The foundation.** Added the complete runnable system in one commit (84 files, ~17,700 lines): the Gymnasium placement environment, edge-GNN policy and masked-PPO trainer, baseline evaluator, Python-native runtime over the vendored TILOS PLC core replacing the TensorFlow stack, the first desktop GUI and Streamlit console, design discovery/import with yosys synthesis, MCP-style assistant with offline fallbacks, settings outside the repo, CI, and tool shims for yosys/OpenROAD. This is the commit that turned an upstream research snapshot into a product you can install and run. |
| `91a7772` | 2026-10-29 | **The EDA upgrade.** Grew the desktop into a workstation (the bulk of its current 3,600 lines): live QPainter canvas with playback, Compare (Baseline vs. AI), SAM/critical-path overlays, technology-node presets (sky130/asap7/gf180mcu), PPA metrics and the Auto-Optimize loop, simulated-annealing polish and orientation flips, the RTL co-pilot, and most of the test suite that guards them. |
| `7f6ba12` | 2026-10-29 | **The documentation pass.** Rewrote this README into a complete user document (setup, GUI/console/CLI runs, import, connectors, settings, outputs, troubleshooting) so a new user can go from clone to first floorplan without reading source. |

### In the current (uncommitted) workspace

- The **Neuro AI** chat upgrade: WhatsApp-style bubbles with timestamps, the
  `Assistant` → `Neuro AI` rename, a Suggestions dropdown plus type-ahead
  autocomplete replacing the truncated quick-ask buttons, and project import
  from chat (button or `import <path>`) — each covered by offscreen widget tests.
- [docs/FEATURES.md](docs/FEATURES.md): a full feature reference synthesized
  from a line-by-line inspection of the package, including explicit
  non-features and evidence limits.
- The Amber SoC conversion (`Chip models/amber/hw/vlog/system/system.v` →
  `imported: amber`, 85 macros, best proxy 1.1295) proving the import
  pipeline on a 787k-cell design, plus a documented triage of 30 open-source
  chip models against the bundled toolchain.

### Upstream era (2021–2026, grouped)

The 291 upstream commits built the original Circuit Training framework:
the placement-cost simulator and its API, DREAMPlace integration and
regioning, fake-net/packing heatmaps, distributed PPO collection with
Reverb, pre-training on multiple netlists, the Ariane example flow, and
years of correctness work (NumPy 2.0 migration, legalization, density and
divergence fixes). NeuroSilicon keeps that code untouched for reference;
none of it runs in the current system. Per-commit detail is in
`git log circuit_training/` and `docs/ARIANE.md` / `docs/PRETRAINING.md`.

Release notes per version are in [CHANGELOG.md](CHANGELOG.md).

---

## Upstream AlphaChip

NeuroSilicon builds on the open-source AlphaChip / Circuit Training
framework (Mirhoseini et al., *Nature* 2021), retained under
`circuit_training/` with its own documentation in that tree. The original
TensorFlow/TF-Agents stack (Python ≤ 3.11 plus a retired `plc_wrapper_main`
binary) is replaced here by a Python-native runtime over TILOS's
open-source PLC reimplementation (vendored, BSD-3). See `LICENSE`,
`PRINCIPLES.md`, and `docs/` for licensing, principles, and format
references (netlist format, placement cost, blockages, pre-training).

---

## Contributing and license

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Please keep tests green
(`pytest neurosilicon/tests/ -q`), keep AI keys out of the tree, and keep
proxy metrics labelled as proxies wherever they are shown to users.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). Vendored
third-party code keeps its own licenses (see `neurosilicon/thirdparty/`
notices).

*NeuroSilicon — place smarter, explore faster, keep it open.*


