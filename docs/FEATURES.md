# NeuroSilicon — Feature Reference

Version covered: `1.0.0` (`neurosilicon/_version.py`).
Date of analysis: 2026-10-30.
Method: full code inspection of the `neurosilicon/` package, root docs,
`docs/`, `tools/`, and test inventory. Behaviour below is proven from
source unless marked otherwise. Runtime launches and test runs were not
part of this analysis — see [Evidence limits](#evidence-limits).

## 1. What the project is

NeuroSilicon is an AI chip floorplanner. It places hard macros
sequentially with an edge-GNN policy trained by PPO. Each finished
episode is scored by an open-source proxy cost (wirelength + congestion
+ density) after force-directed standard-cell relaxation.

Key positioning, stated in `README.md`:

- Runnable system is the `neurosilicon/` package only.
- Native Qt desktop GUI is the primary interface.
- Streamlit web console is the secondary interface.
- CLI tools cover baselines and training.
- No TensorFlow. No proprietary binaries. CPU-friendly (CPU PyTorch is
  enough; GPU not required).
- OS: Linux (Fedora/Ubuntu-class). Python 3.12+ (CI uses 3.12; the
  project `.venv` uses 3.14). Desktop needs an X11 or Wayland session.
- Optional tools: `yosys` for RTL synthesis on import, `openroad` for
  routed validation. Paths are configurable in Settings.
- Optional AI provider, off by default: any OpenAI-compatible API, or a
  local Ollama / LM Studio server.

> Every reported number is a documented proxy for interactive
> exploration. Nothing here is signoff timing, power, or routing.
> This caveat is in `README.md` and is repeated in the GUIs.

## 2. How to run it

All commands run from the repository root. The environment lives in the
project as `.venv`.

| Entry point | Command | Source |
| --- | --- | --- |
| Desktop GUI (primary) | `.venv/bin/python -m neurosilicon.desktop` | `neurosilicon/desktop.py` |
| Web console | `.venv/bin/python -m streamlit run neurosilicon/app.py --server.port 8765` then open `http://localhost:8765` (omit the port flag for Streamlit default 8501) | `neurosilicon/app.py` |
| Headless desktop check | `QT_QPA_PLATFORM=offscreen .venv/bin/python -m neurosilicon.desktop` | `README.md` |
| CLI baselines | `.venv/bin/python -m neurosilicon.evaluate --netlist <netlist.pb.txt> [--init_placement <initial.plc>] [--episodes 5] [--outdir logs/eval]` | `neurosilicon/evaluate.py` |
| CLI training | `.venv/bin/python -m neurosilicon.train --netlist <netlist.pb.txt> [--outdir logs/train] [--iterations 10]` (`--netlist` repeatable) | `neurosilicon/train.py` |
| Flat DEF CLI | `python -m neurosilicon.flatten_def --netlist … --plc … --json … --lef … --top … --out …` | `neurosilicon/flatten_def.py` |
| Synth-to-proto CLI | `python -m neurosilicon.synth_to_proto --json … --lef … --top … --outdir …` | `neurosilicon/synth_to_proto.py` |

There are no `console_scripts` / `pyproject` entry points for
NeuroSilicon. The root `setup.py` packages the upstream
`circuit_training` code, not `neurosilicon` — see
[Section 11](#11-upstream-provenance-and-scope).

## 3. Architecture map

| Module | Role |
| --- | --- |
| `neurosilicon/env.py` | Gymnasium `ChipPlacementEnv`: sequential macro placement MDP |
| `neurosilicon/model.py` | `CircuitPolicy`: PyTorch edge-centric GCN actor-critic |
| `neurosilicon/train.py` | Single-process masked PPO training |
| `neurosilicon/evaluate.py` | Random / first-legal / policy rollouts and `run_baselines` |
| `neurosilicon/liveplace.py` | Live rollout, PPA/thermal/timing proxies, SA polish, orientation search, RUDY congestion, PPA summary |
| `neurosilicon/priors.py` | Keep-out/allow/perimeter masks, chat-to-mask parsing, broadcast and thermal penalties |
| `neurosilicon/finetune.py` | Coordinate-descent finetuning (`run_cd`) |
| `neurosilicon/grid_select.py` | Automatic placement-grid selection |
| `neurosilicon/setup.py` | Tolerant PLC construction (`create_plc`) |
| `neurosilicon/tech.py` | Technology-node presets and corner power |
| `neurosilicon/desktop.py` | Qt desktop application (primary GUI) |
| `neurosilicon/app.py` | Streamlit web console, run persistence, validation, report bundle |
| `neurosilicon/render.py` | Headless matplotlib figures and PNG export |
| `neurosilicon/ui.py` | Streamlit presentational helpers (CSS, header, pills, stepper) |
| `neurosilicon/gnn_viz.py` | Policy-rollout simulation recording for scrubbing |
| `neurosilicon/assistant.py` | MCP-style assistant, Auto-Optimize diagnosis, RTL co-pilot, 3-agent review |
| `neurosilicon/llm.py` | OpenAI-compatible API + Ollama/local LLM transport |
| `neurosilicon/settings.py` | Settings schema, persistence, secret redaction |
| `neurosilicon/discover.py` | Design discovery for folder / file / `.zip` import |
| `neurosilicon/project.py` | RTL staging, yosys synthesis, import pipeline, PDK/tool resolution, design registry |
| `neurosilicon/synth_to_proto.py` | Synthesis JSON + LEF → `netlist.pb.txt` / `initial.plc` / `clusters.json` |
| `neurosilicon/export_def.py` | Viewer DEF and OpenROAD Tcl export |
| `neurosilicon/flatten_def.py` | Flat cell-grade DEF for routed validation |
| `neurosilicon/runlog.py` | Run logging and self-describing `manifest.json` |
| `neurosilicon/thirdparty/tilos/` | Vendored BSD-3 TILOS PLC core (see evidence limits) |

## 4. Placement engine

### 4.1 Sequential placement MDP

- `ChipPlacementEnv` places each non-fixed hard macro once, in
  descending-size order. Soft macros are excluded from the action
  sequence (`neurosilicon/env.py`).
- Canvas is a padded 128×128 centred grid (`MAX_GRID_SIZE=128`), giving
  16,384 discrete actions. Illegal, off-grid, or mask-zero placements
  return the infeasible penalty (`-4.0` by default). Intermediate
  reward is 0; only the terminal step is scored.
- Each reset unplaces all nodes, so episodes are isolated.

### 4.2 Proxy cost and scoring

- Proxy cost is `-(w_wl·wirelength + w_cong·congestion + w_dens·density)`
  plus any `extra_costs`. Defaults: wirelength 1.0, congestion 0.5,
  density 1.0 (`neurosilicon/env.py`, `neurosilicon/settings.py`).
- Terminal info reports `wirelength`, `congestion`, `density`, and
  `proxy_cost`. Lower proxy is better.
- Wirelength, congestion, and density come from the PLC cost object.
  The exact `PlacementCost` semantics live in the vendored TILOS core,
  which this analysis did not read line by line — see evidence limits.

### 4.3 Force-directed relaxation

- Soft macros and standard cells relax with a force-directed schedule
  at construction and again before scoring (`fd_placement_schedule`).
  GUI/CLI expose FD steps per stage (default 10 in the GUIs).

### 4.4 Rollouts and baselines

- `rollout_random`: uniform choice among legal cells.
- `rollout_first_legal`: first legal cell (deterministic).
- `rollout_policy`: `CircuitPolicy` with masked logits, greedy argmax
  or seeded softmax sampling. Infeasible episodes report proxy `-1`.
- `run_baselines` runs episodes of both heuristic methods, saves
  `best_baseline.plc`, `results.json`, and a run manifest, and prints a
  table from the CLI (`neurosilicon/evaluate.py`).

### 4.5 PPA, congestion, thermal, and timing proxies

All of these are proxies, labelled as such in code and UI
(`neurosilicon/liveplace.py`):

- RUDY congestion: grid demand with overall (top-10% mean) and peak
  values, top-5 peak tiles, and up to 8 congested macros.
- SAM hotspot segmentation: flood-fill over the top-15% hotspot cells
  (`top_hotspot`, `segment_hotspots`).
- Power grid and per-macro power: area-proportional by default, or from
  an uploaded `{macro: watts}` map where the UI accepts one.
- Critical path: DAG longest path with Manhattan wire delay
  (0.002 ns/µm) plus logic delay; `est. Fmax = 1000 / delay_ns`. This
  is not signoff STA.
- `ppa_metrics` / `ppa_delta` / `format_ppa_summary` produce the
  before/after summaries used by Compare and Auto-Optimize.

### 4.6 Live rollout and snapshots

- `run_live_rollout` supports `first_legal` and `random`, emits
  snapshots through callbacks, and thins to at most 160 frames.
- Snapshots carry hierarchy-group colours (cache / ALU / SRAM /
  default families), orientation, pin edges, and per-macro power heat.
  The desktop canvas replays these frames; the web console shows the
  best floorplan preview per episode.

### 4.7 Post-placement polish

- Simulated annealing (`simulated_annealing`, 200 steps in the GUIs):
  swaps compatible transposed-size pairs, flips orientations, or
  nudges a macro. Acceptance is Metropolis on pin-aware HPWL. Pinned
  macros are held fixed. The search is read-only and returns the best
  positions/orientations for the caller to apply.
- Orientation optimisation (`optimize_orientations`): one greedy pass
  over 0°/90°/180°/270° per hard macro, minimising pin-aware HPWL.
  Orientations export as `R0/R90/R180/R270` in the OpenROAD Tcl.
- Coordinate-descent finetune (`run_cd`): greedy position/orientation
  local search (distance bound 5, 2 epochs by default) with
  force-directed stdcell placement. DreamPlace is not used here.

### 4.8 GNN policy and PPO training

- `CircuitPolicy` is an edge-centric GCN: metadata and feature
  encoders, 3 GCN layers with residual mean-scatter, current-node
  attention, a value MLP, and a transposed-convolution policy head
  producing 16,384 logits. Infeasible logits are masked at
  `-(2^32)+1`. Dirichlet exploration noise and Glorot init are
  supported (`neurosilicon/model.py`).
- Training is single-process masked PPO: GAE (γ=1.0, λ=0.95), 4 epochs,
  batch 128, clip 0.2, value coefficient 0.5, entropy 0.01, gradient
  clip 1.0, cosine LR from 0.004 to 10%. Multi-netlist training is
  round-robin. Evaluation saves `best.pt`, `latest.pt`,
  `history.csv`, and `best_policy_<i>.plc`. `--resume` and
  `--dirichlet_weight` are supported (`neurosilicon/train.py`).
- Modes: `heuristic` (no model needed), `neurosilicon` (PPO checkpoint;
  missing checkpoint falls back to heuristic with a warning), and
  `llm_assisted` (heuristic plus LLM review).

### 4.9 Priors, masks, and extra costs

- Hard masks from rectangles: `make_keepout_fn` (block rectangles),
  `make_allow_fn` (restrict to rectangles), `make_perimeter_fn`
  (edge bands, 0.25 margin default). These plug in as `extra_mask_fn`.
- Rule-based chat parsing (`chat_to_spec` and friends): centre channel
  (20% default), edge allow/block (25% default). Unrecognised text
  produces no mask (`neurosilicon/priors.py`).
- Extra costs: broadcast penalty for high-fanout nets (fanout ≥16;
  flat variant ≥64) and thermal penalties — overlap penalty for close
  high-power macro pairs plus corner-based dynamic/leakage terms.
  Both are off or zero by default and marked experimental where the
  docs say so.

### 4.10 Grid selection, PLC setup, technology nodes

- `grid_select` sweeps 10–128 columns/rows, keeps grids that place all
  hard macros, and picks the grid minimising waste/empty ratio
  (within 5% of best, smallest wins). Fallback is 32×32.
- `create_plc` resolves canvas/grid/routes with precedence: explicit
  args > `.plc` header > netlist comments > defaults (default grid 32,
  aspect-preserving). It validates the initial placement node set,
  fixes ports and blockages, and squares soft macros.
- Technology nodes (`neurosilicon/tech.py`, documented proxies, not
  signoff PDK data):

| Node | Voltage | Clock | Notes |
| --- | --- | --- | --- |
| `sky130` | 1.8 V | 100 MHz | Baseline preset (default fallback) |
| `asap7` | 0.7 V | 1000 MHz | Higher leakage/congestion multipliers |
| `gf180mcu` | 3.3 V | 50 MHz | Lower leakage/congestion multipliers |

Switching the node rescales corner power
(`(V/1.8)² × (f/100) × (activity/0.2)` plus area leakage) and scaled
congestion instantly in both GUIs.

## 5. Desktop GUI (primary)

Entry: `python -m neurosilicon.desktop`. Native Qt window, dark theme,
minimum 1024×640 (`neurosilicon/desktop.py`).

### 5.1 Layout

- Three panes: left Designs, centre canvas + KPIs + tabs, right Run
  status + assistant tabs. Menus: File / Run / View / Help. Toolbar
  toggles for Designs, AI Panel, and Log Terminal.
- Centre tabs: Import / Place / Results / Thermal.
- Right tabs: Assistant / Connectors / Guide / Settings / Log.
- Drag-and-drop a folder, file, or `.zip` onto the window to
  auto-detect a design.
- Header KPI cards: proxy, wirelength, congestion, macros placed,
  est. Fmax. Tooltips state the Fmax/thermal numbers are proxies, not
  signoff STA.

### 5.2 Canvas

- Views: Floorplan, Congestion, Thermal, Wires, the three combined
  `+Wires` views, and Compare (Baseline vs. AI). Rendering is native
  `QPainter` — no PNG round-trip.
- Hard macros, soft macros (translucent grey), pending (dashed), and
  the macro currently being placed (amber outline) are visually
  distinct. Ports, keep-outs/blockages (red dashed), orientation
  notches, and a hierarchy legend are drawn.
- Interactions: click a macro to select it and show flylines and
  facts. Right-click or double-click a hard macro to pin it (amber
  badge). Drag a hard macro to snap its pin to another grid tile.
- Toggles: SAM (top-15% hotspot contours, Thermal/Congestion only)
  and Show Critical Path (gold/magenta Manhattan path).

### 5.3 Playback

- Play/Pause, Step, scrub slider, and Slow (450 ms) / Normal (140 ms)
  / Fast (45 ms) speeds. Live snapshots are coalesced at ~20 FPS for
  display while every frame is recorded for replay.

### 5.4 Import tab

- Path field with Folder… / File… / ZIP… pickers and Auto-detect
  (`discover_design`). Shows the discovery plan, an editable Top
  module combo, project name, parameters, Verilog defines, baseline
  episodes (1–5), and a checkable RTL file list.
- A ready `netlist.pb.txt` + `initial.plc` pair registers instantly.
  RTL goes through yosys synthesis. Also hosts Suggest RTL Fix.

### 5.5 Place tab

- Animate placement live (on by default), live rollout policy
  (`first_legal` / `random`), baseline episodes (1–10), FD relax steps
  (default 10), random seed, finetune epochs, SA polish checkbox, and
  orientation-flip checkbox.
- Pinned-macros and chat-constraint rows, each with a Clear button.
- Buttons: Run floorplan (primary) and Auto-Optimize (AI Loop).
- A run honours pins, the keep-out mask, and current engine weights,
  then optionally applies orientation optimisation and SA polish.

### 5.6 Results tab and downloads

- Details view: die/grid, hard/soft/port/net counts, best-run
  breakdown, hotspots, tech node, pins/flips, placed file.
- Episode table: method / episode / proxy / wirelength / congestion /
  density / seconds, with a Δ% line versus the previous run of the
  same design. A Pareto widget shows normalised proxy/wirelength/
  congestion bars (lower is better).
- Downloads: placed `.plc`, viewer DEF, OpenROAD `.tcl`, timelapse
  GIF, floorplan/thermal/congestion PNGs, report ZIP, and Open output
  folder. Downloads are blocked until a run exists.

### 5.7 Compare (Baseline vs. AI)

- Split screen: baseline left, AI-optimised right, with proxy /
  wirelength / congestion / Fmax lines, heat grids, an amber divider,
  and Δ badges (green improved, red worsened) for congestion,
  wirelength, and peak power.
- The pair comes from the two best snapshots or from Auto-Optimize.
  Until two scored runs exist, the view shows placeholder guidance.

### 5.8 Thermal tab

- Technology Node combo (kept in sync with Settings), Clock
  (10–1200 MHz), Voltage, and Activity sliders. Updates recompute
  corner power and the hottest-macro label. The panel states the
  power formula and that it is a proxy, not signoff.

### 5.9 Execution model

- One background `Worker` on a `QThread` at a time. Starting a second
  job is refused or queued. Progress runs 0–60% during placement and
  60–100% across episodes, with an elapsed timer. Busy state disables
  Import/Run/Auto/RTL/Apply/Review buttons. Failures surface as a
  dialog plus details in Results/Import/Log.

### 5.10 Supporting renderers

- `neurosilicon/render.py` (headless matplotlib): floorplan,
  congestion (V+H), thermal, cost-breakdown, history, simulation-step,
  critical-path, comparison, and KPI-bar figures; `figure_png` at
  110 dpi; `save_figures` writes the standard PNG set.
- `neurosilicon/ui.py` serves the web console only (CSS, header,
  status pills, workflow stepper). It holds no compute logic.
- `neurosilicon/gnn_viz.py` records a policy rollout (per-step
  probabilities, value, chosen cell) for GNN scrubbing. The desktop
  canvas itself uses `liveplace` frames, not `gnn_viz`.

## 6. Web console and CLI

Entry: `streamlit run neurosilicon/app.py`. The sidebar has seven
sections (`neurosilicon/app.py`):

| Section | What it does |
| --- | --- |
| Dashboard | Toolchain pills (yosys/OpenROAD/PDK/LLM), designs table, 8 newest runs, quick-start steps |
| Run floorplan | Netlist picker, mode/checkpoint/episodes/FD/CD controls, SA + orientation options, mask/chat constraints, macro-power JSON upload, Run and Auto-Optimize |
| Results & downloads | Run picker with Figures / Inspector / Files / Validation / Curves tabs |
| Thermal lab | Tech-node rescaling, clock/voltage/activity/leakage sliders, clock-sweep animation, hotspot table |
| GNN simulation | Policy-checkpoint rollout replay with per-step heatmap and value chart |
| AI assistant | Provider status/test, Auto-Optimize, 3-Agent Review, RTL Fix, Q&A over the last run |
| Settings | Dynamic form over the five settings sections, with redacted-secrets view |

Details worth knowing:

- Run floorplan modes are `heuristic` / `neurosilicon` /
  `llm_assisted`. Policy mode falls back to `first_legal` with a note
  when the checkpoint is missing. GNN simulation has no fallback —
  the checkpoint must exist.
- Results Figures tab shows best proxy, est. Fmax / critical delay,
  canvas views including Compare, the episode table, and the KPI
  Pareto figure. Inspector shows run provenance JSON and a per-macro
  x/y/w/h/area/orientation table.
- A stale-code guard detects outdated loaded modules and offers
  Reload updated code & retry. Console exceptions are logged to
  `logs/gui_runs/console_errors.log`.
- Pure compute is shared with the desktop flow: `run_episodes()` and
  `run_auto_optimize()` hold no Streamlit calls.
- Large designs are slow: the console warns that Ariane (133 macros)
  takes roughly 9 minutes per episode. All-infeasible runs error with
  advice to loosen the mask or use the heuristic.

CLI details:

- `evaluate` runs `random` + `first_legal` episodes with optional CD
  finetune, broadcast penalty options, and `--no_init_fd`. It writes
  `best_baseline.plc` and `results.json` and prints a table.
- `train` accepts repeatable `--netlist` / `--init_placement`,
  `--iterations`, `--episodes_per_iter`, `--eval_every`, `--lr`,
  `--resume`, and broadcast options.
- `flatten_def` and `synth_to_proto` expose argparse mains for the
  conversion pipeline.

## 7. AI and MCP-style assistant

Modules: `neurosilicon/assistant.py`, `neurosilicon/llm.py`.

> Precision note: this is an **MCP-style** in-process tool surface.
> `MCP_TOOLS` is a list inside `assistant.py`. No deployable MCP
> server, transport, or entry point ships in `neurosilicon/`. The
> assistant never edits files and never runs placement itself —
> placement always runs in the local engine.

### 7.1 Tool surface

| Tool | Returns |
| --- | --- |
| `get_design_info` | Canvas size, placement grid, macro/port counts, top module |
| `get_run_results` | Episode rows and best proxy/wirelength/congestion/density |
| `get_thermal_info` | Hottest macros at the current corner (area-proportional proxy) |
| `propose_constraint` | Chat text parsed into a keep-out/allow rectangle mask |

### 7.2 Chat behaviour

- With a provider enabled, answers are grounded in MCP-style facts:
  design, canvas, best-run numbers, hottest macros, and RUDY
  congestion (overall/peak/tiles/macros). The system prompt limits
  the model to those facts and forbids claiming it ran placement.
- Connection/status/setup questions return a connection explanation
  (never including the API key).
- On any LLM failure the chat returns an offline answer prefixed
  `[Offline Mode — Answered via local EDA analyzer]`. The technical
  error goes to the log only. The offline path branches on
  constraint, thermal, congestion, best-run, and no-run questions,
  and always labels itself as the rule-based fallback.
- Quick-ask buttons cover: explain the best run, why congestion is
  high, hottest macros, and the centre-channel constraint.

### 7.3 Constraint parsing

- Online: the LLM must reply with one JSON object
  `{mode: block|allow|none, rects: [[x0,y0,x1,y1]], explanation}` in
  microns. Any LLM or parse failure falls back to the rule-based
  parser, and the fallback is noted in the explanation.
- Applying a constraint stores keep-out rectangles plus thermal
  weights. Apply AI Fix & Re-Place re-runs placement with them.

### 7.4 Auto-Optimize (AI Loop)

Available in both GUIs. Three iterations, one background job:

1. Baseline placement (`run_baselines`).
2. SAM hotspot segmentation (`top_hotspot`) plus diagnosis. The
   diagnosis asks the Muse model first, then the configured model,
   for JSON constraints. Rectangles are clamped to the canvas,
   capped at 60% of die area, and weights are never lowered
   (congestion ≥ current + 0.1 and ≥ 0.8; thermal ≥ 0.5; both ≤ 2.0).
   Any failure returns a deterministic offline diagnosis: spread hot
   macros + 20% centre channel + raised weights.
3. AI-guided re-placement with the diagnosed mask and weights. If
   the mask dead-ends, it retries weights-only. Optional SA polish
   and orientation flips follow.

On completion it posts the PPA Improvement Summary and per-iteration
notes to chat, sets the Compare pair, applies the diagnosed weights
and keep-outs, and selects the better placement. It refuses to start
when every hard macro is pinned.

### 7.5 RTL co-pilot (Suggest RTL Fix for Bottleneck)

- `find_bottleneck` prefers the top RUDY-congested hard macro (soft
  macros excluded), else the highest power-density macro.
- `find_verilog_for_macro` locates the source by exact basename, then
  `module <name>` text, then suffix (400-file walk bound). A snippet
  is quoted only when the file actually defines the macro.
- Online replies give a one-line summary, a small Verilog unified
  diff, and expected PPA impact. Offline templates cover memory
  banking (cache/SRAM/RAM), bus pipelining (bus/arbiter/mux), and
  clock gating + pipelining (everything else).

### 7.6 3-Agent Design Review

Deterministic and rule-based — no LLM call. Three roles read the same
assistant context (best run, hotspots, RUDY congestion, PPA, critical
path):

- Thermal Specialist: recommends `thermal_weight` 0.9 / 0.6 / 0.5
  from the hottest macro share.
- Routing & Timing Specialist: recommends `congestion_weight`
  1.2 / 0.8 / 0.5 and `wirelength_weight` 1.3 when est. Fmax is low.
- Lead Physical Design Architect: verdict plus the adopted weights.

The desktop applies the recommended weights to Settings. The review
text states that Fmax is a proxy, not signoff.

### 7.7 Providers and transport

- `api`: `POST {base_url}/chat/completions` with Bearer [REDACTED],
  temperature 0.2. Any OpenAI-compatible gateway works.
- `local` with a `/v1` URL: OpenAI-compatible `POST /chat/completions`
  (LM Studio shape, no key). Otherwise native Ollama
  `POST /api/chat` with `stream: false`.
- Muse-first routing: Auto-Optimize diagnosis and the RTL co-pilot
  try the Muse model (`muse-arcadia-1.0` in code) first, then the
  configured model. Plain chat uses the configured model directly.
- Resilience: 3 attempts with 0.5 s / 1 s / 2 s backoff on SSL EOF,
  connection, timeout, and HTTP-exception failures, plus 502/503/504.
  Requests send `Connection: close`. Errors are sanitised (no raw SSL
  internals, truncated) before reaching the UI.
- Test connection sends a fixed probe sentence and returns
  `(ok, message, seconds)`. List models reads `GET /models` (API /
  OpenAI-compatible local) or `GET /api/tags` (Ollama). The desktop
  renders the returned IDs as click-to-use buttons that fill, save,
  and test. Hints cover 401/403 (key) and 404/model-not-found (copy an
  exact listed ID).
- Failure contract: `chat` / `advise` / `list_models` raise `LLMError`
  for the UI to show. `parse_constraints`, `diagnose_placement`,
  `suggest_rtl_fix`, and `answer` never raise for LLM outages — they
  fall back instantly.

### 7.8 Connectors UI and Guide

- Connectors tab: Provider (off/api/local), API key (password field),
  base URL, model, local URL/model, timeout (5–600 s), Save / Test /
  List models, and a status label. Saving posts a chat message
  explaining what is connected and how the MCP-style tools work.
- Guide tab renders the built-in setup guide: cloud API, Ollama, LM
  Studio, MCP chatbox usage, and troubleshooting.
- Connector status distinguishes: off (rule-based fallback), API key
  missing, model missing, local model missing, and configured.

## 8. Import, synthesis, export, and validation

### 8.1 Discovery

`discover_design` accepts a folder, a single file, or a `.zip`
(unpacked with a zip-slip guard). It classifies the input as
`ready_netlist`, `rtl_project`, `rtl_files`, or `unknown`, and reports
RTL files, candidate tops, a suggested top, LEF/DEF/Liberty lists,
and a found/missing checklist. A `netlist.pb.txt` + `initial.plc`
pair in one folder is preferred as the ready pair
(`neurosilicon/discover.py`).

### 8.2 RTL staging and synthesis

- Folder/file scan collects `.v/.sv/.vh/.h` (case-insensitive),
  extracts module tops, and suggests the top by hint or
  root-most-instantiated heuristic. Testbenches are excluded by
  default. `prune_to_top` keeps the instantiation closure of the
  chosen top (`ifdef`-aware) (`neurosilicon/project.py`).
- Staging copies sources and headers, stubs missing `` `include ``
  files empty with a warning, and blackboxes huge-memory modules
  (≥1 Mbit) and simulation-only modules so yosys/ABC cannot blow up.
  Synthesis text is normalised for the Verilog-2005 frontend.
- Yosys flow: `read_verilog [-sv]`, `hierarchy -check -top`, `synth`,
  `dfflibmap -liberty`, `opt`, `autoname`, then `write_json` and
  `write_verilog`. Runs as a subprocess (3600 s default timeout),
  writes `synth.log`, and raises with the log tail on failure.
  SystemVerilog needs the old frontend and can fail verbatim — the
  user guide lists modern-parser cores (ibex, neorv32) as out of
  scope for the bundled yosys.
- End-to-end: `import_project` / `import_files` prune, stage, resolve
  the PDK, synthesise, convert, run baselines, and write `import.json`
  plus a persistent design registry. Ready pairs register directly
  after netlist and canvas-header validation.

### 8.3 Netlist conversion

`Converter` parses the LEF for cell sizes and pin directions, streams
synthesis JSON over 32 MB via `ijson`, and walks the hierarchy. Leaf
modules become hard macros (square, `sqrt(area/utilization)`).
Dissolved standard cells become clusters (fanout and area capped).
Nets resolve drivers and sinks. Output is `netlist.pb.txt`,
`initial.plc`, and `clusters.json` with automatic grid selection
(fallback 32×32) (`neurosilicon/synth_to_proto.py`).

### 8.4 Exports

| Export | Desktop | Web | Notes |
| --- | --- | --- | --- |
| Placed `.plc` | Yes | Yes | Copy / text download of the best placement |
| Viewer DEF | Yes | Yes | DEF 5.8, viewer-grade. Explicitly not signoff: best-effort NETS, HARDMACRO/SOFTMACRO placeholders |
| OpenROAD `.tcl` | Yes | Yes | `place_macro` per hard macro with `R0/R90/R180/R270`; desktop also writes `placed.def` alongside |
| Flat DEF | Via validation | When synth JSON exists | Cell-grade DEF by tiling leaf cells inside macro boxes. Geometric tiling without legalisation; overflow is counted, no NETS |
| PNGs | Floorplan / thermal / congestion | Plus cost-breakdown | Matplotlib at 110 dpi |
| Timelapse GIF | Yes | No | PIL, 760×540, 90 ms per frame, looping, from playback frames |
| Report ZIP | Whole run folder | Curated bundle | Web bundle: best `.plc`, viewer DEF, 4 PNGs, results, inspector, advice, redacted settings, README |

### 8.5 OpenROAD routed validation

Available when a consistent synthesis JSON + Verilog pair exists
(built-in servant/picosoc/ultraembedded and qualifying imports). The
best placement is flattened, then OpenROAD runs
`tools/validate_route.tcl` (1800 s timeout) with `NS_*` environment
variables. Success requires the `GR_DONE` marker. Overflow and
hotspots are parsed from the congestion report, which is downloadable.
The Tcl does `place_pins` and `global_route -allow_congestion`. This
is a routed check, still not signoff.

### 8.6 PDK and tool resolution

- Blank LEF/Liberty/techlef auto-detect the Ciel sky130 install
  (`sky130_fd_sc_hd`, TT 1.80 V library, nominal techlef) and are
  validated to exist. Relative tool paths resolve against the repo
  root, not the working directory.
- `tools/yosys` and `tools/openroad` are bash shims that exec local
  OpenLane-extracted binaries with `tools/oldlibs` on
  `LD_LIBRARY_PATH`. They are not containers.

### 8.7 Built-in designs

| Design | Macros | Source fixtures |
| --- | --- | --- |
| `toy` | 2 | `circuit_training/environment/test_data/toy_macro_stdcell` |
| `ariane` | 133 | `circuit_training/environment/test_data/ariane` |
| `servant SoC (sky130)` | 18 | Synthesis-backed (flat DEF + validation available) |
| `picorv32` | 2 + clusters | Netlist + placement pair |
| `picosoc` | 6 + clusters | Synthesis-backed |
| `ultraembedded riscv_core` | 10 + clusters | Synthesis-backed |

Imported designs join the same lists through the local registry.

## 9. Settings, persistence, and run artifacts

### 9.1 Settings

Settings live outside the repo at
`~/.config/neurosilicon/settings.json` (`XDG_CONFIG_HOME` honoured).
API keys never enter the repository or logs. Loading merges over
defaults and ignores unknown keys. Saving writes only known keys,
atomically via a temp file plus rename. Secrets display as `(set)` /
`(empty)` through `redacted()` (`neurosilicon/settings.py`).

| Section | Fields (defaults) |
| --- | --- |
| `run` | mode `heuristic`, episodes 3, FD steps 10, seed 0, outdir `logs/gui_runs` |
| `engine` | wirelength 1.0, congestion 0.5, density 1.0, broadcast 0.0 (threshold 16), thermal 0.0, infeasible penalty 4.0 |
| `model` | checkpoint `logs/joint4_train/best.pt` |
| `llm` | provider `off`, API key (secret), base URL `https://api.openai.com/v1`, model `gpt-4o-mini`, local URL `http://localhost:11434`, local model `llama3.1`, timeout 60 s |
| `pdk` | LEF/Liberty/techlef blank (sky130 auto-detect), yosys `tools/yosys`, OpenROAD `tools/openroad`, utilisation 0.8, canvas utilisation 0.35 |

The desktop Settings tab edits PDK paths, engine weights
(0.0–10.0, step 0.1), the PPO checkpoint, and the technology node.
The web Settings page renders the same schema dynamically. Run
controls in Place/Run pre-fill from these defaults.

### 9.2 Run persistence and artifacts

- Desktop runs: `logs/desktop_runs/` (including `auto_baseline/` and
  `auto_ai/` for Auto-Optimize). Web runs: `logs/gui_runs/runs/<id>/`
  with `run.json`, `best.plc`, and `blocked.npy`. They survive tab
  switches, refreshes, and restarts.
- Every CLI/import output folder is self-describing: `manifest.json`
  (tool, version 1.0.0, git hash, start time, argv, resolved args),
  `run.log`, plus `results.json` and `best_baseline.plc` for
  evaluation, or `history.csv` / `latest.pt` / `best.pt` for training
  (`neurosilicon/runlog.py`).
- The web console also keeps a training-history browser over
  `logs/*/history.csv` with a downloadable curve PNG.

## 10. Tests

- 135 test functions across 19 `*_test.py` files in
  `neurosilicon/tests/` (counted by grep and corroborated against the
  README claim during this analysis).
- Coverage by area: env MDP/masking/padding, GNN shapes/mask/Dirichlet,
  PLC parsing/costs/blockages, priors and chat masks, tiny PPO and
  cosine LR, baselines and policy rollout, CD finetune, project
  scan/synth/import/registry, settings and LLM request shapes, DEF/PNG
  export, desktop discovery/assistant/live rollout, Auto-Optimize and
  RTL templates, SA polish/orientations/pins/tech presets/OpenROAD Tcl,
  and the 3-agent review.
- Run them with `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q`.
  One env test skips itself when the openc906 design has not been
  imported on that machine. That skip is by design (`README.md`).

## 11. Upstream provenance and scope

- NeuroSilicon builds on AlphaChip / Circuit Training (Mirhoseini et
  al., Nature 2021). The original TensorFlow / TF-Agents stack and the
  retired `plc_wrapper_main` binary were replaced by a Python-native
  runtime over the vendored BSD-3 TILOS PLC reimplementation. Project
  licence is Apache 2.0; vendored licences are preserved.
- `circuit_training/` is provenance only. No `neurosilicon/` runtime
  code imports it. Two exceptions are data, not code:
  `circuit_training/environment/test_data/` fixtures (toy, ariane)
  are used by the apps and tests.
- In-scope tooling: `tools/yosys`, `tools/openroad`,
  `tools/validate_route.tcl`, `tools/oldlibs`.
- Out-of-scope upstream artefacts: `tools/docker/` images, DREAMPlace
  build scripts, `tools/e2e_smoke_test.sh`. `tools/nix-run.sh` is a
  sandbox-only workaround, not a user feature.
- Reference docs: `docs/ARIANE.md` (upstream scale example),
  `docs/BLOCKAGE.md`, `docs/NETLIST_FORMAT.md`,
  `docs/PLACEMENT_COST.md`, `docs/PRETRAINING.md`.

## 12. Explicit non-features and limits

- Not signoff: proxy cost, RUDY congestion, thermal/power, and est.
  Fmax are interactive proxies. Routed validation is a global-route
  check, not signoff.
- Not an MCP server: the assistant is MCP-style and in-process only.
- Viewer DEF is for KLayout/OpenROAD viewing, not tapeout. Flat DEF
  tiles cells geometrically without legalisation.
- Converter geometry is approximate: square macros, centred cluster
  pins, perimeter macro pins, square canvas. Huge-memory and
  simulation-only blackbox sizes are nominal.
- Single-job desktop: one background job at a time.
- Scale: force-directed cost grows ~n². Very large designs need
  reduced FD steps and clustering; the docs record per-design timings
  and a deferred-baseline path.

## Evidence limits

This document synthesises a multi-agent code inspection. The
following remain unproven here and should not be read as verified:

- Runtime behaviour of the desktop GUI, Streamlit console, PPO
  training, yosys synthesis, and OpenROAD validation (no launches in
  this analysis; your Auto-Optimize run on 2026-10-30 is separate
  first-hand evidence for that one flow).
- Test pass/fail status (tests were inventoried and counted, not run
  by the analysis workflow).
- `PlacementCost` internals in `neurosilicon/thirdparty/tilos/`
  (5,428 lines) — proxy-cost claims rest on the calling code, not a
  line-by-line read of that core.
- Known doc drift, flagged not resolved: `NEUROSILICON_HOW_IT_WORKS.md`
  still contains upstream Docker/TF-Agents steps and a stale 89-test
  figure (suite is now 135); `NEUROSILICON_README.md` predates the
  desktop-primary layout; `NeuroSilicon eg features.md` is an
  aspirational blueprint, not a spec of shipped code;
  `docs/PLACEMENT_COST.md` grid default (100×100) differs from the
  NeuroSilicon `default_grid` of 32. Whether the desktop upgrade
  ships inside version 1.0.0 or above it (`CHANGELOG.md`
  `[Unreleased]` vs `_version.py` 1.0.0) is ambiguous in the tree.
- `PRINCIPLES.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md` are
  community/governance files and were out of feature scope.
