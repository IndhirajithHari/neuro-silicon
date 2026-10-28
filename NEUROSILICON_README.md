# NeuroSilicon (Python-native)

AI floorplanner: sequential macro placement with an edge-GNN policy trained
by PPO, scored by an open-source proxy-cost engine (wirelength + congestion
+ density), with force-directed standard-cell relaxation. No TensorFlow, no
proprietary binaries, CPU-friendly.

Branch: `neurosilicon/base` (restored AlphaChip reference lives in
`circuit_training/`; the runnable system is `neurosilicon/`).

## Layout

- `neurosilicon/setup.py` — builds a configured `PlacementCost`.
- `neurosilicon/env.py` — Gymnasium env (sequential placement MDP).
- `neurosilicon/model.py` — PyTorch edge-GNN actor-critic.
- `neurosilicon/train.py` — masked PPO trainer.
- `neurosilicon/evaluate.py` — baselines + results tables.
- `neurosilicon/priors.py` — keep-outs, chat constraints, thermal cost.
- `neurosilicon/render.py`, `export_def.py` — plots + viewer DEF.
- `neurosilicon/app.py` — Streamlit GUI.
- `neurosilicon/thirdparty/tilos/` — vendored open-source PLC core
  (BSD-3, see `NOTICE.md` + `LICENSE`).
- `neurosilicon/tests/` — pytest suite (`*_test.py`).

## Environment

A project `.venv` (Python 3.14) is already set up. To recreate:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib pytest gymnasium absl-py gin-config streamlit
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
export MPLCONFIGDIR=/tmp/mpl   # matplotlib cache (sandbox-friendly)
```

## Commands (run from repo root)

```bash
# Tests
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q

# Baselines (toy: seconds; ariane: minutes, CPU)
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.evaluate \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --episodes 5 --outdir logs/toy_baseline

# Train PPO (toy smoke: 1 iter; scale up for real runs)
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.train \
  --netlist .../toy_macro_stdcell/netlist.pb.txt \
  --init_placement .../toy_macro_stdcell/initial.plc \
  --outdir logs/train --iterations 10 --episodes_per_iter 4

# Desktop app (primary)
.venv/bin/python -m neurosilicon.desktop
# Legacy web console
source .venv/bin/activate
streamlit run neurosilicon/app.py
# then open http://localhost:8501 in a browser
```

Pages: **Dashboard** (toolchain status pills, designs, recent runs,
quick start), **Run floorplan** (heuristic / neurosilicon-policy /
LLM-assisted modes — our model is optional; project import lives at
the top of this page: folder scan → yosys synth → proto → baselines,
registered into the netlist picker), **Results & downloads** (tabbed Figures / Files / Validation
/ Curves (+ Inspector tab with full macro table, chip counts, and
run provenance): floorplan/congestion/thermal/cost PNGs, .plc, viewer +
flat DEF, one-click OpenROAD global-route validation, full-report ZIP,
training curves (proxy + wirelength/congestion/density +
AlphaChip-style top-10% overflow curves)), **Thermal lab** (live clock/voltage/activity corner
sliders + hotspot monitor), **GNN simulation** (step-through policy
rollout with placement heatmaps, value curve, and circuit-graph
overlay), **AI assistant** (LLM review + Q&A on real runs), **Settings**
(industrial sections with per-field explanations, saved outside the
repo at `~/.config/neurosilicon/settings.json`).

AI providers (Settings > AI Providers): `off` (rule-based parsing),
`api` (OpenAI-compatible key + base URL + model), `local` (Ollama
native `/api/chat` or `/v1` OpenAI-compatible servers like LM Studio).
Thermal maps use an uploaded per-macro power JSON when given, else a
documented area-proportional estimate.

Industrial footing: versioned releases (`neurosilicon/_version.py`,
`CHANGELOG.md`), dark theme (`.streamlit/config.toml`), structured
logging + `manifest.json`/`run.log` in every train/evaluate/import
outdir, CI running the pytest suite (`.github/workflows/ci.yml`),
and an operator guide (`NEUROSILICON_USER_GUIDE.md`). Unexpected
console errors are caught per page and appended to
`logs/gui_runs/console_errors.log` without losing saved runs.

Console extras: **Thermal lab** page (clock/voltage/activity/leakage
sliders, live balance score, hotspot table, animated clock sweep, PNG
download; model `P = C·(V/1.8)²·(f/100)·(α/0.2) + leakage·area`),
native **Browse…** folder picker on Import (zenity → tkinter → typed
path fallback), and `?` help on every control. Runs stream live
(per-episode rows + best-so-far floorplan as each lands, with run ID,
timestamp, seed, and per-episode seconds), neurosilicon mode
pre-fills the verified-good checkpoint per design, and ariane carries
a time warning. Result rows also carry `top10cong` (AlphaChip-style
mean overflow of the worst-10% grid cells), Inspector shows placed
macro utilization, the floorplan overlay draws the active keep-out
mask (hatched grey), and GNN simulation shows per-step placement
order badges with an order caption row. Every finished run is saved
to `logs/gui_runs/runs/<run_id>/` (run.json + best.plc + mask) and
auto-restored, with a run picker on Results — tab switches,
refreshes, and restarts never lose results. If the console ever
shows a stale-server error after a code update, press its
**Reload updated code & retry** button (hot-reloads modules in place);
if that is unavailable, stop with Ctrl+C and re-run
`streamlit run neurosilicon/app.py`.

## RTL to training (RISC-V cores)

Yosys runs via `tools/yosys` (binary extracted from the local OpenLane
image layers + compat libs in `tools/oldlibs`; no containers needed).

```bash
# 1. Synthesize (example: servant SoC, small RAM for demo scale)
tools/yosys -p "read_verilog src/*.v; synth -top servant; \
  abc -liberty $SKY130TT; dfflibmap -liberty $SKY130TT; abc -liberty $SKY130TT; \
  opt; stat; write_json servant.json"
# 2. Convert to floorplan format (macros + greedy stdcell clusters)
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.synth_to_proto \
  --json servant.json --lef $SKY130LEF --top servant \
  --outdir logs/servant_proto --utilization 0.8 --canvas_util 0.35
# 3. Baselines + PPO (repeat --netlist/--init_placement for joint training)
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.evaluate \
  --netlist logs/servant_proto/netlist.pb.txt \
  --init_placement logs/servant_proto/initial.plc --episodes 5 \
  --outdir logs/servant_baseline
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.train \
  --netlist logs/servant_proto/netlist.pb.txt \
  --init_placement logs/servant_proto/initial.plc \
  --outdir logs/servant_train --iterations 20 --episodes_per_iter 8
```

Where `$SKY130TT` is `sky130_fd_sc_hd__tt_100C_1v80.lib` and `$SKY130LEF`
is `sky130_fd_sc_hd.lef` in the local volare PDK (`~/.ciel`).

## Results

Toy netlist (2 macros, 12 stdcells, 12 ports), 5 episodes each:

| method      | best proxy | wl     | cong   | dens   |
|-------------|------------|--------|--------|--------|
| random      | 0.9782     | 0.1360 | 0.7334 | 0.4755 |
| first_legal | 1.0568     | 0.2138 | 0.7337 | 0.4761 |

Ariane (133 hard macros, 799 soft, 1231 ports), 3 episodes each,
full FD (30,30,30), ~9 min/episode on CPU:

| method      | proxy  | wl     | cong   | dens   | note                    |
|-------------|--------|--------|--------|--------|-------------------------|
| random      | -      | -      | -      | -      | all 3 dead-ended (mask emptied mid-episode) |
| first_legal | 1.7762 | 0.1057 | 1.4065 | 0.9671 | deterministic reference |

Toy PPO (15 iters x 8 eps): trains end-to-end, greedy eval cost 1.1056
vs random best 0.9782 on the tiny 2-macro toy (little headroom to learn).

Servant SoC, sky130 (18 macros, 398 nets), 5 episodes each:

| method      | proxy  | wl    | cong   | note               |
|-------------|--------|-------|--------|--------------------|
| random      | ~1.31  | ~0.44 | ~0.74  | 3/5 complete       |
| first_legal | 1.2096 | 0.337 | 0.744  | deterministic      |
| first_legal+CD | 1.148 | —  | —      | coordinate-descent tuned |
| PPO-20      | 1.246  | —     | —      | best greedy eval |
| PPO-80      | 1.2135 | —     | —      | best eval at iter 67; ties first_legal, entropy 5.8→1.1 |

OpenROAD routed validation (servant, sky130, `flatten_def` + global_route):

| placement   | proxy  | GR overflow | hotspots | note |
|-------------|--------|-------------|----------|------|
| servant first_legal | 1.2096 | 3266   | 953      | `logs/servant_route_cmp/gr_cong_fl.rpt` |
| servant first_legal+CD | 1.148 | 2747 | 767      | `logs/servant_baseline/gr_cong.rpt` |
| servant first_legal+CD + detailed_place | 1.148 | 4171 | 896 | legalized, `logs/servant_baseline/gr_cong_dp.rpt` |
| picosoc first_legal | 1.2257 | 521179 | 20560 | clk + cpu↔RAM broadcast dominated; 179210 excl. those |
| servant joint-policy+CD | 1.212 | 477 | 302 | equal proxy, 7x less overflow than first_legal |
| picorv32-v2 first_legal | 1.4731 | 119734 | 9211 | consistent autoname re-synth |
| picorv32-v2 joint-policy | 1.3388 | 116809 | 8655 | better proxy, marginally better routed |
| picosoc joint-policy | 1.1973 | 508774 | 20171 | beats first_legal on proxy and routed |
| ue random-best | 1.2233 | 78774 | 10365 | clk_i/rst_i dominated; lighter macro interiors than picosoc |
| ue joint4-policy | 1.2528 | 80480 | 9115 | worse proxy and routed than random-best; ordering consistent |

Proxy cost tracks routed overflow within a design (servant: 5% better
proxy → 16% less GR overflow), but absolute overflow is not comparable
across designs: picosoc's is dominated by no-CTS clock + wide cpu↔RAM
broadcast nets that the fanout-insensitive proxy under-predicts.
Learned placements can also break the trend: the joint policy's servant
placement ties first_legal on proxy (1.212 vs 1.2096) yet routes 7x
cleaner (477 vs 3266) — routed validation stays the arbiter.
Next proxy upgrade: fanout-aware congestion / CTS-mode clock handling.

Broadcast-penalty upgrade (shipped, experimental, default off):
`priors.broadcast_penalty` (cluster-level span x log fanout) is wired into
`extra_costs` via `--broadcast_weight/--broadcast_threshold` in `train`,
`evaluate`, and `finetune`. `Converter.flat_net_spans` +
`priors.flat_broadcast_penalty` add true flat fanout x cluster span.
Validation on the six routed placements was NEGATIVE for ranking:
flat load ranks pico32v2 (5837) above picosoc (3313) while routed
overflow ranks them 117k < 509k — the gap is macro-interior routing
(flop-RAM interiors are tiled flat by `flatten_def` but solid blocks to
the proxy), invisible to any placement-level term. Real fix: a static
macro-interior load estimate added to the proxy (future work); the
broadcast terms remain useful within-design co-location guides.
Flow: DEF (rows + margin) → link → `flat.def.place.tcl` sets locations →
power nets marked special → `place_pins` → `global_route -allow_congestion`.
Reusable script `tools/validate_route.tcl` (env: NS_LIB/NS_TLEF/NS_LEF/
NS_VER/NS_DEF/NS_PLACE/NS_TOP/NS_RPT, optional NS_RUN_DP=1);
`flatten_def` now also emits `<out>.nopins.def` for validation.

picorv32, sky130 (2 macros + 511 stdcell clusters, 7428 nets):

| method      | proxy  | wl    | cong   | dens   |
|-------------|--------|-------|--------|--------|
| first_legal | 1.4905 | 0.261 | 0.878  | 0.791  |

picosoc (picorv32 + UART/SPI/mem), sky130 (6 macros + 485 clusters,
7025 nets, canvas 914.8): synthesized 24998 cells
(`logs/picosoc_build/`), proto in `logs/picosoc_proto/`, GUI entry added.

| method      | proxy  | wl    | cong  | dens  | note        |
|-------------|--------|-------|-------|-------|-------------|
| random      | 1.2661 | 0.255 | 0.883 | 0.570 | best of 3, 3/3 complete |
| first_legal | 1.2257 | 0.241 | 0.815 | 0.578 | deterministic |

ultraembedded riscv_core, sky130 (10 macros + 225 clusters, 3901 nets,
canvas 780.3): synthesized 25218 cells (`logs/ue_build/`),
proto in `logs/ue_proto/`, GUI entry added.

| method      | proxy  | wl    | cong  | dens  | note        |
|-------------|--------|-------|-------|-------|-------------|
| random      | 1.2233 | 0.286 | 0.835 | 0.520 | best of 3; beats first_legal |
| first_legal | 1.2879 | 0.289 | 0.862 | 0.568 | deterministic |

Joint training (`logs/joint3_train`, 20 iters x 3 eps,
servant+picorv32+picosoc round-robin, fd_steps 5): eval (3-net mean)
1.3246 → 1.3011 → 1.3937 → 1.3063 at iters 4/9/14/19; best `best.pt`
at iter 9. (Earlier `joint_train`/`joint_train2` died with their sandbox
namespaces — `setsid` does not survive here; long runs must be
session-managed foreground commands.)

Joint-4 (`logs/joint4_train`, 20 iters x 4 eps, +ultraembedded):
eval (4-net mean) 1.3419 → 1.2919 → 1.2762 → 1.2771, best at iter 14.
Per-design greedy: picorv32-v1 1.4008 (beats FL 1.4905), picosoc 1.1752
(beats joint3 policy 1.1973 and FL 1.2257), UE 1.2528 (between random
1.2233 and FL 1.2879), servant INFEASIBLE (greedy dead-end — negative
transfer from the 4th net on servant's tight canvas).

Full tables: `logs/toy_baseline/results.json`, `logs/ariane_baseline/results.json`.

## Notes and deviations

- Google's `plc_wrapper_main` binary is no longer publicly downloadable and
  the TF stack needs Python <= 3.11, so the runtime is Python-native on top
  of TILOS's open-source PLC reimplementation (vendored, BSD-3).
- Raw stdcells are kept as small soft macros (nets, FD, costs all cover them).
- Padded 128x128 canvas puts the real grid at top-left (Google centers it).
- DEF export is viewer-grade, not signoff-grade.
