# NeuroSilicon Operator Guide

Day-to-day use of the NeuroSilicon console and CLI. For the system
design, see `NEUROSILICON_HOW_IT_WORKS.md`; for commands and results,
see `NEUROSILICON_README.md`.

## Start the desktop app (primary)

NeuroSilicon runs as a native desktop program (Qt window) — no web
server, no browser:

```bash
cd /path/to/neuro-silicon
.venv/bin/python -m neurosilicon.desktop
```

Import a chip from **Import chip**: select the whole design — a
project folder, any file inside it, or a `.zip` (drag-and-drop onto
the window works too). Auto-detect fetches what floorplanning needs:
a ready `netlist.pb.txt` + `initial.plc` pair places directly; RTL
gets an auto-picked top module and yosys synthesis. Then
**Place → Run floorplan**, watch the macros land live on the native
canvas (Play/Pause, step, scrub, speed; Floorplan / Congestion /
Thermal / Wires views), replay the run, and download the placed
`.plc`, DEF, PNGs, or a report ZIP from **Results & Downloads**.

Optional AI: **AI Connectors** (right panel) enables a cloud
OpenAI-compatible API or a local Ollama / LM Studio model — off by
default, keys stay in `~/.config/neurosilicon/settings.json`. The
**Guide** tab walks through setup; the **AI Assistant** chatbox reads
the current floorplan (MCP-style tools), explains runs, ranks thermal
hotspots, and turns text like “keep a 20% routing channel open in
the center” into a keep-out mask for the next run. It also answers
offline with a rule-based fallback.

### Legacy web console (retained, not the default)

```bash
source .venv/bin/activate
streamlit run neurosilicon/app.py
```

Open http://localhost:8501. The **Dashboard** shows toolchain status
(yosys, OpenROAD, PDK files, LLM provider), available designs, and
recent runs. Red pills name missing tools — fix the path in
Settings > PDK & Toolchain.

## Standard workflow

1. **Run floorplan** — pick a design, choose heuristic /
   neurosilicon-policy / llm-assisted, press **Run floorplan**.
   Episodes stream live; the best placement is kept.
2. **Results & downloads** — Figures (floorplan, congestion, thermal,
   cost breakdown), Inspector (every macro, provenance), Files (.plc,
   viewer + flat DEF, report ZIP), Validation (OpenROAD global
   route), Curves (training history).
3. **Thermal lab** — clock/voltage/activity sliders over the current
   best placement (documented proxy, not signoff).
4. **Report bundle** — one ZIP with PNGs, DEFs, results, advice, and
   redacted settings.

Every finished run is saved under `logs/gui_runs/runs/<run_id>/`
and survives tab switches, refreshes, and restarts; switch runs with
the picker on Results.

## Import a chip project

On Run floorplan, top section: enter the RTL folder (Browse… for a
native picker), tick files, choose the top module, set `NAME=value`
parameters if needed, press **Import and baseline**. The flow runs
yosys synth → proto conversion → baselines and registers
`imported: <name>` in the picker. Verilog is fully supported;
SystemVerilog depends on the yosys parser.

## CLI (batch / training)

```bash
export MPLCONFIGDIR=/tmp/mpl
# Baselines
.venv/bin/python -m neurosilicon.evaluate --netlist <pb> \
  --init_placement <plc> --episodes 5 --outdir logs/eval
# PPO training (repeat --netlist/--init_placement for joint runs)
.venv/bin/python -m neurosilicon.train --netlist <pb> \
  --init_placement <plc> --outdir logs/train --iterations 20 \
  --episodes_per_iter 8
# Tests
.venv/bin/python -m pytest neurosilicon/tests/ -q
```

Every output directory carries `manifest.json` (tool, version, git
hash, args) and `run.log` (timestamped log).

## Routined checks

- **Before a demo**: Dashboard all green, one fresh run placed.
- **After a code update**: if the console reports a stale server,
  press **Reload updated code & retry**, else restart Streamlit.
- **Console errors**: unexpected page errors append tracebacks to
  `logs/gui_runs/console_errors.log`; saved runs are unaffected.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `yosys binary not found` | Set the absolute yosys path in Settings > PDK & Toolchain |
| `PDK file missing` | Point lef/liberty/techlef at the sky130 install (volare `~/.ciel`) |
| All episodes infeasible | Loosen the constraint mask; try heuristic / first_legal |
| Flat DEF unavailable | Needs the synth JSON (sky130 built-ins + imports only) |
| GNN rollout dead-ends | That policy cannot place this design greedily; try another checkpoint |
| LLM advice unavailable | Configure Settings > AI Providers (API key or local Ollama) |

## Limits

Thermal/power figures are area-proportional proxies, not signoff
analysis. Joint policies can underperform per-design specialists
(the console pre-fills the verified checkpoint per design).
SystemVerilog cores needing a modern parser (ibex, neorv32) are out
of scope for the bundled yosys.
