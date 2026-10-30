<div align="center">

# ⚡ NEUROSILICON

### *Autonomous AI Chip Floorplanning with Graph Neural Networks & Deep Reinforcement Learning*

<p align="center">
  <b>An open-source, interactive EDA workstation that places hard macros the way a learning agent plays a game — scored on wirelength, congestion, density, and thermal gradients live on your desktop.</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%20|%203.14-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CPU%20Friendly-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="neurosilicon/tests/"><img src="https://img.shields.io/badge/Tests-136%20Passed-success?logo=pytest&logoColor=white" alt="Tests"></a>
  <a href="#-architectural-workflow"><img src="https://img.shields.io/badge/RL%20Policy-Edge--GNN%20%2B%20Masked%20PPO-8A2BE2" alt="Architecture"></a>
  <a href="#-native-workstation-gui"><img src="https://img.shields.io/badge/Workstation-PySide6%20Qt%20%7C%20Streamlit-00BC8C" alt="GUI"></a>
  <a href="#-open-eda-interoperability"><img src="https://img.shields.io/badge/Toolchain-Yosys%20%7C%20OpenROAD%20%7C%20Sky130-107C41" alt="Open EDA"></a>
</p>

---

</div>

## 💡 Overview: The Game of Silicon

Floorplanning is the pivotal first hour in physical chip design. Before standard cells can be placed and detailed signal wires can be routed, an engineer must position the massive building blocks — SRAM memories, register files, and execution cores — onto a bare silicon die. A mistake here locks in long wire delays, routing congestion hot spots, and thermal hazards that no downstream EDA pass can ever repair.

Traditionally, this critical step is held behind expensive commercial licensing, high-barrier scripting, and lengthy manual design iterations.

**NeuroSilicon re-imagines floorplanning as an open, observable, learning game:**

* 🎮 **Sequential Placement as a Reinforcement Learning MDP:** Macros are placed sequentially onto a 128×128 masked grid by an Edge-GNN policy trained with Masked PPO. Illegal placements are dynamically masked out in real time.
* ⚡ **Live Force-Directed Physics:** Standard cells continuously float and relax around placed macros, calculating transparent proxy costs (wirelength, density, RUDY routing congestion, and thermal gradients).
* 🤖 **Neuro AI Co-Pilot:** An integrated EDA intelligence capable of explaining physical trade-offs, enforcing conversational natural-language constraints, diagnosing bottlenecks, and executing a self-driving **Auto-Optimize Loop** (Baseline ➔ SAM Hotspot Diagnostics ➔ Constrained Re-Placement).
* 💻 **100% Open-Source & CPU-Native:** Free of commercial EDA license locks, closed binaries, or heavy cloud GPU dependencies. Runs from Verilog to placed DEF in minutes on standard Linux machines.

> [!NOTE]
> **Exploration, Not Signoff:** NeuroSilicon is built for the rapid exploratory phase *before* expensive signoff tooling is invoked. Every metric is an honest, mathematically rigorous proxy designed for interactive optimization.

---

## 🗺️ Architectural Workflow

```mermaid
flowchart TD
    subgraph INGESTION ["1. Design Ingestion"]
        RTL["Verilog / SystemVerilog\n(e.g., Amber, Servant, Ibex)"] --> YOSYS["Yosys Synthesis\n(Bundled / System)"]
        YOSYS --> PROTO["Proto Netlist Builder\n(.pb.txt + .plc)"]
        READY["Ready Netlist Pair\n(ariane, toy, picorv32)"] --> PROTO
    end

    subgraph ENGINE ["2. AI Placement Engine"]
        PROTO --> MDP["Gymnasium 128x128 MDP"]
        MDP --> GNN["Edge-GNN Actor-Critic\n(PyTorch CPU)"]
        GNN --> MASK["Dynamic Legal Cell Masking"]
        MASK --> PLACE["Sequential Macro Placement"]
        PLACE --> RELAX["Force-Directed Standard-Cell Relaxation"]
    end

    subgraph POLISH ["3. Physical Polish & Kinetics"]
        RELAX --> SA["Simulated Annealing\nMacro Swaps"]
        SA --> FLIP["0° / 90° / 180° / 270°\nOrientation Optimizer"]
        FLIP --> PROXY["Multi-Objective Proxy Engine\nWirelength + RUDY Congestion + Density + Fmax"]
    end

    subgraph AI_LOOP ["4. Neuro AI & Autonomous Loop"]
        PROXY --> REVIEW["3-Agent Design Review\n(Thermal, Routing, Lead Architect)"]
        PROXY --> AUTOLOOP["⚡ Auto-Optimize Loop"]
        AUTOLOOP --> SAM["SAM Hotspot Segmentation"]
        SAM --> ADVICE["Bottleneck Diagnosis & RTL Advice"]
        ADVICE --> REPUTE["Constrained AI Re-Placement"]
    end

    subgraph OUTPUTS ["5. Export & Verification"]
        REPUTE --> DEF["Standard DEF Export\n(Flat & Viewer)"]
        REPUTE --> TCL["OpenROAD Placement Script\n(.tcl with R0..R270)"]
        REPUTE --> ROUTE["OpenROAD Global Route Validation\n(Optional Reality Check)"]
    end
```

---

## 📊 How NeuroSilicon Compares

| Dimension | Commercial EDA Systems | AlphaChip (2021 Research Snapshot) | NeuroSilicon (v1.0.0 Workstation) |
| :--- | :--- | :--- | :--- |
| **Licensing** | Proprietary (\$100k+ / seat) | Research snapshot (TF/TF-Agents) | **100% Free & Open-Source (Apache 2.0)** |
| **Hardware** | Massive Server Clusters | Multi-GPU + Reverb distributed clusters | **Runs on any modern Linux Laptop / CPU** |
| **Interactivity** | Batch CLI / Heavy legacy GUI | Batch command-line scripts | **Native QPainter Desktop Workstation + Web UI** |
| **Design Ingestion** | Complex proprietary TCL scripts | Hardcoded Protobuf netlists | **Drag-and-Drop RTL, folders, or `.zip` files** |
| **AI Co-Pilot** | None / Generic cloud chatbot | None | **Neuro AI (Conversational, Tool-Grounded, Offline)** |
| **Self-Optimization** | Manual engineering iterations | Offline RL training scripts | **⚡ One-Click Autonomous Auto-Optimize Loop** |
| **Technology Nodes** | Closed proprietary PDKs | Hardcoded technology parameters | **Sky130, ASAP7, GF180MCU dynamic presets** |

---

## 🌟 Key Highlights

### 🖥️ Native Workstation GUI
* **Multi-Layer QPainter Canvas:** Inspect floorplans with sub-millisecond redraws across **Floorplan**, **Congestion (RUDY)**, **Thermal**, **Flightlines (Wires)**, and **Baseline vs. AI Compare** split-views.
* **Interactive Kinetics:** Step through placement rollouts with Play/Pause, speed control, and scrub bars.
* **Macro Pinning:** Right-click or drag macros to lock them in place while the RL policy places remaining components around them.

### 🧠 Neuro AI Assistant
* **Context-Aware Insights:** Grounded in live simulation metrics (never hallucinates file writes or placement numbers).
* **Conversational Constraints:** Speak your physical intent (*"Keep a 20% routing channel open in the center"* or *"Spread hot macros to the perimeter"*).
* **Type-Ahead & Suggestion Dropdown:** Auto-completes design queries with intelligent engineering presets.
* **100% Offline Resilience:** Operates with a deterministic rule-based engine when no LLM is configured; effortlessly connects to local **Ollama** / **LM Studio** or cloud API endpoints when desired.

### ⚡ Autonomous Auto-Optimize Loop
Press **⚡ Auto-Optimize (AI Loop)** to launch an autonomous three-phase refinement:
1. **Phase 1: Baseline Acquisition** ➔ Simulates heuristic or unconstrained rollouts.
2. **Phase 2: SAM Segmentation & Diagnosis** ➔ Pinpoints congestion centers, identifies bottleneck hard macros, and formats constraint keep-out zones.
3. **Phase 3: AI-Guided Re-Placement** ➔ Enforces optimal placement, executes simulated annealing, and presents a **PPA Improvement Summary** with side-by-side delta badges.

### 🛠️ RTL Co-Pilot & 3-Agent Review
* **RTL Synthesis Co-Pilot:** Identifies thermal/wirelength bottleneck instances, locates their source Verilog, and generates synthesizable architectural patches (memory banking, pipeline slicing, clock gating).
* **3-Agent Review:** Simulates an engineering review board (*Thermal Engineer*, *Routing Specialist*, *Lead Architect*) to dynamically balance proxy cost weights.

---

## 🚀 Quick Start (in Under 2 Minutes)

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/IndhirajithHari/neuro-silicon.git
cd neuro-silicon

# Create an isolated Python 3.12+ environment
python3 -m venv .venv

# Install dependencies (CPU PyTorch keeps it light and portable)
.venv/bin/pip install numpy scipy matplotlib pytest gymnasium absl-py \
    gin-config streamlit PySide6 Pillow
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch

# Avoid sandbox cache warnings
export MPLCONFIGDIR=/tmp/mpl
```

### 2. Verify with the Test Suite

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest neurosilicon/tests/ -q
```
*Expected Result:* **136 passed, 1 skipped** *(openc906 skips until imported)*.

### 3. Launch the Desktop Workstation

```bash
.venv/bin/python -m neurosilicon.desktop
```

*(For headless servers or remote SSH, launch with `QT_QPA_PLATFORM=offscreen` or use the Web Console below).*

### 4. Or Launch the Streamlit Web Console

```bash
.venv/bin/python -m streamlit run neurosilicon/app.py --server.port 8765
```
Open **`http://localhost:8765`** in your browser.

---

## 📦 Ingesting Custom Designs

You can bring your own chip from RTL to floorplan across four flexible ingestion methods:

| Ingestion Method | How It Works | Ideal Usage |
| :--- | :--- | :--- |
| **Drag & Drop** | Drag any folder, `.v`/`.sv` file, or `.zip` directly onto the workstation window | Rapid testing of external open-source cores |
| **Neuro AI Chat Button** | Open the right panel ➔ **Neuro AI** ➔ click `Import project file…` | Guided, interactive import while chatting |
| **Conversational Command** | Type `import /path/to/my_chip_rtl` directly in the chat box | Fast keyboard-driven workflow |
| **Import Tab** | Select Center Panel ➔ **Import** ➔ Browse ➔ Auto-Detect ➔ Import | Deep review of detected top modules and hierarchy |

* **Ready Netlists (`.pb.txt` + `.plc`):** Registers in milliseconds with zero synthesis overhead.
* **Raw RTL (`.v` / `.sv`):** NeuroSilicon automatically discovers module hierarchies, detects top modules, filters testbenches, runs Yosys synthesis, and generates floorplanning netlists on the fly.
* **Pre-bundled Reference Chips:** Includes `toy` (2 macros), `ariane` (133 macros), `servant SoC` (Sky130), `picorv32`, `picosoc`, and `ultraembedded riscv_core`.

---

## 🔌 AI Connectors (Optional & Privacy-First)

NeuroSilicon works **100% offline out-of-the-box**. No API keys are required. If you want to connect a foundation model, open **Connectors** in the desktop right panel:

* **Local & Private:** Connect to **Ollama** (`http://localhost:11434`) or **LM Studio** (`http://localhost:1234/v1`). Zero tokens leave your workstation.
* **Cloud API:** Point to any OpenAI-compatible inference gateway (API Key + Base URL + Model Name).
* **Storage Safety:** Credentials are saved exclusively in your personal `~/.config/neurosilicon/settings.json` (strictly isolated from git tracking).

---

## 📁 Repository Layout

| Directory / File | Description |
| :--- | :--- |
| `neurosilicon/desktop.py` | Native PySide6 Qt Workstation (Canvas, Pins, Compare, Exports) |
| `neurosilicon/app.py` | Modern Streamlit Web Console |
| `neurosilicon/env.py` | Gymnasium 128×128 placement MDP |
| `neurosilicon/model.py` & `train.py` | Edge-GNN PyTorch policy & masked PPO trainer |
| `neurosilicon/liveplace.py` | Physics relaxation, RUDY congestion, SAM hotspots, Fmax |
| `neurosilicon/assistant.py` & `llm.py` | Neuro AI conversational engine & LLM connectors |
| `neurosilicon/priors.py` | Constraint masks, thermal penalties, routing channels |
| `neurosilicon/tech.py` | Sky130 / ASAP7 / GF180MCU technology presets |
| `neurosilicon/project.py` | Yosys RTL synthesis & project discovery pipeline |
| `neurosilicon/render.py` & `export_def.py` | High-res plotting & standard DEF/TCL exporters |
| `neurosilicon/tests/` | 19 comprehensive test suites (136 tests) |
| `neurosilicon/thirdparty/tilos/` | Vendored open-source placement cost engine (BSD-3) |
| `circuit_training/` | Upstream AlphaChip reference (provenance only) |
| `docs/` | Technical specs, blockages, netlist formats, and feature guides |
| `tools/` | Yosys, OpenROAD, and container tool shims |
| `NEUROSILICON_HOW_IT_WORKS.md` | Mathematical & algorithmic deep dive |
| `CHANGELOG.md` | Detailed release notes |

---

## ⚙️ CLI Power Tools

For researchers, batch experiments, and automated pipelines:

```bash
# 1. Run baseline placement evaluations (Random / First-Legal / Heuristics)
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.evaluate \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --episodes 10 --outdir logs/cli_baseline

# 2. Train the Edge-GNN Policy via Masked PPO
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m neurosilicon.train \
  --netlist circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt \
  --init_placement circuit_training/environment/test_data/toy_macro_stdcell/initial.plc \
  --outdir logs/cli_train --iterations 5 --episodes_per_iter 4
```

---

## 📜 History & Provenance

NeuroSilicon stands on the shoulders of Google Research's landmark **AlphaChip / Circuit Training** project (*Mirhoseini et al., Nature 2021*), whose 291 foundational commits remain preserved in `circuit_training/`.

NeuroSilicon modernizes this legacy into a lightweight, Python-native engine:
1. **Commit `54dcd45`:** Replaced the TensorFlow, TF-Agents, and proprietary binary stack with an open Python-native runtime over the vendored TILOS PLC core, introducing the PyTorch Edge-GNN, Gymnasium MDP, Yosys import pipeline, and initial GUI.
2. **Commit `91a7772`:** Engineered the full EDA workstation upgrade: real-time QPainter canvas, SAM hotspot segmenter, ⚡ Auto-Optimize loop, Technology Node presets, orientation flipping (`R0..R270`), and PPA metric suite.
3. **Commit `17d365c`:** Introduced **Neuro AI**, suggestion dropdowns with type-ahead completion, direct chat import, and exhaustive feature documentation.

---

## 🤝 Contributing & License

We believe open hardware requires truly open EDA tools. Contributions of all kinds — GNN policy enhancements, new technology presets, macro placers, or UI upgrades — are warmly welcomed!

* **Code of Conduct:** See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
* **Contributing Guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md).
* **License:** Licensed under the [Apache License, Version 2.0](LICENSE). Third-party vendored components maintain their original permissive licenses.

<p align="center">
  <sub><b>NeuroSilicon</b> — Place smarter. Explore faster. Keep silicon design open.</sub>
</p>
