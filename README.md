```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗     ███████╗██╗██╗     ██╗ ██████╗ ██████╗ ███╗   ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗    ██╔════╝██║██║     ██║██╔════╝██╔═══██╗████╗  ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║    ███████╗██║██║     ██║██║     ██║   ██║██╔██╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║    ╚════██║██║██║     ██║██║     ██║   ██║██║╚██╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝    ███████║██║███████╗██║╚██████╗╚██████╔╝██║ ╚████║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ╚══════╝╚═╝╚══════╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝
```

# NeuroSilicon ⚡ AI Chip Floorplanner

<p align="center">
  <b>Autonomous, Interactive Macro Placement with Graph Neural Networks & Deep RL — Live on Your Desktop.</b>
</p>

<p align="center">
  <a href="https://github.com/IndhirajithHari/neuro-silicon/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%20|%203.14-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CPU%20Friendly-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://pytest.org/"><img src="https://img.shields.io/badge/Tests-136%20Passed-success?logo=pytest&logoColor=white" alt="Tests"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Architecture-Edge--GNN%20%2B%20Masked%20PPO-8A2BE2" alt="Architecture"></a>
  <a href="#desktop-gui-primary"><img src="https://img.shields.io/badge/GUI-PySide6%20Qt%20%7C%20Streamlit-00BC8C" alt="GUI"></a>
  <a href="https://github.com/IndhirajithHari/neuro-silicon"><img src="https://img.shields.io/badge/Open%20EDA-Yosys%20%7C%20OpenROAD%20%7C%20Sky130-107C41" alt="Open EDA"></a>
</p>

---

## 💡 Overview: The Game of Silicon

Floorplanning is the pivotal first hour in physical chip design. Before standard cells are routed, an engineer must position the massive building blocks — SRAMs, register files, and execution units — onto a bare silicon canvas. A mistake here locks in long wire delays, routing congestion bottlenecks, and thermal hot zones that no downstream EDA pass can ever recover from.

**NeuroSilicon re-imagines floorplanning as an interactive, observable learning game:**

* 🎮 **Placement as a Reinforcement Learning MDP:** Macros are placed sequentially onto a 128×128 masked grid by an Edge-GNN policy trained with PPO.
* ⚡ **Live Force-Directed Physics:** Standard cells continuously float and relax around macros in real time, calculating transparent proxy costs (wirelength, density, RUDY routing congestion, and thermal gradients).
* 🤖 **Neuro AI Co-Pilot:** An integrated EDA intelligence capable of explaining trade-offs, enforcing conversational natural-language keep-out rules, diagnosing bottlenecks, and executing a self-driving **Auto-Optimize Loop** (Baseline ➔ SAM Hotspot Diagnostics ➔ Constrained Re-Placement).
* 💻 **100% Open-Source & CPU-Native:** Free of commercial EDA license locks, closed binaries, or heavy cloud GPU dependencies. Runs from Verilog to placed DEF in minutes on standard Linux machines.

> [!NOTE]
> **Exploration, Not Signoff:** NeuroSilicon is built for the rapid exploratory phase *before* expensive signoff tooling is invoked. Every metric is an honest, mathematically rigorous proxy designed for interactive optimization.

---

## 🗺️ Architectural Workflow

```mermaid
flowchart LR
    classDef ingestion fill:#0d2137,stroke:#0ea5e9,stroke-width:2px,color:#f1f5f9;
    classDef engine fill:#241242,stroke:#a855f7,stroke-width:2px,color:#f1f5f9;
    classDef polish fill:#062e2e,stroke:#14b8a6,stroke-width:2px,color:#f1f5f9;
    classDef ailoop fill:#381e05,stroke:#f59e0b,stroke-width:2px,color:#f1f5f9;
    classDef outputs fill:#06331e,stroke:#10b981,stroke-width:2px,color:#f1f5f9;

    subgraph S1 ["1. Design Ingestion"]
        direction TB
        RTL["Verilog / SystemVerilog"]:::ingestion
        YOSYS["Yosys Synthesis"]:::ingestion
        READY["Pre-built Netlists"]:::ingestion
        PROTO["Proto Netlist (.pb.txt)"]:::ingestion
        RTL --> YOSYS --> PROTO
        READY --> PROTO
    end

    subgraph S2 ["2. AI Placement Engine"]
        direction TB
        MDP["Gymnasium 128x128 MDP"]:::engine
        GNN["Edge-GNN Actor-Critic"]:::engine
        MASK["Legal Cell Masking"]:::engine
        PLACE["Sequential Placement"]:::engine
        RELAX["Force-Directed Relaxation"]:::engine
        MDP --> GNN --> MASK --> PLACE --> RELAX
    end

    subgraph S3 ["3. Physical Polish"]
        direction TB
        SA["Simulated Annealing Swaps"]:::polish
        FLIP["Orientation Optimizer (R0..R270)"]:::polish
        PROXY["Proxy Scoring (RUDY / Fmax)"]:::polish
        SA --> FLIP --> PROXY
    end

    subgraph S4 ["4. Neuro AI Loop"]
        direction TB
        AUTOLOOP["Auto-Optimize Loop"]:::ailoop
        SAM["SAM Hotspot Segmenter"]:::ailoop
        ADVICE["Bottleneck Diagnosis"]:::ailoop
        REPUTE["Constrained Re-Placement"]:::ailoop
        AUTOLOOP --> SAM --> ADVICE --> REPUTE
    end

    subgraph S5 ["5. Export & Verification"]
        direction TB
        DEF["Standard DEF (Flat / Viewer)"]:::outputs
        TCL["OpenROAD Script (.tcl)"]:::outputs
        ROUTE["Global Route Validation"]:::outputs
        DEF --> TCL --> ROUTE
    end

    PROTO --> MDP
    RELAX --> SA
    PROXY --> AUTOLOOP
    REPUTE --> DEF
```

---

## 📊 How NeuroSilicon Compares

| Dimension | Commercial EDA | AlphaChip (2021 Snapshot) | NeuroSilicon (v1.0.0) |
| :--- | :--- | :--- | :--- |
| **Licensing** | Proprietary (\$100k+/seat) | Research snapshot (TF/TF-Agents) | **100% Free & Open-Source (Apache 2.0)** |
| **Hardware** | Large Compute Clusters | Multi-GPU + Reverb distributed servers | **Runs on any modern Linux Laptop / CPU** |
| **Interactivity** | Batch CLI / Heavy GUI | Batch script execution | **Native QPainter Desktop Workstation + Web UI** |
| **Design Ingestion** | Complex proprietary TCL scripts | Hardcoded Protobuf netlists | **Drag-and-Drop RTL, folders, or `.zip` files** |
| **AI Co-Pilot** | None / Generic cloud wrapper | None | **Neuro AI (Conversational, Tool-Grounded, Offline)** |
| **Self-Optimization** | Manual design passes | Offline RL training scripts | **⚡ One-Click Autonomous Auto-Optimize Loop** |
| **Technology Nodes** | Closed PDK libraries | Fixed technology models | **Sky130, ASAP7, GF180MCU dynamic presets** |

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
* Press **⚡ Auto-Optimize (AI Loop)** to launch an autonomous three-phase refinement:
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
*Expected:* **136 passed, 1 skipped** *(openc906 skips until imported)*.

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

You can bring your own chip from RTL to floorplan in four ways:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CHIP INGESTION MODES                            │
├───────────────────┬────────────────────────────────────────────────────┤
│ Drag-and-Drop     │ Drag a folder, file, or .zip directly onto the UI │
│ Neuro AI Button   │ Click "Import project file…" in the Neuro AI tab   │
│ Natural Chat      │ Type "import /path/to/my_chip_rtl" in Neuro AI     │
│ Import Tab        │ Use Center Panel ➔ Browse ➔ Auto-Detect ➔ Import  │
└───────────────────┴────────────────────────────────────────────────────┘
```

* **Ready Netlists (`.pb.txt` + `.plc`):** Registers in milliseconds with zero synthesis overhead.
* **Raw RTL (`.v` / `.sv`):** NeuroSilicon automatically discovers module hierarchies, detects top modules, filters testbenches, runs Yosys synthesis, and generates floorplanning netlists on the fly.
* **Pre-bundled Reference Chips:** Includes `toy` (2 macros), `ariane` (133 macros), `servant SoC` (Sky130), `picorv32`, `picosoc`, and `ultraembedded riscv_core`.

---

## 🔌 AI Connectors (Optional & Privacy-First)

NeuroSilicon works **100% offline out-of-the-box**. No API keys are required. If you want to connect a foundation model, open **Connectors** in the desktop right panel:

* **Local / Private:** Connect to **Ollama** (`http://localhost:11434`) or **LM Studio** (`http://localhost:1234/v1`). Zero tokens leave your workstation.
* **Cloud API:** Point to any OpenAI-compatible inference gateway (API Key + Base URL + Model Name).
* **Storage Safety:** Credentials are saved exclusively in your personal `~/.config/neurosilicon/settings.json` (strictly isolated from git tracking).

---

## 📁 Repository Layout

```
neuro-silicon/
├── neurosilicon/               # ⚡ Runnable Python-native package
│   ├── desktop.py              # Native PySide6 Qt Workstation (Canvas, Pins, Compare)
│   ├── app.py                  # Modern Streamlit Web Console
│   ├── env.py                  # Gymnasium 128x128 placement MDP
│   ├── model.py & train.py     # Edge-GNN PyTorch policy & masked PPO trainer
│   ├── liveplace.py            # Physics relaxation, RUDY congestion, SAM, Fmax
│   ├── assistant.py & llm.py   # Neuro AI conversational engine & LLM connectors
│   ├── priors.py               # Constraint masks, thermal penalties, channels
│   ├── tech.py                 # Sky130 / ASAP7 / GF180MCU technology presets
│   ├── project.py              # Yosys RTL synthesis & project discovery
│   ├── render.py & export_def.py # Plotting & standard DEF/TCL exporters
│   ├── tests/                  # 19 comprehensive test suites (136 tests)
│   └── thirdparty/tilos/       # Vendored open-source placement cost engine (BSD-3)
├── circuit_training/           # 🏛️ Restored AlphaChip reference (provenance only)
├── docs/                       # Technical specs, blockages, netlist formats
│   ├── FEATURES.md             # Complete feature specification & evidence
│   └── ARIANE.md               # Ariane 133-macro design reference
├── tools/                      # Yosys, OpenROAD, and container tool shims
├── NEUROSILICON_HOW_IT_WORKS.md# Mathematical & algorithmic deep dive
└── CHANGELOG.md                # Detailed release notes
```

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
