# kiki-syn-agent

A conversational AI agent that bridges natural language with **Synera CAD** software and the **KIKI ML model**. Users describe what they want in plain language; a multi-agent system routes the request to the right specialist, calls the appropriate tool, and reports back the result.

**Part of:** KIKI project — WP 300 AI Optimization / WP 320 Dissemination

---

## What it does

```
User: "Recommend a lattice for an elderly patient, 60 kg"

  → Supervisor routes to KIKI agent
  → KIKI agent collects inputs, asks confirmation, calls KIKI ML API
  → KIKI: "GYR lattice recommended — vol. fraction 0.25, rotation 17°/18°/10°"

  → Supervisor routes to Synera agent
  → Synera agent confirms parameters, calls reconstruct_lattice tool
  → syneraheadless.exe runs reconstruct_lattice.syn, exports lattice_image.png
  → Synera: "GYR lattice created. Ready in Synera."
```

---

## Architecture

Three specialised agents coordinated by a supervisor:

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Reads conversation, routes to correct agent | None (structured output routing) |
| **KIKI agent** | Biomechanical optimisation — collects patient data and calls KIKI ML API | `kiki_recommend` |
| **Synera agent** | CAD geometry creation — runs Synera workflows | `reconstruct_lattice`, `create_sphere`, `create_pattern` |
| **Responder** | General questions and clarifications | None |

LLM: **OpenAI** (primary) with **Google Gemini** automatic fallback on rate limits or connection errors.
Observability: **LangSmith** tracing enabled via `.env`.
Persistence: **SQLite** checkpointer — conversations survive restarts.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.11 |
| Synera with headless license | `syneraheadless.exe` must be on PATH or set in `.env` |
| OpenAI API key | Primary LLM — get one at platform.openai.com |
| Google AI API key | Fallback LLM — free tier at aistudio.google.com |
| KIKI ML API running | Local FastAPI server at `http://127.0.0.1:8000` |

---

## Setup

**1. Create and activate a virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```powershell
pip install -r requirements.txt
```

**3. Configure `.env`**

```powershell
copy .env.example .env
```

Open `.env` and fill in your API keys. At minimum you need `OPENAI_API_KEY` and `GOOGLE_API_KEY`.

**4. Run the app**

```powershell
# Browser UI
streamlit run app.py

# Terminal chat
python agent.py
```

---

## Project structure

```
kiki-syn-agent/
│
├── agent.py                    # Active agent — multi-agent supervisor with LLM fallback
├── app.py                      # Streamlit browser UI
├── requirements.txt
├── .env                        # API keys and config (never commit — gitignored)
├── .env.example                # Template for new contributors
│
├── tools/                      # One subfolder per Synera tool
│   ├── kiki_recommend/         # Calls the KIKI ML API
│   ├── reconstruct_lattice/    # Runs reconstruct_lattice.syn via syneraheadless
│   ├── create_sphere/          # Runs create_sphere.syn
│   └── create_pattern/         # Runs create_pattern.syn
│
├── config/                     # Shared configuration
│   └── presets.py              # Bone condition presets for KIKI
│
├── tests/                      # Test suite
│   ├── test_langsmith.py       # Verifies LangSmith connection
│   └── test_fallback.py        # Verifies OpenAI → Gemini fallback
│
├── scripts/                    # Utility scripts (not part of the running app)
│   ├── visualize_graph.py      # Saves LangGraph diagrams to diagrams/
│   └── db_setup.py             # One-time PostgreSQL setup (optional)
│
├── synera_run/                 # Developer utilities for working with Synera directly
│   ├── run_workflow.py         # Manually run any .syn with a JSON input
│   ├── get_template.py         # Inspect inputs/outputs of any .syn file
│   └── synera_headless_reference.txt  # Full syneraheadless CLI reference
│
└── archive/                    # Previous agent versions kept for reference
    ├── agent_v1_single.py      # Original single ReAct agent
    └── agent_v2_multi.py       # Multi-agent without LLM fallback
```

---

## Available tools

### `kiki_recommend`
Calls the KIKI ML API with patient data and returns the optimal lattice parameters.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `bone_condition` | str | Yes | `osteoporotic`, `elderly`, `normal`, `athletic` |
| `body_weight` | float | Yes | Patient body weight in kg |

### `reconstruct_lattice`
Reconstructs a 3D lattice unit cell in Synera and exports `lattice_image.png`.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `cell_type` | str | Yes | — | `GYR`, `BCC`, `FCC`, `SCH`, `DIA`, `FLU`, `OCT`, `KEV`, `DTPMS`, `SPP`, `SC`, `DDK2`, `HCG`, `PSM2`, `RDO`, `TEG3` |
| `vol_frac` | float | No | 0.25 | Volume fraction (0.0–1.0) |
| `x_rotation` | float | No | 0.0 | Rotation around X axis in degrees |
| `y_rotation` | float | No | 0.0 | Rotation around Y axis in degrees |
| `z_rotation` | float | No | 0.0 | Rotation around Z axis in degrees |

### `create_sphere`
Creates a parametric 3D sphere in Synera.

| Parameter | Type | Required | Default |
|---|---|---|---|
| `radius` | float | Yes | — |
| `center_x/y/z` | float | No | 0.0 |

### `create_pattern`
Creates a 2D bio-inspired surface pattern in Synera.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern_type` | int | 1 | 0 = Curved Beam, 1 = Glass Sponge 1, 2 = Glass Sponge 2 |
| `u_spacing` | float | 0.5 | Spacing in U direction |
| `v_spacing` | float | 0.5 | Spacing in V direction |

---

## Running tests

```powershell
# Check LangSmith connection
python tests/test_langsmith.py

# Verify OpenAI → Gemini fallback
python tests/test_fallback.py

# Run all tests with pytest
pytest tests/
```

---

## Developer utilities

```powershell
# Inspect inputs/outputs of a Synera workflow
python synera_run/get_template.py tools/reconstruct_lattice/reconstruct_lattice.syn

# Run a workflow manually without the agent
python synera_run/run_workflow.py tools/reconstruct_lattice/reconstruct_lattice.syn tools/reconstruct_lattice/reconstruct_lattice_input.json

# Generate LangGraph architecture diagrams
python scripts/visualize_graph.py
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | Missing key in `.env` | Copy `.env.example` to `.env` and fill in keys |
| `syneraheadless.exe not found` | Not on PATH | Set `SYNERA_EXE` in `.env` to the full path |
| `Synera did not produce an output file` | Wrong `.syn` file path | Check `SYN_FILE` in the tool file |
| `KIKI API connection refused` | KIKI FastAPI server not running | Start the KIKI server before running the agent |
| Rate limit error | OpenAI quota hit | Gemini fallback fires automatically — no action needed |
