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

User: "What hip contact forces occur during walking?"

  → Supervisor routes to Knowledge agent
  → Knowledge agent calls rag_search tool
  → rag_search retrieves relevant chunks from indexed research papers
  → Agent: "According to Bergmann et al. (2001), peak hip contact force during
             normal walking is 238% of body weight..."
```

---

## Architecture

Four specialised agents coordinated by a supervisor:

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Reads conversation, routes to correct agent | None (structured output routing) |
| **KIKI agent** | Biomechanical optimisation — collects patient data and calls KIKI ML API | `kiki_recommend` |
| **Synera agent** | CAD geometry creation — runs Synera workflows | `reconstruct_lattice`, `create_sphere`, `create_pattern` |
| **Knowledge agent** | Answers questions from indexed research papers | `rag_search` |
| **Responder** | General questions and clarifications | None |

**LLM:** OpenAI (primary) with Google Gemini automatic fallback on rate limits or connection errors.  
**Observability:** LangSmith tracing enabled via `.env`.  
**Conversation persistence:** PostgreSQL checkpointer — conversations survive restarts, stored alongside the vector store in Docker.  
**Knowledge base:** PostgreSQL + pgvector in Docker — semantic search over indexed research papers.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.11 |
| Docker Desktop | Runs PostgreSQL + pgvector for the knowledge base and conversation history |
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

Open `.env` and fill in your keys. At minimum you need:

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
POSTGRES_URI=postgresql://postgres:yourpassword@localhost:5432/kiki_agent
POSTGRES_PASSWORD=yourpassword
```

**4. Start the database container**

```powershell
docker compose up -d
```

**5. Set up the database (first time only)**

```powershell
python scripts/db_setup.py
```

**6. Index your research documents (first time only)**

Add PDFs to `tools/rag_search/documents/` then run:

```powershell
python tools/rag_search/ingest.py
```

Only new or changed files are re-embedded on subsequent runs.

**7. Run the app**

```powershell
# Start everything with one command (Docker + Streamlit)
.\start.ps1

# Or run Streamlit directly (container must already be running)
streamlit run app.py
```

---

## Daily usage

```powershell
.\start.ps1
```

Starts the `rag_postgres` Docker container and launches the Streamlit UI. The container reconnects to the existing database volume — no re-ingestion needed.

---

## Project structure

```
kiki-syn-agent/
│
├── agent.py                    # Multi-agent supervisor with HITL and LLM fallback
├── app.py                      # Streamlit browser UI
├── docker-compose.yml          # PostgreSQL 16 + pgvector container definition
├── start.ps1                   # One-command startup (Docker + Streamlit)
├── requirements.txt
├── .env                        # API keys and config (never commit — gitignored)
├── .env.example                # Template for new contributors
│
├── tools/                      # One subfolder per tool
│   ├── kiki_recommend/         # Calls the KIKI ML API
│   ├── reconstruct_lattice/    # Runs reconstruct_lattice.syn via syneraheadless
│   ├── create_sphere/          # Runs create_sphere.syn
│   ├── create_pattern/         # Runs create_pattern.syn
│   └── rag_search/             # Semantic search over research papers
│       ├── rag_search_tool.py  # LangChain tool — called by knowledge_agent
│       ├── ingest.py           # Incremental PDF indexing into pgvector
│       ├── evaluate.py         # RAG quality evaluation (21 test cases)
│       └── documents/          # Place research PDFs here (gitignored)
│
├── config/                     # Shared configuration
│   └── presets.py              # Bone condition presets for KIKI
│
├── scripts/                    # Utility scripts
│   ├── db_setup.py             # One-time setup: pgvector, LangGraph tables, conversations tables
│   └── visualize_graph.py      # Saves LangGraph diagrams to diagrams/
│
├── tests/                      # Test suite
│   ├── test_langsmith.py       # Verifies LangSmith connection
│   └── test_fallback.py        # Verifies OpenAI → Gemini fallback
│
└── synera_run/                 # Developer utilities for Synera
    ├── run_workflow.py         # Manually run any .syn with a JSON input
    ├── get_template.py         # Inspect inputs/outputs of any .syn file
    └── synera_headless_reference.txt
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

### `rag_search`
Semantic search over indexed research papers using pgvector cosine similarity.

| Parameter | Type | Description |
|---|---|---|
| `query` | str | Plain-English question or search phrase |

Returns relevant excerpts with source filename. Filters low-relevance results via cosine distance threshold to prevent hallucination.

---

## RAG knowledge base

### Adding documents

1. Drop PDFs into `tools/rag_search/documents/`
2. Run `python tools/rag_search/ingest.py`

Only new or modified files are re-embedded — unchanged files are skipped (MD5 hash tracking).

### Testing retrieval quality

```powershell
python tools/rag_search/evaluate.py
```

Runs 21 test cases derived from real facts in the indexed papers and reports an overall quality score.

### Docker commands

```powershell
docker compose up -d        # start the database
docker compose down         # stop (data preserved — conversations and vectors kept)
docker compose down -v      # stop and delete ALL data (conversations + vectors lost)
docker ps                   # check container status
docker logs rag_postgres    # view database logs
docker volume ls            # list named volumes (rag-postgres-data)
```

### Backing up the database

Backs up both the RAG vectors and all conversation history in one command:

```powershell
docker exec rag_postgres pg_dump -U postgres kiki_agent > backup.sql
```

---

## Running tests

```powershell
# Check LangSmith connection
python tests/test_langsmith.py

# Verify OpenAI → Gemini fallback
python tests/test_fallback.py

# Verify RAG retrieval quality
python tools/rag_search/evaluate.py
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | Missing key in `.env` | Copy `.env.example` to `.env` and fill in keys |
| `syneraheadless.exe not found` | Not on PATH | Set `SYNERA_EXE` in `.env` to the full path |
| `docker: command not found` | Docker not in PATH | Open a new terminal after Docker Desktop starts |
| `POSTGRES_URI is not set` | Missing in `.env` | Add `POSTGRES_URI=postgresql://postgres:pw@localhost:5432/kiki_agent` |
| `Knowledge base unavailable` | Container not running | Run `docker compose up -d` |
| `KIKI API connection refused` | KIKI FastAPI server not running | Start the KIKI server before running the agent |
| Rate limit error | OpenAI quota hit | Gemini fallback fires automatically — no action needed |
