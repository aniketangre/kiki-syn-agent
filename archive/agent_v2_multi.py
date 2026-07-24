"""
agent2.py
---------
True Supervisor multi-agent system built with LangGraph.

Differences from agent.py
--------------------------
agent.py is a single ReAct agent with all tools in one loop.
agent2.py replaces it with three specialised agents coordinated by a supervisor:

  Supervisor   — reads the conversation and routes to the right agent each turn.
                 Uses with_structured_output() so routing is typed, not free text.

  KIKI agent   — compiled create_react_agent subgraph with its own ReAct loop.
                 Owns kiki_recommend tool only. Has a separate KIKIAgentState
                 so its internal steps are isolated from the supervisor state.

  Synera agent — compiled create_react_agent subgraph with its own ReAct loop.
                 Owns create_sphere, create_pattern, reconstruct_lattice tools.
                 Separate SyneraAgentState for the same reason.

  Responder    — plain LLM call for greetings and general questions (no tools).

Additional changes vs agent.py:
  - Three TypedDicts (SupervisorState, KIKIAgentState, SyneraAgentState) instead
    of one, each with RemainingSteps to prevent infinite loops in sub-agents.
  - _trim() context window helper keeps the last 10 messages and strips Gemini
    thinking blocks before re-sending to the API.
  - stream_chat() and stream_tokens() added for optional streaming UIs.
  - LangSmith tracing enabled automatically via .env (LANGSMITH_TRACING=true).
  - SQLite checkpointer lives on the supervisor graph only; sub-agents are
    stateless computation units that do not persist state between turns.

Public interface (identical to agent.py — swap the import in app.py):
    chat(), load_messages(), list_conversations(), save_conversation(),
    save_message_image(), delete_conversation()

Usage:
    python agent2.py          ← terminal chat
    streamlit run app.py      ← browser UI (change import to agent2)
"""

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

from tools.create_sphere.create_sphere_tool import create_sphere
from tools.create_pattern.create_pattern_tool import create_pattern
from tools.reconstruct_lattice.reconstruct_lattice_tool import reconstruct_lattice
from tools.kiki_recommend.kiki_recommend_tool import kiki_recommend
from tools.kiki_recommend.bounds import get_bounds_summary
from config.presets import get_preset_summary

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "conversations.db")

# ---------------------------------------------------------------------------
# States — each agent owns its own TypedDict
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    """Shared conversation state owned by the supervisor graph."""
    messages:   Annotated[list, add_messages]
    next_agent: str  # routing label set each turn by supervisor_node

class KIKIAgentState(TypedDict):
    """
    Private state for the KIKI sub-agent's internal ReAct loop.
    remaining_steps is managed automatically by LangGraph — it is injected at
    the start of each invocation and decremented after every node execution.
    When it reaches 0 the agent is forced to stop, preventing infinite loops.
    """
    messages:        Annotated[list, add_messages]
    remaining_steps: RemainingSteps

class SyneraAgentState(TypedDict):
    """
    Private state for the Synera sub-agent's internal ReAct loop.
    Same remaining_steps contract as KIKIAgentState.
    """
    messages:        Annotated[list, add_messages]
    remaining_steps: RemainingSteps

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_kiki_tools   = [kiki_recommend]
_synera_tools = [create_sphere, create_pattern, reconstruct_lattice]

# ---------------------------------------------------------------------------
# LLM — uncomment one option, comment the rest
# ---------------------------------------------------------------------------

# ── Option A: Groq (cloud, fast) ───────────────────────────────────────────
# from langchain_groq import ChatGroq
# _base_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# ── Option B: Ollama (local, private) ──────────────────────────────────────
# from langchain_ollama import ChatOllama
# _base_llm = ChatOllama(model="qwen3.5", base_url="http://localhost:11434")

# ── Option C: Google Gemini (cloud, large context) ─────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
_base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
# _base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)

# ── Option D: OpenAI (cloud, GPT models) ───────────────────────────────────
# from langchain_openai import ChatOpenAI
# _base_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

# Supervisor uses structured output to produce a routing label (no tools)
class _RouteDecision(BaseModel):
    next: Literal["kiki", "synera", "responder"]

_supervisor_llm = _base_llm.with_structured_output(_RouteDecision)

# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_MESSAGES = 10

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = SystemMessage(content="""
You are a routing supervisor. Read the conversation and decide which agent
should handle the user's current request. Reply with exactly one routing label.

Available agents:
  kiki      — patient-specific lattice optimisation using the KIKI ML model.
               Use when the user wants a recommendation based on bone condition
               or biomechanical properties.
  synera    — 3D geometry creation (lattice, sphere, pattern) in Synera.
               Use when the user gives explicit geometry parameters, OR when
               the kiki agent has just produced cell_type and rotation values
               that should now be passed to reconstruct_lattice.
  responder — everything else: greetings, clarifications, general questions.

Routing rules (apply the first that matches):
  1. "recommend / optimise / best lattice for [patient type]"              → kiki
  2. User replies yes/confirm/proceed/ok/sure when kiki last asked         → kiki
  3. kiki already returned recommended_cell_type in this turn              → synera
  4. User replies yes/confirm/proceed/ok/sure when synera last asked       → synera
  5. "reconstruct / create / make [lattice / sphere / pattern]"            → synera
  6. Everything else                                                       → responder

To apply rules 2 and 4: check the last assistant message in the conversation.
If it ends with a question ("Shall I proceed?") from kiki → rule 2.
If it ends with a question ("Shall I proceed?") from synera → rule 4.
""")

_KIKI_PROMPT = SystemMessage(content=f"""
You are a biomechanical optimisation specialist. Collect patient data and
call the KIKI ML model to recommend the optimal lattice structure.

Follow these steps in order:

  STEP 1 — Collect required inputs if not already given:
              a) Bone condition: osteoporotic | elderly | normal | athletic
              b) Body weight in kg
            If either is missing, ask for it and stop.

  STEP 2 — Once both inputs are known, summarise them and ask:
              "Bone condition: <value>, body weight: <value> kg. Shall I proceed?"
            Stop here. Do NOT call kiki_recommend yet.

  STEP 3 — Only after the user confirms (yes / ok / proceed / sure / go ahead):
            call kiki_recommend with the collected inputs.

  STEP 4 — Present the result clearly:
              • Recommended cell type
              • Volume fraction
              • Rotation angles (X, Y, Z)
            Do NOT call reconstruct_lattice — that is the Synera agent's job.

IMPORTANT: Never call kiki_recommend in the same turn you ask for confirmation.

Valid parameter ranges:
{get_bounds_summary()}

Bone presets and their material constants:
{get_preset_summary()}

Never reveal internal file paths or engineering constants to the user unless asked.
""")

_SYNERA_PROMPT = SystemMessage(content="""
You are a CAD geometry specialist. Create 3D geometry in Synera using the
available tools. Follow these two steps:

STEP 1 — Before calling any tool, list the parameters you will use and ask:
            "I'll use these parameters:
             - Cell type: <value>
             - Volume fraction: <value>
             - Rotation (X / Y / Z): <value>° / <value>° / <value>°  (omit if all zero)
             Shall I proceed?"
          Stop here. Do NOT call any tool yet.

STEP 2 — Only after the user confirms (yes / ok / proceed / sure / go ahead):
          call the appropriate tool with those parameters.
          After the tool call, briefly confirm what was created and that it is ready in Synera.

IMPORTANT: Never call a tool in the same turn you ask for confirmation.

Tools and their required / optional parameters:
  reconstruct_lattice
    REQUIRED  cell_type   — one of: GYR SCH DTPMS SPP SC BCC FCC DIA
                                    FLU OCT KEV DDK2 HCG PSM2 RDO TEG3
    OPTIONAL  vol_frac    — volume fraction, 0.0–1.0  (default 0.25)
              x_rotation, y_rotation, z_rotation — degrees (default 0)

  create_sphere
    REQUIRED  radius      — sphere radius
    OPTIONAL  center_x, center_y, center_z (default 0.0)

  create_pattern
    OPTIONAL  u_size, v_size, u_spacing, v_spacing
              pattern_type — 0 = Curved Beam, 1 = Glass Sponge 1, 2 = Glass Sponge 2

Never mention file paths, export directories, or internal storage locations.
""")

_GENERAL_PROMPT = SystemMessage(content="""
You are a helpful assistant for the Synera CAD and KIKI Lattice Optimiser system.
Answer general questions about the system, lattice structures, or biomechanics.
For geometry creation or patient optimisation, let the user know they can ask you.
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Normalise LLM content to plain string (handles Gemini thinking blocks)."""
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content or ""


def _trim(messages: list) -> list:
    """
    Return the last CONTEXT_WINDOW_MESSAGES messages, guaranteeing the slice
    starts with a HumanMessage (required by Gemini's turn-ordering rules).
    Also strips Gemini thinking blocks so re-serialisation to the API never fails.
    """
    recent = messages[-CONTEXT_WINDOW_MESSAGES:]
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    if not recent:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                recent = [msg]
                break

    safe = []
    for msg in recent:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            safe.append(AIMessage(
                content=_extract_text(msg.content),
                tool_calls=msg.tool_calls,
            ))
        else:
            safe.append(msg)
    return safe

# ---------------------------------------------------------------------------
# Sub-agent compiled graphs (true ReAct agents with their own state + loop)
#
# create_react_agent compiles a full StateGraph internally:
#     input → llm_node → (tool call?) → tool_node → llm_node → ... → END
#
# Each invocation runs the complete loop until the LLM stops calling tools.
# The sub-agents have no checkpointer — they are stateless computation units.
# Only the supervisor's checkpointer persists the conversation to SQLite.
# ---------------------------------------------------------------------------

kiki_subgraph = create_react_agent(
    model=_base_llm,
    tools=_kiki_tools,
    prompt=_KIKI_PROMPT,
    state_schema=KIKIAgentState,
    name="kiki_agent",
)

synera_subgraph = create_react_agent(
    model=_base_llm,
    tools=_synera_tools,
    prompt=_SYNERA_PROMPT,
    state_schema=SyneraAgentState,
    name="synera_agent",
)

# ---------------------------------------------------------------------------
# Supervisor graph nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: SupervisorState) -> dict:
    """Read the conversation and set the routing label for this turn."""
    decision = _supervisor_llm.invoke([_SUPERVISOR_PROMPT] + _trim(state["messages"]))
    return {"next_agent": decision.next}


def kiki_node(state: SupervisorState) -> dict:
    """
    Invoke the KIKI sub-agent with the current conversation context.
    The sub-agent runs its full internal ReAct loop (collect → confirm → call API)
    and returns when it has no more tool calls to make.
    Only the messages the sub-agent *added* are merged back into the supervisor state.
    """
    trimmed  = _trim(state["messages"])
    result   = kiki_subgraph.invoke({"messages": trimmed})
    new_msgs = result["messages"][len(trimmed):]
    return {"messages": new_msgs}


def synera_node(state: SupervisorState) -> dict:
    """
    Invoke the Synera sub-agent with the current conversation context.
    The sub-agent runs its full internal ReAct loop and returns once all
    geometry tool calls are complete.
    """
    trimmed  = _trim(state["messages"])
    result   = synera_subgraph.invoke({"messages": trimmed})
    new_msgs = result["messages"][len(trimmed):]
    return {"messages": new_msgs}


def responder_node(state: SupervisorState) -> dict:
    """Handle general conversation — a plain LLM call, no sub-agent needed."""
    response = _base_llm.invoke([_GENERAL_PROMPT] + _trim(state["messages"]))
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Routing functions (conditional edges on the supervisor graph)
# ---------------------------------------------------------------------------

def _route_from_supervisor(state: SupervisorState) -> str:
    return state.get("next_agent", "responder")


def _route_after_kiki(state: SupervisorState) -> str:
    """
    After the KIKI sub-agent returns:
    - If it completed a successful API call (recommended_cell_type present in a
      ToolMessage from this turn), return to the supervisor so it can chain to
      the Synera agent.
    - Otherwise (KIKI was asking a question), go to END and wait for the user.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
                if payload.get("success") and payload.get("recommended_cell_type"):
                    return "supervisor"
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
    return END

# ---------------------------------------------------------------------------
# Build the supervisor graph
# ---------------------------------------------------------------------------

_builder = StateGraph(SupervisorState)

_builder.add_node("supervisor",   supervisor_node)
_builder.add_node("kiki_agent",   kiki_node)
_builder.add_node("synera_agent", synera_node)
_builder.add_node("responder",    responder_node)

_builder.set_entry_point("supervisor")

_builder.add_conditional_edges(
    "supervisor",
    _route_from_supervisor,
    {"kiki": "kiki_agent", "synera": "synera_agent", "responder": "responder"},
)
_builder.add_conditional_edges(
    "kiki_agent",
    _route_after_kiki,
    {"supervisor": "supervisor", END: END},
)
_builder.add_edge("synera_agent", END)
_builder.add_edge("responder",    END)

# ---------------------------------------------------------------------------
# SQLite checkpointer — lives on the supervisor graph only.
# Sub-agents are stateless computation units; the supervisor state is what
# persists across user turns.
# ---------------------------------------------------------------------------

_sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_sqlite_conn)
graph        = _builder.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# Conversations metadata tables (shared schema with agent.py)
# ---------------------------------------------------------------------------

_sqlite_conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        thread_id  TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
""")
_sqlite_conn.execute("""
    CREATE TABLE IF NOT EXISTS message_images (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id   TEXT NOT NULL,
        image_path  TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
""")
_sqlite_conn.commit()

# ---------------------------------------------------------------------------
# Public SQLite helpers — identical signatures to agent.py
# ---------------------------------------------------------------------------

def save_conversation(thread_id: str, title: str) -> None:
    """Register a conversation title. INSERT OR IGNORE — saves only once."""
    _sqlite_conn.execute(
        "INSERT OR IGNORE INTO conversations (thread_id, title) VALUES (?, ?)",
        (thread_id, title[:60]),
    )
    _sqlite_conn.commit()


def list_conversations() -> list[dict]:
    """Return all conversations ordered newest first."""
    rows = _sqlite_conn.execute(
        "SELECT thread_id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def save_message_image(thread_id: str, image_path: str) -> None:
    """Persist the image path produced by a tool run."""
    _sqlite_conn.execute(
        "INSERT INTO message_images (thread_id, image_path) VALUES (?, ?)",
        (thread_id, image_path),
    )
    _sqlite_conn.commit()


def delete_conversation(thread_id: str) -> None:
    """Delete a conversation: image files, metadata rows, and LangGraph checkpoints."""
    rows = _sqlite_conn.execute(
        "SELECT image_path FROM message_images WHERE thread_id = ?",
        (thread_id,),
    ).fetchall()
    for (image_path,) in rows:
        try:
            p = Path(image_path)
            if p.exists():
                p.unlink()
        except Exception:
            pass
    _sqlite_conn.execute("DELETE FROM message_images WHERE thread_id = ?", (thread_id,))
    _sqlite_conn.execute("DELETE FROM conversations WHERE thread_id = ?",  (thread_id,))
    for table in ("checkpoints", "writes"):
        try:
            _sqlite_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except Exception:
            pass
    _sqlite_conn.commit()


def load_messages(thread_id: str) -> list[dict]:
    """
    Reconstruct the display message list for a thread from the supervisor state.
    Images are matched positionally to the assistant reply that followed the
    successful tool run that produced them.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state  = graph.get_state(config)
    if not state or not state.values:
        return []

    rows = _sqlite_conn.execute(
        "SELECT image_path FROM message_images WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    images      = [Path(r[0]) for r in rows]
    image_idx   = 0
    pending_img = False

    result = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content, "image": None})

        elif isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
                if payload.get("success") and payload.get("exports"):
                    pending_img = True
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            img = None
            if pending_img and image_idx < len(images):
                img        = images[image_idx]
                image_idx  += 1
                pending_img = False
            result.append({
                "role":    "assistant",
                "content": _extract_text(msg.content),
                "image":   img,
            })

    return result

# ---------------------------------------------------------------------------
# Public chat interface — identical signature to agent.py
# ---------------------------------------------------------------------------

def stream_chat(thread_id: str, user_message: str):
    """
    Yield graph update chunks for streaming UI rendering.
    Each chunk is {node_name: state_update} so the UI can show which agent
    is active and extract messages as they arrive.
    Yields {"error": str} on failure so the caller never raises.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        yield from graph.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="updates",
        )
    except Exception as e:
        yield {"error": str(e)}


def stream_tokens(thread_id: str, user_message: str):
    """
    Yield (token_or_message, node_name) tuples for token-by-token streaming.

    - token_or_message is a str token when the LLM is generating text
    - token_or_message is a ToolMessage when a tool finishes (for PNG extraction)
    - Yields ("__error__", error_str) on failure so the caller never raises.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "")
            if isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    yield text, node
            elif isinstance(chunk, ToolMessage):
                yield chunk, node
    except Exception as e:
        yield "__error__", str(e)


def chat(thread_id: str, user_message: str) -> tuple[str, list[str]]:
    """
    Send a user message through the multi-agent graph and return the reply
    plus any PNG export paths produced by tool calls in this turn.

    Returns:
        (reply_text, png_paths)  — png_paths is [] when no tool produced an image.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        # Collect all conversational AI messages from this turn (after last HumanMessage).
        # A multi-agent turn can have several: KIKI summary + Synera confirmation.
        # We join them so the user sees the full narrative, not just the final line.
        conversational: list[str] = []
        png_paths: list[str] = []

        for msg in reversed(result["messages"]):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                try:
                    payload = (
                        msg.content
                        if isinstance(msg.content, dict)
                        else json.loads(msg.content)
                    )
                    if payload.get("success") and payload.get("exports"):
                        for v in payload["exports"].values():
                            if str(v).endswith(".png"):
                                png_paths.append(str(v))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                text = _extract_text(msg.content)
                if text:
                    conversational.append(text)

        reply = "\n\n".join(reversed(conversational)) if conversational else ""
        return reply, png_paths

    except Exception as e:
        return f"An error occurred: {e}", []

# ---------------------------------------------------------------------------
# Terminal interactive loop (python agent2.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  Synera AI Agent — True Supervisor Multi-Agent Mode")
    print("  Specialists: KIKI Optimiser | Synera CAD | Responder")
    print("  Type 'exit' to quit")
    print("=" * 60)

    print("\nEnter a thread ID to resume a previous conversation,")
    print("or press Enter to start a new one:")
    thread_input = input("Thread ID: ").strip()

    if thread_input:
        thread_id = thread_input
        print(f"\nResuming conversation: {thread_id}\n")
    else:
        thread_id = str(uuid.uuid4())
        print(f"\nNew conversation started.")
        print(f"Thread ID: {thread_id}")
        print(f"Conversations saved to: {SQLITE_DB_PATH}")
        print("(Save the thread ID above to resume this conversation later.)\n")

    print("Try: 'Recommend a lattice for an elderly patient'")
    print("  or: 'Reconstruct a BCC lattice with 30% volume fraction'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"\nGoodbye. Thread ID: {thread_id}")
            break

        reply, _ = chat(thread_id, user_input)
        print(f"\nAgent: {reply}\n")
