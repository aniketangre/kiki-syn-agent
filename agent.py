"""
agent.py  —  Multi-agent supervisor with HITL interrupts and LLM fallback
--------------------------------------------------------------------------
Extends agent3.py with Human-in-the-Loop (HITL) confirmation before every
tool call. The graph pauses automatically before executing any tool and
waits for the caller to call resume() or cancel().

Architecture change from agent3.py
------------------------------------
kiki_subgraph and synera_subgraph are now added as REAL sub-graph nodes
in the parent graph (instead of being called via .invoke() inside wrapper
functions). This is required for interrupt_before=["tools"] to propagate
to the parent graph's checkpointer and be resumable.

HITL public API
---------------
chat(thread_id, message)  → (reply, png_paths, pending_tool)
resume(thread_id)         → (reply, png_paths, pending_tool)
cancel(thread_id)         → (cancel_text, [], None)

pending_tool is None when the graph completed normally, or
{"name": str, "args": dict} when paused before a tool call.

LLM setup (unchanged)
----------------------
Primary  : OpenAI  (gpt-4o-mini)
Fallback : Google Gemini  (gemini-2.5-flash)
Trigger  : RateLimitError | APIConnectionError | APITimeoutError | InternalServerError
Excluded : AuthenticationError — bad key, needs a human fix
"""

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal

import openai
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
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
    remaining_steps is managed automatically by LangGraph.
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
# LLM — OpenAI primary, Gemini fallback
# ---------------------------------------------------------------------------

_FALLBACK_ON = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

_primary_llm  = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
_fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

_base_llm = _primary_llm.with_fallbacks(
    [_fallback_llm],
    exceptions_to_handle=_FALLBACK_ON,
)

class _RouteDecision(BaseModel):
    next: Literal["kiki", "synera", "responder"]

_supervisor_llm = _primary_llm.with_structured_output(_RouteDecision).with_fallbacks(
    [_fallback_llm.with_structured_output(_RouteDecision)],
    exceptions_to_handle=_FALLBACK_ON,
)

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
  1. User asks to design an implant, wants a lattice recommendation, or
     asks about optimising for a patient (hip implant, bone implant,
     "recommend", "optimise", "best lattice for", etc.)                    → kiki
  2. The last assistant message was kiki asking for missing patient data
     (bone condition or body weight) and the user is providing it          → kiki
  3. kiki already returned recommended_cell_type in this turn              → synera
  4. "reconstruct / create / make [lattice / sphere / pattern]"            → synera
  5. Everything else                                                       → responder

To apply rule 2: look at the last assistant message in the conversation.
If it asked for bone condition or body weight, the user's reply belongs to kiki.
""")

_KIKI_PROMPT = SystemMessage(content=f"""
You are a biomechanical optimisation specialist. Collect patient data and
call the KIKI ML model to recommend the optimal lattice structure.

Follow these steps in order:

  STEP 1 — Collect required inputs if not already given:
              a) Bone condition: osteoporotic | elderly | normal | athletic
              b) Body weight in kg
            If either is missing, ask for it and stop.

  STEP 2 — Once both inputs are known, call kiki_recommend immediately.
            Do NOT ask for confirmation — a confirmation dialog will appear
            automatically before the tool executes.

  STEP 3 — Present the result clearly:
              • Recommended cell type
              • Volume fraction
              • Rotation angles (X, Y, Z)
            Do NOT call reconstruct_lattice — that is the Synera agent's job.

Valid parameter ranges:
{get_bounds_summary()}

Bone presets and their material constants:
{get_preset_summary()}

Never reveal internal file paths or engineering constants to the user unless asked.
""")

_SYNERA_PROMPT = SystemMessage(content="""
You are a CAD geometry specialist. Create 3D geometry in Synera using the
available tools.

Call the appropriate tool with the provided parameters immediately.
Do NOT ask for confirmation — a confirmation dialog will appear automatically
before the tool executes.
After the tool call, briefly confirm what was created and that it is ready in Synera.

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


def _kiki_messages_modifier(state) -> list:
    """Trim history and prepend KIKI system prompt before each LLM call."""
    messages = state["messages"] if isinstance(state, dict) else list(state)
    return [_KIKI_PROMPT] + _trim(messages)


def _synera_messages_modifier(state) -> list:
    """Trim history and prepend Synera system prompt before each LLM call."""
    messages = state["messages"] if isinstance(state, dict) else list(state)
    return [_SYNERA_PROMPT] + _trim(messages)

# ---------------------------------------------------------------------------
# Sub-agent compiled graphs (real sub-graph nodes — required for HITL)
# ---------------------------------------------------------------------------

kiki_subgraph = create_react_agent(
    model=_base_llm,
    tools=_kiki_tools,
    prompt=_kiki_messages_modifier,   # trims history + prepends system prompt
    state_schema=KIKIAgentState,
    interrupt_before=["tools"],        # pause before every tool call for HITL
    name="kiki_agent",
)

synera_subgraph = create_react_agent(
    model=_base_llm,
    tools=_synera_tools,
    prompt=_synera_messages_modifier,  # trims history + prepends system prompt
    state_schema=SyneraAgentState,
    interrupt_before=["tools"],        # pause before every tool call for HITL
    name="synera_agent",
)

# ---------------------------------------------------------------------------
# Supervisor graph nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: SupervisorState) -> dict:
    """Read the conversation and set the routing label for this turn."""
    decision = _supervisor_llm.invoke([_SUPERVISOR_PROMPT] + _trim(state["messages"]))
    return {"next_agent": decision.next}


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
    After KIKI finishes: if it produced a successful recommendation this turn,
    return to supervisor to chain to Synera. Otherwise wait for the user.
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
_builder.add_node("kiki_agent",   kiki_subgraph)   # real sub-graph — supports HITL
_builder.add_node("synera_agent", synera_subgraph) # real sub-graph — supports HITL
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
# SQLite checkpointer
# ---------------------------------------------------------------------------

_sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_sqlite_conn)
graph        = _builder.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# Conversations metadata tables
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
# Public SQLite helpers
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
# HITL result processor — shared by chat(), resume(), cancel()
# ---------------------------------------------------------------------------

def _process_result(result: dict, config: dict) -> tuple[str, list[str], dict | None]:
    """
    Extract reply text, PNG paths, and pending tool info from a graph result.

    When the graph is paused at a HITL interrupt, the pending AIMessage with
    tool_calls lives inside the sub-graph's state. We access it via
    graph.get_state(subgraphs=True) and return it as pending_tool so the UI
    can show a confirmation dialog.
    """
    snapshot = graph.get_state(config)
    pending_tool: dict | None = None

    if snapshot.next:
        # Peek inside sub-graph state to find the pending tool call.
        try:
            sub_snap = graph.get_state(config, subgraphs=True)
            for task in sub_snap.tasks:
                if hasattr(task, "state") and task.state:
                    sub_msgs = task.state.values.get("messages", [])
                    for msg in reversed(sub_msgs):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            tc = msg.tool_calls[0]
                            pending_tool = {"name": tc["name"], "args": tc["args"]}
                            break
                if pending_tool:
                    break
        except Exception:
            pass

        # Fallback: some LangGraph versions flush sub-graph messages to parent state.
        if pending_tool is None:
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tc = msg.tool_calls[0]
                    pending_tool = {"name": tc["name"], "args": tc["args"]}
                    break

    conversational: list[str] = []
    png_paths: list[str] = []

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            try:
                payload = (
                    msg.content if isinstance(msg.content, dict)
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
    return reply, png_paths, pending_tool

# ---------------------------------------------------------------------------
# Public chat interface
# ---------------------------------------------------------------------------

def stream_chat(thread_id: str, user_message: str):
    """Yield graph update chunks for streaming UI rendering."""
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
    """Yield (token_or_message, node_name) tuples for token-by-token streaming."""
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


def chat(thread_id: str, user_message: str) -> tuple[str, list[str], dict | None]:
    """
    Send a user message through the multi-agent graph.

    Returns:
        (reply_text, png_paths, pending_tool)
        pending_tool is None when the graph completed normally.
        pending_tool is {"name": str, "args": dict} when paused at a HITL interrupt.
        Call resume() to allow the tool to run, or cancel() to abort it.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        return _process_result(result, config)
    except Exception as e:
        return f"An error occurred: {e}", [], None


def resume(thread_id: str) -> tuple[str, list[str], dict | None]:
    """
    Resume a graph paused at a HITL interrupt, allowing the pending tool to execute.

    Returns the same 3-tuple as chat(). pending_tool will be non-None again if
    the resumed run hits a second interrupt (e.g., synera tool call after kiki).
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(Command(resume=True), config=config)
        return _process_result(result, config)
    except Exception as e:
        return f"An error occurred: {e}", [], None


def cancel(thread_id: str) -> tuple[str, list[str], dict | None]:
    """
    Cancel a pending HITL interrupt without executing the tool.

    Injects a cancellation AIMessage as the output of the interrupted sub-agent
    node, then drives the graph to END. The routing edge (_route_after_kiki) sees
    no ToolMessage with recommended_cell_type and routes to END.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if not snapshot.next:
        return "No pending tool call to cancel.", [], None

    interrupted_node = snapshot.next[0]
    cancel_reply = AIMessage(
        content="Tool call cancelled. Let me know how you'd like to proceed."
    )

    # Mark the interrupted node as completed with the cancellation message.
    graph.update_state(config, {"messages": [cancel_reply]}, as_node=interrupted_node)

    # Drive the graph forward from the now-completed node to reach END cleanly.
    try:
        graph.invoke(None, config=config)
    except Exception:
        pass

    return _extract_text(cancel_reply.content), [], None

# ---------------------------------------------------------------------------
# Terminal interactive loop (python agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  Synera AI Agent — Multi-Agent + HITL + LLM Fallback")
    print("  Primary: OpenAI  |  Fallback: Google Gemini")
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

        reply, _, pending_tool = chat(thread_id, user_input)
        if reply:
            print(f"\nAgent: {reply}\n")

        # HITL loop — confirm each tool call before execution
        while pending_tool is not None:
            print(f"\n[CONFIRM] About to call: {pending_tool['name']}")
            print(f"          Parameters: {json.dumps(pending_tool['args'], indent=2)}")
            choice = input("Proceed? [y/n]: ").strip().lower()
            if choice in ("y", "yes"):
                reply, _, pending_tool = resume(thread_id)
                if reply:
                    print(f"\nAgent: {reply}\n")
            else:
                cancel_text, _, _ = cancel(thread_id)
                print(f"\nAgent: {cancel_text}\n")
                pending_tool = None
