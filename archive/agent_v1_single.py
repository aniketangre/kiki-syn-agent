"""
agent.py
--------
A conversational AI agent built with LangGraph StateGraph and the Groq API.

The agent can chat naturally with the user AND call Synera tools to create
3D geometry when asked. Conversation state is persisted to a local SQLite
database (conversations.db) via SqliteSaver, so conversations survive
process restarts and can be resumed using their thread ID.

Graph structure:
                     ┌─────────┐
    START ──────────►│ chatbot │◄──────────┐
                     └────┬────┘           │
                          │                │
               Does the LLM want          │
               to call a tool?            │
                          │                │
               Yes ───────▼────────  No   │
                     ┌─────────┐     │    │
                     │  tools  │     └───►END
                     └────┬────┘
                          │
               Tool result goes back
               to chatbot for a reply
                          │
                          └────────────────┘

Usage:
    python agent.py          ← terminal chat
    streamlit run app.py     ← browser UI
"""

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated

import groq as groq_module
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# Import our Synera tools
from tools.create_sphere.create_sphere_tool import create_sphere
from tools.create_pattern.create_pattern_tool import create_pattern
from tools.reconstruct_lattice.reconstruct_lattice_tool import reconstruct_lattice
from tools.kiki_recommend.kiki_recommend_tool import kiki_recommend

# Prompt-time data — injected into SYSTEM_PROMPT so the agent knows
# valid ranges and preset details without needing to call a tool.
from tools.kiki_recommend.bounds import BOUNDS, get_bounds_summary
from config.presets import get_preset_summary


# ---------------------------------------------------------------------------
# Load configuration from .env
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Path to the SQLite database file. Configurable via .env; defaults to the
# project root. The file is created automatically on first run.
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "conversations.db")


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class State(TypedDict):
    """
    The state that flows through the graph at every step.

    `messages` holds the full conversation history.
    `add_messages` is a LangGraph reducer that appends new messages to the
    existing list rather than overwriting it on each graph step.
    """
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

# All available Synera tools — register new tools here as you build them
tools = [create_sphere, create_pattern, reconstruct_lattice, kiki_recommend]

# Bind tools to the LLM so it knows their names, parameters, and descriptions.
# The LLM uses this to decide when and how to call each tool.

# ── Option A: Groq (cloud) ─────────────────────────────────────────────────
# llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# ── Option B: Ollama (local) ───────────────────────────────────────────────
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="qwen3.5", base_url="http://localhost:11434")

# ── Option C: Google Gemini (cloud, free tier) ─────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# Context window limit
#
# Only the last CONTEXT_WINDOW_MESSAGES messages are sent to the LLM on each
# turn. The full conversation history is always stored in SQLite and shown in
# the UI — this limit only affects what the model sees.
#
# Note: a single tool-calling turn produces 3 messages
# (HumanMessage + AIMessage(tool_call) + ToolMessage + AIMessage reply = 4).
# Keep this ≥ 4 to ensure the model always sees at least one full tool turn.
# ---------------------------------------------------------------------------
CONTEXT_WINDOW_MESSAGES = 10  # change this value to adjust context size

# System prompt — sent at the top of every LLM call so the model always has
# its instructions regardless of how long the conversation has grown.
SYSTEM_PROMPT = SystemMessage(content=f"""
You are a helpful assistant that creates 3D geometry using Synera CAD tools.

Rules for calling tools:
- ALWAYS include ALL required parameters when calling a tool.
- For create_sphere, you MUST always provide the 'radius' parameter. It is required and has no default.
  Optional: center_x, center_y, center_z (default 0.0 each).
- For create_pattern, all parameters are optional and have sensible defaults.
  Key parameters: u_spacing, v_spacing (spacing between elements),
  u_size, v_size (pattern size as a single float each, default 1.0).
  pattern_type selects the pattern: 0 = Curved Beam, 1 = Glass Sponge 1, 2 = Glass Sponge 2.
  Do NOT pass plane parameters — the plane is set internally by the Synera workflow.
  It exports geometry.step and pattern_image.png.
- For reconstruct_lattice, you MUST always provide the 'cell_type' parameter. It is required.
  Accepted cell_type string values (always pass the tag, never a number):
    GYR, SCH, DTPMS, SPP, SC, BCC, FCC, DIA, FLU, OCT, KEV, DDK2, HCG, PSM2, RDO, TEG3
  If the user names a lattice by full name (e.g. "Gyroid" → "GYR", "Diamond" → "DIA", "BCC" → "BCC"), map it to the correct tag string.
  Optional parameters:
    vol_frac   — volume fraction 0.0–1.0, default 0.25 (lower = more porous)
    x_rotation — rotation around X axis in degrees, default 0.0
    y_rotation — rotation around Y axis in degrees, default 0.0
    z_rotation — rotation around Z axis in degrees, default 0.0
  It exports pattern_image.png.
- For kiki_recommend, follow these steps strictly:
  STEP 1 — Collect inputs. If the user has not provided them, ask for:
      a) Bone condition — one of: osteoporotic, elderly, normal, athletic.
         Map free-text descriptions using these keywords:
           "osteoporotic" — osteoporosis, fragile, brittle, severely reduced density
           "elderly"      — aged, senior, 65+, reduced density, age-related
           "normal"       — healthy, standard, typical adult
           "athletic"     — young, athlete, dense, strong, active, high density
      b) Patient body weight in kg (allowed range {BOUNDS.body_wt.min}–{BOUNDS.body_wt.max} kg).
  STEP 2 — Confirm. Before calling the tool, summarise what you will do, e.g.:
      "I will optimise a lattice for an elderly patient weighing 68 kg using
       default stress (50 MPa) and displacement (5 mm) limits. Shall I proceed?"
      Wait for the user to confirm. Do NOT call the tool until they say yes.
  STEP 3 — Call kiki_recommend with the confirmed inputs. Do not pass values
      for max_stress, max_disp, vol_frac, or vox_to_surf unless the user
      explicitly provided them — let the tool use its defaults.
  STEP 4 — Immediately call reconstruct_lattice using the returned
      recommended_cell_type, x_rotation_deg, y_rotation_deg, z_rotation_deg,
      and vol_frac. No further confirmation needed for this second call.
  Allowed ranges for user-provided parameters:
{get_bounds_summary()}
  Available bone presets and their material constants:
{get_preset_summary()}
  If the user asks about preset values or material properties, answer directly
  from the information above — do not say you lack access to it.
- If a required value is missing, ask the user for it before calling the tool.
- After a tool succeeds, always confirm completion clearly. Tell the user what was created
  and the key parameters used (e.g. cell type, volume fraction, spacing, pattern type).
  NEVER mention file paths, export directories, image filenames, or storage locations —
  the user does not need this information.
- After a tool fails, explain what went wrong in plain language without mentioning file paths.
- For casual conversation, reply normally without calling any tools.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """
    Normalise LLM message content to a plain string.

    Groq / Ollama return a plain string.
    Gemini 2.5 returns a list of typed blocks, e.g.:
        [{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}]
    This helper extracts only the 'text' blocks and joins them.
    Applied consistently in chat() and load_messages() so the UI
    never receives a raw list.
    """
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content or ""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def chatbot_node(state: State) -> dict:
    """
    The main chatbot node.

    Prepends the system prompt to the most recent CONTEXT_WINDOW_MESSAGES
    messages and sends them to the LLM. Older messages are excluded from the
    LLM call but remain stored in SQLite and visible in the UI.
    """
    recent = state["messages"][-CONTEXT_WINDOW_MESSAGES:]
    # Gemini requires conversations to start with a user (Human) turn.
    # Trimming can leave an AI/Tool message at the front — drop leading
    # non-human messages until the slice starts with a HumanMessage.
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    if not recent:
        # Search backwards for the most recent HumanMessage as fallback.
        # Using state["messages"][-1] is unsafe — it could be a ToolMessage.
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                recent = [msg]
                break
    # Normalize AIMessage content to a plain string before re-sending.
    # Gemini stores thinking blocks as a list in SQLite; the Gemini client
    # cannot re-serialize that format and throws "contents are required".
    safe = []
    for msg in recent:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            safe.append(AIMessage(
                content=_extract_text(msg.content),
                tool_calls=msg.tool_calls,
            ))
        else:
            safe.append(msg)
    messages = [SYSTEM_PROMPT] + safe
    response = llm_with_tools.invoke(messages)
    # Return as a list — add_messages appends it to the existing history
    return {"messages": [response]}


# ToolNode automatically executes whichever tool the LLM requested and
# wraps the result in a ToolMessage that goes back into the conversation.
tool_node = ToolNode(tools=tools)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_use_tool(state: State) -> str:
    """
    Conditional edge: decides where to go after the chatbot node.

    Returns "tools" if the LLM wants to call a tool, or END if the LLM
    has produced a final text reply and the graph should stop.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools",   tool_node)

# Every invocation starts at the chatbot node
graph_builder.set_entry_point("chatbot")

# After the chatbot responds: either call a tool or finish
graph_builder.add_conditional_edges(
    "chatbot",
    should_use_tool,
    {"tools": "tools", END: END},
)

# After a tool runs: always return to the chatbot so it can read the
# tool result and produce a natural language reply
graph_builder.add_edge("tools", "chatbot")


# ---------------------------------------------------------------------------
# SQLite checkpointer + graph compilation
#
# SqliteSaver persists the full conversation state (all messages, tool calls,
# and results) to a local SQLite file, keyed by thread_id. This means:
#   - Conversations survive process restarts
#   - Multiple independent conversations are stored in the same file
#   - A conversation can be resumed at any time by reusing its thread_id
#
# check_same_thread=False is required because Streamlit may access the
# connection from different threads across reruns.
# ---------------------------------------------------------------------------

_sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_sqlite_conn)

# Compile the graph with the checkpointer baked in. Every graph.invoke()
# call will automatically load and save state for the given thread_id.
graph = graph_builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Conversations metadata table
# Stores a title and timestamp for each thread so the UI can list them.
# Lives in the same SQLite file alongside LangGraph's checkpoint tables.
# ---------------------------------------------------------------------------

_sqlite_conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        thread_id  TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
""")

# message_images stores the timestamped image path for each assistant reply
# that triggered a Synera tool run. Rows are ordered by `id` (insert order)
# so they can be matched positionally to tool-result messages in the checkpoint.
_sqlite_conn.execute("""
    CREATE TABLE IF NOT EXISTS message_images (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id   TEXT NOT NULL,
        image_path  TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
""")
_sqlite_conn.commit()


def save_conversation(thread_id: str, title: str) -> None:
    """Register a conversation title. INSERT OR IGNORE — only saves the first time."""
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
    """Persist the timestamped image path produced by a tool run."""
    _sqlite_conn.execute(
        "INSERT INTO message_images (thread_id, image_path) VALUES (?, ?)",
        (thread_id, image_path),
    )
    _sqlite_conn.commit()


def delete_conversation(thread_id: str) -> None:
    """
    Permanently delete a conversation and all its associated data:
      - PNG image files from disk  (from message_images)
      - message_images rows        (our metadata table)
      - conversations row          (our metadata table)
      - checkpoints rows           (LangGraph internal table)
      - writes rows                (LangGraph internal table)

    File-deletion errors are swallowed so the DB cleanup always completes
    even if a file was already removed manually.
    """
    # 1. Collect image paths before touching the DB
    rows = _sqlite_conn.execute(
        "SELECT image_path FROM message_images WHERE thread_id = ?",
        (thread_id,),
    ).fetchall()

    # 2. Delete image files from disk
    for (image_path,) in rows:
        try:
            p = Path(image_path)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # 3. Delete from our own metadata tables
    _sqlite_conn.execute(
        "DELETE FROM message_images WHERE thread_id = ?", (thread_id,)
    )
    _sqlite_conn.execute(
        "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
    )

    # 4. Delete from LangGraph's internal checkpoint tables.
    #    Wrapped in try/except because table names can vary across LangGraph versions.
    for table in ("checkpoints", "writes"):
        try:
            _sqlite_conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
            )
        except Exception:
            pass

    _sqlite_conn.commit()


def get_last_image_exports(thread_id: str) -> list[str]:
    """
    Return PNG export paths reported by the most recent tool run in this thread.

    Walks the message list backward from the end, collecting image paths from
    every ToolMessage in the latest turn, and stops at the preceding HumanMessage.
    Only returns paths from successful tool calls that include an 'exports' key.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state  = graph.get_state(config)
    if not state or not state.values:
        return []
    paths = []
    for msg in reversed(state.values.get("messages", [])):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            print(f"[image-debug] ToolMessage content type: {type(msg.content).__name__}")
            print(f"[image-debug] ToolMessage content[:300]: {str(msg.content)[:300]}")
            try:
                # content may be a JSON string or already a dict
                if isinstance(msg.content, dict):
                    payload = msg.content
                else:
                    payload = json.loads(msg.content)
                print(f"[image-debug] success={payload.get('success')}  exports={payload.get('exports')}")
                if payload.get("success") and payload.get("exports"):
                    for v in payload["exports"].values():
                        if str(v).endswith(".png"):
                            paths.append(str(v))
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f"[image-debug] parse error: {e}")
    print(f"[image-debug] paths found: {paths}")
    return paths


def load_messages(thread_id: str) -> list[dict]:
    """
    Reconstruct the display message list for a thread.

    Images are matched to assistant replies by detecting successful tool runs
    in the LangGraph checkpoint: when a ToolMessage contains a JSON result with
    "success": true and an "exports" key, the next final AIMessage gets the
    next stored image (by insertion order) from message_images.
    """
    from pathlib import Path as _Path

    config = {"configurable": {"thread_id": thread_id}}
    state  = graph.get_state(config)
    if not state or not state.values:
        return []

    # Load stored image paths for this thread, oldest first
    rows = _sqlite_conn.execute(
        "SELECT image_path FROM message_images WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    images = [_Path(r[0]) for r in rows]
    image_idx   = 0
    pending_img = False   # a tool ran successfully — next AIMessage gets an image

    result = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content, "image": None})

        elif isinstance(msg, ToolMessage):
            # Check if this tool result signals a successful image-generating run
            try:
                payload = json.loads(msg.content)
                if payload.get("success") and payload.get("exports"):
                    pending_img = True
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            img = None
            if pending_img and image_idx < len(images):
                img = images[image_idx]
                image_idx  += 1
                pending_img = False
            result.append({"role": "assistant", "content": _extract_text(msg.content), "image": img})

    return result


# ---------------------------------------------------------------------------
# Public interface: send one message and get a reply
# ---------------------------------------------------------------------------

def chat(thread_id: str, user_message: str) -> tuple[str, list[str]]:
    """
    Send a user message through the agent graph and return the reply plus
    any PNG export paths produced by tool calls in this turn.

    Returns:
        (reply_text, png_paths)  — png_paths is [] when no tool produced an image.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        reply = _extract_text(result["messages"][-1].content)

        # Scan in-memory messages from this turn for image exports.
        # We work on result["messages"] directly — these are live Python objects,
        # not SQLite-deserialized data, so isinstance checks are reliable.
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

        return reply, png_paths

    except groq_module.RateLimitError:
        return "I am being rate-limited by the API. Please wait a moment and try again.", []

    except groq_module.BadRequestError as e:
        return (
            "I made an error constructing the tool call. "
            "Could you please repeat your request, making sure to include all required values? "
            f"(Internal error: {e})"
        ), []


# ---------------------------------------------------------------------------
# Terminal interactive loop (python agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("  Synera AI Agent")
    print("  Powered by LangGraph + Groq + SQLite")
    print("  Type 'exit' to quit")
    print("=" * 55)

    # Allow the user to resume a previous conversation by entering its
    # thread ID, or press Enter to start a fresh one.
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
        print(f"Conversations are saved to: {SQLITE_DB_PATH}")
        print("(Save the thread ID above to resume this conversation later.)\n")

    print("I can chat with you and create 3D geometry using Synera.")
    print("Try: 'Reconstruct a BCC lattice'\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"\nGoodbye. Thread ID: {thread_id}")
            print(f"Conversations saved to: {SQLITE_DB_PATH}")
            break

        reply = chat(thread_id, user_input)
        print(f"\nAgent: {reply}\n")
