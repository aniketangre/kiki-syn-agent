"""
app.py  —  Streamlit UI for the Synera AI Agent
Run with:  streamlit run app.py
"""

import html
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from agent import (
    cancel,
    chat,
    delete_conversation,
    list_conversations,
    load_messages,
    resume,
    save_conversation,
    save_message_image,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent
_IMAGES_DIR = _ROOT / "images"
_IMAGES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Synera AI Agent",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS — only touch what Streamlit can't do natively
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Hide default chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Hide every variant of the sidebar collapse/expand button across Streamlit versions */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNavCollapseIcon"],
section[data-testid="stSidebar"] > div > button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    display: none !important;
}

/* User message bubble (right-aligned) */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}
.user-bubble {
    background: #E8E4DC;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 15px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.65;
    word-wrap: break-word;
    white-space: pre-wrap;
    color: #1C1B10;
}

/* Thinking animation */
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
    30%           { transform: translateY(-5px); opacity: 1; }
}
.thinking-dots { display: flex; gap: 5px; align-items: center; padding: 8px 0; }
.thinking-dots span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #C96433;
    animation: bounce 1.3s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.18s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.36s; }

/* Result images */
.result-image img {
    border-radius: 10px;
    border: 1px solid #E0DDD6;
    margin-top: 10px;
    max-width: 380px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id"         not in st.session_state:
    st.session_state.thread_id         = str(uuid.uuid4())
if "messages"          not in st.session_state:
    st.session_state.messages          = []
if "pending_prompt"    not in st.session_state:
    st.session_state.pending_prompt    = None
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None  # None or {"name": str, "args": dict}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    # ── Branding ─────────────────────────────────────────────────────────────
    st.markdown("## 🧊 Synera AI")
    st.caption("Conversational 3D geometry assistant powered by LangGraph")
    st.divider()

    # ── New conversation ──────────────────────────────────────────────────────
    if st.button("＋  New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id         = str(uuid.uuid4())
        st.session_state.messages          = []
        st.session_state.pending_prompt    = None
        st.session_state.pending_interrupt = None
        st.rerun()

    st.divider()

    # ── Conversation history ──────────────────────────────────────────────────
    st.markdown("**Conversations**")

    convs = list_conversations()
    if not convs:
        st.caption("No conversations yet — send your first message!")
    else:
        for conv in convs:
            is_active = conv["thread_id"] == st.session_state.thread_id

            # Format timestamp
            try:
                ts    = datetime.strptime(conv["created_at"], "%Y-%m-%d %H:%M:%S")
                label = ts.strftime("%H:%M") if ts.date() == datetime.now().date() \
                        else ts.strftime("%b %d")
            except Exception:
                label = ""

            title = conv["title"][:40] + ("…" if len(conv["title"]) > 40 else "")

            if is_active:
                st.markdown(
                    f"<div style='background:#F4EDE6;border-left:3px solid #C96433;"
                    f"border-radius:6px;padding:8px 10px 8px 9px;margin:3px 0;"
                    f"font-size:13px;color:#3C3828;font-weight:500;'>"
                    f"{html.escape(title)}"
                    f"<br><span style='font-size:10px;color:#A09D90;'>"
                    f"{label} · {conv['thread_id'][:8]}…</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                col_title, col_del = st.columns([5, 1])
                with col_title:
                    btn_label = f"{title}  —  {label}"
                    if st.button(btn_label, key=f"conv_{conv['thread_id']}",
                                 use_container_width=True):
                        st.session_state.thread_id         = conv["thread_id"]
                        st.session_state.messages          = load_messages(conv["thread_id"])
                        st.session_state.pending_prompt    = None
                        st.session_state.pending_interrupt = None
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{conv['thread_id']}",
                                 use_container_width=True):
                        delete_conversation(conv["thread_id"])
                        # If the deleted thread was active, start a fresh one
                        if st.session_state.thread_id == conv["thread_id"]:
                            st.session_state.thread_id      = str(uuid.uuid4())
                            st.session_state.messages       = []
                            st.session_state.pending_prompt = None
                        st.rerun()

    st.divider()

    # ── Session ID ────────────────────────────────────────────────────────────
    st.markdown("**Session ID**")
    st.code(st.session_state.thread_id, language=None)
    st.caption("Copy this ID to resume the session later.")

    st.divider()

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("**About**")
    st.markdown("""
Synera AI bridges natural language with **Synera CAD** workflows.
Describe what you want to design and the agent calls the right tool automatically.

**Available tools**
- 🔩 `reconstruct_lattice` — 16 cell types, volume fraction, rotation
- 🔵 `create_sphere` — parametric sphere geometry
- 🔷 `create_pattern` — Glass Sponge & Curved Beam surface patterns

**Stack:** LangGraph · SQLite · Synera Headless
    """)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THINKING_HTML = """
<div class="thinking-dots">
    <span></span><span></span><span></span>
</div>
"""


def _render_user(content: str):
    safe = html.escape(content)
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def _render_assistant(content: str, image: Path | None = None):
    with st.chat_message("assistant", avatar="🧊"):
        st.markdown(content)
        if image and image.exists():
            st.image(image.read_bytes(), width=380)


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Welcome screen when no messages yet
if not st.session_state.messages:
    st.markdown("### What can I design for you?")
    st.markdown(
        "Describe the 3D geometry you need — lattice structures, spheres, "
        "or surface patterns — and I'll run the Synera workflow automatically."
    )
    st.markdown("---")
    st.markdown("**Try one of these:**")

    col1, col2, col3 = st.columns(3)
    examples = [
        ("🔩", "BCC Lattice",      "Reconstruct a BCC lattice with 30% volume fraction"),
        ("🌀", "Gyroid",           "Reconstruct a Gyroid lattice rotated 45° around Z"),
        ("🔷", "Surface Pattern",  "Create a Glass Sponge surface pattern with 0.3 spacing"),
    ]
    for col, (icon, title, prompt) in zip([col1, col2, col3], examples):
        with col:
            if st.button(f"{icon} **{title}**\n\n{prompt}",
                         key=f"ex_{title}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()
    st.markdown("---")

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        _render_user(msg["content"])
    else:
        _render_assistant(msg["content"], msg.get("image"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_tool(name: str, args: dict) -> str:
    """Return a plain-English sentence describing what the agent is about to do."""
    if name == "kiki_recommend":
        condition = args.get("bone_condition", "unknown")
        weight    = args.get("body_weight", "?")
        return (f"I have everything I need. I'll run the **KIKI ML model** to recommend "
                f"the optimal lattice for a **{condition}** patient weighing **{weight} kg**.")
    if name == "reconstruct_lattice":
        cell = args.get("cell_type", "?")
        vf   = args.get("vol_frac", 0.25)
        rx   = args.get("x_rotation", 0)
        ry   = args.get("y_rotation", 0)
        rz   = args.get("z_rotation", 0)
        rot  = f"{rx}° / {ry}° / {rz}°" if any([rx, ry, rz]) else "no rotation"
        return (f"I'll reconstruct a **{cell}** lattice in Synera — "
                f"volume fraction **{vf}**, rotation {rot}.")
    if name == "create_sphere":
        return f"I'll create a sphere with radius **{args.get('radius', '?')}** in Synera."
    if name == "create_pattern":
        labels = {0: "Curved Beam", 1: "Glass Sponge 1", 2: "Glass Sponge 2"}
        pt = labels.get(args.get("pattern_type", 1), "surface pattern")
        return f"I'll create a **{pt}** surface pattern in Synera."
    return f"I'm ready to call `{name}`."


# ---------------------------------------------------------------------------
# HITL interrupt UI — shown whenever the graph is paused before a tool call
# ---------------------------------------------------------------------------

if st.session_state.pending_interrupt is not None:
    tool_info = st.session_state.pending_interrupt

    # Short plain-English description shown as an assistant bubble
    _render_assistant(_describe_tool(tool_info["name"], tool_info["args"]))

    with st.container(border=True):
        st.markdown("#### Confirm Tool Execution")
        st.markdown(f"**Tool:** `{tool_info['name']}`")
        import json as _json
        st.code(_json.dumps(tool_info["args"], indent=2), language="json")
        st.caption("Review the parameters above, then confirm or cancel.")

        col_confirm, col_cancel = st.columns(2)

        with col_confirm:
            if st.button("Confirm — Run Tool", type="primary",
                         use_container_width=True, key="hitl_confirm"):
                st.session_state.pending_interrupt = None

                thinking_slot_h = st.empty()
                with thinking_slot_h.container():
                    with st.chat_message("assistant", avatar="🧊"):
                        st.markdown(THINKING_HTML, unsafe_allow_html=True)

                reply_r, png_paths_r, next_interrupt = resume(st.session_state.thread_id)
                st.session_state.pending_interrupt = next_interrupt

                new_image_r = None
                for raw_path in png_paths_r:
                    raw = Path(raw_path)
                    if raw.exists():
                        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest = _IMAGES_DIR / f"{raw.stem}_{ts}.png"
                        shutil.copy2(raw, dest)
                        save_message_image(st.session_state.thread_id, str(dest))
                        new_image_r = dest
                        break

                thinking_slot_h.empty()
                if reply_r:
                    _render_assistant(reply_r, new_image_r)
                    st.session_state.messages.append({
                        "role": "assistant", "content": reply_r, "image": new_image_r,
                    })

                st.rerun()

        with col_cancel:
            if st.button("Cancel — Skip Tool", type="secondary",
                         use_container_width=True, key="hitl_cancel"):
                st.session_state.pending_interrupt = None
                cancel_text, _, _ = cancel(st.session_state.thread_id)
                _render_assistant(cancel_text)
                st.session_state.messages.append({
                    "role": "assistant", "content": cancel_text, "image": None,
                })
                st.rerun()

# ---------------------------------------------------------------------------
# Input + agent call
# ---------------------------------------------------------------------------

pending    = st.session_state.pending_prompt
user_input = st.chat_input(
    "Message Synera AI…",
    disabled=st.session_state.pending_interrupt is not None,
)

to_process = pending or user_input
if pending:
    st.session_state.pending_prompt = None

# Do not process new input while a tool confirmation is pending
if to_process and st.session_state.pending_interrupt is None:
    # Save conversation title on first message
    if not st.session_state.messages:
        save_conversation(st.session_state.thread_id, to_process)

    # Show user message
    st.session_state.messages.append({"role": "user", "content": to_process, "image": None})
    _render_user(to_process)

    # Thinking indicator
    thinking_slot = st.empty()
    with thinking_slot.container():
        with st.chat_message("assistant", avatar="🧊"):
            st.markdown(THINKING_HTML, unsafe_allow_html=True)

    # Run agent — returns reply text, PNG paths, and optional pending tool
    reply, png_paths, pending_tool = chat(st.session_state.thread_id, to_process)
    st.session_state.pending_interrupt = pending_tool

    new_image = None
    for raw_path in png_paths:
        raw = Path(raw_path)
        if raw.exists():
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest      = _IMAGES_DIR / f"{raw.stem}_{ts}.png"
            shutil.copy2(raw, dest)
            save_message_image(st.session_state.thread_id, str(dest))
            new_image = dest
            break

    # Show reply (may be empty if the graph paused immediately at the interrupt)
    thinking_slot.empty()
    if reply:
        _render_assistant(reply, new_image)
        st.session_state.messages.append({
            "role": "assistant", "content": reply, "image": new_image,
        })

    if pending or pending_tool:
        st.rerun()
