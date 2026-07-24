"""
test_fallback.py
----------------
Verifies that the OpenAI → Gemini fallback in agent3.py works correctly.

Simulates an OpenAI RateLimitError and confirms that Gemini picks up the call.

Usage:
    python test_fallback.py
"""

from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()

import openai
from agent3 import _base_llm, _supervisor_llm, _RouteDecision
from langchain_core.messages import HumanMessage


def _make_rate_limit_error():
    """Build a realistic openai.RateLimitError for the mock to raise."""
    response = openai.BadRequestError.__new__(openai.BadRequestError)
    return openai.RateLimitError(
        message="Rate limit exceeded (simulated)",
        response=None,
        body=None,
    )


print("=" * 55)
print("  agent3.py — Fallback Test")
print("  Simulating OpenAI RateLimitError → expect Gemini reply")
print("=" * 55)

# ── Test 1: _base_llm fallback ──────────────────────────────
print("\n[1] Testing _base_llm (used by sub-agents and responder)...")
try:
    with patch(
        "langchain_openai.ChatOpenAI.invoke",
        side_effect=openai.RateLimitError(
            message="Rate limit exceeded (simulated)",
            response=None,
            body=None,
        ),
    ):
        reply = _base_llm.invoke([HumanMessage(content="Say 'fallback ok' and nothing else.")])
        text  = reply.content if isinstance(reply.content, str) else str(reply.content)
        print(f"   ✓ Got reply from fallback model: {text[:80]}")
except Exception as e:
    print(f"   ✗ Fallback did not fire — error: {e}")

# ── Test 2: _supervisor_llm fallback ───────────────────────
print("\n[2] Testing _supervisor_llm (routing with structured output)...")
try:
    with patch(
        "langchain_openai.ChatOpenAI.invoke",
        side_effect=openai.RateLimitError(
            message="Rate limit exceeded (simulated)",
            response=None,
            body=None,
        ),
    ):
        decision = _supervisor_llm.invoke([
            HumanMessage(content="The user wants to create a BCC lattice. Route this.")
        ])
        print(f"   ✓ Got routing decision from fallback model: next='{decision.next}'")
except Exception as e:
    print(f"   ✗ Fallback did not fire — error: {e}")

print("\n" + "=" * 55)
print("  Done. Both ✓ = fallback is working correctly.")
print("=" * 55)
