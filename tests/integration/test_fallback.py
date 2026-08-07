"""
Integration test — verifies that the OpenAI → Gemini fallback in agent.py fires
correctly when OpenAI raises a rate limit or connection error.

Requires: OPENAI_API_KEY and GOOGLE_API_KEY set in .env
Run with: pytest tests/test_fallback.py -v -s
"""

import pytest
import openai
from unittest.mock import patch
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def agent_llms():
    from agent import _base_llm, _supervisor_llm, _RouteDecision
    return _base_llm, _supervisor_llm, _RouteDecision


def _rate_limit_error():
    import httpx
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return openai.RateLimitError(
        message="Rate limit exceeded (simulated)",
        response=response,
        body=None,
    )


def test_base_llm_falls_back_to_gemini(agent_llms):
    """_base_llm should transparently switch to Gemini on RateLimitError."""
    base_llm, _, _ = agent_llms
    with patch("langchain_openai.ChatOpenAI.invoke", side_effect=_rate_limit_error()):
        reply = base_llm.invoke([HumanMessage(content="Say 'fallback ok' and nothing else.")])
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    assert text.strip() != ""


def test_supervisor_llm_falls_back_to_gemini(agent_llms):
    """_supervisor_llm should return a valid RouteDecision from Gemini on RateLimitError."""
    _, supervisor_llm, RouteDecision = agent_llms
    with patch("langchain_openai.ChatOpenAI.invoke", side_effect=_rate_limit_error()):
        decision = supervisor_llm.invoke([
            HumanMessage(content="The user wants to create a BCC lattice. Route this.")
        ])
    assert isinstance(decision, RouteDecision)
    assert decision.next in ("kiki", "synera", "knowledge", "responder")
