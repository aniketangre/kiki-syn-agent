"""
Integration tests for the RAG pipeline (ingest → search → retrieve).

Requires:
  - Docker container running:  docker compose up -d
  - POSTGRES_URI and OPENAI_API_KEY set in .env
  - Documents ingested:  python tools/rag_search/ingest.py

These tests hit the real pgvector database and make real OpenAI embedding
calls — they verify end-to-end retrieval quality rather than tool wiring.

Run with:  pytest tests/integration/test_rag_pipeline.py
Skip with: pytest -m "not integration"
"""

import os

import pytest

from tools.rag_search.rag_search_tool import rag_search

pytestmark = pytest.mark.integration

# Skip if embeddings API is not configured
_no_openai = not os.environ.get("OPENAI_API_KEY")
_skip_msg  = "OPENAI_API_KEY not set — skipping RAG pipeline test"


# ---------------------------------------------------------------------------
# Domain queries — should return relevant results from indexed papers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_no_openai, reason=_skip_msg)
class TestDomainQueries:
    def test_biomechanics_query_returns_results(self):
        # Broad biomechanics query — verifies general retrieval works
        result = rag_search.invoke({"query": "hip contact forces during walking"})
        assert "No relevant information" not in result
        assert "unavailable" not in result.lower()

    def test_specific_topic_returns_results(self):
        # Specific topic from a different paper — verifies multiple documents are indexed
        result = rag_search.invoke({"query": "lattice structure endoprosthesis graded density"})
        assert "No relevant information" not in result


# ---------------------------------------------------------------------------
# Anti-hallucination — out-of-scope query should return nothing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_no_openai, reason=_skip_msg)
def test_off_topic_query_returns_not_found():
    result = rag_search.invoke({"query": "recipe for pasta carbonara"})
    assert "No relevant information" in result


# ---------------------------------------------------------------------------
# Source attribution — results should cite a source file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_no_openai, reason=_skip_msg)
def test_results_include_source_filename():
    result = rag_search.invoke({"query": "hip contact forces"})
    if "No relevant information" not in result:
        assert ".pdf" in result.lower()
