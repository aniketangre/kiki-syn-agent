"""
Unit tests for the rag_search tool.

The vector store and embeddings are mocked — no PostgreSQL or OpenAI calls.
Run with:  pytest tests/unit/test_rag_tool.py
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from tools.rag_search.rag_search_tool import rag_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, source: str = "paper.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _mock_vs(hits: list[tuple[Document, float]]) -> MagicMock:
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = hits
    return vs


# ---------------------------------------------------------------------------
# Successful retrieval
# ---------------------------------------------------------------------------

class TestRelevantResults:
    def test_returns_content_and_source(self):
        hits = [(_doc("Hip contact force is 238% body weight.", "Bergmann2001.pdf"), 0.2)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "hip contact force"})

        assert "238%" in result
        assert "Bergmann2001.pdf" in result

    def test_numbers_results(self):
        hits = [
            (_doc("First chunk.", "a.pdf"), 0.1),
            (_doc("Second chunk.", "b.pdf"), 0.3),
        ]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "something"})

        assert "[1]" in result
        assert "[2]" in result

    def test_includes_score_in_output(self):
        hits = [(_doc("Content here.", "paper.pdf"), 0.25)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "query"})

        assert "0.250" in result or "0.25" in result


# ---------------------------------------------------------------------------
# Threshold filtering — scores >= 0.5 should be discarded
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    def test_filters_high_distance_results(self):
        hits = [(_doc("Irrelevant content.", "other.pdf"), 0.8)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "hip forces"})

        assert "No relevant information" in result

    def test_keeps_results_just_below_threshold(self):
        hits = [(_doc("Borderline relevant.", "paper.pdf"), 0.49)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "hip forces"})

        assert "Borderline relevant" in result

    def test_discards_results_at_threshold(self):
        hits = [(_doc("Exactly at threshold.", "paper.pdf"), 0.5)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "hip forces"})

        assert "No relevant information" in result

    def test_mixed_relevance_returns_only_relevant(self):
        hits = [
            (_doc("Relevant chunk.", "good.pdf"), 0.2),
            (_doc("Irrelevant chunk.", "bad.pdf"), 0.9),
        ]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "query"})

        assert "Relevant chunk" in result
        assert "Irrelevant chunk" not in result


# ---------------------------------------------------------------------------
# Empty and unavailable states
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_result_set(self):
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs([])):
            result = rag_search.invoke({"query": "quantum computing"})

        assert "No relevant information" in result

    def test_db_unavailable_returns_error_message(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.side_effect = Exception("connection refused")
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=mock_vs):
            result = rag_search.invoke({"query": "hip forces"})

        assert "unavailable" in result.lower()

    def test_out_of_scope_query_filtered(self):
        hits = [(_doc("Recipe for pasta carbonara.", "food.pdf"), 0.95)]
        with patch("tools.rag_search.rag_search_tool._get_vectorstore", return_value=_mock_vs(hits)):
            result = rag_search.invoke({"query": "recipe for pasta carbonara"})

        assert "No relevant information" in result
