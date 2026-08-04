"""
rag_search_tool.py — Semantic search tool over the project knowledge base.

Overview
--------
Exposes a single LangChain tool (`rag_search`) that the knowledge_agent calls
whenever the user asks a question that should be answered from the indexed
research documents rather than from the LLM's general training knowledge.

How it works
------------
1. The user's query is embedded into a vector using the same OpenAI model
   that was used during ingestion (text-embedding-3-small).
2. The vector is compared against all stored chunk vectors in PostgreSQL
   using pgvector's cosine distance.
3. The four closest chunks are returned, filtered by a distance threshold
   to suppress low-relevance results.
4. The agent synthesises an answer from those chunks and cites the source.

Dependencies
------------
- PostgreSQL with the pgvector extension (Docker: `docker compose up -d`).
- POSTGRES_URI and OPENAI_API_KEY must be set in .env.
- Run tools/rag_search/ingest.py once to build the index before using this tool.

Score threshold
---------------
_SCORE_THRESHOLD controls how strict the relevance filter is.
  0.0 = only return a perfect match
  0.5 = return reasonably relevant chunks  (current setting)
  1.0 = return almost anything
Raise the value if the agent says "not found" too often; lower it if it
returns irrelevant content.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

_COLLECTION      = "kiki_knowledge"
_vectorstore     = None
_SCORE_THRESHOLD = 0.5


def _get_connection() -> str:
    uri = os.environ.get("POSTGRES_URI", "")
    if not uri:
        raise RuntimeError("POSTGRES_URI is not set in .env")
    # langchain-postgres requires the psycopg3 driver prefix
    return uri.replace("postgresql://", "postgresql+psycopg://", 1)


def _get_vectorstore() -> PGVector:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.environ.get("OPENAI_API_KEY"),
            ),
            collection_name=_COLLECTION,
            connection=_get_connection(),
        )
    return _vectorstore


@tool
def rag_search(query: str) -> str:
    """
    Search the project knowledge base for relevant information about Synera,
    KIKI, lattice structures, implant design, or biomechanics.

    Use this when the user asks how something works, requests an explanation,
    or needs reference information from the documentation.

    Args:
        query: Plain-English question or search phrase.

    Returns:
        Relevant excerpts from the knowledge base, or a message if nothing found.
    """
    try:
        docs_with_scores = _get_vectorstore().similarity_search_with_score(query, k=4)
    except Exception as e:
        return f"Knowledge base unavailable: {e}. Run ingest.py to build the index."

    relevant = [(doc, score) for doc, score in docs_with_scores if score < _SCORE_THRESHOLD]

    if not relevant:
        return "No relevant information found in the knowledge base for this query."

    parts = []
    for i, (doc, score) in enumerate(relevant, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[{i}] {source} (score: {score:.3f})\n{doc.page_content}")

    return "\n\n".join(parts)
