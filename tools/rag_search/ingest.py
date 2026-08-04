"""
ingest.py — Incrementally index documents into PostgreSQL + pgvector.

Run whenever documents change:
    python tools/rag_search/ingest.py

Incremental behaviour
---------------------
Each file's MD5 hash is stored in the `ingested_files` table after indexing.
On subsequent runs:
  - Unchanged files  → skipped (hash matches)
  - New files        → indexed and recorded
  - Modified files   → old chunks deleted, re-indexed with new hash
  - Deleted files    → chunks removed from the vector store

This means only genuinely new or changed content is sent to OpenAI,
keeping re-index runs fast and cheap.

Supported file types: PDF, .txt, .md
"""

import hashlib
import logging
import os
import sys
import warnings
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Suppress noisy PyPDF page-label warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

load_dotenv()

_DOCS_DIR    = Path(__file__).parent / "documents"
_COLLECTION  = "kiki_knowledge"
_MAX_FILE_MB = 50  # skip files larger than this to avoid memory issues


def _get_connection() -> str:
    uri = os.environ.get("POSTGRES_URI", "")
    if not uri:
        print("ERROR: POSTGRES_URI is not set in your .env file.")
        sys.exit(1)
    # langchain-postgres requires the psycopg3 driver prefix
    return uri.replace("postgresql://", "postgresql+psycopg://", 1)


def _get_raw_uri() -> str:
    """Plain URI for psycopg (without the +psycopg driver suffix)."""
    return os.environ.get("POSTGRES_URI", "")


def _md5(path: Path) -> str:
    """Compute MD5 hash of a file to detect changes between runs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_seen_hashes(conn) -> dict[str, str]:
    """Return {filename: hash} for all previously indexed files."""
    rows = conn.execute("SELECT filename, file_hash FROM ingested_files").fetchall()
    return {row[0]: row[1] for row in rows}


def _delete_chunks(conn, filename: str) -> None:
    """Remove all pgvector chunks that belong to a given source file."""
    conn.execute("""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = %s
        )
        AND cmetadata->>'source' LIKE %s
    """, (_COLLECTION, f"%{filename}%"))


def _record_file(conn, filename: str, file_hash: str, chunk_count: int) -> None:
    """Insert or update the ingestion record for a file."""
    conn.execute("""
        INSERT INTO ingested_files (filename, file_hash, chunk_count, ingested_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (filename) DO UPDATE
            SET file_hash   = EXCLUDED.file_hash,
                chunk_count = EXCLUDED.chunk_count,
                ingested_at = EXCLUDED.ingested_at
    """, (filename, file_hash, chunk_count))


def _remove_record(conn, filename: str) -> None:
    conn.execute("DELETE FROM ingested_files WHERE filename = %s", (filename,))


def ingest():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_postgres import PGVector
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_openai import OpenAIEmbeddings

    connection = _get_connection()
    raw_uri    = _get_raw_uri()

    print(f"Documents folder : {_DOCS_DIR}")
    print(f"Vector store     : PostgreSQL + pgvector ({_COLLECTION})\n")

    # Collect all supported files under the documents/ folder
    all_files: list[Path] = []
    for pattern in ("**/*.pdf", "**/*.txt", "**/*.md"):
        all_files.extend(sorted(_DOCS_DIR.glob(pattern)))

    # Filter out the .gitkeep placeholder
    all_files = [f for f in all_files if f.name != ".gitkeep"]

    if not all_files:
        print("No documents found — add PDFs or text files to tools/rag_search/documents/")
        sys.exit(0)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    splitter = RecursiveCharacterTextSplitter(
        # 800 chars keeps chunks within a single topic; 100-char overlap preserves
        # context across chunk boundaries so sentences aren't split mid-thought.
        chunk_size=800,
        chunk_overlap=100,
    )

    with psycopg.connect(raw_uri) as conn:
        seen = _load_seen_hashes(conn)
        current_filenames = {f.name for f in all_files}

        # Remove chunks for files that no longer exist in documents/
        for old_filename in list(seen.keys()):
            if old_filename not in current_filenames:
                print(f"  - {old_filename}  (deleted — removing from index)")
                _delete_chunks(conn, old_filename)
                _remove_record(conn, old_filename)
        conn.commit()

        # Process each file
        for file_path in all_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)

            if size_mb > _MAX_FILE_MB:
                print(f"  ! {file_path.name}  ({size_mb:.0f} MB) — skipped (exceeds {_MAX_FILE_MB} MB limit)")
                continue

            current_hash = _md5(file_path)

            if seen.get(file_path.name) == current_hash:
                print(f"  = {file_path.name}  — unchanged, skipped")
                continue

            action = "updated" if file_path.name in seen else "new"
            print(f"  + {file_path.name}  ({size_mb:.1f} MB) — {action}")

            # Load pages/sections
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            try:
                documents = loader.load()
            except Exception as e:
                print(f"    Warning: could not load — {e}")
                continue

            # Chunk
            chunks = splitter.split_documents(documents)

            # Strip NUL bytes — some PDFs embed binary artifacts that PostgreSQL rejects
            for chunk in chunks:
                chunk.page_content = chunk.page_content.replace('\x00', '')

            # Remove old chunks for this file before adding new ones
            if file_path.name in seen:
                _delete_chunks(conn, file_path.name)
                conn.commit()

            # Embed and store the new chunks
            vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=_COLLECTION,
                connection=connection,
            )
            vectorstore.add_documents(chunks)

            # Record the new hash so this file is skipped next run
            _record_file(conn, file_path.name, current_hash, len(chunks))
            conn.commit()

            print(f"    → {len(chunks)} chunks indexed")

    print("\nDone. The rag_search tool is ready to use.")


if __name__ == "__main__":
    ingest()
