"""
Shared fixtures and configuration for the kiki-syn-agent test suite.

Unit tests run with no external dependencies (all I/O is mocked).
Integration tests require a live PostgreSQL container and are marked with
@pytest.mark.integration — skip them with:  pytest -m "not integration"
"""

import os
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()

# Derive a dedicated test DB URI so integration tests never touch the real DB.
# E.g. postgresql://postgres:pw@localhost:5432/kiki_agent
#    → postgresql://postgres:pw@localhost:5432/kiki_agent_test
_base_uri = os.environ.get("POSTGRES_URI", "")
if _base_uri:
    _parsed = urlparse(_base_uri)
    POSTGRES_TEST_URI = urlunparse(_parsed._replace(path="/kiki_agent_test"))
else:
    POSTGRES_TEST_URI = ""


def pytest_runtest_setup(item):
    """Skip integration tests automatically when PostgreSQL is unreachable."""
    if "integration" in item.keywords and not POSTGRES_TEST_URI:
        pytest.skip("POSTGRES_URI not set — skipping integration test")


# ---------------------------------------------------------------------------
# Session-scoped fixture: create test DB and tables once per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_test_conn():
    """
    Connect to kiki_agent_test (created if absent), set up required tables,
    yield the connection, then drop the tables on teardown.
    """
    parsed   = urlparse(POSTGRES_TEST_URI)
    db_name  = parsed.path.lstrip("/")
    admin_uri = urlunparse(parsed._replace(path="/postgres"))

    with psycopg.connect(admin_uri, autocommit=True) as admin:
        if not admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        ).fetchone():
            admin.execute(f'CREATE DATABASE "{db_name}"')

    conn = psycopg.connect(POSTGRES_TEST_URI, autocommit=True, prepare_threshold=0)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id  TEXT        PRIMARY KEY,
            title      TEXT        NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_images (
            id          BIGSERIAL   PRIMARY KEY,
            thread_id   TEXT        NOT NULL,
            image_path  TEXT        NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    yield conn

    conn.execute("DROP TABLE IF EXISTS message_images")
    conn.execute("DROP TABLE IF EXISTS conversations")
    conn.close()
