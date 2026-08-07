"""
Integration test — verifies that LangSmith tracing is reachable.

Requires: LANGSMITH_API_KEY set in .env
Run with: pytest tests/test_langsmith.py -v
"""

import os
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="LANGSMITH_API_KEY not set — skipping LangSmith connectivity test",
)
def test_langsmith_connection():
    """Confirm the LangSmith client can list projects without error."""
    from langsmith import Client
    client   = Client()
    projects = list(client.list_projects())
    assert isinstance(projects, list)
