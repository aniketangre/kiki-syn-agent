"""
Integration tests for the PostgreSQL conversation helper functions in agent.py.

Requires:
  - Docker container running:  docker compose up -d
  - POSTGRES_URI set in .env

The helpers are tested against a dedicated kiki_agent_test database so the
real conversation data is never touched. agent._pg_conn is patched with the
test connection for each test.

Run with:  pytest tests/integration/test_db_helpers.py
Skip with: pytest -m "not integration"
"""

import pytest
import agent

pytestmark = pytest.mark.integration


@pytest.fixture
def db(pg_test_conn, monkeypatch):
    """
    Patch agent._pg_conn with the test DB connection and wipe tables
    before and after each test so tests are fully isolated.
    """
    monkeypatch.setattr(agent, "_pg_conn", pg_test_conn)
    pg_test_conn.execute("DELETE FROM message_images")
    pg_test_conn.execute("DELETE FROM conversations")
    yield pg_test_conn
    pg_test_conn.execute("DELETE FROM message_images")
    pg_test_conn.execute("DELETE FROM conversations")


# ---------------------------------------------------------------------------
# save_conversation / list_conversations
# ---------------------------------------------------------------------------

class TestSaveAndListConversations:
    def test_saved_conversation_appears_in_list(self, db):
        agent.save_conversation("t-001", "First conversation")
        convs = agent.list_conversations()
        match = next((c for c in convs if c["thread_id"] == "t-001"), None)
        assert match is not None
        assert match["title"] == "First conversation"

    def test_save_is_idempotent(self, db):
        agent.save_conversation("t-002", "Original title")
        agent.save_conversation("t-002", "Should be ignored")
        convs = agent.list_conversations()
        matches = [c for c in convs if c["thread_id"] == "t-002"]
        assert len(matches) == 1
        assert matches[0]["title"] == "Original title"

    def test_title_truncated_at_60_chars(self, db):
        agent.save_conversation("t-003", "A" * 100)
        convs = agent.list_conversations()
        match = next(c for c in convs if c["thread_id"] == "t-003")
        assert len(match["title"]) <= 60

    def test_list_ordered_newest_first(self, db):
        agent.save_conversation("t-004", "First saved")
        agent.save_conversation("t-005", "Second saved")
        convs = agent.list_conversations()
        ids = [c["thread_id"] for c in convs]
        assert ids.index("t-005") < ids.index("t-004")

    def test_empty_list_when_no_conversations(self, db):
        assert agent.list_conversations() == []


# ---------------------------------------------------------------------------
# save_message_image
# ---------------------------------------------------------------------------

class TestSaveMessageImage:
    def test_image_path_is_stored(self, db):
        agent.save_conversation("t-010", "Conv with image")
        agent.save_message_image("t-010", "/tmp/test_image.png")
        rows = db.execute(
            "SELECT image_path FROM message_images WHERE thread_id = %s", ("t-010",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "/tmp/test_image.png"

    def test_multiple_images_per_thread(self, db):
        agent.save_conversation("t-011", "Multi-image conv")
        agent.save_message_image("t-011", "/tmp/img1.png")
        agent.save_message_image("t-011", "/tmp/img2.png")
        rows = db.execute(
            "SELECT image_path FROM message_images WHERE thread_id = %s ORDER BY id", ("t-011",)
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "/tmp/img1.png"
        assert rows[1][0] == "/tmp/img2.png"


# ---------------------------------------------------------------------------
# delete_conversation
# ---------------------------------------------------------------------------

class TestDeleteConversation:
    def test_delete_removes_conversation_row(self, db):
        agent.save_conversation("t-020", "To delete")
        agent.delete_conversation("t-020")
        convs = agent.list_conversations()
        assert not any(c["thread_id"] == "t-020" for c in convs)

    def test_delete_removes_associated_images(self, db):
        agent.save_conversation("t-021", "Conv with image")
        agent.save_message_image("t-021", "/tmp/img.png")
        agent.delete_conversation("t-021")
        rows = db.execute(
            "SELECT id FROM message_images WHERE thread_id = %s", ("t-021",)
        ).fetchall()
        assert rows == []

    def test_delete_nonexistent_thread_does_not_raise(self, db):
        agent.delete_conversation("nonexistent-thread-id")
