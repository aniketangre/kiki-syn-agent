"""
db_setup.py
-----------
One-time setup script for the kiki-syn-agent conversation database.

Run this ONCE before starting the agent for the first time:
    python db_setup.py

What it does:
  1. Connects to your local PostgreSQL server
  2. Creates the 'kiki_agent' database if it does not exist
  3. Creates the LangGraph checkpointer tables inside that database

Requirements:
  - PostgreSQL must be installed and running locally
  - The user in POSTGRES_URI must have CREATE DATABASE permissions

Configuration:
  Set POSTGRES_URI in your .env file, for example:
      POSTGRES_URI=postgresql://postgres:yourpassword@localhost:5432/kiki_agent
"""

import sys

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import os

load_dotenv()

POSTGRES_URI = os.environ.get("POSTGRES_URI", "")

if not POSTGRES_URI:
    print("ERROR: POSTGRES_URI is not set in your .env file.")
    print("Add a line like:")
    print("  POSTGRES_URI=postgresql://postgres:yourpassword@localhost:5432/kiki_agent")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1: Create the database if it doesn't exist
# ---------------------------------------------------------------------------

# Parse out the database name and build a URI pointing to the default 'postgres' DB
# so we can run CREATE DATABASE without being connected to the target DB.
from urllib.parse import urlparse, urlunparse

parsed = urlparse(POSTGRES_URI)
db_name = parsed.path.lstrip("/")
admin_uri = urlunparse(parsed._replace(path="/postgres"))

print(f"Connecting to PostgreSQL at {parsed.hostname}:{parsed.port} ...")

try:
    with psycopg.connect(admin_uri, autocommit=True) as conn:
        cur = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        )
        if cur.fetchone() is None:
            conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")
except psycopg.OperationalError as e:
    print(f"\nERROR: Could not connect to PostgreSQL.\n{e}")
    print("\nMake sure PostgreSQL is running and the credentials in POSTGRES_URI are correct.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Create LangGraph checkpointer tables inside the target database
# ---------------------------------------------------------------------------

print("Setting up LangGraph checkpointer tables ...")

with psycopg.connect(POSTGRES_URI) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

print("\nSetup complete. You can now run the agent:")
print("  python agent.py")
