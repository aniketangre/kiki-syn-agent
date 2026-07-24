"""
test_langsmith.py
-----------------
Quick check that LangSmith tracing is reachable before running the full app.

Usage:
    python test_langsmith.py
"""

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

try:
    client = Client()
    projects = list(client.list_projects())
    print("LangSmith connection successful.")
    print(f"Projects found: {[p.name for p in projects]}")
except Exception as e:
    print(f"LangSmith connection failed: {e}")
    print("Check: LANGSMITH_API_KEY and LANGSMITH_ENDPOINT in .env")
