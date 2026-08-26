"""Shared pytest fixtures.

Env vars are set at module level — before any app module is imported — so
that agent/config.py reads the test values when it constructs Settings().
"""

import os
import tempfile
from uuid import uuid4

# Must come before any import from the agent package.
os.environ["INTERNAL_API_KEY"] = "test-internal-key"
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test"
os.environ["NODE_INTERNAL_BASE_URL"] = "http://node-mock:3000/api"
os.environ["D10_INTERNAL_KEY"] = "test-d10-internal-key"
os.environ["D10_INTERNAL_BASE_URL"] = "http://d10-mock:3000/api"
os.environ["D10_USAGE_OUTBOX_PATH"] = os.path.join(
    tempfile.gettempdir(), f"d10-usage-test-{uuid4()}.sqlite3"
)
os.environ["D10_USAGE_FLUSH_INTERVAL_SECS"] = "3600"

import pytest
from fastapi.testclient import TestClient  # noqa: E402

TEST_KEY = "test-internal-key"
D10_TEST_KEY = "test-d10-internal-key"


@pytest.fixture(scope="session")
def client():
    """FastAPI test client shared across the session."""
    from server import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
