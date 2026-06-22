"""Shared pytest fixtures.

Env vars are set at module level — before any app module is imported — so
that agent/config.py reads the test values when it constructs Settings().
"""

import os

# Must come before any import from the agent package.
os.environ["INTERNAL_API_KEY"] = "test-internal-key"
os.environ["DEEPSEEK_API_KEY"] = "sk-test"
os.environ["NODE_INTERNAL_BASE_URL"] = "http://node-mock:3000/api"

import pytest
from fastapi.testclient import TestClient  # noqa: E402

TEST_KEY = "test-internal-key"


@pytest.fixture(scope="session")
def client():
    """FastAPI test client shared across the session."""
    from server import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
