def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["service"] == "laby-adk"
    assert data["provider"] == "openrouter"
    # Every model is routed through OpenRouter, never a provider API directly.
    assert data["model"].startswith("openrouter/")
