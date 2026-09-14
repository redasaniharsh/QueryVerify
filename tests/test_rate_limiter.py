"""
Unit tests for the sliding-window rate limiter and /ask rate-limiting behavior.

These tests do not require a live backend, database, or Ollama instance so
they run cleanly in CI and stay green offline.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.rate_limiter import SlidingWindowRateLimiter, LOOPBACK_HOSTS
from app.main import app, rate_limiter


def test_rate_limiter_basic_window():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60.0, exempt_loopback=False)
    client = "192.0.2.1"

    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is True
    # 4th request within window must be rejected
    assert limiter.is_allowed(client) is False


def test_rate_limiter_independent_clients():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60.0, exempt_loopback=False)
    client_a = "192.0.2.1"
    client_b = "192.0.2.2"

    assert limiter.is_allowed(client_a) is True
    assert limiter.is_allowed(client_a) is True
    assert limiter.is_allowed(client_a) is False

    # Client B should still be allowed their own quota
    assert limiter.is_allowed(client_b) is True
    assert limiter.is_allowed(client_b) is True
    assert limiter.is_allowed(client_b) is False


def test_rate_limiter_window_expiry(monkeypatch):
    current_time = 1000.0

    def mock_monotonic():
        return current_time

    monkeypatch.setattr("time.monotonic", mock_monotonic)

    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60.0, exempt_loopback=False)
    client = "192.0.2.1"

    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is False

    # Advance time past the 60-second window
    current_time += 61.0

    # Should allow requests again
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is False


def test_rate_limiter_exempt_loopback():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60.0, exempt_loopback=True)

    for host in LOOPBACK_HOSTS:
        for _ in range(5):
            assert limiter.is_allowed(host) is True, f"Host {host} should be exempt"

    # Non-loopback host is still subject to the limit
    external_ip = "203.0.113.50"
    assert limiter.is_allowed(external_ip) is True
    assert limiter.is_allowed(external_ip) is False


def test_rate_limiter_disabled_when_non_positive():
    limiter = SlidingWindowRateLimiter(limit=0, window_seconds=60.0, exempt_loopback=False)
    client = "192.0.2.1"
    for _ in range(10):
        assert limiter.is_allowed(client) is True


def test_rate_limiter_reset():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60.0, exempt_loopback=False)
    client = "192.0.2.1"
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is True
    assert limiter.is_allowed(client) is False

    limiter.reset()
    assert limiter.is_allowed(client) is True


# --- FastAPI Endpoint Integration Tests (using TestClient) ---

@pytest.fixture(autouse=True)
def reset_global_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_ask_rate_limiting_with_test_client():
    client = TestClient(app)
    fake_ip = "198.51.100.99"
    headers = {"X-Forwarded-For": fake_ip}

    # Set limiter limit to 3 for this test
    original_limit = rate_limiter.limit
    original_exempt = rate_limiter.exempt_loopback
    rate_limiter.limit = 3
    rate_limiter.exempt_loopback = True

    try:
        mock_result = {
            "sql": "SELECT 1",
            "result": {"columns": ["col"], "rows": [{"col": 1}]},
            "confidence": "high",
            "confidence_score": 0.95,
            "explanation": "Test answer",
            "attempts": 1,
            "success": True,
            "needs_clarification": False,
        }

        with patch("app.main.handle_question", return_value=mock_result), \
             patch("app.main.get_engine"):
            # First 3 requests must succeed
            for i in range(3):
                resp = client.post(
                    "/ask",
                    json={"question": f"Question {i}"},
                    headers=headers,
                )
                assert resp.status_code == 200, f"Request {i+1} should succeed, got {resp.status_code}"
                assert resp.json()["question"] == f"Question {i}"

            # 4th request must return 429 Too Many Requests
            resp = client.post(
                "/ask",
                json={"question": "Excess question"},
                headers=headers,
            )
            assert resp.status_code == 429
            data = resp.json()
            assert "Rate limit exceeded" in data["detail"]

            # /health must still succeed even when /ask is rate limited for this IP
            health_resp = client.get("/health", headers=headers)
            assert health_resp.status_code == 200
            assert health_resp.json() == {"status": "ok"}
    finally:
        rate_limiter.limit = original_limit
        rate_limiter.exempt_loopback = original_exempt


def test_health_always_unmetered():
    client = TestClient(app)
    headers = {"X-Forwarded-For": "198.51.100.100"}

    rate_limiter.limit = 1
    rate_limiter.exempt_loopback = False

    try:
        # Exhaust quota on /ask
        with patch("app.main.handle_question", return_value={"success": True, "attempts": 1, "needs_clarification": False}), \
             patch("app.main.get_engine"):
            resp = client.post("/ask", json={"question": "q1"}, headers=headers)
            assert resp.status_code == 200
            resp_blocked = client.post("/ask", json={"question": "q2"}, headers=headers)
            assert resp_blocked.status_code == 429

        # /health should still respond normally
        for _ in range(5):
            resp_health = client.get("/health", headers=headers)
            assert resp_health.status_code == 200
            assert resp_health.json() == {"status": "ok"}
    finally:
        rate_limiter.limit = 10
        rate_limiter.exempt_loopback = True
