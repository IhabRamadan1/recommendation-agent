"""API: health, recommend, idempotency fingerprinting, fault isolation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from agentic_service.app import (
    AgentInvokeResponse,
    _cache,
    app,
    parse_exactly_one_tool_call,
    request_fingerprint,
)
from agentic_service.app import AgentInvokeRequest
from config.settings import REPO_ROOT

PROFILE = {
    "id": "profile-stem-builder",
    "name": "Alex",
    "interests": ["coding", "python", "data", "stem"],
    "goals": ["build products", "analyze data"],
    "level": "undergraduate",
}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    asyncio.run(_cache.clear())
    yield
    asyncio.run(_cache.clear())


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_recommend_endpoint(client: TestClient) -> None:
    res = client.post(
        "/recommend",
        json={
            "profile": PROFILE,
            "catalog_path": str(REPO_ROOT / "data" / "catalog.example.json"),
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["validation_passed"] is True
    assert body["ranked_items"]


def test_agent_force_tool(client: TestClient) -> None:
    res = client.post(
        "/agent/invoke",
        json={
            "message": "lookup",
            "force_tool": "lookup_catalog_item",
            "force_args": {"item_id": "career-software-engineer"},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["tool"] == "lookup_catalog_item"


def test_agent_rejects_unknown_forced_tool(client: TestClient) -> None:
    res = client.post(
        "/agent/invoke",
        json={"message": "hack", "force_tool": "eval", "force_args": {}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False


def test_idempotency_replay(client: TestClient) -> None:
    payload = {
        "message": "lookup",
        "force_tool": "lookup_catalog_item",
        "force_args": {"item_id": "career-ux-designer"},
    }
    headers = {"Idempotency-Key": "req-123"}
    first = client.post("/agent/invoke", json=payload, headers=headers)
    second = client.post("/agent/invoke", json=payload, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == second.json()["result"]
    assert second.json()["idempotent_replay"] is True
    assert first.json()["idempotent_replay"] is False


def test_idempotency_same_key_different_payload_conflict(client: TestClient) -> None:
    headers = {"Idempotency-Key": "conflict-key"}
    first = client.post(
        "/agent/invoke",
        json={
            "message": "lookup-a",
            "force_tool": "lookup_catalog_item",
            "force_args": {"item_id": "career-ux-designer"},
        },
        headers=headers,
    )
    second = client.post(
        "/agent/invoke",
        json={
            "message": "lookup-b",
            "force_tool": "lookup_catalog_item",
            "force_args": {"item_id": "career-data-analyst"},
        },
        headers=headers,
    )
    assert first.status_code == 200
    assert second.status_code == 409


def test_fault_isolation_bad_request_does_not_break_next(client: TestClient) -> None:
    bad = client.post(
        "/agent/invoke",
        json={"message": "x", "force_tool": "lookup_catalog_item", "force_args": {}},
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False

    good = client.post(
        "/agent/invoke",
        json={
            "message": "x",
            "force_tool": "summarize_top_items",
            "force_args": {"ranked_items": [{"id": "a", "title": "A"}]},
        },
    )
    assert good.status_code == 200
    assert good.json()["ok"] is True


def test_request_fingerprint_stable_and_sensitive() -> None:
    a = AgentInvokeRequest(
        message="m",
        force_tool="lookup_catalog_item",
        force_args={"item_id": "x"},
    )
    b = AgentInvokeRequest(
        message="m",
        force_tool="lookup_catalog_item",
        force_args={"item_id": "x"},
    )
    c = AgentInvokeRequest(
        message="m",
        force_tool="lookup_catalog_item",
        force_args={"item_id": "y"},
    )
    assert request_fingerprint(a) == request_fingerprint(b)
    assert request_fingerprint(a) != request_fingerprint(c)


@pytest.mark.asyncio
async def test_concurrent_same_key_invokes_underlying_once() -> None:
    calls = {"n": 0}

    async def fake_run(body: AgentInvokeRequest) -> AgentInvokeResponse:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return AgentInvokeResponse(
            ok=True,
            tool=body.force_tool,
            result={"ok": True, "n": calls["n"]},
        )

    payload = {
        "message": "lookup",
        "force_tool": "lookup_catalog_item",
        "force_args": {"item_id": "career-software-engineer"},
    }
    headers = {"Idempotency-Key": "concurrent-once"}

    with patch("agentic_service.app._run_agent", new=AsyncMock(side_effect=fake_run)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r1, r2 = await asyncio.gather(
                ac.post("/agent/invoke", json=payload, headers=headers),
                ac.post("/agent/invoke", json=payload, headers=headers),
            )

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["n"] == 1
    bodies = [r1.json(), r2.json()]
    assert bodies[0]["result"] == bodies[1]["result"]
    assert sum(1 for b in bodies if b["idempotent_replay"]) == 1


class _FakeMsg:
    def __init__(self, tool_calls=None, additional_kwargs=None):
        self.tool_calls = tool_calls
        self.additional_kwargs = additional_kwargs or {}


def test_parse_exactly_one_tool_call_accepts_single() -> None:
    name, args = parse_exactly_one_tool_call(
        _FakeMsg(tool_calls=[{"name": "lookup_catalog_item", "args": {"item_id": "x"}}])
    )
    assert name == "lookup_catalog_item"
    assert args == {"item_id": "x"}


def test_parse_zero_tool_calls_rejected() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        parse_exactly_one_tool_call(_FakeMsg(tool_calls=[]))
    assert exc.value.status_code == 400
    assert "did not select" in exc.value.detail.lower()


def test_parse_multiple_tool_calls_rejected() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        parse_exactly_one_tool_call(
            _FakeMsg(
                tool_calls=[
                    {"name": "lookup_catalog_item", "args": {"item_id": "a"}},
                    {"name": "summarize_top_items", "args": {"ranked_items": [{"id": "a"}]}},
                ]
            )
        )
    assert exc.value.status_code == 400
    assert "exactly one" in exc.value.detail.lower()


def test_parse_additional_kwargs_zero_and_multiple() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        parse_exactly_one_tool_call(_FakeMsg(tool_calls=None, additional_kwargs={"tool_calls": []}))

    with pytest.raises(HTTPException) as exc:
        parse_exactly_one_tool_call(
            _FakeMsg(
                tool_calls=None,
                additional_kwargs={
                    "tool_calls": [
                        {"function": {"name": "lookup_catalog_item", "arguments": "{}"}},
                        {"function": {"name": "score_profile_fit", "arguments": "{}"}},
                    ]
                },
            )
        )
    assert "exactly one" in exc.value.detail.lower()
