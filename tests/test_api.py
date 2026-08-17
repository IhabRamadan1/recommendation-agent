"""API: health, recommend, idempotency, fault isolation."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agentic_service.app import _cache, app
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
