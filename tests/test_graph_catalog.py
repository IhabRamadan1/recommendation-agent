"""Catalog load / empty / malformed paths for Part A."""

from pathlib import Path

from recommendation_graph.graph import run_recommendation
from recommendation_graph.nodes import load_catalog

REPO = Path(__file__).resolve().parents[1]
PROFILE = {
    "id": "t1",
    "name": "Test",
    "interests": ["python", "data"],
    "goals": ["analyze data"],
    "level": "undergraduate",
}


def test_load_good_catalog() -> None:
    state = load_catalog({"catalog_path": str(REPO / "data" / "catalog.example.json")})
    assert state["catalog_ok"] is True
    assert len(state["catalog"]) >= 3


def test_load_empty_catalog() -> None:
    state = load_catalog({"catalog_path": str(REPO / "data" / "catalog.empty.json")})
    assert state["catalog_ok"] is False
    assert state["catalog"] == []
    assert any("empty" in e.lower() for e in state["errors"])


def test_load_malformed_catalog() -> None:
    state = load_catalog({"catalog_path": str(REPO / "data" / "catalog.malformed.json")})
    assert state["catalog_ok"] is False
    assert state["catalog"] == []


def test_graph_fails_closed_on_empty_catalog() -> None:
    result = run_recommendation(PROFILE, REPO / "data" / "catalog.empty.json")
    assert result["validation_passed"] is False
    assert result["catalog_ok"] is False


def test_graph_fails_closed_on_malformed_catalog() -> None:
    result = run_recommendation(PROFILE, REPO / "data" / "catalog.malformed.json")
    assert result["validation_passed"] is False
    assert result["catalog_ok"] is False


def test_load_rejects_duplicate_catalog_ids() -> None:
    state = load_catalog(
        {"catalog_path": str(REPO / "data" / "catalog.duplicate_ids.json")}
    )
    assert state["catalog_ok"] is False
    assert state["catalog"] == []
    assert any("duplicate catalog id" in e.lower() for e in state["errors"])


def test_graph_fails_closed_on_duplicate_catalog_ids() -> None:
    result = run_recommendation(PROFILE, REPO / "data" / "catalog.duplicate_ids.json")
    assert result["validation_passed"] is False
    assert result["catalog_ok"] is False
