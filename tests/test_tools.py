"""Allowlist tool execution."""

from pathlib import Path

from agentic_service.tools import TOOL_SPECS, execute_tool

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "data" / "catalog.example.json"


def test_reject_unknown_tool() -> None:
    result = execute_tool("rm_rf", {}, catalog_path=CATALOG)
    assert result["ok"] is False
    assert "allowlist" in result["error"].lower() or "not in the allowlist" in result["error"]


def test_lookup_catalog_item() -> None:
    result = execute_tool(
        "lookup_catalog_item",
        {"item_id": "career-data-analyst"},
        catalog_path=CATALOG,
    )
    assert result["ok"] is True
    assert result["item"]["title"] == "Data Analyst"


def test_lookup_missing_item() -> None:
    result = execute_tool(
        "lookup_catalog_item",
        {"item_id": "nope"},
        catalog_path=CATALOG,
    )
    assert result["ok"] is False


def test_score_profile_fit() -> None:
    result = execute_tool(
        "score_profile_fit",
        {
            "profile": {
                "id": "p",
                "name": "A",
                "interests": ["python", "data"],
                "goals": ["analyze data"],
                "level": "undergraduate",
            }
        },
        catalog_path=CATALOG,
    )
    assert result["ok"] is True
    assert result["ranked_items"]


def test_summarize_top_items() -> None:
    result = execute_tool(
        "summarize_top_items",
        {"ranked_items": [{"id": "a", "title": "Alpha"}]},
        catalog_path=CATALOG,
    )
    assert result["ok"] is True
    assert "Alpha" in result["summary"]


def test_invalid_args_rejected() -> None:
    result = execute_tool("lookup_catalog_item", {}, catalog_path=CATALOG)
    assert result["ok"] is False


def test_allowlist_names() -> None:
    assert set(TOOL_SPECS) == {
        "lookup_catalog_item",
        "score_profile_fit",
        "summarize_top_items",
    }
