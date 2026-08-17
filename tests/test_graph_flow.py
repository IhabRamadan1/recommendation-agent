"""End-to-end graph flow: parallel merge and failed-branch handling."""

from pathlib import Path
from unittest.mock import patch

from recommendation_graph.graph import run_recommendation
from recommendation_graph.nodes import rank_items

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "data" / "catalog.example.json"
PROFILE = {
    "id": "profile-stem-builder",
    "name": "Alex",
    "interests": ["coding", "python", "data", "stem"],
    "goals": ["build products", "analyze data"],
    "level": "undergraduate",
}


def test_happy_path_produces_ranked_and_narrative() -> None:
    result = run_recommendation(PROFILE, CATALOG)
    assert result["catalog_ok"] is True
    assert result["validation_passed"] is True
    assert result["ranked_items"]
    assert result["narrative"]
    ranked_titles = {r["title"] for r in result["ranked_items"]}
    # Narrative template mentions top titles from ranked set.
    assert any(t in result["narrative"] for t in ranked_titles)


def test_rank_is_catalog_constrained() -> None:
    result = run_recommendation(PROFILE, CATALOG)
    catalog_ids = {c["id"] for c in result["catalog"]}
    for item in result["ranked_items"]:
        assert item["id"] in catalog_ids


def test_failed_rank_branch_fails_validation() -> None:
    with patch(
        "recommendation_graph.graph.rank_items",
        side_effect=lambda state: {
            "ranked_items": [],
            "rank_failed": True,
            "errors": ["injected rank failure"],
        },
    ):
        result = run_recommendation(PROFILE, CATALOG)
    assert result["validation_passed"] is False
    assert result.get("rank_failed") is True

def test_narrative_failure_still_merges_rank_when_rank_ok() -> None:
    """If narrative fails but rank succeeds, validator should fail (empty narrative)."""

    def boom_narrative(state):
        return {
            "narrative": "",
            "narrative_failed": True,
            "errors": ["injected narrative failure"],
        }

    with patch("recommendation_graph.graph.write_narrative", side_effect=boom_narrative):
        result = run_recommendation(PROFILE, CATALOG)
    assert result.get("ranked_items")  # rank branch still contributed
    assert result["validation_passed"] is False


def test_rank_items_unit() -> None:
    from recommendation_graph.nodes import load_catalog

    loaded = load_catalog({"catalog_path": str(CATALOG)})
    out = rank_items({**loaded, "profile": PROFILE})
    assert out["rank_failed"] is False
    assert len(out["ranked_items"]) <= 5
