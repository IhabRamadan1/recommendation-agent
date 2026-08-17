"""Shared ranking helper consistency."""

from pathlib import Path

from recommendation_graph.nodes import load_catalog, rank_items, write_narrative
from recommendation_graph.ranking import compute_ranked_items
from recommendation_graph.state import CatalogItem, UserProfile

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "data" / "catalog.example.json"
PROFILE = {
    "id": "profile-stem-builder",
    "name": "Alex",
    "interests": ["coding", "python", "data", "stem"],
    "goals": ["build products", "analyze data"],
    "level": "undergraduate",
}


def test_compute_ranked_items_deterministic_order() -> None:
    import json

    catalog = [CatalogItem.model_validate(c) for c in json.loads(CATALOG.read_text())]
    profile = UserProfile.model_validate(PROFILE)
    a = compute_ranked_items(profile, catalog, limit=5)
    b = compute_ranked_items(profile, catalog, limit=5)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert len(a) <= 5
    assert all(item.id in {c.id for c in catalog} for item in a)


def test_rank_items_node_uses_shared_helper() -> None:
    loaded = load_catalog({"catalog_path": str(CATALOG)})
    out = rank_items({**loaded, "profile": PROFILE})
    catalog = [CatalogItem.model_validate(c) for c in loaded["catalog"]]
    expected = compute_ranked_items(UserProfile.model_validate(PROFILE), catalog, limit=5)
    assert out["ranked_items"] == [r.model_dump() for r in expected]


def test_write_narrative_uses_shared_helper_not_rank_node(monkeypatch) -> None:
    """Narrative must call compute_ranked_items, not the LangGraph rank_items node."""
    loaded = load_catalog({"catalog_path": str(CATALOG)})
    state = {**loaded, "profile": PROFILE, "ranked_items": []}

    called = {"helper": 0, "node": 0}
    real_helper = compute_ranked_items

    def tracking_helper(*args, **kwargs):
        called["helper"] += 1
        return real_helper(*args, **kwargs)

    def boom_rank_node(state):
        called["node"] += 1
        raise AssertionError("write_narrative must not call rank_items node")

    monkeypatch.setattr("recommendation_graph.nodes.compute_ranked_items", tracking_helper)
    monkeypatch.setattr("recommendation_graph.nodes.rank_items", boom_rank_node)

    out = write_narrative(state)
    assert called["helper"] >= 1
    assert called["node"] == 0
    assert out["narrative_failed"] is False
    assert out["narrative"]
