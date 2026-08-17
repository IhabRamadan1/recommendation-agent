"""Deterministic validator constraints."""

from recommendation_graph.validators import validate_recommendation_state

CATALOG = [
    {
        "id": "a",
        "title": "Alpha Career",
        "tags": ["stem"],
        "score_hint": 0.9,
    },
    {
        "id": "b",
        "title": "Beta Career",
        "tags": ["design"],
        "score_hint": 0.8,
    },
]

RANKED = [
    {"id": "a", "title": "Alpha Career", "score": 0.9, "matched_tags": ["stem"]},
]


def test_validator_passes_when_narrative_uses_ranked_only() -> None:
    passed, messages = validate_recommendation_state(
        catalog=CATALOG,
        catalog_ok=True,
        ranked_items=RANKED,
        narrative="Alpha Career is a strong fit for this student.",
        rank_failed=False,
    )
    assert passed is True
    assert any("passed" in m.lower() for m in messages)


def test_validator_rejects_narrative_citing_non_ranked_catalog_item() -> None:
    passed, messages = validate_recommendation_state(
        catalog=CATALOG,
        catalog_ok=True,
        ranked_items=RANKED,
        narrative="Consider Beta Career as well as Alpha Career.",
        rank_failed=False,
    )
    assert passed is False
    assert any("outside the ranked" in m for m in messages)


def test_validator_rejects_unknown_ranked_id() -> None:
    passed, _ = validate_recommendation_state(
        catalog=CATALOG,
        catalog_ok=True,
        ranked_items=[
            {"id": "zzz", "title": "Ghost", "score": 1.0, "matched_tags": []},
        ],
        narrative="Ghost is great.",
        rank_failed=False,
    )
    assert passed is False


def test_validator_fails_closed_when_catalog_not_ok() -> None:
    passed, _ = validate_recommendation_state(
        catalog=[],
        catalog_ok=False,
        ranked_items=RANKED,
        narrative="Alpha Career",
        rank_failed=False,
    )
    assert passed is False
