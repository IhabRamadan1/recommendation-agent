"""Deterministic (non-LLM) validation gate for recommendation graph outputs.

Business: never ship recommendations or narratives that invent items outside the
ranked, catalog-constrained set — the core safety pattern for agentic features.
Technical: pure Python checks; no model calls.
"""

from __future__ import annotations

import re

from recommendation_graph.state import CatalogItem, RankedItem


def validate_recommendation_state(
    *,
    catalog: list[dict],
    catalog_ok: bool,
    ranked_items: list[dict],
    narrative: str,
    rank_failed: bool,
    errors: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return (passed, messages). Fail closed on empty/malformed catalog or bad refs."""
    messages: list[str] = []
    errors = errors or []

    if not catalog_ok:
        messages.append("Catalog was empty or malformed; validation failed closed.")
        return False, messages

    if not catalog:
        messages.append("Catalog is empty; cannot validate recommendations.")
        return False, messages

    catalog_by_id: dict[str, CatalogItem] = {}
    try:
        for raw in catalog:
            item = CatalogItem.model_validate(raw)
            catalog_by_id[item.id] = item
    except Exception as exc:  # noqa: BLE001 — surface as validation failure
        messages.append(f"Catalog failed schema validation: {exc}")
        return False, messages

    if rank_failed or not ranked_items:
        messages.append("Rank branch produced no usable ranked list.")
        return False, messages

    ranked: list[RankedItem] = []
    try:
        for raw in ranked_items:
            ranked.append(RankedItem.model_validate(raw))
    except Exception as exc:  # noqa: BLE001
        messages.append(f"Ranked items failed schema validation: {exc}")
        return False, messages

    for item in ranked:
        if item.id not in catalog_by_id:
            messages.append(f"Ranked id {item.id!r} is not in the reference catalog.")
            return False, messages
        if catalog_by_id[item.id].title != item.title:
            messages.append(
                f"Ranked title for {item.id!r} does not match catalog title."
            )
            return False, messages

    allowed_titles = {r.title for r in ranked}
    allowed_ids = {r.id for r in ranked}
    narrative_text = (narrative or "").strip()
    if not narrative_text:
        messages.append("Narrative is empty.")
        return False, messages

    # Titles/ids from the full catalog that are NOT in the ranked set must not appear.
    illicit_titles = [
        c.title
        for c in catalog_by_id.values()
        if c.title not in allowed_titles and _mentions_phrase(narrative_text, c.title)
    ]
    illicit_ids = [
        c.id
        for c in catalog_by_id.values()
        if c.id not in allowed_ids and _mentions_phrase(narrative_text, c.id)
    ]
    if illicit_titles or illicit_ids:
        messages.append(
            "Narrative references catalog items outside the ranked output: "
            f"titles={illicit_titles}, ids={illicit_ids}"
        )
        return False, messages

    if errors:
        # Soft signal: branch warnings exist but core constraints passed.
        messages.append(f"Passed with branch warnings: {errors}")

    messages.append("Validation passed.")
    return True, messages


def _mentions_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive whole-phrase / token-friendly mention check."""
    if not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    return pattern.search(text) is not None
