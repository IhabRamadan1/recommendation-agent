"""Pure deterministic ranking shared by graph nodes and tools.

Business: one scoring algorithm so parallel narrative and rank stay consistent.
Technical: no GraphState / I/O — callers adapt state and call this helper.
"""

from __future__ import annotations

from recommendation_graph.state import CatalogItem, RankedItem, UserProfile


def compute_ranked_items(
    profile: UserProfile,
    catalog: list[CatalogItem],
    limit: int = 5,
) -> list[RankedItem]:
    """Score catalog items by trait/tag overlap + score_hint; return top-N."""
    traits = profile.trait_tokens
    scored: list[RankedItem] = []
    for item in catalog:
        matched = sorted(traits & set(item.tags))
        overlap = len(matched)
        # Blend overlap density with publisher hint so weak matches still order stably.
        overlap_score = overlap / max(len(item.tags), 1)
        score = round(0.7 * overlap_score + 0.3 * item.score_hint, 4)
        if overlap == 0 and item.score_hint < 0.5:
            continue
        scored.append(
            RankedItem(
                id=item.id,
                title=item.title,
                score=score,
                matched_tags=matched,
            )
        )

    scored.sort(key=lambda r: (-r.score, r.title))
    return scored[:limit]
