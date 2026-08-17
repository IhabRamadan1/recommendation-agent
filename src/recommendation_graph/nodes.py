"""Graph nodes: catalog load, rank, narrative, merge, validate.

Business: produce catalog-constrained rankings and a narrative that only cites them.
Technical: nodes return partial state dicts; LangGraph merges them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recommendation_graph.state import CatalogItem, RankedItem, UserProfile
from recommendation_graph.validators import validate_recommendation_state


def load_catalog(state: dict[str, Any]) -> dict[str, Any]:
    """Load and schema-validate the reference catalog. Never crash on bad files."""
    path = Path(state.get("catalog_path") or "")
    errors: list[str] = []

    if not path.is_file():
        return {
            "catalog": [],
            "catalog_ok": False,
            "errors": [f"Catalog file not found: {path}"],
        }

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "catalog": [],
            "catalog_ok": False,
            "errors": [f"Catalog unreadable or invalid JSON: {exc}"],
        }

    if not isinstance(raw, list):
        return {
            "catalog": [],
            "catalog_ok": False,
            "errors": ["Catalog must be a JSON list of items."],
        }

    if len(raw) == 0:
        return {
            "catalog": [],
            "catalog_ok": False,
            "errors": ["Catalog loaded empty."],
        }

    catalog: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        try:
            item = CatalogItem.model_validate(entry)
            catalog.append(item.model_dump())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Item at index {idx} invalid: {exc}")

    if not catalog:
        return {
            "catalog": [],
            "catalog_ok": False,
            "errors": errors or ["No valid catalog items after schema validation."],
        }

    result: dict[str, Any] = {"catalog": catalog, "catalog_ok": True}
    if errors:
        result["errors"] = errors
    return result


def rank_items(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic ranking from trait/tag overlap + score_hint. Catalog-constrained."""
    if not state.get("catalog_ok"):
        return {
            "ranked_items": [],
            "rank_failed": True,
            "errors": ["Rank skipped: catalog not ok."],
        }

    try:
        profile = UserProfile.model_validate(state.get("profile") or {})
        catalog = [CatalogItem.model_validate(c) for c in state.get("catalog") or []]
    except Exception as exc:  # noqa: BLE001
        return {
            "ranked_items": [],
            "rank_failed": True,
            "errors": [f"Rank branch failed: {exc}"],
        }

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
    top = scored[:5]

    if not top:
        return {
            "ranked_items": [],
            "rank_failed": True,
            "errors": ["Rank branch produced an empty list for this profile."],
        }

    return {
        "ranked_items": [r.model_dump() for r in top],
        "rank_failed": False,
    }


def write_narrative(state: dict[str, Any]) -> dict[str, Any]:
    """Short narrative that may only reference ranked items (LLM or template fallback)."""
    if not state.get("catalog_ok"):
        return {
            "narrative": "",
            "narrative_failed": True,
            "errors": ["Narrative skipped: catalog not ok."],
        }

    ranked_raw = state.get("ranked_items") or []
    # Parallel fan-out: narrative may run before merge sees rank output.
    # Prefer ranked_items if present; otherwise recompute rank locally from catalog.
    if not ranked_raw and state.get("catalog"):
        rank_partial = rank_items(state)
        ranked_raw = rank_partial.get("ranked_items") or []
        if rank_partial.get("rank_failed"):
            return {
                "narrative": "",
                "narrative_failed": True,
                "errors": ["Narrative skipped: could not obtain ranked items."],
            }

    try:
        profile = UserProfile.model_validate(state.get("profile") or {})
        ranked = [RankedItem.model_validate(r) for r in ranked_raw]
    except Exception as exc:  # noqa: BLE001
        return {
            "narrative": "",
            "narrative_failed": True,
            "errors": [f"Narrative branch failed: {exc}"],
        }

    if not ranked:
        return {
            "narrative": "",
            "narrative_failed": True,
            "errors": ["Narrative skipped: empty ranked list."],
        }

    try:
        narrative = _generate_narrative(profile, ranked)
    except Exception as exc:  # noqa: BLE001
        return {
            "narrative": "",
            "narrative_failed": True,
            "errors": [f"Narrative generation failed: {exc}"],
        }

    return {"narrative": narrative, "narrative_failed": False}


def merge_branches(state: dict[str, Any]) -> dict[str, Any]:
    """Identity merge node: state already holds both branch outputs after fan-in."""
    return {}


def validate_output(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic gate after merge."""
    passed, messages = validate_recommendation_state(
        catalog=state.get("catalog") or [],
        catalog_ok=bool(state.get("catalog_ok")),
        ranked_items=state.get("ranked_items") or [],
        narrative=state.get("narrative") or "",
        rank_failed=bool(state.get("rank_failed")),
        errors=state.get("errors") or [],
    )
    return {
        "validation_passed": passed,
        "validation_messages": messages,
    }


def _generate_narrative(profile: UserProfile, ranked: list[RankedItem]) -> str:
    """Prefer LLM when configured; always fall back to a deterministic template."""
    titles = [r.title for r in ranked]
    try:
        from config.settings import get_settings

        settings = get_settings()
        if settings.has_openai:
            return _llm_narrative(profile, ranked, settings.openai_model, settings.openai_api_key)
    except Exception:
        # Fall through to template — offline / misconfig must not break the graph.
        pass

    top = ", ".join(titles[:3])
    return (
        f"Based on {profile.name}'s interests in "
        f"{', '.join(profile.interests[:4]) or 'general exploration'}, "
        f"the strongest catalog-aligned fits are {top}. "
        f"These options stay within the ranked recommendation set only."
    )


def _llm_narrative(
    profile: UserProfile,
    ranked: list[RankedItem],
    model: str,
    api_key: str,
) -> str:
    from langchain_openai import ChatOpenAI

    allowed = ", ".join(f"{r.title} ({r.id})" for r in ranked)
    prompt = (
        "Write 2 short sentences recommending careers/majors for this student. "
        "You may ONLY mention items from this allowed list (by title): "
        f"{allowed}. Do not invent other programs or careers.\n"
        f"Student: {profile.model_dump_json()}"
    )
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    return str(content).strip()
