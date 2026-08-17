"""Allowlisted LLM-selectable tools. No eval / blind getattr.

Business: the model proposes a tool; the service decides if it is real and safe.
Technical: registry maps name → (argument model, callable).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from recommendation_graph.state import CatalogItem, UserProfile
from recommendation_graph.nodes import rank_items


class LookupArgs(BaseModel):
    item_id: str = Field(min_length=1)


class ScoreFitArgs(BaseModel):
    profile: dict[str, Any]
    catalog_path: str | None = None


class SummarizeArgs(BaseModel):
    ranked_items: list[dict[str, Any]] = Field(min_length=1)


def lookup_catalog_item(item_id: str, catalog_path: Path) -> dict[str, Any]:
    """Return one catalog item by id, or an error payload if missing."""
    items = _load_catalog(catalog_path)
    for raw in items:
        item = CatalogItem.model_validate(raw)
        if item.id == item_id:
            return {"ok": True, "item": item.model_dump()}
    return {"ok": False, "error": f"Unknown item_id: {item_id}"}


def score_profile_fit(profile: dict[str, Any], catalog_path: Path) -> dict[str, Any]:
    """Run deterministic ranking for a profile against the reference catalog."""
    UserProfile.model_validate(profile)  # fail fast on bad shape
    state = {
        "profile": profile,
        "catalog_path": str(catalog_path),
        "catalog": _load_catalog(catalog_path),
        "catalog_ok": True,
    }
    if not state["catalog"]:
        return {"ok": False, "error": "Catalog empty or unreadable."}
    # Re-validate catalog_ok after load
    try:
        state["catalog"] = [CatalogItem.model_validate(c).model_dump() for c in state["catalog"]]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    ranked = rank_items(state)
    return {"ok": True, **ranked}


def summarize_top_items(ranked_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a short deterministic summary of already-ranked items."""
    titles = [str(r.get("title", r.get("id", "?"))) for r in ranked_items[:5]]
    return {
        "ok": True,
        "summary": "Top recommendations: " + ", ".join(titles),
        "count": len(titles),
    }


ToolHandler = Callable[..., dict[str, Any]]


class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}


# Allowlist: only these names may be executed.
TOOL_SPECS: dict[str, ToolSpec] = {
    "lookup_catalog_item": ToolSpec(
        name="lookup_catalog_item",
        description="Look up a single catalog item by its id.",
        args_model=LookupArgs,
    ),
    "score_profile_fit": ToolSpec(
        name="score_profile_fit",
        description="Score and rank catalog items for a user trait profile.",
        args_model=ScoreFitArgs,
    ),
    "summarize_top_items": ToolSpec(
        name="summarize_top_items",
        description="Summarize an already ranked list of items.",
        args_model=SummarizeArgs,
    ),
}


def tool_openai_schemas() -> list[dict[str, Any]]:
    """JSON schemas exposed to the LLM for function-calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_catalog_item",
                "description": TOOL_SPECS["lookup_catalog_item"].description,
                "parameters": LookupArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "score_profile_fit",
                "description": TOOL_SPECS["score_profile_fit"].description,
                "parameters": ScoreFitArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_top_items",
                "description": TOOL_SPECS["summarize_top_items"].description,
                "parameters": SummarizeArgs.model_json_schema(),
            },
        },
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    catalog_path: Path,
) -> dict[str, Any]:
    """Validate tool name against allowlist, validate args, then execute."""
    if name not in TOOL_SPECS:
        return {
            "ok": False,
            "error": f"Tool {name!r} is not in the allowlist.",
            "allowed": sorted(TOOL_SPECS),
        }

    spec = TOOL_SPECS[name]
    try:
        parsed = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}

    if name == "lookup_catalog_item":
        assert isinstance(parsed, LookupArgs)
        return lookup_catalog_item(parsed.item_id, catalog_path)
    if name == "score_profile_fit":
        assert isinstance(parsed, ScoreFitArgs)
        path = Path(parsed.catalog_path) if parsed.catalog_path else catalog_path
        return score_profile_fit(parsed.profile, path)
    if name == "summarize_top_items":
        assert isinstance(parsed, SummarizeArgs)
        return summarize_top_items(parsed.ranked_items)

    # Unreachable if TOOL_SPECS and branches stay in sync — still fail closed.
    return {"ok": False, "error": f"No handler wired for allowlisted tool {name!r}."}


def _load_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    if not catalog_path.is_file():
        return []
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return raw
