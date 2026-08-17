"""Allowlisted LLM-selectable tools. No eval / blind getattr.

Business: the model proposes a tool; the service decides if it is real and safe.
Technical: TOOL_SPECS is the single source of truth (args model + handler).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from recommendation_graph.ranking import compute_ranked_items
from recommendation_graph.state import CatalogItem, UserProfile


class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)


class ScoreFitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: dict[str, Any]
    catalog_path: str | None = None


class SummarizeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    try:
        user = UserProfile.model_validate(profile)
        raw_items = _load_catalog(catalog_path)
        if not raw_items:
            return {"ok": False, "error": "Catalog empty or unreadable."}
        catalog = [CatalogItem.model_validate(c) for c in raw_items]
        top = compute_ranked_items(user, catalog, limit=5)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    if not top:
        return {
            "ok": True,
            "ranked_items": [],
            "rank_failed": True,
        }
    return {
        "ok": True,
        "ranked_items": [r.model_dump() for r in top],
        "rank_failed": False,
    }


def summarize_top_items(ranked_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a short deterministic summary of already-ranked items."""
    titles = [str(r.get("title", r.get("id", "?"))) for r in ranked_items[:5]]
    return {
        "ok": True,
        "summary": "Top recommendations: " + ", ".join(titles),
        "count": len(titles),
    }


ToolHandler = Callable[[BaseModel, Path], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """Allowlist entry: name, description, args schema, and explicit handler."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler


def _handle_lookup(parsed: BaseModel, catalog_path: Path) -> dict[str, Any]:
    assert isinstance(parsed, LookupArgs)
    return lookup_catalog_item(parsed.item_id, catalog_path)


def _handle_score_fit(parsed: BaseModel, catalog_path: Path) -> dict[str, Any]:
    assert isinstance(parsed, ScoreFitArgs)
    path = Path(parsed.catalog_path) if parsed.catalog_path else catalog_path
    return score_profile_fit(parsed.profile, path)


def _handle_summarize(parsed: BaseModel, catalog_path: Path) -> dict[str, Any]:
    assert isinstance(parsed, SummarizeArgs)
    return summarize_top_items(parsed.ranked_items)


# Allowlist: only these names may be executed. Handlers are wired explicitly here.
TOOL_SPECS: dict[str, ToolSpec] = {
    "lookup_catalog_item": ToolSpec(
        name="lookup_catalog_item",
        description="Look up a single catalog item by its id.",
        args_model=LookupArgs,
        handler=_handle_lookup,
    ),
    "score_profile_fit": ToolSpec(
        name="score_profile_fit",
        description="Score and rank catalog items for a user trait profile.",
        args_model=ScoreFitArgs,
        handler=_handle_score_fit,
    ),
    "summarize_top_items": ToolSpec(
        name="summarize_top_items",
        description="Summarize an already ranked list of items.",
        args_model=SummarizeArgs,
        handler=_handle_summarize,
    ),
}


def tool_openai_schemas() -> list[dict[str, Any]]:
    """JSON schemas exposed to the LLM for function-calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_model.model_json_schema(),
            },
        }
        for spec in TOOL_SPECS.values()
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    catalog_path: Path,
) -> dict[str, Any]:
    """Validate tool name against allowlist, validate args, then execute handler."""
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

    return spec.handler(parsed, catalog_path)


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
