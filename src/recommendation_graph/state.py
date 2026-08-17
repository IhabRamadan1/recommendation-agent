"""Graph state and domain models (Pydantic).

Business: shared state is the contract between ranking, narrative, and validation.
Technical: LangGraph merges node updates into this TypedDict-shaped state.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


def _merge_errors(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer so parallel branches can append errors without clobbering each other."""
    return list(left or []) + list(right or [])


class CatalogItem(BaseModel):
    """One reference-catalog entry. Catalog is treated as untrusted input until validated."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    score_hint: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("tags must be a list")
        return [str(t).strip().lower() for t in value if str(t).strip()]


class UserProfile(BaseModel):
    """Small user trait profile used to score catalog fit."""

    id: str = Field(default="anonymous")
    name: str = Field(default="User")
    interests: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    level: str = Field(default="undergraduate")

    @field_validator("interests", "goals", mode="before")
    @classmethod
    def _normalize_traits(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("interests/goals must be lists")
        return [str(t).strip().lower() for t in value if str(t).strip()]

    @property
    def trait_tokens(self) -> set[str]:
        return set(self.interests) | set(self.goals)


class RankedItem(BaseModel):
    """A catalog item plus a computed fit score (always constrained to catalog ids)."""

    id: str
    title: str
    score: float
    matched_tags: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """LangGraph shared state. Parallel nodes write disjoint keys; errors use a reducer."""

    profile: dict[str, Any]
    catalog_path: str
    catalog: list[dict[str, Any]]
    catalog_ok: bool
    ranked_items: list[dict[str, Any]]
    narrative: str
    errors: Annotated[list[str], _merge_errors]
    validation_passed: bool
    validation_messages: list[str]
    rank_failed: bool
    narrative_failed: bool
