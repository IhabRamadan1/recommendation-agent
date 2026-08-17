"""LangGraph StateGraph wiring for the recommendation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from recommendation_graph.nodes import (
    load_catalog,
    merge_branches,
    rank_items,
    validate_output,
    write_narrative,
)
from recommendation_graph.state import GraphState, UserProfile


def build_graph():
    """Compile: load catalog → parallel rank + narrative → merge → validate."""
    graph = StateGraph(GraphState)
    graph.add_node("load_catalog", load_catalog)
    graph.add_node("rank_items", rank_items)
    graph.add_node("write_narrative", write_narrative)
    graph.add_node("merge", merge_branches)
    graph.add_node("validate", validate_output)

    graph.add_edge(START, "load_catalog")
    # Fan-out: both branches run after catalog load (LangGraph runs them in parallel).
    graph.add_edge("load_catalog", "rank_items")
    graph.add_edge("load_catalog", "write_narrative")
    # Fan-in: merge waits for both branches.
    graph.add_edge("rank_items", "merge")
    graph.add_edge("write_narrative", "merge")
    graph.add_edge("merge", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def run_recommendation(
    profile: dict[str, Any] | UserProfile,
    catalog_path: str | Path,
) -> dict[str, Any]:
    """Execute the recommendation graph and return the final state."""
    if isinstance(profile, UserProfile):
        profile_data = profile.model_dump()
    else:
        profile_data = UserProfile.model_validate(profile).model_dump()

    app = build_graph()
    initial: GraphState = {
        "profile": profile_data,
        "catalog_path": str(catalog_path),
        "catalog": [],
        "catalog_ok": False,
        "ranked_items": [],
        "narrative": "",
        "errors": [],
        "validation_passed": False,
        "validation_messages": [],
        "rank_failed": False,
        "narrative_failed": False,
    }
    return app.invoke(initial)
