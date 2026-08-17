"""Hardened async FastAPI agentic service.

Fixes applied (see documentations/BUGS_AND_FIXES.md):
1. Correct package imports so startup succeeds.
2. No sync blocking on the async request path.
3. Async-safe TTL cache instead of a bare shared dict.

Adds allowlisted LLM tool-calling with per-request isolation and
fingerprint-bound per-key single-flight idempotency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agentic_service.cache import AsyncTTLCache
from agentic_service.tools import TOOL_SPECS, execute_tool, tool_openai_schemas
from config.settings import REPO_ROOT, get_settings

app = FastAPI(title="Hardened Agentic Service", version="0.1.0")
_settings = get_settings()
_cache = AsyncTTLCache(ttl_seconds=_settings.cache_ttl_seconds)


def _catalog_path() -> Path:
    path = Path(_settings.catalog_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


class RecommendRequest(BaseModel):
    profile: dict[str, Any]
    catalog_path: str | None = None


class AgentInvokeRequest(BaseModel):
    """Natural-language instruction; model may pick one allowlisted tool."""

    message: str = Field(min_length=1)
    # Optional forced tool path for offline / tests (skips LLM).
    force_tool: str | None = None
    force_args: dict[str, Any] | None = None


class AgentInvokeResponse(BaseModel):
    ok: bool
    tool: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    idempotent_replay: bool = False


def request_fingerprint(body: AgentInvokeRequest) -> str:
    """Deterministic hash of the semantically relevant request payload."""
    payload = {
        "message": body.message,
        "force_tool": body.force_tool,
        "force_args": body.force_args,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_exactly_one_tool_call(response: Any) -> tuple[str, dict[str, Any]]:
    """Require exactly one tool call from LangChain / OpenAI response shapes.

    Handles both ``response.tool_calls`` and ``additional_kwargs["tool_calls"]``.
    """
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    source = "tool_calls"

    if not tool_calls:
        additional = getattr(response, "additional_kwargs", {}) or {}
        tool_calls = list(additional.get("tool_calls") or [])
        source = "additional_kwargs"

    if len(tool_calls) == 0:
        raise HTTPException(status_code=400, detail="Model did not select a tool.")
    if len(tool_calls) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly one tool call, got {len(tool_calls)}.",
        )

    first = tool_calls[0]
    if source == "additional_kwargs" and isinstance(first, dict):
        name = first.get("function", {}).get("name") or first.get("name")
        arg_str = first.get("function", {}).get("arguments") or "{}"
        args = json.loads(arg_str) if isinstance(arg_str, str) else dict(arg_str or {})
    elif isinstance(first, dict):
        name = first.get("name")
        args = dict(first.get("args") or {})
    else:
        name = getattr(first, "name", None)
        args = dict(getattr(first, "args", None) or {})

    if not name or name not in TOOL_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Model selected non-allowlisted tool {name!r}",
        )
    return str(name), args


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
async def recommend(body: RecommendRequest) -> dict[str, Any]:
    """Run the Part A recommendation graph without blocking the event loop."""
    from recommendation_graph import run_recommendation

    catalog = Path(body.catalog_path) if body.catalog_path else _catalog_path()
    if not catalog.is_absolute():
        catalog = REPO_ROOT / catalog

    # Bug #2 fix: run sync graph in a worker thread instead of blocking the loop.
    result = await asyncio.to_thread(run_recommendation, body.profile, catalog)
    return {
        "validation_passed": result.get("validation_passed"),
        "ranked_items": result.get("ranked_items"),
        "narrative": result.get("narrative"),
        "errors": result.get("errors"),
        "validation_messages": result.get("validation_messages"),
    }


@app.post("/agent/invoke", response_model=AgentInvokeResponse)
async def agent_invoke(
    body: AgentInvokeRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AgentInvokeResponse:
    """LLM tool-selection (or forced tool) with allowlist validation + idempotency."""
    fingerprint = request_fingerprint(body)

    async def _compute() -> AgentInvokeResponse:
        try:
            return await _run_agent(body)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — isolate failure to this request
            return AgentInvokeResponse(ok=False, error=f"Request failed: {exc}")

    if not idempotency_key:
        return await _compute()

    async def _factory() -> dict[str, Any]:
        result = await _compute()
        return {
            "request_hash": fingerprint,
            "response": result.model_dump(),
        }

    # Fast path: cache hit without taking the per-key lock.
    cached = await _cache.get(idempotency_key)
    if cached is not None:
        return _replay_or_conflict(cached, fingerprint)

    entry, computed = await _cache.get_or_compute(idempotency_key, _factory)
    # get_or_compute may return an entry produced by another waiter; still check hash.
    if not computed:
        return _replay_or_conflict(entry, fingerprint)

    # This caller computed: verify we stored our fingerprint (should match).
    if entry.get("request_hash") != fingerprint:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used with a different request payload.",
        )
    return AgentInvokeResponse.model_validate(entry["response"])


def _replay_or_conflict(entry: Any, fingerprint: str) -> AgentInvokeResponse:
    if not isinstance(entry, dict) or "request_hash" not in entry or "response" not in entry:
        raise HTTPException(status_code=500, detail="Corrupt idempotency cache entry.")
    if entry["request_hash"] != fingerprint:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used with a different request payload.",
        )
    replay = AgentInvokeResponse.model_validate(entry["response"])
    replay.idempotent_replay = True
    return replay


async def _run_agent(body: AgentInvokeRequest) -> AgentInvokeResponse:
    catalog = _catalog_path()

    if body.force_tool:
        # Deterministic path for tests / offline demos.
        result = execute_tool(
            body.force_tool,
            body.force_args or {},
            catalog_path=catalog,
        )
        ok = bool(result.get("ok"))
        return AgentInvokeResponse(
            ok=ok,
            tool=body.force_tool,
            result=result,
            error=None if ok else str(result.get("error")),
        )

    tool_name, tool_args = await _select_tool_with_llm(body.message)
    result = execute_tool(tool_name, tool_args, catalog_path=catalog)
    ok = bool(result.get("ok"))
    return AgentInvokeResponse(
        ok=ok,
        tool=tool_name,
        result=result,
        error=None if ok else str(result.get("error")),
    )


async def _select_tool_with_llm(message: str) -> tuple[str, dict[str, Any]]:
    """Ask the model to pick one allowlisted tool; validate the name before return."""
    settings = get_settings()
    if not settings.has_openai:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_API_KEY not set. Pass force_tool/force_args for offline use, "
                f"or configure a key. Allowed tools: {sorted(TOOL_SPECS)}"
            ),
        )

    # Bug #2 fix: use async LLM client; never call blocking HTTP on the event loop.
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tool_openai_schemas())

    if hasattr(llm_with_tools, "ainvoke"):
        response = await llm_with_tools.ainvoke(message)
    else:
        response = await asyncio.to_thread(llm_with_tools.invoke, message)

    return parse_exactly_one_tool_call(response)
