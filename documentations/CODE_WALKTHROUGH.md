# Code walkthrough

This document explains **what each module does**, **why it exists (business)**, and **why it is implemented this way (technical)**. Source files stay lightly commented on purpose — deep “why” lives here so the code remains readable.

Related docs:

- [business_objective.md](business_objective.md)
- [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md)
- [../diagrams/workflow_diagram.md](../diagrams/workflow_diagram.md)
- [../diagrams/sequence_diagram.md](../diagrams/sequence_diagram.md)

---

## Repository layout (why this structure)

| Path | Role | Why |
|------|------|-----|
| `src/config/` | Settings | Educatlyy-style `config` package; under `src/` so `uv` installs make `import config` work |
| `src/recommendation_graph/` | Part A LangGraph | Maps to production `app/workflow/` conceptually |
| `src/agentic_service/` | Part B FastAPI | Maps to production `app/api/` conceptually |
| `src/recommendation_agent/` | CLI entry | `uv run recommendation-agent` demo |
| `data/` | Mock catalog + profiles | Reference catalog = untrusted fixture input |
| `documentations/`, `diagrams/` | Docs | Same convention as the multi-agent Educatlyy repo |
| `tests/` | Pytest | Graph + service safety nets |

We intentionally **did not** copy Kafka/Docker from the production repo — out of scope for this half-day task.

---

## `src/config/settings.py`

**What:** Loads `OPENAI_API_KEY`, `OPENAI_MODEL`, `CATALOG_PATH`, `CACHE_TTL_SECONDS` from env / `.env`.

**Business why:** Operators configure secrets and catalog location without code changes.

**Technical why:**

- `pydantic-settings` gives typed config and validation.
- `REPO_ROOT` is derived from `__file__` so paths work regardless of cwd.
- `get_settings()` is cached (`lru_cache`) so the app does not re-parse env on every call.
- `has_openai` lets narrative / tool-calling fall back offline when no key is set.

---

## Part A — `src/recommendation_graph/`

### `state.py`

**What:** Pydantic models (`CatalogItem`, `UserProfile`, `RankedItem`) and `GraphState` TypedDict.

**Business why:** The shared state is the **contract** between ranking, storytelling, and compliance (validation). If the contract is vague, agents invent fields.

**Technical why:**

- Pydantic validates shapes early (tags normalized to lowercase).
- `GraphState` uses LangGraph’s `Annotated[list, reducer]` for `errors` so **parallel branches can both append errors** without overwriting each other.
- `total=False` TypedDict allows gradual filling across nodes.

### `nodes.py`

#### `load_catalog`

- Reads JSON from `catalog_path`.
- On missing file / bad JSON / non-list / empty list / **duplicate ids** → `catalog_ok=False`, empty list, error strings — **no exception escapes**.
- Valid entries become `CatalogItem` dumps; invalid rows are skipped with an error note.

**Business:** Bad reference data must not crash the product; it must fail closed later at validation.  
**Technical:** Failures are data, not stack traces, so the graph can continue to a deterministic gate.

#### `rank_items`

- Skips if `catalog_ok` is false.
- Adapts GraphState → calls **`compute_ranked_items`** (shared pure helper) → returns top 5.
- Empty result → `rank_failed=True`.

**Business:** Ranking is constrained to the catalog — never invents careers.  
**Technical:** Deterministic scoring lives in `ranking.py` so narrative and tools cannot drift from the graph node.

#### `write_narrative`

- Needs a ranked set. Because rank and narrative run **in parallel**, this node calls **`compute_ranked_items`** (not the `rank_items` LangGraph node) when `ranked_items` is not yet in state.
- Tries LLM if `OPENAI_API_KEY` is set; otherwise uses a template that only inserts ranked titles.
- On failure → `narrative_failed=True`, empty narrative.

**Business:** Narrative must not advertise items that were not recommended.  
**Technical:** Parallelism requires each branch to be self-sufficient; the shared helper (not the LangGraph node) keeps scores identical without a barrier before narrative.

### `ranking.py`

**What:** Pure `compute_ranked_items(profile, catalog, limit=5)` — trait/tag overlap + `score_hint`.

Used by `rank_items`, `write_narrative`, and `score_profile_fit` so the algorithm exists in exactly one place.

#### `merge_branches`

- Returns `{}`. Fan-in already merged keys into state; this node is an explicit graph stage for clarity and future enrichment.

#### `validate_output`

- Calls `validators.validate_recommendation_state` and stores `validation_passed` + messages.

### `validators.py`

**What:** Pure Python gate — no LLM.

Checks:

1. `catalog_ok` and non-empty schema-valid catalog with **unique ids**.
2. Rank branch produced a usable list; every ranked id/title matches the catalog.
3. `narrative_failed` is treated as an explicit failure (not only empty string).
4. Narrative is non-empty and does **not** mention titles/ids from the catalog that are outside the ranked set (regex word-boundary style).

**Business:** This is the compliance checkpoint before results are trusted.  
**Technical:** Keeping validation non-LLM means an adversarial or buggy model cannot grade its own homework.

### `graph.py`

**What:** Builds `StateGraph(GraphState)`:

`START → load_catalog → (rank_items ∥ write_narrative) → merge → validate → END`

Exports `build_graph()` and `run_recommendation(profile, catalog_path)`.

**Business:** One callable produces a full, gated recommendation package.  
**Technical:** Parallel edges after `load_catalog` are how LangGraph fans out; both edges into `merge` create a join barrier.

### Package `__init__.py`

Re-exports `build_graph` and `run_recommendation` for clean imports.

---

## CLI — `src/recommendation_agent/__init__.py`

**What:** Loads the first sample profile + configured catalog, runs the graph, prints JSON.

**Business:** A quick demo for stakeholders without starting HTTP.  
**Technical:** Wired as the `recommendation-agent` script in `pyproject.toml`.

---

## Part B — `src/agentic_service/`

### `cache.py` — `AsyncTTLCache`

**What:** Dict + store lock + **per-key** locks + TTL timestamps + `get_or_compute`.

**Business:** Idempotent retries must return the same answer (payment-like semantics for agent side effects).  
**Technical:** Fixes Bug #3 — no bare shared dict across `await` points. Same idempotency key is single-flight; different keys do not share one global compute lock.

### `tools.py`

**What:** Three real functions + allowlist registry (`TOOL_SPECS`) as the **single source of truth**:

| Tool | Purpose |
|------|---------|
| `lookup_catalog_item` | Fetch one catalog row by id |
| `score_profile_fit` | Deterministic rank via `compute_ranked_items` |
| `summarize_top_items` | Short summary of ranked rows |

`execute_tool(name, arguments, catalog_path=...)`:

1. Reject unknown names (must be in `TOOL_SPECS`).
2. Validate args with the tool’s Pydantic model (`extra="forbid"`).
3. Dispatch via the explicit `handler` on that allowlist entry (not `getattr` / `eval`).

**Business:** The model may *ask* to call something dangerous; the service simply refuses.  
**Technical:** Allowlist + schema validation is the standard tool-calling safety pattern.

`tool_openai_schemas()` exposes JSON schemas for `bind_tools`.

### `app.py`

**What:** FastAPI app with:

| Route | Behavior |
|-------|----------|
| `GET /health` | Liveness |
| `POST /recommend` | Runs Part A graph via `asyncio.to_thread` (Bug #2 fix) |
| `POST /agent/invoke` | Allowlisted tool call; fingerprint-bound `Idempotency-Key` |

Idempotency flow:

```text
Idempotency-Key → request fingerprint → per-key lock → double-check cache → execute once → cache
```

Same key + different payload → **409 Conflict**. `parse_exactly_one_tool_call` rejects zero or multiple LLM tool calls.

`force_tool` / `force_args` skip the LLM for offline demos and tests while still going through `execute_tool`.

Per-request `try/except` isolates unexpected errors (`HTTPException` is re-raised).

**Business:** HTTP is how other services will call recommendations and tools.  
**Technical:** Correct package imports fix Bug #1; thread offload / async LLM fix Bug #2; cache fixes Bug #3.

---

## Data files

| File | Purpose |
|------|---------|
| `data/catalog.example.json` | Happy-path education/career catalog |
| `data/catalog.empty.json` | Empty list → fail closed |
| `data/catalog.malformed.json` | Object instead of list → fail closed |
| `data/catalog.duplicate_ids.json` | Duplicate ids → fail closed |
| `data/sample_profiles.json` | Mock trait profiles for demos |

---

## Tests (what they prove)

| File | Proves |
|------|--------|
| `test_graph_catalog.py` | Empty/malformed/duplicate catalog handling |
| `test_graph_flow.py` | Happy path, catalog constraint, failed branches |
| `test_ranking.py` | Shared `compute_ranked_items` used by nodes |
| `test_validators.py` | Narrative constraints, `narrative_failed`, duplicate ids |
| `test_cache.py` | Per-key single-flight; different keys concurrent |
| `test_tools.py` | Allowlist, `extra=forbid`, shared ranking |
| `test_api.py` | Idempotent replay, 409 fingerprint mismatch, one tool call |
| `test_setup.py` | Imports still resolve |

---

## Implementation plan (why this order)

1. **Config + fixtures** — everything else needs paths and mock data.
2. **State → validators → nodes → graph** — validator first clarifies the safety contract nodes must satisfy.
3. **Part A tests** — lock graph behavior before HTTP complexity.
4. **Cache → tools → app** — fix async foundations before wiring LLM selection.
5. **Part B tests** — prove idempotency and allowlist under HTTP.
6. **Docs/diagrams** — capture business + debugging narrative for reviewers.

Ranking is deterministic and narrative is LLM-optional so the exercise evaluates **orchestration and validation**, not prompt tuning. Tool-calling still demonstrates the allowlist pattern with a real (or forced) selection path.
