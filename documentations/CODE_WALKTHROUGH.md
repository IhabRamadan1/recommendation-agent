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
- On missing file / bad JSON / non-list / empty list → `catalog_ok=False`, empty list, error strings — **no exception escapes**.
- Valid entries become `CatalogItem` dumps; invalid rows are skipped with an error note.

**Business:** Bad reference data must not crash the product; it must fail closed later at validation.  
**Technical:** Failures are data, not stack traces, so the graph can continue to a deterministic gate.

#### `rank_items`

- Skips if `catalog_ok` is false.
- Scores each item: `0.7 * tag_overlap_density + 0.3 * score_hint`.
- Returns top 5 as `RankedItem`s; empty result → `rank_failed=True`.

**Business:** Ranking is constrained to the catalog — never invents careers.  
**Technical:** Deterministic scoring makes tests stable and avoids paying for an LLM on the rank branch (half-day budget + reliability). Overlap density prevents long tag lists from dominating unfairly.

#### `write_narrative`

- Needs a ranked set. Because rank and narrative run **in parallel**, this node **recomputes the same deterministic rank** when `ranked_items` is not yet in state.
- Tries LLM if `OPENAI_API_KEY` is set; otherwise uses a template that only inserts ranked titles.
- On failure → `narrative_failed=True`, empty narrative.

**Business:** Narrative must not advertise items that were not recommended.  
**Technical:** Parallelism requires each branch to be self-sufficient; duplicating deterministic rank is cheaper and safer than adding a barrier before narrative.

#### `merge_branches`

- Returns `{}`. Fan-in already merged keys into state; this node is an explicit graph stage for clarity and future enrichment.

#### `validate_output`

- Calls `validators.validate_recommendation_state` and stores `validation_passed` + messages.

### `validators.py`

**What:** Pure Python gate — no LLM.

Checks:

1. `catalog_ok` and non-empty schema-valid catalog.
2. Rank branch produced a usable list; every ranked id/title matches the catalog.
3. Narrative is non-empty and does **not** mention titles/ids from the catalog that are outside the ranked set (regex word-boundary style).

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

**What:** Dict + `asyncio.Lock` + TTL timestamps.

**Business:** Idempotent retries must return the same answer (payment-like semantics for agent side effects).  
**Technical:** Fixes Bug #3 — no bare shared dict across `await` points. `get` / `set` always take the lock; expired entries are deleted on read.

### `tools.py`

**What:** Three real functions + allowlist registry:

| Tool | Purpose |
|------|---------|
| `lookup_catalog_item` | Fetch one catalog row by id |
| `score_profile_fit` | Deterministic rank for a profile |
| `summarize_top_items` | Short summary of ranked rows |

`execute_tool(name, arguments, catalog_path=...)`:

1. Reject unknown names.
2. Validate args with the tool’s Pydantic model.
3. Dispatch with an explicit `if name == ...` chain (not `getattr`).

**Business:** The model may *ask* to call something dangerous; the service simply refuses.  
**Technical:** Allowlist + schema validation is the standard tool-calling safety pattern; avoiding `getattr`/`eval` removes a whole RCE class.

`tool_openai_schemas()` exposes JSON schemas for `bind_tools`.

### `app.py`

**What:** FastAPI app with:

| Route | Behavior |
|-------|----------|
| `GET /health` | Liveness |
| `POST /recommend` | Runs Part A graph via `asyncio.to_thread` (Bug #2 fix) |
| `POST /agent/invoke` | Allowlisted tool call; optional `Idempotency-Key` header |

`force_tool` / `force_args` skip the LLM for offline demos and tests while still going through `execute_tool`.

`_select_tool_with_llm` binds only allowlisted schemas, then **re-checks** the returned name ∈ `TOOL_SPECS` before execution.

Per-request `try/except` in the idempotent compute path converts unexpected errors into a structured `ok=False` response (fault isolation).

**Business:** HTTP is how other services will call recommendations and tools.  
**Technical:** Correct package imports fix Bug #1; thread offload / async LLM fix Bug #2; cache fixes Bug #3.

---

## Data files

| File | Purpose |
|------|---------|
| `data/catalog.example.json` | Happy-path education/career catalog |
| `data/catalog.empty.json` | Empty list → fail closed |
| `data/catalog.malformed.json` | Object instead of list → fail closed |
| `data/sample_profiles.json` | Mock trait profiles for demos |

---

## Tests (what they prove)

| File | Proves |
|------|--------|
| `test_graph_catalog.py` | Empty/malformed catalog handling |
| `test_graph_flow.py` | Happy path, catalog constraint, failed branches |
| `test_validators.py` | Narrative cannot cite non-ranked catalog items |
| `test_cache.py` | Async cache get/set / concurrency smoke |
| `test_tools.py` | Allowlist reject + real tool success paths |
| `test_api.py` | Health, recommend, idempotent replay, isolation |
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
