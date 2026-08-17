# Recommendation Agent

Task 2 — **Recommendation Graph + Harden a Broken Agentic Service**

A small, end-to-end agentic recommendation system that:

1. **Part A** — Runs a LangGraph pipeline: load a reference catalog → rank items and write a narrative **in parallel** → merge → **deterministic (non-LLM) validation**
2. **Part B** — Exposes a hardened async FastAPI service with allowlisted LLM tool-calling, per-request fault isolation, and idempotent retries

**Core safety idea:** treat both the reference catalog and the LLM’s tool/narrative choices as **untrusted** — validate before use.

---

## Stack

| Layer | Tech |
|-------|------|
| Orchestration | LangGraph `StateGraph` |
| LLM / tools | LangChain + OpenAI tool-calling |
| Models / config | Pydantic, pydantic-settings |
| API | FastAPI + asyncio + uvicorn |
| Packaging / tests | `uv`, pytest, pytest-asyncio |

Python **3.13+** (see `.python-version`).

---

## Quick start

```powershell
cd E:\recommendation-agent
uv sync
copy .env.example .env
# Optional: set OPENAI_API_KEY in .env for LLM narrative + live tool selection
```

Verify:

```powershell
uv run pytest -q
```

---

## Part A — Recommendation graph (CLI)

Runs the graph on the first sample profile (`Alex`) and prints JSON:

```powershell
uv run recommendation-agent
```

### Use another profile (Sam / Jordan)

```powershell
uv run python -c "import json; from pathlib import Path; from recommendation_graph import run_recommendation; profiles=json.loads(Path('data/sample_profiles.json').read_text()); print(json.dumps(run_recommendation(profiles[1], 'data/catalog.example.json'), indent=2))"
```

- `profiles[0]` → Alex (STEM)
- `profiles[1]` → Sam (creative / UX)
- `profiles[2]` → Jordan (business)

### Force a validation failure

```powershell
uv run python -c "from recommendation_graph import run_recommendation; r=run_recommendation({'name':'Sam','interests':['design'],'goals':['lead'],'level':'graduate'}, 'data/catalog.empty.json'); print(r['validation_passed'], r['validation_messages'], r['errors'])"
```

Also try `data/catalog.malformed.json`.

### Python API

```python
from recommendation_graph import run_recommendation

result = run_recommendation(
    {
        "name": "Alex",
        "interests": ["python", "data", "stem"],
        "goals": ["analyze data"],
        "level": "undergraduate",
    },
    "data/catalog.example.json",
)

assert result["validation_passed"] is True
print(result["ranked_items"])
print(result["narrative"])
```

### Part A flow

```text
profile + catalog
       │
       ▼
 load_catalog ──► (rank_items ∥ write_narrative) ──► merge ──► validate ──► result
```

- Ranked items are **catalog-constrained**
- Narrative may only mention items in the **ranked** set
- Empty / malformed catalogs **fail closed**

Diagram: [diagrams/workflow_diagram.md](diagrams/workflow_diagram.md)

---

## Part B — Hardened agentic API

```powershell
uv run uvicorn agentic_service.app:app --reload --host 127.0.0.1 --port 8000
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/docs | Swagger UI (try requests here) |
| http://127.0.0.1:8000/health | Liveness |
| http://127.0.0.1:8000/recommend | Run Part A graph over HTTP |
| http://127.0.0.1:8000/agent/invoke | Allowlisted tool calling |

### `POST /recommend`

```json
{
  "profile": {
    "id": "profile-stem-builder",
    "name": "Alex",
    "interests": ["coding", "python", "data", "stem"],
    "goals": ["build products", "analyze data"],
    "level": "undergraduate"
  }
}
```

Optional: `"catalog_path": "data/catalog.empty.json"` to demo fail-closed behavior.

### `POST /agent/invoke` (offline / no API key)

```json
{
  "message": "lookup software engineer",
  "force_tool": "lookup_catalog_item",
  "force_args": { "item_id": "career-software-engineer" }
}
```

Add header `Idempotency-Key: demo-1`, execute twice — second response should set `"idempotent_replay": true`.

Same key with a **different body** returns **HTTP 409 Conflict** (fingerprint mismatch).

Concurrent retries with the same key are **single-flight** (underlying work runs once).

### `POST /agent/invoke` (live LLM tool selection)

1. Set `OPENAI_API_KEY` in `.env`
2. Call **without** `force_tool` — the model picks one of:

   - `lookup_catalog_item`
   - `score_profile_fit`
   - `summarize_top_items`

3. The service **re-checks** the tool name against the allowlist before executing (no `eval` / blind `getattr`)

Diagram: [diagrams/sequence_diagram.md](diagrams/sequence_diagram.md)

---

## Project layout

```text
recommendation-agent/
├── data/
│   ├── catalog.example.json         # happy-path reference catalog
│   ├── catalog.empty.json           # fail-closed fixture
│   ├── catalog.malformed.json       # fail-closed fixture
│   ├── catalog.duplicate_ids.json   # fail-closed fixture
│   └── sample_profiles.json         # Alex / Sam / Jordan
├── diagrams/                        # mermaid workflow + sequence
├── documentations/                  # business, bugs, code walkthrough
├── src/
│   ├── config/                      # settings from .env
│   ├── recommendation_agent/        # CLI entrypoint
│   ├── recommendation_graph/        # Part A (state, ranking, nodes, graph, validators)
│   └── agentic_service/             # Part B (app, cache, tools)
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Configuration

Copy `.env.example` → `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | _(empty)_ | Enables LLM narrative + live tool selection |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `CATALOG_PATH` | `data/catalog.example.json` | Default catalog for API |
| `CACHE_TTL_SECONDS` | `300` | Idempotency cache TTL |

Without an API key, Part A still works (template narrative) and Part B works via `force_tool`.

---

## Tests

```powershell
uv run pytest -q
```

Coverage includes: empty/malformed/duplicate catalog, parallel graph + failed branches, shared ranking helper, validator rules (`narrative_failed`), per-key single-flight cache, strict tool args, exactly-one tool call, fingerprint 409 conflicts, and fault isolation.

---

## Documentation

| Doc | What you’ll learn |
|-----|-------------------|
| [documentations/business_objective.md](documentations/business_objective.md) | Why catalog + tool choice are untrusted |
| [documentations/BUGS_AND_FIXES.md](documentations/BUGS_AND_FIXES.md) | Three layered Part B bugs (before → after) |
| [documentations/CODE_WALKTHROUGH.md](documentations/CODE_WALKTHROUGH.md) | File-by-file what / why (business + technical) |
| [diagrams/workflow_diagram.md](diagrams/workflow_diagram.md) | Part A graph |
| [diagrams/sequence_diagram.md](diagrams/sequence_diagram.md) | Part B request / tool flow |

---

## What this task evaluates

| Skill | Where it shows up |
|-------|-------------------|
| LangGraph orchestration | Parallel rank + narrative, merge, validate |
| Deterministic validation | `validators.py` — no LLM in the gate; duplicate ids / `narrative_failed` |
| Shared domain ranking | `ranking.compute_ranked_items` used by graph + tools |
| Async debugging | Import path, non-blocking handlers, per-key cache locks |
| Safe tool-calling | Allowlist handlers + `extra="forbid"` args; exactly one tool call |
| Isolation + idempotency | Per-request errors; fingerprint + single-flight `Idempotency-Key` |

---

## License / status

Internal Task 2 exercise — Part A and Part B implemented with tests and documentation.
