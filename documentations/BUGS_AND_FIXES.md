# Part B bugs and fixes

The service was designed around three **layered** failures that commonly appear in async agentic APIs. The tree ships the **fixed** code; this document records what was wrong and why each fix matters.

---

## Bug 1 — Startup import error

**Symptom:** `uvicorn agentic_service.app:app` fails at import time (module not found / wrong relative import).

**Broken pattern:**

```python
# Wrong: treats the package as a script or uses a non-existent module path
from cache import AsyncTTLCache
from tools import execute_tool
```

**Why it hurts:** The API never comes up. In layered debugging this often masks the other two bugs until imports are fixed.

**Fix:** Use package-absolute imports that match the `src/` layout:

```python
from agentic_service.cache import AsyncTTLCache
from agentic_service.tools import execute_tool
from config.settings import get_settings
```

Run with:

```powershell
uv run uvicorn agentic_service.app:app --reload
```

---

## Bug 2 — Synchronous blocking call on an async request path

**Symptom:** Under concurrent load, latency spikes and the event loop stalls even though handlers are `async def`.

**Broken pattern:**

```python
@app.post("/recommend")
async def recommend(body: RecommendRequest):
    time.sleep(0.5)  # or requests.get(...) — blocks the event loop
    return run_recommendation(profile, path)  # heavy sync CPU/IO on the loop
```

**Why it hurts:** Asyncio concurrency is cooperative. One blocking call freezes **all** in-flight requests on that worker.

**Fix:**

- Run sync graph work in a thread: `await asyncio.to_thread(run_recommendation, ...)`.
- Prefer async LLM clients (`ainvoke`) or `asyncio.to_thread` for sync SDK calls.
- Never use `time.sleep` / sync `requests` inside `async def` handlers.

---

## Bug 3 — Shared mutable dict used as an async-unsafe cache

**Symptom:** Intermittent lost updates, corrupted entries, or inconsistent idempotent replays under concurrency.

**Broken pattern:**

```python
_CACHE: dict[str, Any] = {}

async def agent_invoke(...):
    if key in _CACHE:          # race: check
        return _CACHE[key]
    result = await compute()   # await yields; another task interleaves
    _CACHE[key] = result       # race: write
```

**Why it hurts:** Between `await` points, another coroutine can read/write the same dict. Idempotency guarantees collapse.

**Fix:** `AsyncTTLCache` uses a store lock for get/set **and** a **per-key** lock for single-flight `get_or_compute` (see `src/agentic_service/cache.py`). Different keys still run concurrently; the same key computes at most once.

Idempotency is further bound to a **request fingerprint** (canonical JSON hash of the payload). Same key + same payload → replay; same key + different payload → HTTP **409 Conflict**.

---

## After the fixes — tool-calling safety

Even with a healthy async shell, LLM tool selection is still untrusted:

1. Model returns tool call(s); the service requires **exactly one** call.
2. Service checks `name in TOOL_SPECS` (allowlist with explicit handlers).
3. Arguments are validated with Pydantic models (`extra="forbid"`).
4. Only then is the allowlisted handler invoked — **no** `eval`, **no** blind `getattr`.

Per-request `try/except` isolates failures; `Idempotency-Key` + fingerprint + per-key single-flight make retries safe.
