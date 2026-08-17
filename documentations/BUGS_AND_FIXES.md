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

**Fix:** `AsyncTTLCache` wraps the store with `asyncio.Lock` for get/set (see `src/agentic_service/cache.py`). Idempotency keys are stored only through that API.

---

## After the fixes — tool-calling safety

Even with a healthy async shell, LLM tool selection is still untrusted:

1. Model returns a tool name + arguments.
2. Service checks `name in TOOL_SPECS` (allowlist).
3. Arguments are validated with Pydantic models.
4. Only then is a real Python function invoked — **no** `eval`, **no** blind `getattr`.

Per-request `try/except` isolates failures; `Idempotency-Key` replays cached responses on retries.
