# Sequence diagram (Part B)

Hardened agent invoke path with fingerprint-bound, per-key single-flight idempotency and allowlisted tools.

```text
Idempotency-Key
     ↓
request fingerprint (canonical JSON hash)
     ↓
per-key single-flight lock
     ↓
double-check cache
     ↓
execute once
     ↓
cache {request_hash, response}
```

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI_async
  participant Cache as AsyncTTLCache
  participant LLM as OpenAI_tools
  participant Tools as allowlisted_tools

  Client->>API: POST /agent/invoke Idempotency-Key
  API->>API: fingerprint payload
  API->>Cache: get key
  alt cache_hit_same_hash
    Cache-->>API: prior_response
    API-->>Client: replay JSON
  else cache_hit_different_hash
    API-->>Client: 409 Conflict
  else miss
    API->>Cache: acquire per_key lock
    API->>Cache: double_check
    alt won_single_flight
      API->>LLM: bind_tools allowlist schemas
      LLM-->>API: exactly_one tool call
      API->>API: name in TOOL_SPECS
      API->>Tools: execute_tool validated args
      Tools-->>API: result
      API->>Cache: store hash plus response
      API-->>Client: JSON result
    else other_waiter
      Cache-->>API: cached entry
      API-->>Client: replay or 409
    end
  end
```

Offline / tests may skip the LLM by sending `force_tool` + `force_args`; allowlist validation still applies.
