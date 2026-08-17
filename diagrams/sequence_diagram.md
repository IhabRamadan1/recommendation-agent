# Sequence diagram (Part B)

Hardened agent invoke path with idempotency and allowlisted tools.

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI_async
  participant Cache as AsyncTTLCache
  participant LLM as OpenAI_tools
  participant Tools as allowlisted_tools

  Client->>API: POST /agent/invoke Idempotency-Key
  API->>Cache: get key
  alt cache_hit
    Cache-->>API: prior_response
    API-->>Client: replay JSON
  else miss
    API->>LLM: bind_tools allowlist schemas
    LLM-->>API: tool_name plus args
    API->>API: name in TOOL_SPECS
    API->>Tools: execute_tool validated args
    Tools-->>API: result
    API->>Cache: set key
    API-->>Client: JSON result
  end
```

Offline / tests may skip the LLM by sending `force_tool` + `force_args`; allowlist validation still applies.
