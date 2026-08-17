# Business objective

## What this system does

This Task 2 deliverable is a **reference-constrained recommendation engine** plus a **hardened agentic API**.

Given a small **user trait profile** (interests, goals, level) and a **reference catalog** (careers, majors, electives), the system:

1. Ranks catalog items that fit the profile.
2. Writes a short narrative explaining the picks.
3. **Validates** both outputs with deterministic (non-LLM) rules before accepting them.

Separately, an async service lets an LLM **select one of a few real tools**, but only after the service checks that the chosen tool is on an **allowlist**.

## Why catalog and tool choice are untrusted

Two inputs must never be trusted blindly:

| Input | Why untrusted | Safety pattern |
|-------|---------------|----------------|
| Reference catalog file | Can be missing, empty, malformed, or edited incorrectly | Load + schema-validate; fail closed |
| LLM outputs (narrative text, tool name, tool args) | Models invent ids, cite wrong items, or request unsafe tools | Deterministic validator + allowlist execution |

This mirrors the core safety pattern behind agentic features on the Educatlyy-style stack: **models propose; code disposes**.

## Business rules encoded in Part A

- Ranked items must exist in the reference catalog (no invented careers/majors).
- The narrative may only mention items that appear in **that ranked list**, not the full catalog.
- If the catalog is empty/malformed, or ranking fails, the system **fails closed** (no silent “best effort” hallucinations).

## Business rules encoded in Part B

- The model may pick a tool; the service executes it only if the name is allowlisted.
- Retries with the same `Idempotency-Key` must return the same result (no double side effects).
- One bad request must not crash the process or poison other requests.

## Domain flavor

Mock catalog items are **education / career** oriented (careers, majors, electives) so this exercise aligns with the upcoming multi-agent recommendation work, without depending on Kafka or the full production monorepo.
