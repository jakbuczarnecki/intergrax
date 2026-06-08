# Reliability, Failure Model, and Human-in-the-Loop

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 22  
---

---

# 30. Failure Model

Failures are expected.

The system must treat failure as normal.

Failure types:

- agent failure
- tool failure
- adapter failure
- timeout
- invalid output
- missing data
- low confidence
- unsafe action
- human rejection
- incomplete result

Failure handling options:

- retry same step
- retry with different agent
- ask human
- degrade gracefully
- return partial result
- stop execution
- mark as failed

---


---

# 31. Retry Policy

Retries must be controlled.

Every retry should have:

- reason
- retry count
- changed strategy if possible
- stop condition

Do not retry endlessly.

Retries should be visible in traces.

### 31.1 Two retry layers (do not double-retry)

Intergrax has **two independent retry layers**. Configure each explicitly; avoid aggressive values on both for the same step without trace events.

| Layer | Location | Scope | Policy |
|-------|----------|-------|--------|
| **Graph / validation** | `RetryEngine` (`runtime/nexus/retry/retry_engine.py`) | Nexus execution graph after agent step validation fails | `RetryPolicy` (`max_retries`, `retry_alternate_agent`); may switch agent via `AgentRegistry`; `RetryRecord` on task result; hooks `BEFORE_RETRY` / `AFTER_RETRY` when middleware wired |
| **Run-level** | `RuntimeEngine` / `runtime_steps` | Transient LLM or tool failures inside one agent run (`RuntimeErrorCode.LLM_ERROR`, `TOOL_ERROR`, …) | `RuntimeConfig.max_run_retries`, `retry_run_on`; re-executes pipeline step; does not change Nexus agent selection |

A future `RetryCoordinator` may delegate to both with explicit `RETRY_SCHEDULED` / `RETRY_STARTED` events (§42.34). Until then, agents emit **intent** (`AgentDecision.RETRY`); runtime executes policy — no agent-internal `for attempt in range(n)` against adapters.

---


---

# 32. Human In The Loop

Human approval may be required for:

- sending external messages
- modifying external systems
- deleting data
- financial actions
- legal conclusions
- risky automation
- uncertain results

Nexus manages human approval.

Agents may request approval via `AgentDecision.REQUEST_HUMAN`, but Nexus controls the approval flow (§42.10).

Agents MUST NOT implement ad-hoc human gates or send approval messages directly.

---

---

# 33. Reliability Primitives

Reliability is enforced at **graph**, **run**, and **integration** layers.

## 33.1 Idempotency and deduplication

- Side-effectful tools SHOULD accept `idempotency_key` on `ToolRequest` (§42.12).
- Tier-3 `ReliabilityProfile` enables idempotency stores via integration `key_value_cache` backends.
- Duplicate intake deduplication uses stable task/run identifiers on `TaskEnvelope`.

## 33.2 Circuit breaker and timeouts

| Layer | Mechanism |
|-------|-----------|
| Integration calls | Circuit breaker on provider hosts; wired from `ReliabilityProfile` |
| LLM adapters | Retry/backoff profiles on `LLMProfile` |
| Graph steps | `RetryEngine` + `RetryPolicy` (§31.1) |
| UAEP run | `RuntimeConfig.max_run_retries` |

## 33.3 Checkpoint, resume, compensation

- `RuntimeCheckpoint` captures plan snapshot, graph snapshot, UAEP cursor (§42.9).
- HITL pause creates `PauseRecord`; resume restores checkpoint.
- Long-running tasks expose partial results API + scheduler hooks (§26 in [`ORCHESTRATION.md`](ORCHESTRATION.md)).

## 33.4 Error taxonomy (Harness)

| Class | Examples | Typical response |
|-------|----------|------------------|
| `UserError` | Invalid input, denied permission | Fail fast, no retry |
| `PolicyError` | Guardrail violation | DENY / REQUIRE_HUMAN |
| `DependencyError` | Provider down | Retry + circuit breaker |
| `RuntimeError` | Timeout, state corruption | Retry run or escalate |
| `QualityError` | Schema / rubric failure | Retry alternate agent or critic loop |

## 33.5 Code map

| Module | Role |
|--------|------|
| `runtime/nexus/retry/retry_engine.py` | Graph-level retry |
| `runtime/resilience/` | Circuit breaker helpers |
| `applications/_shared/reliability_wiring.py` | Profile → runtime |
| `runtime/sandbox/`, `runtime/shadow/` | Isolated risky execution |
| `runtime/human/` | HITL approval flow (§32, UAEP §42.10) |

**Plan:** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md) Phase REL.

---
