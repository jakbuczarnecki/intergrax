# Reliability, Failure Model, and Human-in-the-Loop

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 29. Validation Model

Validation is mandatory.

Validation should not rely only on LLM confidence.

Possible validation types:

- schema validation
- rule-based validation
- data completeness validation
- source citation validation
- secondary model review
- separate validator agent
- human review
- executable tests
- consistency checks

Validation should be defined before or during planning.

For high-risk tasks, Nexus should create a validation contract before execution.

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

A future `RetryCoordinator` may delegate to both with explicit `RETRY_SCHEDULED` / `RETRY_STARTED` events ([§42](UNIFIED_EXECUTION_RUNTIME.md).34). Until then, agents emit **intent** (`AgentDecision.RETRY`); runtime executes policy — no agent-internal `for attempt in range(n)` against adapters.

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

Agents may request approval via `AgentDecision.REQUEST_HUMAN`, but Nexus controls the approval flow ([§42](UNIFIED_EXECUTION_RUNTIME.md).10).

Agents MUST NOT implement ad-hoc human gates or send approval messages directly.

---

