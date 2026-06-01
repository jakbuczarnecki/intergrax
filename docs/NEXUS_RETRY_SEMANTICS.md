# Nexus retry semantics (Phase Q-N.3)

Intergrax has **two retry layers**. They must not double-retry the same failure without a trace event.

## 1. Graph / validation retry — `RetryEngine`

- **Location:** `intergrax/runtime/nexus/retry/retry_engine.py`
- **Scope:** Nexus execution graph after agent step validation fails.
- **Policy:** `RetryPolicy` (`max_retries`, `retry_alternate_agent`).
- **Behavior:** May switch to an alternate agent from `AgentRegistry`; records `RetryRecord` on the task result.
- **Hooks:** `BEFORE_RETRY` / `AFTER_RETRY` when a `MiddlewarePipeline` is wired on the engine path.

## 2. Run-level retry — `RuntimeConfig.max_run_retries`

- **Location:** `RuntimeEngine` / pipeline execution (`runtime_steps`).
- **Scope:** Transient LLM or tool failures inside a single agent run (`RuntimeErrorCode.LLM_ERROR`, `TOOL_ERROR`, …).
- **Policy:** `max_run_retries`, `retry_run_on` on `RuntimeConfig` / `TraceRuntimeConfig`.
- **Behavior:** Re-executes the runtime pipeline step; does not change Nexus agent selection.

## Optional coordinator

A future `RetryCoordinator` may delegate to both layers with explicit trace events (`RETRY_SCHEDULED`, `RETRY_STARTED`). Until then, configure each layer independently and avoid setting both to aggressive values for the same step.

**Canon pointer:** architecture §31 (validation retry), §42 retry events.
