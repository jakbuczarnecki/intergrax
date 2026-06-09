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
- guardrail denial (`LlmGuardrailMiddleware` BLOCK at input/output/tool hooks — composes with HITL when policy escalates; see [`INTEGRATIONS.md`](INTEGRATIONS.md) §47)
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

# 34. Configurable Resilience Policy Framework

Fault tolerance in Intergrax is **policy-driven and modular** — not hardcoded retry loops in agents or ad-hoc exception handlers in Tier-3 hosts.

## 34.1 Design goals

| Goal | Mechanism |
|------|-----------|
| **Continuity of work** | Retry, alternate agent, checkpoint resume, partial completion |
| **Configurable per host** | `ReliabilityProfile`, `OrchestrationProfile`, `RuntimeConfig` |
| **Composable policies** | Named `ResiliencePolicy` bundles resolved at assembly time |
| **Observable** | `RETRY_*`, `CHECKPOINT_*`, `TASK_PROGRESS`, `GRAPH_BACKPRESSURE` events |
| **Safe failure** | Circuit breakers, timeouts, stop conditions, escalation |

## 34.2 Resilience policy modules

Policies are **orthogonal modules** composed through profiles — authors enable only what the product needs.

| Module | Scope | Key controls | Canon |
|--------|-------|--------------|-------|
| **Graph retry** | Nexus node after validation failure | `max_retries`, `retry_alternate_agent`, `RetryRecord` | §31.1, `RetryEngine` |
| **Run retry** | UAEP step (LLM/tool transient) | `max_run_retries`, `retry_run_on` | §31.1, `RuntimeEngine` |
| **Run-level graph retry** | Whole graph re-execution | `max_run_retries` on `OrchestrationProfile` | [`ORCHESTRATION.md`](ORCHESTRATION.md) §52.1 Layer C |
| **Circuit breaker** | Integration / provider calls | Open/half-open thresholds via `ReliabilityProfile` | §33.2 |
| **Idempotency** | Side-effectful tools | `idempotency_key` on `ToolRequest` | §33.1 |
| **Checkpoint / resume** | Long-running and HITL tasks | `RuntimeCheckpoint`, `resume_token` | §33.3, UAEP §42.9 |
| **Partial completion** | Multi-node graphs | `PARTIALLY_COMPLETED` when policy allows | [`ORCHESTRATION.md`](ORCHESTRATION.md) §52.2 |
| **Failover (agent)** | Same logical role | Alternate `agent_id` on node retry | §31.1 Layer A |
| **Redundancy (infra)** | Host availability | Nexus replicas, queue workers | [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) |
| **Recovery reboot** | Corrupted / stuck run | `RUNTIME_RECOVERY_REQUIRED` interrupt → policy: retry graph, cold restart node, or escalate | UAEP §42.8, §34.4 |

**Rule:** agents emit **intent** (`AgentDecision.RETRY`, `INTERRUPT`); Nexus + PolicyEngine **execute** policy — no agent-internal unbounded retry against adapters.

## 34.3 ResiliencePolicy contract (target)

```text
ResiliencePolicy:
    policy_id: str
    version: str
    on_dependency_error: RETRY | CIRCUIT_BREAK | FAIL | DEGRADE
    on_quality_error: RETRY_ALTERNATE | RETRY_SAME | REQUEST_HUMAN | FAIL
    on_timeout: RETRY | FAIL | PARTIAL
    on_runtime_error: RETRY_RUN | RETRY_GRAPH | RECOVERY_REBOOT | ESCALATE
    max_attempts: int
    backoff: fixed | exponential | none
    alternate_agent_ids: list[str] | null
    allow_partial_result: bool
    checkpoint_on_pause: bool
    reboot_strategy: NONE | RE_EXECUTE_NODE | RE_EXECUTE_GRAPH | COLD_AGENT_RELOAD
```

**As-built (2026-06-09):** unified `ResiliencePolicy` on `ReliabilityProfile`; resolved via `policy_resolver` into `RetryEngine` and trace `RECOVERY_REBOOT`. **Tier-3 debt:** `apply_reliability_task_defaults` wired on **lab host only** — other hosts need profile enricher or H-APP-WIRING.1.

## 34.4 Recovery reboot semantics

**Reboot** means **controlled re-execution** of a bounded unit — not OS process restart unless the host operator chooses infra recycle.

| Strategy | When | Effect |
|----------|------|--------|
| `RE_EXECUTE_NODE` | Transient agent/tool failure | Same graph node from clean UAEP cursor |
| `RE_EXECUTE_GRAPH` | Plan/graph corruption or run-level retry budget | `RetryCoordinator` full graph |
| `COLD_AGENT_RELOAD` | Agent state suspected corrupt | Re-instantiate agent from registry; preserve checkpoint |
| `ESCALATE` | Budget exhausted or policy deny | HITL queue or `FAILED` with audit |

Reboot MUST preserve: `task_id`, trace lineage, idempotency keys for external side effects.

## 34.5 Wiring map

```text
ApplicationEnvironmentProfile.reliability
    → reliability_wiring.py
    → ReliabilityProfile (circuit breaker, idempotency store)
    → runtime_config_bridge (max_run_retries)
OrchestrationProfile (max_run_retries, long_running, checkpoint)
    → GraphExecutor + RetryEngine + long_running_bridge
PolicyEngine
    → maps failure class (§33.4) → ResiliencePolicy action
```

**Cross-ref:** orchestration resilience matrix [`ORCHESTRATION.md`](ORCHESTRATION.md) §52 · execution flow [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §14 · interrupt model UAEP §42.8.

---

# 35. Autonomy Control Model (Autonomy Slider)

Users and operators MUST be able to **steer how much the system acts without asking** — at session, task, or step granularity. This is distinct from host **execution posture** (`ExecutionMode`: STRICT | BALANCED | EXPLORATORY) and agent **dispatch mode** (`AgentExecutionMode`: SYNC | ASYNC).

## 35.1 Autonomy levels

| Level | User experience | Harness behaviour |
|-------|-----------------|-------------------|
| **MANUAL** | User drives each meaningful action | Tools with side effects blocked unless explicitly approved; planner may suggest but not execute; HITL default-on for external writes |
| **ASK** | Agent proposes; user confirms high-impact steps | Policy routes risky tools and low-confidence outputs to approval queue; auto-continue for read-only / safe tools per allowlist |
| **AUTONOMOUS** | Agent executes within policy envelope | Full tool policy + cost caps; HITL only on policy triggers (risk class, confidence threshold, regulated pathways) |

```text
AutonomyLevel:
    MANUAL
    ASK
    AUTONOMOUS
```

**Mid-run changes:** autonomy MAY change at any time via `TaskExecutionOptions.autonomy_level` or operator API — PolicyEngine re-evaluates **before the next UAEP step** and before each tool invocation (UAEP §42.11).

## 35.2 Resolution order

```text
effective_autonomy = min(
    user_requested_autonomy,      # slider / task option
    tenant_policy_ceiling,        # org governance
    execution_mode_ceiling,       # STRICT caps at ASK for destructive tools
    agent_contract.risk_level     # high-risk agents never fully AUTONOMOUS without override
)
```

| Host `execution_mode` | Typical autonomy ceiling |
|-----------------------|--------------------------|
| `EXPLORATORY` | Up to `AUTONOMOUS` (lab) |
| `BALANCED` | Up to `ASK` for destructive tools; `AUTONOMOUS` for read-only |
| `STRICT` | Default `ASK`; `AUTONOMOUS` only with explicit policy exception |

## 35.3 Mapping to runtime primitives

| Autonomy | Tool execution | Planning | HITL |
|----------|----------------|----------|------|
| MANUAL | `PolicyDecision.DENY` except allowlisted reads; `REQUEST_HUMAN` before writes | Plan visible; execute only after approval | Default for most steps |
| ASK | Risk-scored: auto vs queue | Auto plan; confirm on `risk >= threshold` | Queue for gated tools |
| AUTONOMOUS | Policy + budget only | Auto plan and execute | On interrupt types only (§42.8) |

**Implementation anchors:** `PolicyEngine.evaluate_tool_call`, `AgentDecision.REQUEST_HUMAN`, `HumanDecisionStore`, `hitl.*` tools — UAEP §42.10.

## 35.4 UX contract (platform)

- Slider state MUST be **persisted** on `Task` / session metadata and echoed in trace (`AUTONOMY_LEVEL_SET`, `AUTONOMY_LEVEL_CHANGED`).
- Downgrade (AUTONOMOUS → MANUAL) MUST be **immediate** for new steps; in-flight tool calls follow cancel-or-complete policy per `CancellationCoordinator`.
- Upgrade (MANUAL → AUTONOMOUS) MUST NOT bypass unresolved HITL items.

**As-built (2026-06-09):** `AutonomyLevel` on `TaskExecutionOptions`; effective level via `autonomy_resolver` + `AutonomyGovernanceMiddleware`; trace events `AUTONOMY_LEVEL_SET` / `AUTONOMY_LEVEL_CHANGED`. **Mid-run HTTP API** (`POST …/tasks/{id}/autonomy`) mounted on **lab host only** — runtime downgrade/upgrade works on all paths when set on task envelope.

**Tier-3 debt:** product hosts without `mount_harness_task_routes` require client to set `autonomy_level` on task create or resume payload.

**Plan:** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md) Phase REL-ADV (**Done**); surface parity → H-APP-WIRING.1.

---
