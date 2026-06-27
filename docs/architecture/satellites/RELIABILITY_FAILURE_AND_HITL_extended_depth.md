# RELIABILITY_FAILURE_AND_HITL — extended depth

**Parent hub:** [`RELIABILITY_FAILURE_AND_HITL.md`](../RELIABILITY_FAILURE_AND_HITL.md)

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
| **Run retry** | UAEP step (LLM/tool transient) | `max_run_retries`, `retry_run_on` | §31.1, `AgentEngine` |
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

**Cross-domain maintenance (§6.1av):** durable async queue opt-in — [`ORCH-MAINT-04`](../plan/ORCHESTRATION.md#61av-harness-implementation-queue--orchestration-audit-maintenance-planned) (REL-MAINT-03); LLM profile failover on retriable provider errors — [`LLM-MAINT-03`](../plan/LLM_ADAPTERS.md#61av-harness-implementation-queue--llm-adapters-audit-maintenance-planned) (REL-MAINT-04). Verification: `scripts/maintenance/check_harness_resilience_policy.py`.

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
