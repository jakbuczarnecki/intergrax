# ADR-HARNESS-W4-PRODUCTION-TOOL-BOUNDEDNESS: Production ToolRuntime dependency admission configuration ownership

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED — INDEPENDENTLY APPROVED ARCHITECTURE DECISION** |
| **Date** | 2026-09-26 |
| **Baseline SHA** | `c6d6f87d3206b93387257af5f6a5e6b1c0978d39` (`development` = `origin/development` at P0-R1 replay) |
| **Parent** | `HARNESS-W4` — Scale / Resilience / Cancellation recertification |
| **Child** | `HARNESS-W4-R1` — Production Tool Boundedness & Admission Closure |
| **Related** | [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) · [`ADR_ENTERPRISE_TOOL_DEPENDENCY_ATTEMPT_BOUNDARY.md`](ADR_ENTERPRISE_TOOL_DEPENDENCY_ATTEMPT_BOUNDARY.md) · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) |

Independent review approved Outcome B: **`ReliabilityProfile` is the canonical Tier-3 configuration owner** to be extended for production dependency-concurrency admission.

---

## 1. Problem

`HARNESS-W4` requires current-HEAD proof of bounded concurrency, queue/backpressure, overload/saturation, and fail-closed rejection semantics for production tool execution. W2 qualified `DependencyAttemptExecutionBoundary` + optional `RuntimeToolInvoker` wiring, but **strict production composition** can still build `RuntimeToolInvoker` with `dependency_attempt_boundary=None` while `production_mode=True`. Stdlib `ThreadPoolExecutor` worker defaults do not satisfy enterprise admission-before-submit / queue-protection invariants ([`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) tool row).

This ADR closes **only** the architecture question:

> What existing platform-owned typed configuration/composition surface is the canonical owner of production `DependencyConcurrencyIdentity → DependencyConcurrencyPolicy` selection for **ToolRuntime**, so strict production can require `DependencyAttemptExecutionBoundary` without magic defaults or a second configuration authority?

**No production code changes in this task.**

---

## 2. Current topology

```text
ApplicationEnvironmentProfile
  ├── orchestration_profile (W0 graph caps: max_parallel_nodes / max_inflight_nodes)
  └── reliability_profile (idempotency, circuit breaker threshold, ResiliencePolicy, recovery)

build_production_runtime_tool_invoker(production_mode=True, dependency_attempt_boundary=None)  # allowed
  → RuntimeToolInvoker._execution_pool.submit(...) without admission when boundary is None

build_production_runtime_tool_invoker(..., dependency_attempt_boundary=DependencyAttemptExecutionBoundary)
  → dep_boundary.acquire(...) before submit; permit until worker completion (W2-B2)
```

LLM provider path (comparison only): `set_llm_provider_dependency_boundary` process hook + `LLMRuntimeLifecycleBinding.bind_provider_dependency_boundary` — still **no** typed host profile field for `DependencyConcurrencyPolicy` map; tests/harness construct `LocalDependencyConcurrencyAdmission` inline.

---

## 3. Ownership matrix (concern separation)

| Concern | Canonical owner | Contract | Configuration owner | Composition owner | Lifecycle owner | Current production wiring | W4 status |
|---------|-----------------|----------|---------------------|-------------------|-----------------|---------------------------|-----------|
| Root execution admission | `ExecutionRuntime` / host task intake | `ExecutionCapacityAdmissionPort` | Optional port injection (not on `ApplicationEnvironmentProfile`) | Host execution composition | `ExecutionRuntime` permit `finally` | Often `None` (legacy) | OPEN (orthogonal to tool gap) |
| Graph/fan-out concurrency | `GraphExecutor` | Orchestration caps + semaphore | `ApplicationEnvironmentProfile.orchestration_profile` | Nexus / graph composition | Graph batch / inflight semaphores | Strict requires W0 caps | Partial (graph only) |
| **Tool dependency concurrency** | `RuntimeToolInvoker` + resilience boundary | `DependencyConcurrencyAdmissionPort` / `DependencyAttemptExecutionBoundary` | **`ReliabilityProfile` (accepted extension owner; field pending R1)** | `build_production_runtime_tool_invoker` (optional boundary today) | `RuntimeToolInvoker.close` → boundary `drain_and_close` | **Boundary omitted** on all strict production roots | **BLOCKER (HARNESS-W4-R1)** |
| LLM provider dependency concurrency | `LLMAdapter._execute` | Same admission contract | `LLMCallConfig` (rate/retry only; **not** concurrency map) | `set_llm_provider_dependency_boundary` + registry apply | Adapter-bound boundary | Optional global setter only | OPEN (parallel seam) |
| Provider rate limiting | `execute_with_resilience` | `ProviderRateLimitPort` | `LLMCallConfig` / adapter defaults | LLM adapter stack | Per-attempt acquire | Wired in resilience path | W2-C qualified; W4 FRZ-REL-11 still OPEN at freeze |
| Retry budget | `execute_with_resilience` / Nexus retry | `RetryBudgetPort` / `ResiliencePolicy` | `ReliabilityProfile.resilience_policy` | Retry engine / task metadata | Per-attempt | Task metadata from reliability wiring | W2 qualified; freeze OPEN |
| External-operation cancellation intent | External-operation stores / ports | `ExternalOperationCancellationPort` | Composition-injected | Tool/LLM composition | Store + owner lifecycle | Partially wired on invoker when provided | W4-A/C qualified; FRZ-REL-04/05 OPEN |
| Physical provider termination | LLM/tool termination ports | `ExternalOperationTerminationPort` | Composition | Lifecycle binding | Port + future bind | Optional | W4-D qualified; FRZ-REL-05 OPEN |
| Tool physical termination | Tool termination port on invoker | Same family | Composition | `RuntimeToolInvoker` | Future bind on submit | Optional | OPEN |

---

## 4. Production caller inventory (`build_production_runtime_tool_invoker`)

| Path | Class | `production_mode` source | `dependency_attempt_boundary` | Policy source if supplied | Lifecycle owner |
|------|-------|--------------------------|-------------------------------|---------------------------|-----------------|
| `intergrax/applications/_shared/declarative_tool_wiring.py` | **production** (strict host) | `environment.execution_mode == strict` or caller flag | **No** | — | Host / catalog invoker (no boundary shutdown today) |
| `intergrax/runtime/nexus/engine/runtime_context.py` | **production** (Nexus runtime) | `config.production_mode` | **No** | — | `RuntimeContext` / runtime teardown |
| `intergrax/runtime/execution/execution_bound_catalog_tool_composition.py` | **production** (execution-bound catalog) | `production_mode` parameter | **No** | — | Execution-bound composition host |
| `intergrax/runtime/nexus/tools/invoker.py` (`with_idempotency_store`) | recomposition | caller `production_mode` | **Transfers** existing (may be `None`) | Prior invoker only | Replacement invoker owns boundary |
| `tests/**`, `tests/qualification/**` | test / harness | varies | Sometimes **Yes** (harness W2-R6) | Inline `LocalDependencyConcurrencyAdmission` in tests | Test fixture |

No other production `build_production_runtime_tool_invoker(` call sites on baseline SHA (symbol search).

---

## 5. Reused contracts (frozen)

- `intergrax/contracts/dependency_concurrency_admission.py` — `DependencyConcurrencyAdmissionPort`, `DependencyConcurrencyPolicy`, overload modes `REJECT` / `WAIT_WITH_TIMEOUT` only
- `intergrax/runtime/resilience/local_dependency_concurrency_admission.py` — process-local port implementation
- `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` — sync bridge; acquire before submit

Do **not** introduce a second contract family, manager, or scheduler.

---

## 6. Decision question (restated)

Canonical typed owner for production **tool** `DependencyConcurrencyIdentity → DependencyConcurrencyPolicy` materialization, feeding `DependencyAttemptExecutionBoundary` into **every** strict production `build_production_runtime_tool_invoker` root without hidden defaults.

---

## 7. Considered outcomes

| Outcome | Assessment on baseline SHA |
|---------|----------------------------|
| **A — Existing canonical owner exists** | **Rejected.** No `ApplicationEnvironmentProfile` (or sibling) field materializes dependency concurrency policies. `applications/` has zero `DependencyConcurrency*` references. |
| **B — Extend existing owner without new authority** | **Accepted.** `ReliabilityProfile` (`ApplicationEnvironmentProfile.reliability_profile`) already owns host-declared operational reliability controls (idempotency, circuit breaker threshold, `ResiliencePolicy`, recovery). W2-ADR assigns policy values to “application/composition configuration,” not `ToolContract` or `RuntimeToolInvoker`. Graph caps remain on `OrchestrationProfile`; per-adapter LLM call policy remains on `LLMCallConfig`. |
| **C — No owner** | Would apply only if Outcome B is rejected by maintainers (no legitimate profile extension). **Not selected.** |

---

## 8. Selected outcome

**Selected outcome:** EXISTING OWNER REQUIRES APPROVED EXTENSION (Outcome B)

**Approval:** ACCEPTED (independent architecture review)

**Configuration owner:** `ReliabilityProfile` — canonical Tier-3 configuration owner for production dependency-concurrency admission configuration

**Implementation:** PENDING `HARNESS-W4-R1`

### 8.1 Configuration owner

| Item | Value |
|------|-------|
| **Owner** | `ReliabilityProfile` in `intergrax/applications/contracts/environment_profile/sub_profiles.py` |
| **Contract / type (implementation child)** | Typed immutable/frozen value object in `intergrax/contracts/` describing `DependencyConcurrencyIdentity` + `DependencyConcurrencyPolicy` entries (proposed implementation type name e.g. `DependencyConcurrencyAdmissionConfiguration` — **semantics frozen; final class name not mandated**) |
| **Why responsibility belongs here** | Sole Tier-3 host profile for dependency-adjacent **operational** reliability (breaker, recovery, retry policy, idempotency posture). Orthogonal to orchestration graph caps and tool semantic contracts. Matches W2-ADR “application/composition configuration” without a new top-level authority. |
| **Policy materialization flow (implementation child)** | `ApplicationEnvironmentProfile.reliability_profile` → `wire_application_reliability(...)` / extended reliability composition → validate strict-mode completeness → build `LocalDependencyConcurrencyAdmission(policy_map)` → `DependencyAttemptExecutionBoundary(admission)` → pass to `build_production_runtime_tool_invoker` |
| **Composition root (consumer)** | Extended Tier-3 wiring beside `wire_application_reliability` (e.g. `materialize_tool_dependency_attempt_boundary(env)`) called from **all** production roots in §4 |
| **Boundary lifecycle owner** | `RuntimeToolInvoker` (existing `close` / reconfiguration transfer semantics) |
| **Strict-production fail-closed rule (implementation child)** | Strict production `ToolRuntime` **must not** silently materialize with unbounded dependency admission. Final implementation must materialize explicit typed policy **or** fail closed at composition (mirror `validate_strict_host_execution_capacity` / `ProductionRuntimeToolInvokerCompositionError` pattern). No magic default capacities; no implicit unlimited strict production. |

### 8.2 Target implementation shape (child scope only)

```text
ReliabilityProfile
    ↓
typed dependency-concurrency configuration (immutable/frozen)
    ↓
existing Tier-3 reliability composition
    ↓
DependencyConcurrencyAdmissionPort (LocalDependencyConcurrencyAdmission)
    ↓
DependencyAttemptExecutionBoundary
    ↓
build_production_runtime_tool_invoker(..., dependency_attempt_boundary=..., production_mode=True)
    ↓
all strict-production RuntimeToolInvoker roots (§4)
    ↓
acquire before ThreadPoolExecutor.submit
```

### 8.3 Why this is not a duplicate authority

- Does not add `BulkheadManager` / `ConcurrencyManager` / registry.
- Does not put capacity on `ToolContract`.
- Does not let `RuntimeToolInvoker` choose default capacities.
- Reuses existing port + boundary; only adds **declarative host policy** on an existing profile.

**Production boundedness remains blocked** until `HARNESS-W4-R1` implements the extension on every production root.

---

## 9. Structural proofs (baseline SHA, no code changes in P0-R1)

| ID | Result | Evidence |
|----|--------|----------|
| **P1** Builder optionality | **YES** | `build_production_runtime_tool_invoker(..., production_mode=True)` does not require `dependency_attempt_boundary`; tests compose without boundary (`test_gr10_r9_production_composition_custom_mse_port_wired`). |
| **P2** Strict production roots | **NO** | §4 — declarative, runtime_context, execution_bound_catalog: none supply boundary. |
| **P3** Queue protection when boundary set | **YES** | `RuntimeToolInvoker._execute_once`: `dep_boundary.acquire` before `self._execution_pool.submit` when boundary non-`None` (`invoker.py`). |
| **P4** Failure semantics | **YES** | `DependencyConcurrencyPolicy` + `LocalDependencyConcurrencyAdmission`: `REJECT`, `WAIT_WITH_TIMEOUT`, no `WAIT_FOREVER`; missing policy → `DependencyConcurrencyPolicyMissingError`. |
| **P5** Ownership | **YES** | Invoker does not select policies; boundary does not own retry/rate-limit/governance; port does not mint execution identity. |
| **P6** Authority | **YES** | Admission rejects/sleeps for slots only; no governance permission, no `ExecutionId`, no alternate execution authority. |

---

## 10. Forbidden alternatives

As listed in HARNESS-W4-R1-P0: `ToolConcurrencyManager`, `GlobalCapacityManager`, `ResourceScheduler`, `BulkheadManager`, magic invoker defaults, `ToolContract` capacity fields, `WAIT_FOREVER`, env-only semantic authority, `dict[str, int]` pseudo-contract, generic metadata, reflection probing, `Any`/`object` policy surfaces, duplicate sync admission contracts.

---

## 11. Test / evidence matrix (P0-R1 replay on `c6d6f87d…`, `-p no:xdist`)

### 11.1 Prior incorrect attribution (corrected)

| Previous claim | Actual evidence | Corrected classification |
|----------------|-----------------|--------------------------|
| `test_llm_final_flow_tenant_then_resilience_then_admission_inside_physical` failed with `LLMRateLimitError` in a worker thread | Test is a **static** architecture gate (`Path.read_text`, `str.index`, ordering asserts). It does not run LLM, rate limiter, or worker threads. P0-R1 replay: **`ValueError: substring not found`** for `"def _execute("` in `llm_adapter.py` (no such substring on HEAD). | **REAL TEST / QUALIFICATION FAILURE** — static gate out of sync with source layout; **not** an environment flake and **not** attributable to W2-C rate-limit behavior. |

`LLMRateLimitError` / `PytestUnhandledThreadExceptionWarning` observed only in **`test_enterprise_scale_resilience_w2_c_retry_containment.py`** (W2-C behavioral suite); pytest **PASS** with warning/background-thread evidence (Case C).

### 11.2 Replay runs

| Run | Scope | Result | Warnings / errors | Classification |
|-----|-------|--------|-------------------|----------------|
| **T1** | `…w2_final_qualification.py::test_llm_final_flow_tenant_then_resilience_then_admission_inside_physical` | **FAIL** | `ValueError: substring not found` at `adapter.index("def _execute(")` | **REAL TEST / QUALIFICATION FAILURE** (test body, synchronous) |
| **T2** | Full `test_enterprise_scale_resilience_w2_final_qualification.py` (3 tests) | **1 failed**, 2 passed | Same `ValueError` on static gate | **REAL TEST / QUALIFICATION FAILURE** |
| **T3** | Full `test_enterprise_scale_resilience_w2_c_retry_containment.py` (7 tests) | **PASS** | 100 warnings; `PytestUnhandledThreadExceptionWarning` / `LLMRateLimitError` in `_worker` threads (e.g. `test_retry_storm_caps_provider_calls`) | **warning/background-thread evidence** — not test failures |
| **T4** | Combined batch (W4-A/C/D, boundary unit, invoker admission, W2 final, W2-C) with `-x` | **FAIL** at 68/78 collected | Failed node: same static gate; 68 passed before stop | **REAL TEST / QUALIFICATION FAILURE** (isolated T1 identical) |

**W4-A / W4-C / W4-D:** green in T4 prefix (68 tests) — historical cancellation evidence only; **does not** prove strict production tool boundedness or close FRZ-REL-08..10.

**Production wiring gap** (§4, P2) remains independent of W2 final static gate failure.

---

## 12. Applicable FRZ criteria (all remain OPEN)

| ID | Relevance to this ADR |
|----|------------------------|
| FRZ-REL-08 | Tool production admission not explicit in host profile / composition |
| FRZ-REL-09 | Pool queue unprotected when boundary omitted |
| FRZ-REL-10 | No strict fail-closed materialization for missing tool admission policy |
| FRZ-REL-11 | LLM rate/retry orthogonal; tool bypass does not satisfy freeze |

Supporting (not closed by this task): FRZ-REL-04..07 — **OPEN**, no PASS.

This ADR supplies **architecture** evidence (owner, contracts, fail-closed topology intent) only — **not** implementation, behavioral overload, or current production boundedness proof.

---

## 13. STOP conditions

- **STOP — ROADMAP STATE CHANGED** if `HARNESS-W4` ≠ CURRENT before implementation child.
- **STOP — W4 BASELINE CHANGED** if a concurrent commit mandates boundary on all production roots without the approved profile extension (re-audit required).
- W2 final static gate failure on HEAD is **qualification debt** (repair outside P0-R1 doc scope); does not reverse accepted Outcome B.

---

## 14. Required implementation child scope (`HARNESS-W4-R1`)

1. Typed field on `ReliabilityProfile` + frozen contract value object (no magic capacities).
2. `materialize_tool_dependency_attempt_boundary(env)` (or equivalent) in `applications/_shared/`.
3. Inject boundary in: `declarative_tool_wiring.py`, `runtime_context.py`, `execution_bound_catalog_tool_composition.py`, and preserve `with_idempotency_store` transfer semantics.
4. `production_mode=True` + strict: fail closed if boundary required but not materialized.
5. Structural tests: all `build_production_runtime_tool_invoker` production roots; behavioral overload/reject tests mirroring W2 units; repair W2 final static gate anchor if still required.
6. Lifecycle: boundary shutdown via existing `RuntimeToolInvoker.close` / reconfiguration transfer.

---

## 15. No-implementation statement

This ADR records corrected qualification replay evidence and an **accepted** configuration owner only. No production runtime behavior, capacity values, managers, or composition changes were made in HARNESS-W4-R1-P0-R1.

---

## 16. Parent roadmap recommendation (do not edit roadmap in P0-R1)

| Item | Value |
|------|-------|
| `HARNESS-W4` current | CURRENT |
| `HARNESS-W4` recommended | **BLOCKED** (pending R1 implementation after architecture acceptance) |
| Child | `HARNESS-W4-R1` Production Tool Boundedness & Admission Closure |

Reason: strict production tool composition lacks mandatory dependency admission / queue protection required by W4 and FRZ-REL-08..10.

---

## 17. Enterprise audit matrix (post P0-R1 correction)

| Area | Required conclusion |
|------|------------------------|
| Boundaries | No layer boundary change |
| Communication | Existing typed admission boundary retained |
| Composition | One canonical Tier-3 policy owner selected (`ReliabilityProfile`) |
| Ownership | `ReliabilityProfile` accepted; no shadow owner |
| Contracts | Existing admission contracts retained |
| Strong typing | Future config must be typed/frozen; no dict/string pseudo-contract |
| Pluginability | `DependencyConcurrencyAdmissionPort` remains swappable |
| Replaceability | Local admission implementation remains replaceable |
| Bypass resistance | Current production gap remains open until R1 implementation |
| Governance | Capacity admission remains orthogonal to permission |
| Execution | Admission does not mint or widen execution authority |
| Evidence | Correct test attribution only; W2 final static gate failure recorded honestly |
| Fail-closed | Required future strict-production rule frozen in §8.1 |
| Regression | R1 must add structural gate over all production roots |
