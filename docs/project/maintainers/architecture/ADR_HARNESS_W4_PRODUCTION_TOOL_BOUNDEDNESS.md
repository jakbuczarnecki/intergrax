# ADR-HARNESS-W4-PRODUCTION-TOOL-BOUNDEDNESS: Production ToolRuntime dependency admission configuration ownership

| Field | Value |
|-------|-------|
| **Status** | **PROPOSED — ARCHITECTURE DECISION REQUIRED** |
| **Date** | 2026-09-26 |
| **Baseline SHA** | `6c86cb11c6f6240506d97b964537347aec6e0892` (`development` = `origin/development` at audit) |
| **Parent** | `HARNESS-W4` — Scale / Resilience / Cancellation recertification |
| **Child** | `HARNESS-W4-R1` — Production Tool Boundedness & Admission Closure |
| **Related** | [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) · [`ADR_ENTERPRISE_TOOL_DEPENDENCY_ATTEMPT_BOUNDARY.md`](ADR_ENTERPRISE_TOOL_DEPENDENCY_ATTEMPT_BOUNDARY.md) · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) |

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
| **Tool dependency concurrency** | `RuntimeToolInvoker` + resilience boundary | `DependencyConcurrencyAdmissionPort` / `DependencyAttemptExecutionBoundary` | **None typed on host profile today** | `build_production_runtime_tool_invoker` (optional boundary) | `RuntimeToolInvoker.close` → boundary `drain_and_close` | **Boundary omitted** on all strict production roots | **BLOCKER (HARNESS-W4-R1)** |
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
| **B — Extend existing owner without new authority** | **Proposed.** `ReliabilityProfile` (`ApplicationEnvironmentProfile.reliability_profile`) already owns host-declared operational reliability controls (idempotency, circuit breaker threshold, `ResiliencePolicy`, recovery). W2-ADR assigns policy values to “application/composition configuration,” not `ToolContract` or `RuntimeToolInvoker`. Graph caps remain on `OrchestrationProfile`; per-adapter LLM call policy remains on `LLMCallConfig`. |
| **C — No owner** | Would apply only if Outcome B is rejected by maintainers (no legitimate profile extension). |

---

## 8. Selected outcome (pending approval)

**EXISTING OWNER REQUIRES APPROVED EXTENSION** (Outcome B proposal — **not** Accepted).

### 8.1 Configuration owner (proposed)

| Item | Value |
|------|-------|
| **Owner** | `ReliabilityProfile` in `intergrax/applications/contracts/environment_profile/sub_profiles.py` |
| **Contract / type (new field, implementation child)** | Typed value object in `intergrax/contracts/` (e.g. `DependencyConcurrencyAdmissionConfiguration` with a **frozen list** of `{identity: DependencyConcurrencyIdentity, policy: DependencyConcurrencyPolicy}` entries — **not** `dict[str, int]`, not metadata keys) |
| **Why responsibility belongs here** | Sole Tier-3 host profile for dependency-adjacent **operational** reliability (breaker, recovery, retry policy). Orthogonal to orchestration graph caps and tool semantic contracts. Matches W2-ADR “application/composition configuration” without a new top-level authority. |
| **Policy materialization flow (implementation child)** | `env.reliability_profile` → validate strict-mode completeness → build `LocalDependencyConcurrencyAdmission(policy_map)` → `DependencyAttemptExecutionBoundary(admission)` → pass to `build_production_runtime_tool_invoker` |
| **Composition root (consumer)** | New or extended Tier-3 wiring beside `wire_application_reliability` (e.g. `materialize_tool_dependency_attempt_boundary(env)`) called from **all three** production roots in §4 |
| **Boundary lifecycle owner** | `RuntimeToolInvoker` (existing `close` / reconfiguration transfer semantics) |
| **Strict-production fail-closed rule (implementation child)** | When `execution_mode=strict` and tool runtime is composed for production effects: missing admission configuration or enabled admission without required `TOOL` policies must **fail closed at composition** (mirror `validate_strict_host_execution_capacity` / `ProductionRuntimeToolInvokerCompositionError` pattern). No implicit unlimited pool. |

### 8.2 Target implementation shape (child scope only)

```text
ReliabilityProfile.dependency_concurrency_admission (typed, approved)
    ↓
LocalDependencyConcurrencyAdmission
    ↓
DependencyAttemptExecutionBoundary
    ↓
build_production_runtime_tool_invoker(..., dependency_attempt_boundary=..., production_mode=True)
    ↓
acquire before ThreadPoolExecutor.submit
```

### 8.3 Why this is not a duplicate authority

- Does not add `BulkheadManager` / `ConcurrencyManager` / registry.
- Does not put capacity on `ToolContract`.
- Does not let `RuntimeToolInvoker` choose default capacities.
- Reuses existing port + boundary; only adds **declarative host policy** on an existing profile.

**Production implementation remains blocked** until this extension is independently approved and implemented.

---

## 9. Structural proofs (baseline SHA, no code changes)

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

## 11. Test / evidence matrix (replay on baseline SHA)

Command (single process, `-p no:xdist`):

```bash
uv run --frozen pytest \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_a_cancellation_qualification.py \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_c_distributed_cancellation_qualification.py \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_d_provider_cancellation.py \
  tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py \
  tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_final_qualification.py \
  tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py \
  -q --tb=line -p no:xdist
```

| Suite | Result | Notes |
|-------|--------|-------|
| W4-A cancellation qualification | **PASS** | Historical cancellation evidence; does not close production tool admission gap |
| W4-C distributed cancellation | **PASS** | Includes tool admission in test fixtures only |
| W4-D provider cancellation | **PASS** | Provider plane |
| W2-B2 boundary unit | **PASS** | Acquire/release semantics |
| W2 tool invoker admission unit | **PASS** | Boundary wired in tests |
| W2 final qualification | **1 FAIL** | `test_llm_final_flow_tenant_then_resilience_then_admission_inside_physical` — `LLMRateLimitError` in worker thread (flake/environment); 77 other tests passed |
| W2-C retry containment | **PASS** (with thread warning) | Rate-limit path exercised |

**Classification:** W4-A/C/D and W2 admission unit tests **do not** prove strict production tool composition mandates boundary. W2 final matrix failure is **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** for that single test on this run; does not remove the production wiring gap.

---

## 12. Applicable FRZ criteria (all remain OPEN)

| ID | Relevance to this ADR |
|----|------------------------|
| FRZ-REL-08 | Tool production admission not explicit in host profile / composition |
| FRZ-REL-09 | Pool queue unprotected when boundary omitted |
| FRZ-REL-10 | No strict fail-closed materialization for missing tool admission policy |
| FRZ-REL-11 | LLM rate/retry orthogonal; tool bypass does not satisfy freeze |

Supporting (not closed by this task): FRZ-REL-04..07.

---

## 13. STOP conditions

- **STOP — ROADMAP STATE CHANGED** if `HARNESS-W4` ≠ CURRENT before implementation child.
- **STOP — W4 BASELINE CHANGED** if a concurrent commit mandates boundary on all production roots without the approved profile extension (re-audit required).
- **STOP — ARCHITECTURE DECISION REQUIRED** while §8 remains PROPOSED (maintainer rejection of `ReliabilityProfile` extension → Outcome C; do not implement ad hoc owners).

---

## 14. Required implementation child scope (after approval)

1. Approve typed field on `ReliabilityProfile` + contract value object (no magic capacities).
2. `materialize_tool_dependency_attempt_boundary(env)` (or equivalent) in `applications/_shared/`.
3. Inject boundary in: `declarative_tool_wiring.py`, `runtime_context.py`, `execution_bound_catalog_tool_composition.py`.
4. `production_mode=True` + strict: fail closed if boundary required but not materialized.
5. Structural tests: all `build_production_runtime_tool_invoker` production roots; behavioral overload/reject tests mirroring W2 units.
6. Lifecycle: boundary shutdown via existing `RuntimeToolInvoker.close` / reconfiguration transfer.

---

## 15. No-implementation statement

This ADR records evidence and a **proposed** configuration owner only. No production runtime behavior, capacity values, managers, or composition changes were made in the HARNESS-W4-R1-P0 audit commit.

---

## 16. Parent roadmap recommendation (do not edit roadmap in P0)

| Item | Value |
|------|-------|
| `HARNESS-W4` current | CURRENT |
| `HARNESS-W4` recommended | **BLOCKED** (pending R1 implementation after architecture approval) |
| Child | `HARNESS-W4-R1` Production Tool Boundedness & Admission Closure |

Reason: strict production tool composition lacks mandatory dependency admission / queue protection required by W4 and FRZ-REL-08..10.
