# S24-GAP-02-P0 — Qualified Marketplace Tool Handoff Architecture Lock

## 1. Metadata and baseline

| Field | Value |
| ----- | ----- |
| **Task** | `S24-GAP-02-P0` (architecture lock — no implementation) |
| **Pre-audit baseline** | `4958c7e4bae6d18308426c6dc70d6595d67a4d5f` |
| **Lock audit HEAD** | `38b1e1d115302cabb57b82bec66e1c73764c38ea` |
| **Branch** | `development` (`HEAD == origin/development` at lock time) |
| **Diff since pre-audit** | Qualification harness / roadmap docs only — **no** Marketplace handoff, UCA, ToolRuntime, or EE production changes |
| **Artifact role** | Closed-world design record before `S24-GAP-02-P1` implementation |
| **Status** | **ARCHITECTURE LOCKED — Class A extension path** |

---

## 2. Problem

Canonical UCA Marketplace gap acquisition already ends in `domain_handoff_reference = handoff://<handoff_id>`. After Capability Qualification and binding, there is **no** production Tool-domain path that:

1. Stages the Marketplace-selected exact release **without** ToolRegistry activation,
2. Qualifies and binds `DOMAIN_HANDOFF_REFERENCE` for Marketplace Tools,
3. Materializes the exact qualified release **only after** `QUALIFIED` and Execution admission,
4. Invokes through **canonical ToolRuntime** (not `ToolMarketplaceAcquisitionBridge` → `DynamicToolAcquisitionPort.acquire`).

---

## 3. Root cause

Marketplace and Tool domains already support discover → select → governed handoff and exact release resolution → host activation respectively. The gap is the **missing typed seam** between UCA handoff reference and post-qualification executable Tool execution under Execution Engine identity — not missing Marketplace discovery.

**Hard rule (unchanged):** Marketplace acquisition MUST NOT mutate ToolRegistry before Capability Qualification outcome is `QUALIFIED`.

`ToolMarketplaceAcquisitionBridge` (`intergrax/marketplace/handoff/adapters/tool_acquisition_bridge.py`) remains valid for **non-UCA** Marketplace lifecycle flows but MUST NOT be used as the UCA acquisition consumer (activation before qualification).

---

## 4. Canonical owners

| Concern | Owner |
| ------- | ----- |
| Capability gap | Capability Catalog / canonical recovery |
| Marketplace discovery / selection | Marketplace |
| Marketplace handoff envelope | Marketplace (`CapabilityHandoffEnvelope`, ME-10 orchestrator) |
| Staged Tool release record | **Tool domain** (`MarketplaceQualifiedToolStage` + repository) |
| Capability qualification | Capability Qualification service + provider registry |
| Tool qualification facts | Tool-domain `MarketplaceToolCapabilityQualificationProvider` |
| Qualified binding | `QualifiedCapabilityBindingService` + Tool `MarketplaceToolQualifiedCapabilityBindingProvider` |
| Execution lifecycle | Execution Engine |
| Execution identity | ExecutionIdentityAuthority |
| Tool activation / materialization | Tool domain (post-qualification, in execution handler) |
| Tool invocation | ToolRuntime (`RuntimeToolInvoker` / `ExecutionBoundCatalogToolInvoker`) |
| Authority | Canonical Governance / Collaborative authority |
| Business responsibility | Autonomous Work / application |
| Proof | Scenario proof layer |

---

## 5. Existing reused contracts (no semantic change)

| Contract | Location |
| -------- | -------- |
| `QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE` | `intergrax/contracts/capability_qualification/qualified_subject.py` |
| `CapabilityHandoffEnvelope`, `CapabilityHandoffConsumer`, `CapabilityHandoffDeliveryAdmission`, `CapabilityHandoffTraceEvidenceConsumer` | `intergrax/contracts/marketplace/handoff_traceability.py` |
| `MarketplaceDiscoveryHandoffOrchestrator` | `intergrax/marketplace/handoff_traceability/orchestrator.py` |
| UCA gap handoff reference | `intergrax/marketplace/acquisition/gap_acquisition_service.py` (`handoff://`) |
| `CapabilityQualificationProvider` + `CapabilityQualificationService` | `intergrax/contracts/capability_qualification/provider.py`, `intergrax/capability_qualification/qualification_service.py` |
| `QualifiedCapabilityBindingProvider` + `QualifiedCapabilityBindingService` | `intergrax/contracts/capability_qualification/qualified_capability_binding.py`, `intergrax/capability_qualification/qualified_capability_binding_service.py` |
| `QualifiedCapabilityExecutionBindingHandler` + registry | `intergrax/runtime/execution/qualified_capability_execution_handlers.py` |
| Governed Task binding for dispatch | `intergrax/runtime/execution/governed_task_scoped_qualified_capability_execution_dispatch.py` |
| Catalog ToolRuntime gateway | `intergrax/contracts/execution_bound_catalog_tool_invocation.py`, `intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py` |
| Reference CodeCraft qualified path (pattern only) | `intergrax/runtime/codecraft/wiring_bound_capability_execution.py` |

---

## 6. Frozen surfaces — NO CHANGE in GAP-02

- `QualifiedCapabilitySubjectKind` (no new kind for Marketplace Tool)
- `QualifiedCapabilityBindingProvider` public semantics
- `WorkerQualifiedCapabilityExecutionRequest` fields / semantics
- UCA acquisition success evidence shape (`domain_handoff_reference`)
- `CapabilityHandoffEnvelope` schema
- Execution Engine lifecycle ownership
- GCF / UCA freeze invariants (`GOVERNED_CAPABILITY_FULFILLMENT.md`)

---

## 7. Target flow (order frozen)

```text
Canonical capability gap
    ↓
MarketplaceGapAcquisitionService
    ↓
governed Marketplace selection
    ↓
CapabilityHandoffEnvelope
    ↓
TOOL DOMAIN STAGING CONSUMER (CapabilityHandoffConsumer)
    ↓
MarketplaceQualifiedToolStage persisted (immutable)
    NO ToolRegistry activation
    NO executable materialization
    ↓
domain_handoff_reference = handoff://<handoff_id>
    ↓
Capability Qualification
    ↓
MarketplaceToolCapabilityQualificationProvider
    reads staged record by handoff_id
    ↓
QUALIFIED (evidence preserves domain_handoff_reference)
    ↓
MarketplaceToolQualifiedCapabilityBindingProvider
    identity continuity acquisition → qualification → binding
    ↓
QualifiedCapabilityExecutionTarget (opaque)
    ↓
Execution Engine admission + identity
    ↓
MarketplaceToolQualifiedCapabilityExecutionHandler
    (handler registry by binding_provider_id)
    ↓
exact qualified Tool materialization / activation
    ↓
QualifiedToolInvocationResolver → ExecutionBoundCatalogToolInvokeRequest
    ↓
ToolRuntime (ExecutionBoundCatalogToolInvoker → RuntimeToolInvoker)
    ↓
exact Tool invocation
```

---

## 8. Staging model

### P0-A audit answers

**Q1 — Public `handoff_id` → `CapabilityHandoffEnvelope` read contract?**

**NO.**

Evidence:

- `CapabilityHandoffTraceEvidenceConsumer` (`intergrax/contracts/marketplace/handoff_traceability.py`) exposes only `record_handoff` — no read port.
- `CapabilityHandoffDeliveryAdmission` exposes `reserve` / `mark_delivered` / `mark_delivery_failed` — no envelope lookup on the protocol.
- `InMemoryCapabilityHandoffTraceEvidenceConsumer.get` (`intergrax/marketplace/handoff_traceability/evidence.py`) is implementation-specific and not a production contract.

**Q2 — Restart-safe handoff persistence for Tool staging?**

**STAGING REPOSITORY REQUIRED.**

ME-10 in-memory admission/evidence is process-local reference material only. Tool-domain staging MUST NOT rely on Marketplace private stores or trace evidence consumers for qualification reads.

### Approved Tool-owned staging design

**Record:** `MarketplaceQualifiedToolStage` (immutable, typed, frozen Pydantic or dataclass per repo convention)

Minimum identity:

- `handoff_id`
- `tenant_id`
- `selected_release` (`CapabilityReleaseIdentity`)
- `discovery_correlation_id`
- `selection_id`
- `consumer_target == TOOL_DOMAIN`
- `recorded_at`

Exclusions: no executable handle, no ToolRegistry entry, no credentials, no authority token, no mutation beyond idempotent `stage`.

**Port:** `MarketplaceQualifiedToolStageRepository`

Conceptual operations:

- `stage(record)` — idempotent identical replay; conflict if same `handoff_id` with different payload; tenant-safe
- `get(handoff_id, tenant_id)` — fail closed on missing / tenant mismatch

**Consumer:** `tool_qualification_staging_consumer.py` under `intergrax/marketplace/handoff/adapters/` implements `CapabilityHandoffConsumer`, writes stage only, does not call `DynamicToolAcquisitionPort`.

**Handoff ID parsing:** `domain_handoff_reference` prefix `handoff://` from `gap_acquisition_service._domain_handoff_reference` — strip prefix to obtain `handoff_id` for repository lookup (no UCA contract change).

---

## 9. Qualification model

### P0-C audit

**Q3 — Production qualification provider for Marketplace Tool `DOMAIN_HANDOFF_REFERENCE`?**

**NO.**

Evidence: `CapabilityQualificationProvider` is defined in `intergrax/contracts/capability_qualification/provider.py`; production registry is populated only via composition — no Tool/Marketplace handoff provider under `intergrax/` outside tests.

**Approved extension:** `MarketplaceToolCapabilityQualificationProvider` implementing existing `CapabilityQualificationProvider`:

- `supports()` — technical compatibility only (qualified subject kind + staged TOOL handoff)
- `qualify()` — verify staged record integrity; preserve `domain_handoff_reference`; typed evidence; **no** execution, **no** ToolRegistry activation, **no** authority minting
- Fail closed: missing / conflicting / incomplete / wrong tenant / non-TOOL consumer target

Qualification integrity rules already require acquisition ↔ qualification handoff continuity (`intergrax/contracts/capability_qualification/qualification_integrity.py`).

---

## 10. Binding model

### P0-D audit

**YES** — existing `QualifiedCapabilityBindingProvider`, `QualifiedCapabilityBindingService`, and `DOMAIN_HANDOFF_REFERENCE` are sufficient **without semantic modification**.

Reference binding implementation pattern: `CodeCraftQualifiedCapabilityBindingProvider` (`intergrax/runtime/codecraft/qualified_capability_binding_provider.py`) — subject kind gate, opaque `QualifiedCapabilityExecutionTarget`, no execution.

**Approved provider:** `MarketplaceToolQualifiedCapabilityBindingProvider`

- Only `DOMAIN_HANDOFF_REFERENCE`
- Requires `QUALIFIED`
- Resolves staged TOOL subject; validates identity chain
- No execution, no Marketplace rediscovery, no re-qualification, no UCA mutation

---

## 11. Execution semantics

### P0-E audit — Q4

**Q4 — Where does canonical Tool execution get operation / input arguments for `task_id`-correlated capability execution?**

**Answer (canonical production paths only):**

1. **Execution admission** carries `task_id`, `worker_need_id`, and opaque `execution_target` in `WorkerQualifiedCapabilityExecutionRequest` (`intergrax/contracts/autonomous_work/worker_qualified_capability_resume.py`) — **not** tool operation or input payload.

2. **Before runtime**, `GovernedTaskScopedQualifiedCapabilityExecutionDispatchService` resolves the live governed `Task` from `ActiveTaskRegistry` and binds it via `bind_governed_execution_task` (`intergrax/runtime/execution/governed_task_scoped_qualified_capability_execution_dispatch.py`).

3. **ToolRuntime invocation** requires a typed `ExecutionBoundCatalogToolInvokeRequest` (`tool_id`, `input: BaseModel`, `tenant_id`, `task_id`, `run_id`, `agent_id`, `step_id`, …) per `intergrax/contracts/execution_bound_catalog_tool_invocation.py`.

4. **Reference qualified execution (CodeCraft):** `CodeCraftQualifiedCapabilityExecutionHandler` → `WiringCodeCraftBoundCapabilityExecution` builds `ExecutionBoundCatalogToolInvokeRequest` from **domain-owned session state** (craft code, sandbox wiring), not from fields on `WorkerQualifiedCapabilityExecutionRequest`. ToolRuntime still receives governed `Task` via `peek_governed_execution_task()` inside `RuntimeToolInvoker` / `NexusExecutionBoundCatalogToolInvoker` (`intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py`, `intergrax/runtime/nexus/tools/invoker.py`).

5. **Worker capability need (AW durable):** `WorkerCapabilityNeed.required_operations` (`intergrax/contracts/autonomous_work/capability_acquisition.py`) records required tool logical operations for discovery/gap context. Durable read: `WorkerRecoveryObstacleCapabilityNeedReadPort` (`intergrax/autonomous_work/worker_recovery_capability_fulfillment_episode_context_ports.py`). `worker_need_id` is derivable and encodes `worker_instance_id`, `obstacle_id`, and `recovery_decision_id` via `derive_worker_capability_need_id`.

6. **Task payload:** governed `Task.message`, `Task.context`, and `Task.metadata` (`intergrax/runtime/task/task.py`) supply obstacle/work context for invocation materialization — consumed when ToolRuntime and MSE gates run under active governed task.

**Verdict: B — NEW TOOL-DOMAIN INPUT-RESOLUTION PORT NEEDED**

Existing data exists across governed `Task` + durable `WorkerCapabilityNeed` + staged `CapabilityReleaseIdentity`, but there is **no** reusable Tool-owned contract to produce `ExecutionBoundCatalogToolInvokeRequest` (or equivalent ToolRuntime material) without changing EE/UCA request types.

**Approved port (new contract only):** `QualifiedToolInvocationResolver` in `intergrax/contracts/tools/qualified_tool_invocation.py`

Responsibilities (implementation in Tool domain, P3):

- Inputs: qualified staged release, active governed `Task`, resolved `WorkerCapabilityNeed` (from `worker_need_id` + read port), active execution identity (`run_id`, `step_id` policy)
- Output: `ExecutionBoundCatalogToolInvokeRequest` (or narrow wrapper) for `ExecutionBoundCatalogToolInvoker`
- Must not mint authority, own lifecycle, or mutate UCA/EE contracts

**Approved handler:** `MarketplaceToolQualifiedCapabilityExecutionHandler` under canonical `QualifiedCapabilityExecutionBindingHandler` registry — same pattern as `CodeCraftQualifiedCapabilityExecutionHandler` (`intergrax/runtime/codecraft/qualified_capability_execution_handler.py`).

---

## 12. ToolRuntime invocation source (summary)

| Material | Source | Owner |
| -------- | ------ | ----- |
| `tool_id` / exact release | Staged `CapabilityReleaseIdentity` + post-qualification activation | Tool domain |
| Operation selection | `WorkerCapabilityNeed.required_operations` (+ release capability contract) | AW need store + Tool resolver |
| Input arguments | Governed `Task` (+ need `evidence_refs` where applicable) | Task registry + Tool resolver |
| `tenant_id`, `task_id`, `run_id`, `step_id` | Execution identity + dispatch request + EE context | Execution Engine |
| Runtime admission / MSE | `RuntimeToolInvoker` with `peek_governed_execution_task()` | ToolRuntime / governance |

---

## 13. Persistence / restart semantics

| Store | Restart-safe | GAP-02 action |
| ----- | ------------ | ------------- |
| ME-10 handoff admission (in-memory ref) | No | Not used for Tool qualification reads |
| ME-10 trace evidence consumer | No | Observability only |
| `MarketplaceQualifiedToolStageRepository` | **Must be pluggable durable** | **P1 implement** |
| AW obstacle need repositories | Yes (PostgreSQL / document store adapters exist) | Reuse read port in resolver |

Qualification and binding MUST fail closed if staged record missing after restart.

---

## 14. Idempotency and concurrency

- **Staging:** identical envelope replay → same record; conflicting payload → `CapabilityHandoffIdentityConflictError` semantics aligned with ME-10 admission
- **Qualification:** service-level idempotency via existing qualification request identity
- **Binding:** `binding_operation_id` cache pattern per CodeCraft provider
- **Execution:** `execution_request_id` derivation unchanged; handler activation idempotent per operation id
- **Duplicate binding / execution:** map to `CONFLICT` / `REJECTED` — no silent double activation

---

## 15. Authority and security

All paths fail closed. No silent fallback to direct Tool invoke, registry mutation, or Marketplace bridge acquisition.

Authority flows only through admitted governance identity and collaborative authority on existing UCA resume/dispatch requests — Tool staging and resolver MUST NOT mint scopes.

---

## 16. Failure matrix

| # | Condition | Disposition |
| - | --------- | ----------- |
| 1 | Handoff missing | Qualification/binding/execution: `FAILED` / `NOT_SUPPORTED` — closed |
| 2 | Handoff identity conflict | Staging conflict error; downstream unavailable |
| 3 | Wrong tenant | Repository/get: reject |
| 4 | Non-TOOL handoff | Consumer rejects; qualification `supports()` false |
| 5 | Incomplete release identity | Staging/qualification reject |
| 6 | Package/version/digest mismatch | Qualification or activation reject |
| 7 | Qualification rejected | No activation |
| 8 | Qualification provider unavailable | `NO_PROVIDER` / `UNAVAILABLE` |
| 9 | Binding provider unavailable | `NO_PROVIDER` / `UNAVAILABLE` |
| 10 | Staging unavailable | Acquisition consumer `UNAVAILABLE` |
| 11 | Activation unavailable | Execution `UNAVAILABLE` |
| 12 | Execution handler unavailable | `execution_handler_unavailable` |
| 13 | ToolRuntime rejection | Mapped EE disposition — no bypass |
| 14 | Duplicate binding replay | Idempotent `BOUND` with same target |
| 15 | Duplicate execution replay | EE/idempotency rules |
| 16 | Authority denied | `REJECTED` / governance errors |
| 17 | Stale/tampered handoff reference | Integrity check fail — closed |

---

## 17. Change classification

| Classification | **Class A — safe extension** |
| -------------- | ------------------------------ |
| Execution verdict | **B** — new Tool-domain `QualifiedToolInvocationResolver` contract only |
| UCA reopen | **NO** |
| Class C triggers | None identified |

Rationale: new Tool-domain providers, staging repository, and resolver implement against frozen SPIs and registries; no frozen UCA/EE payload semantics change.

---

## 18. Future implementation scope (closed world)

### Contracts

- `intergrax/contracts/tools/marketplace_qualified_capability.py` — stage model + repository protocol
- `intergrax/contracts/tools/qualified_tool_invocation.py` — resolver protocol (Verdict B)

### Tool domain

- `intergrax/tools/marketplace_qualified_capability_staging.py`
- `intergrax/tools/marketplace_qualified_capability_qualification_provider.py`
- `intergrax/tools/marketplace_qualified_capability_binding_provider.py`
- `intergrax/tools/marketplace_qualified_capability_execution_handler.py`

### Marketplace adapter

- `intergrax/marketplace/handoff/adapters/tool_qualification_staging_consumer.py`

Leaf-module imports only; no package-root re-export unless an existing canonical package API requires it.

**Explicitly out of scope for implementation:** changing `ToolMarketplaceAcquisitionBridge` behavior for legacy flows; scenario-local shortcuts.

---

## 19. Future implementation gates

### Contract gates

- Immutable typed staging; no `Any` semantic payload; no reflection routing; no private cross-class reads

### Acquisition gates

- Marketplace produces handoff; staging consumer only; ToolRegistry activation count **0**

### Qualification gates

- Staged subject required; failure → activation count **0**

### Binding gates

- Same qualified subject; tenant match; no execution during bind

### Execution gates

- Handler via `binding_provider_id`; activation only after `QUALIFIED`; digest/version preserved; ToolRuntime only; Tool execution count **0** before qualification success

### Regression gates

- CodeCraft UCA path green; direct reuse green; ME-14/ME-17 Marketplace semantics green; no GCF invariant changes

---

## 20. Exit criteria (P0)

| Criterion | Result |
| --------- | ------ |
| Q1 answered with evidence | PASS |
| Q2 staging requirement stated | PASS |
| Q3 provider gap + extension approved | PASS |
| Q4 answered with canonical paths | PASS |
| Binding SPI sufficient | PASS |
| No Class C / STOP condition | PASS |
| Design record complete | PASS |
| Production code changed in P0 | PASS (none) |
| UCA semantic change required | FAIL (not needed — lock proceeds) |

---

## 21. No-UCA-reopen conclusion

GAP-02 closes via **Class A** Tool-domain extensions and one new Tool invocation resolver contract. No modification to frozen UCA resume types, binding SPI semantics, or Execution Engine ownership. **Implementation may proceed to P1 after independent audit of this commit.**

---

## 22. Materialization timing proof (test gates for P3/CERT)

Negative guarantees (automated in `S24-GAP-02-CERT`):

```text
before QUALIFIED:
  ToolRegistry activation count == 0
  Tool execution count == 0

qualification failure:
  ToolRegistry activation count == 0
  Tool execution count == 0

after QUALIFIED + EE admission only:
  activation allowed once for exact staged release
  invocation only via ToolRuntime
```

Ordering evidence: staging consumer runs in Marketplace delivery before UCA returns handoff reference; qualification provider reads stage; binding references qualified subject only; handler runs under `QualifiedCapabilityExecutionBindingHandlerRegistry` after `GovernedTaskScopedQualifiedCapabilityExecutionDispatchService` binds task.
