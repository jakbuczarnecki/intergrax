# S24-GAP-02-P0 — Qualified Marketplace Tool Handoff Architecture Lock

## 1. Metadata and baseline

| Field | Value |
| ----- | ----- |
| **Task** | `S24-GAP-02-P0` (architecture lock) · `S24-GAP-02-P0-R1` (typed invocation material boundary correction) · **`S24-GAP-02-P2-P0`** (tenant-safe qualification subject resolution) · **`S24-GAP-02-P2-P0-R1`** (tenant-distinct Marketplace handoff identity correction) · **`S24-GAP-02-P3-P0`** (canonical invocation intent propagation) · **`S24-GAP-02-P3-P0-R1`** (deterministic pre-EE tool execution intent recording correction) |
| **Pre-audit baseline** | `4958c7e4bae6d18308426c6dc70d6595d67a4d5f` |
| **Lock audit HEAD** | `d28b6f81cd721ca0ab2bbac002a78421073b9735` (P0) · **P0-R1** invocation boundary · **P2-P0** tenant resolution · **P2-P0-R1** handoff identity + association model (see §23) · **P3-P0** invocation intent (see §25) · **P3-P0-R1** pre-EE intent hook (see §25.2) |
| **P3-P0 session HEAD** | `d3a85c83910c858a78fc8b243cd42b73e3491f29` (P3-P0 lock commit) |
| **P3-P0-R1 audit baseline** | `d3a85c83910c858a78fc8b243cd42b73e3491f29` — code evidence from **committed** `development` at correction time (local WIP under `intergrax/runtime/execution/**` is not architecture evidence) |
| **Branch** | `development` |
| **Diff since pre-audit** | Qualification harness / roadmap docs only — **no** Marketplace handoff, UCA, ToolRuntime, or EE production changes (P3-P0: **docs only**) |
| **Artifact role** | Closed-world design record before `S24-GAP-02-P1` implementation; **P2-P0 / P2-P0-R1** extend lock before `S24-GAP-02-P2` providers; **P3-P0** extends lock before `S24-GAP-02-P3` execution handler |
| **Status** | **ARCHITECTURE LOCKED — Class A/B extension path** · **P3 invocation intent: Class B (§25)** · **UCA REOPEN: NO** · **EE REOPEN: NO** |

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
| Business Tool input material | Application / domain (`QualifiedToolInvocationMaterialProvider` implementation) |
| Tool invoke request assembly | Tool domain (`QualifiedToolInvocationResolver`) |
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
    resolves tenant via context resolver → staged record (tenant_id + handoff_id)
    ↓
QUALIFIED (evidence preserves domain_handoff_reference)
    ↓
MarketplaceToolQualifiedCapabilityBindingProvider
    identity continuity acquisition → qualification → binding
    ↓
QualifiedCapabilityExecutionTarget (opaque)
    ↓
QualifiedMarketplaceToolExecutionIntent (durable, Tool-owned — **P3-P0**)
    keyed by execution_request_id; selected_operation + correlation only
    ↓
Execution Engine admission + identity
    ↓
MarketplaceToolQualifiedCapabilityExecutionHandler
    lookup intent by execution_request_id; integrity vs bound dispatch
    (handler registry by binding_provider_id)
    ↓
exact qualified Tool materialization / activation
    ↓
QualifiedToolInvocationMaterialProvider → typed invocation material (immutable contract / BaseModel)
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

**Handoff ID parsing:** `domain_handoff_reference` prefix `handoff://` from gap acquisition — strip prefix to obtain `handoff_id`. **P2+** Marketplace gap production ids use **v2** tenant-distinct derivation (§23); qualification resolves tenant via context association + integrity checks before P1 `get(tenant_id, handoff_id)` (no UCA contract change).

**Repository read (P1 actual SPI):** `MarketplaceQualifiedToolStageRepository.get(tenant_id=…, handoff_id=…)` — tenant scope is mandatory; global `get_by_handoff_id` is forbidden.

---

## 23. P2 Tenant-Safe Qualification Subject Resolution

**Task:** `S24-GAP-02-P2-P0` · **`S24-GAP-02-P2-P0-R1`** · **Status:** **LOCKED** · **Verdict:** **Class B — tenant-distinct handoff identity + Tool-domain context resolver; NO UCA reopen**

### Problem (post-P1)

P1 introduced tenant-scoped durable staging (`MarketplaceQualifiedToolStage.tenant_id` + partitioned `ConditionalDocumentStore`). Frozen UCA-4 qualification dispatch still exposes only `domain_handoff_reference` (via nested `CapabilityAcquisitionResult` / evidence) and coordination ids — **no `tenant_id`**. A qualification provider cannot call `repository.get(tenant_id, handoff_id)` safely without a **durable, tenant-known association** created when the tenant is authoritative (handoff delivery / staging), not by inferring tenant from AW private state or a global stage scan.

### Root cause corrected (P2-P0-R1)

P2-P0 correctly forbade **global** `get_by_handoff_id` on the P1 stage repository and rejected inferring tenant from UCA ids alone. It **incorrectly** proposed a **global association primary key** on `acquisition_request_id` with **CONFLICT** when two tenants share the same acquisition id. That models a **non-existent global uniqueness** invariant: `derive_capability_acquisition_request_id` has **no** tenant dimension, so multi-tenant qualification must **not** require tenant-independent acquisition-id exclusivity.

**Approved fix:** **tenant-aware deterministic opaque Marketplace handoff identity** (`marketplace-gap-handoff:v2:…`) so the same `operation_id` under different tenants yields **different** `handoff_id` values; association lookup is by **tenant-distinct** `handoff_id`, not by acquisition id alone.

### P2-P0 audit — Q1–Q5 (unchanged facts; R1 updates Q4)

| Q | Verdict | Evidence |
| - | ------- | -------- |
| **Q1 — Canonical tenant source on production qualification path?** | **NO** (for `CapabilityQualificationProvider.qualify(CapabilityQualificationRequest)`) | `CapabilityQualificationRequest` (`intergrax/contracts/capability_qualification/qualification_request.py`) has no `tenant_id`. `CapabilityAcquisitionRequest`, `CapabilityAcquisitionResult`, `CapabilityAcquisitionEvidence` likewise omit tenant. `WorkerCapabilityNeed` / `WorkerCapabilityAcquisitionRequest` (`intergrax/contracts/autonomous_work/capability_acquisition.py`) carry `worker_instance_id` but not tenant. `WorkerCapabilityRecoveryCoordinator` builds `CapabilityQualificationRequest` from acquisition result only (`intergrax/autonomous_work/worker_capability_recovery_coordinator.py`). **Tenant is typed and validated only at Marketplace handoff staging:** `CapabilityHandoffEnvelope.tenant_id` → `ToolQualificationStagingConsumer` (`intergrax/marketplace/handoff/adapters/tool_qualification_staging_consumer.py`). That tenant is **not** on the frozen qualification request surface. |
| **Q2 — Acquisition / handoff identity globally tenant-unique?** | **NO** (UCA ids) · **YES** (after v2 handoff derivation) | `derive_capability_acquisition_request_id(gap_id, request_nonce)` has no tenant dimension. Legacy `handoff_id = marketplace-gap-handoff:{operation_id}` is **not** tenant-distinct. **P2 implementation** MUST emit/consume **v2** handoff ids via `derive_marketplace_gap_handoff_id(tenant_id, operation_id)` (below). P1 `get(tenant_id, handoff_id)` remains mandatory; **no** global stage lookup. |
| **Q3 — Production-safe `identity → tenant_id` resolver port on UCA ids?** | **NO** | No protocol maps `worker_need_id`, `recovery_decision_id`, `acquisition_request_id`, or `qualification_request_id` → `tenant_id` for qualification. AW episode context resolves tenant for fulfillment — **not** exposed to `CapabilityQualificationProvider`, and MUST NOT be parsed as a qualification-core bypass. |
| **Q4 — Class B without frozen `CapabilityQualificationRequest` change?** | **CLASS B — YES** | New Tool-domain contracts/providers plus **production handoff-id generation change** (tenant-bound v2 identity) without frozen UCA contract mutation. Conservative **Class B** gates for the full P2 track. |
| **Q5 — Requires `tenant_id` on frozen UCA contracts?** | **NO — not required for correct design** | Tenant scope is carried in Tool-owned association + staged record; UCA evidence shape unchanged (`handoff://<handoff_id>` only). **Not Class C.** |

### Tenant source (frozen)

Canonical tenant at staging:

```text
MarketplaceQueryContext.tenant_id
    → CapabilityHandoffEnvelope.tenant_id
```

Envelope already validates `envelope.tenant_id == discovery_trace.marketplace_query_context.tenant_id`. **Do not** resolve tenant via AW private state.

### Marketplace gap handoff identity (frozen — P2-P0-R1)

Conceptual helper (Tool/Marketplace implementation; contracts may expose as pure function):

```text
derive_marketplace_gap_handoff_id(
    *,
    tenant_id: str,
    operation_id: str,
) -> str
```

Semantics: deterministic; opaque; tenant-bound; stable across retries; **no** Python `hash()`; **no** random UUID; **no** raw tenant leakage in the final id.

Canonical derivation:

```text
domain_separator = "intergrax.marketplace-gap-handoff.v2"

digest = SHA-256(
    UTF8(
        domain_separator
        + "\0"
        + tenant_id
        + "\0"
        + operation_id
    )
)

handoff_id = "marketplace-gap-handoff:v2:" + lowercase_hex(digest)
```

If the repo already exposes a stable SHA-256 helper with identical properties, P2 implementation MAY reuse it — **no** new crypto abstraction without need.

**Invariants:**

```text
same tenant_id + same operation_id → same handoff_id   (idempotent retry)
same operation_id + different tenant_id → different handoff_id
```

`operation_id` on the Marketplace gap path equals `CapabilityAcquisitionRequest.request_id` / acquisition `request_id` used in qualification cross-checks.

**Domain reference (unchanged UCA evidence):**

```text
domain_handoff_reference = handoff://<handoff_id>
```

Forbidden on frozen acquisition evidence: `tenant_id` query params, extra metadata fields, or raw tenant embedded in the public URI.

### Approved solution shape (frozen for P2)

```text
CapabilityHandoffEnvelope (tenant_id known, validated)
    ↓
ToolQualificationStagingConsumer
    handoff_id = derive_marketplace_gap_handoff_id(tenant_id, operation_id)
    ↓
MarketplaceQualifiedToolStageRepository.stage (tenant partition)
    +
MarketplaceQualifiedToolStageContextAssociationRepository.record (same consumer flow)
    ↓
domain_handoff_reference = handoff://<handoff_id>
    ↓
CapabilityQualificationRequest (unchanged — no tenant_id)
    ↓
MarketplaceToolCapabilityQualificationProvider
    ↓
parse domain_handoff_reference (Tool-domain strict helper)
    ↓
MarketplaceQualifiedToolStageContextResolver
    get_by_handoff_id(handoff_id)  # allowed: handoff_id is tenant-distinct by construction
    ↓
MarketplaceQualifiedToolStageContext { tenant_id, handoff_id, acquisition_request_id }
    ↓
integrity: acquisition_request_id + re-derive handoff_id
    ↓
MarketplaceQualifiedToolStageRepository.get(tenant_id=…, handoff_id=…)
```

**Hard rules:**

- **Forbidden:** global P1 stage `get_by_handoff_id`; weakening P1 `get(tenant_id, handoff_id)`; primary association keyed only by `acquisition_request_id`; global acquisition-id exclusivity across tenants; guessing `tenant_id` from `gap_id` / `correlation_id` / AW stores; mutating frozen UCA evidence or qualification request semantics.
- **Allowed:** `ContextAssociationRepository.get_by_handoff_id(handoff_id)` **only** because v2 `handoff_id` is cryptographically tenant-distinct — this is **not** P1 global stage lookup.
- **Association write timing:** same handoff consumer invocation as `stage`, from the same authoritative envelope facts; fail-closed partial-write policy below (not a single cross-store transaction unless platform contract guarantees one).

### New contracts (conceptual SPI — P2 implement)

**Association (immutable, typed):**

```text
MarketplaceQualifiedToolStageContextAssociation
  handoff_id          # primary durable lookup identity
  tenant_id
  acquisition_request_id
```

**Context (immutable, typed):**

```text
MarketplaceQualifiedToolStageContext
  tenant_id
  handoff_id
  acquisition_request_id
```

**Resolver (read port):**

```text
MarketplaceQualifiedToolStageContextResolver
  resolve_for_qualification(
    *,
    domain_handoff_reference,
    acquisition_request_id,
    strategy_id,
  ) -> MarketplaceQualifiedToolStageContext
```

**Resolver flow (mandatory):**

```text
CapabilityQualificationRequest
    → domain_handoff_reference
    → strict parse → handoff_id
    → ContextAssociationRepository.get_by_handoff_id(handoff_id)
    → ctx { tenant_id, handoff_id, acquisition_request_id }
    → verify: ctx.acquisition_request_id == request.acquisition_request_id
    → verify: handoff_id == derive_marketplace_gap_handoff_id(ctx.tenant_id, request.acquisition_request_id)
    → MarketplaceQualifiedToolStageRepository.get(tenant_id=ctx.tenant_id, handoff_id=ctx.handoff_id)
```

Any mismatch → **INTEGRITY** / **FAIL CLOSED**.

**Association repository (write at staging, read at qualification):**

```text
MarketplaceQualifiedToolStageContextAssociationRepository
  record(association) -> WriteResult   # idempotent identical replay
  get_by_handoff_id(handoff_id) -> association | None
```

Persistence: **reuse `ConditionalDocumentStore`** (or same backend-neutral pattern as P1 staging) — **not** process-local dict/singleton. Same `handoff_id` with different immutable association payload → **CONFLICT** (no silent repair).

**Parsing:** one Tool-domain utility — strict `handoff://<handoff_id>` only; integrity via resolver re-derivation + acquisition id cross-check (no duplicated ad-hoc `split(":")` in providers).

**Owner:** **Tool domain** (contracts under `intergrax/contracts/tools/`, implementations under `intergrax/tools/`, staging consumer + minimal Marketplace gap handoff id touch). **Not** Capability Qualification core, AW coordinator, EE, or scenario proof.

### Identity collision model (mandatory — P2-P0-R1)

| Case | Expected behavior |
| ---- | ----------------- |
| Same `acquisition_request_id`, **different** `tenant_id` | **Different** v2 `handoff_id` each; **independent** stage + association; **NO CONFLICT**; both tenants may qualify their own stage |
| Same `tenant_id` + same `acquisition_request_id` | **Same** `handoff_id`; idempotent replay on stage + association |
| `handoff_id` ≠ `derive_marketplace_gap_handoff_id(tenant_id, acquisition_request_id)` | **INTEGRITY FAIL** (corrupt association or reference) |
| Same `handoff_id`, different association payload (`tenant_id` / `acquisition_request_id`) | **CONFLICT** |
| Same raw `handoff_id` across tenants (v2) | Should be **impossible** for correctly generated ids except cryptographic collision; still run **full integrity validation** on stored association |
| Same `handoff_id`, different `selected_release` | Staging `MarketplaceQualifiedToolStageConflictError` (P1); association must not mask stage conflict |
| Parsed `domain_handoff_reference` ≠ association `handoff_id` | Resolver **INTEGRITY** — fail-closed |
| Tenant A resolves tenant B stage | **FAIL CLOSED** — wrong partition / integrity / tenant mismatch on `get(tenant_id, handoff_id)` |

**Security invariant:** Tenant A MUST NEVER read Tenant B staged release — **tenant-scoped stage partition** + **tenant-distinct handoff_id** + **association integrity checks** + **no** global acquisition-id lock.

### Partial failure and retry (staging consumer — frozen)

Platform MUST NOT claim atomicity across two document-store writes unless the storage contract provides it.

**Qualification invariant:**

```text
qualification MUST NOT succeed unless BOTH
  staged MarketplaceQualifiedToolStage exists
  AND matching context association exists
```

| Partial state | Disposition |
| ------------- | ----------- |
| Stage created; association write failed | Consumer **FAILED / UNAVAILABLE**; retry may `ALREADY_STAGED_IDENTICAL` + create association and complete delivery |
| Association exists; stage missing | Qualification **FAIL CLOSED**; consumer retry must recreate missing stage |
| Either record conflicts on immutable fields | **BLOCKED / INTEGRITY** — never silent repair |

Idempotent replay MUST safely complete partially finished writes.

### Qualification provider expectation (P2 — after this lock)

`MarketplaceToolCapabilityQualificationProvider` receives **no** direct `tenant_id`. It MUST:

1. Gate `supports()` on strategy / `DOMAIN_HANDOFF_REFERENCE` technical compatibility.
2. Read exact `domain_handoff_reference` from acquisition evidence (unchanged UCA shape).
3. Call `MarketplaceQualifiedToolStageContextResolver` (not inline tenant guess; **no** global stage scan; **no** AW lookup).
4. Load `repository.get(tenant_id=ctx.tenant_id, handoff_id=ctx.handoff_id)`.
5. Validate staged TOOL release vs evidence; emit `CapabilityQualificationEvidence` preserving **exact** `domain_handoff_reference`.
6. Never activate Tool, bind, execute, or mint authority.

**Binding provider:** `MarketplaceToolQualifiedCapabilityBindingProvider` MAY use the same `domain_handoff_reference` → context resolver → exact staged release path; still **NO** Tool activation, **NO** ToolRegistry mutation, **NO** execution during bind.

### P2-P0-R1 classification

| Item | Value |
| ---- | ----- |
| **P2 tenant resolution + handoff identity** | **NEW TOOL-DOMAIN CONTEXT RESOLVER + v2 TENANT-DISTINCT HANDOFF ID** |
| **Class** | **B** (new Tool contracts/providers; production handoff-id generation change; frozen UCA untouched) |
| **UCA reopen** | **NO** |

Rationale: extension-shaped Tool-domain surface, but P2 implementation changes existing Marketplace gap handoff-id generation to remove multi-tenant identity defect without altering frozen UCA contracts, ownership, or authority — **Class B** gates apply to the full P2 track.

### P2 implementation test gates (frozen minimum — 15)

1. Same tenant + same acquisition → same handoff id.
2. Different tenant + same acquisition → different handoff id.
3. Deterministic across process restart.
4. Tenant id not visible in generated handoff id.
5. Association identical replay idempotent.
6. Same handoff id + different association → conflict.
7. Resolver validates acquisition id against association.
8. Resolver re-derives handoff id and checks integrity.
9. Corrupted tenant association → fail closed.
10. Association exists / stage missing → qualification fails.
11. Stage exists / association missing → qualification fails.
12. Two tenants using same acquisition id both qualify their own stage without conflict.
13. Tenant A cannot resolve tenant B stage.
14. UCA evidence schema unchanged.
15. No Tool activation before qualification success.

---

## 9. Qualification model

### P0-C audit

**Q3 — Production qualification provider for Marketplace Tool `DOMAIN_HANDOFF_REFERENCE`?**

**NO.**

Evidence: `CapabilityQualificationProvider` is defined in `intergrax/contracts/capability_qualification/provider.py`; production registry is populated only via composition — no Tool/Marketplace handoff provider under `intergrax/` outside tests.

**Approved extension:** `MarketplaceToolCapabilityQualificationProvider` implementing existing `CapabilityQualificationProvider`:

- `supports()` — technical compatibility only (qualified subject kind + staged TOOL handoff)
- `qualify()` — resolve tenant via **§23 context resolver**; verify staged record integrity; preserve `domain_handoff_reference`; typed evidence; **no** execution, **no** ToolRegistry activation, **no** authority minting
- Fail closed: missing / conflicting / incomplete / wrong tenant / non-TOOL consumer target / ambiguous association

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

1. **Execution admission** carries `task_id`, `worker_need_id`, and opaque `execution_target` in `WorkerQualifiedCapabilityExecutionRequest` (`intergrax/contracts/autonomous_work/worker_qualified_capability_resume.py`) — **not** tool operation or business input payload.

2. **Before runtime**, `GovernedTaskScopedQualifiedCapabilityExecutionDispatchService` resolves the live governed `Task` from `ActiveTaskRegistry` and binds it via `bind_governed_execution_task` (`intergrax/runtime/execution/governed_task_scoped_qualified_capability_execution_dispatch.py`). That binding is **runtime-internal** for governance/MSE admission — it is **not** a public Tool-contract input channel.

3. **ToolRuntime invocation** requires a typed `ExecutionBoundCatalogToolInvokeRequest` (`tool_id`, `input: BaseModel`, `tenant_id`, `task_id`, `run_id`, `agent_id`, `step_id`, …) per `intergrax/contracts/execution_bound_catalog_tool_invocation.py`.

4. **Reference qualified execution (CodeCraft):** `CodeCraftQualifiedCapabilityExecutionHandler` → `WiringCodeCraftBoundCapabilityExecution` builds `ExecutionBoundCatalogToolInvokeRequest` from **domain-owned session state** (craft code, sandbox wiring), not from fields on `WorkerQualifiedCapabilityExecutionRequest` and **not** by parsing governed `Task.message` / `Task.context` / `Task.metadata`. ToolRuntime may still use `peek_governed_execution_task()` inside `RuntimeToolInvoker` / `NexusExecutionBoundCatalogToolInvoker` for admission only (`intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py`, `intergrax/runtime/nexus/tools/invoker.py`).

5. **Worker capability need (AW durable):** `WorkerCapabilityNeed.required_operations` (`intergrax/contracts/autonomous_work/capability_acquisition.py`) identifies the **required logical capability operation** for discovery/gap/execution routing — **not** business invocation arguments. Durable read: `WorkerRecoveryObstacleCapabilityNeedReadPort` (`intergrax/autonomous_work/worker_recovery_capability_fulfillment_episode_context_ports.py`). `worker_need_id` is derivable via `derive_worker_capability_need_id`. Fields `required_data_domains`, `required_resource_refs`, and `evidence_refs` may carry typed context/evidence for application logic; they MUST NOT be treated as automatic Tool `input` payload or parsed by a generic platform resolver.

6. **Business invocation inputs** are owned by application/domain/host. The platform MUST materialize them through a **replaceable typed provider** (see §11.1) — never by reading `Task.message`, `Task.context`, or `Task.metadata` (`Any`) as the public semantic contract for Tool arguments, and never via string/dict inspection, reflection, or scenario-specific platform conditionals.

**Verdict B — new Tool-domain invocation-resolution extension required, with a separate typed invocation-material provider boundary.**

Two concerns are intentionally split:

```text
business invocation material  ≠  tool invocation resolution
```

Existing EE/UCA admission types, staged `CapabilityReleaseIdentity`, and durable `WorkerCapabilityNeed` supply **identity, operation selection, and correlation** — not a reusable, strongly typed business-input contract. Implementation requires **two** new Tool-domain contract surfaces (P3), without changing frozen UCA/EE request semantics.

### 11.1 Typed invocation material and resolver (frozen responsibilities)

**Contract-layer purity (mandatory):**

```text
intergrax/contracts/tools/**
MUST NOT import intergrax/runtime/**
```

New Tool contracts use only stable contract-layer types. If a design required a public contract importing runtime `Task`, that would be **STOP — architecture boundary violation**.

**Strong typing chain (mandatory):**

```text
business/application state
    → typed domain invocation model
    → QualifiedToolInvocationMaterialProvider
    → typed invocation material (immutable BaseModel — not dict[str, Any])
    → QualifiedToolInvocationResolver
    → concrete Tool input BaseModel inside ExecutionBoundCatalogToolInvokeRequest
    → ToolRuntime
```

Forbidden at any public semantic boundary: `dict[str, Any]` as invocation payload; magic metadata keys; `getattr`/reflection routing; generic parsing of `Task.message`; platform scenario-specific branches.

#### A. `QualifiedToolInvocationMaterialProvider` (application/domain-owned seam)

**Location (contracts):** `intergrax/contracts/tools/qualified_tool_invocation.py` (or a sibling leaf module if repo conventions favor split files — responsibilities frozen either way).

**Role:** Map **execution/business context identity** (correlation handles known to the handler/runtime adapter — e.g. `task_id`, `worker_need_id`, tenant, episode ids) to **typed invocation material** (immutable Pydantic `BaseModel` or equivalent frozen contract type).

- Replaceable per scenario/application; platform does not interpret business payload.
- MUST NOT return `dict[str, Any]` as the semantic contract.
- MAY be implemented in application or domain modules; handler wires a concrete provider.

Runtime `Task` MAY remain an **internal** source inside a runtime adapter that **calls** the provider — never as a dependency of `intergrax/contracts/tools/**`.

```text
Execution Runtime
    → resolves active Task internally (governance only)
    → application/domain QualifiedToolInvocationMaterialProvider
    → typed invocation material
    → QualifiedToolInvocationResolver
```

Not permitted:

```text
intergrax/contracts/tools → intergrax/runtime/task.Task
resolver → task.metadata["..."]
```

#### B. `QualifiedToolInvocationResolver` (generic Tool-domain)

**Location (contracts):** same module family as the material provider protocol.

**Inputs (typed only):**

- Exact qualified Tool identity / staged release (`CapabilityReleaseIdentity` or post-activation tool identity policy)
- Selected **required operation** (from `WorkerCapabilityNeed.required_operations` + release capability contract — operation id only, not arguments)
- **Typed invocation material** from the provider (immutable contract)
- Execution identity fields required by ToolRuntime (`tenant_id`, `task_id`, `run_id`, `step_id`, `agent_id` policy per existing gateway)

**Output:** `ExecutionBoundCatalogToolInvokeRequest` (or a typed intermediate contract if an existing layer boundary requires separation from the runtime gateway).

**Resolver MUST NOT:** read arbitrary Task metadata; know Asterion or other scenario semantics; know Marketplace internals; mint authority; execute the Tool; activate ToolRegistry.

**Approved handler:** `MarketplaceToolQualifiedCapabilityExecutionHandler` under canonical `QualifiedCapabilityExecutionBindingHandler` registry — same pattern as `CodeCraftQualifiedCapabilityExecutionHandler` (`intergrax/runtime/codecraft/qualified_capability_execution_handler.py`). Handler orchestrates: resolve release → post-qualification activation → **material provider** → **resolver** → ToolRuntime.

### 11.2 P3 orchestration shape (frozen)

```text
Execution Engine
    ↓
MarketplaceToolQualifiedCapabilityExecutionHandler
    ↓
resolve exact qualified Tool release
    ↓
post-qualification activation / materialization
    ↓
QualifiedToolInvocationMaterialProvider
    ↓
typed invocation material
    ↓
QualifiedToolInvocationResolver
    ↓
ExecutionBoundCatalogToolInvokeRequest
    ↓
ToolRuntime
```

---

## 12. ToolRuntime invocation source (summary)

| Material | Source | Owner |
| -------- | ------ | ----- |
| `tool_id` / exact release | Staged `CapabilityReleaseIdentity` + post-qualification activation | Tool domain |
| Operation selection | `WorkerCapabilityNeed.required_operations` (+ release capability contract) | AW need store + resolver (operation id only) |
| Business input arguments | `QualifiedToolInvocationMaterialProvider` → typed invocation material | Application / domain / host |
| Concrete Tool `input` BaseModel | `QualifiedToolInvocationResolver` (maps material + operation → gateway request) | Tool domain |
| `tenant_id`, `task_id`, `run_id`, `step_id` | Execution identity + dispatch request + EE context | Execution Engine |
| Runtime admission / MSE | `RuntimeToolInvoker` with `peek_governed_execution_task()` (runtime internal) | ToolRuntime / governance |

---

## 13. Persistence / restart semantics

| Store | Restart-safe | GAP-02 action |
| ----- | ------------ | ------------- |
| ME-10 handoff admission (in-memory ref) | No | Not used for Tool qualification reads |
| ME-10 trace evidence consumer | No | Observability only |
| `MarketplaceQualifiedToolStageRepository` | **Must be pluggable durable** | **P1 implement** |
| AW obstacle need repositories | Yes (PostgreSQL / document store adapters exist) | Reuse read port in **handler/runtime adapter** for operation selection; not for inferring Tool input |

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

Authority flows only through admitted governance identity and collaborative authority on existing UCA resume/dispatch requests — Tool staging, invocation-material provider contracts, and resolver MUST NOT mint scopes.

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
| Execution verdict | **B** — typed `QualifiedToolInvocationMaterialProvider` + generic `QualifiedToolInvocationResolver` (Tool contracts); no runtime `Task` / `Any` public dependency |
| UCA reopen | **NO** |
| Class C triggers | None identified — model realizable without changing UCA contracts, EE request semantics, `QualifiedCapabilityBindingProvider`, or `QualifiedCapabilityExecutionTarget` |

Rationale: new Tool-domain providers, staging repository, material provider, and resolver implement against frozen SPIs and registries; no frozen UCA/EE payload semantics change; business inputs stay application-owned.

---

## 18. Future implementation scope (closed world)

### Contracts

- `intergrax/contracts/tools/marketplace_qualified_capability.py` — stage model + repository protocol (**P1 done**)
- `intergrax/contracts/tools/marketplace_qualified_tool_stage_context.py` — **P2 (P2-P0-R1 lock):** `MarketplaceQualifiedToolStageContext`, `MarketplaceQualifiedToolStageContextAssociation`, `MarketplaceQualifiedToolStageContextResolver`, `MarketplaceQualifiedToolStageContextAssociationRepository`
- `intergrax/contracts/tools/marketplace_handoff_reference.py` — **P2:** strict `handoff://` parse + `derive_marketplace_gap_handoff_id` + integrity helpers (no raw tenant in URI)
- `intergrax/contracts/tools/qualified_tool_invocation.py` — typed invocation material contract + `QualifiedToolInvocationMaterialProvider` protocol + `QualifiedToolInvocationResolver` protocol (Verdict B; split leaf modules if conventions require)

### Tool domain

- `intergrax/tools/marketplace_qualified_capability_staging.py` (**P1 done**)
- `intergrax/tools/marketplace_qualified_tool_stage_context_association.py` — **P2:** `ConditionalDocumentStore` association repository
- `intergrax/tools/marketplace_qualified_tool_stage_context_resolver.py` — **P2:** fail-closed resolver implementation
- `intergrax/tools/marketplace_qualified_capability_qualification_provider.py`
- `intergrax/tools/marketplace_qualified_capability_binding_provider.py`
- `intergrax/tools/marketplace_qualified_capability_execution_handler.py`

### Marketplace adapter

- `intergrax/marketplace/handoff/adapters/tool_qualification_staging_consumer.py` (**P1 done**; **P2:** record context association alongside `stage`; same envelope-authoritative `tenant_id` + v2 `handoff_id`)
- **Minimal Marketplace touch (P2):** `intergrax/marketplace/acquisition/gap_acquisition_service.py` (or shared helper consumed there) — emit `derive_marketplace_gap_handoff_id` using envelope/query-context `tenant_id`; **no** UCA contract change

### Tests (P2)

- `tests/unit/tools/test_marketplace_qualified_tool_stage_context.py` — §23 fifteen gates: tenant-distinct id, collision, idempotent replay, parse + re-derive integrity, cross-tenant isolation, partial-write qualification failures
- `tests/unit/tools/test_marketplace_qualified_capability_qualification_provider.py` — resolver + repository integration (mocked ports)
- `tests/unit/tools/test_marketplace_qualified_capability_binding_provider.py` — tenant-safe bind path
- Extend `tests/unit/marketplace/test_uca5_marketplace_acquisition_strategy.py` only if needed for staging+association wiring smoke (no UCA semantic change)

Leaf-module imports only; no package-root re-export unless an existing canonical package API requires it.

**Explicitly out of scope for implementation:** changing `ToolMarketplaceAcquisitionBridge` behavior for legacy flows; scenario-local shortcuts.

---

## 19. Future implementation gates

### Contract gates

- Immutable typed staging; no `Any` semantic payload at Tool invocation boundaries; no reflection routing; no private cross-class reads; `intergrax/contracts/tools/**` MUST NOT import `intergrax/runtime/**`

### Acquisition gates

- Marketplace produces handoff; staging consumer only; ToolRegistry activation count **0**

### Qualification gates

- Staged subject required; failure → activation count **0**

### Binding gates

- Same qualified subject; tenant match; no execution during bind

### Execution gates

- Handler via `binding_provider_id`; activation only after `QUALIFIED`; digest/version preserved; ToolRuntime only; Tool execution count **0** before qualification success; invocation material via provider (typed BaseModel), not Task metadata parsing

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
| P0-R1: no public Tool contract dependency on runtime `Task` / `Any` invocation semantics | PASS |
| P0-R1: typed invocation-material provider + generic resolver frozen | PASS |
| Binding SPI sufficient | PASS |
| No Class C / STOP condition | PASS |
| Design record complete | PASS |
| Production code changed in P0 | PASS (none) |
| UCA semantic change required | FAIL (not needed — lock proceeds) |

---

## 21. No-UCA-reopen conclusion

GAP-02 closes via **Class A** Tool-domain extensions: staging, qualification/binding providers, **typed invocation-material provider**, and **generic invocation resolver** contracts. Public Tool contracts do not depend on runtime `Task` or `dict[str, Any]` invocation semantics. No modification to frozen UCA resume types, binding SPI semantics, or Execution Engine ownership. **Implementation may proceed to P1 after independent audit of this commit.**

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

---

## 25. P3 Canonical Invocation Intent Propagation (`S24-GAP-02-P3-P0` · `S24-GAP-02-P3-P0-R1`)

**Scope:** architecture lock only — **no** P3 production/tests in this step.

**Session evidence:** **P3-P0** at `d3a85c83910c858a78fc8b243cd42b73e3491f29`; **P3-P0-R1** corrects §25.2 pre-EE hook against the same committed AW/contract symbols. Uncommitted working-tree drift under `intergrax/runtime/execution/**` is not architecture evidence.

### 25.1 Required verdict

```text
S24-GAP-02-P3
CLASS B

UCA REOPEN = NO
EE REOPEN = NO
```

**P3 invocation intent:** new typed durable Tool execution intent + Tool-domain contracts/repository/provider/handler (extension). **AW:** optional `QualifiedCapabilityExecutionIntentPreparationPort` on `WorkerCapabilityFulfillmentCoordinator` (composition). **Frozen:** `WorkerQualifiedCapabilityResumeRequest`, `WorkerQualifiedCapabilityExecutionRequest`, `QualifiedCapabilityExecutionIntakePayload`, `BoundCapabilityExecutionDispatchRequest`, EE lifecycle, execution identity ownership, qualification/binding semantics — unchanged unless **Class C** triggers fire (§25.2.6).

Rationale: `BoundCapabilityExecutionDispatchRequest` correctly stays minimal; `QualifiedCapabilityExecutionIntakePayload` / `WorkerQualifiedCapabilityExecutionRequest` already carry `worker_need_id` but the binding handler SPI does **not**. There is **no** public `worker_need_id → WorkerCapabilityNeed` read port. Handler restart cannot rely on in-memory AW state or forbidden string parsing of `worker_need_id`. A Tool-owned durable sidecar keyed by `execution_request_id` is required without changing frozen EE dispatch semantics.

### 25.2 Q1 — last canonical `WorkerCapabilityNeed` before EE

| Stage | File | Symbol | Owner | Need form |
| ----- | ---- | ------ | ----- | --------- |
| Qualified resume orchestration (in-memory) | `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py` | `WorkerCapabilityFulfillmentCoordinator._fulfill_qualified` / `_fulfill_qualified_async` (`need = request.acquisition_request.need`) | AW | **In memory** — full `WorkerCapabilityNeed` |
| Same episode (durable) | `intergrax/autonomous_work/worker_recovery_capability_fulfillment_episode_context_provider.py` | `DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider.resolve_episode_context` → `WorkerRecoveryObstacleCapabilityNeedReadPort.get_obstacle_capability_need` | AW | **Durable read** — requires `worker_instance_id` + `obstacle_id` |
| Post-binding resume handoff | `intergrax/autonomous_work/worker_qualified_capability_resume_coordinator.py` | `_prepare_execution_handoff` → `WorkerQualifiedCapabilityExecutionRequest` | AW → EE ingress | **Not present** — only `worker_need_id` |
| EE intake (full provenance) | `intergrax/runtime/execution/qualified_capability_execution_dispatch_service.py` | `_build_intake_payload` → `QualifiedCapabilityExecutionIntakePayload` | EE ingress | **Not present** — only `worker_need_id` |
| Handler SPI | `intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py` | `QualifiedCapabilityExecutionRuntimeDelegate.execute` → `BoundCapabilityExecutionDispatchRequest` | EE | **Not present** |

**Last production point with full typed `WorkerCapabilityNeed` in memory (Marketplace qualified resume path):** `WorkerCapabilityFulfillmentCoordinator._fulfill_qualified` / `_fulfill_qualified_async` — locals `need`, `qualification`, `worker_need_id`, `resume_operation_id` after `qualification.outcome == QUALIFIED` and **before** `self._resume.resume` / `resume_async`.

**Where EE first admits execution (unchanged):** `WorkerQualifiedCapabilityResumeCoordinator` → `_prepare_execution_handoff` → `derive_qualified_capability_execution_request_id` → `WorkerQualifiedCapabilityExecutionPort.execute`. Resume coordinator **re-derives** the same public ids; it does **not** receive full need on frozen resume types.

**P3 write hook (frozen — `S24-GAP-02-P3-P0-R1`):** file `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py`, methods `_fulfill_qualified` / `_fulfill_qualified_async`, **after** `QUALIFIED` and after `need`, `worker_need_id`, and `resume_operation_id` are available, **after** pre-EE canonical `execution_request_id` derivation (§25.2.1), call optional `QualifiedCapabilityExecutionIntentPreparationPort.prepare(...)`, then **before** `WorkerQualifiedCapabilityResumeRequest` construction and **before** `self._resume.resume(...)`. Full need meets canonical `execution_request_id` **here** — not via frozen resume/EE request fields.

#### 25.2.1 Pre-EE canonical execution identity (single algorithm)

Fulfillment coordinator MUST **reuse** the same public derivation functions as `WorkerQualifiedCapabilityResumeCoordinator` (`intergrax/autonomous_work/worker_qualified_capability_resume_coordinator.py`). **Forbidden:** local reimplementation of binding/execution id formulas.

| Step | Public function | Module |
| ---- | --------------- | ------ |
| Qualified subject | `qualified_capability_subject_from_result(qualification)` | `intergrax/contracts/capability_qualification/qualified_subject.py` |
| Binding operation id | `derive_qualified_capability_binding_operation_id(resume_operation_id=..., qualified_subject_reference=...)` | `intergrax/contracts/capability_qualification/qualified_capability_binding.py` |
| Execution request id | `derive_qualified_capability_execution_request_id(resume_operation_id=..., binding_operation_id=...)` | `intergrax/contracts/autonomous_work/worker_qualified_capability_resume.py` |

Frozen sequence in `_fulfill_qualified*` (conceptual):

```text
subject = qualified_capability_subject_from_result(qualification)
binding_operation_id = derive_qualified_capability_binding_operation_id(
    resume_operation_id=resume_operation_id,
    qualified_subject_reference=subject.qualified_subject_reference,
)
execution_request_id = derive_qualified_capability_execution_request_id(
    resume_operation_id=resume_operation_id,
    binding_operation_id=binding_operation_id,
)
```

Pre-EE derivation uses qualification facts only; binding may still fail later (`BLOCKED`, `UNAVAILABLE`, `FAILED`, `HITL`). The id is still deterministic for that qualified subject. Resume coordinator MUST produce **identical** `binding_operation_id` and `execution_request_id` when execution handoff proceeds (§25.2.7).

#### 25.2.2 Generic intent preparation extension seam

**Contract owner:** AW → Tool execution boundary (suggested: `intergrax/contracts/tools/qualified_capability_execution_intent_preparation.py` — final path/naming at implementation).

**Implementation owner:** Tool domain (Marketplace qualified Tool provider + `QualifiedMarketplaceToolExecutionIntentRepository`).

**Injection (frozen shape):**

```text
WorkerCapabilityFulfillmentCoordinator(
    ...
    intent_preparation: QualifiedCapabilityExecutionIntentPreparationPort | None = None,
)
```

`prepare(...)` accepts **typed facts** — prefer **not** passing the whole `WorkerCapabilityFulfillmentRequest`. Minimum inputs include: `need`, `qualification_result`, `execution_request_id`, `tenant_id`, `task_id`, `worker_need_id`, `resume_operation_id`; optional `binding_operation_id`, `handoff_id`, correlation fields when needed for durable intent integrity.

**Outcomes (conceptual):**

| Outcome | AW behavior |
| ------- | ----------- |
| `NOT_APPLICABLE` | Non-Marketplace qualified capability kinds/providers — continue existing resume path (CodeCraft unchanged) |
| Success (`CREATED` / `ALREADY_RECORDED_IDENTICAL`) | Proceed to `self._resume.resume(...)` |
| `UNAVAILABLE`, `CONFLICT`, `INTEGRITY_FAILURE`, `INVALID_OPERATION` | **Fail closed** — **do not** call `resume()` on Marketplace qualified Tool path (§25.2.4) |

Marketplace Tool provider (inside Tool domain, not AW):

- Recognizes qualified subject for `marketplace.tool.qualified_binding.v1` (frozen kind string at implementation).
- `need.required_operations` + staged Marketplace Tool → `QualifiedMarketplaceToolOperationSelector` → `selected_operation` (§25.4).
- Persists `QualifiedMarketplaceToolExecutionIntent` (§25.7) — **does not** copy full `WorkerCapabilityNeed`.

**Forbidden in `WorkerCapabilityFulfillmentCoordinator`:** Marketplace/Asterion branches; imports of `DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository`, `DynamicToolAcquisitionService`, `ToolRuntime`, or other Tool-runtime implementations.

#### 25.2.3 Orphan intent semantics and retention

Recorded execution intent is **inert data** — not authority, not admission, not activation, not permission. A stored intent **cannot** start Tool execution. `MarketplaceToolQualifiedCapabilityExecutionHandler` may load intent **only** when EE delivers `BoundCapabilityExecutionDispatchRequest` with:

```text
intent.execution_request_id == dispatch.execution_request_id
intent.tenant_id == dispatch.tenant_id
intent.qualified_subject_reference == dispatch.execution_target.qualified_subject_reference
```

If intent carries `task_id`: `intent.task_id == dispatch.task_id`. Mismatch → handler disposition `FAILED` (§25.15).

Binding/resume may complete without EE execution; intent may remain **orphan** audit evidence. **Retention (preferred):** immutable append/idempotent store under existing backing retention — same `ConditionalDocumentStore` pattern as `DocumentStoreMarketplaceQualifiedToolStageRepository` / stage association repos. **No** P3 delete-on-binding-failure; **no** new garbage-collector framework.

#### 25.2.4 Pre-resume intent failure → `WorkerCapabilityFulfillmentDisposition`

When `intent_preparation` is configured and the provider returns a failure outcome (not `NOT_APPLICABLE`) for a Marketplace qualified Tool subject, fulfillment **MUST NOT** invoke `self._resume.resume` / `resume_async`.

| Preparation failure | `WorkerCapabilityFulfillmentDisposition` |
| ------------------- | ---------------------------------------- |
| `UNAVAILABLE` | `DISCOVERY_UNAVAILABLE` |
| `CONFLICT` | `DISCOVERY_CONFLICT` |
| `INTEGRITY_FAILURE` | `FAIL_CLOSED` |
| `INVALID_OPERATION` | `FAIL_CLOSED` |

No new global fulfillment disposition enum members for P3.

#### 25.2.5 `WorkerQualifiedCapabilityResumeCoordinator` unchanged

Hard requirement: **no** new fields on resume coordinator; **no** full-need dependency. Production semantics unchanged — coordinator continues to derive `subject`, `binding_operation_id`, `execution_request_id` via the same public functions during binding/execution handoff. Later re-derivation is the end-to-end integrity check (§25.2.7).

#### 25.2.6 Class C reopen triggers

**STOP** — classify **CLASS C CANDIDATE · ARCHITECTURE REOPEN** if implementation requires:

- Field addition/change on `WorkerQualifiedCapabilityResumeRequest`, `WorkerQualifiedCapabilityExecutionRequest`, `QualifiedCapabilityExecutionIntakePayload`, or `BoundCapabilityExecutionDispatchRequest`;
- EE lifecycle change, execution identity ownership change, or qualification/binding semantic change.

#### 25.2.7 P3 execution identity test invariant (future implementation)

```text
execution_request_id predicted in WorkerCapabilityFulfillmentCoordinator (pre-resume)
==
execution_request_id derived in WorkerQualifiedCapabilityResumeCoordinator execution handoff
==
execution_request_id on BoundCapabilityExecutionDispatchRequest at EE handler
```

#### 25.2.8 Sync and async qualified fulfillment

Identical pre-EE preparation semantics in `_fulfill_qualified` and `_fulfill_qualified_async`. Intent `record` may remain synchronous when the repository contract is synchronous and the coordinator follows existing sync/async split. If intent persistence requires a new async-only repository API → **architecture STOP** (separate design).

### 25.3 Q2 — public read by `worker_need_id`

```text
NO
```

Evidence: `WorkerRecoveryObstacleCapabilityNeedReadPort` (`intergrax/autonomous_work/worker_recovery_capability_fulfillment_episode_context_ports.py`) exposes only `get_obstacle_capability_need(worker_instance_id, obstacle_id)`. No production `Protocol` or service method accepts `worker_need_id` alone. `derive_worker_capability_need_id` (`intergrax/contracts/autonomous_work/capability_acquisition.py`) is a one-way hash of need fields — not a lookup key API.

### 25.4 Q3 — where `selected_operation` is chosen

**Decision:** **A — before EE**, persisted as immutable Tool execution intent.

| Concern | Owner |
| ------- | ----- |
| `WorkerCapabilityNeed.required_operations` semantics | AW |
| Operation selection policy vs staged Tool release | Tool domain (`QualifiedMarketplaceToolOperationSelector` — new P3 contract; not EE) |
| Persisted `selected_operation` on intent | Tool domain (`QualifiedMarketplaceToolExecutionIntent`) |

Handler MUST NOT re-select from durable need at execution time via obstacle lookup (resume/EE path lacks `obstacle_id`; parsing `worker_need_id` is forbidden).

**Selection rules (frozen):**

| `len(required_operations)` | Behavior |
| -------------------------- | -------- |
| 0 | Invalid at need construction (`_validate_operations` — fail closed upstream) |
| 1 | `selected_operation = required_operations[0]` |
| >1 | **Fail closed** unless a typed Tool-domain selection policy resolves exactly one operation against staged release + need (no “first element”, no string parsing). Until that policy contract exists, intent build returns NOT EXECUTABLE — **P3 blocker** for multi-operation Marketplace needs |

### 25.5 Q4 — frozen EE dispatch change?

```text
NO
```

`BoundCapabilityExecutionDispatchRequest` (`intergrax/contracts/execution/bound_capability_execution_dispatch.py`) remains unchanged. Propagation uses durable `QualifiedMarketplaceToolExecutionIntent` keyed by `execution_request_id`, loaded inside `MarketplaceToolQualifiedCapabilityExecutionHandler`.

### 25.6 Q5 — stable invocation-intent key

| Candidate | Verdict |
| --------- | ------- |
| `execution_request_id` | **Preferred** — deterministic via `derive_qualified_capability_execution_request_id(resume_operation_id, binding_operation_id)`; identifies one qualified execution intake; present on bound dispatch; EE ingress ledger uses `(tenant_id, execution_request_id)` (`QualifiedCapabilityExecutionDispatchService`) |
| `binding_operation_id` | Insufficient alone (multiple executions could be modeled per binding in future) |
| `qualified_subject_reference` | Not unique per execution attempt |
| `task_id` | Not unique per qualified Tool execution |
| `execution_target_reference` | Opaque target id — not minted per EE attempt |

Intent repository API: `get(execution_request_id=...)` with integrity cross-check of `tenant_id` and `qualified_subject_reference` against bound dispatch + `execution_target`.

### 25.7 Q6 — invocation-intent repository owner

**Owner:** Tool domain execution extension (AW → Tool handoff boundary), **not** EE core, UCA qualification, Marketplace, or scenario proof.

Suggested contracts (P3):

- `intergrax/contracts/tools/qualified_marketplace_tool_execution_intent.py` — `QualifiedMarketplaceToolExecutionIntent`, write outcomes (`CREATED`, `ALREADY_RECORDED_IDENTICAL`, `CONFLICT`)
- `QualifiedMarketplaceToolExecutionIntentRepository` — `record` / `get` by `execution_request_id`
- Implementation: `DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository` using existing `ConditionalDocumentStore` (same pattern as `DocumentStoreMarketplaceQualifiedToolStageRepository`)

Stores **Tool execution intent only** (selected operation + correlation), not full `WorkerCapabilityNeed`.

### 25.8 Q7 — restart semantics

| Artifact | Durable? | Replay |
| -------- | -------- | ------ |
| `MarketplaceQualifiedToolStage` | Yes (`MarketplaceQualifiedToolStageRepository`) | `get(tenant_id, handoff_id)` via context resolver |
| Stage context association | Yes (P2) | Resolver on qualification/binding/execution |
| `WorkerCapabilityNeed` | Yes (obstacle store) | **Not reachable** on handler path without `obstacle_id` |
| Invocation intent | **Must be durable** (new store) | Handler `get(execution_request_id)` after restart |
| EE execution | Yes | Canonical EE + suspended-operation reentry (no second HITL) |
| Activation | Idempotent | See §25.12 |

Reconstruction from AW records alone without intent store would require forbidden `worker_need_id` parsing or Task metadata — **rejected**.

Material provider restart semantics: provider input is stable public identity (`QualifiedToolInvocationMaterialRequest` — §25.11); provider implementation MUST be safe under EE replay (idempotent or explicitly documented side-effect policy); platform does not cache business material in EE.

### 25.9 Q8 — `CapabilityReleaseIdentity` → `DynamicToolAcquisitionRequest`

**Verdict:** **LOSSLESS** for material release fields **when** `content_digest` and `package_reference` / `version_label` are populated on the staged release (validated at staging).

| `DynamicToolAcquisitionRequest` field | Source from `CapabilityReleaseIdentity` |
| ------------------------------------- | --------------------------------------- |
| `capability_identity_key` | `CapabilityIdentityKey.from_discovery_identity(selected_release.discovery)` |
| `selected_identity.catalog_source_id` | `selected_release.discovery.source.source_id` |
| `selected_identity.package.logical_tool_id` | `selected_release.discovery.logical.logical_id` |
| `package_reference` / `package_version` / `package_digest` | `package_reference`, `version_label`, `content_digest` |
| `catalog_entry_id` | Optional — omit unless staged with catalog entry id |
| `host_profile_id` | **Not on release** — composition-injected host execution policy (`ApplicationEnvironmentProfile.profile_id` or dedicated host profile port wired into handler composition) |
| `operation_id` | **Not on release** — derive deterministically for post-qualification activation (§25.12) |

**STOP conditions:** missing digest/package facts on stage → activation fail closed (no rediscovery, no “nearest” tool).

### 25.10 Q9 — complete invocation identity matrix

| Field | Required by | Canonical source | Available at handler? | Resolution |
| ----- | ----------- | ---------------- | --------------------- | ---------- |
| `tool_id` | ToolRuntime | `DynamicToolAcquisitionResult.registry_tool_id` after exact activation | After activation step inside handler | Post-qualification `DynamicToolAcquisitionPort.acquire` |
| `selected_operation` | Resolver | `QualifiedMarketplaceToolExecutionIntent.selected_operation` | After intent load | Durable intent recorded pre-EE |
| `input` (`BaseModel`) | ToolRuntime | `QualifiedToolInvocationMaterialProvider` | Inside handler | Typed material contract — no Task metadata |
| `tenant_id` | ToolRuntime | `BoundCapabilityExecutionDispatchRequest.tenant_id` | YES | Integrity vs intent |
| `task_id` | ToolRuntime | `BoundCapabilityExecutionDispatchRequest.task_id` | YES | Pass-through |
| `run_id` | ToolRuntime | EE active identity (`require_active_execution_identity`) | YES | Handler parameter + invoker |
| `agent_id` | ToolRuntime | `ExecutionBoundCatalogToolInvoker.caller_agent_id` | YES (injected) | Composition: `caller_agent_id` from application host wiring (`build_execution_bound_catalog_tool_composition`, same as UCA-6C CodeCraft qualified path) |
| `step_id` | ToolRuntime | Deterministic qualified-execution step | YES (derived) | `f"qmte:{execution_request_id}"` (parallel to CodeCraft `f"qce:{execution_request_id}"` in `intergrax/runtime/codecraft/wiring_bound_capability_execution.py`) |
| `correlation_request_id` | Optional | EE `execution_id` | YES | Handler `execution_id` parameter |
| `idempotency_key` | Optional / policy | Tool idempotency coordinator | Optional | Derive `f"qmte:{execution_request_id}:{selected_operation}"` unless host policy overrides |

### 25.11 Business material provider input (frozen minimum)

Public request type (P3 contract, no runtime imports):

```text
execution_request_id
tenant_id
task_id
selected_operation
qualified_tool_identity   # registry tool id post-activation OR staged release identity policy frozen in contract
handoff_id                # correlation for domain materialization
worker_need_id            # correlation only — provider MUST NOT parse it for operations
```

Implementations live in application/domain; handler wires provider instance via composition.

### 25.12 Activation model (post-qualification)

Inside `MarketplaceToolQualifiedCapabilityExecutionHandler` after intent + stage integrity:

1. Resolve `handoff_id` from qualified subject / execution target reference (existing binding provider conventions).
2. Load `MarketplaceQualifiedToolStage` — fail closed on missing/conflict.
3. Map stage `selected_release` → `DynamicToolAcquisitionRequest` (§25.9).
4. `host_profile_id` from constructor-injected host execution context (same source as other Tool acquisition in app composition).
5. **Activation `operation_id` (deterministic):** `f"marketplace-qualified-tool-activation:{execution_request_id}"` — distinct from acquisition `operation_id` and from `binding_operation_id`.

Reuse `DynamicToolAcquisitionService` mechanics only **inside** the handler after EE admission — no Marketplace bridge, no pre-qualification registry mutation.

### 25.13 ToolRuntime invoker dependency

Canonical handler port: `ExecutionBoundCatalogToolInvoker` (`intergrax/contracts/execution_bound_catalog_tool_invocation.py`). Production adapter: `NexusExecutionBoundCatalogToolInvoker` via `build_execution_bound_catalog_tool_composition` — **no** direct Nexus imports in handler module; composition-only wiring.

### 25.14 Q12 — release operation declarations

**NO** — `CapabilityReleaseIdentity` (`intergrax/contracts/capability_catalog/release_identity.py`) and `MarketplaceQualifiedToolStage` do not declare Tool logical operations. Operation truth remains `WorkerCapabilityNeed.required_operations` plus Tool-domain selection policy (§25.4). P3 may add a **Tool-domain** operation selection contract; it is not implied by release identity alone.

### 25.15 Failure model → `QualifiedCapabilityExecutionDispatchDisposition`

| Condition | Disposition | Notes |
| --------- | ----------- | ----- |
| Intent missing | `FAILED` | `intent_not_found` |
| Intent conflict / integrity mismatch | `FAILED` | fail closed |
| Stage missing / corrupt | `FAILED` or `UNAVAILABLE` | map staging errors |
| Activation resolution failure | `FAILED` | no rediscovery |
| Activation conflict | `FAILED` | |
| Activation unavailable | `UNAVAILABLE` | |
| Material unavailable | `UNAVAILABLE` | |
| Material invalid | `FAILED` | |
| Resolver invalid operation | `FAILED` | |
| ToolRuntime unavailable | `UNAVAILABLE` | |
| Tool execution failed | `FAILED` | |
| Tool suspended / HITL | Propagate `ExecutionSuspendedWorkPauseRequired` | EE-owned suspension only |
| Handler / binding mismatch | `FAILED` | `binding_provider_id` / target validation |

### 25.16 Updated P3 execution flow

```text
WorkerCapabilityFulfillmentCoordinator
    has full WorkerCapabilityNeed (in memory)
    → derive resume_operation_id
    → qualified_capability_subject_from_result(qualification)
    → derive binding_operation_id (canonical public function)
    → derive execution_request_id (canonical public function)
    → QualifiedCapabilityExecutionIntentPreparationPort (optional; Tool provider for Marketplace)
        → QualifiedMarketplaceToolOperationSelector
        → durable QualifiedMarketplaceToolExecutionIntent (record before resume)
    → existing self._resume.resume(...)  [frozen WorkerQualifiedCapabilityResumeRequest — no full need]
WorkerQualifiedCapabilityResumeCoordinator
    → re-derives same binding_operation_id / execution_request_id (canonical public functions)
    → EE admission
QualifiedCapabilityExecutionDispatchService
    → BoundCapabilityExecutionDispatchRequest (minimal — frozen)
MarketplaceToolQualifiedCapabilityExecutionHandler
    → intent lookup by execution_request_id + integrity checks (§25.2.3)
    → stage + post-qualification activation + material + resolver + ToolRuntime
```

### 25.17 P3 implementation scope (post-lock)

**Contracts**

- `intergrax/contracts/tools/qualified_marketplace_tool_execution_intent.py` — `QualifiedMarketplaceToolExecutionIntent` minimum: `execution_request_id`, `tenant_id`, `qualified_subject_reference`, `selected_operation`, `worker_need_id`; optional `binding_operation_id`, `task_id`, `handoff_id` when needed for material/integrity
- `intergrax/contracts/tools/qualified_capability_execution_intent_preparation.py` — `QualifiedCapabilityExecutionIntentPreparationPort` + typed preparation outcomes (`NOT_APPLICABLE`, success, failure vocabulary)
- Extend `intergrax/contracts/tools/qualified_tool_invocation.py` (material + resolver + material request identity)
- Optional: `qualified_marketplace_tool_operation_selection.py` if >1 operation policy is required before CERT

**Tool domain**

- `intergrax/tools/qualified_marketplace_tool_execution_intent_repository.py`
- `intergrax/tools/qualified_marketplace_tool_operation_selector.py`
- Marketplace Tool `QualifiedCapabilityExecutionIntentPreparationPort` implementation (provider)
- `intergrax/tools/marketplace_qualified_capability_execution_handler.py`
- Composition: `intergrax/tools/marketplace_qualified_capability_execution_composition.py` (or application shared wiring sibling)

**AW (minimal production touch — no frozen resume/EE request field changes)**

- `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py` — optional port call at §25.2 hook; pre-EE id derivation via public functions only
- Fulfillment coordinator factory/composition sibling (inject `intent_preparation` provider)
- **Not** `worker_qualified_capability_resume_coordinator.py` for production semantics (test instrumentation only if needed)

**Tests (P3 minimum)**

- `tests/unit/tools/test_qualified_marketplace_tool_execution_intent.py`
- `tests/unit/tools/test_marketplace_qualified_capability_execution_handler.py`
- Coordinator / composition tests for §25.2 hook and identity chain

**Future P3 test gates (required before CERT)**

1. Coordinator without `intent_preparation` → existing paths unchanged
2. Non-Marketplace qualified subject → preparation `NOT_APPLICABLE`; resume proceeds
3. Marketplace Tool → intent recorded **before** `resume()` call
4. Intent `CONFLICT` → `resume()` **not** called
5. Intent `UNAVAILABLE` → `resume()` **not** called
6. `selected_operation` copied exactly from selector into durable intent
7. Predicted pre-resume `execution_request_id` == resume coordinator `execution_request_id`
8. `_fulfill_qualified` vs `_fulfill_qualified_async` — identical execution identity
9. CodeCraft qualified path regression unchanged
10. No field changes on frozen EE/resume dispatch request types

### 25.18 Contract purity (P3-P0 confirmation)

```text
intergrax/contracts/tools/** → NO intergrax/runtime/** imports
NO Task.message / Task.context / Task.metadata public semantics
NO dict[str, Any] invocation payload
NO reflection dispatch
NO worker_need_id string parsing for operation selection
```
