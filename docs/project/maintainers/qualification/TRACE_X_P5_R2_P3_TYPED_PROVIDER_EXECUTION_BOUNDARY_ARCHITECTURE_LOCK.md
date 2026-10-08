# TRACE-X-P5-R2-P3-R1-R2 — Typed Provider Execution Boundary Architecture Lock

## A. Revision / status

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R2` — Typed Provider Execution Boundary Architecture Lock |
| **Child reconciliation** | `TRACE-X-P5-R2-P3-R1-R3` — Governance + typed resolution reconciliation @ `63ba8a237449ce7e78043c480dff97ed1f96918e` (§O) · `TRACE-X-P5-R2-P3-R1-R4` — Configured-Adoption Pilot Admission @ `fcfa59797c86f5933a7e53cb6784768a4f1d1787` (§P) · `TRACE-X-P5-R2-P3-R1-R5` — Deferred Configured Provider Dependency Projection @ `62ec2457268ac7287be9c643d7d3cf1b97f32f07` (§Q) |
| **START_HEAD (R1-R2)** | `d39b39ee94982d74acc7042c4dfa63a887a97c05` (`development` = `origin/development` @ R1-R2 task start) |
| **START_HEAD (R1-R3)** | `63ba8a237449ce7e78043c480dff97ed1f96918e` (`development` = `origin/development` @ R1-R3 task start) |
| **START_HEAD (R1-R4)** | `fcfa59797c86f5933a7e53cb6784768a4f1d1787` (`development` = `origin/development` @ R1-R4 task start) |
| **START_HEAD (R1-R5)** | `62ec2457268ac7287be9c643d7d3cf1b97f32f07` (`development` = `origin/development` @ R1-R5 task start) |
| **Audited HEAD (R1-R4)** | `fcfa59797c86f5933a7e53cb6784768a4f1d1787` (docs amendment follows in same task commit) |
| **Audited HEAD (R1-R5)** | docs amendment in same task commit after `62ec2457268ac7287be9c643d7d3cf1b97f32f07` |
| **Parent** | `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **Steering authority** | [`TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md`](TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md) (P0) · [`TRACE_X_P5_R2_P3_R1_ACTUAL_USE_JOIN_POINT_ARCHITECTURE_RECONCILIATION.md`](TRACE_X_P5_R2_P3_R1_ACTUAL_USE_JOIN_POINT_ARCHITECTURE_RECONCILIATION.md) (P3-R1 / R1-R1) |
| **Primary FRZ** | `FRZ-TRC-11` — **OPEN** (no PASS) |
| **Primary blocker** | `P5-GAP-04` — **IMPLEMENTATION IN PROGRESS**; causality owned by **`R2-P3-EFFECTIVE-USE-CAUSALITY-03`** |
| **Production delta (this child)** | **0** — architecture lock only |
| **Recommended disposition** | **`TRACE-X-P5-R2-P3-R1-R3` = READY FOR AUDIT** (supersedes R1-R2 audit line only for governance + resolution typing) |
| **Parent P3** | **BLOCKED PENDING ARCHITECTURE AUDIT / IMPLEMENTATION** — not CLOSED |
| **TRACE-X-P5-R2** | **CURRENT / BLOCKED ON P3** (roadmap: **P3 NEXT**) |
| **P4** | **NOT ENTERED** — forbidden in this task |

**Concurrent change @ START_HEAD:** none intersecting this lock; pin `d39b39ee…` contained in HEAD.

---

## B. Current before graph (disconnect @ `d39b39ee…`)

```text
CONFIGURE_EXISTING / fulfillment
  → ExecutionIntegrationConfigurationAdoption (neutral qualified intake only)

Execution admission
  → ExecutionId minted / active (ExecutionRuntime)

QualifiedCapabilityExecutionRuntimeDelegate.execute
  → pin_configured_adoption_for_execution
      → ExecutionBoundIntegrationConfigurationExecutionPinningAdapter
      → ExecutionBoundIntegrationResolution.resolve_and_pin
          → _observe_effective_identity
              → resolve / resolve_from_profile (materialize)
              → observe provider_id
              → DISCARD materialized instance
          → validate_configured_adoption_match
          → pin ExecutionIntegrationConfigurationPinningStore
  → handler.dispatch_once(BoundCapabilityExecutionDispatchRequest)
      → category path (e.g. Marketplace tool invoker)
          → ToolInvocationWiringResolver / separate resolve paths
          → first provider/tool I/O (no typed link to pin materialization)
```

**Architectural defect:** provenance records *effective identity* from a **transient** materialization; **actual** provider I/O may use a **second** resolution graph. String equality of `provider_id` across independent resolves is **not** a mechanical causality proof (P3-R1 §1, P0 §17).

**Rejected carry paths (audit-closed):** live `CategoryIntegrationInstance` / `PlatformIntegrationContract` on `QualifiedCapabilityExecutionDispatchRequest`, `QualifiedCapabilityExecutionIntakePayload`, or `BoundCapabilityExecutionDispatchRequest` (P3-R1-R1 §2).

**Rejected semantic owner:** Marketplace / `ToolInvocationWiringResolver` as integration configuration or configured/effective truth (task §10; tool wiring = dependency projection only).

---

## C. Locked after graph (ownership boundaries)

```text
domain / application / qualified execution caller
  → typed category operation request (category-specific port)
  → Integrations-owned Configured Provider Execution Boundary
        │
        ├─ inputs: tenant_id, ExecutionId, ExecutionIntegrationConfigurationAdoption
        │          (+ Integrations-owned materialization selectors: IntegrationProfile | catalog_slug — §O.4)
        │
        ├─ single materialization (resolve / resolve_from_profile) — ONCE per adoption subject
        ├─ EffectiveIntegrationIdentity from SAME local instance
        ├─ validate_configured_adoption_match (fail closed)
        ├─ build ExecutionIntegrationConfigurationProvenance + IntegrationConfigurationSubject
        ├─ pin (ExecutionIntegrationConfigurationPinningStore)
        └─ invoke category-specific typed operation on SAME local materialized provider
              (no second resolve / resolve_from_profile for that subject)

Execution layer supplies: ExecutionId, tenant, execution target, optional adoption on **qualified** surfaces only.
Governance: **operation-level permission only** — never materializes providers, never substitutes Execution admission, never infers permission from pin/adoption (§O).
AW: opaque configuration_ref + adoption handoff — not provider execution owner.
Integrations: **no** ALLOW/DENY policy decisions; **no** synthesized governance evidence; configured binding ≠ operation permission (§O.2).
Reconstruction: ExecutionReconstructor + neutral provenance reader only — no live provider.
```

**Mechanical invariant (locked):**

```text
pinned effective provider_id
==
provider_id observed from the materialized instance that receives the first category business call
==
same object reference (local continuity inside Integrations/category boundary)
```

---

## D. Contract model

### D.1 Pattern decision — **Pattern A** (not a universal `execute()`)

| Option | Disposition |
|---|---|
| **Pattern A** — shared lifecycle coordinator + category-specific typed operation ports | **SELECTED** — matches existing Intergrax seams: `ExecutionBoundIntegrationResolution` (lifecycle), `PlatformIntegrationContract` / per-category contracts (operations), EBH-2G RAG precedent (Integrations-owned materialization + typed category ports). |
| **Pattern B** — only category ports, duplicated lifecycle | **REJECTED** — duplicates P1/P2 pinning and adoption validation already centralized in `ExecutionBoundIntegrationResolution`. |
| Universal `execute(operation: object) -> object` | **FORBIDDEN** (task §6, FRZ-TYP-*) |

**Reuse vs new abstraction**

| Existing mechanism | Role after lock |
|---|---|
| `ExecutionBoundIntegrationResolution` | **Extend** (conceptually `resolve_materialize_validate_pin`) — same owner for materialize → identity → validate → pin; **return materialized instance to Integrations/category callers only** (not on neutral Execution DTOs). |
| `ExecutionIntegrationConfigurationAdoption`, `ConfiguredCapabilityBinding`, `EffectiveIntegrationIdentity`, provenance DTOs | **Reuse** — no field duplication on new envelopes. |
| `ExecutionBoundIntegrationMaterializationPort` | **Reuse** — tighten `resolve_from_profile` → `CategoryIntegrationInstance` with CONFIGURED_ADOPTED v1 **contract branch only** (`R2-P3-MATERIALIZATION-PORT-TYPING-04`). |
| `QualifiedCapabilityExecutionBindingHandler` | **Extend protocol** — optional `ExecutionIntegrationConfigurationAdoption` on `dispatch_once` (**parameter**, not `BoundCapabilityExecutionDispatchRequest` field) so handlers invoke Integrations boundary without neutral provider leak. |
| `ExecutionIntegrationConfigurationExecutionPinningPort` + delegate-only pin | **Demote** on configured-required paths — pin must occur inside the same boundary stack frame as first provider I/O (P3-R1 §9). |
| New conceptual name | **`ConfiguredProviderExecutionCoordinator`** — thin orchestration over extended resolution + injected category operation port; **may be implemented as methods on `ExecutionBoundIntegrationResolution` + category port** without a second resolver. |

### D.2 Caller → boundary contract (minimum fields)

**`ConfiguredProviderExecutionRequest` (conceptual — category operation envelope)**

| Field | Source | Notes |
|---|---|---|
| `tenant_id` | Caller / execution context | Must match adoption.binding.tenant_id before any provider call. |
| `execution_id` | Execution (`ExecutionId`) | Already canonical; not re-minted here. |
| `adoption` | `ExecutionIntegrationConfigurationAdoption` | Required on CONFIGURED_ADOPTED paths; carries `ConfiguredCapabilityBinding`. |
| `category_operation` | Category-specific typed request | **Not** `object`, `Any`, or string dispatch — per-category port. |
| Materialization selectors | Optional `IntegrationProfile`, `catalog_slug`; `resource_scope` from adoption unless explicitly overridden in boundary request | **Integrations-owned** typed selectors only — see §O.4. **No** `resolve_config` / generic config bag on the semantic contract. |

**`ConfiguredProviderExecutionResult` (conceptual)**

| Field | Notes |
|---|---|
| Category-specific typed result | Per port. |
| `effective` / `provenance` / `subject` | Optional echoes for diagnostics; durable truth is pin store — not a second authority. |

**Failure types (reuse):** `ExecutionIntegrationConfigurationAdoptionError`, `ExecutionIntegrationConfigurationPinningError`, category port typed errors — **no** silent EFFECTIVE_ONLY fallback when adoption required (P0 §1A).

### D.3 Provider instance lifetime

| Phase | Location |
|---|---|
| Materialize | Inside `ExecutionBoundIntegrationResolution` (or coordinator delegating to it) |
| Hold | **Local variable / closure** in Integrations-owned boundary method until first category operation completes |
| Pin | After validation, before first mutating/external provider business call (ordering §I) |
| Forbidden | Fields on `BoundCapabilityExecutionDispatchRequest`, qualified dispatch DTOs, handler registries exposed to Execution neutrality |

---

## E. Common lifecycle vs category operation

```text
COMMON (Integrations — ConfiguredProviderExecutionCoordinator / extended resolution):
  (operation-level Governance authorization on canonical path — §O.3 — before first governed provider business I/O)
  adopt? → tenant check → materialize ONCE → EffectiveIntegrationIdentity(instance)
  → validate_configured_adoption_match
  → provenance + subject
  → pin store
  → hand local instance to category port

CATEGORY-SPECIFIC (provider implementation via platform category contract):
  port.execute_configured_operation(local_instance, category_operation) -> CategoryResult
  — examples are per IntegrationCategory contracts, NOT a shared semantic execute()
```

**Tool catalog note:** `ExecutionBoundCatalogToolInvoker` may remain the **tool-category I/O surface**, but configured adoption **must** receive wiring/materialization derived from the **same** pinned local integration instance (or an explicit typed adapter from that instance). `ToolInvocationWiringResolver` **must not** become a parallel integration resolver for configured adoption.

---

## F. Exactly-one ownership matrix

### F.1 Governance / authority stages (R1-R3 — no inferred permission)

| Stage | Owner | Meaning |
|---|---|---|
| `CONFIGURE_EXISTING` decision | AW acquisition (`WorkerCapabilityAcquisitionDecisionService`) | Proposal / selection only — not permission |
| INT-CONFIG realization authorization | Governance (control-plane mutation) | Permission to **mutate/configure** binding — **not** provider business operation |
| Execution admission | Execution (`ExecutionRuntime`, active `ExecutionId`) | Legal execution lifecycle — **not** operation-level side-effect permission |
| Operation-level authorization | Canonical execution-time Governance boundary (existing ports on the normal handler/domain path — e.g. agent runtime governance, meaningful side-effect authorization, canonical inner guard where wired) | Permission for the **actual governed operation** before provider business I/O |
| Provider materialization | Integrations (`ExecutionBoundIntegrationResolution` / coordinator) | Provider selection + instance creation for configured adoption — **not** policy owner |
| Configured/effective validation | Integrations (`validate_configured_adoption_match`) | Factual match — **not** permission |
| Provenance pin | Integrations (`ExecutionIntegrationConfigurationPinningStore`) | **Evidence** of configured/effective facts — **pinning creates evidence; pinning does NOT create permission** |
| Provider business operation | Category-specific typed port / `PlatformIntegrationContract` implementation | First governed external/mutating I/O |

**Non-inference (locked):** configuration mutation ALLOW ≠ execution admission ≠ operation authorization ≠ provider invocation. No authority may be inferred across these stages.

### F.2 Composition ownership (unchanged from R1-R2)

| Concern | Owner |
|---|---|
| Acquisition decision (`CONFIGURE_EXISTING`) | `WorkerCapabilityAcquisitionDecisionService` |
| CONFIGURE_EXISTING orchestration | `WorkerCapabilityFulfillmentCoordinator` |
| Configuration opportunity | Integrations (`ExistingCapabilityConfigurationOpportunity`) |
| Configuration realization | Integrations INT-CONFIG (`ExistingCapabilityConfigurationRealizationPort`) |
| Governance authorization (realization) | Governance |
| Configured binding identity | `ConfiguredCapabilityBinding` from realization — no rediscovery |
| Explicit configured adoption | AW fulfillment → `ExecutionIntegrationConfigurationAdoption` |
| Provider resolution / materialization | Integrations (`resolve` / `resolve_from_profile` via `ExecutionBoundIntegrationMaterializationPort`) |
| Effective identity observation | Integrations (from **local** materialized instance — P0 §1A.3) |
| Configured/effective validation | Integrations (`validate_configured_adoption_match`) |
| Provenance pin | Integrations (`ExecutionBoundIntegrationResolution` + `ExecutionIntegrationConfigurationPinningStore`) |
| **Configured provider execution composition** | Integrations **`ConfiguredProviderExecutionCoordinator`** (single composition owner for materialize→pin→operate) |
| Execution identity | Execution (`ExecutionRuntime`, active execution context) |
| Provider business operation | Category-specific port / provider contract implementation |
| Provenance reconstruction | `ExecutionReconstructor` (neutral reader) |
| Tool wiring projection | ToolRuntime — **subordinate** to category execution port on configured paths; **not** configuration authority |

---

## G. Layer dependency graph

**Permitted (downward / inward)**

```text
applications → agents → intergrax (runtime, integrations, contracts)
execution handlers → Integrations configured provider execution port (typed)
category ports → PlatformIntegrationContract / category contracts
Integrations coordinator → registry factory, pinning store, adoption validators
```

**Forbidden**

| From | To | Reason |
|---|---|---|
| `intergrax/contracts/execution/*` neutral DTOs | live provider instances | P0 §17 |
| Execution delegate | sole owner of pin without category coupling | `R2-P3-EFFECTIVE-USE-CAUSALITY-03` |
| ToolRuntime wiring resolver | integration configuration / adoption truth | task §10 |
| Governance | provider materialization or category business APIs | separation |
| AW | provider APIs or configuration semantics beyond opaque ref | P0 §1B |
| Marketplace handler | ownership of resolve/pin truth | consumer of Integrations port only |
| Second provider registry | parallel materialization | bypass resistance |

---

## H. Pluginability / replaceability

**Pluginability (FRZ-PLG-*):** External provider packages register via existing Integrations catalog / `PlatformIntegrationContract` implementations. Category operation ports depend on **contract types**, not concrete vendor classes. No Execution/Governance/AW changes required to add a provider that satisfies an admitted category contract.

**Replaceability (FRZ-RPL-*):** Provider A → B via configuration/resolution (`provider_id`, profile, catalog slug) without consumer code changes, provided both implement the same category contract and admission class is `CONFIGURED_ADOPTION_EXECUTION_SUPPORTED`.

**Anti-patterns (qualification failure):** `getattr`/`setattr` dispatch, `cast`/`type: ignore` to bridge contract mismatch, `dict[str, Any]` operation payloads.

---

## I. Failure ordering (fail-closed matrix)

| Order | Condition | Provider business call | Pin |
|---|---|---|---|
| 1 | `tenant_id` ≠ adoption.binding.tenant_id | **0** | **0** |
| 2 | Unsupported category / not admitted | **0** | **0** |
| 3 | `EXTERNAL_WORK` / excluded | **0** | **0** |
| 4 | Resolution / materialization failure | **0** | **0** |
| 5 | Identity unavailable / not `PlatformIntegrationContract` when required | **0** | **0** |
| 6 | Configured ≠ effective (provider, scope, category) | **0** | **0** |
| 7 | Subject/provenance validation failure | **0** | **0** |
| 8 | Pin store failure | **0** | **0** |
| 9 | Adoption required but missing on configured-required handler path | **0** | **0** |
| 10 | Operation-level Governance **DENY** (canonical boundary) | **0** | **0** — pin/adoption confer **no** authority |
| 11 | Required operation authorization **unavailable** (fail closed) | **0** | **0** |
| 12 | Config typing / materialization selector contract violation (§O.4) | **0** | **0** — **no** generic payload fallback |
| 13 | Success path | **1** (same local instance) | **1** (before first external/mutating call; after required operation authorization — §O.3) |

**Structural impossibility (locked):** pin success **without** required operation authorization when that authorization is a prerequisite for provider business I/O → qualification **FAIL** (must be enforced by call ordering, not prose).

**Second resolve** after successful pin for the same adoption subject → **architecture FAIL** (qualification gate).

---

## J. Tenant isolation

Single `tenant_id` chain: fulfillment → adoption.binding → resolution request → provenance → pin store → reconstruction reader. Any cross-tenant hint in resolution hints → fail at step 1. No global FRZ-TEN promotion in this doc-only child.

---

## K. Category classification (admission)

| Class | Meaning | CONFIGURED_ADOPTED v1 |
|---|---|---|
| **`CONFIGURED_ADOPTION_EXECUTION_SUPPORTED`** | May use full coordinator lifecycle + typed category port | **Pilot categories only** — must be explicitly listed when implementation wave starts; **not fabricated @ this lock** |
| **`EFFECTIVE_ONLY`** | Pin mode without configured slice; no configured adoption enforcement on operation path | Allowed where P0 §1A.7 already permits |
| **`BOOTSTRAP_OR_INFRASTRUCTURE`** | Composition/DI-only; not consumer configured-adoption certification | No coordinator required for FRZ-TRC-11 configured path |
| **Excluded** | `ExternalWorkIntegration` | **Forbidden** in CONFIGURED_ADOPTED v1 (P0 C1) |

**Handler classification @ `d39b39ee…` (closed-world):**

| Handler | Class |
|---|---|
| `CodeCraftQualifiedCapabilityExecutionHandler` | Unrelated substrate — not Integrations configured-adoption provider path |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | May **call** Integrations category port when adoption present — **not** semantic owner |
| Others in inventory | Unrelated until explicitly admitted |

**Future admission:** new category requires architecture note + typed category port + qualification gates — no silent enum expansion.

---

## L. Implementation wave (bounded file families — not implemented here)

1. **Integrations execution-bound resolution / contracts** — extend `ExecutionBoundIntegrationResolution`, materialization port typing, coordinator façade, category port protocol(s).
2. **First admitted `CONFIGURED_ADOPTION_EXECUTION_SUPPORTED` category** — one typed operation port + provider contract wiring (minimal vertical slice).
3. **Execution / AW handoff** — handler protocol adoption parameter; relocate pin off delegate-only path for configured-required handlers; composition root continuity (`R2-P3-PINNING-COMPOSITION-CONTINUITY-02`).
4. **Applications composition** — wire coordinator + handler + pinning store in `build_worker_recovery_governed_fulfillment_wiring` lineage (single path).
5. **Qualification** — extend `tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py` family + causality/typing gates (§M).

**Explicitly out of scope for bounded P3:** ToolRuntime core rewrite, all providers, all categories, generic Execution contract expansion beyond handler protocol, Governance core changes.

---

## M. Qualification plan (future implementation)

| Family | Intent |
|---|---|
| **Positive causality** | A configured → materialize A → observe A → pin A → same object executes operation |
| **Negative** | A/B mismatch; tenant mismatch; pin failure; unsupported category; missing adoption; second resolve; provider on neutral DTO |
| **Structural** | one resolution owner; one configured execution composition owner; no ToolRuntime as config authority; no delegate-only decoupled pin on configured path |
| **Typing** | FRZ-TYP-01..04 — no semantic `Any`/`object` on operation seams |
| **Regression** | existing P1/P2/P3 trace_x gate modules remain green |

---

## N. STOP conditions (implementation discoveries)

Stop with **`STOP — ARCHITECTURE DECISION REQUIRED`** if:

- configured operation requires provider instance on neutral Execution DTO;
- only a generic `execute()` can unify categories;
- admitted category contract cannot express operations without cross-layer redesign;
- Governance or AW must call provider APIs;
- ToolRuntime wiring must own configuration truth;
- second registry/resolver required;
- ExternalWork pulled into CONFIGURED_ADOPTED v1;
- broad edits outside §L families needed.

---

## Closed-world Q&A (§8)

| # | Answer |
|---|---|
| **Q1 Semantic owner** | Binding/adoption: AW fulfillment + Integrations validation; materialization + effective identity + pin + execution composition: **Integrations**; ExecutionId: **Execution**; business operation: **category port / provider contract**; Governance: **permission only**. |
| **Q2 Contract shape** | §D.2 — coordinator request = tenant + ExecutionId + adoption + typed category_operation + optional Integrations resolution hints. |
| **Q3 Lifetime** | Local to Integrations/category boundary method stack; never neutral Execution DTOs. |
| **Q4 Pluginability** | Yes — via catalog + category contracts + port injection (§H). |
| **Q5 Replaceability** | Yes — configuration/resolution swap (§H). |
| **Q6 Operation typing** | Yes — per-category typed operations; no generic semantic contract. |
| **Q7 Causality** | Single materialize → identity from instance → validate → pin → same reference → category port call (§C, §E). |
| **Q8 Failure ordering** | §I. |
| **Q9 Governance** | INT-CONFIG ALLOW ≠ operation permission; operation authorization on existing canonical boundary before provider business I/O; Integrations never ALLOW/DENY (§O). |
| **Q10 Execution** | No new identity minting; ExecutionId consumed from active context only (§C). |

---

## P3 blocker resolution mapping

| Blocker | Lock resolution |
|---|---|
| `R2-P3-CONFIGURE-EXISTING-REACHABILITY-01` | Unchanged — `WorkerCapabilityAcquisitionDecisionService` remains sole decision owner; fulfillment propagates adoption only. |
| `R2-P3-PINNING-COMPOSITION-CONTINUITY-02` | Single wiring: store → resolution/coordinator → handler path in use. |
| `R2-P3-EFFECTIVE-USE-CAUSALITY-03` | **Owned here** — Pattern A coordinator + same-instance operation (§C). |
| `R2-P3-MATERIALIZATION-PORT-TYPING-04` | Typed `CategoryIntegrationInstance` / contract branch; **remove** semantic `resolve_config` / `Mapping[str, object]` from execution-bound request + port (§O.4). |
| `R2-P3-GOVERNANCE-CONTINUITY-05` | **Owned in R1-R3** — operation authorization before provider business I/O; demote delegate-only pin-before-handler for configured-required paths (§O.3). |

---

## O. TRACE-X-P5-R2-P3-R1-R3 — Governance continuity + `resolve_config` typing reconciliation

**START_HEAD:** `63ba8a237449ce7e78043c480dff97ed1f96918e` · **Production delta:** 0 · **Pattern A:** preserved (§C–E).

### O.1 Audit closure scope

Independent audit @ `63ba8a23…` accepts Pattern A (single materialization → effective identity → validate → pin → **same** local provider → category-specific typed operation) but requires:

1. execution-time **Governance authorization continuity** before provider business I/O;
2. **strong typing** of provider resolution configuration — no semantic `resolve_config` / generic mapping at the configured-provider execution boundary.

This section reconciles those items **without** reopening Pattern A, single materialization owner, same-instance continuity, or ToolRuntime/configuration authority separation.

### O.2 Four-way permission distinction (locked)

| # | Stage | Not the same as |
|---|---|---|
| 1 | Configuration mutation authorization (INT-CONFIG Governance ALLOW) | Operation permission or execution admission |
| 2 | Execution admission (`ExecutionId` active) | Operation permission or configured binding |
| 3 | Operation-level Governance authorization (canonical execution-time boundary) | INT-CONFIG ALLOW, adoption, or pin |
| 4 | Provider invocation (typed category operation) | Any prior stage |

**Integrations MUST NOT:** make ALLOW/DENY decisions; synthesize Governance evidence; treat configured binding or pin as permission; bypass ToolRuntime/domain authorization; become a second Governance owner.

### O.3 Locked execution ordering (configured-adoption provider path)

**Target shape (ownership/call ordering @ `63ba8a23…` evidence):**

```text
canonical Execution admission (ExecutionId available)
  → qualified handler / domain operation path entry
  → applicable canonical operation-level Governance authorization
        (existing ports on the normal path — no new Integrations mechanism)
  → Configured Provider Execution Boundary (Pattern A coordinator)
        materialize ONCE
        → EffectiveIntegrationIdentity (same local instance)
        → validate_configured_adoption_match
        → provenance + subject
        → pin (evidence only)
        → SAME local instance → category-specific typed operation (first provider business I/O)
```

**Invariant constraints (all admitted categories):**

| Constraint | Rule |
|---|---|
| ExecutionId | Available before boundary materialization on configured-required paths |
| Operation authorization | Applicable canonical authorization **MUST succeed** before the **first** governed provider **business** I/O (external/mutating call on the category contract) |
| Pin | Evidence only — **never** substitutes operation authorization |
| Materialization side effects | If materialization performs external/mutating I/O, it **MUST NOT** run before required operation authorization (category may place authorization earlier, never later) |
| Post-pin | No re-resolution between successful pin and provider call; pin failure ⇒ provider call = 0 |
| Authority widening | No downstream stage may widen permission implied by an earlier stage |

**Before graph @ `63ba8a23…` (production gap — not audit PASS for FRZ-TRC-11):**

```text
Execution admission
  → QualifiedCapabilityExecutionRuntimeDelegate.execute
  → pin_configured_adoption_for_execution (materialize + validate + pin)  ← before handler
  → handler.dispatch_once (e.g. Marketplace tool path)
  → ExecutionBoundCatalogToolInvoker / ToolRuntime
  → agent_runtime_governance + meaningful_side_effect_authorization (inside tool invoker composition)
  → first tool/provider effect
```

**After graph (architecture lock):** handler (or domain path) reaches **operation-level authorization** before the coordinator performs any materialization that is not strictly non-mutating identity observation, then coordinator stack as in §C with same-instance operation. Delegate-only pin-before-handler is **demoted** for configured-required paths (`R2-P3-GOVERNANCE-CONTINUITY-05`).

**Pilot category:** R1-R2 did **not** certify a `CONFIGURED_ADOPTION_EXECUTION_SUPPORTED` category where integration category == provider operation category. **No pilot name is locked here.** Future admission requires proof per §K; Marketplace qualified tool handler is **not** evidence of configured integration adoption alignment (tool I/O ≠ `ExecutionIntegrationConfigurationAdoption.integration_category` without explicit admission).

**Bounded code evidence (governance ports exist on tool path, not on Integrations pin):**

- `intergrax/runtime/execution/execution_bound_catalog_tool_composition.py` — wires `AgentRuntimeGovernanceBoundary`, `MeaningfulSideEffectAuthorizationPort`, `CanonicalInnerExecutionGuardPort` into production tool invoker.
- `intergrax/tools/marketplace_qualified_capability_execution_handler.py` — `catalog_tool_invoker.invoke` after handler-local validation; does not consume adoption on `dispatch_once`.
- `intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py` — pins when adoption present **before** `handler.dispatch_once`.

### O.4 Configuration typing decision — `resolve_config` removed from semantic contract

**Architecture outcome:** **Option B** (binding/profile capture) **+** **Option A** (existing typed profile contract) — **not** Option C.

| Semantic boundary input | Typed owner | Role |
|---|---|---|
| `ExecutionIntegrationConfigurationAdoption` | Integrations contracts (`execution_integration_configuration.py`) | `ConfiguredCapabilityBinding` + category + `resource_scope` — configured identity |
| `IntegrationProfile` (optional) | Integrations contracts (`integration_profile.py`) | Declarative per-category `IntegrationBinding` slots; per-slug factory options via **`IntegrationProfile.options_for_slug`** (profile-internal; not a separate execution-bound bag) |
| `catalog_slug` (optional) | Integrations registry semantics | Catalog factory path when profile slot absent |
| INT-CONFIG realization payload | `IntegrationConfigurationPayload` (+ codecs) | **Configuration-time only** — fingerprinted in binding; **not** re-exposed as `Mapping[str, object]` on execution boundary |

**Removed from configured-provider **semantic** contract:** `resolve_config`, `Mapping[str, object]`, `Mapping[str, Any]`, `dict[str, Any]`, and any arbitrary metadata bag at the Pattern A coordinator / `ConfiguredProviderExecutionRequest` envelope.

**Implementation debt @ `63ba8a23…` (internal only until `R2-P3-MATERIALIZATION-PORT-TYPING-04`):**

- `ExecutionBoundIntegrationResolutionRequest.resolve_config: Mapping[str, object] | None`
- `ExecutionBoundIntegrationMaterializationPort.resolve_*` `config: Mapping[str, object] | None`
- `intergrax/integrations/registry/factory.py` `config: Optional[Mapping[str, Any]]`

**Exact typed target for port typing-04:** delete execution-bound `resolve_config` field; materialization port accepts **`IntegrationProfile | None`**, **`catalog_slug: str | None`**, and category from adoption — factory `config` merge, when needed, is derived **inside Integrations** from `IntegrationProfile.options_for_slug` for the resolved slug, not from a caller-supplied generic mapping. Pinning adapter @ `63ba8a23…` already omits `resolve_config` (composition-injected profile/slug only).

### O.5 R1-R3 enterprise self-audit (incremental)

| Criterion | R1-R3 result |
|---|---|
| Governance continuity | **PASS @ lock** — §O.3 invariants + §F.1 matrix |
| Pin ≠ permission | **PASS @ lock** — explicit §F.1 |
| Strong typing @ semantic boundary | **PASS @ lock** — §O.4; **OPEN @ code** until typing-04 |
| Bypass resistance | **PASS @ lock** — Integrations forbidden policy role §O.2 |
| Production ordering | **OPEN** — `R2-P3-GOVERNANCE-CONTINUITY-05` |

### O.6 Applicable FRZ (R1-R3)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | **OPEN** — governance ordering + typing must be **implemented** to close |
| FRZ-GOV-* (permission vs execution) | Supporting — §O.2, §F.1 |
| FRZ-EXE-01 | Supporting — ExecutionId authority unchanged |
| FRZ-CTR-01, FRZ-CTR-02 | Supporting |
| FRZ-TYP-01..04 | Supporting — §O.4 |
| FRZ-PLG-01..05, FRZ-RPL-01/02/04 | Supporting — §H |

### O.7 R1-R3 disposition

| Item | Status |
|---|---|
| `TRACE-X-P5-R2-P3-R1-R3` | **READY FOR AUDIT** |
| `TRACE-X-P5-R2-P3-R1-R2` | Superseded for governance/typing closure only — Pattern A unchanged |
| `TRACE-X-P5-R2-P3` | **BLOCKED PENDING ARCHITECTURE AUDIT / IMPLEMENTATION** |
| P4 | **NOT ENTERED** |

---

## P. TRACE-X-P5-R2-P3-R1-R4 — Configured-Adoption Pilot Admission

**START_HEAD:** `fcfa59797c86f5933a7e53cb6784768a4f1d1787` · **Production delta:** 0 · **Pattern A (R1-R3):** not reopened.

### P.1 Required question — answer **B (NO)**

> Does current HEAD already contain one legal category/provider/caller combination that can satisfy the accepted Pattern A without introducing a new cross-layer semantic contract, new authority, new ToolRuntime wiring mechanism, or broad typing redesign?

**Answer: NO.** No candidate in the bounded closed-world inventory (§P.2) passes **all** admission gates PA-01..PA-10 with code evidence.

**Disposition:** **`STOP — ARCHITECTURE DECISION REQUIRED`**

**Architecture blocker:** **`R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06`** — Pattern A is locked (R1-R3), but @ `fcfa59797c86f5933a7e53cb6784768a4f1d1787` there is **no** existing production vertical slice that simultaneously proves configured category alignment, strong typed provider operation, Governance continuity on the category business call, and same-instance provider causality from configured adoption through pin to business I/O.

**Admitted pilot:** **none** — silent promotion of Marketplace tool execution or profile-resolved database tools is **forbidden**.

**Minimum next architecture decision (design only — not chosen here):** introduce an explicit typed mechanism so an **already materialized** category provider instance from the Pattern A coordinator can reach an **existing governed** category operation consumer (e.g. typed configured-provider execution port and/or typed dependency projection from pin/coordinator into that caller) **without** ToolRuntime becoming configuration authority and **without** weakening `RelationalStore` or generic tool transport into the semantic boundary.

### P.2 Bounded candidate inventory (@ `fcfa59797c86f5933a7e53cb6784768a4f1d1787`)

| Candidate | Role @ HEAD | Pre-audit disposition |
|---|---|---|
| **CodeCraft** | `CodeCraftQualifiedCapabilityExecutionHandler` → `CodeCraftBoundCapabilityExecutionPort.execute` | **N/A** — not Integrations configured-provider execution |
| **Marketplace qualified tool** | `MarketplaceToolQualifiedCapabilityExecutionHandler` → intent/stage/material → `ExecutionBoundCatalogToolInvoker` → ToolRuntime | **NOT ADMITTED** — no typed relation `adoption.integration_category` → integration provider used by tool |
| **`RELATIONAL_STORE` / `sqlite`** | CONFIGURE_EXISTING: `SQLiteRelationalStoreConfigurationRealizationStrategy` + `WorkerConfiguredCapabilityFulfillmentService` adoption; separate consumer: `database.*` tools via `ToolWiringContext.relational_store` from `ToolWiringContext.from_integration_profile` | **CANDIDATE ONLY** — fails PA-02, PA-04, PA-05, PA-08 (and related gates) |

**Qualified handler registry @ HEAD:** two real binding handlers — CodeCraft (`intergrax/runtime/codecraft/qualified_capability_execution_handler.py`), Marketplace tool (`intergrax/tools/marketplace_qualified_capability_execution_handler.py`). Neither implements a category-specific configured-provider operation port for `ExecutionIntegrationConfigurationAdoption.integration_category`.

**CONFIGURE_EXISTING → Execution (partial chain exists):** `WorkerCapabilityFulfillmentCoordinator` passes `integration_configuration_adoption` into qualified execution intake (`worker_capability_fulfillment_coordinator.py`); `QualifiedCapabilityExecutionRuntimeDelegate` calls `pin_configured_adoption_for_execution` before `handler.dispatch_once` (`qualified_capability_execution_runtime_delegate.py`); pin adapter delegates to `ExecutionBoundIntegrationResolution.resolve_and_pin` (`execution_bound_integration_pinning_adapter.py`). **Gap:** pin path observes effective **identity** and stores provenance; it does **not** retain or forward the materialized `PlatformIntegrationContract` instance to any category business operation (`execution_bound_integration_resolution.py` — `materialized` used only for `provider_id`, instance discarded).

### P.3 Call graphs (code-bounded)

**CodeCraft**

```text
QualifiedCapabilityExecutionRuntimeDelegate.execute
  → [optional pin — adoption not consumed by handler]
  → CodeCraftQualifiedCapabilityExecutionHandler.dispatch_once
  → CodeCraftBoundCapabilityExecutionPort.execute(CodeCraftBoundCapabilityExecutionRequest)
```

No `ExecutionIntegrationConfigurationAdoption`, no Integrations category, no `CONFIGURED_ADOPTION_EXECUTION_SUPPORTED` path.

**Marketplace qualified tool**

```text
WorkerCapabilityFulfillmentCoordinator (may attach adoption)
  → QualifiedCapabilityExecutionRuntimeDelegate.execute
  → pin_configured_adoption_for_execution (ExecutionBoundIntegrationResolution.resolve_and_pin)
  → MarketplaceToolQualifiedCapabilityExecutionHandler.dispatch_once
  → intent / stage / activation / material / invocation resolution
  → ExecutionBoundCatalogToolInvoker.invoke
  → ToolRuntime (governance wired in tool invoker composition)
  → tool operation
```

`MarketplaceToolQualifiedCapabilityExecutionHandler.dispatch_once` does not read `integration_configuration_adoption` or `integration_category` (handler API is `BoundCapabilityExecutionDispatchRequest` only). Tool identity is marketplace/catalog semantics, not adoption category proof.

**`RELATIONAL_STORE` / `sqlite` (discontinuous)**

```text
CONFIGURE_EXISTING path:
  WorkerConfiguredCapabilityFulfillmentService
    → ExecutionIntegrationConfigurationAdoption(integration_category=opportunity.integration_category, …)
  → coordinator → qualified execution + pin (as above)

Database tool path (independent):
  ToolWiringContext.from_integration_profile
    → profile.slug_for_category(RELATIONAL_STORE) / resolve_from_profile
    → ctx.relational_store
  → database_query | database_execute | database_describe_schema
    → RelationalStore.fetch_all | execute
```

No code edge proves `ctx.relational_store` is the **same object** materialized during `resolve_and_pin` for the execution carrying the adoption.

### P.4 Typed contracts found (candidate-relevant)

| Surface | Location | P3 pilot relevance |
|---|---|---|
| `RelationalStore` | `intergrax/integrations/contracts/relational_store.py` | `execute(..., params: Sequence[Any])`, `fetch_all` → `Sequence[Mapping[str, Any]]` — **fails PA-08** for category business seam |
| Database tool DTOs | `intergrax/tools/providers/database/contracts.py` (via `service.py`) | Strongly typed **tool** inputs/outputs; tool layer ≠ configured-provider category port |
| `ExecutionIntegrationConfigurationAdoption` | Integrations contracts | Neutral adoption envelope — not a category operation port |
| `ExecutionBoundIntegrationResolutionResult` | `execution_bound_integration_resolution.py` | Returns `effective` identity + provenance — **no** pinned provider instance handle for consumers |
| CodeCraft / Marketplace dispatch | qualified handlers | Typed **execution** dispatch — not category provider operations |

### P.5 Governance boundary found

| Path | Governance | Pilot relevance |
|---|---|---|
| Marketplace → ToolRuntime | `AgentRuntimeGovernanceBoundary` + meaningful side-effect authorization on catalog tool invoker composition (`execution_bound_catalog_tool_composition.py` per §O.3) | **Does not** tie authorization to adoption category or materialized integration provider instance |
| Pin / `resolve_and_pin` | Validates adoption vs effective identity; pins provenance — **not** operation-level Governance before category business I/O on a retained provider | Materialization for identity observation can occur in pin path before handler (§O.3 **OPEN** `R2-P3-GOVERNANCE-CONTINUITY-05`) |
| Database tools | ToolRuntime governance when invoked as tools | **Separate** from configured-adoption execution port; profile-based store resolution is not adoption causality |

### P.6 Same-instance continuity — **FAIL @ HEAD (all candidates)**

- **Pin path:** `resolve_and_pin` materializes via `ExecutionBoundIntegrationMaterializationPort` but returns only `EffectiveIntegrationIdentity` / provenance — **no** consumer receives the materialized instance (`execution_bound_integration_resolution.py`).
- **Tool wiring:** `ToolWiringContext.from_integration_profile` may call `resolve_from_profile` / `resolve` again for `RELATIONAL_STORE` (`wiring.py`) — **second resolution**, not pin-projected instance (**PA-05**).

### P.7 Tenant continuity

| Candidate | Result |
|---|---|
| CodeCraft | **N/A** (no configured adoption path) |
| Marketplace | Fulfillment/adoption tenant checks exist on fulfillment and handler intent (`intent.tenant_id != request.tenant_id`); **cannot** prove provider-operation tenant equals adoption tenant for integration category — tool path lacks adoption category/provider binding |
| `RELATIONAL_STORE` / `sqlite` | Fulfillment validates binding tenant/category/provider (`worker_configured_capability_fulfillment_service.py`); database tools use execution/tool tenant via ToolRuntime — **no proof** same configured binding drives the `RelationalStore` instance used in SQL I/O |

**FRZ-TEN-*:** no global PASS; tenant alignment on adoption creation does **not** admit pilot without operation causality.

### P.8 Pluginability / replaceability

| Candidate | PA-09 |
|---|---|
| CodeCraft | **N/A** |
| Marketplace | **FAIL** — replaceability of integration provider behind adoption is **unproven**; tool catalog slug ≠ configured `provider_id` causality |
| `RELATIONAL_STORE` / `sqlite` | **FAIL @ pilot** — sqlite could be replaced in principle via Integrations registry, but **without** admitted same-instance port, replaceability does not satisfy Pattern A vertical slice |

### P.9 Pilot admission gate matrix (PA-01..PA-10)

Legend: **PASS** = PASS WITH CODE EVIDENCE · **FAIL** = FAIL WITH CODE EVIDENCE · **N/A**

| Gate | CodeCraft | Marketplace tool | `RELATIONAL_STORE` / `sqlite` + database tool |
|---|---|---|---|
| **PA-01** Same semantic category | N/A | **FAIL** — handler/tool path has no `adoption.integration_category` ≡ tool integration category proof | **FAIL** — adoption category `RELATIONAL_STORE` not wired to tool store instance |
| **PA-02** Existing typed category operation | N/A | **FAIL** — tool ops typed as tool contracts, not integration category port | **FAIL** — `RelationalStore` uses `Any` / `Mapping[str, Any]` |
| **PA-03** Existing governed caller | N/A | **PASS** — ToolRuntime governance on invoke | **PASS** — tool path governed; **FAIL** as configured-adoption category caller (discontinuous) |
| **PA-04** Same-instance continuity | N/A | **FAIL** — pin does not pass provider; tool resolves independently | **FAIL** — `from_integration_profile` vs pin materialization |
| **PA-05** No second resolver | N/A | **FAIL** — tool/materialization chain re-resolves tool wiring | **FAIL** — `_optional(RELATIONAL_STORE)` re-resolve |
| **PA-06** No ToolRuntime authority mutation | N/A | **PASS** — ToolRuntime does not select adoption | **FAIL** — profile slug selection is ToolRuntime wiring authority, not adoption-projected instance |
| **PA-07** No Governance bypass | N/A | **FAIL** — pin/materialization ordering vs operation auth (§O.3 OPEN) for configured-required story | **FAIL** — SQL via profile wiring bypasses configured-provider coordinator operation auth |
| **PA-08** Strong typing | N/A | **PASS** on tool DTOs; **FAIL** on adoption→provider category seam | **FAIL** — `relational_store.py` `Any` |
| **PA-09** Pluginability | N/A | **FAIL** — unproven adoption-aligned replaceability | **FAIL** — no admitted port |
| **PA-10** Bounded P3 implementation | N/A | **FAIL** — would require new adoption↔tool semantic edge | **FAIL** — requires typed port + instance projection and/or coordinator consumer |

### P.10 Impact on P3 · FRZ · tenant

| Item | Status |
|---|---|
| `TRACE-X-P5-R2-P3` | **NEXT / REQUIRED / NOT ENTERED** — blocked on **`R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06`** before first `CONFIGURED_ADOPTION_EXECUTION_SUPPORTED` implementation wave |
| `TRACE-X-P5-R2` | **CURRENT / P3 NEXT** (unchanged) |
| `P5-GAP-04` | **IMPLEMENTATION IN PROGRESS** |
| **FRZ-TRC-11** | **OPEN** — no pilot ⇒ no production configured→effective→operation causality proof |
| FRZ-CTR-01/02, FRZ-TYP-01..04, FRZ-PLG-01..05, FRZ-RPL-01/02/04, FRZ-EXE-01, FRZ-GOV-* | Supporting — pilot admission **does not** promote PASS |
| **FRZ-TEN-*** | **OPEN** — tenant checks on fulfillment insufficient without admitted operation path |

**Why no silent promotion:** Marketplace success at governed **tool** execution does not prove `ExecutionIntegrationConfigurationAdoption` category alignment; database tools using `RelationalStore` from **IntegrationProfile** do not prove configured-adoption materialization continuity; CodeCraft is outside Integrations configured-provider scope.

### P.11 R1-R4 disposition

| Item | Status |
|---|---|
| **`TRACE-X-P5-R2-P3-R1-R4`** | **`STOP — ARCHITECTURE DECISION REQUIRED`** (`R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06`) |
| Admitted pilot | **none** |
| `TRACE-X-P5-R2-P3-R1-R3` | **READY FOR AUDIT** (unchanged — Pattern A) |
| P4 | **NOT ENTERED** |

### P.12 Unresolved findings (R1-R4 classification)

| Finding | Class |
|---|---|
| `R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06` | **IN-SCOPE BLOCKER** |
| No production Pattern A coordinator + category port @ HEAD | **IN-SCOPE BLOCKER** (carried from §O) |
| `R2-P3-GOVERNANCE-CONTINUITY-05` | **IN-SCOPE BLOCKER** |
| `R2-P3-MATERIALIZATION-PORT-TYPING-04` | **TRACKED FREEZE DEBT** |
| `RelationalStore` `Any` on operation seam | **IN-SCOPE BLOCKER** for sqlite-as-pilot without new typed contract |

---

## Enterprise self-audit matrix (@ lock design)

| Criterion | Result |
|---|---|
| Layer boundaries | **PASS** — provider local to Integrations/category |
| Communication direction | **PASS** — handlers call Integrations port inward |
| Semantic ownership | **PASS** — §F |
| Composition ownership | **PASS** — single coordinator owner |
| Contracts over implementations | **PASS** |
| Strong typing | **PASS** — explicit bans §H |
| Pluginability / replaceability | **PASS** — §H |
| Fail-closed / bypass resistance | **PASS** — §I, §G |
| Governance separation | **PASS** — §O.2–O.3; production ordering **OPEN** (`R2-P3-GOVERNANCE-CONTINUITY-05`) |
| Governance continuity (operation before I/O) | **PASS @ lock** — §O.3 |
| Semantic boundary typing (`resolve_config`) | **PASS @ lock** — §O.4; code debt typing-04 |
| Execution authority | **PASS** — no new lifecycle |
| Tenant continuity | **PASS** — §J |
| Traceability | **OPEN** until implementation proves FRZ-TRC-11 |
| Regression protection | **PLANNED** — §M |

**Unresolved (implementation, not lock):**

| Finding | Class |
|---|---|
| No production coordinator + category port @ `63ba8a23…` | `IN-SCOPE BLOCKER` |
| Delegate pin-before-handler; materialization may precede operation governance | `IN-SCOPE BLOCKER` (`R2-P3-GOVERNANCE-CONTINUITY-05`) |
| `resolve_config` / `Mapping[str, object]` on execution-bound request + port | `TRACKED FREEZE DEBT` (`R2-P3-MATERIALIZATION-PORT-TYPING-04`) |

---

## Applicable FRZ evidence (this child)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | Primary — **OPEN**; this lock specifies mechanical path only |
| FRZ-CTR-01, FRZ-CTR-02 | Supporting — contracts-only consumer surfaces |
| FRZ-TYP-01..04 | Supporting — typing rules §H |
| FRZ-PLG-01..05, FRZ-RPL-01,02,04 | Supporting — §H |
| FRZ-EXE-01 | Supporting — Execution identity unchanged |
| FRZ-GOV-* (as cited in P0 / INT-CONFIG) | Supporting — §O.2 permission separation |

**No global FRZ PASS promotion.**

---

## Recommended roadmap status (bookkeeping — not independent audit)

| Stage | Status |
|---|---|
| `TRACE-X-P5-R2-P3-R1-R5` | **READY FOR AUDIT** (§Q) |
| `TRACE-X-P5-R2-P3-R1-R4` | **SUPERSEDED @ architecture** for admission-06 — historical §P STOP |
| `TRACE-X-P5-R2-P3-R1-R3` | **READY FOR AUDIT** |
| `TRACE-X-P5-R2-P3-R1-R2` | Architecture base — governance/typing superseded by R1-R3 §O |
| `TRACE-X-P5-R2-P3` | **BLOCKED PENDING R1-R5 AUDIT / IMPLEMENTATION** |
| `TRACE-X-P5-R2` | **CURRENT / P3 NEXT** |
| `P5-GAP-04` | **IMPLEMENTATION IN PROGRESS** |
| `FRZ-TRC-11` | **OPEN** |

Do **not** mark P3 CLOSED. Do **not** enter P4.

---

## Q. TRACE-X-P5-R2-P3-R1-R5 — Deferred Configured Provider Dependency Projection Architecture Lock

**START_HEAD:** `62ec2457268ac7287be9c643d7d3cf1b97f32f07` · **Production delta:** 0 · **Supersedes:** R1-R4 disposition on **`R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06`** at architecture level only (§P.11 historical STOP → §Q architecture lock).

### Q.1 START_HEAD and steering alignment

| Field | Value |
|---|---|
| **START_HEAD** | `62ec2457268ac7287be9c643d7d3cf1b97f32f07` |
| **Parent** | `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **TRACE-X-P5-R2** | **CURRENT / P3 NEXT** |
| **TRACE-X-P5-R2-P3** | **NEXT / REQUIRED / NOT ENTERED** (architecture unblocked for implementation wave; parent not CLOSED) |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **FRZ-TRC-11** | **OPEN** — no PASS |
| **P4** | **NOT ENTERED** |

Canonical trackers @ START_HEAD match expected: `development` = `origin/development` = `62ec2457268ac7287be9c643d7d3cf1b97f32f07`.

### Q.2 Selected architecture (mandatory — not open for rediscovery)

**Deferred, post-Governance, invocation-scoped typed category dependency projection.**

```text
qualified Execution
  → handler receives factual adoption (separate from neutral catalog invoke request)
  → ExecutionBoundCatalogToolInvokeRequest carries per-invocation ToolInvocationWiringResolver
  → ToolRuntime: canonical Governance (inner guard → agent runtime → declarative policy → MSE when applicable)
  → ToolRuntime: invocation dependency projection (_apply_invocation_wiring)
  → ToolExecutor
  → database tool consumes ONE typed relational execution port
  → port invokes Integrations Pattern A lazily on first category operation
  → materialize provider P ONCE per bound port lifecycle
  → effective identity(P) from SAME P
  → validate configured/effective
  → pin
  → SAME P performs category operation
```

**Hard invariant:** provider object **NEVER** crosses Execution contracts, `ExecutionBoundCatalogToolInvokeRequest`, or `ToolInvocationWiring`.

**Resolves §P minimum next decision:** typed execution-scoped port + sanctioned projection bridge — **without** ToolRuntime configuration authority and **without** promoting weak `RelationalStore` to the semantic boundary.

### Q.3 RuntimeToolInvoker ordering evidence (@ `62ec2457268ac7287be9c643d7d3cf1b97f32f07`)

Production structure (code-bounded):

```text
RuntimeToolInvoker.invoke
  → _prepare_invocation
      → registry bind
      → _require_canonical_inner_execution_guard
      → _require_current_attempt_authorization
          → _require_agent_runtime_governance
          → scope policy
          → [sandbox: optional invocation_context.wiring_resolver.resolve — see Q.4]
          → declarative policy / meaningful side-effect authorization paths
      → input validation
  → idempotency / protected-work admission (when applicable)
  → _execute_external_effect
  → _execute_with_policy (retries re-call _require_current_attempt_authorization when attempt > 1)
  → _execute_once
      → _apply_invocation_wiring (invocation_wiring_resolver.resolve)
      → ToolExecutor.execute
```

**Evidence:** `intergrax/runtime/nexus/tools/invoker.py` — `invoke` @ L301–408; `_prepare_invocation` @ L411–488; `_require_current_attempt_authorization` @ L566+; `_execute_once` @ L1674+ with `_apply_invocation_wiring` @ L1613+ before executor.

**Consequence:** invocation-scoped configured dependency may legally reach tool execution **after** canonical operation authorization on the success path. Pattern A provider business I/O remains **downstream** of Governance when the port is only invoked from `ToolExecutor` / tool handler.

### Q.4 Resolver early-call constraint (HARD RULE)

`ToolInvocationWiringResolver.resolve(...)` **MUST** be:

- side-effect free;
- provider-I/O free;
- materialization-free;
- pin-free;
- authorization-free.

It **MAY** create/project a lightweight typed execution-scoped port object (e.g. `ExecutionBoundConfiguredRelationalStorePort`).

It **MUST NOT**:

- call Integrations `resolve` / `resolve_from_profile`;
- instantiate external provider connections;
- validate configured/effective provider by materialization;
- pin provenance;
- perform provider business I/O.

**Early invocation evidence:** when `contract.requires_sandbox_isolation`, `_require_current_attempt_authorization` calls `invocation_context.wiring_resolver.resolve` **before** sandbox admission completes (`invoker.py` L612–639). Configured projection implementations must remain safe under **multiple** `resolve()` calls (Q.9).

Actual Pattern A starts only when the projected category execution port method (`query` / `execute`) is invoked on the ToolExecutor path **after** Governance.

### Q.5 Selected pilot

| Field | Lock |
|---|---|
| **Integration category** | `IntegrationCategory.RELATIONAL_STORE` |
| **Reference configured realization** | `sqlite` (reference only — not semantic owner) |
| **Admitted tool operations** | `database.query`, `database.execute` |
| **Explicitly excluded** | `database.describe_schema` (SQLite-specific `PRAGMA` / `sqlite_master` — not replaceability proof for generic category contract) |

Pilot proves vertical slice only; alternate provider replaceability remains mandatory (Q.14).

### Q.6 Typed relational execution contract (Integrations-owned)

Do **not** expose weak transport `RelationalStore` (`Sequence[Any]`, `Mapping[str, Any]`) as the configured-provider semantic execution boundary.

**Lock (conceptual name; repository naming may vary):** `ConfiguredRelationalStoreExecutionPort`

Category operations (no universal `execute(operation: object)`):

```text
query(typed relational query request) → typed relational query result
execute(typed relational execute request) → typed relational execute result
```

**Forbidden on semantic port:** arbitrary operation strings; Tools `BaseModel` as Integrations semantic payload; Tools-layer DTO dependency from Integrations.

**Transport adapter (internal during P3):**

```text
ConfiguredRelationalStoreExecutionPort
  → Integrations category adapter
  → existing RelationalStore transport (legacy weak typing contained behind adapter)
```

Does **not** promote global **FRZ-TYP** PASS; `Any` on legacy transport remains tracked debt outside the new boundary.

### Q.7 Typed SQL scalar / value model

Explicit closed scalar domain on the new contract (no `Any`):

```text
SqlScalar = str | int | float | bool | bytes | None
```

(Expand only if repository/provider evidence requires another scalar — not for compatibility widening.)

- Query parameters: immutable sequence/tuple of `SqlScalar`.
- Query row: typed mapping/record with values from the same domain.
- Adapter **fail-closed** if provider returns unsupported runtime values — **no** widen-to-`Any`.

### Q.8 Lazy bound-port lifecycle

Project into ToolRuntime wiring — **not** the provider:

**`ExecutionBoundConfiguredRelationalStorePort`** (conceptual) holds only:

- `tenant_id`;
- canonical `ExecutionId`;
- `ExecutionIntegrationConfigurationAdoption`;
- Integrations-owned coordinator / resolution dependency;
- typed materialization selectors allowed by R1-R3 (§O.4).

**MUST NOT** materialize provider in constructor or in `ToolInvocationWiringResolver.resolve`.

**First** `query` / `execute` on the port:

```text
ConfiguredRelationalStoreExecutionPort operation
  → Integrations configured-provider coordinator (Pattern A)
  → tenant/adoption checks
  → materialize P (once per bound port — Q.9)
  → EffectiveIntegrationIdentity from SAME P
  → validate configured/effective
  → build provenance/subject → pin
  → SAME P executes relational-store operation via category adapter
```

No second resolution; no provider handoff to Tools.

### Q.9 Retry / multiple resolver invocation semantics

ToolRuntime may:

- call wiring resolver during sandbox/governance preparation (Q.4);
- call again in `_apply_invocation_wiring`;
- retry physical tool execution (`_execute_with_policy`).

**Required semantics:**

| Phase | Provider materializations |
|---|---|
| Many `ToolInvocationWiringResolver.resolve()` | **0** — same logical execution-scoped category port instance (or equivalent idempotent projection) |
| First category business operation on bound port | **1** — validate → pin → operation |
| ToolRuntime retry of same invocation | **reuse** validated/pinned provider on same bound port — **no** second materialization |

Process-local bound instance is **not** durable authority; restart/reconstruction = **P4** scope — **not** solved in R1-R5.

### Q.10 ToolInvocationWiring projection model

Extend invocation-scoped wiring with **one** typed pilot slot:

```text
ToolInvocationWiring.configured_relational_store_execution: ConfiguredRelationalStoreExecutionPort | None
```

(+ registration/effective context projection required for database tools.)

**Forbidden inside `ToolInvocationWiring`:** provider object; `PlatformIntegrationContract`; `RelationalStore`; adoption; config dict.

Only the typed execution port crosses this boundary.

### Q.11 Projection owner and qualified invocation pass-through

**Sanctioned composition surface (single owner):** `ConfiguredIntegrationToolInvocationProjectionPort`

Given factual `tenant_id`, `ExecutionId`, `ExecutionIntegrationConfigurationAdoption`, target tool identity → produce `ToolInvocationWiringResolver` **only** when category/tool pair is admitted (Q.5).

- Validate admission; fail closed on mismatch.
- No provider creation; no authorization; no registry/provider selection authority.
- MAY dispatch by `IntegrationCategory` to category-specific projection implementations (dependency projection, **not** provider selection).

**Handler adoption propagation (architecture):** extend qualified handler dispatch so Marketplace (and future configured paths) receive `integration_configuration_adoption` as a **factual typed argument** on `dispatch_once` (or repository-conformant equivalent) — **not** on `ExecutionBoundCatalogToolInvokeRequest`.

**`QualifiedToolInvocationResolver`:** smallest extension — optional caller-supplied `ToolInvocationWiringResolver` passed through to `ExecutionBoundCatalogToolInvokeRequest.wiring_resolver`. Resolver remains mapping-only (no Integrations calls).

**Permanent forbidden pattern:** `if adoption.category == RELATIONAL_STORE` embedded in generic Marketplace handler business logic as the long-term architecture.

### Q.12 Single consumer + invocation overlay precedence

Database tools converge on **one** semantic dependency: typed relational execution port.

| Composition path | Source |
|---|---|
| Ordinary non-adoption | adapter around registration-time `RelationalStore` |
| CONFIGURED_ADOPTED + admitted tool | invocation overlay replaces adapter with execution-scoped configured port |

**Forbidden permanently:**

```text
if configured_port: ... else: use old relational_store  # two semantic mechanisms
```

**Forbidden:**

```text
adoption present + admitted tool + projection missing → silent profile relational_store fallback
```

**Required:**

```text
adoption present + admitted tool → configured projection mandatory → failure = fail closed
```

Closes R1-R4 second-resolution / profile bypass gap (§P.3).

### Q.13 Governance ordering (successful path)

```text
Execution admission
  → qualified handler (+ adoption as factual arg)
  → ToolRuntime canonical inner Governance
  → agent runtime Governance
  → declarative policy
  → MSE authorization when side-effecting
  → invocation dependency projection
  → ToolExecutor
  → typed configured relational port (lazy Pattern A)
  → materialize → validate → pin → SAME provider I/O
```

- `database.execute` **MUST** prove MSE/side-effect authorization before provider write.
- `database.query` behind normal applicable ToolRuntime Governance; MSE may not apply.
- Integrations: **no** ALLOW/DENY policy decisions (§O.2).

**Note @ HEAD:** delegate `pin_configured_adoption_for_execution` before handler (§P) may still materialize for identity observation — implementation wave must align configured-required paths with **lazy** port Pattern A after ToolRuntime Governance for category business I/O (`R2-P3-GOVERNANCE-CONTINUITY-05`).

### Q.14 Tenant continuity

```text
qualified execution tenant
  == Tool invocation tenant
  == adoption.binding.tenant_id
  == Pattern A request tenant
  == provenance tenant
```

`ExecutionId` only from active canonical Execution — no new minting; correlation/request ID is **not** execution authority.

Wrong tenant → materialize **0**, pin **0**, provider I/O **0**.

### Q.15 Pluginability / replaceability

Semantic contract **MUST NOT** depend on SQLite concrete class, path/config types, or SQLite module imports in generic coordinator/tool consumer.

Qualification must include alternate in-memory/fake structural provider implementing relational category transport; proof:

```text
sqlite provider ↔ alternate relational provider
```

swapped via canonical Integrations configuration/materialization **without** changing database tool, Execution, Governance, or Marketplace handler. No global provider migration.

### Q.16 Ownership lock

| Concern | Owner |
|---|---|
| CONFIGURE_EXISTING decision | AW acquisition |
| configuration realization | Integrations / INT-CONFIG |
| configuration realization permission | Governance |
| explicit adoption | AW fulfillment using Integrations binding |
| Execution lifecycle / ExecutionId | Execution |
| operation authorization | ToolRuntime Governance |
| invocation dependency projection | sanctioned Tools/Integrations composition bridge (`ConfiguredIntegrationToolInvocationProjectionPort`) |
| provider materialization | Integrations |
| effective identity | Integrations |
| configured/effective validation | Integrations |
| provenance pin | Integrations |
| relational category execution contract | Integrations |
| provider business implementation | external/provider plugin |
| tool ABI | Tools |
| diagnostics | Observability only |

### Q.17 Failure matrix

| Condition | Materialize | Pin | Provider I/O |
|---|---:|---:|---:|
| Governance DENY | 0 | 0 | 0 |
| MSE required but unavailable | 0 | 0 | 0 |
| sandbox/pre-auth `resolve()` only | 0 | 0 | 0 |
| adoption missing on admitted configured path | 0 | 0 | 0 |
| category/tool mismatch | 0 | 0 | 0 |
| tenant mismatch | 0 | 0 | 0 |
| provider identity mismatch | ≤1 | 0 | 0 |
| pin failure | ≤1 | attempted/failed | 0 |
| invocation wiring resolution failure | 0 | 0 | 0 |
| success | 1 | 1 | same provider |
| ToolRuntime retry | no second materialization | same prior pin | same provider instance |

**Pin semantics:** pin = configured/effective fact for execution — not permission, not successful effect, not successful tool outcome. If provider business I/O occurs: `provider_object_used.provider_id == pinned provenance.effective.provider_id`.

### Q.18 Blocker mapping

| Blocker | R1-R5 resolution |
|---|---|
| **`R2-P3-CONFIGURED-ADOPTION-PILOT-ADMISSION-06`** | **Architecture RESOLVED** — RELATIONAL_STORE/sqlite `database.query` + `database.execute` vertical slice + §Q projection model |
| **`R2-P3-EFFECTIVE-USE-CAUSALITY-03`** | Same bound port; lazy Pattern A; same provider instance for pin and I/O |
| **`R2-P3-GOVERNANCE-CONTINUITY-05`** | Category business I/O only after ToolRuntime Governance; resolver/pre-auth calls materialization-free |
| **`R2-P3-PINNING-COMPOSITION-CONTINUITY-02`** | Configured port via same `wiring_resolver` / `_apply_invocation_wiring` path ToolExecutor consumes |
| **`R2-P3-MATERIALIZATION-PORT-TYPING-04`** | **Implementation OPEN** — remove `resolve_config` / weak materialization typing per R1-R3 §O.4 |
| **`R2-P3-CONFIGURE-EXISTING-REACHABILITY-01`** | AW CONFIGURE_EXISTING + adoption chain remains sole upstream source |

### Q.19 Bounded implementation wave (after independent audit)

**One consolidated wave** — do not split into independent semantic authorities:

1. handler adoption propagation;
2. `ConfiguredIntegrationToolInvocationProjectionPort`;
3. `QualifiedToolInvocationResolver` wiring_resolver pass-through;
4. `ToolInvocationWiring.configured_relational_store_execution`;
5. single typed relational execution consumer for database tools;
6. lazy Pattern A bound port/coordinator;
7. execution-bound materialization typing cleanup (typing-04);
8. production composition continuity;
9. P3 qualification/adversarial gates.

### Q.20 Qualification plan

Extend `tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py` family with:

- admitted vs non-admitted tool/category fail-closed;
- resolver called twice before executor — materialization count 0;
- Governance DENY / MSE deny — no materialize/pin/I/O;
- tenant mismatch adversarial;
- sqlite ↔ alternate provider swap without tool/handler change;
- retry reuses same provider instance;
- no silent profile fallback when adoption present.

Relevant ToolRuntime invocation-wiring / database-tool unit tests as identified during implementation — not broad suite discovery.

### Q.21 STOP conditions

**STOP — ARCHITECTURE DECISION REQUIRED** if design requires any of:

- provider in Execution DTO / `ExecutionBoundCatalogToolInvokeRequest` / `ToolInvocationWiring`;
- materialization inside wiring resolver;
- provider I/O before Governance;
- ToolRuntime deciding provider ID/configuration;
- generic `execute(object)` or generic operation registry replacing category contracts;
- permanent database-tool dual semantic path;
- second integration resolver/catalog;
- Governance core modification;
- all-provider migration;
- ExternalWork admission in CONFIGURED_ADOPTED v1;
- provider re-materialization per retry;
- correlation/request ID as canonical `ExecutionId`.

§Q design **does not** trigger these STOP conditions.

### Q.22 Recommended status

| Stage | Status |
|---|---|
| **`TRACE-X-P5-R2-P3-R1-R5`** | **READY FOR AUDIT** |
| **`TRACE-X-P5-R2-P3-R1-R4`** | **SUPERSEDED @ architecture** for admission-06 (historical STOP in §P preserved) |
| **`TRACE-X-P5-R2-P3`** | **BLOCKED PENDING R1-R5 AUDIT / IMPLEMENTATION** |
| **`TRACE-X-P5-R2`** | **CURRENT / P3 NEXT** |
| **`P5-GAP-04`** | **IMPLEMENTATION IN PROGRESS** |
| **`FRZ-TRC-11`** | **OPEN** |
| **P4** | **NOT ENTERED** |

### Q.23 Unresolved findings (R1-R5 classification)

| Finding | Class |
|---|---|
| No production implementation of §Q projection @ HEAD | **IN-SCOPE BLOCKER** (implementation wave) |
| `R2-P3-MATERIALIZATION-PORT-TYPING-04` | **IN-SCOPE BLOCKER** (implementation) |
| Delegate pin-before-handler may pre-materialize for identity @ HEAD | **IN-SCOPE BLOCKER** — align with lazy port + Governance continuity in implementation |
| Legacy `RelationalStore` `Any` behind adapter | **TRACKED FREEZE DEBT** |
| Restart/reconstruction of pinned provider | **TRACKED FREEZE DEBT** (P4) |

### Q.24 Applicable FRZ (R1-R5)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | Primary — **OPEN** |
| FRZ-OWN-* | Supporting — Q.16 |
| FRZ-CTR-01, FRZ-CTR-02 | Supporting — neutral vs typed contracts |
| FRZ-TYP-01..04 | Supporting — Q.6–Q.7 |
| FRZ-PLG-01..05, FRZ-RPL-01/02/04 | Supporting — Q.14–Q.15 |
| FRZ-GOV-05 | Supporting — Q.13 |
| FRZ-EXE-01 | Supporting — ExecutionId authority |
| Relevant FRZ-TEN-* | Supporting — Q.14 |

**No global FRZ PASS promotion.**

### Q.25 Before / after call graph (configured pilot target)

**Before @ HEAD (§P.3 — discontinuous):**

```text
Adoption + pin (delegate, may discard materialized instance)
  ‖ parallel
database.* → ToolWiringContext.from_integration_profile → RelationalStore
```

**After (architecture lock):**

```text
Qualified execution (+ adoption to handler)
  → ConfiguredIntegrationToolInvocationProjectionPort → ToolInvocationWiringResolver
  → ExecutionBoundCatalogToolInvokeRequest.wiring_resolver
  → ToolRuntime Governance → _apply_invocation_wiring
  → ToolInvocationWiring.configured_relational_store_execution
  → database.query | database.execute → ONE typed port
  → lazy Pattern A → pin → SAME provider I/O
```
