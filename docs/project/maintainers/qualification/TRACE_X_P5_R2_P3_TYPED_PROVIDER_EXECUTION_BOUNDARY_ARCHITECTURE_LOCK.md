# TRACE-X-P5-R2-P3-R1-R2 — Typed Provider Execution Boundary Architecture Lock

## A. Revision / status

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R2` — Typed Provider Execution Boundary Architecture Lock |
| **Child reconciliation** | `TRACE-X-P5-R2-P3-R1-R3` — Governance + typed resolution reconciliation @ `63ba8a237449ce7e78043c480dff97ed1f96918e` (§O) |
| **START_HEAD (R1-R2)** | `d39b39ee94982d74acc7042c4dfa63a887a97c05` (`development` = `origin/development` @ R1-R2 task start) |
| **START_HEAD (R1-R3)** | `63ba8a237449ce7e78043c480dff97ed1f96918e` (`development` = `origin/development` @ R1-R3 task start) |
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
| `TRACE-X-P5-R2-P3-R1-R3` | **READY FOR AUDIT** |
| `TRACE-X-P5-R2-P3-R1-R2` | Architecture base — governance/typing superseded by R1-R3 §O |
| `TRACE-X-P5-R2-P3` | **BLOCKED PENDING ARCHITECTURE AUDIT / IMPLEMENTATION** |
| `TRACE-X-P5-R2` | **CURRENT / BLOCKED ON P3** |
| `P5-GAP-04` | **IMPLEMENTATION IN PROGRESS** |
| `FRZ-TRC-11` | **OPEN** |

Do **not** mark P3 CLOSED. Do **not** enter P4.
