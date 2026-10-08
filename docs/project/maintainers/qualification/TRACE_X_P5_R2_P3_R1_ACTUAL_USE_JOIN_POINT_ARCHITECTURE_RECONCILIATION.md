# TRACE-X-P5-R2-P3-R1 — Actual-Use Join Point Architecture Reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** (parent P3-R1 architecture superseded by **R1-R1** §2 for causality join) |
| **Child (reconciliation)** | **`TRACE-X-P5-R2-P3-R1-R1`** — Provider Instance Boundary Reconciliation |
| **START_HEAD** | `94c856d2d0dd9111be4aec41d71984cf08a04d99` |
| **Parent** | `TRACE-X-P5-R2-P3` (implementation remediation **not** in this child) |
| **Primary FRZ** | `FRZ-TRC-11` — **OPEN** (no PASS) |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **Production delta (this child)** | **0** — qualification architecture decision only |

## 1. Current state

| Item | Value |
|---|---|
| **AUDITED_HEAD** | `94c856d2d0dd9111be4aec41d71984cf08a04d99` |
| **Branch** | `development` (aligned with `origin/development`) |

Independent audit of `94c856d2…` rejected **TRACE-X-P5-R2-P3** with four blockers (this document closes architecture for **#3**; specifies remediation for **#1–#2–#4**). **TRACE-X-P5-R2-P3-R1-R1** re-audited the prior P3-R1 **provider carry** proposal against P0 §17 and **rejects** it (§2).

| ID | Summary |
|---|---|
| `R2-P3-CONFIGURE-EXISTING-REACHABILITY-01` | CONFIGURE_EXISTING decision / phase does not reliably reach configured fulfillment → resume → execution with adoption. |
| `R2-P3-PINNING-COMPOSITION-CONTINUITY-02` | `build_worker_recovery_governed_fulfillment_wiring` builds pinning adapter on wiring DTO while `inner_dispatch` may be a delegate **without** the same port installed. |
| `R2-P3-EFFECTIVE-USE-CAUSALITY-03` | `ExecutionBoundIntegrationResolution` materializes for identity observation then **discards** instance; `QualifiedCapabilityExecutionRuntimeDelegate` calls `handler.dispatch_once` via an independent handler pipeline. |
| `R2-P3-MATERIALIZATION-PORT-TYPING-04` | `ExecutionBoundIntegrationMaterializationPort.resolve_from_profile` → `object`; loose `Mapping[str, object]` config seams. |

**Current causality (code fact @ `94c856d2…`):** pin proves `effective.provider_id` from a **transient** `resolve` / `resolve_from_profile` result inside `_observe_effective_identity`; no typed carry-forward ties that **object identity** to the first `PlatformIntegrationContract` business call (or execution-bound tool path that must consume the same materialization). Generic delegate pinning does not pass `ExecutionIntegrationConfigurationAdoption` into binding handlers (`BoundCapabilityExecutionDispatchRequest` is identity + target only).

---

## 2. REJECTED BY INDEPENDENT AUDIT (P3-R1-R1 vs P0)

**Authority:** [`TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md`](TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md) §1A, §1B, §10.2, **§17** — *Neutral contract leaks provider instances → **Forbidden** by design*.

The following P3-R1 (pre-R1-R1) proposals are **REJECTED BY INDEPENDENT AUDIT** and MUST NOT appear in future P3 implementation:

| Rejected proposal | Why (P0) |
|---|---|
| `ExecutionBoundIntegrationResolutionResult` → carry live `CategoryIntegrationInstance` into generic Execution dispatch | Provider materialization stays Integrations-owned; neutral Execution DTOs carry factual adoption only |
| `BoundCapabilityExecutionDispatchRequest` → `CategoryIntegrationInstance` | Contract doc: *Minimal binding-handler dispatch surface — execution target and identity only* (`intergrax/contracts/execution/bound_capability_execution_dispatch.py`) |
| `QualifiedCapabilityExecutionRuntimeDelegate` → inject materialized instance into `BoundCapabilityExecutionDispatchRequest` | Same forbidden neutral leak |
| Generic qualified Execution intake carrying live integration materialization | `QualifiedCapabilityExecutionDispatchRequest` / `QualifiedCapabilityExecutionIntakePayload` may carry **`ExecutionIntegrationConfigurationAdoption`** only — not provider objects |

**Misread corrected:** P0 §10.2 (*before downstream work relies on the instance where feasible*) requires **object continuity inside the Integrations/provider execution boundary**, not transport through neutral Execution contracts. §10.2 does **not** authorize §17 violation.

**Preserved P3-R1 findings (still valid):** causality gap (`R2-P3-EFFECTIVE-USE-CAUSALITY-03`); reachability (`CONFIGURE_EXISTING` sole decision owner); composition continuity (pinning store → execution-bound mechanism → dispatch in use); strong typing debt on `resolve_from_profile`; handler/provider identity must not be conflated.

---

## 3. Required architecture shape (R1-R1 reconciled)

```text
canonical Execution (ExecutionId + tenant + target + optional ExecutionIntegrationConfigurationAdoption on qualified intake)
    ↓
sanctioned category-specific execution handler (binding_provider_id)
    ↓
Integrations-owned execution-bound resolution boundary (same stack frame / typed port — not neutral DTO)
    ↓
materialize exact provider (resolve / resolve_from_profile)
    ↓
derive independent EffectiveIntegrationIdentity
    ↓
validate configured adoption
    ↓
pin provenance under canonical ExecutionId
    ↓
same materialized provider (local reference continuity)
    ↓
first provider/category business operation
```

**Allowed on neutral Execution contracts:** `ExecutionId`, tenant identity, execution target, `ExecutionIntegrationConfigurationAdoption`.

**Forbidden on neutral Execution contracts:** `CategoryIntegrationInstance`, `PlatformIntegrationContract`, concrete provider objects, `object`/`Any`, dict metadata bags, reflection.

---

## 4. Actual execution graph (@ `94c856d2…`)

Closed-world **production-intent** path (Applications host wires `build_worker_recovery_governed_fulfillment_wiring` + `build_qualified_capability_execution_dispatch_service`):

```text
WorkerCapabilityAcquisitionDecisionService.decide
  → CapabilityAcquisitionDisposition.CONFIGURE_EXISTING (sole decision producer)
WorkerCapabilityRecoveryCoordinator.coordinate_recovery
  → phase CONFIGURE_EXISTING_REQUIRED | REALIZATION_REQUIRED+decision | …
WorkerCapabilityFulfillmentCoordinator._fulfill_configure_existing
  → WorkerConfiguredCapabilityFulfillmentService.fulfill_configure_existing
      → ExistingCapabilityConfigurationOpportunityReadPort.read_exact
      → ExistingCapabilityConfigurationRealizationPort.realize
      → ConfiguredCapabilityBinding
      → ExecutionIntegrationConfigurationAdoption
  → (only if QUALIFICATION_COMPLETE) WorkerCapabilityFulfillmentCoordinator._fulfill_qualified
      → WorkerQualifiedCapabilityResumeCoordinator
      → WorkerQualifiedCapabilityExecutionEngineAdapter
      → QualifiedCapabilityExecutionDispatchService
      → ExecutionRuntime (Governance admission + checkpoint admission)
      → QualifiedCapabilityExecutionRuntimeDelegate.execute
          → require_active_execution_identity / peek_active_execution_id → ExecutionId
          → [today] pin_configured_adoption_for_execution → resolve_and_pin (materialize → discard instance)
          → handler.dispatch_once(BoundCapabilityExecutionDispatchRequest only — no adoption on handler surface)
```

**Corrected target graph (Marketplace configured-adoption — future P3):** delegate **must not** be the sole owner of materialize+pin if the handler subsequently re-resolves providers. Pin+materialize+first I/O collapse into the **Marketplace / tool category boundary** (§6).

---

## 5. Join-point classification (closed-world inventory)

| Surface | Class | Evidence |
|---|---|---|
| `ExecutionBoundIntegrationResolution.resolve_and_pin` | **A** (partial) | Correct owner for materialize + validate + pin; `_observe_effective_identity` already calls `resolve` / `resolve_from_profile` but returns only `EffectiveIntegrationIdentity` — instance discarded @ `94c856d2…` |
| `ExecutionIntegrationConfigurationExecutionPinningAdapter` | **A** | Thin adapter; no provider carry |
| `QualifiedCapabilityExecutionRuntimeDelegate.execute` | **B** | Has `ExecutionId` + adoption on `QualifiedCapabilityExecutionIntakePayload`; must **stop** being the place that pins without coupling to the handler’s provider path; may pass **adoption** to handlers via protocol extension — not provider instance |
| `QualifiedCapabilityExecutionBindingHandler.dispatch_once` | **B** | Needs typed optional `ExecutionIntegrationConfigurationAdoption` parameter (factual contract) so configured-adoption handlers can invoke Integrations boundary; **not** `BoundCapabilityExecutionDispatchRequest` extension |
| `CodeCraftQualifiedCapabilityExecutionHandler` | **N/A — WITH EVIDENCE** | `CodeCraftBoundCapabilityExecutionPort.execute` — CodeCraft substrate, **not** `PlatformIntegrationContract` configured-adoption path |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` → `ExecutionBoundCatalogToolInvoker` | **B** | Handler + invoker are the sanctioned tool category boundary; today **does not** read adoption; tool `invoke` uses `wiring_resolver` / registry paths independent of pin materialization — **no legal existing end-to-end join (A)** without P3 wiring |
| Neutral `BoundCapabilityExecutionDispatchRequest` | **C** if provider carry attempted | **Forbidden** — use category boundary instead |

**Core question answer:** Smallest sanctioned boundary where **(1)** `ExecutionId` + adoption are available and **(2)** the resolved provider can perform the first business operation without crossing a neutral Execution DTO carrying the instance:

→ **`ExecutionBoundIntegrationResolution` (extended `resolve_materialize_validate_pin`) invoked from the category-specific handler (or a handler-injected Integrations/tool port), retaining the materialized reference only in that call stack until the first provider/tool business operation.**

**Not** `STOP — ARCHITECTURE DECISION REQUIRED` — existing ports **`ExecutionBoundIntegrationResolution`** + **`ExecutionBoundCatalogToolInvoker`** + handler protocol extension (**B**) suffice; no new generic provider transport.

---

## 6. Handler-by-handler configured-adoption classification

| Handler | Participates in CONFIGURED_ADOPTED provider causality? | Classification | Action for P3 |
|---|---|---|---|
| `CodeCraftQualifiedCapabilityExecutionHandler` | **No** — not an Integrations `PlatformIntegrationContract` execution path | **Unrelated** | No CONFIGURED_ADOPTED materialization forced; optional adoption on qualified intake must not imply CodeCraft is the certified integration provider |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | **Yes** (when qualified dispatch carries adoption for an integration category consumed by catalog tool wiring) | **Configured-adoption participant** | After protocol carries adoption: single Integrations boundary call (materialize → validate → pin → invoke with **same** instance-fed wiring); **zero** post-pin `resolve` / `resolve_from_profile` for that adoption subject |
| Other binding handlers @ `94c856d2…` | None in closed-world inventory | **Unrelated** | Unchanged |

**Marketplace trace (minimum):** `dispatch_once` → `QualifiedMarketplaceToolActivationResolver.ensure_exact_active` → `QualifiedToolInvocationMaterialProvider.provide` → `QualifiedToolInvocationResolver.resolve` → `NexusExecutionBoundCatalogToolInvoker.invoke` (`ToolInvocationContext.wiring_resolver` / catalog host). **Adoption is not consumed** on this path today; pin in delegate is causally **decoupled** from `invoke`.

---

## 7. Reconciliation answers (R1-R1 checklist)

| # | Question | Answer @ reconciled architecture |
|---|---|---|
| 1 | Where does effective provider materialization happen? | Inside `ExecutionBoundIntegrationResolution` (`resolve` / `resolve_from_profile` via `ExecutionBoundIntegrationMaterializationPort`) — **on the category path that will perform first provider I/O**, not in neutral delegate DTOs |
| 2 | Where does canonical `ExecutionId` become available? | Execution admission / `require_active_execution_identity` + `peek_active_execution_id` in `QualifiedCapabilityExecutionRuntimeDelegate.execute` (P0 §1B.12) |
| 3 | Who owns the first provider business call? | Category/provider boundary: Marketplace → `ExecutionBoundCatalogToolInvoker.invoke` (tool/provider I/O); Integrations owns materialization+pin immediately before that call on configured-adoption paths |
| 4 | Can materialize + validate + pin + first call occur in one typed boundary? | **Yes** — single Integrations-layer method + same-stack handler/invoker orchestration (**B**); not yet implemented @ `94c856d2…` |
| 5 | Provider instance outside neutral Execution DTOs? | **Yes** — required; P0 §17 |
| 6 | Re-resolve between pin and business call? | **Forbidden** — architecture **FAIL** if allowed |
| 7 | | See row 6 |
| 8 | Mechanical call graph (target) | `QualifiedCapabilityExecutionIntakePayload` (adoption) → `MarketplaceToolQualifiedCapabilityExecutionHandler.dispatch_once` (+ adoption param) → `ExecutionBoundIntegrationResolution.resolve_materialize_validate_pin` → local `CategoryIntegrationInstance` / `PlatformIntegrationContract` → pin store → `ExecutionBoundCatalogToolInvoker.invoke` using wiring from **that** instance → first tool/provider operation |
| 9 | Handlers in configured-adoption execution? | **Marketplace** (when adoption present and path classified); **not CodeCraft** |
| 10 | Unrelated handlers? | **CodeCraft** and any handler not classified `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` (P0 §1B.16) |

**Causality invariant:** `provider observed as effective` **==** object reference that performs first provider business call — **local object continuity** inside Integrations/category boundary only.

---

## 8. Ownership matrix (unchanged)

| Concern | Exactly-one owner |
|---|---|
| Acquisition decision | `WorkerCapabilityAcquisitionDecisionService` |
| CONFIGURE_EXISTING sequencing | `WorkerCapabilityFulfillmentCoordinator` |
| Configuration opportunity | Integrations (`ExistingCapabilityConfigurationOpportunity` read ports) |
| INT-CONFIG realization | `ExistingCapabilityConfigurationRealizationPort` / Integrations facade |
| Governance authorization | Governance (`ControlPlaneMutationAuthorizationPort` inside realization facade — not AW) |
| Effective materialization | Integrations `resolve` / `resolve_from_profile` via `ExecutionBoundIntegrationResolution` |
| Execution identity | Execution (`ExecutionRuntime`, `require_active_execution_identity`) |
| Provenance pin | `ExecutionBoundIntegrationResolution` + `ExecutionIntegrationConfigurationPinningStore` |
| Provider business execution | Category execution boundary (`ExecutionBoundCatalogToolInvoker`, future category handlers) **after** same-boundary materialization |
| Reconstruction | `ExecutionReconstructor` via neutral provenance reader only |

---

## 9. Reachability & composition (retained)

### Reachability (`R2-P3-CONFIGURE-EXISTING-REACHABILITY-01`)

- Propagate **canonical** `WorkerCapabilityAcquisitionDecision` from `WorkerCapabilityAcquisitionDecisionService` through recovery provenance into `WorkerCapabilityFulfillmentCoordinator` without `WorkerCapabilityRecoveryCoordinator` becoming a second CONFIGURE_EXISTING decision owner.
- **Forbidden:** recovery re-deriving CONFIGURE_EXISTING independently.
- **Missing seam (if proven in implementation):** typed carry of the already-canonical decision object from recovery outcome into fulfillment when phase is `CONFIGURE_EXISTING_REQUIRED` — architecture requires propagation, not re-decision.

### Production composition (`R2-P3-PINNING-COMPOSITION-CONTINUITY-02`)

- Collapse to **one** wiring path: `configuration_pinning_store` → `ExecutionBoundIntegrationResolution` → adapter → `build_qualified_capability_execution_dispatch_service(..., integration_configuration_pinning=…)` → `inner_dispatch` used at runtime.
- **Forbidden:** orphan pinning adapter; **forbidden:** solving causality by stuffing provider instances into generic dispatch DTOs.

### Actual-use causality (`R2-P3-EFFECTIVE-USE-CAUSALITY-03`) — **R1-R1 remediation**

- Refactor `resolve_and_pin` → **`resolve_materialize_validate_pin`** returning materialized instance **to Integrations/category callers only** (not neutral Execution contracts).
- **Remove** generic delegate-only pin that materializes and discards while handlers re-resolve (**anti-pattern** @ baseline).
- Extend `QualifiedCapabilityExecutionBindingHandler.dispatch_once` with optional **`ExecutionIntegrationConfigurationAdoption`** (typed).
- Marketplace path: handler (or invoker wrapper in allow-list) calls `resolve_materialize_validate_pin` then `invoke` with instance-derived wiring — **no** second resolution.
- **REJECTED (§2):** `BoundCapabilityExecutionDispatchRequest` / qualified intake provider fields.

### Strong typing (`R2-P3-MATERIALIZATION-PORT-TYPING-04`)

- `resolve_from_profile` → `CategoryIntegrationInstance` (or `PlatformIntegrationContract` where branch is contract-only).
- **Forbidden:** `Any`, semantic `object`, reflection, string-equality causality proof.

---

## 10. Strong typing target

| Seam | Canonical type |
|---|---|
| Catalog materialization | `PlatformIntegrationContract` |
| Profile / union materialization | `CategoryIntegrationInstance` — CONFIGURED_ADOPTED v1: **contract branch only** |
| Effective identity (pin) | `EffectiveIntegrationIdentity` |
| Adoption input | `ExecutionIntegrationConfigurationAdoption` + `ConfiguredCapabilityBinding` |
| Provenance | `ExecutionIntegrationConfigurationProvenance`, `IntegrationConfigurationSubject` |
| Neutral handler intake | `BoundCapabilityExecutionDispatchRequest` — **unchanged** (no provider fields) |

---

## 11. Tenant isolation audit

**Verdict: PASS — local P3-R1 / R1-R1 architecture scope**

Evidence: fail-closed tenant checks on configured fulfillment, `ExecutionBoundIntegrationResolution.resolve_and_pin`, adoption validators, qualified dispatch intake. Remediation preserves:

```text
fulfillment tenant == configured binding tenant == adoption tenant == execution tenant == resolution tenant == provenance tenant
```

**No global `FRZ-TEN-*` PASS promotion.**

---

## 12. Implementation file budget (future P3 remediation)

Bounded allow-list (updated R1-R1 — **no** provider fields on neutral dispatch contracts):

| Area | Files |
|---|---|
| Integrations resolution | `intergrax/integrations/execution_bound_integration_resolution.py` |
| Execution pin ports / adapter | `intergrax/runtime/execution/execution_integration_configuration_pinning_ports.py`, `intergrax/runtime/execution/execution_bound_integration_pinning_adapter.py` |
| Runtime delegate / composition | `intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py`, `intergrax/runtime/execution/qualified_capability_execution_composition.py` |
| Dispatch / intake contracts | `qualified_capability_execution_intake.py`, `qualified_capability_execution_dispatch.py` — adoption only; **`bound_capability_execution_dispatch.py` unchanged** |
| Handler protocol | `intergrax/runtime/execution/qualified_capability_execution_handlers.py` |
| AW reachability / wiring | `worker_capability_fulfillment_coordinator.py`, `worker_capability_recovery_coordinator.py` (phase mapping only), `worker_recovery_governed_fulfillment_composition.py`, `worker_qualified_capability_resume_coordinator.py` |
| Handlers | `qualified_capability_execution_handler.py` (CodeCraft), `marketplace_qualified_capability_execution_handler.py`, `nexus_execution_bound_catalog_tool_invoker.py` (or sibling `ExecutionBoundCatalogToolInvoker` impl) |
| Tests / gates | `test_trace_x_p5_r2_p3_configured_fulfillment.py`, `test_trace_x_p5_r2_p3_production_flow_gates.py`, targeted adversarial under `tests/qualification/trace_x/` |

Changes outside this list require a new architecture child (**STOP**).

---

## 13. Exit criteria for future P3 remediation

- [ ] Canonical CONFIGURE_EXISTING decision reaches fulfillment without a second decision owner.
- [ ] Exact `ConfiguredCapabilityBinding` preserved on `ExecutionIntegrationConfigurationAdoption` end-to-end.
- [ ] Effective provider identified via canonical materialization path only.
- [ ] **Same object reference** from materialization performs first provider/tool business call (not string match).
- [ ] Pin failure → provider business call count = 0.
- [ ] Provider mismatch → pin count = 0 and provider call count = 0.
- [ ] Missing required adoption → provider call count = 0 on configured-required paths.
- [ ] After successful pin, **no** second resolution for the same adoption subject.
- [ ] Neutral Execution contracts contain **no** provider instance.
- [ ] Production composition auto-wires pinning into the dispatch delegate **and** category path continuity.
- [ ] Tenant isolation adversarial cases fail closed before provider I/O.
- [ ] No `Any` / generic `object` semantic boundary on materialization port.
- [ ] P1/P2 qualification gates remain green.

---

## FRZ evidence (revalidated, no PASS)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | **OPEN** — primary; R1-R1 defines legal join without §17 violation |
| FRZ-TYP-* | Revalidated — materialization typing remediation still required |
| FRZ-CTR-* / FRZ-OWN-* | Revalidated — §8 |
| FRZ-GOV-* / FRZ-BND-* | Revalidated — AW does not execute provider APIs |

---

## Unresolved findings

| Classification | Item |
|---|---|
| **IN-SCOPE BLOCKER** | All four R2-P3 blockers — remediation §9; **not** implemented @ `94c856d2…` |
| **ARCHITECTURE CLOSED (R1-R1)** | Provider carry through neutral Execution — **rejected**; replacement join §5–§7 |
| **TRACKED FREEZE DEBT** | `FRZ-TRC-11` global closure deferred to CERT |

---

## Recommended roadmap status

| Node | Status |
|---|---|
| **TRACE-X-P5-R2-P3-R1-R1** | **READY FOR AUDIT** |
| **TRACE-X-P5-R2-P3-R1** | **BLOCKED PENDING R1-R1 AUDIT** (parent doc amended; prior READY superseded for causality transport) |
| **TRACE-X-P5-R2-P3** | **BLOCKED ON IMPLEMENTATION REMEDIATION** |
| **TRACE-X-P5-R2** | **CURRENT / BLOCKED ON P3** |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **FRZ-TRC-11** | **OPEN** |

**Final architecture disposition:** **READY FOR AUDIT** — not `STOP — ARCHITECTURE DECISION REQUIRED` (category **B** extensions on existing Integrations + Marketplace/tool ports; P0 §17 preserved).
