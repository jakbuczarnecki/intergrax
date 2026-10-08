# TRACE-X-P5-R2-P3-R1 — Actual-Use Join Point Architecture Reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** |
| **START_HEAD** | `d5fbe8843462ff30a25958efc765fb710ed6cb8f` |
| **Parent** | `TRACE-X-P5-R2-P3` (implementation remediation **not** in this child) |
| **Primary FRZ** | `FRZ-TRC-11` — **OPEN** (no PASS) |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **Production delta (this child)** | **0** — qualification architecture decision only |

## 1. Current state

| Item | Value |
|---|---|
| **AUDITED_HEAD** | `d5fbe8843462ff30a25958efc765fb710ed6cb8f` |
| **Branch** | `development` (aligned with `origin/development`) |

Independent audit of `d5fbe884…` rejected **TRACE-X-P5-R2-P3** with four blockers (this child closes architecture for **#3** only; specifies remediation for **#1–#2–#4**):

| ID | Summary |
|---|---|
| `R2-P3-CONFIGURE-EXISTING-REACHABILITY-01` | CONFIGURE_EXISTING decision / phase does not reliably reach configured fulfillment → resume → execution with adoption. |
| `R2-P3-PINNING-COMPOSITION-CONTINUITY-02` | `build_worker_recovery_governed_fulfillment_wiring` builds pinning adapter on wiring DTO while `inner_dispatch` may be a delegate **without** the same port installed. |
| `R2-P3-EFFECTIVE-USE-CAUSALITY-03` | `ExecutionBoundIntegrationResolution` materializes for identity observation then **discards** instance; `QualifiedCapabilityExecutionRuntimeDelegate` calls `handler.dispatch_once` via an independent handler pipeline. |
| `R2-P3-MATERIALIZATION-PORT-TYPING-04` | `ExecutionBoundIntegrationMaterializationPort.resolve_from_profile` → `object`; loose `Mapping[str, object]` config seams. |

**Current causality (code fact):** pin proves `effective.provider_id` from a **transient** `resolve` / `resolve_from_profile` result; no typed carry-forward ties that object identity to the first `PlatformIntegrationContract` business call (or execution-bound tool path that must consume the same materialization).

---

## 2. Actual execution graph (configured-adoption paths @ `d5fbe884…`)

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
          → ExecutionIntegrationConfigurationExecutionPinningPort.pin_configured_adoption_for_execution
              → ExecutionBoundIntegrationConfigurationExecutionPinningAdapter
              → ExecutionBoundIntegrationResolution.resolve_and_pin
                  → resolve | resolve_from_profile (materialize → observe provider_id → pin → return result without instance)
          → QualifiedCapabilityExecutionBindingHandlerRegistry.resolve(binding_provider_id)
          → handler.dispatch_once(BoundCapabilityExecutionDispatchRequest, …)
```

**Sanctioned binding handlers (qualified dispatch):**

| Handler | Adoption on intake | First provider business boundary (today) |
|---|---|---|
| `CodeCraftQualifiedCapabilityExecutionHandler` | optional `integration_configuration_adoption` on dispatch chain | `CodeCraftBoundCapabilityExecutionPort.execute` — **not** `PlatformIntegrationContract` |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | same | `QualifiedMarketplaceToolActivationResolver.ensure_exact_active` → … → `ExecutionBoundCatalogToolInvoker.invoke` — **separate** from pin materialization |

**Direct test / application composition path** (same execution seam):

```text
build_qualified_capability_execution_dispatch_service(integration_configuration_pinning=…)
  → QualifiedCapabilityExecutionRuntimeDelegate (pin port optional)
  → same delegate.execute graph as above
```

**P0 authority (unchanged):** [`TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md`](TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md) §10.2 — materialize via existing `resolve` / `resolve_from_profile`, pin after `ExecutionId`, **before downstream work relies on the instance**.

---

## 3. Exactly-one actual-use join point (target architecture)

| Role | Owner |
|---|---|
| **Semantic owner** | `ExecutionBoundIntegrationResolution` (Integrations) — sole canonical materialize + configured/effective validate + provenance pin |
| **Contract owner** | `ExecutionBoundIntegrationResolutionResult` **extended** to include materialized `CategoryIntegrationInstance`; execution intake extended to carry the **same** instance reference into dispatch (typed field on `BoundCapabilityExecutionDispatchRequest` / qualified intake — not a metadata bag) |
| **Composition owner** | `build_qualified_capability_execution_dispatch_service` — **only** sanctioned assembler of `QualifiedCapabilityExecutionRuntimeDelegate` + `integration_configuration_pinning`; `build_worker_recovery_governed_fulfillment_wiring` **must** obtain `inner_dispatch` from that builder (or equivalent single factory) with pinning injected — **not** a detached adapter on `WorkerRecoveryGovernedFulfillmentWiring` only |
| **Join point (timing)** | `QualifiedCapabilityExecutionRuntimeDelegate.execute`: immediately **after** canonical `ExecutionId` is available and **after** `resolve_materialize_validate_pin` succeeds, **before** `handler.dispatch_once` and **before** any code path can reach the first `PlatformIntegrationContract` business method (or execution-bound tool invoke that is classified as consuming the adopted integration materialization for that `adoption.integration_category`) |

**Named seam:** **Execution-bound configured integration materialization handle** — one `CategoryIntegrationInstance` per `(tenant_id, ExecutionId, ExecutionIntegrationConfigurationAdoption)` pin attempt, produced only inside `ExecutionBoundIntegrationResolution`.

`ExternalWorkIntegration` remains **out of** `CONFIGURED_ADOPTED` v1 (P0 C1).

---

## 4. Causality proof

| State | `Can pinned effective provider differ from provider actually used?` |
|---|---|
| **@ `d5fbe884…` (current)** | **YES** — materialized instance is not retained; handlers/tool wiring may select providers independently (`tools/registry/wiring.py` `resolve_from_profile`, activation/material pipelines). String equality of `provider_id` after two resolutions is **not** causal proof. |
| **After P3 remediation (this design)** | **NO** — mechanical: pin and first provider business call must use the **same** `CategoryIntegrationInstance` object reference obtained once from `resolve` / `resolve_from_profile` inside `ExecutionBoundIntegrationResolution`; pin failure or identity mismatch **blocks** dispatch; no second registry resolution on the configured-adoption path after pin. |

---

## 5. Remediation contract (P3 implementation — do not implement in P3-R1)

### Reachability (`R2-P3-CONFIGURE-EXISTING-REACHABILITY-01`)

- Propagate **canonical** `WorkerCapabilityAcquisitionDecision` from `WorkerCapabilityAcquisitionDecisionService` through recovery provenance into `WorkerCapabilityFulfillmentCoordinator` without `WorkerCapabilityRecoveryCoordinator` becoming a second CONFIGURE_EXISTING decision owner.
- Map recovery phase so CONFIGURE_EXISTING fulfillment → qualification/resume can run with `ExecutionIntegrationConfigurationAdoption` on `WorkerQualifiedCapabilityExecutionRequest` / `QualifiedCapabilityExecutionDispatchRequest`.
- **Forbidden:** recovery re-deriving CONFIGURE_EXISTING independently.

### Production composition (`R2-P3-PINNING-COMPOSITION-CONTINUITY-02`)

- Collapse to **one** wiring path: `configuration_pinning_store` → `ExecutionBoundIntegrationResolution` → `ExecutionBoundIntegrationConfigurationExecutionPinningAdapter` → `build_qualified_capability_execution_dispatch_service(..., integration_configuration_pinning=…)` → `inner_dispatch` passed to `GovernedTaskScopedQualifiedCapabilityExecutionDispatchService`.
- Remove reliance on operators manually copying `WorkerRecoveryGovernedFulfillmentWiring.integration_configuration_pinning` onto dispatch.
- **Forbidden:** orphan pinning adapter that is not the delegate installed in the dispatch service used at runtime.

### Actual-use causality (`R2-P3-EFFECTIVE-USE-CAUSALITY-03`)

- Refactor `resolve_and_pin` → **`resolve_materialize_validate_pin`** returning `CategoryIntegrationInstance` + provenance artifacts.
- Extend `ExecutionIntegrationConfigurationExecutionPinningPort` to return the materialized instance (or fail closed).
- `QualifiedCapabilityExecutionRuntimeDelegate` passes instance into `BoundCapabilityExecutionDispatchRequest` (typed).
- Handlers and `ExecutionBoundCatalogToolInvoker` / tool wiring on configured-adoption paths **must** accept and use the injected instance for first integration-category provider I/O — **no** post-pin `resolve` / `resolve_from_profile` for the same adoption subject.
- **Forbidden:** pre-resolve → pin → discard → independent handler resolution (current shape).

### Strong typing (`R2-P3-MATERIALIZATION-PORT-TYPING-04`)

- Align `ExecutionBoundIntegrationMaterializationPort.resolve_from_profile` with registry factory: `-> CategoryIntegrationInstance`.
- Align `resolve_catalog` with `-> PlatformIntegrationContract` (already typed in default impl).
- Replace `Mapping[str, object]` resolution config with platform-typed config merge inputs where already defined (`Optional[Mapping[str, Any]]` at factory boundary only if unavoidable — prefer `IntegrationProfile` + typed merge from `intergrax/integrations/registry/factory.py`).
- **Forbidden:** `Any`, semantic `object`, `getattr`/`setattr`, reflection, post-hoc string compare as causality proof.

---

## 6. Strong typing target

| Seam | Canonical type |
|---|---|
| Catalog materialization | `PlatformIntegrationContract` (`intergrax/runtime/integrations/contracts.py`) |
| Profile / union materialization | `CategoryIntegrationInstance` = `PlatformIntegrationContract \| ExternalWorkIntegration` (`intergrax/runtime/integrations/contract_metadata.py`) — CONFIGURED_ADOPTED v1 uses **contract branch only** |
| Effective identity (pin) | `EffectiveIntegrationIdentity` (`intergrax/integrations/contracts/execution_integration_configuration.py`) |
| Adoption input | `ExecutionIntegrationConfigurationAdoption` + `ConfiguredCapabilityBinding` |
| Provenance | `ExecutionIntegrationConfigurationProvenance`, `IntegrationConfigurationSubject` |

Do **not** introduce a generic wrapper type if `CategoryIntegrationInstance` + `ExecutionBoundIntegrationResolutionResult` extension suffices.

---

## 7. Ownership matrix

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
| Provider business execution | Execution binding handlers + ports (`CodeCraftBoundCapabilityExecutionPort`, `ExecutionBoundCatalogToolInvoker`, future category handlers) **using** materialized instance from join |
| Reconstruction | `ExecutionReconstructor` via neutral provenance reader only |

---

## 8. Tenant isolation audit

**Verdict: PASS — local P3-R1 architecture scope**

Evidence: existing fail-closed tenant checks on `WorkerConfiguredCapabilityFulfillmentService`, `ExecutionBoundIntegrationResolution.resolve_and_pin` (`binding.tenant_id` vs `request.tenant_id`), adoption validators, and qualified dispatch intake. Remediation preserves single `tenant_id` chain (fulfillment == opportunity == binding == adoption == execution == pin). **No global `FRZ-TEN-*` PASS promotion.**

---

## 9. Implementation file budget (future P3 remediation)

Bounded allow-list:

| Area | Files |
|---|---|
| Integrations resolution | `intergrax/integrations/execution_bound_integration_resolution.py` |
| Execution pin ports / adapter | `intergrax/runtime/execution/execution_integration_configuration_pinning_ports.py`, `intergrax/runtime/execution/execution_bound_integration_pinning_adapter.py` |
| Runtime delegate / composition | `intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py`, `intergrax/runtime/execution/qualified_capability_execution_composition.py` |
| Dispatch / intake contracts | `intergrax/contracts/execution/bound_capability_execution_dispatch.py`, `intergrax/contracts/execution/qualified_capability_execution_intake.py`, `intergrax/contracts/execution/qualified_capability_execution_dispatch.py` |
| AW reachability / wiring | `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py`, `intergrax/autonomous_work/worker_capability_recovery_coordinator.py` (phase mapping only — **no** new decision logic), `intergrax/autonomous_work/worker_recovery_governed_fulfillment_composition.py`, `intergrax/autonomous_work/worker_qualified_capability_resume_coordinator.py` |
| Handlers (minimal coupling) | `intergrax/runtime/codecraft/qualified_capability_execution_handler.py`, `intergrax/tools/marketplace_qualified_capability_execution_handler.py`, plus **one** `ExecutionBoundCatalogToolInvoker` implementation file if tool path must consume materialized integration |
| Tests / gates | `tests/unit/autonomous_work/test_trace_x_p5_r2_p3_configured_fulfillment.py`, `tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py`, targeted adversarial additions under `tests/qualification/trace_x/` |

Changes outside this list require a new architecture child (**STOP**).

---

## 10. Exit criteria for future P3 remediation

- [ ] Canonical CONFIGURE_EXISTING decision reaches fulfillment without a second decision owner.
- [ ] Exact `ConfiguredCapabilityBinding` preserved on `ExecutionIntegrationConfigurationAdoption` end-to-end.
- [ ] Effective provider identified via canonical materialization path only.
- [ ] **Pinned `provider_id` == provider used** (same `CategoryIntegrationInstance` reference).
- [ ] Pin failure → provider business call count = 0.
- [ ] Provider mismatch → pin count = 0 and provider call count = 0.
- [ ] Missing required adoption → provider call count = 0.
- [ ] Production composition auto-wires pinning into the dispatch delegate in use.
- [ ] No manual caller wiring to make production flow work.
- [ ] No `Any` / generic `object` semantic boundary on materialization port.
- [ ] No bypass around execution-bound path for CONFIGURED_ADOPTED.
- [ ] Tenant A cannot use tenant B adoption / provider / provenance.
- [ ] P1/P2 qualification gates remain green.

---

## FRZ evidence (revalidated, no PASS)

| FRZ | Role in P3-R1 |
|---|---|
| **FRZ-TRC-11** | **OPEN** — primary; this document defines actual-use join only |
| FRZ-TYP-* (family) | Revalidated — materialization must use `CategoryIntegrationInstance` / `PlatformIntegrationContract` |
| FRZ-CTR-* / FRZ-OWN-* | Revalidated — single owners in §7 |
| FRZ-GOV-* / FRZ-BND-* | Revalidated — AW does not execute provider APIs; Governance does not select providers |

---

## Unresolved findings

| Classification | Item |
|---|---|
| **IN-SCOPE BLOCKER** | `R2-P3-CONFIGURE-EXISTING-REACHABILITY-01`, `R2-P3-PINNING-COMPOSITION-CONTINUITY-02`, `R2-P3-EFFECTIVE-USE-CAUSALITY-03`, `R2-P3-MATERIALIZATION-PORT-TYPING-04` — remediation specified §5; not implemented in P3-R1 |
| **TRACKED FREEZE DEBT** | Full adversarial matrices for P3 production flow (partial per P3 flow doc); `FRZ-TRC-11` global closure deferred to CERT |
| **ENVIRONMENT/TEST ISSUE** | None identified in gate run below |

---

## Qualification gates executed

```text
uv run pytest -p no:xdist \
  tests/qualification/trace_x/test_trace_x_p5_r2_p1_contract_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p2_persistence_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py
```

**Result:** 39 passed in 11.29s (log: `.tmp/session/trace-x-p5-r2-p3-r1/pytest-gates.log`).

---

## Recommended roadmap status

| Node | Status |
|---|---|
| **TRACE-X-P5-R2-P3-R1** | **READY FOR AUDIT** |
| **TRACE-X-P5-R2-P3** | **BLOCKED ON IMPLEMENTATION REMEDIATION** |
| **TRACE-X-P5-R2** | **CURRENT / BLOCKED ON P3** |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **FRZ-TRC-11** | **OPEN** |

**Architecture decision:** **not** `STOP — ARCHITECTURE DECISION REQUIRED` — P0 §10.2 already implies retain materialized instance before downstream use; P3-R1 extends that into a single execution join without new resolver, Governance change, AW provider execution, or External Work scope expansion.
