# TRACE-X-P5-R2-P3-R1-R1-R1 — Configured Execution Subject & Canonical Intake Architecture Lock

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R1-R1` |
| **Parent** | `TRACE-X-P5-R2-P3-R1-R1` → `TRACE-X-P5-R2-P3-R1` → `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **START_HEAD** | `f517fd787d0e10488eed9a0da97c1ea1c5166f0c` |
| **Disposition** | **SUPERSEDED (duplicate-prone sections)** — see [`TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md`](TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md) for execution ingress, business identity, intent, and implementation-map convergence. Non-superseded: Variant B ∩ UCA = ∅, subject reference shape, governance ordering, tenant continuity. |
| **Production delta** | **0** (architecture / qualification gates only) |
| **FRZ-TRC-11** | **OPEN** (scoped evidence; no PASS promotion) |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |

## Canonical stage (@ START_HEAD)

| Stage | Status |
|---|---|
| TRACE-X | CURRENT |
| TRACE-X-P5 | CURRENT / BLOCKED ON R2 |
| TRACE-X-P5-R2 | CURRENT / P3 NEXT |
| TRACE-X-P5-R2-P3 | NEXT / REQUIRED / NOT ENTERED |
| P4 | NOT ENTERED |

## Blockers reconciled

| Blocker | Resolution |
|---|---|
| **R2-P3-CONFIGURED-EXECUTION-SUBJECT-12** | Pure Variant B requires **`ConfiguredCapabilityExecutionSubject`** + configured binding/intake seam; **no** UCA facts; minimum typed business-target anchor (§5–§6). |
| **R2-P3-VARIANT-B-UCA-CONFLATION-13** | **REJECT** R1-R1 Q3 UCA reuse; lock Variant B ∩ UCA acquisition/qualification = **∅** per episode (§2). |
| **R2-P3-CANONICAL-DISPATCH-COMPOSITION-09** | **Preserved** from R1-R1 — one host-composed `QualifiedCapabilityExecutionBindingHandlerRegistry`. |
| **R2-P3-AW-TO-CANONICAL-DISPATCH-REACHABILITY-10** | **Preserved** — configured ingress feeds same canonical `ExecutionRuntime` family as qualified/DIRECT_REUSE. |
| **R2-P3-CONFIGURE-EXISTING-ENTRY-SEAM-11** | **Preserved** — `WorkerCapabilityFulfillmentRecoveryAdapter` owns `decide` → `CONFIGURE_EXISTING_REQUIRED` (§12). |

---

## 1 — Audit rejection of R1-R1 Q3

Independent audit @ `f517fd787…` rejected R1-R1 because **Q3 conflates Variant B with Variant C facts**.

| Path | Normative graph |
|---|---|
| **Variant B — CONFIGURE_EXISTING** | existing capability/provider → configuration realization (INT-CONFIG) → adoption → **configured execution subject** → typed binding → canonical Execution. **No** Capability Acquisition. **No** Qualification. |
| **Variant C — TRUE GAP** | CapabilityGap → Capability Acquisition → Qualification → Binding → Execution. |

**Forbidden (R1-R1 Q3):**

```text
CONFIGURE_EXISTING → reuse acquisition_result → reuse qualification_result → qualified resume
```

That path is Variant C provenance smuggled into Variant B. **REJECT.**

---

## 2 — Locked invariants (do not reopen)

- Pure `CONFIGURE_EXISTING` **MUST NOT** require or synthesize: `CapabilityAcquisitionResult`, `CapabilityQualificationResult`, `acquisition_request_id`, `qualification_request_id`, `QualifiedCapabilitySubject`, or fake IDs.
- `CONFIGURE_EXISTING ∩ CapabilityGap/CapabilityAcquisition = ∅` for the same configured realization episode.
- Prior unrelated UCA episodes **do not** authorize configured execution.
- `ConfiguredCapabilityBinding` owner = **Integrations / INT-CONFIG** only.
- `ExecutionIntegrationConfigurationAdoption` = configuration adoption fact only (not authorization, not effective proof, not business execution target).
- **One** canonical Execution authority — no second Execution Engine.

---

## 3 — Current contract incompatibility inventory (@ HEAD)

| Contract | UCA-only / incompatible with pure Variant B |
|---|---|
| `WorkerQualifiedCapabilityResumeCoordinator` | Requires `CapabilityQualificationResult` → `QualifiedCapabilitySubject` → `QualifiedCapabilityBindingRequest`. |
| `QualifiedCapabilityBindingRequest` | Embeds `qualification_result` + `qualified_subject`. |
| `QualifiedCapabilityExecutionDispatchRequest` | Requires `resume_operation_id`, `binding_operation_id`, `qualification_request_id`, `acquisition_request_id`, `qualified_subject_reference`. |
| `MarketplaceToolQualifiedCapabilityBindingProvider.bind` | Requires `QualifiedCapabilityBindingRequest` with UCA qualification + subject. |
| `MarketplaceQualifiedToolExecutionIntentPreparation` | Gated on `CapabilityQualificationResult` + marketplace strategy evidence. |
| `WorkerCapabilityCandidate` (EXISTING_CONFIGURATION) | Has `capability_ref`, `configuration_ref`, `operations` — **no** `QualifiedCapabilityExecutionTarget`, handoff id, or catalog identity key. |
| `ExistingCapabilityConfigurationOpportunity` | Integration/configuration identity only — **does not** identify Marketplace staged-tool execution target. |
| R1-R1 Q4 `_fulfill_qualified` continuation | Requires `acquisition_result` + `qualification_result` on recovery outcome. |

**Precedent (compatible):** `ExecutionBoundCapabilityExecutionDispatchRequest` + `HostAvailableCapabilityBindingRequest` + `ExecutionBoundCapabilityExecutionDispatchService` — non-UCA provenance, same `QualifiedCapabilityExecutionTarget`, same handler registry pattern.

---

## 4 — Decision matrix

| Candidate design | Verdict |
|---|---|
| Reuse UCA acquisition/qualification facts on Variant B | **REJECT** |
| Synthesize UCA facts | **REJECT** |
| Optional UCA IDs on qualified dispatch | **REJECT** (no formal weakening proof) |
| `capability_ref` / `configuration_ref` string as execution target | **REJECT** |
| Reuse DIRECT_REUSE contracts unchanged for configured semantics | **REJECT** (host-available-specific) |
| New configured subject + configured binding + configured ingress (Option A) | **PREFERRED** |
| Generalize execution-bound intake (Option B) | **Only if bounded; default reject** — risks DIRECT_REUSE semantic migration |
| Second Execution Engine | **REJECT** |

**Dispatch ingress:** **Option A** — dedicated `ConfiguredCapabilityExecutionDispatchRequest` + configured runtime delegate (mirror `ExecutionBoundCapabilityExecutionDispatchService`; add `integration_configuration_adoption`).

---

## 5 — `ConfiguredCapabilityExecutionSubject` (normative)

Immutable typed subject **after** successful INT-CONFIG + adoption, **before** binding.

| Field (conceptual) | Semantics |
|---|---|
| `tenant_id` | Fulfillment tenant |
| `worker_need_id` | From `derive_worker_capability_need_id(need)` |
| `recovery_decision_id` | From `WorkerCapabilityNeed.recovery_decision_id` |
| `decision_id` | From `WorkerCapabilityAcquisitionDecision.decision_id` |
| `configuration_adoption_identity` | Deterministic digest of `ExecutionIntegrationConfigurationAdoption` + `ConfiguredCapabilityBinding.configuration_fingerprint` |
| `business_execution_target` | Typed `WorkerCapabilityBusinessExecutionTarget` (§6) — **not** provider, permission, or effective identity |
| `selected_operations` | Copy of need operations subset used for this dispatch (pilot: one of `database.query` / `database.execute`) |

**MUST NOT be:** `QualifiedCapabilitySubject`, raw dict/string payload, or qualification evidence.

**Subject reference (binding handle):**

```text
configured-capability-execution:{recovery_decision_id}:{decision_id}:{configuration_fingerprint}
```

---

## 6 — Authoritative business execution target source

**Finding @ HEAD:** No existing contract on the pure CONFIGURE_EXISTING path carries a typed Marketplace staged-tool handoff or catalog execution identity. `WorkerCapabilityCandidate.capability_ref` and `configuration_ref` are **opaque integration anchors**, not execution targets.

**Minimum additional typed contract (implementation wave — not in this child):**

`WorkerCapabilityBusinessExecutionTarget` (discriminated union), with pilot arm:

`MarketplaceQualifiedToolBusinessTarget(handoff_id: str)` — non-empty typed id, **not** parsed from `capability_ref`.

**Authoritative producers (discovery episode, before `decide`):**

1. **Primary:** discovery projection sets `business_execution_target` on the selected `WorkerCapabilityCandidate` when emitting `EXISTING_CONFIGURATION` for Marketplace database tools (Integrations/Marketplace discovery adapter — exact module in implementation map).
2. **Corroboration:** `WorkerCapabilityNeed.required_operations` supplies canonical operation names (`database.query` / `database.execute`) for `ConfiguredMarketplaceToolOperationSelector` equivalent on the configured path.
3. **Rejected sources:** parsing `capability_ref`, `configuration_ref`, `provider_id`, `resource_scope`, or operation string alone; opportunity record alone; adoption alone.

If `business_execution_target` is missing at configured binding time → **fail-closed** (`CONFIGURED_EXECUTION_TARGET_UNAVAILABLE`).

---

## 7 — Binding seam — `ConfiguredCapabilityExecutionBindingPort`

**Owner:** Platform contract in `intergrax/contracts/` (implementation wave); **Marketplace** reference provider behind SPI.

**Input:** `ConfiguredCapabilityExecutionBindingRequest` — subject, tenant, task/worker correlation, `ExecutionIntegrationConfigurationAdoption`, `configured_binding_operation_id`, `configured_execution_operation_id`, `requested_at`.

**Output:** `ConfiguredCapabilityExecutionBindingResult` with existing `QualifiedCapabilityExecutionTarget`.

**Provider MUST NOT:** materialize integration provider, execute I/O, authorize, or re-run INT-CONFIG.

**Marketplace pilot (RELATIONAL_STORE + `database.*`):**

- **Do not** call `MarketplaceToolQualifiedCapabilityBindingProvider` unchanged (UCA request).
- **Preferred:** factor shared **Marketplace qualified-tool stage → execution target reference** resolver (already used by qualified provider: `execution_target_reference_for_marketplace_qualified_tool`, stage repository, context resolver) into domain-internal module consumed by:
  - `MarketplaceToolQualifiedCapabilityBindingProvider` (UCA), and
  - `MarketplaceConfiguredCapabilityExecutionBindingProvider` (Variant B).
- **Ownership:** Marketplace/Tools domain resolves `MarketplaceQualifiedToolBusinessTarget` → `QualifiedCapabilityExecutionTarget`; AW and Integrations do not import staging internals.

**Deterministic IDs:**

| Identity | Derivation |
|---|---|
| `configured_execution_operation_id` | `configured-capability-execution:{recovery_decision_id}:{decision_id}` |
| `configured_binding_operation_id` | `configured-capability-binding:{configured_execution_operation_id}:{subject_reference}` |
| `execution_request_id` | `derive_qualified_capability_execution_request_id(resume_operation_id=configured_execution_operation_id, binding_operation_id=configured_binding_operation_id)` (same EE id family as qualified/DIRECT_REUSE) |

---

## 8 — `QualifiedCapabilityExecutionTarget` reuse decision

**Verdict:** **Reuse type; legacy field naming only.**

- `execution_target_reference` + `binding_provider_id` are semantically neutral routing handles.
- `qualified_subject_reference` is a **binding-subject handle** on all paths (DIRECT_REUSE already maps `host_subject_reference` into this field without implying UCA qualification).
- Configured path sets it to **`ConfiguredCapabilityExecutionSubject` subject reference** (§5) — **never** a UCA `QualifiedCapabilitySubject` reference.
- **No** new successor target type in this wave (minimality). Document invariants at bind time: configured provider MUST NOT accept UCA qualification artifacts.

---

## 9 — Configured execution intake

**Contract:** `ConfiguredCapabilityExecutionDispatchRequest` (new, Option A).

**Required fields:** `execution_request_id`, `execution_target`, `tenant_id`, `task_id`, `worker_instance_id`, `worker_need_id`, `configured_execution_operation_id`, `configured_binding_operation_id`, `recovery_decision_id`, `decision_id`, `integration_configuration_adoption`, `admitted_governance_identity`, `effective_authority_decision`, `collaborative_authority_scopes`, `requested_at`, optional `run_id` / `attempt_id`.

**Forbidden fields:** `acquisition_request_id`, `qualification_request_id`, `resume_operation_id` (UCA semantics).

**Runtime:** `ConfiguredCapabilityExecutionDispatchService` + delegate → canonical `ExecutionRuntime` → mints canonical `ExecutionId` (same as qualified/execution-bound).

**Handler registry:** Same host-composed `QualifiedCapabilityExecutionBindingHandlerRegistry` as UCA qualified and DIRECT_REUSE.

---

## 10 — Adoption propagation

- Configured intake **carries** `ExecutionIntegrationConfigurationAdoption`.
- Configured runtime delegate passes **exact adoption** into `QualifiedCapabilityExecutionBindingHandler.dispatch_once(...)`.
- **Forbidden:** adoption lookup by `ExecutionId`, global adoption registry, reconstruction on live path.

---

## 11 — Option A after INT-CONFIG (P0 §1B.11)

```text
successful INT-CONFIG → adoption → ConfiguredCapabilityExecutionSubject
  → ConfiguredCapabilityExecutionBindingPort.bind
  → WorkerExecutionAdmissionPort
  → ConfiguredCapabilityExecutionDispatchRequest
  → canonical ExecutionRuntime
  → Marketplace handler (Pattern A)
```

**No:** rediscovery, recovery reconcile gate, new acquisition, UCA qualification, `_fulfill_qualified`, discard-and-lookup.

---

## 12 — CONFIGURE_EXISTING decision entry (blocker 11)

**Unchanged from R1-R1 Q1–Q2:**

| Concern | Owner |
|---|---|
| Call `decide` | `WorkerCapabilityFulfillmentRecoveryAdapter` (`WorkerCapabilityRecoveryPort`) |
| Carry decision | `WorkerCapabilityRecoveryOutcome.worker_acquisition_decision` |
| Phase | `CONFIGURE_EXISTING_REQUIRED` when disposition is `CONFIGURE_EXISTING` |
| Coordinator | `WorkerCapabilityFulfillmentCoordinator.fulfill` → `_fulfill_configure_existing` → **new** `_fulfill_configured_execution` (implementation) |
| UCA recovery | Delegated to `WorkerCapabilityRecoveryCoordinator` **only** when `decide` routes `ACQUIRE_CAPABILITY` — never to manufacture Variant B facts |

---

## 13 — Governance ordering

```text
configured binding complete
  → WorkerExecutionAdmissionPort
  → admitted governance identity
  → canonical ExecutionRuntime admission
  → Marketplace handler
  → ToolRuntime Governance
  → Pattern A provider materialization / pin / I/O
```

INT-CONFIG authorization does **not** replace Execution admission or ToolRuntime authorization.

---

## 14 — Tenant continuity

```text
fulfillment tenant
  → decision / opportunity tenant
  → configured binding tenant
  → adoption tenant
  → configured subject tenant
  → binding request tenant
  → configured execution intake tenant
  → Execution tenant
  → pin tenant
```

Mismatch before materialization: materialization = 0, pin = 0, provider I/O = 0.

---

## 15 — Failure matrix (fail-closed; no UCA fallback)

| Condition | Outcome |
|---|---|
| Missing `worker_acquisition_decision` | FAIL_CLOSED |
| Disposition ≠ `CONFIGURE_EXISTING` | Deny configured path |
| Missing / invalid `configuration_ref` | Configured fulfillment deny |
| INT-CONFIG failure / deny | REALIZATION_FAILED / FAIL_CLOSED |
| Missing adoption after realization | FAIL_CLOSED |
| Missing `business_execution_target` | CONFIGURED_EXECUTION_TARGET_UNAVAILABLE |
| Ambiguous / unsupported operation | BINDING_NOT_SUPPORTED |
| Binding unavailable | BINDING_FAILED |
| Tenant mismatch (any hop) | 0 materialization / pin / I/O |
| Execution admission deny | REJECTED |
| Missing handler | Dispatch UNAVAILABLE |
| ToolRuntime Governance deny | Handler REJECTED |
| Configured vs effective mismatch at pin | FAIL_CLOSED |

---

## 16 — Ownership matrix

| Concern | Owner |
|---|---|
| CONFIGURE_EXISTING decision | `WorkerCapabilityAcquisitionDecisionService.decide` |
| CONFIGURE_EXISTING_REQUIRED phase | `WorkerCapabilityFulfillmentRecoveryAdapter` |
| INT-CONFIG / `ConfiguredCapabilityBinding` | Integrations |
| Adoption DTO | `WorkerConfiguredCapabilityFulfillmentService` |
| `ConfiguredCapabilityExecutionSubject` assembly | AW fulfillment (post-adoption) |
| Business target on candidate | Discovery projection (Marketplace/Integrations adapter) |
| Configured binding SPI | Platform contracts; Marketplace reference impl |
| Target resolution (handoff → EE target ref) | Marketplace/Tools (shared resolver) |
| Configured dispatch ingress | Execution runtime (new service, Option A) |
| Handler registry aggregation | Applications `uca6c_qualified_capability_execution_host_composition` |
| Execution authority | `ExecutionRuntime` / existing dispatch family |
| Composition wiring | Applications Tier-3 (no semantic logic) |

---

## 17 — Implementation file map (next wave)

| Area | Files |
|---|---|
| Subject + business target contracts | `intergrax/contracts/autonomous_work/configured_capability_execution.py` (**NEW**) |
| Binding port | `intergrax/contracts/capability_qualification/configured_capability_execution_binding.py` (**NEW**) |
| Configured dispatch | `intergrax/contracts/execution/configured_capability_execution_dispatch.py` (**NEW**) |
| Runtime delegate/service | `intergrax/runtime/execution/configured_capability_execution_dispatch_service.py` (**NEW**) |
| AW continuation | `worker_capability_fulfillment_coordinator.py`, `worker_configured_capability_execution_adapter.py` (**NEW**) |
| Recovery adapter | `worker_capability_fulfillment_recovery_adapter.py` (**NEW**, per R1-R1) |
| Marketplace configured binding | `intergrax/tools/marketplace_configured_capability_execution_binding_provider.py` (**NEW**) + factored resolver module |
| Shared target resolver | Factor from `marketplace_qualified_capability_binding_provider.py` |
| Intent (configured) | `ConfiguredMarketplaceToolExecutionIntent` or extend intent prep with non-UCA request (**NEW** wave) |
| Host composition | `uca6c_qualified_capability_execution_host_composition.py` |
| Discovery | Candidate projection setting `business_execution_target` |

**This child:** docs + gates only — **no** edits to rows above.

---

## 18 — STOP conditions

Return **STOP — ARCHITECTURE DECISION REQUIRED** if implementation cannot identify `business_execution_target` without UCA, string parsing, second Execution Engine, provider object in AW, or optional Governance.

**This lock:** STOP **not** triggered — minimum typed anchor defined (§6); path mirrors DIRECT_REUSE structurally.

---

## 19 — Next implementation wave exit criteria

1. `WorkerCapabilityBusinessExecutionTarget` on discovery-produced candidates (pilot).
2. End-to-end Variant B: adoption → subject → bind → configured dispatch → Pattern A with adoption on handler.
3. Qualified UCA + DIRECT_REUSE paths unchanged in behavior tests.
4. No synthetic UCA IDs in configured dispatch payloads.
5. Host single registry dispatches all three ingress kinds.

---

## Applicable FRZ (evidence only)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | OPEN — primary |
| FRZ-OWN-01, 02, 03, 05 | Ownership |
| FRZ-CTR-01, 02 | Contracts |
| FRZ-TYP-* (scoped) | Typed subjects |
| FRZ-PLG-* (scoped) | Binding SPI |
| FRZ-GOV-05 | Governance ordering |
| FRZ-EXE-01, 02 | Single execution authority |
| FRZ-TEN-* (scoped) | Tenant continuity |

---

## Tests (this child)

```text
uv run pytest -p no:xdist \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_production_composition_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_handoff_architecture_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_r1_configured_execution_subject_architecture_gates.py
```

**Status:** `TRACE-X-P5-R2-P3-R1-R1-R1` = **READY FOR AUDIT**
