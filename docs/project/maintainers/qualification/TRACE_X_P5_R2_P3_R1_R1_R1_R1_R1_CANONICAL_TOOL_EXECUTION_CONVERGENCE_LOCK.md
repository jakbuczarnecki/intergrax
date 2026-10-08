# TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 — Canonical Tool Target, Activation & Intent Convergence Lock

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1` |
| **Parent** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1` → `TRACE-X-P5-R2-P3-R1-R1-R1` → `TRACE-X-P5-R2-P3-R1-R1` → `TRACE-X-P5-R2-P3-R1` → `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **START_HEAD** | `79365c021c4637d13edb0e90fe4f65cd14023b88` |
| **Disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Production delta** | **0** |
| **FRZ-TRC-11** | **OPEN** |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **Supersedes (partial)** | [`TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md`](TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md) §8 target-resolution, §10 intent, handler/material convergence, tenant §17 imprecise `CapabilityIdentityKey tenant scope` wording; parent R1-R1-R1-R1 unresolved target/activation/intent portions |

## 1 — Canonical state (@ START_HEAD)

| Stage | Status |
|---|---|
| TRACE-X | CURRENT |
| TRACE-X-P5 | CURRENT / BLOCKED ON R2 |
| TRACE-X-P5-R2 | CURRENT / P3 BLOCKED |
| TRACE-X-P5-R2-P3 | BLOCKED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 | **CURRENT** (this lock) |
| P4 | NOT ENTERED |
| DUP-X | FINAL / MANDATORY (roadmap §3.0.3) |

---

## 2 — Independent audit acceptance @ `79365c021c4637d13edb0e90fe4f65cd14023b88`

**Accepted:** elimination of full configured DispatchService/Delegate duplicate direction; shared execution-bound ingress direction; `CapabilityIdentityKey` reuse; rejection of UCA handoff identity for pure Variant B; one handler registry; one `ExecutionRuntime`; one provider-resolution mechanism; one intent truth requirement.

**Rejected (new parent blockers):** blockers **16** and **17** (below).

---

## 3 — Blockers 16 / 17

### R2-P3-CAPABILITY-IDENTITY-TO-TOOL-TARGET-RESOLUTION-16

@ HEAD `MarketplaceToolQualifiedCapabilityBindingProvider` resolves:

```text
qualification → domain_handoff_reference → acquisition_request_id
  → MarketplaceQualifiedToolStageContext → handoff_id → stage → execution_target_reference
```

This is **Variant C / UCA-only**. It does **not** resolve `CapabilityIdentityKey → execution target` for CONFIGURE_EXISTING.

**Resolution:** **FACTOR SHARED CORE** `marketplace_tool_execution_target_resolution` (name TBD) with:

- **UCA thin adapter:** existing handoff/stage/context path → `QualifiedCapabilityExecutionTarget` (preserve `marketplace-qualified-tool:v1:{handoff_id}` encoding).
- **CONFIGURE_EXISTING thin adapter:** typed `CapabilityIdentityKey` + scope correlation evidence → **distinct** execution target encoding `catalog-tool-capability:v1:{stable_capability_identity_key_payload}` (typed serialization of `CapabilityIdentityKey` — **not** parsing `capability_ref` / `configuration_ref`).

Binding providers remain **no activation**; they only emit opaque `QualifiedCapabilityExecutionTarget`.

### R2-P3-MARKETPLACE-EXECUTION-LINEAGE-CONVERGENCE-17

@ HEAD `MarketplaceToolQualifiedCapabilityExecutionHandler` couples UCA lineage (handoff, stage, qualified activation resolver, handoff-gated intent/material) with Tool invocation.

**Resolution:** decompose handler into:

```text
UCA provenance adapter ─────────────┐
configured provenance adapter ─────┼→ execution-ready context (registry_tool_id + intent core + adoption)
                                     ↓
              ONE Marketplace Tool execution core (material → invocation_resolver → invoker)
```

**Forbidden:** second configured handler copying activation/invoke steps; `ConfiguredToolActivationResolver`; `ConfiguredToolRegistry`; `ConfiguredToolPackageResolver`; parallel intent stores.

---

## 4 — Current UCA execution lineage (@ HEAD)

```text
QualifiedCapabilityBindingRequest (DOMAIN_HANDOFF_REFERENCE + gap acquisition strategy)
  → MarketplaceToolQualifiedCapabilityBindingProvider.bind
  → execution_target_reference = marketplace-qualified-tool:v1:{handoff_id}
  → QualifiedCapabilityExecutionTarget

Pre-EE: QualifiedMarketplaceToolExecutionIntentPreparation (stage + handoff + operation selector)
  → QualifiedMarketplaceToolExecutionIntentRepository

Execution: MarketplaceToolQualifiedCapabilityExecutionHandler
  → parse handoff from target
  → load intent (handoff_id required)
  → load MarketplaceQualifiedToolStage
  → QualifiedMarketplaceToolActivationResolver.ensure_exact_active(stage)
  → QualifiedToolInvocationMaterialProvider (handoff_id in request)
  → QualifiedToolInvocationResolver → ExecutionBoundCatalogToolInvoker
  → optional ConfiguredIntegrationToolInvocationProjectionPort (Pattern A wiring only)
```

---

## 5 — Known-capability Tool lifecycle (@ HEAD)

```text
ToolPackageResolutionForIdentityPort.resolve_for_identity(CapabilityIdentityKey)
  → assert_exact_tool_package_resolution_for_identity
  → ToolHostActivationPort (+ ToolHostActivationMaterializer)
  → ToolRegistry / ToolRegistryRead
  → ToolRuntimeActivationMetadata (activation truth)
```

`ToolKnownCapabilityRealizationService` orchestrates the same ports for **UCA-2 known capability realization** (idempotent `operation_id`, may activate outside Marketplace EE handler).

**Locked reuse decision for Variant B execution path:** **do not** call `ToolKnownCapabilityRealizationService.realize` directly from Marketplace execution — it can activate outside the sanctioned EE admission point and carries UCA handoff disposition semantics. **Reuse** underlying contracts: `ToolPackageResolutionForIdentityPort`, `assert_exact_tool_package_resolution_for_identity`, `ToolHostActivationPort`, `QualifiedMarketplaceToolActivationResolver` **pattern** factored into a **configured provenance activation step** inside the handler (shared exact-release + metadata match logic with UCA resolver).

---

## 6 — Canonical identities

| Identity | Owner | Notes |
|---|---|---|
| Logical capability | `CapabilityIdentityKey` | Source-qualified; **no `tenant_id` field** @ `identity_key.py` |
| Exact release | `CapabilityReleaseIdentity` | Sole immutable release descriptor |
| Discovery projection | `CapabilityDiscoveryIdentity` → `CapabilityIdentityKey.from_discovery_identity` | Catalog evidence |
| Host activation truth | `ToolRuntimeActivationMetadata` on `ToolRegistryRead` | No second activation store |
| Execution request | `execution_request_id` + Execution intake | Distinct from capability identity |
| Configured subject | `ConfiguredCapabilityExecutionSubject` (future) | Carries `CapabilityIdentityKey` per R1-R1-R1 |

**Distinction:** logical identity ≠ exact release ≠ host `registry_tool_id` ≠ execution_request_id. Tenant/application scope is **orthogonal** to `CapabilityIdentityKey`.

---

## 7 — Production discovery gap (@ HEAD)

| Fact | Evidence |
|---|---|
| `WorkerConfigurationOpportunityDiscoveryPort` exists | `capability_acquisition_ports.py` |
| `EXISTING_CONFIGURATION` candidate kind exists | `capability_acquisition.py` |
| No production adapter writes `EXISTING_CONFIGURATION` candidates with `configuration_ref` | grep `intergrax/` + `agents/` — only tests / fulfillment consume |
| `WorkerCapabilityCandidate` lacks `CapabilityIdentityKey` | `capability_acquisition.py` @ HEAD |

**Locked future projection owner:** **Tier-3 host-composed** `WorkerConfigurationOpportunityDiscoveryPort` implementation that **joins** (no re-selection):

```text
scope-visible catalog / known-capability evidence (CapabilityIdentityKey, kind=TOOL)
  + tenant-scoped ExistingCapabilityConfigurationOpportunity (configuration_ref only)
  → WorkerCapabilityCandidate(
        kind=EXISTING_CONFIGURATION,
        capability_identity=CapabilityIdentityKey,  # contract MODIFY
        configuration_ref=...,
        capability_ref=derived display ref only — not identity authority
    )
```

**Join authority:** Integrations owns opportunity truth; Capability Catalog owns identity truth; AW discovery adapter performs **deterministic join** on correlated scope keys — **not** a second capability ranker.

**Forbidden:** inventing capability identity from `configuration_ref`, `provider_id`, or parsed `capability_ref`.

---

## 8 — Exact release decision (Variant B)

**YES** — Tool execution requires a resolved exact release before mutating activation.

| Path | Release source | Owner |
|---|---|---|
| UCA Variant C | `MarketplaceQualifiedToolStage.selected_release: CapabilityReleaseIdentity` | UCA staging (provenance carrier) |
| CONFIGURE_EXISTING | `ToolPackageResolutionForIdentityPort` + `assert_exact_tool_package_resolution_for_identity` | Tools domain — **no silent latest** |

If registry already holds exact active metadata matching resolved release → `ALREADY_ACTIVE_EXACT`; else activate via `ToolHostActivationPort` inside handler after Execution admission.

---

## 9 — Convergence-level decision

**Lowest truthful common execution level: D — `registry_tool_id` + `ToolRuntimeActivationMetadata` + canonical intent core.**

| Level | UCA | Variant B | Verdict |
|---|---|---|---|
| A `CapabilityIdentityKey` only | + exact release from stage | + package resolution | Too early for shared invoke core |
| B `CapabilityReleaseIdentity` | yes | yes (via resolution) | Shared for activation policy, not invoke |
| C `ToolHostActivationPort` | yes | yes | Shared lifecycle boundary |
| D active tool + metadata | yes | yes | **Shared invoke core starts here** |

Source-specific provenance resolves through C; **one** neutral core from D onward.

---

## 10 — Activation authority & timing

| Concern | Owner |
|---|---|
| Activation mutation | `ToolHostActivationPort` only |
| Activation truth read | `ToolRegistryRead.activation_metadata` |
| UCA exact activation policy | `QualifiedMarketplaceToolActivationResolver` (UCA adapter) |
| Configured exact activation policy | **FACTOR SHARED CORE** from resolver + `ToolPackageResolutionForIdentityPort` (configured adapter) |

**Timing:** mutating activation occurs **inside active canonical Execution**, in the Marketplace handler **after** intent/subject validation and **before** material/invoke — same as UCA @ HEAD. **Forbidden:** activation during discovery, configuration opportunity projection, binding-only adapter, or pre-`WorkerExecutionAdmissionPort` fulfillment steps.

---

## 11 — Target reference semantics

| Path | `execution_target_reference` | `binding_provider_id` | `qualified_subject_reference` |
|---|---|---|---|
| UCA | `marketplace-qualified-tool:v1:{handoff_id}` | `marketplace.tool.qualified_binding.v1` | UCA qualified subject |
| CONFIGURE_EXISTING | `catalog-tool-capability:v1:{typed CapabilityIdentityKey payload}` | same provider id | configured subject reference string |

`QualifiedCapabilityExecutionTarget` **REUSE** — reference encoding is adapter-specific; DTO remains source-neutral envelope.

**Forbidden for Variant B:** `marketplace-qualified-tool:v1:*`, synthetic `handoff_id`, parsing `capability_ref`.

---

## 12 — Intent convergence

**One semantic Tool execution intent** stored in **one** repository implementing `QualifiedMarketplaceToolExecutionIntentRepository`.

| Layer | Fields |
|---|---|
| **Common execution truth** | `execution_request_id`, `tenant_id`, `task_id`, `worker_need_id`, `selected_operation`, `qualified_subject_reference`, `binding_operation_id`, logical tool correlation (`capability_identity` and/or `activated_tool_id` after preparation) |
| **UCA provenance** | `handoff_id`, `resume_operation_id` |
| **CONFIGURE_EXISTING provenance** | configuration adoption / binding operation ids (typed extension or parallel provenance record — **same store**, versioned schema evolution) |

**Direction:** **MODIFY** `QualifiedMarketplaceToolExecutionIntent` (or v2 schema) — factor nullable UCA fields; add typed configured provenance; **one** `DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository`.

**Forbidden:** `ConfiguredMarketplaceToolExecutionIntentRepository`.

Preparation:

```text
UCA intent preparation adapter ─────┐
configured intent preparation adapter ├→ QualifiedMarketplaceToolExecutionIntentRepository
                                      ↓
                        MarketplaceToolQualifiedCapabilityExecutionHandler (core)
```

---

## 13 — Operation-selector convergence

`DefaultQualifiedMarketplaceToolOperationSelector` semantic core = **cardinality policy on `required_operations`** (0 / 1 / explicit multi policy). `stage` / `handoff_id` are **UCA policy context only**.

**FACTOR SHARED CORE** `tool_execution_operation_selection` (0/1/N policy); **THIN UCA ADAPTER** passes stage into optional multi-op policy; **CONFIGURE_EXISTING adapter** calls core with `required_operations` only.

**Forbidden:** `ConfiguredMarketplaceToolOperationSelector` duplicate.

---

## 14 — Handler decomposition

| Component | Classification |
|---|---|
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | **MODIFY** → orchestrator + shared core |
| UCA target/intent/stage parse block | **THIN SOURCE ADAPTER** |
| Configured target/intent/capability parse block | **THIN SOURCE ADAPTER — NEW** |
| Activation (UCA resolver) | **THIN ADAPTER** around shared exact-activation core |
| Activation (configured) | **THIN ADAPTER** using package resolution + `ToolHostActivationPort` |
| Material + invoke + Pattern A projection | **SHARED CORE** (existing lines 171–229 @ HEAD) |

---

## 15 — Material-provider convergence

`QualifiedToolInvocationMaterialRequest` currently requires `handoff_id`. Business material for configured DB tools does **not** require UCA handoff semantics.

**FACTOR SHARED CORE** material request without mandatory `handoff_id`; optional provenance bag for UCA correlation. **REUSE** `QualifiedToolInvocationMaterialProvider` implementations — widen request via factored contract, not second provider.

---

## 16 — Invocation-core reuse

**REUSE unchanged:** `QualifiedToolInvocationResolver` → `ExecutionBoundCatalogToolInvokeRequest` → `ExecutionBoundCatalogToolInvoker` → ToolRuntime Governance.

Provider Pattern A (`ExecutionIntegrationConfigurationAdoption` → projection → pin → I/O) **unchanged** — orthogonal to Tool identity.

---

## 17 — Activation responsibility matrix

| Concern | UCA Variant C | CONFIGURE_EXISTING | Shared canonical |
|---|---|---|---|
| logical identity | stage / qualification evidence | catalog join → `CapabilityIdentityKey` | `CapabilityIdentityKey` |
| exact release | `stage.selected_release` | package resolution port | `CapabilityReleaseIdentity` |
| package resolution | dynamic acquisition from stage | `ToolPackageResolutionForIdentityPort` | Tools catalog contracts |
| activation authorization | UCA qualification + EE admission | CONFIGURE_EXISTING decision + adoption + EE admission | `WorkerExecutionAdmissionPort` + Execution |
| activation mutation | `QualifiedMarketplaceToolActivationResolver` → `ToolHostActivationPort` | configured activation adapter → same port | `ToolHostActivationPort` |
| runtime activation truth | `ToolRegistryRead` | same | `ToolRuntimeActivationMetadata` |
| registry_tool_id | from activation result | from activation result | post-activation fact |

---

## 18 — Tenant continuity

```text
tenant_id / application scope (need, opportunity, binding, intake)
  → catalog visibility evidence (scope-filtered discovery)
  → EXISTING_CONFIGURATION candidate (capability_identity + configuration_ref)
  → CONFIGURE_EXISTING decision
  → configuration opportunity read (tenant match)
  → INT-CONFIG binding + adoption
  → configured subject
  → target bind (subject + identity encoding)
  → intent prepare (tenant_id match)
  → WorkerExecutionAdmissionPort
  → Execution intake (tenant_id match)
  → activation read (registry scope)
  → provider pin (tenant_id)
```

**Verdict:** **PASS — architecture proof complete** (hops explicit; `CapabilityIdentityKey` is not tenant authority; cross-tenant mismatch must fail closed before I/O). **Runtime evidence @ HEAD:** Variant B production path **not yet wired** — implementation must prove each hop.

---

## 19 — Governance ordering

```text
CONFIGURE_EXISTING realization → adoption → configured subject/binding
  → WorkerExecutionAdmissionPort
  → shared bound-execution ingress → ExecutionRuntime
  → provenance adapter → shared Tool execution core
  → ToolRuntime Governance
  → Pattern A provider materialization (optional adoption)
  → provider I/O
```

Configuration authorization ≠ Tool activation ≠ Execution admission ≠ ToolRuntime authorization.

---

## 20 — Ownership matrix

| Concern | Exactly-one owner |
|---|---|
| Capability identity | Capability Catalog (`CapabilityIdentityKey`) |
| Release identity | `CapabilityReleaseIdentity` |
| Catalog visibility | Discovery adapters + scope context |
| Configuration opportunity | Integrations (`ExistingCapabilityConfigurationOpportunity`) |
| CONFIGURE_EXISTING decision | `WorkerCapabilityAcquisitionDecisionService` |
| Provider realization | INT-CONFIG |
| Configured adoption | `WorkerConfiguredCapabilityFulfillmentService` |
| EXISTING_CONFIGURATION join projection | Tier-3 host discovery adapter (thin) |
| Package resolution | `ToolPackageResolutionForIdentityPort` |
| Tool host activation | `ToolHostActivationPort` |
| Runtime activation truth | `ToolRegistryRead` |
| Operation selection | factored selection core |
| Execution intent store | `QualifiedMarketplaceToolExecutionIntentRepository` |
| Handler routing | `QualifiedCapabilityExecutionBindingHandlerRegistry` |
| Invocation material | application `QualifiedToolInvocationMaterialProvider` |
| Invocation resolution | `QualifiedToolInvocationResolver` |
| ToolRuntime authorization | ToolRuntime Governance |
| Integration provider materialization | `ExecutionBoundIntegrationResolution` |

---

## 21 — Duplicate classification matrix

| Component | Classification |
|---|---|
| `MarketplaceToolQualifiedCapabilityBindingProvider` | **CANONICAL OWNER** (UCA bind) + **FACTOR CORE** for target encoding |
| Future configured binding adapter | **THIN TYPED ADAPTER** |
| `ToolKnownCapabilityRealizationService` | **DISTINCT RESPONSIBILITY** (UCA-2 realization port) — reuse ports only |
| `QualifiedMarketplaceToolActivationResolver` | **CANONICAL OWNER** (UCA activation policy) |
| Shared exact-activation core (future) | **SHARED CORE** |
| `ToolHostActivationPort` | **CANONICAL OWNER** |
| `ToolRegistryRead` | **CANONICAL OWNER** (read truth) |
| UCA intent preparation | **THIN SOURCE ADAPTER** |
| Configured intent preparation (future) | **THIN SOURCE ADAPTER** |
| `QualifiedMarketplaceToolExecutionIntentRepository` | **CANONICAL OWNER** (single store) |
| `DefaultQualifiedMarketplaceToolOperationSelector` | **FACTOR SHARED CORE** |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | **MODIFY** — shared core + adapters |
| `QualifiedToolInvocationMaterialProvider` | **SHARED CORE** (factored request) |
| `QualifiedToolInvocationResolver` | **CANONICAL OWNER** |
| `ConfiguredIntegrationToolInvocationProjectionPort` | **THIN ADAPTER** (Pattern A only) |
| `ConfiguredTool*` registries/resolvers (forbidden names) | **N/A — must not exist** @ HEAD verified |

---

## 22 — Responsibility-delta proofs (summary)

1. **UCA binding provider** — owns qualification-to-handoff bind only; cannot replace configured identity bind without handoff (**adapter** required).
2. **Configured bind adapter** — validates configured subject + projects identity encoding; no activation (**thin**).
3. **UCA handler provenance block** — loads stage/handoff; cannot be deleted without losing UCA evidence (**adapter**).
4. **Shared invoke core** — identical material/invoke/Governance for both paths (**core**).
5. **ToolKnownCapabilityRealizationService** — owns UCA-2 operation_id realization lifecycle; EE Variant B uses same ports at EE timing (**distinct** lifecycle — not duplicate if ports reused).

---

## 23 — Failure matrix (fail closed)

| Condition | Result |
|---|---|
| missing typed `CapabilityIdentityKey` on configured path | reject before bind |
| capability kind ≠ TOOL | reject |
| tenant/scope mismatch (need/opportunity/binding/intent/intake) | reject |
| exact release unresolved / ambiguous | reject |
| release conflict with active metadata | reject |
| Tool unavailable / activation denied | reject |
| ambiguous operation (0 or >1 without policy) | reject |
| intent missing/corrupt/subject mismatch | reject |
| configured adoption mismatch | reject |
| handler/provider/Governance deny | reject |
| configured/effective provider mismatch | reject |

**No** fallback to UCA, alternate Tool release, or profile provider shortcut.

---

## 24 — Future implementation map

| Component | Action |
|---|---|
| `capability_acquisition.py` (`WorkerCapabilityCandidate`) | **MODIFY** — add `capability_identity: CapabilityIdentityKey \| None` |
| Tier-3 `WorkerConfigurationOpportunityDiscoveryPort` adapter | **THIN ADAPTER — NEW** |
| `marketplace_tool_execution_target_resolution` (factor) | **FACTOR SHARED CORE** from binding provider |
| `marketplace_configured_capability_execution_binding_provider.py` | **THIN ADAPTER — NEW** |
| `qualified_marketplace_tool_execution_intent.py` | **MODIFY** — source-neutral + provenance fields |
| `marketplace_qualified_tool_execution_intent_preparation.py` | **MODIFY** — UCA adapter |
| configured intent preparation module | **THIN ADAPTER — NEW** |
| `qualified_tool_invocation.py` (`MaterialRequest`) | **MODIFY** — factor handoff optional |
| `marketplace_qualified_capability_execution_handler.py` | **MODIFY** — adapters + shared core |
| `qualified_marketplace_tool_activation_resolver.py` | **FACTOR SHARED CORE** with configured activation adapter |
| `qualified_marketplace_tool_operation_selector.py` | **FACTOR SHARED CORE** |
| `ToolKnownCapabilityRealizationService` | **REUSE** ports only on configured path |
| Forbidden `ConfiguredTool*` services | **NEW SEMANTIC MECHANISM = 0** |

---

## 25 — STOP conditions

Return **STOP — ARCHITECTURE DECISION REQUIRED** if implementation requires: second activation authority; second registry; duplicate release resolver; second intent store; configured-only executor; UCA handoff on pure Variant B; string inference targets; pre-admission activation without sanction; silent latest release; provider id as Tool identity; universal nullable intent DTO as permanent dual truth.

**This lock:** STOP **not** triggered.

---

## 26 — Implementation exit criteria

1.–23. per task §53 — locked in sections above; parent P3 remains BLOCKED until independent implementation audit.

**Status:** `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1` = **READY FOR AUDIT**

**Proposed new semantic mechanisms:** **0**

---

## Applicable FRZ (evidence only)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | **OPEN** — primary |
| FRZ-OWN-01, 03, 04, **05** | Single ownership |
| FRZ-CTR-01, 02 | Contracts |
| FRZ-TYP-* (scoped) | Strong typing |
| FRZ-PLG-01, 02, 05 | Adapters |
| FRZ-RPL-01, 02, 04 | Replaceability |
| FRZ-GOV-05 | Governance ordering |
| FRZ-EXE-01, 02 | Execution authority |
| FRZ-TEN-* (scoped) | Tenant continuity |
| FRZ-REG-* (scoped) | Registry truth |

---

## Tests (this child)

```text
uv run pytest -p no:xdist \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_production_composition_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_handoff_architecture_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_r1_configured_execution_subject_architecture_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_r1_r1_convergence_architecture_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_r1_r1_r1_tool_execution_convergence_architecture_gates.py
```
