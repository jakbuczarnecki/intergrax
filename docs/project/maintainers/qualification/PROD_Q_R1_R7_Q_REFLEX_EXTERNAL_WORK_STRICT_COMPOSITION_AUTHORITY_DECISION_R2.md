# PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-STRICT-COMPOSITION-AUTHORITY-DECISION-R2

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (bounded architecture decision — documentation only) |
| **Date** | 2026-10-18 |
| **Branch** | `development` |
| **START_HEAD / evidence SHA** | `a7f3a46296ae58cd4728db492429e99dbfdb8c82` |
| **Parent** | **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** — independent audit **REJECTED** IMPL-01-R1 (unqualified composition authority + production durability) |
| **Preserved (not reopened)** | [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md) **A+B** · [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md) · [`PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md`](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md) **Option B** · [`PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md`](PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md) |
| **Roadmap** | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) (PROD-Q-R1-R7-Q slice — no stage promotion) |
| **Freeze checklist** | [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — **no global FRZ PASS** |
| **Production delta (this task)** | **0** |
| **Recommended child** | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01-R2-STRICT-COMPOSITION-RECOVERY` |
| **Does not close** | **IMPL-01** · **IMPL-01-R1** · **R7-Q** · **PROD-Q** · **Package P** interpretation **B** · any global **FRZ-*** PASS |

**Purpose:** One coherent corrective architecture decision for independent-audit findings **F-R2-01…03** @ `a7f3a462…` — strict external-work composition authority, revision-bound AC3 materialization, and provider capability identity — without new orchestrator, PIS, dispatch gate, or follow-on store beyond ratified F02.

---

## 1. Git baseline and closed-world evidence

| Check | Result |
|-------|--------|
| Branch | `development` |
| `HEAD` | `a7f3a46296ae58cd4728db492429e99dbfdb8c82` (= required **START_HEAD**) |
| Ancestry | `a7f3a462…` is ancestor of `HEAD` (trivial equality) |
| Production / test code changed in this task | **0** |
| Test execution in this task | **0** (`pytest` / `uv run` forbidden) |

---

## 2. Closed-world contract / owner / caller inventory

### 2.1 Target pipeline (single sanctioned graph)

```text
Production admission + revision-bound artifact authority (AC-3 / RosterAgentContractAuthority)
  → exactly-one sanctioned agent materialization (no overlay)
  → host-injected typed SanctionedExternalWorkOperationPort
  → GovernedExternalWorkOrchestrator (existing)
  → production-qualified GCA store bundle + durable PIS
  → GovernedProviderInvocationDispatchGate (existing — no bypass)
```

### 2.2 Semantic owners (unchanged from ratified ADRs)

| Step | Semantic owner | Composition owner | Canonical contracts / symbols |
|------|----------------|-------------------|--------------------------------|
| HTTP strict admission | GCA serving + P3-A2 | `factory.py` · `fastapi_router.py` | `GovernedContractorRunRequestV1` · `StrictProductionServingTenantAuthorityTaskExecutionPort` |
| Task execution routing | Runtime execution | `StrategyExecutionRouter` · `HostTaskExecutionPort` | `Task` · `ExecutionRequest` |
| Agent contract identity | Agent Distribution AC-3 / AC-5 | `MaterializedRegistryProjection` · `RosterAgentContractAuthority` | `AgentContract` · `RuntimeAgentFactoryResolver` |
| Tier-2 reflex entry | `ExternalContractorAdapterAgent` | **Must** match revision-bound materialization | `external_contractor.adapt` |
| Mutating external work | `GovernedExternalWorkOrchestrator` | `build_governed_external_work_production_runtime` | `SanctionedExternalWorkOperationPort` · `GovernedSanctionedExternalWorkOperationPort` |
| PIS intent/outcome | ER + GCA gate | `ProductionProcessComposition.provider_invocation_store` | `ProviderInvocationStore` · `GovernedProviderInvocationDispatchGate` |
| GCA execution / receipt / bundle / continuation | GCA orchestrator | **This decision:** process-scoped injected bundle | `GovernedExecutionStore` · `ProofReceiptStore` · `PolicyBundleArtifactStore` · `ContinuationStateStore` |
| Follow-on create snapshot | Orchestrator only | F02 Option B on **same** `GovernedExecutionStore` | `GovernedExternalWorkCreateFollowOnRecordV1` (ratified — child IMPL) |
| Provider feature flags | Orchestrator policy | **This decision:** integration discover + operator pin | `ExternalWorkProviderCapabilities` · `ExternalWorkIntegration.discover()` → `ExternalWorkProviderDescriptor` |

### 2.3 Callers @ evidence SHA

| Caller | Callee | Finding relevance |
|--------|--------|-------------------|
| `create_governed_contractor_backend_app` (`factory.py` L262–270) | `build_strict_external_work_host_session_resources` | F-R2-01 fallback stores; F-R2-03 fixture capabilities |
| `create_governed_contractor_backend_app` (`factory.py` L346–355) | `rebind_strict_sanctioned_external_contractor_adapter_registry` | F-R2-02 overlay |
| `build_strict_external_work_host_session_resources` (`strict_external_work_host_wiring.py` L70, L77) | `build_process_scoped_governed_external_work_host_stores` · `quote_first_partner_capability_fixture` | F-R2-01 · F-R2-03 |
| `build_governed_external_work_production_runtime` (`production_external_work_composition.py` L291–308) | `provider_invocation_store.is_durable` | **Correct** strict gate for PIS only |
| Revision-bound registry | `registry_projection.agent_registry` (`harness_registry_authority.py` L93–112) | `builders=` **forbidden** — overlay is not an approved seam |
| Manifest scaffold | `build_governed_contractor_external_contractor_adapter_from_context` (`agent_factories.py` L11–20) | AC3 materialization without host ports |

---

## 3. Independent audit findings (F-R2-01…03)

### F-R2-01 — Production store fallback

**Source:** `applications/governed_contractor_application/host/strict_external_work_host_wiring.py`

| Observation @ `a7f3a462…` | Classification |
|---------------------------|----------------|
| `resolved_stores = host_stores or build_process_scoped_governed_external_work_host_stores()` (L70) | **IN-SCOPE BLOCKER** — silent in-memory GCA store minting on strict graph while PIS may be durable |
| `build_process_scoped_governed_external_work_host_stores()` (`governed_external_work_host_stores.py` L31–37) | `InMemoryGovernedExecutionStore` · `InMemoryProofReceiptStore` · `InMemoryPolicyBundleArtifactStore` · `InMemoryContinuationStateStore` |
| Injection boundary exists | `host_stores: GovernedExternalWorkHostStoreBundle \| None` parameter (L54) — **optional**, not wired from `ProductionProcessComposition` |
| Durable GCA implementations in-tree | **None qualified for production** — only `InMemory*` + `FilesystemHostStore` (`stores.py` L252+, docstring: demo; used by `offline_demo.py`, CLI, unit tests) |
| PIS durability gate | `production_external_work_composition.py` — `is_durable` required; **no equivalent** on GCA host store bundle today |
| Startup when durable GCA missing | **No fail-closed** — fallback constructs ephemeral stores |

**Mandatory decision (F-R2-01):** Strict external-work **cannot** silently construct in-memory GCA stores. Lab/demo filesystem persistence **must not** be promoted to production without separate qualification evidence.

---

### F-R2-02 — AC3 registry overlay

**Source:** `applications/governed_contractor_application/host/strict_revision_bound_agent_registry_rebind.py`

| Observation | Classification |
|-------------|----------------|
| `_StrictSanctionedExternalWorkAgentRegistryOverlay` wraps `AgentRegistryRead` | **IN-SCOPE BLOCKER** — second agent resolution authority post materialization |
| `get` / `get_contract` override one `agent_id` | Partial override — `list_contracts` (L64–65) **does not** substitute strict agent contract |
| `find_by_capability` / `find_best_match` | Replaces instances only for `external_contractor.adapt` / matching id — duplicate lookup path vs revision-bound registry |
| `StrategyExecutionRouter` / execution | Uses materialized registry from projection; overlay patched **after** `build_harness_host_runtime` (`factory.py` L346–355) |
| Sanctioned host factory mapping | `build_governed_contractor_agent_builders` (`agent_builders.py` L51–64) — **correct** mapping, **not** used for REVISION_BOUND assembly (`harness_registry_authority.py` L103–106 forbids `builders`) |
| Scaffold factory | `build_governed_contractor_external_contractor_adapter_from_context` — no port, no boundary |

**Mandatory decision (F-R2-02):** Registry, contract, and implementation resolution **must** derive from **one** authoritative revision-bound composition. **Remove** overlay module in child IMPL — **no** compatibility retention.

**Approved dependency-injection seam (smallest change to existing owners):**

| Owner | Extension | Operator ratification |
|-------|-----------|----------------------|
| `ColocatedServingUnitRuntimeAccessor` + `ServingRuntimeHandoff` (`serving_runtime_handoff.py` L53–64) | Host supplies `RuntimeAgentFactoryResolver` that **delegates** to revision-bound venv resolver for all contracts **except** `external_contractor_adapter`, where it invokes `build_governed_contractor_agent_builders(resolved_host_runtime)` factory | **REQUIRED** — new handoff field or accessor method; not a second registry |
| `create_governed_contractor_backend_app` (`factory.py`) | Register accessor closure capturing **frozen** `GovernedContractorHostRuntimeComposition` **after** sanctioned port attach (L271–274) | Implementation detail of child IMPL |
| `establish_serving_runtime_handoff` / colocated composition | Wire accessor into serving unit warmup | Same |

If operator **rejects** extending `ColocatedServingUnitRuntimeAccessor` / handoff → **STOP — ARCHITECTURE DECISION REQUIRED** (no overlay workaround permitted).

---

### F-R2-03 — Provider capability and identity

**Sources:** `strict_external_work_host_wiring.py` L55, L77 · `production_external_work_composition.py`

| Observation | Classification |
|-------------|----------------|
| Default `provider_id="gec3_deterministic_fake"` (L55) | Qualification / fake adapter identity — **not** operator-authorized production config |
| `capabilities=quote_first_partner_capability_fixture(provider_id=provider_id)` (L77) | **IN-SCOPE BLOCKER** — qualification fixture on strict composition path reachable from `factory.py` L262–270 when strict external-work stack attaches |
| Canonical production contract for capabilities | `ExternalWorkProviderCapabilities` (`intergrax/contracts/external_work_provider_capabilities.py`) — orchestrator input |
| Canonical provider identity discovery | `ExternalWorkIntegration.discover()` → `ExternalWorkProviderDescriptor` (`intergrax/integrations/contracts/external_work.py` L72–74) |
| Strict reachability | **Yes** when `strict_topology_reliability` + integration + durable PIS + policy bundle + repos + task scope (factory L239–274) |
| Operator authorization / revision bind | **NOT PROVEN** — no settings pin or digest check @ HEAD |

**Mandatory rule (F-R2-03):** Never infer production capabilities from `quote_first_partner_capability_fixture`. Strict path must use **typed** projection from `integration.discover()` plus **fail-closed** operator pin (expected `provider_id` / descriptor digest) from approved settings — not hardcoded defaults.

---

## 4. Options considered and rejected

| Option | Rejection rationale |
|--------|---------------------|
| Retain `_StrictSanctionedExternalWorkAgentRegistryOverlay` as “compat” | Violates FRZ-PRD-08 / FRZ-OWN-* / ADR-01 single materialization; incomplete `list_contracts`; hidden alternate lookup |
| Default `host_stores or InMemory*` in strict mode | Split-brain vs durable PIS; violates FRZ-REL-03 / FRZ-REL-05 durability posture for host execution truth |
| Promote `FilesystemHostStore` to strict production default | Demo/lab only — no production durability qualification; violates enterprise env rule (no silent lab persistence) |
| `compose_builders` on `build_harness_host_runtime` for strict production | **Forbidden** for `RegistryAssemblyMode.REVISION_BOUND` (`harness_registry_authority.py` L103–106) |
| New agent registry / resolver / materialization service | New architectural authority — **not ratified** |
| Infer capabilities from PIS or GER | Forbidden closed-world inference (F02 §3) |
| HTTP / metadata-supplied `provider_id` alone | Not revision-bound or operator-authorized |
| Second dispatch gate or invocation store | Explicitly forbidden by ADR-01 |

---

## 5. Recommended unified architecture (single decision)

### 5.1 Composition authority (F-R2-01)

1. **Exactly-one composition owner** for the **instance bundle** fed to `build_governed_external_work_production_runtime`: `create_governed_contractor_backend_app` in `applications/governed_contractor_application/host/factory.py`, sourcing stores from **`ProductionProcessComposition`** (new optional field) or an explicitly injected `GovernedExternalWorkHostStoreBundle` on the process opener — **same lifetime** as `provider_invocation_store` (process-scoped, shared across strict HTTP requests via `GovernedExternalWorkHostSessionResources`).

2. **Protocol mirror (child IMPL):** extend `GovernedExternalWorkHostStoreBundle` or each store protocol with `is_durable: bool` (same semantic as `ProviderInvocationStore.is_durable` @ `stores.py` L191–192 pattern). `build_governed_external_work_production_runtime` gains a **strict-only** precondition: all four GCA stores `is_durable is True` when `execution_mode` is strict — parallel to PIS gate @ `production_external_work_composition.py` L301–307.

3. **Fail-closed:** Remove optional fallback in `build_strict_external_work_host_session_resources` — `host_stores` **required** when caller is strict factory path; missing bundle → `HarnessHostOrchestrationTopologyReliabilityCompositionError`. Missing durable implementation when strict policy demands durability → startup **reject** (no in-memory silent default).

4. **`build_process_scoped_governed_external_work_host_stores`:** retain **only** for explicit lab/tests/offline_demo call sites — **not** callable from strict factory path (static gate in child).

5. **F02:** `GovernedExternalWorkCreateFollowOnRecordV1` remains **only** follow-on authority on extended `GovernedExecutionStore` — no parallel snapshot service (ratified F02 unchanged).

### 5.2 Revision-bound AC3 (F-R2-02)

1. **Delete** `strict_revision_bound_agent_registry_rebind.py` and remove `rebind_strict_sanctioned_external_contractor_adapter_registry` from `factory.py`.

2. **Materialize** `ExternalContractorAdapterAgent` through **one** `RuntimeAgentFactoryResolver` chain established at serving handoff, with host closure supplying `sanctioned_external_work_operation_port` + `require_sanctioned_external_work_mutations=True` + `meaningful_side_effect_authorization_boundary` + `external_work_integration` from `GovernedContractorHostRuntimeComposition`.

3. **Registry projection** remains derived-only; **no** post-hoc `AgentRegistryRead` wrapper.

4. **Manifest scaffold** `build_governed_contractor_external_contractor_adapter_from_context` stays for non-strict manifest development only — strict serving must not route traffic to agents built solely through this symbol.

### 5.3 Provider identity and capabilities (F-R2-03)

1. **New host composition function (child IMPL):** e.g. `resolve_strict_external_work_provider_capabilities(integration, *, settings)` in `applications/governed_contractor_application/host/` (or `intergrax/contracts` mapper if pure):

   - Call `descriptor = integration.discover()`.
   - Map `ExternalWorkProviderDescriptor` → `ExternalWorkProviderCapabilities` via **explicit typed table** (`ExternalWorkCapability` → boolean flags) — **no** `quote_first_partner_capability_fixture` in strict path.
   - Validate `descriptor.identity` against operator pin from `GovernedContractorBackendSettings` (new typed fields — child IMPL; env-driven per FRZ env rules).
   - On absent integration, failed discover, or pin mismatch → **startup fail-closed** for strict external-work stack.

2. Remove default `provider_id="gec3_deterministic_fake"` from strict wiring public API.

3. Qualification tests may continue importing `quote_first_partner_capability_fixture` **only** in test modules — not in `strict_external_work_host_wiring.py` / `factory.py` strict path.

---

## 6. Before / after call graphs

### 6.1 Before (@ `a7f3a462…` — rejected)

```text
create_governed_contractor_backend_app
  ├─ ProductionProcessComposition.provider_invocation_store (durable) ✓
  ├─ build_strict_external_work_host_session_resources
  │     ├─ [FALLBACK] build_process_scoped_governed_external_work_host_stores() → InMemory*
  │     └─ [FIXTURE] quote_first_partner_capability_fixture(provider_id=…)
  ├─ attach_sanctioned_external_work_to_host_runtime
  ├─ build_harness_host_runtime → MaterializedRegistryProjection.agent_registry
  │     └─ ExternalContractorAdapterAgent (scaffold / revision-bound without ports)
  └─ replace(registry → _StrictSanctionedExternalWorkAgentRegistryOverlay)  [F-R2-02]
        └─ StrategyExecutionRouter → patched get / find_best_match only
```

**Negative paths (before):** strict stack attaches with durable PIS + ephemeral GCA stores; router may see scaffold contracts via `list_contracts`; capabilities overstated via fixture.

### 6.2 After (R2 recovery target)

```text
ProductionProcessComposition
  ├─ provider_invocation_store (durable, existing)
  └─ governed_external_work_host_store_bundle (injected, is_durable=True when qualified)

create_governed_contractor_backend_app
  ├─ resolve_strict_external_work_provider_capabilities(integration, settings)
  ├─ build_strict_external_work_host_session_resources(host_stores=bundle, capabilities=resolved)
  │     └─ build_governed_external_work_production_runtime → orchestrator + gate
  ├─ attach_sanctioned_external_work_to_host_runtime
  ├─ establish_serving_runtime_handoff + ColocatedServingUnitRuntimeAccessor
  │     └─ RuntimeAgentFactoryResolver (host delegates AC3 adapter factory only)
  └─ build_harness_host_runtime(registry_projection=…)  [no registry replace]
        └─ StrategyExecutionRouter → single registry authority
              └─ ExternalContractorAdapterAgent (strict ports) → domain_job → SanctionedExternalWorkOperationPort
```

**Negative paths (after):**

| Condition | Outcome |
|-----------|---------|
| Strict + external work + missing `governed_external_work_host_store_bundle` | `HarnessHostOrchestrationTopologyReliabilityCompositionError` at startup |
| Strict + bundle present + any store `is_durable is False` | `HarnessHostOrchestrationTopologyReliabilityCompositionError` (mirror PIS message class) |
| Strict + capability pin mismatch vs `discover()` | Startup fail-closed |
| `require_sanctioned_external_work_mutations` + missing port on materialized agent | Tier-2 deny (`domain_job.py`) + composition must prevent reachability |
| Accept/cancel without F02 follow-on record | `prior_create_snapshot_missing` — no provider mutation |

---

## 7. File-level corrective implementation blueprint (child — not this task)

| File | Action |
|------|--------|
| `intergrax/applications/_shared/production_process_composition.py` | Add optional `governed_external_work_host_store_bundle: GovernedExternalWorkHostStoreBundle \| None` (TYPE_CHECKING import from GCA host module or protocol in `intergrax/contracts` if boundary requires) — **operator ratification** |
| `intergrax/applications/_shared/serving_runtime_handoff.py` | Extend handoff / `ColocatedServingUnitRuntimeAccessor` to carry host `RuntimeAgentFactoryResolver` decoration — **operator ratification** |
| `applications/governed_contractor_application/host/factory.py` | Wire bundle from composition; remove overlay `replace`; register serving accessor; fail-closed gaps per ownership ADR F05 |
| `applications/governed_contractor_application/host/strict_external_work_host_wiring.py` | Require `host_stores`; remove in-memory fallback; remove fixture/default provider_id; accept resolved `ExternalWorkProviderCapabilities` |
| `applications/governed_contractor_application/host/governed_external_work_host_stores.py` | Document lab-only builder; optional `is_durable` on bundle |
| `applications/governed_contractor_application/host/stores.py` | `is_durable` on GCA store implementations; F02 follow-on methods per ratified F02 |
| `applications/governed_contractor_application/host/production_external_work_composition.py` | Strict GCA store durability precondition (symmetric to PIS) |
| `applications/governed_contractor_application/host/strict_revision_bound_agent_registry_rebind.py` | **DELETE** |
| `applications/governed_contractor_application/host/agent_builders.py` | Unchanged — canonical port mapping |
| `applications/governed_contractor_application/host/sanctioned_external_work_operation_port.py` | Remove session shadow dict (ownership ADR F02) |
| `applications/governed_contractor_application/host/orchestrator.py` | F02 writer/resolver + follow-on record |
| `intergrax/contracts/governed_external_work_create_follow_on.py` | F02 typed record (ratified) |
| `applications/governed_contractor_application/host/settings.py` | Typed strict external-work provider pin fields (env-backed) |
| New: `resolve_external_work_provider_capabilities_from_descriptor.py` (location TBD in child) | Typed mapper descriptor → capabilities |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_ownership_recovery_architecture_gate.py` (extend) | Static gates: no overlay import; no fixture in strict wiring; no silent store fallback |

**Forbidden:** new registry overlay; Tier-2 `applications/` imports; second orchestrator; new invocation store; strict use of `FilesystemHostStore` without qualification record.

---

## 8. Migration / removal plan

| Phase | Action |
|-------|--------|
| R2-IMPL-1 | Add composition fields + durability protocol + factory wiring (fail-closed) |
| R2-IMPL-2 | Serving handoff resolver hook; delete overlay module |
| R2-IMPL-3 | Capability resolver + settings pin; remove fixture from strict path |
| R2-IMPL-4 | F02 store extension + orchestrator resolver (may overlap IMPL-01-R1 ownership child — same branch policy as operator directs) |
| R2-QUAL | Independent exact-SHA audit + deferred tests below |

No feature flag to re-enable overlay or in-memory strict fallback.

---

## 9. Acceptance gates (positive / negative) — child qualification

| Gate | Type | Criterion |
|------|------|-----------|
| G-R2-01 | Positive | Strict factory with durable PIS + durable GCA bundle starts; single `GovernedExternalWorkHostSessionResources` lifetime |
| G-R2-02 | Negative | Strict factory without `host_stores` / bundle → composition error (no InMemory mint) |
| G-R2-03 | Negative | Non-durable GCA store in strict bundle → same class of error as PIS non-durable |
| G-R2-04 | Negative | Static gate: `strict_revision_bound_agent_registry_rebind` absent; `factory.py` no `replace(registry,…)` |
| G-R2-05 | Positive | Materialized AC3 agent has `require_sanctioned_external_work_mutations=True` and non-None port without registry wrapper |
| G-R2-06 | Negative | Strict wiring / factory import path does not reference `quote_first_partner_capability_fixture` |
| G-R2-07 | Negative | Provider pin mismatch → startup fail-closed |
| G-R2-08 | Positive | Integrated reflex CREATE traverses PIS + gate (extends Package P interpretation **B** — **NOT PROVEN** until executed) |
| G-R2-09 | Negative | Restart accept without F02 record → `prior_create_snapshot_missing` |

---

## 10. Deferred test inventory — **NOT EXECUTED**

| Test (proposed) | Module | Gate |
|-----------------|--------|------|
| `test_strict_external_work_rejects_missing_host_store_bundle` | `applications/.../tests/host/` | G-R2-02 |
| `test_strict_external_work_rejects_non_durable_gca_stores` | same | G-R2-03 |
| `test_strict_factory_does_not_import_capability_fixture` | `tests/unit/applications/governed_contractor/` | G-R2-06 |
| `test_revision_bound_ac3_agent_has_sanctioned_port_without_overlay` | same | G-R2-05 |
| `test_provider_capability_pin_mismatch_fail_closed` | host tests | G-R2-07 |
| Extend architecture gate @ git status path | `test_prod_q_r1_r7_q_ownership_recovery_architecture_gate.py` | G-R2-04 |

**DEFERRED — NOT EXECUTED** (task resource limit).

---

## 11. FRZ mapping (evidence contribution only — criteria remain **OPEN**)

| FRZ ID | R2 decision contribution |
|--------|---------------------------|
| **FRZ-PRD-08** | Removes strict-path production bypass: in-memory GCA fallback, registry overlay, qualification capability fixture |
| **FRZ-SEC-01** | Process-scoped store bundle + single materialization reduces cross-request authority drift |
| **FRZ-SEC-04** | Fail-closed startup when stores/capabilities unqualified; no silent strict→lab execution |
| **FRZ-REL-03** | Explicit idempotency/durability ownership for GCA host stores aligned with PIS |
| **FRZ-REL-05** | Host execution truth + PIS co-located durability policy; F02 follow-on unchanged |
| **FRZ-CTR-01 / FRZ-CTR-06** | Typed ports and bundles — no `Any` at new boundaries |
| **FRZ-BND-02 / FRZ-BND-04** | No Tier-2 import of applications; resolver hook stays in applications/shared handoff |
| **FRZ-OWN-01 / FRZ-OWN-03** | Single composition owner per store class and agent materialization |
| **FRZ-TYP-01 / FRZ-TYP-03** | Capability mapper + F02 record — frozen Pydantic |
| **FRZ-RPL-02 / FRZ-RPL-03** | Injected durable backends remain replaceable |
| **FRZ-REG-02 / FRZ-REG-03** | Gates G-R2-01…09 — mechanical negatives required in child |

**No global FRZ PASS** is claimed.

---

## 12. Operator decisions required

| # | Decision | If rejected |
|---|----------|-------------|
| OD-R2-01 | Add `governed_external_work_host_store_bundle` (or equivalent) to `ProductionProcessComposition` | **STOP** — no ad-hoc global singleton store owner |
| OD-R2-02 | Extend `ColocatedServingUnitRuntimeAccessor` / `ServingRuntimeHandoff` for host-scoped `RuntimeAgentFactoryResolver` decoration (AC3 only) | **STOP — ARCHITECTURE DECISION REQUIRED** |
| OD-R2-03 | GCA host store `is_durable` gate symmetric to ADR-03 PIS for strict production | Alternative: explicit operator waiver document — **not** assumed |
| OD-R2-04 | Typed settings pin for strict `provider_id` / descriptor digest vs `discover()` | Without pin, only discover mapping allowed — weaker authorization; operator must choose |

---

## 13. NOT PROVEN / unresolved (standard classes)

| Item | Class |
|------|-------|
| Durable physical GCA store implementation qualified for production | **TRACKED FREEZE DEBT** (mirror ADR-03 Case B until backend exists) |
| Integrated strict HTTP → PIS → provider in one request (Package P **B**) | **TRACKED FREEZE DEBT** |
| Restart durability for F02 follow-on record | **TRACKED FREEZE DEBT** (F02 ratified, IMPL not qualified) |
| `FilesystemHostStore` production durability | **NOT PROVEN** — lab/demo only |
| Global FRZ / R7-Q / PROD-Q closure | **IN-SCOPE BLOCKER** (parent gates) |
| Serving handoff resolver extension | **IN-SCOPE BLOCKER** until OD-R2-02 ratified |

---

## 14. Recommended parent disposition

| Artifact | Status |
|----------|--------|
| **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** | **NOT CLOSED** — proceed to child **IMPL-01-R2-STRICT-COMPOSITION-RECOVERY** after operator OD-R2-01…04 |
| **This record** | **READY FOR AUDIT** |
| **R7-Q / PROD-Q** | **NOT QUALIFIED** |

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
