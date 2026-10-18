# PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-OWNERSHIP-RECOVERY-CORRECTIVE-ADR-01

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (corrective architecture decision — documentation only) |
| **Date** | 2026-10-18 |
| **Branch** | `development` |
| **START_HEAD / evidence SHA** | `903f4be08ec9cadd4d3ded133761cd0bd79fe0d3` (F02 decision @ same lineage) |
| **F02 decision** | [`PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md`](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md) |
| **Parent** | **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** — independent audit **REJECTED** |
| **Ratified design preserved** | [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md) **A+B** (orchestrator-backed port + strict fail-closed) — **not reopened** |
| **PIS authority preserved** | [`PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md`](PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md) |
| **Package P / §C** | [`PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md`](PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md) · [`PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md`](PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md) (DISPATCH-Q1, DIM20) |
| **Production delta (this task)** | **0** |
| **Does not close** | **IMPL-01** · **Package P** · **DIM20** · **R7-Q** · **PROD-Q** · any global **FRZ-*** PASS |

**Purpose:** Reconcile ownership, store authority, revision-bound AC3 materialization, and typed boundaries for the rejected IMPL-01 slice **without** a second orchestrator, dispatch gate, or snapshot store.

---

## 1. Audit findings → corrective decisions (F01–F05)

### F01 — Production stores (`strict_external_work_host_wiring.py`)

**Defect @ evidence SHA:** `build_strict_external_work_host_session_resources` mints **new** `InMemoryGovernedExecutionStore`, `InMemoryProofReceiptStore`, `InMemoryPolicyBundleArtifactStore`, and `InMemoryContinuationStateStore` per call (`applications/governed_contractor_application/host/strict_external_work_host_wiring.py` L81–L84), decoupled from process-lifetime composition.

| Responsibility | Canonical contract (symbol) | Contract file | Semantic owner | Sanctioned composition owner | Strict production binding today |
|----------------|----------------------------|---------------|----------------|------------------------------|--------------------------------|
| Governed execution results + host lifecycle states | `GovernedExecutionStore` | `applications/governed_contractor_application/host/stores.py` | **GCA** `GovernedExternalWorkOrchestrator` | `build_governed_external_work_production_runtime` (`production_external_work_composition.py`) — caller supplies instances | **IMPL gap:** in-memory only inside `build_strict_external_work_host_session_resources` |
| Attestation receipts | `ProofReceiptStore` (host protocol) | same | **GCA** orchestrator attestation / `retry_attestation` | same builder | **IMPL gap:** in-memory only |
| Runtime policy bundle artifacts | `PolicyBundleArtifactStore` | same | **GCA** orchestrator + `ImmutableRuntimePolicyBundle` | same builder | **IMPL gap:** in-memory only |
| External-work **governed** continuation (quote/HITL), not Nexus resume | `ContinuationStateStore` (host protocol) | same | **GCA** orchestrator `surface_continuation` | same builder | **IMPL gap:** in-memory only; **not** `ExecutionContinuationStateStore` |
| Provider invocation intent/outcome | `ProviderInvocationStore` | `intergrax/contracts/provider_invocation_store.py` | **Execution / ER** (+ GCA dispatch gate) | `ProductionProcessComposition.provider_invocation_store` → `factory.py` | **Correct:** durable store required (`production_external_work_composition.py` L291–L308) |
| Nexus / harness execution continuation | `ExecutionContinuationStateStore` | `intergrax/contracts/execution_continuation_state_store.py` | **Runtime execution** (ADR-01) | `ProductionProcessComposition.execution_continuation_state_store` → `factory.py` L159–L173 | **Separate concern** from GCA `ContinuationStateStore` |

**Corrective decision (F01):**

1. **Exactly-one composition owner** for the four GCA host store **instances** used by `build_governed_external_work_production_runtime` is the **same owner as today’s orchestrator builder:** `create_governed_contractor_backend_app` / strict session wiring — but instances must be **process-scoped**, injected once per strict external-work stack, and **shared** across requests (session resources already intended via `GovernedExternalWorkHostSessionResources`).
2. **No new persistence framework.** Reuse existing protocols in `stores.py` and existing injection parameters on `build_governed_external_work_production_runtime` (L349–L355).
3. **Durable physical backends** for the four GCA host protocols are **not implemented in-tree** (only `InMemory*` in `stores.py`). Mirror **ADR-03 Case B:** strict production must **fail closed** if a mandatory store cannot satisfy production durability policy once that policy is defined for GCA host stores — **until then**, classify as **TRACKED FREEZE DEBT** aligned with ADR-03 (PIS durable; GCA execution/receipt/bundle/continuation host stores remain qualification doubles).
4. **Forbidden:** constructing fresh in-memory stores inside `build_strict_external_work_host_session_resources` when strict production already binds durable `ProviderInvocationStore` — creates **split-brain** execution truth vs PIS.

**Implicit identity violation (same file):** `principal_id="strict-host-principal"`, `workspace_id="default-workspace"` (L55–L56) are **not** authoritative strict HTTP / serving tenant bindings — remove in recovery IMPL; bind `principal_id` / `tenant_id` from `StrictProductionServingExecutionTenantBinding` + governed execution task context only.

---

### F02 — Prior-state ownership (`sanctioned_external_work_operation_port.py`)

**Defect:** `GovernedSanctionedExternalWorkOperationPort` keeps `create_result_by_execution_id: MutableMapping[str, ExternalWorkAdapterResult]` (L31–L37, L134–L141, L156) — **session shadow state**, not restart-safe, duplicates orchestrator truth.

**Canonical owners (closed world):**

| Concern | Owner | Mechanism @ evidence SHA |
|---------|-------|---------------------------|
| Execution identity (`execution_id`, task/run/attempt, tenant, principal) | `GovernedExternalWorkOrchestrator._require_canonical_execution_identity` | `orchestrator.py` |
| Host lifecycle state (CREATE_IN_PROGRESS, AWAITING_HUMAN, UNKNOWN, completed, …) | `GovernedExecutionStore.put_state` / `get_state` | `orchestrator.py` |
| Durable mutation admission + outcome facts | `ProviderInvocationStore` + `GovernedProviderInvocationDispatchGate` | ADR-03, `provider_invocation_lifecycle.py` |
| Post-create **typed** adapter view required for accept/cancel | **`GovernedExternalWorkOrchestrator`** (orchestrator already requires `create_result.snapshot` — `accept` L419–L420, `cancel` L524–L525) | Must be **loaded** from orchestrator authority — **not** a second cache on the port |

**`GovernedExecutionResultV2`** (`intergrax/contracts/governed_execution_result.py`) stores `provider_invocation`, `provider_outcome`, and `proof` — **not** a full `ExternalWorkSnapshot`. PIS `ProviderInvocationOutcome` stores digests only (`build_provider_invocation_outcome`, `provider_invocation_lifecycle.py` L114–L147).

**Corrective decision (F02):**

1. **Remove** port-level `create_result_by_execution_id` as authoritative truth (including `GovernedExternalWorkHostSessionResources.create_result_by_execution_id`).
2. **Single resolver owner:** `GovernedExternalWorkOrchestrator` exposes `resolve_create_adapter_result_for_follow_on(execution_id) -> ExternalWorkAdapterResult | None` (name illustrative — **one** method) that:
   - reads `get_state` / `get_result` / stored boundary event JSON via existing `GovernedExecutionStore` APIs;
   - **fail closed** with `prior_create_snapshot_missing` when state is incomplete, stale, or `EXECUTION_OUTCOME_UNKNOWN`;
   - does **not** add a new store table or snapshot service.
3. **F02 serialization seam — decided (Option B, pending operator ratification):** [`PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md`](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md) — **Option A not proven** (GER / PIS / EBE v2 lack full `ExternalWorkSnapshot`). **Ratify:** `GovernedExternalWorkCreateFollowOnRecordV1` on existing `GovernedExecutionStore` (`put_create_follow_on_record` / `get_create_follow_on_record`), written at CREATE finalize after `put_result`, read only via orchestrator resolver. **Forbidden:** session dict fallback; EBE or PIS as snapshot authority; provider `get_work` inference.

**Recovery / restart:** Accept/cancel after process restart must succeed only when durable host store + PIS rows reconstruct the snapshot; otherwise **deny** (`prior_create_snapshot_missing`) — recovery is **not** ordinary HTTP retry (GR-7-A7).

---

### F03 — AC3 actual runtime vs qualification harness

**Defect:** Qualification tests that call `build_governed_contractor_agent_builders(host_runtime)` prove **development assembly** only. Strict production uses **revision-bound** registry authority:

- `build_harness_host_runtime` in `factory.py` does **not** pass `compose_builders` or `builders=build_governed_contractor_agent_builders(resolved_host_runtime)` (contrast `lab_application/host/factory.py`, `research_application/host/factory.py`).
- `resolve_harness_host_registry` (`harness_registry_authority.py` L93–L106): **`builders` forbidden** when `RegistryAssemblyMode.REVISION_BOUND`.
- Materialized agents come from `MaterializedRegistryProjection.agent_registry` built at activation via `assemble_production_registry_projection_input_bundle` → manifest factory `build_governed_contractor_external_contractor_adapter_from_context` (`manifest.py` L35) — **no** `sanctioned_operation_port`, **no** `require_sanctioned_external_work_mutations`, **no** `authorization_boundary`.

**Sanctioned port wiring today (host only):** `factory.py` L236–L259 attaches `sanctioned_external_work_operation_port` to `GovernedContractorHostRuntimeComposition` **after** registry projection is fixed — **does not rebind** materialized `ExternalContractorAdapterAgent` instances.

**Corrective decision (F03) — revision-bound AC3 seam:**

| Layer | File | Symbol / seam | Required behavior |
|-------|------|---------------|-------------------|
| Composition | `factory.py` | `create_governed_contractor_backend_app` | When strict external-work stack attaches, **same** `resolved_host_runtime` used for agent materialization must see port + `require_sanctioned_external_work_mutations=True` |
| Agent factory (strict) | `agent_builders.py` | `build_external_contractor_adapter_factory` / `build_governed_contractor_agent_builders` | Sole sanctioned mapping from `GovernedContractorHostRuntimeComposition` → `ExternalContractorAdapterAgent` constructor args |
| Tier-2 entry | `external_contractor_adapter_agent.py` → `domain_job.py` | `require_sanctioned_external_work_mutations` | Fail closed `sanctioned_external_work_operation_port_missing` before `adapt_from_step_metadata` |
| Revision-bound materialization | `production_registry_projection_input_bundle.py` | `factory_resolver` / `RuntimeAgentFactoryResolver` | **Must** resolve `external_contractor_adapter` factory through host runtime ports at **serving-unit** materialization or **deterministic post-rehydration rebinding** owned by GCA host — **not** manifest scaffold |

**Composition owner:** `create_governed_contractor_process_app` (`main.py`) → `create_governed_contractor_backend_app` + `package_contract_authority` (`resolve_strict_package_contract_authority_for_serving_handoff`, `factory.py` L281–L288) for contract binding.

**If** revision-bound artifact factories cannot accept host-injected ports without a new approved materialization hook → **STOP — ARCHITECTURE DECISION REQUIRED** (extend `RuntimeAgentFactoryResolver` / serving rehydration — **no** Tier-2 import of `applications/`).

**Child implementation:** `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01-R1-OWNERSHIP-RECOVERY`

---

### F04 — Semantic typing (`intergrax/contracts/sanctioned_external_work_operation.py`)

**Defect:** `Mapping[str, Any]` on create/accept/cancel `metadata` and `SanctionedExternalWorkOperationResult.domain_summary`.

**Minimal existing contracts to use (no generic metadata envelope):**

| Field today | Replace with (existing) |
|-------------|-------------------------|
| Create payload | `TaskId`, `RunId`, `ExecutionId`, tenant/principal strings — **keep**; `message` — keep; metadata for reflex | Step metadata already parsed in `resolve_sanctioned_operation_from_metadata` (`sanctioned_operation_from_metadata.py`) — constrain to **typed keys** already used by adapter: `META_SANCTIONED_MUTATION_OPERATION`, `META_PRIOR_EXECUTION_ID`, plus `intergrax/contracts/external_work.py` evidence types where applicable |
| Accept | `QuoteAcceptanceEvidence` — **already typed** |
| Cancel | idempotency + execution identity — **already typed** |
| Port result | `SanctionedExternalWorkOperationResult` should carry **`ExternalWorkAdapterResult`-derived structured fields** (or a frozen DTO mirroring `to_domain_summary()` keys) instead of open `Mapping[str, Any]`; correlation fields already on result: `execution_id`, `provider_invocation_id`, `host_state`, `adapter_used`, `provider_mutation_dispatched` |
| Tier-2 boundary type | `ExternalWorkAdapterResult` (`agents/external_contractor_adapter/schemas/adapt_result.py`) composes `ExternalWorkSnapshot`, `GovernedContinuationRequest`, `GovernedProofProfile` |

**Corrective decision (F04):** Recovery IMPL narrows port models to **forbid** open metadata maps at the Tier-2 ↔ host boundary; reflex metadata parsing stays in Tier-2 (`resolve_sanctioned_operation_from_metadata`), host port accepts **only** typed request DTOs. **Do not** introduce a second generic envelope contract.

---

### F05 — Strict configuration (`factory.py`)

**Proven fail-closed (mutating path):** `elif` branch `factory.py` L260–L275 raises `HarnessHostOrchestrationTopologyReliabilityCompositionError` when `strict_topology_reliability` + `external_work_integration` + `runtime_policy_bundle` but missing `provider_invocation_store`, `collaborative_work_repositories`, or `active_execution_task_scope`.

**Gaps to close in recovery IMPL:**

| Condition | Current behavior | Target |
|-----------|------------------|--------|
| Strict + external work + **missing** `settings.runtime_policy_bundle` | Sanctioned attach skipped; `compose_governed_contractor_host_runtime` may still build boundary if bundle present in settings path only | **Fail closed** when integration present and policy bundle absent on strict graph |
| Strict + external work + integration; attach succeeded but **AC3 agent** still scaffold factory | Mutations may reach `adapt_from_step_metadata` (`domain_job.py` L94+) with boundary only | **Fail closed** via `require_sanctioned_external_work_mutations` on materialized agent |
| `external_work_integration is None` | No external-work mutations — legitimate observational / non-mutating host | **Allowed** — not a bypass |
| `require_sanctioned_external_work_mutations=True` + port missing | Tier-2 deny (`domain_job.py` L63–L67) | **Keep** — composition must guarantee port wired before traffic |

---

## 2. Target operation types (three mutations)

| Operation | Request type (host port) | Orchestrator method | Adapter action constant | Result type |
|-----------|-------------------------|---------------------|-------------------------|-------------|
| CREATE | `SanctionedExternalWorkCreateRequest` inside `SanctionedExternalWorkOperationRequest` | `GovernedExternalWorkOrchestrator.create` | `ACTION_CREATE_EXTERNAL_WORK` | `SanctionedExternalWorkOperationResult` ← `OrchestratorStepResult` / `ExternalWorkAdapterResult` |
| ACCEPT_QUOTE | `SanctionedExternalWorkAcceptQuoteRequest` | `GovernedExternalWorkOrchestrator.accept` | accept path in adapter | same |
| CANCEL | `SanctionedExternalWorkCancelRequest` | `GovernedExternalWorkOrchestrator.cancel` | cancel path in adapter | same |

Observational adapter methods (`_OBSERVATIONAL_PROVIDER_METHODS` in `external_work_adapter.py`) remain **outside** `SanctionedExternalWorkOperationPort` — no PIS row required.

---

## 3. Strict fail-closed matrix (composition)

| Missing dependency | Strict production + external work integration | Outcome |
|--------------------|-----------------------------------------------|---------|
| `ProductionProcessComposition.provider_invocation_store` (durable) | yes | `HarnessHostOrchestrationTopologyReliabilityCompositionError` (existing) |
| `collaborative_work_repositories` / `active_execution_task_scope` | yes | same (existing) |
| `settings.runtime_policy_bundle` | yes | **Raise** (recovery IMPL) |
| `sanctioned_external_work_operation_port` on host runtime | yes | Agent deny + composition validation |
| `require_sanctioned_external_work_mutations` not true on materialized AC3 | yes | **Composition reject** at factory (recovery IMPL) |
| Prior create snapshot for accept/cancel | yes | `prior_create_snapshot_missing` — **no** provider mutation |
| Implicit default principal/tenant in strict wiring | yes | **Remove** — use serving binding + task scope only |

---

## 4. Dependency graphs

### 4.1 Before (IMPL-01 @ `0bb8fcc8…` — rejected)

```text
create_governed_contractor_backend_app
  ├─ ProductionProcessComposition.provider_invocation_store (durable) ──► PIS + gate
  ├─ build_strict_external_work_host_session_resources
  │     ├─ NEW InMemory* stores (execution/receipt/bundle/continuation)
  │     └─ GovernedSanctionedExternalWorkOperationPort
  │           └─ session dict create_result_by_execution_id  [SHADOW]
  ├─ attach_sanctioned_external_work_to_host_runtime (port on composition)
  └─ build_harness_host_runtime → REVISION_BOUND registry
        └─ ExternalContractorAdapterAgent from artifact/manifest scaffold
              └─ adapt_from_step_metadata path STILL REACHABLE if require_sanctioned=false
```

### 4.2 After (recovery target)

```text
create_governed_contractor_backend_app
  ├─ Process-scoped GovernedExternalWorkHostSessionResources
  │     ├─ execution/receipt/bundle/continuation stores: INJECTED instances (shared lifetime)
  │     ├─ provider_invocation_store: from ProductionProcessComposition (unchanged)
  │     └─ build_governed_external_work_production_runtime → orchestrator (unchanged gate)
  ├─ GovernedSanctionedExternalWorkOperationPort → orchestrator only
  │     └─ prior create via orchestrator.resolve_create_adapter_result_for_follow_on
  ├─ build_governed_contractor_agent_builders(resolved_host_runtime)
  │     └─ require_sanctioned_external_work_mutations=True + port injected
  └─ REVISION_BOUND materialization / rebind seam
        └─ ExternalContractorAdapterAgent (strict) → domain_job → port.execute only
```

---

## 5. File-level recovery blueprint (single child)

| File | Change |
|------|--------|
| `applications/governed_contractor_application/host/strict_external_work_host_wiring.py` | Accept injected store instances; remove per-call `InMemory*` construction; remove default principal/workspace literals |
| `applications/governed_contractor_application/host/factory.py` | Fail closed on policy bundle gap; wire `compose_builders` or approved revision-bound rebind; pass process-scoped stores |
| `applications/governed_contractor_application/host/sanctioned_external_work_operation_port.py` | Drop session dict; delegate prior-create resolution to orchestrator |
| `applications/governed_contractor_application/host/orchestrator.py` | Add single resolver for follow-on mutations (snapshot rehydration per §F02 decision) |
| `applications/governed_contractor_application/host/agent_builders.py` | No semantic change — remains canonical factory mapping |
| `intergrax/contracts/sanctioned_external_work_operation.py` | Tighten types per §F04 |
| `agents/external_contractor_adapter/steps/domain_job.py` | Unchanged fail-closed paths |
| `applications/governed_contractor_application/manifest.py` / materialization seam | Ensure strict path does not use scaffold factory for serving traffic |

**Forbidden files for workaround:** new snapshot store module; Tier-2 imports from `applications/`; second dispatch gate.

---

## 6. Qualification acceptance gates (independent audit)

| Gate | Evidence required | DEFERRED command (not executed in ADR task) |
|------|-------------------|-----------------------------------------------|
| G-OWN-01 | Static: no `InMemory*` store construction in `build_strict_external_work_host_session_resources` for strict production path | `rg "InMemoryGovernedExecutionStore\\(" applications/governed_contractor_application/host/strict_external_work_host_wiring.py` |
| G-OWN-02 | No `create_result_by_execution_id` dict as authority | `rg "create_result_by_execution_id" applications/governed_contractor_application/host` |
| G-AC3-01 | Revision-bound materialized agent has `require_sanctioned_external_work_mutations=True` and non-None port under strict factory | `uv run pytest applications/governed_contractor_application/tests/host/test_gr7_a2_external_work_erl_bridge.py -q` — **DEFERRED — NOT EXECUTED** |
| G-AC3-02 | H09 / integrated strict HTTP uses production registry path, not manual builders only | `uv run pytest tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py -k h09 -q` — **DEFERRED — NOT EXECUTED** |
| G-F05-01 | Factory raises when strict + integration + missing policy bundle | targeted factory composition test — **DEFERRED — NOT EXECUTED** |
| G-PIS-01 | Mutations still traverse `GovernedProviderInvocationDispatchGate` | existing GR-7 / Package P scoped proofs — no regression |
| G-TYP-01 | Port models: no `Mapping[str, Any]` on sanctioned boundary | pyright/architecture gate on contract module — **DEFERRED — NOT EXECUTED** |

---

## 7. FRZ mapping (@ checklist — **no PASS promoted**)

| FRZ | Touchpoint | Status after corrective ADR | Evidence depth |
|-----|------------|----------------------------|----------------|
| **FRZ-PRD-08** | No production bypass of PIS on strict external-work mutations | **OPEN** — F03/F05 gaps documented; recovery IMPL required | Static graph + planned gates G-AC3-* |
| **FRZ-SEC-01** | Tenant/principal on mutations | **OPEN** — implicit strict-host principal debt (F01) | Code path review @ `0bb8fcc8…` |
| **FRZ-SEC-04** | No shadow authority | **OPEN** — session snapshot dict (F02) | Corrective ADR |
| **FRZ-REL-03** | Idempotency ownership | **OPEN** — PIS scoped; HTTP re-entry not proven | Package P / DISPATCH-Q1 |
| **FRZ-REL-05** | Intent before effect | **PARTIAL** — gate wired when orchestrator path used | ADR-03 |
| **FRZ-CTR-01..06** | Contracts at boundaries | **OPEN** — F04 typing debt | Recovery IMPL |
| **FRZ-BND-02/03** | Tier-2 ↔ Apps | **PASS intent** — port-only injection preserved | ADR-01 reflex |
| **FRZ-OWN-01..05** | Exactly-one owner | **OPEN** — store + snapshot + AC3 duplication | This ADR |
| **FRZ-TYP-01..03** | Strong typing | **OPEN** — `Mapping[str, Any]` on port | F04 |
| **FRZ-RPL-01..03** | Replaceable stores | **OPEN** — in-memory construction breaks replaceability story | F01 |
| **FRZ-REG-02/03** | Anti-regression | **OPEN** — gates G-* not yet green | Child IMPL |

---

## 8. Finding classification

| ID | Class | Note |
|----|-------|------|
| F01–F03 | **IN-SCOPE BLOCKER** for IMPL-01 re-qualification | Ownership / AC3 seam |
| F02 snapshot serialization | **OPTION B PROPOSED — AWAITING OPERATOR RATIFICATION** (F02 decision doc) | No second store |
| GCA durable host store media | **TRACKED FREEZE DEBT** | Parallel to ADR-03 Case B |
| Integrated HTTP→PIS E2E | **TRACKED FREEZE DEBT** | Package P interpretation B |
| pytest / uv in ADR task | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Commands listed §6 only |

---

## 9. Enterprise audit matrix (condensed)

| Dimension | @ IMPL-01 HEAD | Recovery target |
|-----------|----------------|-----------------|
| Ownership | Split: PIS durable vs in-memory GCA stores; shadow snapshot dict | Single orchestrator + injected store instances |
| Composition | Port on host composition disconnected from AC3 registry | Revision-bound factory/rebind aligned with `agent_builders.py` |
| Contracts | Open metadata maps on port | Typed DTOs + `ExternalWorkAdapterResult` alignment |
| Security | Default principal in strict wiring | Serving binding + task scope only |
| Persistence | PIS intent durable; execution truth ephemeral | Shared lifetime stores; snapshot rehydration from orchestrator |
| Fail-closed | Partial (factory elif) | Full matrix §3 |

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.*

## F02 operator ratification alignment — 2026-10-18

Option B is **RATIFIED — SCOPED** by operator, with conditions: production-qualified durability for GCA `GovernedExecutionStore` (not process-local `InMemory*`) and preservation of actual provider outcome when follow-on snapshot finalization fails after an external effect. Incomplete finalization must deny follow-on and must **not** authorize ordinary CREATE replay. See [F02 ratification](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md#11-operator-ratification--2026-10-18). F01–F05 implementation and actual AC3 proof remain **BLOCKED / NOT QUALIFIED**; no global FRZ PASS.
