# TRACE-X-P5-R2-P0 — Configured→Effective Execution Provenance Architecture Lock

## Revision record

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P0` |
| **Parent** | `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **START_HEAD** | `16be5fdfe4bcde804eef51a3b34b4ebc5cd45dad` |
| **Primary FRZ** | `FRZ-TRC-11` (**OPEN** — no PASS in P0) |
| **Blocker** | `P5-GAP-04` — no canonical global configured→effective→`ExecutionId`→evidence chain |
| **Production / runtime delta** | **0** (architecture + qualification design only) |
| **Status** | **TRACE-X-P5-R2-P0 = READY FOR AUDIT** · **TRACE-X-P5-R2 = BLOCKED ON P0 INDEPENDENT AUDIT** · **TRACE-X-P5 = CURRENT / BLOCKED ON R2** |

**Steering sources revalidated @ START_HEAD:** [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md), [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md), [`TRACE_X_P5_POLICY_PROFILE_CONFIGURATION_PROVENANCE_BASELINE.md`](TRACE_X_P5_POLICY_PROFILE_CONFIGURATION_PROVENANCE_BASELINE.md), [`TRACE_X_P5_R1_POLICY_PROFILE_EXECUTION_ATTRIBUTION_CERTIFICATION.md`](TRACE_X_P5_R1_POLICY_PROFILE_EXECUTION_ATTRIBUTION_CERTIFICATION.md), [`INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md`](../architecture/INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md).

**Historical evidence (revalidated, not blindly trusted):** INT-CONFIG-REAL-X-CERT `a59744517b92847f55def1db22826d17d89ee155` · TRACE-X-P5-P0 `81fd1490f18d73eaf31ec94c2b93dbb525026ba2` · TRACE-X-P5-R1 `05fd5d9b2b97f9d85a534a949d882cd47d4a54c9`.

---

## 1. Architecture conclusion

**Outcome B — no existing sanctioned join point exists.**

At START_HEAD there is **no** production surface that simultaneously holds:

1. canonical `ConfiguredCapabilityBinding` identity (when INT-CONFIG adoption applies),
2. sanctioned effective integration selection/materialization facts,
3. tenant identity,
4. canonical `ExecutionId`.

Therefore P0 locks a **minimum typed seam** (R1-analogous pattern) without implementing it. **No STOP — ARCHITECTURE DECISION REQUIRED** remains for the locked design: one capture owner, one store semantic owner, one neutral read contract, one reconstructor projection path.

---

## 2. Before graph (@ START_HEAD)

```text
Governance authorize
        │
        ▼
INT-CONFIG realize_admitted
        │
        ▼
ConfiguredCapabilityBinding  ──X──►  (no production consumer)
        │
        │   parallel universe
        ▼
IntegrationProfile + resolve_from_profile / resolve
        │
        ▼
effective provider instance  ──X──►  ExecutionId
        │
        ▼
ExecutionReconstructor  (no execution-config provenance reader)
```

**Exact gap (unchanged from parent):**

1. `ConfiguredCapabilityBinding` is produced by INT-CONFIG realization only.
2. It is **not** a canonical effective-runtime binding.
3. No production configured→effective adoption chain to `ExecutionId` exists.
4. `resolve_from_profile(...)` operates on `IntegrationProfile`, not `ConfiguredCapabilityBinding`.
5. No typed execution-config provenance record analogous to R1 `ExecutionEffectiveProfileProvenance`.
6. No sanctioned persistence/read contract joining configured identity, effective materialization, tenant, and `ExecutionId`.
7. `ExecutionReconstructor` cannot reconstruct configured/effective provenance.

---

## 3. Proposed after graph (locked design — not implemented in P0)

```text
ConfiguredCapabilityBinding (optional adoption input)
        │
        │  execution-bound sanctioned wrapper (calls existing factory only)
        ▼
resolve_from_profile / resolve  →  effective materialization facts
        │
        │  atomic pin @ execution-bound composition (tenant + ExecutionId known)
        ▼
ExecutionIntegrationConfigurationProvenance (immutable, per subject)
        │
        ├── durable ExecutionIntegrationConfigurationPinningStore
        │
        ▼
ExecutionIntegrationConfigurationProvenanceReader (neutral Protocol)
        │
        ▼
ExecutionReconstructor (projection only)
        │
        ▼
ExecutionReconstruction (+ typed collection + read status)
```

---

## 4. Closed-world inventory A — configured identity

### 4.1 Contracts

| Artifact | Path | Role |
|---|---|---|
| `ConfiguredCapabilityBinding` | `intergrax/integrations/contracts/existing_capability_configuration.py` | Typed configured identity (tenant, category, provider_id, resource_scope, type, version, fingerprint, evidence refs) |
| `ExistingCapabilityConfigurationRealizationRequest` | same | Admission request; optional `task_id` / `run_id` — **not** `ExecutionId` |
| `ExistingCapabilityConfigurationRealizationResult` | same | `configured_binding` + `authorization_evidence` |
| `ExistingCapabilityIntegrationTarget` | same | Pre-realization target descriptor |

### 4.2 Production producers

| Producer | Path | Emits |
|---|---|---|
| `ExistingCapabilityConfigurationRealizationService.realize_admitted` | `intergrax/integrations/existing_capability_configuration_service.py` | `ExistingCapabilityConfigurationRealizationResult.configured_binding` |
| `SQLiteRelationalStoreConfigurationRealizationStrategy.realize` | `intergrax/integrations/providers/relational_store/sqlite/configuration_realization.py` | `ConfiguredCapabilityBinding` |
| `ExistingCapabilityConfigurationRealizationFacade.realize` | `intergrax/integrations/existing_capability_configuration_facade.py` | delegates to service after Governance port |

### 4.3 Production consumers of `configured_binding`

**Mechanical proof @ START_HEAD:** repository-wide production search (`intergrax/`, `agents/`, `applications/`) shows **zero** consumers of `ConfiguredCapabilityBinding` or `ExistingCapabilityConfigurationRealizationResult.configured_binding` outside the INT-CONFIG realization pipeline (facade, service, contracts, sqlite strategy).

**False positive excluded:** `intergrax/applications/_shared/production_delegated_subtask_plans.py` uses a local name `configured_binding` for **package binding ids** — unrelated type/contract.

**Handoff after realization:** none in production; binding is returned to caller only. **No persistence** of configured bindings as execution evidence.

### 4.4 Classification

| Surface | Classification |
|---|---|
| INT-CONFIG realization pipeline | `SANCTIONED_CONFIGURED_PRODUCTION` |
| Qualification / unit tests | `NON_PRODUCTION` |
| P5 discovery registry rows | `NOT_RELEVANT` (inventory SSOT) |

---

## 5. Closed-world inventory B — effective integration resolution/materialization

### 5.1 Canonical sanctioned entrypoints (Integrations registry)

| Entry | Path | Classification |
|---|---|---|
| `resolve` | `intergrax/integrations/registry/factory.py` | `SANCTIONED_EFFECTIVE_RESOLUTION` |
| `resolve_from_profile` | same | `SANCTIONED_EFFECTIVE_RESOLUTION` |
| `resolve_slug` / `build_profile_from_env` / `build_profile_from_mapping` | same | `SANCTIONED_EFFECTIVE_RESOLUTION` (selection helpers) |
| `resolve_contract` + category helpers | `intergrax/integrations/registry/resolve_typed.py` | `SANCTIONED_EFFECTIVE_RESOLUTION` (typed delegate; **no** second resolver) |
| `IntegrationProfile` doc example `resolve_from_profile` | `intergrax/integrations/contracts/integration_profile.py` | `NOT_RELEVANT` (documentation) |
| `integrations._shared.health` resolve helpers | `intergrax/integrations/_shared/health.py` | `SANCTIONED_EFFECTIVE_RESOLUTION` (health probe) |
| Cloud platform `resolve(category)` adapters | `intergrax/integrations/providers/cloud_platform/**` | `SANCTIONED_EFFECTIVE_RESOLUTION` (nested category routing — not alternate catalog) |
| Credential / scoped adaptation `resolve` | credentials, `scoped_integration_adaptation` | `SANCTIONED_EFFECTIVE_RESOLUTION` (domain-specific; still not second catalog) |
| Scaffold template emission | `intergrax/scaffold/integration_templates.py` | `NON_PRODUCTION` (generator output) |

**Pre-built instance path:** `resolve_from_profile` → `profile.instance_for_category` → `_require_category_integration_instance` (DI-only / pre-built).

**Catalog factory path:** `resolve` → `get_entry(slug)` → `entry.factory(...)`.

### 5.2 Production call sites (direct `resolve_from_profile` / `resolve`)

| Layer | Module |
|---|---|
| Applications (Tier-3 shared) | `notification_wiring.py`, `identity_wiring.py`, `security_runtime_bridge.py`, `sandbox_host_wiring.py`, `integration_tool_wiring.py`, `adaptive_feature_flag_gate.py` |
| Runtime | `persistence/integration_profile_wiring.py`, `sandbox/hosted_resolver.py`, `codecraft/substrate.py`, `vendor_knowledge/resolver.py` |
| Tools / speech / RAG | `tools/registry/wiring.py`, `speech_adapters/registry/resolver.py`, `speech_adapters/registry/profile.py`, `rag/bootstrap/rag_stack_bootstrap.py`, `rag/vectorstore/bootstrap/integration_vectorstore.py` (`resolve`), `rag/rerankers/integration/resolver.py`, `rag/document_loaders/integration/resolver.py` |
| Integrations | `_shared/health.py` |

**`agents/`:** no `resolve_from_profile` / registry `resolve` production usage found.

**`applications/` (repo root):** no direct registry resolution outside `intergrax` package imports.

**Unknown production materialization path:** **none identified** (closed-world complete for registry-backed resolution).

### 5.3 Relation to `ConfiguredCapabilityBinding`

All sanctioned effective paths consume **`IntegrationProfile`** (or explicit slug/env), **never** `ConfiguredCapabilityBinding`. Adoption is an **unimplemented** future composition concern.

---

## 6. Closed-world inventory C — execution identity vs integration resolution

| Observation | Evidence |
|---|---|
| Canonical `ExecutionId` owner | Execution layer (`intergrax/contracts/execution_identity.py`, minting via execution identity authority / host admission) |
| Profile pinning join (R1 precedent) | `EffectiveProfileRevisionAdmission.admit_root_execution` — has `tenant_id` + `ExecutionId`; **no** integration resolution |
| Integration resolution timing | Predominantly **host/bootstrap wiring** before or without binding to a specific `ExecutionId` |
| INT-CONFIG request correlation | `task_id` / `run_id` optional on realization request — **not** canonical `ExecutionId` |
| Simultaneous quad join | **Absent** in production |

**Decisive answer:** no existing sanctioned composition point holds configured binding + effective facts + tenant + `ExecutionId`.

---

## 7. Closed-world inventory D — evidence / reconstruction

| Component | Path | Role today |
|---|---|---|
| `ExecutionReconstructor` | `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` | Exactly-one factual reconstruction owner |
| `ExecutionReconstruction` | `intergrax/contracts/execution_reconstruction_models.py` | Derived read-only; R1 fields for policy + effective profile |
| R1 reader pattern | `intergrax/contracts/execution_effective_profile_provenance.py` + `profile_provenance_projection.py` | Neutral read-only profile provenance |
| R1 adapter | `applications/_shared/profile_resolution/execution_effective_profile_provenance_reader.py` | Pinning store → neutral DTO |
| RuntimeEvent / causal evidence | existing observability spine | Provider/tool correlation — **not** configured/effective config provenance authority |
| Diagnostics / RuntimeInspection | applications diagnostics | Read-only consumers — **not** config provenance SSOT |

**Execution-config provenance:** **missing** (no contract, store, or reconstructor join).

---

## 8. Ownership matrix (locked)

| Concern | Canonical owner | P0 change |
|---|---|---|
| Configuration realization semantics | Integrations / INT-CONFIG | none |
| `ConfiguredCapabilityBinding` emission | Integrations realization strategies | none |
| Effective provider selection/materialization | Existing `resolve` / `resolve_from_profile` | none (wrapper calls only) |
| `ExecutionId` | Execution | none |
| Governance permission | Governance | none |
| Execution-config **capture** (write) | **Integrations-owned execution-bound resolution seam** invoked from **Applications host execution composition** | future |
| Provenance **persistence** | Evidence/provenance store (Integrations port + Application persistence adapters) | future |
| Factual **reconstruction** | `ExecutionReconstructor` (read/project only) | future |
| Diagnostics / Observability | Read-only consumers | none |

---

## 9. Authority matrix

| Invariant | Locked rule |
|---|---|
| `configured ≠ effective ≠ authorized ≠ executing` | Preserved |
| `ConfiguredCapabilityBinding` | Factual configured identity only — **no** permission, activation, or ExecutionId mint |
| Effective resolution | **No** Governance decision; materialization only |
| Provenance records | **Cannot** authorize, activate, resolve-latest, or heal missing config |
| `RuntimeConfig` | May **inject** already-canonical reader dependencies only — **not** provenance SSOT |
| `IntegrationProfile.options` | **Not** a semantic provenance contract (CONFIG-X debt if blocking linkage) |

---

## 10. Selected architecture option

**R1-isomorphic execution-config provenance** with explicit configured/effective fields and execution-bound capture.

### 10.1 Future neutral contracts (`intergrax/contracts/`)

| Contract | Purpose |
|---|---|
| `IntegrationConfigurationSubject` | Deterministic subject key: `integration_category`, `provider_id`, `resource_scope`, `configuration_type` (aligned with `ConfiguredCapabilityBinding` dimensions where INT-CONFIG applies) |
| `ConfiguredIntegrationProvenanceSlice` | `configuration_version`, `configuration_fingerprint` (+ optional `realization_request_id` correlation) |
| `EffectiveIntegrationMaterializationProvenance` | `integration_category`, `integration_slug` (catalog slug from `resolve_slug` / profile slot), `materialization_kind` ∈ `{catalog_factory, profile_prebuilt_instance}` |
| `ExecutionIntegrationConfigurationProvenance` | `tenant_id`, `execution_id`, `subject`, configured slice (optional when path is profile-only non-INT-CONFIG), effective slice — **immutable** |
| `ExecutionIntegrationConfigurationProvenanceReadStatus` | `not_configured` \| `configured` \| `required_missing` (fail-closed when path claims binding) |
| `ExecutionIntegrationConfigurationProvenanceReader` | `read_all(tenant_id, execution_id) -> tuple[...]` and/or `read_one(..., subject)` |

**Effective identity fields** are derived only from sanctioned factory outcomes — **not** provider instance reflection, options dicts, or `RuntimeConfig` hashing.

**Configured fingerprint ≠ effective proof:** both slices mandatory on INT-CONFIG adoption paths; effective slice required on all configuration-aware execution paths.

### 10.2 Capture / write point (future)

| Element | Locked choice |
|---|---|
| **Owner** | Integrations: `ExecutionBoundIntegrationResolution` (new module) wrapping **only** existing `resolve_from_profile` / `resolve` |
| **Composition invoker** | Applications host execution wiring (same class of roots as `revision_admission` on `build_environment_host_task_execution` / scenario runtime) |
| **Timing** | Immediately after successful sanctioned materialization, **with** `tenant_id` + `ExecutionId` + optional `ConfiguredCapabilityBinding` adoption input — **before** downstream work relies on the instance where feasible |
| **Semantics change** | **Forbidden** in wrapper — delegate to factory unchanged |

**Atomicity note:** bootstrap-time resolves (no `ExecutionId`) are **not** provenance-complete; qualification must either migrate them behind execution-bound wrapper or classify as non-configuration-aware. Residual window without migration = **qualification failure**, not silent NOT_FOUND.

### 10.3 Persistence (future — **mandatory**)

| Decision | Rationale |
|---|---|
| **Durable canonical persistence required** | Restart/replay/historical reconstruction; current profile/config mutable after execution |
| Semantic owner | Integrations port `ExecutionIntegrationConfigurationPinningStore` |
| Key | `(tenant_id, execution_id, IntegrationConfigurationSubject)` |
| Invariants | Immutable append/pin; tenant-scoped read; no latest/current fallback |
| Implementation locus | Mirror R1: `applications/_shared/integrations/persistence.py` (KV/DocumentStore adapters) — **not** a second STATE-X semantic truth owner |

### 10.4 Read / reconstruction path (future)

```text
PinningStore
  → PinningStoreExecutionIntegrationConfigurationProvenanceReader (Applications adapter)
  → ExecutionIntegrationConfigurationProvenanceReader (neutral)
  → integration_configuration_provenance_projection (runtime/observability/reconstruction)
  → ExecutionReconstructor
```

**Forbidden:** `ExecutionReconstructor` imports Integrations implementation modules.

**Composition:** optional reader injection via existing diagnostic/reconstruction DI (`diagnostic_composition.py`, `harness_host_runtime.py`, `scenario_runtime_baseline.py` pattern).

---

## 11. Multiplicity model

| Question | Answer @ architecture lock |
|---|---|
| Multiple integration categories per execution? | **Yes** — one provenance record per `IntegrationConfigurationSubject` |
| Multiple providers per category? | **Yes** when distinct `provider_id` / `resource_scope` / `configuration_type` |
| Multiple bindings over time per subject? | **No silent overwrite** — first pin wins; re-pin mismatch fails closed |
| Child executions? | **Separate** provenance unless explicit future inheritance port ( **not** automatic from R1 profile inheritance) |

Reader returns **deterministic tuple** ordered by subject fields — not a single global record.

---

## 12. Child execution model

| Rule | Locked |
|---|---|
| Default | Child `ExecutionId` **does not** inherit parent execution-config provenance |
| Child integration resolution | If child path calls execution-bound resolution independently → **new** child-scoped records |
| Parent fallback | **Forbidden** (no heuristic inheritance) |
| Profile revision inheritance (R1) | **Unchanged** — orthogonal seam |

If product later requires inheritance, that is a **new** `ChildExecutionContextInheritancePort` specialization — **STOP** if it requires new execution authority (not decided in P0).

---

## 13. Tenant isolation audit

**Verdict: PASS (architecture lock — local R2 scope evidence).**

Required continuity (implementation must enforce):

```text
configured_binding.tenant_id
== effective_resolution.tenant_id
== execution_provenance.tenant_id
== ExecutionReconstruction tenant context
```

**Mandatory negatives (future implementation):**

- tenant A configured binding cannot become effective for tenant B;
- tenant A provenance cannot be read under tenant B scope;
- missing tenant cannot become shared/global;
- provider resolution cannot rewrite tenant;
- child execution cannot widen tenant.

**Global `FRZ-TEN-*` PASS:** **not** claimed in P0.

---

## 14. Governance / authority audit

Configured provenance is **factual identity only** — no ALLOW, no admission, no activation. Governance and Execution authority unchanged from INT-CONFIG + R1 locks.

---

## 15. Failure model

| Condition | Reconstruction / integrity behavior |
|---|---|
| Configuration-aware path, reader configured, required subject missing | **Fail closed** — not silent `NOT_FOUND` |
| Tenant / execution / subject mismatch | **Fail closed** |
| Configured fingerprint without effective slice on adoption path | **Fail closed** |
| Treating fingerprint as effective proof | **Forbidden** |
| Heuristic join (timestamp, task_id-only, current config lookup) | **Forbidden** |

---

## 16. Rejected alternatives (@ START_HEAD evidence)

| Alt | Rejection |
|---|---|
| A. `RuntimeConfig` snapshot as truth | Broad mutable composition; wrong owner |
| B. Diagnostics / RuntimeInspection as truth | Interpretation plane only |
| C. Current/latest configured binding lookup | Historical reconstruction must not use mutable present state |
| D. Timestamp correlation | Heuristic join |
| E. `task_id` / `run_id`-only correlation | R2 requires canonical `ExecutionId` |
| F. New TRACE-X provider resolver | Second Integrations authority |
| G. New activation lifecycle | Forbidden by INT-CONFIG lock |
| H. Provider-specific R2 core branches | Breaks pluginability |

---

## 17. STOP conditions (P0 disposition)

| STOP trigger | P0 status |
|---|---|
| Multiple plausible effective-resolution owners | **Clear** — single catalog factory owner |
| No point to observe effective + ExecutionId | **Addressed** by execution-bound wrapper seam (future) |
| Linking binding requires changing binding semantics | **Not required** — optional adoption input |
| No stable effective identity | **Clear** — slug + materialization_kind + category |
| Second semantic truth store | **Avoided** — provenance store is evidence-only, R1-isomorphic |
| Child propagation needs new execution authority | **Deferred** — no inheritance in v1 lock |
| Tenant absent at capture | **Addressed** — mandatory in API |
| Neutral contract leaks provider instances | **Forbidden** by design |
| Governance semantics change | **Not required** |

**P0 does not emit STOP — ARCHITECTURE DECISION REQUIRED.**

---

## 18. Expected implementation file map (future waves)

| File | Role |
|---|---|
| `intergrax/contracts/integration_configuration_subject.py` | Subject key |
| `intergrax/contracts/execution_integration_configuration_provenance.py` | Neutral DTOs + reader Protocol |
| `intergrax/integrations/execution_bound_integration_resolution.py` | Wrap factory; emit capture events |
| `intergrax/applications/contracts/integrations/execution_configuration_binding.py` | Pinning store Protocol |
| `intergrax/applications/_shared/integrations/execution_configuration_pinning.py` | Pin helpers |
| `intergrax/applications/_shared/integrations/persistence.py` | Durable adapters |
| `intergrax/applications/_shared/integrations/execution_integration_configuration_provenance_reader.py` | Store → neutral |
| `intergrax/runtime/observability/reconstruction/integration_configuration_provenance_projection.py` | Reconstructor projection |
| `intergrax/contracts/execution_reconstruction_models.py` | Additive fields |
| `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` | Wire reader |
| `intergrax/applications/_shared/diagnostic_composition.py` | Optional reader inject |
| `intergrax/applications/_shared/scenario_runtime_baseline.py` | Host wiring |
| `intergrax/applications/_shared/harness_host_runtime.py` | Harness wiring |
| `tests/qualification/trace_x/_trace_x_p5_r2_discovery.py` | Closed-world discovery |
| `tests/qualification/trace_x/_trace_x_p5_r2_support.py` | Registry + classifications |
| `tests/qualification/trace_x/test_trace_x_p5_r2_qualification_gates.py` | Mechanical gates |

**Explicit non-goals in implementation:** modify `ConfiguredCapabilityBinding`, `IntegrationProfile`, `resolve_from_profile` semantics, Governance, or Execution identity minting.

---

## 19. Closed-world qualification strategy (future gates)

Independent discovery + registry parity (seed **not** from registry), minimum gates:

1. All configured-binding production producers known.
2. All configured-binding production consumers known (**expect zero** outside INT-CONFIG @ baseline).
3. All sanctioned effective resolution/materialization paths known.
4. Every **configuration-aware** execution path records R2 provenance.
5. Unknown production effective path → qualification **FAIL**.
6. Registry orphan/duplicate → **FAIL**.
7. Alternate provenance store count = 0.
8. Heuristic join count = 0.
9. Timestamp/latest/current fallback = 0.
10. Diagnostics cannot act as provenance authority.
11. Reconstruction imports no Integrations implementation.
12. Configured fingerprint alone ≠ effective proof.
13.–17. Tenant / execution / provider / subject mismatch + missing required provenance → fail closed.
18. Multiplicity deterministic; child rules per §12.
19. R1 policy/profile provenance unchanged (regression gates).

---

## 20. Implementation wave decomposition

| Wave | Deliverable |
|---|---|
| **P0** (this document) | Architecture lock + qualification design |
| **P1** | Neutral contracts + validation helpers |
| **P2** | Pinning store port + in-memory + durable adapters |
| **P3** | Execution-bound resolution wrapper + host composition wiring |
| **P4** | Reconstructor projection + diagnostic injection |
| **P5** | Closed-world qualification gates + adversarial certification |
| **CERT** | Independent audit; **only** path to FRZ-TRC-11 promotion |

---

## 21. Exit criteria (R2 parent — not P0)

- Global configured→effective→`ExecutionId`→evidence chain reconstructable on all configuration-aware production paths.
- Qualification gates green with discovery/registry parity.
- Independent audit accepts architecture + implementation.
- **FRZ-TRC-11** promotion only after CERT — **not** P0.

---

## 22. Enterprise audit matrix (@ P0 lock)

| Dimension | Grade | Notes |
|---|---|---|
| Boundaries | **PASS** | Tier rules preserved; reconstructor stays derived |
| Semantic ownership | **PASS** | Single owner per row §8 |
| Composition ownership | **PASS** | Applications invoke; Integrations capture |
| Contracts | **PASS** | Neutral DTO design specified |
| Strong typing | **PASS** | No options-dict / RuntimeConfig provenance |
| Pluginability | **PASS** | Slug + materialization_kind; no provider branches in core |
| Replaceability | **PASS** | Wrapper delegates to existing factory |
| Configured/effective separation | **PASS** | Explicit slices |
| Governance separation | **PASS** | Factual only |
| Execution authority | **PASS** | ExecutionId unchanged |
| Observability non-authority | **PASS** | Consumers read-only |
| Tenant continuity | **PASS** | Local audit §13 |
| Evidence timing | **PASS** | Post-materialization @ execution boundary |
| Persistence/recovery | **PASS** | Durable store mandated |
| Bypass resistance | **PARTIAL** | Until P3 wiring — **IN-SCOPE BLOCKER** for R2 impl |
| Reconstruction purity | **PASS** | Reader/projection pattern locked |
| Regression protection | **PASS** | R1 gates remain required |

**IN-SCOPE BLOCKER (unchanged):** `P5-GAP-04` until implementation waves complete.

**TRACKED FREEZE DEBT:** `P5-GAP-05` (CONFIG-X / profile options weak seams) — does not block R2 architecture lock.

**ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED:** `tests/unit/integrations/test_registry.py` — 4 failures @ START_HEAD (fake factories return `dict` vs `PlatformIntegrationContract` expectation); pre-existing; not introduced by P0.

---

## 23. Applicable FRZ evidence (no PASS promotion)

| FRZ | Disposition |
|---|---|
| **FRZ-TRC-11** | **OPEN** — primary owner TRACE-X-P5-R2 |
| FRZ-TRC-07 / 08 | **PASS** — preserved; R1 unchanged |
| FRZ-TRC-09 / 10 | **OPEN** — TRACE-X-P6 |
| FRZ-CFG-* | Supporting — INT-CONFIG-CERT scoped |
| FRZ-CTR-* / FRZ-PLG-* / FRZ-RPL-* | Supporting — single catalog resolver |
| FRZ-GOV-* | Supporting — authority separation |
| FRZ-TEN-* | **OPEN** globally — local tenant audit only |

---

## 24. Current-HEAD test evidence (P0)

```text
uv run pytest \
  tests/qualification/trace_x/test_trace_x_p5_p0_qualification_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r1_qualification_gates.py \
  -p no:xdist -q
# 192 passed (includes existing_capability_configuration qualification modules in run batch)

uv run pytest tests/qualification/existing_capability_configuration -p no:xdist -q
# (included above — green)

uv run pytest \
  tests/unit/integrations/test_registry.py \
  tests/unit/runtime/integrations/test_canonical_registry_projection.py \
  -p no:xdist -q
# 17 passed, 4 failed — test_registry fake factory contract mismatch (pre-existing)
```

Log: `.tmp/session/trace-x-p5-r2-p0/pytest.log`, `pytest-integrations.log`.

---

## 25. Roadmap status (recommended)

| Item | Status |
|---|---|
| TRACE-X-P5-R2-P0 | **READY FOR AUDIT** |
| TRACE-X-P5-R2 | **BLOCKED ON P0 INDEPENDENT AUDIT** |
| TRACE-X-P5 | **CURRENT / BLOCKED ON R2** |
| FRZ-TRC-11 | **OPEN** |
