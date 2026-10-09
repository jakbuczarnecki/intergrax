# TRACE-X-P5-R2-P4-R1 — Reconstruction Requirement Authority & As-Of Semantics Reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** |
| **Rejected P4 baseline** | `055ed448cb890026c8c34e336e7baf7422daa2f6` (lineage only — do not extend) |
| **Parent** | **TRACE-X-P5-R2-P4** = **BLOCKED ON R1** |
| **Production delta @ R1 lock** | **0** |
| **FRZ-TRC-11** | **OPEN** |
| **P5 / CERT** | **NOT ENTERED** |
| **Architecture lock (parent)** | [`TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md`](TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md) |

## Audit blockers addressed

| Blocker | ID | R1 disposition |
|---|---|---|
| Magic payload requirement authority | `R2-P4-PROVENANCE-REQUIREMENT-AUTHORITY-24` | **RESOLVED IN DESIGN** — reject sole authority; lock typed spine evidence + production emitter |
| As-of pin future leak | `R2-P4-AS-OF-CONFIG-PROVENANCE-FUTURE-LEAK-25` | **RESOLVED IN DESIGN** — Option B; no timestamp heuristics |

---

## 1. Requirement truth inventory (closed-world)

Mechanical audit @ rejected P4 baseline (`055ed448…`) and parent locks (P0–P3, P2 persistence). Classifications: **canonical truth** · **factual evidence** · **derived projection** · **insufficient** · **forbidden current-state lookup**.

| Candidate | Locus | Can answer `ExecutionId → provenance required?` | Classification | Notes |
|---|---|---|---|---|
| `RuntimeEvent.payload["execution_integration_configuration_provenance_required"]` | `integration_configuration_provenance_projection.py` | Only if present in positioned history | **derived / test-only @ baseline** | **Zero production emitters** (grep: projection + unit test + P4 qual doc only). **Must not remain semantic authority.** |
| `ExecutionIntegrationConfigurationAdoption` on dispatch/intake | `ExecutionBoundCapabilityExecutionDispatchRequest`, `QualifiedCapabilityExecutionIntakePayload`, `WorkerConfiguredCapabilityExecutionRequest` (mandatory) | At handoff time only | **canonical truth (runtime)** | Typed, execution-scoped, tenant-scoped; **not** persisted into reconstruction inputs today. |
| `ExecutionBoundIntegrationResolution.materialize_validate_and_pin` | `execution_bound_integration_resolution.py` | Implicit when `adoption` present | **factual evidence (write path)** | Emits `CONFIGURED_ADOPTED` provenance to P2 store; **does not** emit requirement spine fact. |
| P2 `ExecutionIntegrationConfigurationPinningStore` pin (`mode`) | `execution_integration_configuration_pinning.py`, `persistence.py` | **No** when pin absent | **canonical truth (provenance content)** | Proves configured/effective **when pin exists**; cannot distinguish “required but missing” vs “not applicable”. Not temporal / as-of. |
| `ConfiguredCapabilityExecutionSubject` | P1 contracts | Pre-execution only | **factual evidence** | Not reconstruction-durable. |
| `ConfiguredMarketplaceToolExecutionProvenance` + durable intent | `marketplace_tool_execution_intent.py`, intent repository | Correlates via `execution_request_id` / handler provenance | **factual evidence** | Marketplace-scoped; **insufficient** as global `ExecutionId` requirement authority without forbidden heuristic joins. |
| `AdmittedRootGovernanceIdentity` / root admission | execution intake payloads | No integration-config dimension | **insufficient** | Governance ≠ integration configuration provenance requirement. |
| Caller registry `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` | Architecture lock §1A.7 | Static classification | **forbidden current-state lookup** | Not execution-instance durable; cannot drive reconstruction. |
| `ExecutionIntegrationConfigurationProvenanceReader.read_all` | P2 reader adapter | Returns pins only | **derived projection** | Current-state store read — **forbidden as as-of historical truth** (blocker 25). |
| Provider/category/profile resolution | Integrations factory | N/A | **forbidden** | No inference from strings or latest config. |

**Closed-world verdict:** No existing **durable, typed, execution-scoped** fact in production today satisfies blocker 24. The nearest canonical fact is **`ExecutionIntegrationConfigurationAdoption`** on the configured execution handoff; it must be **mirrored into canonical positioned runtime evidence** (preferred) without a second semantic state store.

---

## 2. Selected canonical authority (locked)

### 2.1 Requirement: `ExecutionId → integration configuration provenance required`

**Single authority (reconstruction read path):** typed **positioned runtime spine evidence** — a new frozen contract in `intergrax/contracts/` (name locked at implementation: e.g. `ExecutionIntegrationConfigurationProvenanceRequirementEvidence`) carried by a new **`RuntimeEventType`** spine value (e.g. `INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED`), with schema-versioned payload validation (same discipline as other spine facts).

| Property | Satisfied by |
|---|---|
| Typed | Dataclass / validated payload — not `dict` magic keys |
| Execution-scoped | `execution_id` + `tenant_id` on evidence |
| Tenant-scoped | Validated on emit and on reconstruction parse |
| Durable for restart | Persisted via existing **execution runtime evidence** / positioned event store (same durability chain as reconstruction inputs) |
| No current config lookup | Evidence emitted at execution boundary commit |
| No provider lookup | Adoption already resolved at emit point |
| No magic string payload key | `execution_integration_configuration_provenance_required` **authority = 0** |

**Single production emitter (locked):** `ExecutionBoundIntegrationResolution.materialize_validate_and_pin` — emit **once per subject obligation** immediately **after** successful `pinning_store.pin(...)` (same failure domain: failed pin ⇒ no event). Adoption must be non-null (`CONFIGURED_ADOPTED` path). Optional mirror on shared ingress is **forbidden** as a second emitter.

**Reconstruction consumer:** `integration_configuration_provenance_projection` discovers required `ExecutionId` set **only** from typed spine events in **positioned history** (including as-of-truncated history). Magic payload key may remain **only** as deprecated test shim with **zero semantic authority**.

### 2.2 Provenance content: configured / effective pins

**Unchanged owner:** P2 `ExecutionIntegrationConfigurationPinningStore` + neutral `ExecutionIntegrationConfigurationProvenanceReader` (architecture lock §10.3–10.4). `ExecutionReconstructor` remains the sole reconstructor.

### 2.3 Non-configured executions

| Class | Requirement evidence | Pin expectation | Reconstruction |
|---|---|---|---|
| Configured-required (`CONFIGURE_EXISTING` → adoption → pin path) | Spine requirement event present | `CONFIGURED_ADOPTED` pin(s) per subject | Fail closed if event ∧ scoped ID ∧ no valid pin |
| Legitimate effective-only | **Absent** | Optional `EFFECTIVE_ONLY` pin or none | **No** global pin mandate; status not `REQUIRED_MISSING` |
| Not applicable (no integration-config awareness) | **Absent** | None | Empty provenance tuple; status `NOT_CONFIGURED` |

No classification from `provider_id`, category strings, or marketplace binding alone.

---

## 3. Before / after graph

### Before (rejected P4 — blockers 24–25)

```text
Positioned runtime events
  → discover ExecutionIds
  → optional dict flag execution_integration_configuration_provenance_required (no production emitter)
  → reader.read_all(tenant, ExecutionId)  # current P2 store — leaks future pins under execution_as_of
  → empty tuple silently when not “required”
```

### After (post–corrected P4 implementation)

```text
CONFIGURE_EXISTING → adoption → ExecutionId
  → ExecutionBoundIntegrationResolution.pin (P2) + spine requirement event (typed)
  → [full reconstruction] reader.read_all → provenance tuple
  → [execution_as_of set] skip store pin projection → typed UNAVAILABLE status (Option B)

Positioned history (possibly truncated @ execution_as_of)
  → discover ExecutionIds
  → discover required ExecutionIds from typed spine requirement events only
  → required ∧ missing pin → ExecutionReconstructionIntegrityError (fail closed)
```

---

## 4. As-of semantics (locked: Option B)

P2 provenance storage is **not** temporal. **Forbidden:** timestamp filtering, config/opportunity timestamps, `task_id`/`run_id` joins, provider timing heuristics.

**When `execution_as_of is not None`:**

- Integration configuration **provenance enrichment** from `ExecutionIntegrationConfigurationProvenanceReader` is **not performed**.
- Reconstruction **must not** inject current complete pin state into an as-of view.
- Requirement classification may still be derived from **truncated positioned history** (spine events at or before boundary) for status reporting only.

**When `execution_as_of is None` (full reconstruction):** reader projects current durable pins for in-scope execution IDs subject to requirement fail-closed rules.

Option A (temporal provenance reader) is **not** selected — no persisted boundary index exists without new temporal architecture (**STOP** not triggered only because Option B avoids new store).

---

## 5. Status model

### 5.1 Current enum audit

`ExecutionIntegrationConfigurationProvenanceReadStatus`:

| Value | Truthful today | After R1 |
|---|---|---|
| `NOT_CONFIGURED` | No reader wired | Unchanged — **not** overloaded for temporal unsupported |
| `CONFIGURED` | Reader projected pins | Full reconstruction success path |
| `REQUIRED_MISSING` | Defined in P1; underused in rejected P4 | Fail-closed integrity path when requirement evidence demands pin |

### 5.2 Minimum extension (locked)

Add:

```text
UNAVAILABLE_AT_EXECUTION_BOUNDARY = "unavailable_at_execution_boundary"
```

Semantics: `execution_as_of` was set; durable pin projection intentionally withheld (Option B). **Distinct** from `NOT_CONFIGURED` (not applicable / no reader) and from `REQUIRED_MISSING` (full reconstruction integrity violation).

---

## 6. Persistence impact

| Area | R1 production delta | Corrected P4 delta |
|---|---|---|
| P2 pinning store schema | **0** | **0** (no temporal shadow store) |
| Runtime spine / `RuntimeEventType` | **0** | **+1 event type + typed payload contract** |
| Production emitter in `ExecutionBoundIntegrationResolution` | **0** | **Required** |
| Reconstruction projection | **0** | Replace magic payload; branch on `execution_as_of`; extend status enum |
| Second provenance store | **Forbidden** | **Forbidden** |

---

## 7. Authority / ownership matrix

| Concern | Owner | R1 |
|---|---|---|
| Requirement fact emit | Integrations — `ExecutionBoundIntegrationResolution` | Lock |
| Requirement fact contract | `intergrax/contracts/` | Lock |
| Spine persistence | Existing runtime evidence chain | Reuse |
| Provenance pin write | P2 pinning store (unchanged) | Lock |
| Provenance read adapter | Applications `integration_configuration_provenance_reader.py` | Unchanged |
| Reconstruction projection | `integration_configuration_provenance_projection.py` | Spec |
| `ExecutionReconstructor` | Single owner | Unchanged |
| Diagnostics | Read-only injection | No truth ownership |
| Parent→child provenance | No automatic inheritance | Unchanged |

---

## 8. Compatibility impact

| Item | Impact |
|---|---|
| `execution_integration_configuration_provenance_required` in `RuntimeEvent.payload` | **Semantic authority → 0**; remove from reconstruction classification; tests may keep temporary shim until emitter lands |
| Rejected P4 qual doc “READY FOR AUDIT” | **Superseded** — P4 blocked until corrected implementation |
| P3 configured execution path | Must gain spine emitter (no P4-only fake events) |
| API consumers of `integration_configuration_provenance_read_status` | Must handle `UNAVAILABLE_AT_EXECUTION_BOUNDARY` |

---

## 9. Expected implementation delta (corrected P4 — not in R1 commit)

1. **Contracts:** `ExecutionIntegrationConfigurationProvenanceRequirementEvidence` + `RuntimeEventType` spine value; extend `ExecutionIntegrationConfigurationProvenanceReadStatus`.
2. **P3 emit:** After successful pin in `ExecutionBoundIntegrationResolution`, append typed spine event to execution evidence (tenant + execution_id + subject keys + adoption fingerprint).
3. **P4 projection:** Remove magic payload discovery; parse typed spine events; pass `execution_as_of` into projection from `ExecutionReconstructor`; Option B branch skips `reader.read_all` when boundary set.
4. **Fail closed:** `required_ids` from spine ∧ `execution_as_of is None` ∧ empty/conflicting pins → `ExecutionReconstructionIntegrityError` / `REQUIRED_MISSING` as appropriate.
5. **Tests:** Unit gates updated; **mandatory Docker durable restart E2E** (§10).

**Do not** resume rejected baseline `055ed448…`.

---

## 10. Docker E2E plan (mandatory in corrected P4 — not R1)

**Goal:** Prove configured execution → P2/P3 pin → **physical durability** → process/composition restart → `ExecutionReconstructor` → historical configured/effective provenance; then mutate live configuration/opportunity and prove reconstruction unchanged.

| Constraint | Choice |
|---|---|
| Backend | Existing production adapter — `DocumentStoreExecutionIntegrationConfigurationPinningStore` and/or `KvExecutionIntegrationConfigurationPinningStore` over platform `ConditionalDocumentStore` / `DistributedKVStore` |
| Infra pattern | Reuse integration proof style: e.g. `infra/docker/postgresql/docker-compose.yml` or Mongo path from `tests/integration/applications/architecture/harden_*` (operator-run Docker; pytest gated on daemon) |
| Forbidden | `InMemory*` as sole proof; bespoke test-only store |
| Restart boundary | New composition root / new adapter instance / new `ExecutionReconstructor` in **separate process or fresh wiring** after first process destroyed |
| Minimum assertions | Tenant isolation; `ExecutionId` preserved; pins survive restart; required path fail-closed without pin; post-mutation reconstruction returns **pinned** facts |
| P5/CERT | Vendor matrix, adversarial provider cases — **explicitly deferred** (§11) |

---

## 11. P4 vs P5/CERT separation

| Scope | P4 (corrected) | P5/CERT (later) |
|---|---|---|
| Durable restart E2E | **≥1** real backend | Full closed-world backend matrix |
| Configured provider production path | One representative Docker proof | Per-vendor representative + adversarial |
| Requirement authority | Spine + P2 pin | Regression gates across matrix |

---

## 12. STOP rule evaluation

| Trigger | Result |
|---|---|
| New durable semantic store | **Not required** (spine + existing P2) |
| New execution authority | **Not required** |
| New event truth mechanism | **Minimal spine extension only** — pre-approved by parent TRACE-X evidence model |
| Incompatible reconstruction redesign | **No** — additive status + projection branch |
| Temporal persistence architecture | **Avoided** via Option B |

**STOP — ARCHITECTURE DECISION REQUIRED:** **not invoked** for R1.

---

## 13. Tracker sync (intent)

| ID | Status after R1 push |
|---|---|
| TRACE-X-P5-R2-P4 | **BLOCKED ON R1** until corrected implementation; rejected audit baseline superseded |
| TRACE-X-P5-R2-P4-R1 | **READY FOR AUDIT** @ `FINAL_COMMIT` |
| FRZ-TRC-11 | **OPEN** |
| P5 | **NOT ENTERED** |
| CERT | **NOT ENTERED** |

---

## 14. Independent audit notice

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
