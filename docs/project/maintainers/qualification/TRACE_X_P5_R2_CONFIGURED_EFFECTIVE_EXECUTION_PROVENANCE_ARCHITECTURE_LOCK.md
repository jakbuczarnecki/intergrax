# TRACE-X-P5-R2 — Configured→Effective Execution Provenance Architecture Lock

## Revision record

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P0` + **`TRACE-X-P5-R2-P0-R1`** + **`TRACE-X-P5-R2-P0-R1-R1`** + **`TRACE-X-P5-R2-P0-R1-R1-R1`** (configuration opportunity typing) + **`TRACE-X-P5-R2-P0-R1-R1-R1-R1`** (configuration mutation risk classification ownership) + **`TRACE-X-P5-R2-P0-FINAL`** (final architecture lock reconciliation) |
| **P0-FINAL START_HEAD** | `33076146071dd5246691821905ddcba383ad5ee6` (`development` — independently accepted **P0-R1-R1-R1-R1** reconciliation) |
| **P0-R1-R1-R1-R1 accepted reconciliation** | `33076146071dd5246691821905ddcba383ad5ee6` |
| **Parent** | `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **P0 rejection baseline** | `982f945de67577865c1ade4ebbea519cf3a9b284` |
| **P0-R1 START_HEAD** | `982f945de67577865c1ade4ebbea519cf3a9b284` |
| **P0-R1-R1 START_HEAD / AUDIT_BASE** | `6b5b5f2e1fe2da8655e85173b24fd3221e65b602` (`development`) |
| **P0-R1-R1-R1 START_HEAD** | `305ac4a0cb07e303ce2bb0b04c2c4520db35786b` (`development`) |
| **P0-R1-R1-R1-R1 START_HEAD** | `925bcae17cbf1072297296378c9e28c8742f9c92` (`development`) |
| **Primary FRZ** | `FRZ-TRC-11` (**OPEN** — no PASS) |
| **Blocker** | `P5-GAP-04` — **ARCHITECTURALLY SPECIFIED / IMPLEMENTATION OPEN** (no canonical global configured→effective→`ExecutionId`→evidence chain in production until **P1–P5/CERT**) |
| **In-scope P0 blocker (R1-R1-R1)** | `R2-P0-CONFIGURATION-OPPORTUNITY-TYPING-04` = **RESOLVED IN DESIGN (R1-R1-R1)** |
| **In-scope P0 blocker (R1-R1-R1-R1)** | `R2-P0-CONTROL-PLANE-RISK-AUTHORITY-05` = **RESOLVED IN DESIGN (R1-R1-R1-R1)** |
| **Independent-audit blockers** | `R2-P0-EFFECTIVE-IDENTITY-PREBUILT-01` · `R2-P0-CONFIGURED-ADOPTION-AUTHORITY-02` = **RESOLVED IN DESIGN (R1)** · `R2-P0-CONCRETE-ADOPTION-ROOT-03` = **RESOLVED IN DESIGN (R1-R1)** · `R2-P0-CONFIGURATION-OPPORTUNITY-TYPING-04` = **RESOLVED IN DESIGN (R1-R1-R1)** · `R2-P0-CONTROL-PLANE-RISK-AUTHORITY-05` = **RESOLVED IN DESIGN (R1-R1-R1-R1)** @ `33076146…` — **P0-FINAL** reconciles normative precedence; pending **P0-FINAL** independent audit |
| **Production / runtime delta** | **0** (architecture + qualification design only) |
| **Status** | **TRACE-X-P5-R2-P0-FINAL = READY FOR AUDIT** · **TRACE-X-P5-R2-P0 = READY FOR AUDIT** · **TRACE-X-P5-R2-P0-R1-R1-R1-R1 = independently accepted reconciliation @ `33076146…`** · **TRACE-X-P5-R2-P0 initial @ `982f945…` = REJECTED / SUPERSEDED** · **TRACE-X-P5-R2 = CURRENT / BLOCKED ON P0 FINAL INDEPENDENT AUDIT** · **TRACE-X-P5 = CURRENT / BLOCKED ON R2** |

**Steering sources revalidated @ P0-R1-R1-R1-R1 START_HEAD (`925bcae…`):** [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md), [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md), [`TRACE_X_P5_POLICY_PROFILE_CONFIGURATION_PROVENANCE_BASELINE.md`](TRACE_X_P5_POLICY_PROFILE_CONFIGURATION_PROVENANCE_BASELINE.md), [`TRACE_X_P5_R1_POLICY_PROFILE_EXECUTION_ATTRIBUTION_CERTIFICATION.md`](TRACE_X_P5_R1_POLICY_PROFILE_EXECUTION_ATTRIBUTION_CERTIFICATION.md), [`INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md`](../architecture/INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md).

**Historical evidence (revalidated, not blindly trusted):** INT-CONFIG-REAL-X-CERT `a59744517b92847f55def1db22826d17d89ee155` · TRACE-X-P5-P0 `81fd1490f18d73eaf31ec94c2b93dbb525026ba2` · TRACE-X-P5-R1 accepted evidence tip `05fd5d9b2b97f9d85a534a949d882cd47d4a54c9` (docs/bookkeeping lineage e.g. `16be5fdfe4bcde804eef51a3b34b4ebc5cd45dad` — **not** R1 implementation evidence) · P0 draft audited/rejected @ `982f945d…` (identity + adoption authority gaps).

### FINAL NORMATIVE RECONCILIATION (P0-FINAL — authoritative precedence)

Read this subsection first. Earlier prose in §1–§3 without the markers below is **historical context** unless it matches this map.

| Precedence (later wins within P0 chain) | Scope |
|---|---|
| **§1A** | Effective `provider_id` + configured adoption input + provenance modes (`CONFIGURED_ADOPTED` / `EFFECTIVE_ONLY`) + caller disposition (§1A.7) |
| **§1B** | CONFIGURE_EXISTING orchestration owner = Autonomous Work fulfillment; Applications = composition/wiring only; explicit adoption handoff; no binding lookup |
| **§1C** | Integrations-owned `ExistingCapabilityConfigurationOpportunity`; opaque `configuration_ref`; exact `read_exact`; discovery projection; risk continuity (§1C.13) |
| **§1C.26–§1C.31** | Configuration mutation risk: Integrations classifies; Governance authorizes only — rejects `A0 → LOW` and `WorkerCapabilityCandidate.risk_class` → `risk_classification` |

**Superseded / non-normative (audit lineage only):** P0 initial @ `982f945…` (REJECTED); pre-§1A effective identity (`category.value`, slug+category); implicit configured binding discovery; Applications as CONFIGURE_EXISTING sequencer; generic “opportunity resolver” without §1C semantics; `ExternalWorkIntegration` in `CONFIGURED_ADOPTED` v1 (C1 exclusion).

**Single after graph (§3):** opportunity provider → Integrations opportunity owner → mutation risk policy → immutable opportunity → opaque ref → AW discovery → CONFIGURE_EXISTING → fulfillment coordinator → configured fulfillment → `read_exact` → realization request/port → Governance **ALLOW** → `ConfiguredCapabilityBinding` → `ExecutionIntegrationConfigurationAdoption` → execution-bound resolution → effective identity → configured/effective validation → pin (`tenant_id` + `ExecutionId` + subject) → durable store → neutral reader → `ExecutionReconstructor`. No unexplained semantic jump.

**P0-FINAL child history (evidence, not all independently CLOSED as final stages):**

| Child | SHA | Disposition |
|---|---|---|
| P0 initial | `982f945de67577865c1ade4ebbea519cf3a9b284` | **REJECTED / SUPERSEDED** |
| P0-R1 | `6b5b5f2e1fe2da8655e85173b24fd3221e65b602` | Remediation evidence; superseded by later reconciliation |
| P0-R1-R1 | `305ac4a0cb07e303ce2bb0b04c2c4520db35786b` | Remediation evidence; superseded |
| P0-R1-R1-R1 | `925bcae17cbf1072297296378c9e28c8742f9c92` | Remediation + risk-ownership blocker exposed; superseded |
| P0-R1-R1-R1-R1 | `33076146071dd5246691821905ddcba383ad5ee6` | **Independently accepted** child reconciliation |
| P0-FINAL | qualification commit after reconciliation | **READY FOR AUDIT** |

### P0-FINAL exit questionnaire (25 locked answers)

| # | Answer |
|---|---|
| 1 | Configuration Opportunity owner: **Integrations** (§1C) |
| 2 | `configuration_ref` identifies: **immutable Integrations-owned opportunity instance** (exact tenant + ref; no latest) |
| 3 | Configuration mutation risk owner: **Integrations / INT-CONFIG** (§1C.13) |
| 4 | Configuration realization authorization: **Governance only** |
| 5 | Configuration realization executor: **Integrations / INT-CONFIG** (`ExistingCapabilityConfigurationRealizationPort`) |
| 6 | Effective provider materialization: **existing Integrations resolver** (`resolve` / `resolve_from_profile`) inside execution-bound wrapper |
| 7 | Effective provider identity observed: **§1A.3** (catalog tri-equality or `instance.provider_id`; never `category.value`) |
| 8 | Configured binding adoption selection: **explicit `ExecutionIntegrationConfigurationAdoption`** (AW fulfillment path) |
| 9 | Binding lookup forbidden? **YES** |
| 10 | Latest/current fallback forbidden? **YES** |
| 11 | Canonical `ExecutionId` enters: **Execution admission** on governed qualified dispatch (§1B.12) before pin |
| 12 | Provenance pinned: **`ExecutionIntegrationConfigurationPinningStore`** per `tenant_id` + `ExecutionId` + `IntegrationConfigurationSubject` |
| 13 | Persistence durable? **YES** (mandatory for FRZ-TRC-11 path; P2 implements store) |
| 14 | Reconstruction owner: **`ExecutionReconstructor`** via neutral reader only |
| 15 | Evidence creates configuration truth? **NO** |
| 16 | AW owns provider configuration semantics? **NO** (opaque ref only) |
| 17 | Provider plugin owns final risk? **NO** |
| 18 | Governance classifies configuration-domain risk? **NO** |
| 19 | `ExternalWorkIntegration` in CONFIGURED_ADOPTED v1? **NO** (C1) |
| 20 | Child implicit config provenance inheritance? **NO** (§12) |
| 21 | Missing adoption degrades to EFFECTIVE_ONLY? **NO** |
| 22 | Second resolver/catalog/activation lifecycle? **NO** |
| 23 | Tenant identity continuous end-to-end? **YES** (§13, §1B.9, §1C.14) |
| 24 | All five P0 blockers resolved in normative design? **YES** (§1A–§1C.31) |
| 25 | Semantic decisions deferred to P1? **NO** (P1 implements contracts only; file-level choices OK) |

**Tenant isolation audit (P0-FINAL):** **PASS — local R2 architecture scope** (fulfillment → opportunity → INT-CONFIG → adoption → resolution → Execution → pin → reconstruction share one `tenant_id` chain; no global **FRZ-TEN-*** promotion).

---

## 1. Architecture conclusion

**Outcome B — no existing sanctioned join point exists.**

At START_HEAD there is **no** production surface that simultaneously holds:

1. canonical `ConfiguredCapabilityBinding` identity (when INT-CONFIG adoption applies),
2. sanctioned effective integration selection/materialization facts,
3. tenant identity,
4. canonical `ExecutionId`.

Therefore P0 locks a **minimum typed seam** (R1-analogous pattern) without implementing it.

**P0-R1 normative override:** P0 @ `982f945d…` was rejected for (1) unsound effective identity (`slug + materialization_kind + category`, including `category.value` as provider stand-in for pre-built `IntegrationBinding(instance=…)`), and (2) missing configured-adoption handoff contract. **§1A** reconciles both. Where §1A conflicts with earlier P0 prose below, **§1A wins**.

---

## 1A. P0-R1 — effective identity & configured adoption (normative)

### 1A.1 Rejected P0 identity model

| Rejected | Reason |
|---|---|
| `effective provider slug = category.value` for pre-built profile slots | `category` is integration kind, not provider identity; `IntegrationBinding.resolved_slug()` → `None` when `instance` is set (`intergrax/integrations/contracts/binding.py`) |
| `slug + materialization_kind + category` without canonical `provider_id` | Collapses distinct pre-built providers sharing a category; unsound for provenance |
| `configuration_fingerprint` as effective provider proof | Fingerprint is configured payload identity only |

**Invariant:** `category ≠ provider identity` always. Unprovable provider identity → **fail closed** for configured-adoption provenance (`EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE`).

### 1A.2 `EffectiveIntegrationIdentity` (typed conceptual model — future contract)

Minimum fields (no duplicate provider ID registry):

| Field | Semantics |
|---|---|
| `integration_category` | `IntegrationCategory` |
| `provider_id` | Canonical provider identity (see matrix §1A.3) |
| `materialization_kind` | `CATALOG_FACTORY` \| `PROFILE_PREBUILT` — **does not** substitute for `provider_id` |

Same logical provider via different legal composition paths may differ in `materialization_kind` but **must** preserve the same `provider_id`.

### 1A.3 Materialization-path identity matrix (@ `982f945d…`)

| Path | `materialization_kind` | Authoritative `provider_id` | Validation |
|---|---|---|---|
| **A — Catalog factory** (`resolve` → `get_entry` → factory → `PlatformIntegrationContract`) | `CATALOG_FACTORY` | Normalized catalog identity: **`entry.slug` == `IntegrationContractSpec.provider_id` == materialized `PlatformIntegrationContract.provider_id`** where all are exposed (`intergrax/integrations/registry/contract_spec.py` `validate_contract_spec_identity`) | Mismatch at registration or at adoption pin → fail closed (`CONFIGURED_ADOPTION_PROVIDER_MISMATCH`) |
| **B — Pre-built `PlatformIntegrationContract`** (`resolve_from_profile` → `instance_for_category` → instance) | `PROFILE_PREBUILT` | **`instance.provider_id`** (precedent: `intergrax/core/qualification/execution.py::resolve_integration_provider_id`) | Must pass `contract_for_category` typing; independent compare to configured `provider_id` — **never** assign `effective.provider_id = configured.provider_id` |
| **C — DI-only `ExternalWorkIntegration`** (`CategoryIntegrationInstance` union) | — | **No canonical static `provider_id` on Protocol** (`intergrax/integrations/contracts/external_work.py`) | **Option C1 — locked:** excluded from `CONFIGURED_ADOPTED` in R2 v1; effective-only / profile-only; **cannot** close FRZ-TRC-11 configured→effective proof. `discover()` descriptor is runtime I/O — **not** adoption identity authority without contract change (**STOP** if required — separate child; not P0-R1) |

**Forbidden identity sources (all paths):** Python class/module name, `repr`, memory address, `category.value`, reflection, env slug alone without post-materialization `provider_id` check on catalog path.

### 1A.4 Configured adoption authority & handoff

```text
INT-CONFIG realization
      ↓
ConfiguredCapabilityBinding (immutable)
      ↓
explicit host/application composition handoff (caller-owned choice of binding)
      ↓
Integrations execution-bound resolution wrapper
      ↓
existing resolve / resolve_from_profile (unchanged semantics)
      ↓
effective identity extraction + configured/effective match
      ↓
execution provenance pin (ExecutionId owner unchanged)
```

| Concern | Owner |
|---|---|
| Emit configured binding | Integrations / INT-CONFIG |
| Decide **whether** and **which** binding is adopted | Sanctioned **Applications composition root** |
| Validate binding vs effective materialization | Integrations execution-bound wrapper |
| Materialize provider | Existing Integrations factory (**sole** resolver) |
| `ExecutionId` | Execution |
| Durable provenance | Provenance store (port) + Applications adapters |
| Reconstruction | Evidence / `ExecutionReconstructor` (neutral reader only) |

**Wrapper MUST NOT look up configured state** (no current/latest store, global registry, task/run metadata, `RuntimeConfig`, `IntegrationProfile.options`, timestamps, provider lookup, environment). Adoption input is **explicit DI**, not service location.

**Future input contract (Integrations-owned, name illustrative):** `ExecutionIntegrationConfigurationAdoption` — encapsulates exactly one immutable `ConfiguredCapabilityBinding` (reuse preferred over field duplication) plus minimum typed resolution subject context (at minimum: `integration_category`, `resource_scope` for the wiring target). No permission semantics; no `ExecutionId` minting.

### 1A.5 Adoption match invariants (fail closed)

Before provenance pin:

```text
configured.tenant_id == composition tenant == execution tenant == provenance tenant
configured.integration_category == requested effective category
configured.provider_id == effective.provider_id   # independent observation on pre-built/catalog
configured.resource_scope == adoption context resource_scope   # composition-supplied; no string heuristics
configured.configuration_type / configuration_version / configuration_fingerprint retained as configured truth
```

**Missing adoption** on a caller classified `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` → `CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING` — **no** silent downgrade to `EFFECTIVE_ONLY`.

**Conflicting adoption** for same execution + subject → fail closed (no last/latest/first wins).

### 1A.6 Provenance record shape (reconciled — not implemented)

```text
ExecutionIntegrationConfigurationProvenance
    tenant_id
    execution_id
    mode: CONFIGURED_ADOPTED | EFFECTIVE_ONLY

    effective: EffectiveIntegrationIdentity

    configured: ConfiguredIntegrationProvenanceSlice | None
        # tenant_id, integration_category, provider_id, resource_scope,
        # configuration_type, configuration_version, configuration_fingerprint,
        # optional realization_evidence_refs (factual linkage only)
```

| `mode` | Rules |
|---|---|
| `CONFIGURED_ADOPTED` | `configured` **required**; fields match `effective` per §1A.5; counts toward FRZ-TRC-11 when path is classified configured-required |
| `EFFECTIVE_ONLY` | `configured` **absent**; valid runtime on many surfaces today; **does not** close configured→effective traceability; track CONFIG-X if platform must later adopt INT-CONFIG on that surface |

Persistence key remains `(tenant_id, execution_id, IntegrationConfigurationSubject)` with subject dimensions reconciled to **`provider_id`** (not category-as-slug). Effective-only records **do not fabricate** `resource_scope` / `configuration_type`.

### 1A.7 Production caller disposition registry (@ `982f945d…`)

Mechanical inventory of **direct** `resolve` / `resolve_from_profile` in `intergrax/` production modules (tests/scaffold excluded). **Zero** production `ConfiguredCapabilityBinding` consumers outside INT-CONFIG pipeline (false positive: `production_delegated_subtask_plans.configured_binding` = package binding id).

| Module | Disposition | R2 v1 notes |
|---|---|---|
| `applications/_shared/notification_wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | No ExecutionId at wire time |
| `applications/_shared/identity_wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `applications/_shared/security_runtime_bridge.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `applications/_shared/sandbox_host_wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `applications/_shared/integration_tool_wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `applications/_shared/adaptive_feature_flag_gate.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `runtime/persistence/integration_profile_wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `runtime/sandbox/hosted_resolver.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `runtime/codecraft/substrate.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `runtime/vendor_knowledge/resolver.py` | `NON_EXECUTION` | Knowledge source routing |
| `tools/registry/wiring.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `speech_adapters/registry/profile.py`, `resolver.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `rag/bootstrap/rag_stack_bootstrap.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `rag/vectorstore/bootstrap/integration_vectorstore.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | uses `resolve` |
| `rag/rerankers/integration/resolver.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | uses `resolve` |
| `rag/document_loaders/integration/resolver.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | |
| `integrations/_shared/health.py` | `BOOTSTRAP_OR_INFRASTRUCTURE` | Health probe |
| *(none @ baseline)* | `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` | **Reserved** for post–P3 surfaces that adopt INT-CONFIG bindings |

**Post-P3 rule:** `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` callers **must not** call `resolve` / `resolve_from_profile` directly; wrapper bypass → qualification **FAIL**. `EXECUTION_EFFECTIVE_ONLY` / `BOOTSTRAP_OR_INFRASTRUCTURE` may still call factory directly when classification proves non-adoption.

**Resource scope @ execution seam:** Today’s bootstrap callers often lack a platform-wide typed `ExecutionResourceScope` on the composition root. Safe configured adoption therefore requires the **composition root to pass explicit `resource_scope`** alongside adoption (must equal `ConfiguredCapabilityBinding.resource_scope`). Without that typed context, configured adoption for scoped bindings is **illegal** (fail closed), not guessed from profile.

### 1A.8 Failure semantics (future typed errors)

`CONFIGURED_ADOPTION_TENANT_MISMATCH` · `CONFIGURED_ADOPTION_CATEGORY_MISMATCH` · `CONFIGURED_ADOPTION_PROVIDER_MISMATCH` · `CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH` · `EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE` · `CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING` — no free-form-only boundary.

**Adversarial qualification (future):** pre-built `provider-a` configured vs `provider-b` effective must fail; catalog sqlite configured vs postgres effective must fail; missing adoption on configured-required path must not degrade.

### 1A.9 P0-R1 exit questionnaire (locked answers)

| # | Answer |
|---|---|
| 1 | Catalog: tri-equality `slug` / spec `provider_id` / contract `provider_id` |
| 2 | Pre-built contract: `PlatformIntegrationContract.provider_id` |
| 3 | `ExternalWorkIntegration`: **no** `CONFIGURED_ADOPTED` in R2 v1 (C1) |
| 4 | Specific binding via composition → wrapper `ExecutionIntegrationConfigurationAdoption` |
| 5 | Handoff owner: Applications composition; validation: Integrations wrapper |
| 6 | Wrapper lookup of configured state? **NO** |
| 7 | Match: §1A.5 independent `provider_id` observation |
| 8 | `CONFIGURED_ADOPTED` vs `EFFECTIVE_ONLY` via caller registry + explicit adoption input |
| 9 | Missing adoption downgrade? **NO** |
| 10 | Wrapper mandatory for `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` only |
| 11 | Fail closed: §1A.5, §1A.8, identity unavailable, mismatch, conflict |
| 12 | Second resolver/lifecycle? **NO** |

### 1A.10 Implementation sequencing (post-R1)

```text
P0-R1  architecture reconciliation (this revision)
   ↓
P0     parent lock closure / bookkeeping after R1 audit
   ↓
P1     neutral provenance/adoption contracts + validators
   ↓
P2     pinning store + durable adapters
   ↓
P3     execution-bound wrapper + composition migration
   ↓
P4     reconstruction
   ↓
P5     closed-world / adversarial qualification
   ↓
CERT
```

---

## 1B. P0-R1-R1 — concrete CONFIGURE_EXISTING adoption root (normative)

**Blocker `R2-P0-CONCRETE-ADOPTION-ROOT-03`:** at `6b5b5f2e…` there is no production path from `CapabilityAcquisitionDisposition.CONFIGURE_EXISTING` through INT-CONFIG realization to execution-bound effective resolution. **Resolution:** lock exactly one future orchestration root under Autonomous Work fulfillment (parallel to DIRECT_REUSE), wired from Applications composition only.

**P0-R1-R1 normative override:** where §1A.4 implied Applications as the **sequencing** owner for CONFIGURE_EXISTING, **§1B wins for orchestration**: Applications **wires** ports; **Autonomous Work fulfillment** sequences CONFIGURE_EXISTING after an acquisition decision is already made. §1A.3–§1A.6 (identity, adoption input, provenance) remain authoritative.

### 1B.1 Closed-world inventory — CONFIGURE_EXISTING producers (@ `6b5b5f2e…`)

| Surface | Path | Role |
|---|---|---|
| **Decision producer (sole production)** | `WorkerCapabilityAcquisitionDecisionService.decide` | `intergrax/autonomous_work/capability_acquisition_service.py` |
| **Disposition rule** | same | `CONFIGURE_EXISTING` iff selected `WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION` (A0) |
| **Candidate kind producer** | `WorkerConfigurationOpportunityDiscoveryPort.discover` | Port only; production adapters are host-composed (`MappingConfigurationOpportunityDiscoveryAdapter` in tests/scaffold patterns) |
| **Contract invariants** | `validate_acquisition_decision_invariants` | `intergrax/contracts/autonomous_work/capability_acquisition.py` |

**Producer inputs (typed):** `WorkerCapabilityAcquisitionRequest` (need, recovery correlation, profile ref); discovery layers supply `WorkerCapabilityCandidate` (`capability_ref`, optional `configuration_ref`, operations, evidence).

**Tenant:** acquisition request need / fulfillment — **not** on candidate; fulfillment uses `WorkerCapabilityFulfillmentRequest.tenant_id`.

**Recovery correlation:** `recovery_decision_id`, `discovery_correlation_id` via recovery provenance (fulfillment path).

**No other production module emits `CONFIGURE_EXISTING`.**

### 1B.2 Closed-world inventory — CONFIGURE_EXISTING consumers (@ `6b5b5f2e…`)

| Search target | Production consumer branching on `CONFIGURE_EXISTING` |
|---|---|
| `WorkerCapabilityAcquisitionResult` / `WorkerCapabilityAcquisitionDecision` | **None** in fulfillment, recovery, or execution modules |
| `WorkerCapabilityFulfillmentCoordinator` | Routes `DIRECT_REUSE` / `REALIZATION_REQUIRED` (UCA) / `QUALIFICATION_COMPLETE` only — **no** CONFIGURE_EXISTING branch |
| `WorkerCapabilityRecoveryCoordinator` | Canonical recovery never maps CONFIGURE_EXISTING to a phase |
| `WorkerCapabilityDirectReuseFulfillmentService` | DIRECT_REUSE only |

**Confirmed:** no complete realization→execution path exists today. **No alternate canonical path to reuse** beyond the locked future graph below.

### 1B.3 INT-CONFIG public entry (@ `6b5b5f2e…`)

| Artifact | Path | Production callers |
|---|---|---|
| `ExistingCapabilityConfigurationRealizationPort` | `intergrax/integrations/contracts/existing_capability_configuration.py` | **0** |
| `ExistingCapabilityConfigurationRealizationFacade` | `intergrax/integrations/existing_capability_configuration_facade.py` | **0** (tests + qualification only) |
| Pure core | `ExistingCapabilityConfigurationRealizationService` | internal to facade |

Flow today: **INT-CONFIG realization → `ConfiguredCapabilityBinding` → STOP** (unchanged from §4.3).

### 1B.4 Fulfillment / execution precedents

| Precedent | Path | Reuse for CONFIGURE_EXISTING |
|---|---|---|
| **DIRECT_REUSE** | `WorkerCapabilityDirectReuseFulfillmentService` | **Structural only** — host-available binding → `WorkerHostAvailableCapabilityExecutionPort` → Execution; **not** semantic owner |
| **Generic realization** | `WorkerCapabilityFulfillmentCoordinator._fulfill_realization_required` | **UCA** `CapabilityRealizationCoordinatorPort` — **must not** absorb INT-CONFIG (semantic owner stays Integrations) |
| **Qualified execution** | `WorkerQualifiedCapabilityResumeCoordinator` + governed dispatch | **Post-adoption** execution handoff target (same Execution authority as today) |

**Host-available flow (locked precedent, do not duplicate):**

```text
recovery / DIRECT_REUSE
      → HostAvailableCapabilityBindingPort
      → typed execution target
      → WorkerHostAvailableCapabilityExecutionPort
      → canonical Execution
```

### 1B.5 Host / application composition roots (@ `6b5b5f2e…`)

| Root | Path | Classification |
|---|---|---|
| `build_harness_host_runtime` | `intergrax/applications/_shared/harness_host_runtime.py` | `STATIC_HOST_COMPOSITION` — may wire ports; **not** CONFIGURE_EXISTING adoption owner |
| `build_environment_host_task_execution` | runtime execution composition | `EXECUTION_TIME_COMPOSITION` — Execution admission; **not** INT-CONFIG orchestration owner |
| `compose_application_host_orchestration_session` / scenario baselines | applications shared | `STATIC_HOST_COMPOSITION` / wiring |
| `build_worker_recovery_governed_fulfillment_wiring` | `intergrax/autonomous_work/worker_recovery_governed_fulfillment_composition.py` | **Sanctioned EXECUTION_TIME_COMPOSITION** for worker recovery fulfillment — inject future configured-fulfillment port + INT-CONFIG facade here (wiring only) |

**Forbidden:** adding `ConfiguredCapabilityBinding` to `build_harness_host_runtime` as the primary CONFIGURE_EXISTING activation mechanism.

### 1B.6 Exactly-one orchestration owner (R2 v1)

```text
CONFIGURE_EXISTING sequencing
→ Autonomous Work fulfillment layer (future narrow service under WorkerCapabilityFulfillmentCoordinator)
```

**Future production components (names illustrative; ownership normative):**

| Step | Owner module (future) |
|---|---|
| 1. Consume acquisition decision with `CONFIGURE_EXISTING` | `WorkerCapabilityFulfillmentCoordinator` (new branch — **not** DIRECT_REUSE, **not** UCA realization) |
| 2. Project + invoke INT-CONFIG | `WorkerConfiguredCapabilityFulfillmentService` implementing `WorkerConfiguredCapabilityFulfillmentPort` |
| 3. Realization | `ExistingCapabilityConfigurationRealizationFacade` via injected `ExistingCapabilityConfigurationRealizationPort` |
| 4. Build adoption handoff | `WorkerConfiguredCapabilityFulfillmentService` — wraps exact `configured_binding` in `ExecutionIntegrationConfigurationAdoption` |
| 5. Effective resolution + identity compare | `ExecutionBoundIntegrationResolution` (Integrations — §10.2) |
| 6. Execution dispatch | existing `WorkerQualifiedCapabilityResumeCoordinator` / governed dispatch — **same Execution authority**; no provider business call from AW |
| 7. Provenance pin | `ExecutionBoundIntegrationResolution` at `tenant_id` + `ExecutionId` + adoption (P1–P4 waves) |

**Coordinator delegation (preferred):**

```text
WorkerCapabilityFulfillmentCoordinator
       ↓
WorkerConfiguredCapabilityFulfillmentPort
       ↓
WorkerConfiguredCapabilityFulfillmentService
       ↓
ExistingCapabilityConfigurationRealizationPort
       ↓
ConfiguredCapabilityBinding
       ↓
ExecutionIntegrationConfigurationAdoption
       ↓
ExecutionBoundIntegrationResolution
       ↓
resolve / resolve_from_profile (unchanged)
       ↓
effective provider_id (independent observation)
       ↓
provenance pin + canonical Execution
```

**MUST NOT (orchestration service):** implement provider configuration; choose provider implementation; inspect provider objects; mutate `ConfiguredCapabilityBinding`; authorize provider use; mint `ExecutionId`; call provider business APIs.

### 1B.7 Typed request projection (AW → INT-CONFIG)

`WorkerCapabilityCandidate` alone (`capability_ref`, `configuration_ref`) is **discovery/decision correlation only** — **not** a substitute for `ConfiguredCapabilityBinding` (§23).

**Sanctioned enrichment (future P1 — fail closed until present):** Integrations-owned **`ExistingCapabilityConfigurationOpportunityReadPort.read_exact(tenant_id, configuration_ref)`** (§1C) returns immutable **`ExistingCapabilityConfigurationOpportunity`**. Autonomous Work **projects** request envelopes only; **never** parses opaque `configuration_ref` into provider semantics.

| AW / fulfillment source | INT-CONFIG `ExistingCapabilityConfigurationRealizationRequest` field | Validation | Owner |
|---|---|---|---|
| `WorkerCapabilityFulfillmentRequest.tenant_id` | `tenant_id` | must match principal.tenant_id | AW copy; INT-CONFIG validates |
| `WorkerPrincipalBindingRepository` + episode context (same wiring as `build_worker_recovery_governed_fulfillment_wiring`) | `principal: RequestIdentity` | tenant match; **no AW synthesis** of admin/system identity | Applications composition supplies repository; AW reads admitted binding |
| Derived deterministic id (fulfillment operation + decision id) | `request_id` | unique per logical attempt | AW |
| Exact typed opportunity read | `integration_category`, `provider_id`, `resource_scope`, `current_revision` | typed target; fail closed if missing | **Integrations** read port |
| Exact typed opportunity read | `configuration: IntegrationConfigurationPayload` + `configuration_fingerprint` | fingerprint ≡ payload fingerprint (§1C.23) | **Integrations** read port |
| Exact typed opportunity read | `risk_classification: ControlPlaneMutationRisk` | must equal `opportunity.risk_classification`; **no** AW / A0 derivation | **Integrations** (opportunity owner + mutation-risk policy) |
| `WorkerCapabilityFulfillmentRequest.task_id` / `run_id` | `task_id` / `run_id` | optional correlation | AW |
| recovery `discovery_correlation_id` | `correlation_ref` | optional | AW |

**Forbidden:** infer `provider_id` / `resource_scope` / configuration body from `configuration_ref` or `capability_ref` without resolver; metadata dict gaps; latest-binding lookup.

**STOP avoided (opportunity typing):** enrichment boundary is explicit in **§1C**; @ `305ac4a0…` production has **no** Integrations-owned opportunity source — path **fails closed** until P1 contracts + read port exist (not silent skip).

### 1B.8 Principal / Governance

INT-CONFIG façade requires `RequestIdentity` on the realization request. Reuse **existing** worker principal binding read path (`WorkerPrincipalBindingRepository` already composed for governed fulfillment). Autonomous Work **must not** mint configuration-service or system-admin principal. `CONFIGURE_EXISTING` decision **≠** Governance permission; façade `authorize` **≠** provider execution permission (§27).

### 1B.9 Tenant continuity (locked)

```text
WorkerCapabilityFulfillmentRequest.tenant_id
== ExistingCapabilityConfigurationRealizationRequest.tenant_id
== ConfiguredCapabilityBinding.tenant_id
== ExecutionIntegrationConfigurationAdoption (via binding)
== ExecutionBoundIntegrationResolution tenant
== Execution tenant
== provenance tenant
```

Any mismatch → fail closed. No tenant from `configuration_ref`, provider, environment, or host defaults.

### 1B.10 Configuration result handling

Only `ExistingCapabilityConfigurationRealizationResult.configured_binding` proceeds. Forbidden: discard-then-lookup; reconstruct binding from `configuration_ref` after realize.

### 1B.11 Recovery semantics — **Option A (locked)**

After successful INT-CONFIG realization, configuration is **directly adoptable** via explicit `ExecutionIntegrationConfigurationAdoption` + execution-bound resolution; execution continues on the configured-adoption path.

**Not Option B** (mandatory recovery rediscovery as gate) for R2 v1: INT-CONFIG does not imply host-catalog visibility; rediscovery would invite `configuration_ref` / latest lookup substitutes.

**Not** generic UCA `_fulfill_realization_required` reconcile loop for INT-CONFIG: that path owns **catalog capability realization**, not Integrations configuration realization.

**Post-realization:** optional **non-authoritative** recovery reconcile for telemetry only — **must not** replace binding identity or select a different configured fact.

### 1B.12 ExecutionId timing & provenance pin

| Milestone | When |
|---|---|
| CONFIGURE_EXISTING decision | Before fulfillment branch; **no** `ExecutionId` required |
| INT-CONFIG realization | Correlation via `task_id` / `run_id` / `correlation_ref` only |
| `ExecutionId` | Minted at **canonical Execution admission** on qualified/governed dispatch (same family as `WorkerCapabilityDirectReuseFulfillmentService` + `WorkerExecutionAdmissionPort` / governed task-scoped dispatch) |
| Provider effective | After `ExecutionBoundIntegrationResolution` delegates to `resolve` / `resolve_from_profile` and extracts `EffectiveIntegrationIdentity` |
| Provenance pin | When **both** canonical `ExecutionId` and independent effective `provider_id` are known, with exact `ConfiguredCapabilityBinding` on adoption path (§10.2) |

No `task_id` / `run_id` substitution for `ExecutionId` in provenance records.

### 1B.13 Idempotency / retry

Reuse INT-CONFIG **`request_id`** determinism: same logical fulfillment operation + decision + opportunity fingerprint → same `request_id`; repeated compatible realize returns consistent binding facts; conflicting retry → `CONFIGURATION_REALIZATION_FAILED` / fail closed (no second idempotency system).

### 1B.14 Failure families (future typed)

`CONFIGURATION_REALIZATION_UNAVAILABLE` · `CONFIGURATION_REALIZATION_DENIED` · `CONFIGURATION_REALIZATION_FAILED` · `CONFIGURED_BINDING_MISMATCH` · `CONFIGURED_ADOPTION_UNAVAILABLE` · `EFFECTIVE_RESOLUTION_FAILED` · `EXECUTION_ADMISSION_FAILED` — semantic distinctions mandatory.

### 1B.15 DIRECT_REUSE vs CONFIGURE_EXISTING (precedent table)

| Concern | DIRECT_REUSE (`WorkerCapabilityDirectReuseFulfillmentService`) | CONFIGURE_EXISTING future path |
|---|---|---|
| Decision owner | Acquisition / recovery (host key match) | `WorkerCapabilityAcquisitionDecisionService` → CONFIGURE_EXISTING |
| Binding owner | `HostAvailableCapabilityBindingPort` (host-visible) | INT-CONFIG → `ConfiguredCapabilityBinding` |
| Provider materialization | Pre-existing host binding | `ExecutionBoundIntegrationResolution` after adoption |
| Governance | Execution admission | INT-CONFIG façade authorize + execution admission |
| Execution dispatch | `WorkerHostAvailableCapabilityExecutionPort` | Qualified resume / governed dispatch (shared Execution engine) |
| Tenant | `WorkerCapabilityFulfillmentRequest.tenant_id` | Same chain §1B.9 |
| Evidence | Direct reuse operation ids | Realization evidence refs + execution-config provenance pin |

**No duplicated Execution engine** — same governed dispatch/admission family; different binding/adoption seam only.

### 1B.16 Closed-world future classification (additions)

| Class | R2 v1 instance |
|---|---|
| `CONFIGURATION_DECISION_ROOT` | `WorkerCapabilityAcquisitionDecisionService` |
| `CONFIGURATION_REALIZATION_ROOT` | `ExistingCapabilityConfigurationRealizationFacade` |
| `CONFIGURED_ADOPTION_ROOT` | `ExecutionBoundIntegrationResolution` + explicit adoption input |
| `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` | Future worker integration execution surfaces using INT-CONFIG bindings |
| `BOOTSTRAP_OR_INFRASTRUCTURE` | `build_harness_host_runtime`, profile wiring, RAG/bootstrap resolvers (§1A.7) |
| `NON_EXECUTION` | vendor knowledge, diagnostics |

Exactly **one** sanctioned configured-adoption execution path; unknown duplicate → qualification **FAIL**.

### 1B.17 Future file map (P0-R1-R1 additions to §18)

| File (future) | Role |
|---|---|
| `intergrax/contracts/autonomous_work/worker_configured_capability_fulfillment.py` | `WorkerConfiguredCapabilityFulfillmentPort` + request/result |
| `intergrax/autonomous_work/worker_configured_capability_fulfillment_service.py` | Narrow fulfillment: decision → INT-CONFIG → adoption DTO |
| `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py` | Delegate CONFIGURE_EXISTING branch (edit in implementation wave) |
| `intergrax/autonomous_work/worker_capability_fulfillment_composition.py` | Compose configured fulfillment + realization port |
| `intergrax/autonomous_work/worker_recovery_governed_fulfillment_composition.py` | Wire INT-CONFIG facade + adoption consumer ports |
| `intergrax/integrations/contracts/existing_capability_configuration_opportunity.py` (illustrative) | `ExistingCapabilityConfigurationOpportunity` + `ExistingCapabilityConfigurationOpportunityReadPort` |
| `intergrax/integrations/existing_capability_configuration_opportunity_service.py` (illustrative) | Integrations-owned opportunity registry/read (durable adapter optional P2) |
| `intergrax/autonomous_work/integrations_configuration_opportunity_discovery_adapter.py` (illustrative) | AW projection: Integrations source → opaque `configuration_ref` candidates only |
| `intergrax/integrations/execution_bound_integration_resolution.py` | Adoption consumer + provenance capture (§10.2) |

Applications **wire only**; no configuration semantics in Tier-3.

### 1B.18 P0-R1-R1 exit questionnaire (locked)

| # | Answer |
|---|---|
| 1 | **Consumer:** future `WorkerCapabilityFulfillmentCoordinator` branch on `CONFIGURE_EXISTING` (today: **zero** production consumers) |
| 2 | **Invoker of realization port:** `WorkerConfiguredCapabilityFulfillmentService` |
| 3 | **Forward object:** `ExistingCapabilityConfigurationRealizationResult` → exact `configured_binding` only |
| 4 | **Constructs `ExecutionIntegrationConfigurationAdoption`:** `WorkerConfiguredCapabilityFulfillmentService` |
| 5 | **Consumes adoption:** `ExecutionBoundIntegrationResolution` |
| 6 | **Recovery rediscovery required?** **No** (Option A — §1B.11) |
| 7 | **Effective `provider_id` observed:** post-`resolve` / `resolve_from_profile` inside `ExecutionBoundIntegrationResolution` (§1A.3) |
| 8 | **`ExecutionId` available:** canonical Execution admission on governed qualified dispatch (§1B.12) |
| 9 | **Provenance pin:** `ExecutionBoundIntegrationResolution` capture hook with `ExecutionId` + adoption (§10.2) |
| 10 | **Step owners:** §1B.6 table |
| 11 | **Static host roots:** composition/wiring only (§1B.5) |
| 12 | **Exactly one adoption path?** **Yes** — §1B.6 |
| 13 | **Fail-closed mismatches?** **Yes** — §1A.5, §1B.9, §1B.14 |
| 14 | **Tenant end-to-end?** **Yes** — §1B.9 |
| 15 | **New authority required?** **NO** — reuse principal binding + existing Governance/Execution ports |

### 1B.19 P0-R1-R1 STOP disposition

**No STOP — ARCHITECTURE DECISION REQUIRED** for R1-R1 scope. Configuration opportunity typing was **in-scope blocker** `R2-P0-CONFIGURATION-OPPORTUNITY-TYPING-04` — resolved in **§1C (R1-R1-R1)**.

---

## 1C. P0-R1-R1-R1 — configuration opportunity typing & exact-reference resolution (normative)

**Blocker `R2-P0-CONFIGURATION-OPPORTUNITY-TYPING-04`:** `WorkerCapabilityCandidate` for `EXISTING_CONFIGURATION` carries only generic discovery fields (`candidate_id`, `capability_ref`, optional `configuration_ref`, operations, `risk_class`, evidence refs) and **cannot** safely construct `ExistingCapabilityConfigurationRealizationRequest` without Integrations-owned typed opportunity facts. **Resolution:** lock Integrations-owned immutable **Configuration Opportunity** + tenant-scoped **exact read** contract; AW carries **opaque `configuration_ref` only**.

**P0-R1-R1-R1 normative override:** where §1B.7 named a generic “opportunity resolver”, **§1C wins** on opportunity semantics, reference immutability, discovery ownership, and fulfillment read path. §1B orchestration graph unchanged except for the mandatory opportunity seam inserted below.

### 1C.1 Architectural boundary (non-negotiable)

```text
Autonomous Work  →  selects among configuration opportunities (opaque ref)
Integrations     →  owns opportunity contents + exact reference resolution
```

**Forbidden:** expanding `WorkerCapabilityCandidate` with `provider_id`, `integration_category`, `IntegrationConfigurationPayload`, `resource_scope`, `current_revision`, or any provider-specific configuration fields.

### 1C.2 Canonical typed opportunity (future contract)

Immutable Integrations-owned fact (names illustrative; semantics normative):

```text
ExistingCapabilityConfigurationOpportunity
    configuration_ref          # Integrations-issued stable exact key
    tenant_id
    integration_category
    provider_id                  # catalog identity — not effective proof
    resource_scope
    current_revision             # pinned at opportunity creation
    configuration: IntegrationConfigurationPayload
    configuration_fingerprint
```

**Excluded from opportunity (by design):** Governance authorization evidence; `RequestIdentity` / principal; materialized provider object; `ExistingCapabilityIntegrationTarget`; execution identity; effective `provider_id` observation.

### 1C.3 Exact-reference read contract (future)

```text
ExistingCapabilityConfigurationOpportunityReadPort.read_exact(
    tenant_id: str,
    configuration_ref: str,
) -> ExistingCapabilityConfigurationOpportunity
```

| Rule | Requirement |
|---|---|
| Addressing | `(tenant_id, configuration_ref)` — **not** `read(configuration_ref)` alone |
| Cardinality | Exactly one opportunity; `0` → fail closed; `>1` → fail closed |
| Immutability | Referenced opportunity content is immutable for that ref |
| Opacity | AW treats `configuration_ref` as opaque; only Integrations interprets storage |
| Latest/current | **Forbidden** — no `tenant + provider → latest` |

### 1C.4 Semantic ownership

| Concern | Owner |
|---|---|
| Opportunity contents + fingerprint/revision pinning | Integrations |
| `configuration_ref` issuance + storage semantics | Integrations |
| Exact read API | Integrations |
| Capability selection among projected candidates | Autonomous Work |
| Request envelope projection + orchestration | Autonomous Work fulfillment |
| INT-CONFIG realization | Integrations (`ExistingCapabilityConfigurationRealizationPort`) |
| Governance authorization on realize | Governance (façade `authorize`) |
| Effective materialization + identity compare | Integrations (`ExecutionBoundIntegrationResolution`) |
| Durable provenance | Provenance store + Applications adapters |
| Reconstruction | Evidence / `ExecutionReconstructor` (neutral reader) |

No second owner for opportunity truth.

### 1C.5 Discovery projection (locked direction)

```text
Integrations Configuration Opportunity source (canonical — future P3)
        ↓ typed opportunities
AW configuration-opportunity discovery adapter (projection only)
        ↓
WorkerCapabilityCandidate (EXISTING_CONFIGURATION)
        configuration_ref  (+ capability_ref, operations, risk_class, evidence_refs)
```

**AW candidate MAY retain:** `configuration_ref`, `capability_ref`, `operations`, `risk_class`, `evidence_refs`.

**AW candidate MUST NOT copy:** full configuration payload or provider/category/resource_scope/revision fields.

**Canonical source rule:** discovery and fulfillment **must** use the **same** Integrations opportunity authority (read port backs refs emitted by the opportunity source). Divergent dict/fixture sources → qualification **FAIL**.

### 1C.6 Fulfillment resolution (after CONFIGURE_EXISTING)

```text
selected_candidate.configuration_ref
        ↓
ExistingCapabilityConfigurationOpportunityReadPort.read_exact(
    WorkerCapabilityFulfillmentRequest.tenant_id,
    configuration_ref,
)
        ↓
ExistingCapabilityConfigurationOpportunity
        ↓
ExistingCapabilityConfigurationRealizationRequest (projected)
        ↓
ExistingCapabilityConfigurationRealizationPort.realize
        ↓
ConfiguredCapabilityBinding
```

AW orchestrates the read call; AW **does not** interpret provider/configuration payload semantics.

**Forbidden pattern:**

```python
ExistingCapabilityConfigurationRealizationRequest(
    provider_id=parse(candidate.capability_ref),
    ...
)
```

### 1C.7 Distinction from adjacent states / resolvers

| Concept | Meaning |
|---|---|
| **Configuration Opportunity** | Configuration **could** be realized (input truth) |
| **ConfiguredCapabilityBinding** | Configuration realization **succeeded** |
| **EffectiveIntegrationIdentity.provider_id** | Observed effective provider after materialization |

Flow: `Opportunity → INT-CONFIG → ConfiguredCapabilityBinding → adoption → effective identity` — preserves `proposed/configurable ≠ configured ≠ effective`.

**`ExistingCapabilityIntegrationResolver`** remains **distinct**: it accepts a fully built `ExistingCapabilityConfigurationRealizationRequest` and resolves/validates an **existing integration target** — **after** the request exists. Opportunity read occurs **before** request construction. Both Integrations-owned; may share underlying typed catalogs/repositories; **must not** compete as truth owners.

### 1C.8 No second Integration Catalog

Opportunity source owns **configuration possibility/input**, not provider registration. Reuse canonical Integration Catalog identity contracts for `provider_id` / `integration_category`. Opportunity source **must not** register providers.

### 1C.9 Opportunity lifecycle (locked — R2 reconstruction safe)

| Question | Locked answer |
|---|---|
| Who creates? | **Integrations** — via future `ExistingCapabilityConfigurationOpportunityProvider` plugin contract (§1C.12) orchestrated by an Integrations opportunity owner service; **not** Autonomous Work |
| When valid? | When Integrations publishes an immutable opportunity record (creation pins `current_revision` + fingerprint) |
| Durable? | **Yes** when product requires historical exact refs; storage behind Integrations port (P2); **not** Evidence Plane SSOT |
| Expire / revoke? | May mark opportunity **invalid/stale** for new realizes; **must not** silently remap old `configuration_ref` to newer revision |
| After supersession | Exact old `configuration_ref` remains readable **or** read returns typed **stale/not-found** — **never** auto-substitute newer revision |
| Payload retention | Opportunity stores typed `IntegrationConfigurationPayload` (or immutable reference resolved at read time with same fingerprint invariant) |

Model **does not** rely on mutable “current configuration” for R2 attribution.

### 1C.10 Persistence preference

Configuration Opportunity is an Integrations **configuration-input fact**, not Evidence-owned truth. Applications may supply durable adapters behind the Integrations read port. **No** Evidence-owned opportunity repository; **no** second STATE-X semantic authority. **Not implemented in R1-R1-R1.**

### 1C.11 Pluginability

```text
external provider plugin
    → ExistingCapabilityConfigurationOpportunityProvider (platform contract)
    → Integrations opportunity owner aggregates/providers
    → read_exact
```

**Forbidden in core R2 logic:** `if provider_id == "sqlite": … elif …`. Provider-specific opportunity emission stays in provider plugins implementing the platform contract.

### 1C.12 Opportunity creation owner (future — not implemented here)

Preferred shape: Integrations service projects opportunities from **known integrations** + provider-specific configuration capability metadata via **`ExistingCapabilityConfigurationOpportunityProvider`** (typed plugin port). Core aggregates; **no** provider branches.

Production @ `305ac4a0…`: **no** Integrations-owned Configuration Opportunity source exists.

### 1C.13 Configuration mutation risk classification (locked — R1-R1-R1-R1)

**Normatively rejected:** `WorkerAutonomyLevel` (including `A0_KNOWN_CAPABILITY`) → `ControlPlaneMutationRisk` (including `LOW`). Autonomy class is owned by Autonomous Work capability acquisition (`A0`…`A4`); `ControlPlaneMutationRisk` is the cross-domain conservative vocabulary on `ControlPlaneMutationRequest` (`intergrax/contracts/control_plane_mutation.py`). They answer different questions and **must not** be conflated or cross-derived.

| Fact @ `925bcae…` | Disposition |
|---|---|
| Canonical `WorkerAutonomyLevel` → `ControlPlaneMutationRisk` map in production | **Absent** — must remain absent |
| `CONFIGURE_EXISTING` acquisition invariant | Requires `A0_KNOWN_CAPABILITY` (`validate_acquisition_decision_invariants`) — **does not** determine mutation risk |
| Production control-plane pattern | Domain mutation owner constructs `ControlPlaneMutationRequest` with `risk_classification` → `ControlPlaneMutationAuthorizationBoundary.authorize` → Governance `PolicyAction` — boundary **evaluates** risk; it **does not** classify Integrations configuration semantics |

**Ownership lock:**

```text
configuration-mutation risk semantics → Integrations / INT-CONFIG
permission decision               → Governance (ControlPlaneMutationAuthorizationPort)
```

Integrations classifies; Governance authorizes using that classification. **Never:** AW autonomy class → Governance risk. **Never:** Governance → configuration realization semantics.

**Canonical INT-CONFIG flow (future P1 — design only):**

```text
provider configuration facts
        ↓
Integrations Configuration Opportunity owner
        ↓
ExistingCapabilityConfigurationMutationRiskPolicy  (Integrations-owned; name illustrative)
        ↓
ControlPlaneMutationRisk
        ↓
immutable ExistingCapabilityConfigurationOpportunity.risk_classification
        ↓
ExistingCapabilityConfigurationRealizationRequest.risk_classification  (copy only)
        ↓
project_control_plane_mutation_request(...)
        ↓
ControlPlaneMutationAuthorizationPort
        ↓
Governance decision
```

Autonomous Work orchestrates only: select opportunity ref → `read_exact` → carry fields — **must not** classify, override, or lower `ControlPlaneMutationRisk`; **must not** translate `A0`/`A1`/… into `LOW`/`MEDIUM`/…; **must not** call Governance risk logic.

`WorkerCapabilityCandidate.risk_class` remains AW acquisition semantics only — **no** projection to `risk_classification`.

**Risk policy inputs (typed only):** `ExistingCapabilityConfigurationOpportunity` or a narrow projection (`integration_category`, canonical `provider_id`, `resource_scope`, configuration type/version, pinned `current_revision`, configuration fingerprint, platform-defined typed provider/configuration risk metadata if present). **Forbidden inputs:** `WorkerAutonomyLevel`, free-form metadata, `dict[str, Any]`, provider class name, opaque `configuration_ref` parsing, reflection, arbitrary string rules.

**Fail closed:** risk cannot be classified → opportunity is not executable / CONFIGURE_EXISTING cannot build a realization request. No silent `unknown → LOW` or `missing classifier → LOW`. Conceptual failures (exact enum names optional until P1): `CONFIGURATION_MUTATION_RISK_UNAVAILABLE` · `CONFIGURATION_MUTATION_RISK_AMBIGUOUS` · `CONFIGURATION_MUTATION_RISK_INVALID`.

**Immutability:** one `configuration_ref` → one opportunity → one `risk_classification`; policy changes require new opportunity/ref or invalidation — no reinterpretation of historical refs.

**Continuity invariant:**

```text
opportunity.risk_classification
== realization_request.risk_classification
== ControlPlaneMutationRequest.risk_classification
== authorization evidence risk (digest-bound)
```

**Provider plugin boundary:** `ExistingCapabilityConfigurationOpportunityProvider` may supply provider-specific **configuration facts**; it **must not** be final authority over `ControlPlaneMutationRisk`. Provider-supplied risk hints may increase context or raise floor later; they **cannot** unilaterally lower Integrations platform classification. No “lowest risk wins” among multiple classifiers — exactly one Integrations composition owner per opportunity creation.

**No second Governance engine:** Integrations mutation-risk policy outputs **only** `ControlPlaneMutationRisk`, not `ALLOW` / `DENY` / `REQUIRE_HUMAN` / `ESCALATE` / `MODIFY`.

**Reuse:** single cross-domain enum `ControlPlaneMutationRisk` — do not introduce parallel `ConfigurationRisk` / `IntegrationRisk` vocabularies.

### 1C.14 Tenant continuity (opportunity seam)

```text
WorkerCapabilityFulfillmentRequest.tenant_id
== read_exact tenant argument
== ExistingCapabilityConfigurationOpportunity.tenant_id
== ExistingCapabilityConfigurationRealizationRequest.tenant_id
== ConfiguredCapabilityBinding.tenant_id
== adoption tenant
== Execution tenant
== provenance tenant
```

Tenant mismatch on read → fail closed (`CONFIGURATION_OPPORTUNITY_TENANT_MISMATCH`).

### 1C.15 Integrity invariants (pre-request)

Before constructing `ExistingCapabilityConfigurationRealizationRequest`:

```text
opportunity.configuration_fingerprint
    == opportunity.configuration.configuration_fingerprint
```

Mismatch → `CONFIGURATION_OPPORTUNITY_FINGERPRINT_MISMATCH` (fail closed). `current_revision` is **pinned** on the opportunity; fulfillment **must not** fetch “latest revision” separately (stale → INT-CONFIG conflict / typed stale failure — **no auto-refresh**).

### 1C.16 Idempotency

`configuration_ref` identifies the **semantic opportunity**; INT-CONFIG `request_id` identifies one **realization operation** — do not conflate. Repeat realize follows existing INT-CONFIG idempotency only.

### 1C.17 `MappingConfigurationOpportunityDiscoveryAdapter` classification (@ `305ac4a0…`)

| Adapter | Production callers | Classification |
|---|---|---|
| `MappingConfigurationOpportunityDiscoveryAdapter` | **0** (`intergrax/`, `agents/`, `applications/` production) | **TEST / FIXTURE / NON-PRODUCTION** |
| `NotConfiguredConfigurationOpportunityDiscovery` | Host wiring uses via tests/scaffold only in repo inventory | **FALLBACK / TEST WIRING** |
| `UnavailableConfigurationOpportunityDiscovery` | Tests only | **FALLBACK / TEST WIRING** |

Unit/integration tests: `tests/unit/autonomous_work/test_worker_capability_acquisition.py` (mapping adapter), `test_capability_catalog_discovery_adapters.py`, `test_uca6b_r_worker_capability_recovery.py`, `tests/integration/autonomous_work/test_aw_7b_ephemeral_capability_execution.py`.

**Must not** become production authority by wiring a runtime dict. Future production discovery **projects** from Integrations opportunity source only.

### 1C.18 Closed-world inventory — configuration opportunity surfaces (@ `305ac4a0…`)

| Symbol / surface | Path | Production | Role |
|---|---|---|---|
| `configuration_ref` field | `intergrax/contracts/autonomous_work/capability_acquisition.py` (`WorkerCapabilityCandidate`) | Contract only | Optional opaque ref on `EXISTING_CONFIGURATION` candidates |
| `configuration_ref` producers | — | **0** production writers | Only tests construct candidates with refs |
| `WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION` | `capability_acquisition.py` | Contract | Kind for CONFIGURE_EXISTING |
| Kind emission | `WorkerCapabilityAcquisitionDecisionService` | **Conditional** | Emits disposition when discovery supplies kind |
| `WorkerConfigurationOpportunityDiscoveryPort` | `capability_acquisition_ports.py` | Port | Implemented by host-composed adapters |
| `WorkerCapabilityAcquisitionService` | `capability_acquisition_service.py` | **Production** | Invokes configuration discovery layer |
| Production composition of configuration discovery | repo inventory | **0** | No production module wires `MappingConfigurationOpportunityDiscoveryAdapter` |
| `MappingConfigurationOpportunityDiscoveryAdapter` | `capability_discovery_adapters.py` | **NON-PRODUCTION** | Test/fixture dict lookup by operations |
| `ExistingCapabilityConfigurationRealizationRequest` | `integrations/contracts/existing_capability_configuration.py` | Contract | Full typed realize admission |
| `ExistingCapabilityConfigurationRealizationPort` / façade | `existing_capability_configuration_facade.py` | **0** callers | Tests + qualification |
| `ExistingCapabilityIntegrationResolver` | contracts + service | **Tests/qualification fakes only** | Target resolution **after** request exists |
| `ExistingCapabilityConfigurationRealizationStrategy` | sqlite `configuration_realization.py` | Reference provider | `SQLiteRelationalStoreConfigurationRealizationStrategy` |
| `IntegrationConfigurationPayload` | `existing_capability_configuration.py` (Protocol) | Contract | Provider payloads e.g. `SQLiteRelationalStoreConfigurationPayload` |
| Catalog/provider registration | Integration registry / catalog | **Production** | Separate from opportunity — reuse identities only |

### 1C.19 INT-CONFIG request field projection matrix (locked — no guessing)

| Request field | Source |
|---|---|
| `request_id` | Deterministic fulfillment operation identity (AW) |
| `tenant_id` | `WorkerCapabilityFulfillmentRequest.tenant_id` |
| `principal` | Governed worker principal binding (Applications-wired repository; AW read) |
| `integration_category` | `ExistingCapabilityConfigurationOpportunity` |
| `provider_id` | `ExistingCapabilityConfigurationOpportunity` |
| `resource_scope` | `ExistingCapabilityConfigurationOpportunity` |
| `configuration` | `ExistingCapabilityConfigurationOpportunity.configuration` |
| `configuration_fingerprint` | `ExistingCapabilityConfigurationOpportunity` (≡ payload fingerprint) |
| `current_revision` | `ExistingCapabilityConfigurationOpportunity` (pinned) |
| `risk_classification` | `ExistingCapabilityConfigurationOpportunity.risk_classification` (Integrations-owned; §1C.13) |
| `task_id` / `run_id` | `WorkerCapabilityFulfillmentRequest` |
| `correlation_ref` | Recovery provenance (`discovery_correlation_id` family) |

### 1C.20 Typed failure families (future)

`CONFIGURATION_OPPORTUNITY_NOT_FOUND` · `CONFIGURATION_OPPORTUNITY_TENANT_MISMATCH` · `CONFIGURATION_OPPORTUNITY_STALE` · `CONFIGURATION_OPPORTUNITY_AMBIGUOUS` · `CONFIGURATION_OPPORTUNITY_INVALID` · `CONFIGURATION_OPPORTUNITY_FINGERPRINT_MISMATCH` · `CONFIGURATION_OPPORTUNITY_PROVIDER_UNAVAILABLE` — exact enum names optional; semantics mandatory; all fail closed.

### 1C.21 Before / after graph (opportunity seam)

**Before (@ `305ac4a0…`):**

```text
WorkerCapabilityCandidate.configuration_ref (opaque, untyped)
        X  (no Integrations exact read)
ExistingCapabilityConfigurationRealizationRequest
        X  (cannot be built safely from candidate alone)
```

**After (locked design):**

```text
Integrations Configuration Opportunity owner
        ↓ exact typed opportunity
AW discovery projection
        ↓ opaque configuration_ref only
WorkerCapabilityCandidate
        ↓ CONFIGURE_EXISTING
WorkerConfiguredCapabilityFulfillmentService
        ↓ read_exact(tenant_id, configuration_ref)
ExistingCapabilityConfigurationOpportunityReadPort
        ↓ immutable opportunity
ExistingCapabilityConfigurationRealizationRequest
        ↓ … (§1B.6 / §3)
```

### 1C.22 Future qualification gates (R2-P5 — design)

1. Production opportunities originate from exactly one Integrations-owned mechanism.
2. AW candidate carries only exact opaque ref (no full config copy).
3. No string parsing / prefix / regex / JSON-in-ref for provider semantics in AW.
4. No current/latest opportunity lookup.
5. No second provider registry.
6. Tenant-scoped `read_exact` mandatory.
7. Stale / fingerprint / revision violations fail closed.
8. Unknown / ambiguous ref fail closed.
9. `MappingConfigurationOpportunityDiscoveryAdapter` absent from production composition.
10. Opportunity ≠ configured binding ≠ effective provider.
11. Same binding continues through adoption unchanged.
12. Provider plugin opportunity mechanism replaceable.

### 1C.23 P0-R1-R1-R1 STOP disposition

**No STOP — ARCHITECTURE DECISION REQUIRED** for locked scope. Opportunity contracts + read port + discovery projection are **P1**; durable store **P2**; production wiring **P3**.

### 1C.24 P0-R1-R1-R1 exit questionnaire (locked)

| # | Answer |
|---|---|
| 1 | **Configuration Opportunity owner?** Integrations |
| 2 | **Who creates opportunities?** Integrations opportunity owner + `ExistingCapabilityConfigurationOpportunityProvider` plugins (§1C.12) |
| 3 | **Typed fields?** §1C.2 |
| 4 | **`configuration_ref` identifies?** Exactly one immutable tenant-scoped opportunity — not “latest for provider” |
| 5 | **Ref immutable + tenant-scoped?** **Yes** |
| 6 | **AW discovery obtains refs?** Projection adapter over Integrations opportunity source (§1C.5) |
| 7 | **Fulfillment resolves ref?** `read_exact(tenant_id, configuration_ref)` (§1C.6) |
| 8 | **Same canonical source for discovery + fulfillment?** **Yes** |
| 9 | **provider/category/scope/payload/fingerprint/revision source?** Exact opportunity read (§1C.19) |
| 10 | **AW parses provider semantics?** **NO** |
| 11 | **Latest/current lookup?** **NO** |
| 12 | **Second Integration Catalog?** **NO** |
| 13 | **Opportunity ≠ configured binding?** **YES** |
| 14 | **Configured binding ≠ effective identity?** **YES** |
| 15 | **Stale refs silently refresh?** **NO** |
| 16 | **Provider pluginability preserved?** **YES** (§1C.11) |
| 17 | **AW derives mutation risk from A0?** **NO** — Integrations opportunity only (§1C.13) |
| 18 | **Tenant continuity end-to-end?** **YES** (§1C.14) |

### 1C.25 Implementation sequencing (post R1-R1-R1 audit)

```text
P0-R1-R1-R1  opportunity typing lock (this revision)
      ↓
P0           parent architecture closure / bookkeeping
      ↓
P1           opportunity + adoption/provenance contracts + validation
      ↓
P2           durable opportunity/provenance storage where locked
      ↓
P3           production discovery + fulfillment + adoption wiring
      ↓
P4           reconstruction
      ↓
P5           closed-world qualification
      ↓
CERT
```

**Do not enter P1 in R1-R1-R1 task.**

### 1C.26 Closed-world `ControlPlaneMutationRequest` producer inventory (@ `925bcae…`)

Production request builders (tests excluded). **Governance classifier?** = whether the shared boundary derives `risk_classification` from domain semantics (expected **NO** everywhere).

| Domain | Mutation (representative) | Risk source | Constant / derived | Owner | Governance classifier? |
|---|---|---|---|---|---|
| Task control | cancel / resume / autonomy (`intergrax/applications/_shared/task_control_governance.py`) | Domain builder | `MEDIUM` (constant per mutation builder) | Applications / task control | **NO** |
| ECP capacity | scale up/down (`intergrax/runtime/capacity/control_plane_governance.py`) | Domain builder | `MEDIUM` (constant) | Runtime / capacity | **NO** |
| Adaptive (AHI) | control-plane mutations (`intergrax/runtime/adaptive/control_plane_governance.py`) | Domain builder | `MEDIUM` / `HIGH` (per mutation kind) | Runtime / adaptive | **NO** |
| Integration catalog | hot reload (`intergrax/applications/_shared/catalog_hot_reload_governance.py`) | Domain builder | `HIGH` (constant) | Applications / catalog | **NO** |
| Vector index admin | prepare (`intergrax/applications/_shared/vector_index_admin_governance.py`) | Domain builder | `HIGH` (constant) | Applications / vector index | **NO** |
| Agent Distribution | admin mutations (`intergrax/agent_distribution/control_plane_governance.py`) | Domain builder | `HIGH` (constant per builder) | Agent Distribution | **NO** |
| INT-CONFIG | `integration_configuration.realize.v1` (`project_control_plane_mutation_request` in `intergrax/integrations/contracts/existing_capability_configuration.py`) | Realization request field | **Derived @ P1** from `ExistingCapabilityConfigurationOpportunity.risk_classification` (Integrations policy); today callers/tests supply typed request | **Integrations** (classification owner) | **NO** |

Evaluation path (all domains): `ControlPlaneMutationAuthorizationBoundary` (`intergrax/runtime/governance/control_plane_mutation_authorization.py`) + `ControlPlaneMutationPolicyEvaluator` (`intergrax/runtime/governance/control_plane_mutation_policy.py`) consume `request.risk_classification` — they do **not** invent Integrations configuration-domain risk.

### 1C.27 Integrations mutation-risk policy vs Governance policy

| Policy | Owner | Output |
|---|---|---|
| **Integrations mutation-risk policy** (`ExistingCapabilityConfigurationMutationRiskPolicy` — future) | Integrations / INT-CONFIG | `ControlPlaneMutationRisk` for the typed configuration mutation candidate |
| **Governance control-plane policy** | Governance | `PolicyAction` for principal / tenant / resource / revisions / **supplied** risk / bundle |

Complementary authorities — not duplicate engines.

### 1C.28 Tenant Isolation Audit (R1-R1-R1-R1)

**Verdict: PASS** (local invariant design; global `FRZ-TEN-*` remains **OPEN**).

```text
opportunity.tenant_id
== read_exact tenant argument
== risk-classification input tenant scope
== ExistingCapabilityConfigurationRealizationRequest.tenant_id
== ControlPlaneMutationRequest.principal.tenant_id
== authorization evidence.tenant_id
== ConfiguredCapabilityBinding.tenant_id
```

Risk classification **must not** perform cross-tenant lookup or reuse opportunity from tenant A for tenant B.

### 1C.29 Governance audit (R1-R1-R1-R1)

| Proposition | Locked |
|---|---|
| `risk_classification` ≠ permission | **YES** — risk is policy **input**; `PolicyAction` is separate |
| `CONFIGURE_EXISTING` ≠ permission | **YES** — acquisition disposition only |
| Configuration Opportunity ≠ permission | **YES** — input truth only |
| Authorization invocation owner | **`ControlPlaneMutationAuthorizationPort` / boundary only** |

### 1C.30 Before / after ownership graph (risk seam)

**Before (rejected @ R1-R1-R1-R1 audit):**

```text
CONFIGURE_EXISTING + A0_KNOWN_CAPABILITY
        → (implied) ControlPlaneMutationRisk.LOW
        → realization request
```

**After (locked):**

```text
ExistingCapabilityConfigurationOpportunity
        (risk_classification from Integrations mutation-risk policy)
        ↓
AW: opaque ref + orchestration only
        ↓
ExistingCapabilityConfigurationRealizationRequest.risk_classification  (copy)
        ↓
project_control_plane_mutation_request
        ↓
Governance authorize
```

### 1C.31 P0-R1-R1-R1-R1 exit questionnaire (locked)

| # | Answer |
|---|---|
| 1 | **Who owns configuration mutation risk classification?** Integrations / INT-CONFIG |
| 2 | **Does AW classify it?** **NO** |
| 3 | **Does `A0 → LOW` remain?** **NO** |
| 4 | **Does Governance authorize using the classification?** **YES** |
| 5 | **Must Governance classify Integrations domain semantics?** **NO** (evidence: §1C.26) |
| 6 | **Where is final `ControlPlaneMutationRisk` stored before realization?** On immutable `ExistingCapabilityConfigurationOpportunity.risk_classification` |
| 7 | **Who creates it?** Integrations opportunity owner via `ExistingCapabilityConfigurationMutationRiskPolicy` |
| 8 | **Can provider plugin lower it directly?** **NO** |
| 9 | **Undetermined risk?** Fail closed — no realization request |
| 10 | **Immutable per opportunity ref?** **YES** |
| 11 | **Same risk through realization → control-plane → evidence?** **YES** (§1C.13 continuity) |
| 12 | **Exactly one classification owner?** **YES** — Integrations composition owner |
| 13 | **Second Governance engine for classification?** **NO** |
| 14 | **Second provider/configuration registry for risk?** **NO** |
| 15 | **Tenant continuity explicit?** **YES** (§1C.28) |
| 16 | **Global control-plane risk architecture unchanged?** **YES** — reuse `ControlPlaneMutationRisk`; other domains unchanged |

### 1C.32 P0-R1-R1-R1-R1 STOP disposition

**No STOP — ARCHITECTURE DECISION REQUIRED** for locked scope. Production @ `925bcae…` matches domain-supplied risk + Governance evaluation; INT-CONFIG opportunity risk field + Integrations policy are **P1** only.

### 1C.33 Future P1 contracts (shape only — no implementation in R1-R1-R1-R1)

- `ExistingCapabilityConfigurationMutationRiskPolicy` — typed opportunity/mutation facts → `ControlPlaneMutationRisk`
- `ExistingCapabilityConfigurationOpportunity.risk_classification: ControlPlaneMutationRisk` — immutable, Integrations-produced

Placement follows existing Integrations contracts organization (`intergrax/integrations/contracts/…`).

---

## 2. Before graph (@ P0-R1-R1 AUDIT_BASE `6b5b5f2e…`)

```text
AW CONFIGURE_EXISTING decision (WorkerCapabilityAcquisitionDecisionService)
        X  (no production fulfillment consumer)
INT-CONFIG facade (0 production callers)
        ↓
ConfiguredCapabilityBinding
        X  (no production consumer)
        │
        │   parallel universe
        ▼
IntegrationProfile + resolve_from_profile / resolve (bootstrap / effective-only)
        │
        ▼
effective provider instance  ──X──►  ExecutionId (no config provenance join)
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

## 3. Proposed after graph (locked design — not implemented)

```text
WorkerCapabilityAcquisitionDecisionService  →  CONFIGURE_EXISTING
        │
        ▼
WorkerCapabilityFulfillmentCoordinator  (CONFIGURE_EXISTING branch)
        │
        ▼
Integrations Configuration Opportunity source  (P3)
        │
        ▼
WorkerConfiguredCapabilityFulfillmentService
        │  read_exact(tenant_id, configuration_ref)  (P1 port)
        │  → ExistingCapabilityConfigurationOpportunity
        ▼
ExistingCapabilityConfigurationRealizationPort.realize
        │
        ▼
ConfiguredCapabilityBinding (immutable)
        │
        ▼
ExecutionIntegrationConfigurationAdoption  (built by configured fulfillment service)
        │
        ▼
ExecutionBoundIntegrationResolution  (Integrations — no configured lookup)
        │
        ▼
resolve_from_profile / resolve  →  EffectiveIntegrationIdentity (independent provider_id)
        │
        ├── provenance pin (tenant + ExecutionId + binding + effective)
        │
        ▼
WorkerQualifiedCapabilityResumeCoordinator / governed dispatch  →  canonical Execution
        │
        ▼
ExecutionIntegrationConfigurationProvenance → store → ExecutionReconstructor
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
| `IntegrationConfigurationSubject` | Deterministic subject key: `integration_category`, `provider_id`, `resource_scope`, `configuration_type` — **`provider_id` from §1A.3, never `category.value`** |
| `ExecutionIntegrationConfigurationAdoption` | Exactly one immutable `ConfiguredCapabilityBinding` + minimum subject context; Integrations contracts owner |
| `ConfiguredIntegrationProvenanceSlice` | Full configured identity dimensions + `configuration_fingerprint` (not effective proof); optional `realization_evidence_refs` |
| `EffectiveIntegrationIdentity` / materialization slice | `integration_category`, `provider_id`, `materialization_kind` ∈ `{CATALOG_FACTORY, PROFILE_PREBUILT}` per §1A.3 |
| `ExecutionIntegrationConfigurationProvenance` | `tenant_id`, `execution_id`, `mode` (`CONFIGURED_ADOPTED` \| `EFFECTIVE_ONLY`), `effective`, optional `configured` — **immutable** (§1A.6) |
| `ExecutionIntegrationConfigurationProvenanceReadStatus` | Typed read outcomes; `required_missing` fail-closed for configured-required paths |
| `ExecutionIntegrationConfigurationProvenanceReader` | `read_all(tenant_id, execution_id) -> tuple[...]` and/or `read_one(..., subject)` |

**Effective `provider_id`** is derived only from §1A.3 canonical sources — **not** `category.value`, class names, `discover()` on DI-only paths, options dicts, or `RuntimeConfig`.

**Configured fingerprint ≠ effective proof:** both slices mandatory on INT-CONFIG adoption paths; effective slice required on all configuration-aware execution paths.

### 10.2 Capture / write point (future)

| Element | Locked choice |
|---|---|
| **Owner** | Integrations: `ExecutionBoundIntegrationResolution` (new module) wrapping **only** existing `resolve_from_profile` / `resolve` |
| **Orchestration sequencer** | `WorkerConfiguredCapabilityFulfillmentService` — builds and passes **explicit** `ExecutionIntegrationConfigurationAdoption` into `ExecutionBoundIntegrationResolution` |
| **Composition invoker** | `build_worker_recovery_governed_fulfillment_wiring` / `worker_capability_fulfillment_composition` — **wires** realization port + configured fulfillment + execution-bound resolver (no semantic ownership) |
| **Timing** | Immediately after successful sanctioned materialization, **with** `tenant_id` + `ExecutionId` + optional explicit adoption — **before** downstream work relies on the instance where feasible |
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
WorkerCapabilityFulfillmentRequest.tenant_id
== opportunity read tenant
== ExistingCapabilityConfigurationOpportunity.tenant_id
== configured_binding.tenant_id
== composition tenant
== execution tenant (ExecutionId context)
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
| No stable effective identity | **Addressed in P0-R1** — canonical `provider_id` per §1A.3 (P0 slug+category model **rejected**) |
| Second semantic truth store | **Avoided** — provenance store is evidence-only, R1-isomorphic |
| Child propagation needs new execution authority | **Deferred** — no inheritance in v1 lock |
| Tenant absent at capture | **Addressed** — mandatory in API |
| Neutral contract leaks provider instances | **Forbidden** by design |
| Governance semantics change | **Not required** |

**P0-R1:** no STOP — ARCHITECTURE DECISION REQUIRED for locked scope. **ExternalWork `CONFIGURED_ADOPTED`** deferred via C1 (contract change would be separate child).

---

## 18. Expected implementation file map (future waves)

| File | Role |
|---|---|
| `intergrax/contracts/integration_configuration_subject.py` | Subject key |
| `intergrax/integrations/contracts/execution_integration_configuration_adoption.py` (or equivalent) | Explicit adoption input |
| `intergrax/contracts/execution_integration_configuration_provenance.py` | Neutral DTOs + reader Protocol + provenance `mode` |
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
| `intergrax/integrations/contracts/existing_capability_configuration_opportunity.py` | Opportunity DTO + `ExistingCapabilityConfigurationOpportunityReadPort` |
| `intergrax/integrations/providers/*/configuration_opportunity.py` (illustrative) | Provider plugin opportunity emission |
| `intergrax/autonomous_work/integrations_configuration_opportunity_discovery_adapter.py` | AW discovery projection (opaque refs only) |

**Explicit non-goals in implementation:** modify `ConfiguredCapabilityBinding`, `IntegrationProfile`, `resolve_from_profile` semantics, Governance, or Execution identity minting.

---

## 19. Closed-world qualification strategy (future gates)

Independent discovery + registry parity (seed **not** from registry), minimum gates:

1. All configured-binding production producers known.
2. All configured-binding production consumers known (**expect zero** outside INT-CONFIG @ baseline; post-P3 → exactly sanctioned adoption flow).
3. All sanctioned effective resolution/materialization paths known.
4. Every resolver/materialization **production caller** classified (§1A.7 disposition).
5. Every `EXECUTION_CONFIGURED_ADOPTION_REQUIRED` caller routes through wrapper; direct `resolve` / `resolve_from_profile` bypass = 0.
6. Pre-built effective identity = canonical `provider_id`, not category fallback.
7. Category fallback cannot pass as provider ID.
8. Configured/effective `provider_id` mismatch fails; missing adoption cannot downgrade to `EFFECTIVE_ONLY`.
9. `EFFECTIVE_ONLY` records cannot count as FRZ-TRC-11 configured/effective closure.
10. Adoption input explicit from composition; no current/latest configured binding access.
11. No second provider resolver; no second activation lifecycle.
12. `ExternalWorkIntegration` / DI-only classification explicit (C1).
13. Unknown pre-built type without canonical `provider_id` → fail closed.
14. Unknown production effective path / unknown adoption path → **FAIL**.
15. Heuristic join / timestamp / latest / diagnostics-as-authority = 0.
16. Reconstruction imports no Integrations implementation.
17. Tenant / execution / subject / resource_scope mismatch → fail closed.
18. Multiplicity deterministic; child rules per §12.
19. R1 policy/profile provenance unchanged (regression gates).
20. Adversarial: circular pre-built proof + catalog mismatch negatives (§1A.8).
21. Configuration opportunities: single Integrations source; AW opaque ref only (§1C.22).
22. No `configuration_ref` parsing in AW; no latest opportunity lookup.
23. `MappingConfigurationOpportunityDiscoveryAdapter` production composition = 0.
24. Opportunity ≠ binding ≠ effective identity preserved through adoption.

---

## 20. Implementation wave decomposition

| Wave | Deliverable |
|---|---|
| **P0-R1** (§1A) | Effective identity + configured adoption reconciliation |
| **P0-R1-R1** (§1B) | Concrete CONFIGURE_EXISTING adoption root + AW fulfillment orchestration lock |
| **P0-R1-R1-R1** (§1C) | Configuration opportunity typing + exact-reference read lock |
| **P0-R1-R1-R1-R1** (§1C.26–§1C.31) | Configuration mutation risk classification ownership |
| **P0-FINAL** | Final normative reconciliation + canonical tracker sync (**READY FOR AUDIT**) |
| **P0** | Architecture lock complete after **P0-FINAL** independent audit → **P1** |
| **P1** | Typed opportunity + risk policy + adoption/provenance contracts + validation |
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

## 22. Enterprise audit matrix (@ P0-R1-R1-R1-R1 lock)

| Dimension | Grade | Notes |
|---|---|---|
| Boundaries | **PASS** | Tier rules preserved; reconstructor stays derived |
| Semantic ownership | **PASS** | §8 + §1A.4 + §1C.4 |
| Configuration mutation risk ownership | **PASS** | Integrations classifies; Governance authorizes (§1C.13, §1C.26–§1C.27) |
| Governance ownership (permission) | **PASS** | `ControlPlaneMutationAuthorizationBoundary` only; no domain risk derivation |
| AW boundary (no risk classification) | **PASS** | §1C.13; `risk_class` ≠ `risk_classification` |
| Provider/plugin risk boundary | **PASS** | Facts only; no plugin-final LOW (§1C.13) |
| Fail-closed classification | **PASS** | No silent LOW; undetermined → no realize |
| Immutable opportunity risk | **PASS** | §1C.13 immutability + continuity invariant |
| Policy/evidence continuity | **PASS** | opportunity → request → mutation → evidence |
| Exactly-one risk classifier | **PASS** | Integrations composition owner |
| Configuration opportunity ownership | **PASS** | Integrations-only typed opportunity + read port (§1C) |
| Catalog ownership | **PASS** | No second registry; reuse catalog identities (§1C.8) |
| Discovery ownership | **PASS** | Integrations source → AW projection; mapping adapter non-production (§1C.17) |
| Composition ownership | **PASS** | Applications wire ports; AW fulfillment sequences; Integrations validate/capture (§1B) |
| Orchestration ownership | **PASS** | `WorkerCapabilityFulfillmentCoordinator` + configured fulfillment service (§1B.6) |
| Provider identity authority | **PASS** | §1A.3; P0 category/slug model rejected |
| Configured adoption authority | **PASS** | Explicit handoff; no lookup |
| Contracts | **PASS** | Adoption + provenance `mode` + opportunity read (§1C.3) |
| Strong typing | **PASS** | No AW provider payload; `IntegrationConfigurationPayload` Protocol preserved |
| Pluginability | **PASS** | Catalog tri-equality; opportunity provider contract (§1C.11) |
| Replaceability | **PASS** | Single factory; wrapper delegates; replaceable opportunity providers |
| Configured/effective separation | **PASS** | Opportunity ≠ binding ≠ effective (§1C.7) |
| Revision/fingerprint integrity | **PASS** | Pinned revision; fingerprint invariant (§1C.15) |
| Governance separation | **PASS** | Opportunity excludes principal; Governance authorizes realize |
| Execution authority | **PASS** | ExecutionId unchanged |
| Evidence non-authority | **PASS** | Readers/projections only; opportunity not Evidence SSOT |
| Tenant continuity | **PASS** | §13 + §1A.5 + §1C.14 |
| Historical reconstruction suitability | **PASS** | Exact immutable refs; no latest lookup |
| Fail-closed behavior | **PASS** | §1C.20 + §1C.15 |
| Persistence suitability | **PASS** | Durable pin; Integrations-owned opportunity store optional P2 |
| Bypass resistance | **PARTIAL** | Until P3 — wrapper + opportunity wiring |
| Child semantics | **PASS** | §12 unchanged |
| Closed-world qualification design | **PASS** | §19 + §1A.7 + §1C.22 |
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

## 24. Current-HEAD test evidence (P0-FINAL)

Commands (sequential `-p no:xdist`):

```text
uv run pytest \
  tests/qualification/trace_x/test_trace_x_p5_p0_qualification_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r1_qualification_gates.py \
  -p no:xdist -q

uv run pytest tests/qualification/existing_capability_configuration -p no:xdist -q

uv run pytest \
  tests/unit/runtime/governance/test_control_plane_mutation_authorization.py \
  tests/unit/contracts/test_control_plane_mutation_contract.py \
  -p no:xdist -q
```

@ P0-FINAL (`33076146…` START_HEAD + qualification reconciliation commit):

```text
# trace P5 P0+R1 gates: 134 passed (339.65s)
# INT-CONFIG qualification: 58 passed (0.46s)
# control-plane authorization + contract: 27 passed (0.21s)
```

Logs: `.tmp/session/trace-x-p5-r2-p0-final/pytest-trace.log`, `pytest-int-config.log`, `pytest-governance.log`.

**ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED:** `tests/unit/integrations/test_registry.py` — fake factory drift (`dict` vs `PlatformIntegrationContract`); not in mandatory batch. `CapabilityQualificationEvidence` fixture drift in AW fulfillment e2e/hook suites — **not** expanded in R1-R1-R1 batch; classify separately unless production impact proven.

---

## 25. Roadmap status (recommended)

| Item | Status |
|---|---|
| TRACE-X-P5-R2-P0-FINAL | **READY FOR AUDIT** |
| TRACE-X-P5-R2-P0 | **READY FOR AUDIT** |
| TRACE-X-P5-R2-P0-R1-R1-R1-R1 | **independently accepted** @ `33076146…` |
| TRACE-X-P5-R2-P0-R1-R1-R1 | remediation evidence @ `925bcae…` (superseded) |
| TRACE-X-P5-R2-P0-R1-R1 | remediation evidence @ `305ac4a…` (superseded) |
| TRACE-X-P5-R2-P0-R1 | remediation evidence @ `6b5b5f2…` (superseded) |
| TRACE-X-P5-R2-P0 initial | **REJECTED** @ `982f945…` |
| TRACE-X-P5-R2 | **CURRENT / BLOCKED ON P0 FINAL INDEPENDENT AUDIT** |
| TRACE-X-P5 | **CURRENT / BLOCKED ON R2** |
| P5-GAP-04 | **ARCHITECTURALLY SPECIFIED / IMPLEMENTATION OPEN** |
| FRZ-TRC-11 | **OPEN** |
