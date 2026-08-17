# Platform Plugins — Provider qualification architecture (PROVIDER-QUAL-1)

**Status:** Architecture freeze + contract design — **READY_FOR_REVIEW**
**Parent hub:** [`PLATFORM_PLUGINS.md`](../PLATFORM_PLUGINS.md) §18
**Plan:** [`maintainers/plans/PLATFORM_PLUGINS.md`](../../maintainers/plans/PLATFORM_PLUGINS.md) — PROVIDER-QUAL track
**Decision (PROVIDER-QUAL-0):** **EXTEND_EXISTING** — no parallel `ProviderQualificationEngine` or new qualification domain

**Scope:** contract design and ownership freeze only. No runtime implementation, no CI vendor jobs, no LKW integration, no MP-2.

---

## 1. Ownership (FROZEN)

| Layer | Owns | Does not own |
|-------|------|--------------|
| **`intergrax/core/qualification/`** | Reusable qualification primitives: `QualificationStatus`, `QualificationEvidence`, status ordering helpers | Provider semantics, suite definitions, persistence, executor binding |
| **Platform Plugins / platform coordination** (`intergrax/core/plugins/platform_qualification.py` and future provider-scoped extensions) | Cross-domain qualification coordination; provider-scoped subject/run contracts; production admission **hooks** (pure gates, not policy engine) | Domain suite semantics; vendor adapter code; CI execution |
| **Domain program** (e.g. Collaborative Work, RAG, Integrations) | Semantic qualification requirements; suite identity and version; pass/fail thresholds; domain evidence kinds | Global provider registry; platform-wide admission policy |
| **Provider adapter** | Vendor-specific implementation, setup, environment prerequisites | Qualification outcome vocabulary; cross-domain coordination |
| **Execution environment** | Runs qualification workloads; records executor metadata | Qualification semantics; admission policy |
| **Proof / evidence** (`QualificationEvidence`, optional `ProofReceipt` persistence) | Immutable qualification **facts** and safe metadata refs | Public marketing claims |
| **Public Proof and Claims Model** | Allowed public claims derived from evidence | Qualification execution |

**Forbidden:** a parallel `ProviderQualificationEngine`, vendor enums in `intergrax/core/qualification/`, or qualification-core changes required to add Oracle/PostgreSQL/MySQL.

---

## 2. Architectural placement

```text
Domain suite (semantic tests, thresholds)
        |
        v
Provider adapter + execution environment --> ProviderQualificationRun (immutable)
        |                                              |
        |                                              +- QualificationStatus (outcome)
        |                                              +- QualificationEvidenceValidity (separate)
        |                                              +- QualificationEvidence[] (safe refs)
        v
Platform coordination --> production admission hooks (outcome AND validity AND compatibility AND domain policy)
        |
        v
Optional ProofReceipt persistence --> public claims index (PUBLIC_PROOF_AND_CLAIMS_MODEL)
```

Reuses existing PLUGIN-7 contracts (`PluginQualificationSubject`, `PluginQualificationResult`, `PluginQualificationEvidenceKind`) as the **package/capability** qualification path. Provider-scoped qualification **extends** the same primitives with additional subject dimensions and a dedicated run record — it does not fork a second engine.

---

## 3. Provider qualification subject (FROZEN)

`ProviderQualificationSubject` is the canonical identity for **one provider capability in one environment under one suite version**. It extends/reuses `PluginQualificationSubject` field patterns (`domain`, `capability_id`, optional package identity) with provider-scoped dimensions.

### 3.1 Required dimensions

| Field | Type | Semantics |
|-------|------|-----------|
| `provider_id` | `str` | Data-driven vendor/backend identity (e.g. `postgresql`, `oracle`). **Never** a Python/Tier-0 enum. |
| `provider_version` | `str` | Qualified backend/runtime version (e.g. `16.6`, `23ai`). |
| `capability_id` | `str` | Domain capability contract id (e.g. `collaborative_work.persistence.v1`). |
| `domain` | `str` | Owning domain program (e.g. `collaborative_work`, `rag`). |
| `adapter_identity` | `str \| None` | Adapter/package identity when applicable: distribution name, integration module path, or host registration path. |
| `intergrax_revision` | `str` | Intergrax source revision or tested platform version anchor used for the run. |
| `qualification_suite_id` | `str` | Domain-owned suite identity (e.g. `cw.postgresql.repository.v1`). |
| `qualification_suite_version` | `str` | Suite semantic version or content hash label. |
| `environment_id` | `str` | Bounded qualification environment label (e.g. `local-docker-qual-host`, `ci-reference-postgres-16`). |

### 3.2 Optional coordination fields (inherit PLUGIN-7 where relevant)

| Field | When |
|-------|------|
| `package_name` / `package_version` | External wheel/distribution delivers the adapter |
| `entry_point_group` / `entry_point_name` | PEP discovery path |
| `host_registration_path` | Host-embedded adapter |
| `delivery_source` | `external_package` \| `host_embedded_extension` |
| `integration_kind` | When qualification is scoped via Integrations category (see [`INTEGRATIONS.md`](../INTEGRATIONS.md)) |

### 3.3 Identity rule (capability scoping)

Qualification identity is always:

```text
provider_id + provider_version + capability_id + qualification_suite_id
+ qualification_suite_version + environment_id (+ adapter_identity when applicable)
```

A provider is **never globally qualified**. PostgreSQL `production_qualified` for `collaborative_work.persistence.v1` does **not** imply qualification for RAG, memory, events, other PostgreSQL capabilities, or other PostgreSQL versions.

### 3.4 Vendor extensibility invariant

Adding Oracle 23ai requires **only**:

- data (`provider_id=oracle`, `provider_version=23ai`, ...)
- Oracle adapter + environment setup
- domain suite execution against Oracle
- evidence records

It must **not** require changes to `QualificationStatus`, qualification core enums, CI core workflow, or CW semantic contracts.

---

## 4. Provider qualification run (FROZEN)

`ProviderQualificationRun` is the immutable audit record for one executed qualification attempt.

### 4.1 Required dimensions

| Field | Semantics |
|-------|-----------|
| `qualification_run_id` | Stable unique run id **created at qualification run creation/execution time** (UUID or deterministic idempotency key). Optional persistence **preserves** this identity; it must not assign or replace the authoritative run id. |
| `subject` | `ProviderQualificationSubject` (section 3). |
| `status` | `QualificationStatus` — **historical outcome only** (section 5.1). |
| `executed_at` | UTC timestamp of run completion. |
| `executor` | Executor metadata (section 4.2) — not bound to GitHub Actions. |
| `result_summary` | Human-readable summary + structured counts (`passed`, `failed`, `skipped`, ...). |
| `evidence` | `tuple[QualificationEvidence[...], ...]` — safe metadata refs (section 7). |
| `reproducibility` | Reference command or documented reproduction path where safe (no secrets). |
| `limitations` | Explicit scope limits (capability, version, environment, mocks, substitution). |
| `source_revision` | Git SHA or equivalent source anchor for qualification harness + adapter code. |
| `environment_metadata` | Optional bounded facts: `real_backend`, `mocks`, `sqlite_substitution`, host labels. |

### 4.2 Executor metadata (executor-neutral)

| Field | Semantics |
|-------|-----------|
| `executor_kind` | `str` — e.g. `local_cli`, `ci_runner`, `operator_workstation`, `scheduled_qual_host`. |
| `executor_id` | `str` — stable label for the executor instance or job template (not vendor-specific). |
| `executor_version` | `str \| None` | Optional runner/harness version. |

GitHub Actions is **one possible** `executor_kind=ci_runner` implementation. Qualification contracts must not import GHA types, workflow names, or job ids as required fields.

### 4.3 Immutability

Once recorded, `status`, `subject`, `result_summary`, and `evidence` refs for a given `qualification_run_id` are **immutable**. **`validity` is not a field of `ProviderQualificationRun`.** Later drift is modeled only via separate `QualificationEvidenceValidity` evaluation (section 5.2), superseding runs, or explicit revocation records — never by rewriting historical outcome or mutating the run to change validity.

### 4.4 Run identity lifecycle (FROZEN)

```text
qualification trigger
        |
        v
qualification_run_id created (execution-owned identity)
        |
        v
ProviderQualificationRun produced (immutable historical fact)
        |
        v
optional ProofReceipt / index persistence (preserves qualification_run_id)
```

**Why execution-owned run identity matters:**

- stable audit lineage before and after optional persistence
- evidence correlation across executor, harness, and external systems prior to storage
- external executors can emit a stable run id without waiting for persistence
- retry/idempotency handling keyed to the same run identity
- optional persistence without identity replacement
- `ProofReceipt` projection references the run; it does not become the authoritative identity source

---

## 5. Status model (FROZEN)

Three **separate** dimensions. Do not collapse them into `QualificationStatus`.

### 5.1 Qualification outcome — `QualificationStatus`

Canonical vocabulary in `intergrax/core/qualification/status.py`:

| Value | Meaning |
|-------|---------|
| `NOT_QUALIFIED` | No acceptable evidence for required threshold. |
| `QUALIFIED` | Domain threshold met; not necessarily production admission. |
| `PRODUCTION_QUALIFIED` | Domain threshold for production reliance met at execution time. |
| `REJECTED` | Qualification attempted and failed/rejected. |

**Forbidden in `QualificationStatus`:** `STALE`, `REVOKED`, `INCOMPATIBLE`.

Historical example that **must** be representable:

```text
Run A (immutable ProviderQualificationRun):
  qualification_run_id = A
  status = PRODUCTION_QUALIFIED   # historical outcome at execution time

Validity evaluation T1 (separate view/record):
  qualification_run_id = A
  validity = CURRENT

Validity evaluation T2 (append-only superseding evaluation):
  qualification_run_id = A
  validity = STALE
  reason = adapter_revision_changed

Run A remains unchanged.
```

### 5.2 Evidence validity — `QualificationEvidenceValidity` (NEW contract, separate enum)

| Value | Meaning |
|-------|---------|
| `CURRENT` | Evidence still satisfies freshness/policy for admission evaluation. |
| `STALE` | Underlying subject dimensions or platform/domain drift invalidated freshness; outcome history preserved. |
| `REVOKED` | Evidence explicitly withdrawn (security, fraud, operator action). |

Validity is **not** a mutable field of `ProviderQualificationRun`. It is the current admission interpretation associated with a run/evidence set.

| Concept | Role |
|---------|------|
| `QualificationValidityRecord` / `QualificationAdmissionView` | Current or superseding validity evaluation referencing `qualification_run_id`; may include `validity`, `reason`, `evaluated_at`, and policy/context anchors as needed |
| Derived latest view | Admission reads the latest validity evaluation for a run; historical evaluations remain addressable |

Do **not** mutate historical `ProviderQualificationRun` to change validity.

**Lifecycle invariant:** a run that passed remains historically `PRODUCTION_QUALIFIED` even when current validity becomes `STALE` or `REVOKED`.

**Persistence shape (not frozen in R1):** architecture permits either (A) append-only validity evaluation records with a derived latest view, or (B) a derived current admission/validity projection from immutable evidence and current policy/drift inputs. Do not design mutable overwrite semantics for qualification history.

### 5.3 Compatibility evaluation (existing, separate)

Platform compatibility (`PlatformCompatibilityResult`, PLUGIN-6) and domain compatibility checks remain **separate** from qualification outcome. `INCOMPATIBLE` belongs here unless future evidence proves a different owner.

### 5.4 Production admission (conceptual freeze — no policy engine in PROVIDER-QUAL-1)

Production admission requires **all** of:

1. `QualificationStatus` satisfies required level (e.g. `PRODUCTION_QUALIFIED`).
2. `QualificationEvidenceValidity` is `CURRENT`.
3. Compatibility requirements pass (platform + domain).
4. Domain policy accepts evidence scope (environment, suite version, capability).

`PRODUCTION_QUALIFIED` alone is **not** sufficient forever. PLUGIN-7 `evaluate_package_production_admission` remains a **pure hook**; full provider admission policy is a later slice.

---

## 6. Suite ownership (FROZEN)

| Owner | Owns |
|-------|------|
| **Domain** | Suite semantics: CAS rules, idempotency, isolation, reconciliation — e.g. CW PostgreSQL repository suite |
| **Platform coordination** | Suite **identity** registration hooks, run/evidence indexing, cross-domain result vocabulary |
| **Platform coordination** does **not** define | CW CAS semantics, CW idempotency, RAG isolation, VK reconciliation |

---

## 7. Evidence model (FROZEN)

**Decision: C — both, with one canonical mapping.**

| Artifact | Role |
|----------|------|
| **`QualificationEvidence`** (`intergrax/core/qualification/evidence.py`) | Canonical safe metadata inside `ProviderQualificationRun`. No secrets, no raw log payloads. |
| **`ProofReceipt`** (`intergrax/proofs/receipts/`) | Optional **persistence projection** for durable, queryable proof storage via `DocumentStore`. |

### 7.1 Canonical mapping

```text
ProviderQualificationRun
  qualification_run_id  (created at execution; authoritative identity)
  evidence: tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...]
      ref --> ProofReceipt locator / proof_id (when persisted; optional)
      kind --> evidence category (suite_result, live_backend, reproducibility, ...)
        |
        v (optional persistence/projection)
ProofReceipt
  proof_id                (persistence-owned locator; not assumed equal to qualification_run_id)
  qualification_run_id    (references execution-owned run identity)
  proof_kind = provider_qualification / domain-specific kind
  domain_evidence = structured run summary (counts, suite id, subject dimensions)
  provider_evidence = bounded backend facts (real_backend, version, environment_id)
  guardrails = mocks/substitution flags
```

Do **not** equate `proof_id == qualification_run_id` unless explicitly justified for a specific storage backend. Canonical direction: execution creates `qualification_run_id`; optional `ProofReceipt` references it without identity inversion.

- **In-run canonical:** `ProviderQualificationRun.evidence` holds `QualificationEvidence` refs directly (**A**).
- **Durable index:** when persisted, one `ProofReceipt` per run (**B**) with `QualificationEvidence.ref` pointing to the receipt locator.
- **No second vocabulary:** do not invent parallel evidence type names; domain-specific detail lives in `ProofReceipt.domain_evidence` / `provider_evidence` bags and in evidence `code`/`label` fields.

### 7.2 `ProviderQualificationEvidenceKind` (target contract — PROVIDER-QUAL-2)

Extends the PLUGIN-7 kind pattern; exact enum implementation deferred to PROVIDER-QUAL-2. Conceptual kinds:

- `suite_execution` — domain suite pass/fail counts
- `live_backend` — real vendor backend used
- `reproducibility` — safe rerun reference
- `limitation` — explicit scope caps
- `source_anchor` — git revision / harness version

---

## 8. Staleness model (FROZEN — detection deferred)

### 8.1 Semantics

Staleness affects **validity**, not historical `QualificationStatus`. A `PRODUCTION_QUALIFIED` run may become `STALE` when drift dimensions change.

### 8.2 Trigger dimensions (conceptual)

| Trigger | Drift signal (metadata needed) |
|---------|-------------------------------|
| Provider version change | `provider_version` on subject |
| Adapter implementation change | `adapter_identity` + `source_revision` |
| Suite version change | `qualification_suite_version` |
| Domain contract change | `capability_id` + domain contract revision label |
| Critical platform implementation change | `intergrax_revision` |
| Runtime dependency compatibility change | compatibility evaluation inputs |

### 8.3 Metadata for future automatic drift detection (no hash engine in PROVIDER-QUAL-1)

Record and index at minimum:

- full `ProviderQualificationSubject` dimensions (section 3)
- `source_revision`, `intergrax_revision`, `qualification_suite_version`
- `environment_id` + `environment_metadata` guardrail flags
- `executed_at`
- optional content fingerprints on suite definition and adapter package (labels only; computation deferred)

---

## 9. Many-vendor scale (FROZEN)

Architecture must support **5 to 20 to 50+** providers without:

- vendor enums in qualification core
- one CI job per vendor on every PR
- qualification-core modification per provider
- N x M live qualification on every PR

| Shared once | Linear per vendor |
|-------------|-------------------|
| Contracts (`ProviderQualificationSubject`, `ProviderQualificationRun`, evidence kinds) | Adapter + environment setup |
| Evidence schema + parsing/index | Domain suite execution for that provider/capability |
| Runner protocol (executor-neutral) | Evidence records |
| Qualification index / query surface | Maintainer qualification host config |

---

## 10. CI boundary (FROZEN)

| CI owns | CI does not own |
|---------|-----------------|
| Qualification contract unit tests | Executing every real vendor on every PR |
| Schema validation for subject/run/evidence | Proving all provider versions |
| Deterministic harness / reference tests (mocks, SQLite substitution where allowed) | Production admission policy |
| Architecture invariant tests | Public claims authoring |
| Evidence parsing and index correctness | Live vendor matrix as merge gate |

Live vendor qualification runs on **bounded qualification hosts** or scheduled maintainer workflows — not as a universal PR gate.

---

## 11. PostgreSQL proof template (CANONICAL MAPPING — no runtime record yet)

Template for the completed Collaborative Work PostgreSQL proof. **Do not** create the runtime/evidence record until PROVIDER-QUAL-2 integration is approved.

**Immutable run record** (`ProviderQualificationRun` — no `validity` field):

```yaml
qualification_run_id: "<created by qualification execution>"
subject:
  provider_id: postgresql
  provider_version: "16.6"
  capability_id: collaborative_work.persistence.v1
  domain: collaborative_work
  adapter_identity: intergrax.collaborative_work.postgresql
  intergrax_revision: "<git sha at proof execution>"
  qualification_suite_id: cw.postgresql.repository.v1
  qualification_suite_version: "<suite content label>"
  environment_id: local-docker-qual-host
status: PRODUCTION_QUALIFIED
executed_at: "<utc completion timestamp>"
result_summary:
  passed: 15
  failed: 0
  skipped: 0
  label: "CW PostgreSQL repository qualification suite"
environment_metadata:
  real_backend: true
  mocks: false
  sqlite_substitution: false
  bounded_environment: local Docker qualification host
limitations:
  - exact capability collaborative_work.persistence.v1 only
  - exact provider version 16.6 only
  - not transferable to other PostgreSQL capabilities or versions
evidence:
  - kind: suite_execution
    code: all_passed
    label: "15 passed / 0 skipped / 0 failed"
  - kind: live_backend
    code: real_postgresql
    label: "real_backend=true"
  - kind: limitation
    code: scoped_capability_and_version
```

**Separate current validity / admission view** (not part of the immutable run; may be superseded):

```yaml
qualification_run_id: "<same run id as above>"
validity: CURRENT
evaluated_at: "<utc evaluation timestamp>"
# Later evaluation for the same run id (append-only / derived latest view):
# validity: STALE
# reason: adapter_revision_changed
```

The run record above stays `status: PRODUCTION_QUALIFIED` even when validity later becomes `STALE` or `REVOKED`.

---

## 12. Oracle extensibility proof (design-only)

Adding Oracle 23ai for the same CW capability:

```text
UNCHANGED
  QualificationStatus vocabulary
  intergrax/core/qualification/*
  CI core workflow (contract + harness tests only)
  CW semantic contracts / repository ports
  ProviderQualificationSubject / ProviderQualificationRun shape

ADDITIONS ONLY
  provider_id=oracle, provider_version=23ai
  Oracle adapter (domain + integrations)
  Oracle environment setup (Docker/operator host)
  Oracle execution adapter if driver/session differs
  Domain suite execution --> ProviderQualificationRun + evidence
  Optional ProofReceipt persistence
```

No new enums, no qualification-core edits, no per-vendor CI job added to default PR workflow.

---

## 13. Relationship to existing PLUGIN-7 contracts

| Existing (PLUGIN-7) | Provider extension (PROVIDER-QUAL-1 freeze) |
|-----------------------|-----------------------------------------------|
| `PluginQualificationSubject` | Package/capability/delivery identity; provider subject adds `provider_id`, `provider_version`, suite/env dimensions |
| `PluginQualificationResult` | In-process result without run id; provider path uses `ProviderQualificationRun` as authoritative run record |
| `PluginQualificationEvidenceKind` | Platform plugin evidence; provider kinds extend same `QualificationEvidence` pattern |
| `evaluate_package_production_admission` | Remains package-level pure gate; provider admission composes outcome + validity + compatibility + domain policy |

---

## 14. Open for PROVIDER-QUAL-2 (explicitly out of scope here)

**PROVIDER-QUAL-2 implements (contract boundary):**

- `ProviderQualificationSubject`
- `ProviderQualificationRun` (immutable run; **no** embedded mutable `validity` field)
- `QualificationEvidenceValidity` vocabulary
- typed validity/admission view or record (e.g. `QualificationValidityRecord` / `QualificationAdmissionView`) **only if** ownership remains sufficiently frozen here

**Still deferred (do not start in PROVIDER-QUAL-2 scope creep):**

- global admission policy engine
- automatic staleness/hash impact engine
- persistence assigning `qualification_run_id` (identity is execution-owned; persistence preserves it)
- GitHub Actions or other executor implementations


## 15. PROVIDER-QUAL-3A-R1 — provider binding / platform reuse (FROZEN)

**Decision (PROVIDER-QUAL-3A corrected):** **EXTEND_EXISTING_PROVIDER_BINDING**

Integrations provider binding is the canonical foundation and **must** be reused. A minimal typed **domain-provider bridge** is still required before provider qualification evidence integration.

**Not** **REUSE_EXISTING_PROVIDER_BINDING** (too strong): generic Integrations resolution alone does not materialize the Collaborative Work repository bundle contract.

### 15.1 Reuse unchanged (A)

| Capability | Owner | Mechanism |
|------------|-------|-----------|
| Catalog registration | Integrations | `IntegrationManifest`, `register_from_manifest`, `register_integration`, `get_entry`, bootstrap `register_*_integration` |
| Host selection | Application composition | `IntegrationProfile`, `build_profile_from_env`, options merge |
| Provider resolution | Integrations registry | `factory.resolve`, `resolve_from_profile`, `resolve_relational_store` |
| Provider identity | Integrations contracts | `provider_id` as data-driven catalog slug |
| Provider factories | Integrations catalog | e.g. `create_postgresql_relational_store` |
| Config / env ownership | Provider package + host | e.g. `PostgreSQLIntegrationConfig.from_env`, `IntegrationManifest.env_prefix` |
| Generic capability ports | Integrations contracts | e.g. `RelationalStore`, `RelationalStoreIntegrationContract` |
| Typed category resolution | Integrations | `IntegrationCategory.RELATIONAL_STORE`, `PROVIDER_CATEGORY_CONTRACT_REGISTRY` (metadata) |

### 15.2 Extend required (B)

After provider selection/resolution, domain qualification must receive a **typed domain-owned capability/factory** that materializes the authoritative domain repository bundle.

| Gap | Meaning |
|-----|---------|
| **Typed domain-provider materialization bridge** | Connects resolved provider identity/capability to domain-specific persistence contract (e.g. `CollaborativeWorkRepositories`) |

Conceptual names only (no frozen implementation class): `CollaborativeWorkPersistenceProvider`, `CollaborativeWorkRepositoryFactory`.

### 15.3 Forbidden (C)

- New qualification-specific provider registry (`ProviderQualificationRegistry`, `QualificationProviderRegistry`, `QualificationVendorCatalog`)
- New global provider registry outside Integrations catalog
- Vendor enum in qualification core
- Central PostgreSQL/Oracle switch in qualification dispatch
- Direct qualification-core vendor construction (`PostgreSQLConnectionProvider`, `PostgreSQLIntegrationConfig`, `PostgreSQLCollaborativeWorkStore`, `postgresql_repository` imports)

### 15.4 Current reusable Integrations flow

```text
IntegrationProfile
        ¡
IntegrationCategory.RELATIONAL_STORE
        ¡
factory.resolve / resolve_from_profile
        ¡
catalog entry
        ¡
provider factory
        ¡
RelationalStore
```

This flow is **reusable** for provider identity, selection, resolution, and generic relational mechanics.

**However:** `RelationalStore` exposes generic relational operations only. It is **not** the authoritative Collaborative Work repository contract. Collaborative Work requires domain semantics: membership, delegation, principal authority, policy, operation profile repositories; CAS; uniqueness; idempotency; durability; transactional concurrency.

Provider Qualification must **not** run CW qualification semantics directly over generic `RelationalStore` SQL. That would violate **PLATFORM-REUSE-3** (domain owns semantics; platform owns mechanics).

### 15.5 Current CW vendor coupling (implementation evidence only)

Collaborative Work composition currently contains vendor-specific constructors:

- `open_sqlite_collaborative_work_repositories(...)`
- `open_postgresql_collaborative_work_repositories(...)`

`CollaborativeWorkRepositories` currently carries concrete storage implementation knowledge.

This is acceptable as **current implementation lineage**, but **not** acceptable as the future qualification binding boundary. The direct PostgreSQL opener is **historical/implementation evidence only** — **not** canonical provider qualification routing.

Qualification orchestration must **not** call `open_postgresql_collaborative_work_repositories(...)` or `open_oracle_collaborative_work_repositories(...)` directly.

### 15.6 Target responsibility split (FROZEN)

| Layer | Owns | Does not own |
|-------|------|--------------|
| **Integrations** | Provider identity, registration/catalog, selection, configuration, generic factory/resolution, provider lifecycle mechanics | Domain repository ports, CW persistence semantics, qualification subject/run |
| **Collaborative Work (domain)** | Repository ports, repository bundle contract, CW persistence semantics, domain adapter construction **contract** | Provider catalog registration, vendor driver/session details |
| **Provider package** | Concrete implementation behind domain-provider binding; PostgreSQL/Oracle/future vendor mechanics; driver/session/backend | Qualification outcome vocabulary, suite semantics |
| **Qualification** | Subject identity, run/evidence; invokes domain qualification suite using **injected typed domain capability** | Provider selection/construction; vendor dispatch |
| **Platform Plugins** | Qualification metadata coordination, compatibility/trust/admission hooks, subject/run architecture | Runtime integration provider resolution, domain repository construction, vendor session mechanics |

### 15.7 Target binding flow (canonical)

```text
Qualification Executor / Host Composition
        |
        | subject.provider_id
        v
IntegrationProfile / canonical Integrations provider binding
        |
        v
resolved provider identity / provider capability
        |
        v
TYPED DOMAIN-PROVIDER BRIDGE
        |
        v
Collaborative Work persistence capability/factory
        |
        v
CollaborativeWorkRepositories
        |
        v
CW qualification suite
        |
        v
ProviderQualificationRun
```

`IntegrationProfile.resolve(...)` alone is **not** equivalent qualification binding. `open_postgresql_collaborative_work_repositories(...)` alone is **not** equivalent qualification binding.

### 15.8 RelationalStore boundary (do not misuse)

`RelationalStore` remains reusable as a **generic integration capability**.

CW qualification suite must **not** be rewritten to operate directly against:

- `RelationalStore.execute(...)`
- `RelationalStore.fetch_all(...)`

if that bypasses CW repository ports/domain semantics.

The future bridge may internally use provider/session/relational mechanics, but the suite sees Collaborative Work domain contracts.

### 15.9 PostgreSQL target flow

```text
provider_id = "postgresql"
        ¡
IntegrationProfile / Integrations catalog
        ¡
canonical provider resolution
        ¡
typed CW persistence/provider bridge
        ¡
PostgreSQL CW adapter
        ¡
PostgreSQLConnectionProvider / PostgreSQLSession
        ¡
real PostgreSQL
        ¡
CollaborativeWorkRepositories
        ¡
same CW qualification suite
```

Qualification core does **not** import: `PostgreSQLConnectionProvider`, `PostgreSQLIntegrationConfig`, `PostgreSQLCollaborativeWorkStore`, `postgresql_repository`.

### 15.10 Oracle extensibility proof (same binding contract)

```text
provider_id = "oracle"
        ¡
same Integrations resolution
        ¡
same typed CW domain-provider bridge contract
        ¡
Oracle CW adapter
        ¡
Oracle provider mechanics
        ¡
CollaborativeWorkRepositories
        ¡
same CW qualification suite
```

Adding Oracle may require: Oracle CW domain adapter, registration/binding, config/env, vendor implementation.

Must **not** require: qualification core changes, `ProviderQualificationSubject` changes, `ProviderQualificationRun` changes, qualification vendor enum, central dispatch change, CW qualification semantic suite rewrite.

### 15.11 Future-vendor-X acceptance test (FROZEN)

A new provider passes only if it can be added through:

```text
existing Integrations registration
+ implementation of the same domain-provider binding contract
```

without editing:

- `intergrax/core/qualification/*`
- qualification dispatch
- provider enums
- existing vendor branches

If central code gains `if provider_id == "future-vendor-x"` › **FAIL**.

### 15.12 Provider registry decision

**No** new `ProviderQualificationRegistry`, `QualificationProviderRegistry`, or `QualificationVendorCatalog`.

Integrations catalog remains canonical provider registration/resolution. The missing capability is **not** a registry — it is a typed bridge/materialization contract between resolved provider and domain-specific authoritative repository/capability bundle.

### 15.13 Platform Plugins boundary

Platform Plugins owns: qualification metadata coordination, compatibility/trust/admission hooks, provider qualification subject/run architecture.

Platform Plugins does **not** own: runtime integration provider resolution, domain repository construction, vendor session mechanics. Do not expand Platform Plugins into a universal runtime provider registry.

### 15.14 Config / secrets

The typed bridge must reuse existing provider configuration/environment mechanisms.

**Forbidden:** `CWPostgresDSN`, `QualificationOracleCredentials`, `ProviderQualificationSecretConfig`, or equivalent duplicated configuration systems. Provider-specific config remains owned by the provider/Integrations layer; host/executor composition supplies/resolves it.

### 15.15 Typed-wiring requirement

Future implementation must use explicit typed interfaces/factories.

**Forbidden:** `getattr`, `setattr`, `hasattr`, `vars`, `object.__setattr__`, `__dict__`, string method dispatch, private-field unwrapping, method-presence discovery. No dynamic provider capability detection. If provider does not implement the required domain binding contract › fail explicitly.

### 15.16 Binding matrix (PROVIDER-QUAL-3A-R1)

| Concern | Classification | Notes |
|---------|----------------|-------|
| Provider registry / discovery | **EXISTING_REUSABLE** | Integrations catalog |
| Generic capability resolution | **EXISTING_REUSABLE** | `factory.resolve`, `IntegrationProfile` |
| Domain qualification suite | **DOMAIN_SPECIFIC** | CW owns semantics |
| Qualification binding | **EXISTING_EXTENDABLE** | Reuse Integrations resolution + extend with typed bridge |
| **Gap** | **MISSING** | Typed domain-provider materialization bridge |
| Evidence persistence | **MISSING** | **PROVIDER-QUAL-3C** |

Do **not** classify qualification binding as fully reusable without the typed bridge extension.

### 15.17 Stage split (FROZEN)

| Stage | Scope |
|-------|-------|
| **PROVIDER-QUAL-3A-R1** | Architecture correction only (this freeze) |
| **PROVIDER-QUAL-3B** | **READY_FOR_REVIEW** — `CollaborativeWorkPersistenceProvider.materialize_collaborative_work_repositories()`; domain resolver `resolve_collaborative_work_repositories(profile)` composes `resolve_relational_store` |
| **PROVIDER-QUAL-3C** | Qualification evidence persistence/index + record accepted PostgreSQL 16.6 evidence through the canonical binding |

**PROVIDER-QUAL-3B** implementation (review gate): `intergrax/collaborative_work/persistence_provider.py`; vendor adapters on `SqliteRelationalStoreIntegration` / `PostgresqlRelationalStoreIntegration`; `CollaborativeWorkRepositories.store` typed as `CollaborativeWorkStoreOwner` (lifecycle-only).

PostgreSQL evidence must **not** be persisted before **PROVIDER-QUAL-3B** is independently approved. **PROVIDER-QUAL-3C** remains **not started**.

### 15.18 Frozen invariants (PROVIDER-QUAL-INV-1 … INV-8)

| ID | Invariant |
|----|-----------|
| **INV-1** | No vendor-specific qualification execution path in qualification core/coordinator. |
| **INV-2** | Qualification uses canonical Integrations provider resolution (`IntegrationProfile`, catalog `factory.resolve`). |
| **INV-3** | Domain qualification consumes typed domain capability ports/factories — never raw vendor construction. |
| **INV-4** | Adding a provider must not modify qualification core. |
| **INV-5** | `provider_id` remains string/data-driven — never a closed-world enum in qualification core. |
| **INV-6** | No qualification-specific provider registry. |
| **INV-7** | **Domain provider binding:** after provider selection/resolution, domain qualification must receive a typed domain-owned capability/factory. Qualification orchestration must not call `open_postgresql_collaborative_work_repositories(...)`, `open_oracle_collaborative_work_repositories(...)`, or equivalent vendor openers directly. |
| **INV-8** | **Provider selection ? domain materialization:** Integrations owns provider selection/resolution; domain owns construction/materialization of its semantic repository/capability adapter. A typed bridge connects those responsibilities. Neither layer may absorb the other's semantics. |

### 15.19 Identity relationships

| Field | Role |
|-------|------|
| `provider_id` | Vendor/backend catalog slug (e.g. `postgresql`, `oracle`) |
| `integration_kind` | Category / `IntegrationCategory` value (e.g. `relational_store`) |
| `integration_id` | `{provider_id}:{integration_kind}` via `derive_platform_integration_id` |
| `capability_id` | Domain qualification scope (e.g. `collaborative_work.persistence.v1`) — **not** a catalog slug |
| package identity | Distribution/delivery (`package_name`, entry points) — PLUGIN-7 path; orthogonal to runtime slug |

---

## 16. PROVIDER-QUAL-3C — evidence persistence/integration (deferred)

- qualification run persistence/index integration
- mapping to existing `ProofReceipt` where appropriate
- recording the PostgreSQL template as a live `ProofReceipt` (already accepted PostgreSQL 16.6 bounded qualification evidence)
- query/discovery surface for qualification evidence as already architected
- **binding:** canonical flow per §15.7 (Integrations resolution › typed domain-provider bridge › CW suite) — **PROVIDER-QUAL-3B** must land first

**Out of scope until 3C:** persisting PostgreSQL 16.6 evidence before **PROVIDER-QUAL-3B** approval.

---

## 17. References

- [`PLATFORM_PLUGINS.md`](../PLATFORM_PLUGINS.md) section 18
- [`PROOF_RECEIPTS.md`](../PROOF_RECEIPTS.md)
- [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](../../maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)
- [`COLLABORATIVE_WORK.md`](../COLLABORATIVE_WORK.md) — CW-INV-19, durable PostgreSQL adapter
- [`INTEGRATIONS.md`](../INTEGRATIONS.md) — `provider_id` vs `integration_kind`
