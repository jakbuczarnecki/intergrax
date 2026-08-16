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
| `qualification_run_id` | Stable unique run id (UUID or deterministic idempotency key). |
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

Once recorded, `status`, `subject`, `result_summary`, and `evidence` refs for a given `qualification_run_id` are **immutable**. Later drift is modeled only via `QualificationEvidenceValidity` (section 5.2), superseding runs, or explicit revocation records — never by rewriting historical outcome.

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
status = PRODUCTION_QUALIFIED   # immutable outcome at run time
validity = STALE                  # current admission view (section 5.2)
```

### 5.2 Evidence validity — `QualificationEvidenceValidity` (NEW contract, separate enum)

| Value | Meaning |
|-------|---------|
| `CURRENT` | Evidence still satisfies freshness/policy for admission evaluation. |
| `STALE` | Underlying subject dimensions or platform/domain drift invalidated freshness; outcome history preserved. |
| `REVOKED` | Evidence explicitly withdrawn (security, fraud, operator action). |

Validity is evaluated **on evidence/run records** or on an admission view projection — not stored as a replacement for `QualificationStatus`.

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
  evidence: tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...]
      ref --> ProofReceipt row key / proof_id (when persisted)
      kind --> evidence category (suite_result, live_backend, reproducibility, ...)

ProofReceipt (optional persistence)
  proof_kind = provider_qualification / domain-specific kind
  domain_evidence = structured run summary (counts, suite id, subject dimensions)
  provider_evidence = bounded backend facts (real_backend, version, environment_id)
  guardrails = mocks/substitution flags
```

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

```yaml
qualification_run_id: "<assigned at persistence — PROVIDER-QUAL-2>"
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
validity: CURRENT  # at time of recording; may become STALE later
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

- Python dataclass / Pydantic implementations of `ProviderQualificationSubject`, `ProviderQualificationRun`, `QualificationEvidenceValidity`
- Evidence persistence and qualification index
- Recording the PostgreSQL template as a live `ProofReceipt`
- Admission policy engine wiring
- Automatic staleness/hash impact engine
- GitHub Actions or other executor implementations

---

## 15. References

- [`PLATFORM_PLUGINS.md`](../PLATFORM_PLUGINS.md) section 18
- [`PROOF_RECEIPTS.md`](../PROOF_RECEIPTS.md)
- [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](../../maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)
- [`COLLABORATIVE_WORK.md`](../COLLABORATIVE_WORK.md) — CW-INV-19, durable PostgreSQL adapter
- [`INTEGRATIONS.md`](../INTEGRATIONS.md) — `provider_id` vs `integration_kind`
