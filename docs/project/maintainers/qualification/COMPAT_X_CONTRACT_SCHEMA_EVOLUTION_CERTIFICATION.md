# COMPAT-X — Contract, Schema & Evolution Certification (P0 closed-world baseline)

**Status:** **COMPAT-X-P0** = **CLOSED / independently accepted** @ `0612c2a7a263b22a6b72ed5476e0bf0a52cdfea0` · **COMPAT-X-P0-R3** = **CLOSED / independently accepted** (same evidence SHA) · **COMPAT-X-R1-R1** = **REJECTED / superseded by R1-R2** @ `1e79d2a72be8095515f0e6ffa8e59dd94eb5c7f6` · **COMPAT-X-R1-R2** = **REJECTED / superseded by R1-R3** @ `bc6b46b7c300d1db12ba664d327b339a75762532` · **COMPAT-X-R1-R3** = **READY FOR AUDIT** (implementation pending independent audit) · **COMPAT-X-R1** = **BLOCKED ON R1-R3 AUDIT** · **COMPAT-X** parent = **OPEN** / **CURRENT** — **BLOCKED ON R1 AUDIT + R2–R5 REMEDIATION** (not parent closure)

**Branch:** `development`

**Accepted evidence HEAD (independent exact-SHA audit):** `0612c2a7a263b22a6b72ed5476e0bf0a52cdfea0`

**START_HEAD (P0-R3 session):** `d0466ac32ca4296bc1ac5ec5f874eda1039afed2`

**Bookkeeping tip (pre-R3):** `d0466ac32ca4296bc1ac5ec5f874eda1039afed2`

**Rejected P0-R2 implementation (do not use as closure baseline):** `6512ac5df43f45722b0bce50d90586609e6d46f3`

**Rejected P0-R1 implementation:** `065a0dcd875f8b985792e70f4e1f1b20bd8de094`

**Rejected initial P0 baseline:** `cea6393775c8b92a2cefb165fc9930511c776169` (historical only)

**P0-R3 reason:** Independent audit rejected shim discovery / parallel-authority scope mismatch (`compat_adapter_module_scope` narrower than mechanical shim candidacy). R3 introduces typed `CompatibilityCandidateContext`, authority inspection on the same candidate universe, top-level executor/authorizer probes, and production reconciliation (**production delta = 0**).

**Production delta:** `0`

## Purpose

Mechanical closed-world inventory of compatibility-relevant platform surfaces on current HEAD with **candidates = semantic surfaces + evidence-backed exclusions**. Establishes evidence for **FRZ-CMP-01..08** *candidate* state only. Does **not** close **COMPAT-X** or promote global **FRZ-CMP-* = PASS**.

## Inventory summary (mechanical)

| Metric | Count |
| --- | ---: |
| Raw discovered signals (`DISCOVERED_CANDIDATES`) | 7266 |
| Semantic compatibility surfaces (`CLASSIFIED_COMPAT_SURFACES`) | 1879 |
| Evidence-backed exclusions (`EXCLUSIONS_WITH_EVIDENCE`) | 5289 |
| Unclassified candidates | 0 |
| Public/stable facet (inventory) | (see gate `test_cx_p0_r1_inventory_counts_reportable`) |
| Migration mechanisms (discovered) | 157 |
| Compatibility shim candidates (discovered) | 474 |
| Authority-inspected compatibility candidates | 474 |
| Uninspected compatibility candidates | 0 |
| Parallel authority (production inventory) | 0 |
| Registry version conflicts | 0 |

**SSOT:** `tests/qualification/compat_x/_compat_x_inventory.py` (`COMPAT_X_INVENTORY`)

**Discovery / parity:** `tests/qualification/compat_x/_compat_x_closed_world.py` (`build_closed_world_report`)

**Shim authority reconciliation:** `tests/qualification/compat_x/_compat_x_shim_authority.py` (`build_shim_authority_scope_reconciliation`)

**Gate:** `tests/qualification/compat_x/test_compat_x_inventory_gates.py` — parity + adversarial suite (P0-R2 version-removal; P0-R3 scope convergence + outside-compat synthetic probes)

### Discovery counts by mechanism (raw signals)

| Kind | Count |
| --- | ---: |
| `registry.contracts` | 8 |
| `registry.runtime` | 19 |
| `event.payload` | 37 |
| `version.constant` | 443 |
| `class.field.version` | 832 |
| `wire.persistence` (excluded after classification) | 601 |
| `public.export` (excluded after classification) | 4688 |
| `migration.mechanism` | 157 |
| `compat.shim` | 474 |
| `mechanism.policy_owner` | 2 |
| `defect.persisted_without_version` | 5 |

## Five `PERSISTED_SCHEMA_WITHOUT_VERSION` findings (preserved — COMPAT-X-R2)

Classification: **CHILD TASK REQUIRED — blocks COMPAT-X** (not P0 inventory closure). Not remediated in P0-R2.

| Path | Class | Semantic identity | Assessment |
| --- | --- | --- | --- |
| `intergrax/agent_distribution/agent_contract_authority.py` | `AgentPackageContractAuthorityService` | `persisted.without_version:intergrax/agent_distribution/agent_contract_authority.py:AgentPackageContractAuthorityService` | requires R2 validation (likely heuristic false positive — authority service, not wire contract) |
| `intergrax/agent_distribution/agent_contract_authority.py` | `PackageAgentContractAuthorityError` | `persisted.without_version:intergrax/agent_distribution/agent_contract_authority.py:PackageAgentContractAuthorityError` | requires R2 validation (likely heuristic false positive — exception type) |
| `intergrax/agent_distribution/runtime_revision_service.py` | `RuntimeRevisionService` | `persisted.without_version:intergrax/agent_distribution/runtime_revision_service.py:RuntimeRevisionService` | requires R2 validation |
| `intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py` | `DocumentStoreSuspendedExecutionOperationStore` | `persisted.without_version:intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py:DocumentStoreSuspendedExecutionOperationStore` | requires R2 validation (may be real persisted boundary) |
| `intergrax/runtime/nexus/orchestration/long_running_bridge.py` | `RuntimeEventPublisher` | `persisted.without_version:intergrax/runtime/nexus/orchestration/long_running_bridge.py:RuntimeEventPublisher` | requires R2 validation (may be real event wire boundary) |

Discovery evidence kind: `defect.persisted_without_version` / signal `missing_schema_version:<class>` from wire+persist heuristic in `_compat_x_closed_world.py`.

## Canonical registries (complementary owners)

| Registry | Path | Responsibility |
| --- | --- | --- |
| `CONTRACT_SCHEMA_REGISTRY` | `intergrax/contracts/migrations/registry.py` | Tier-0 **public contract** schema ids |
| `RUNTIME_SCHEMA_REGISTRY` | `intergrax/runtime/schema/registry.py` | **Runtime persisted/wire** schema bundle ids |
| Event payload registry | `intergrax/runtime/events/payload_registry.py` | **Runtime event payload** `payload_schema_id` families |

**Registry overlap (normalized identities):** `tests/qualification/compat_x/_compat_x_registry_analysis.py` — **version_conflicts = 0** on current HEAD.

## Owner matrix (responsibility state — not overstated single owner)

See `tests/qualification/compat_x/_compat_x_owner_discovery.py` (`COMPAT_X_OWNER_MATRIX`) with `OwnerResponsibilityState`:

| Concern | State |
| --- | --- |
| Public contract identity/version | CURRENT_CONFIRMED_OWNER |
| Runtime schema registry | CURRENT_CONFIRMED_OWNER |
| Persisted schema migration | FRAGMENTED_UNOWNED |
| Versioning policy | CURRENT_CONFIRMED_OWNER @ architecture hub |
| Event schema evolution | CANDIDATE_OWNER_REMEDIATION_REQUIRED |
| Event evolution policy | FRAGMENTED_UNOWNED |
| Plugin/provider compatibility policy | FRAGMENTED_UNOWNED |
| Deprecation/removal | FRAGMENTED_UNOWNED |
| Compatibility adapters (langchain bridge) | CURRENT_CONFIRMED_OWNER (translation-only evidence) |

## Accepted P0 evidence (independent audit @ `0612c2a7…`)

```text
raw discovery signals = 7266
semantic compatibility surfaces = 1879
evidence-backed exclusions = 5289
unclassified = 0

registry version conflicts = 0

compatibility candidates = 474
authority-inspected compatibility candidates = 474
uninspected compatibility candidates = 0
production PARALLEL_AUTHORITY = 0

production delta = 0
```

**Global FRZ-CMP-* = PASS:** not promoted — checklist rows remain **OPEN** until **COMPAT-X** parent closure.

## FRZ-CMP candidate state (accepted P0 @ `0612c2a7…`)

| Criterion | Candidate | Blocker IDs |
| --- | --- | --- |
| FRZ-CMP-01 | **PASS CANDIDATE** / P0 evidence accepted @ `0612c2a7…` | — (closed-world parity + semantic inventory); checklist **OPEN** until parent closure |
| FRZ-CMP-02 | **PASS CANDIDATE** — R1 evidence pending independent audit | CMP-P0-B02 remediated in R1 qualification layer |
| FRZ-CMP-03 | **BLOCKED** | CMP-P0-B03 (+ 5 `PERSISTED_SCHEMA_WITHOUT_VERSION` — **CHILD TASK REQUIRED**) → COMPAT-X-R2 |
| FRZ-CMP-04 | **BLOCKED** | CMP-P0-B04 → COMPAT-X-R3 |
| FRZ-CMP-05 | **BLOCKED** | CMP-P0-B05 → COMPAT-X-R4 |
| FRZ-CMP-06 | **BLOCKED** | CMP-P0-B06 → COMPAT-X-R2 |
| FRZ-CMP-07 | **BLOCKED** | CMP-P0-B07 → COMPAT-X-R5 |
| FRZ-CMP-08 | **PASS CANDIDATE** / P0 evidence accepted @ `0612c2a7…` | — (0 `PARALLEL_AUTHORITY` production; `CompatibilityCandidateContext` + full candidate authority inspection; multi-shape synthetic probes incl. outside `intergrax/compat/`); checklist **OPEN** until parent closure |

## Proposed remediation grouping (re-derived from corrected discovery)

1. **COMPAT-X-R1** — Contract Versioning & Classification (FRZ-CMP-02)
2. **COMPAT-X-R2** — Persisted Schema Migration (FRZ-CMP-03, FRZ-CMP-06)
3. **COMPAT-X-R3** — Event Evolution (FRZ-CMP-04)
4. **COMPAT-X-R4** — Plugin/Provider Compatibility (FRZ-CMP-05)
5. **COMPAT-X-R5** — Deprecation / Compatibility Shim Closure (FRZ-CMP-07, FRZ-CMP-08 hardening)

## Tenant isolation audit (local COMPAT-X-P0-R3)

**Verdict:** **PASS** (local scope; not global **TENANT-X**)

- Reuse **INT-EXTCOMP-X**: `test_cert_22_resolver_tenant_isolation` invoked from `test_cx_p0_r1_adversarial_11_tenant_invariant_reuse_extcomp`.
- Tenant-bearing compatibility surfaces flagged in inventory (`tenant_relevant`).
- **Not** using LangChain `schema_version` rejection as tenant evidence.
- **Global FRZ-TEN-* promotion:** `0`

## Test evidence

```text
uv run pytest -p no:xdist tests/qualification/compat_x/
uv run pytest -p no:xdist tests/unit/runtime/schema/test_schema_registry_b07.py
uv run pytest -p no:xdist tests/qualification/external_contract_compatibility/test_external_contract_compatibility_certification.py
uv run pyright tests/qualification/compat_x
```

P0-R2 adversarial additions: `test_cx_p0_r2_adversarial_12_version_removal_regression_probe`; parallel-authority probes `test_cx_p0_r2_parallel_authority_*`; sanctioned translation probes `test_cx_p0_r2_translation_only_*` / `test_cx_p0_r2_legacy_decode_*`.

P0-R3 additions: `test_cx_p0_r3_shim_authority_scope_reconciliation`; outside-compat probes `test_cx_p0_r3_adversarial_a`..`e`; `test_cx_p0_r3_frz_cmp_08_pass_candidate_scope`.

## Unresolved findings (inventory blockers — not remediated in P0)

- **5** `PERSISTED_SCHEMA_WITHOUT_VERSION` surfaces (table above) — **CHILD TASK REQUIRED — blocks COMPAT-X**; owned by **COMPAT-X-R2** (not suppressed in P0).
- R1 maps every P0 semantic surface to a typed version-policy classification (`version-policy unclassified = 0`). Five `PERSISTED_SCHEMA_WITHOUT_VERSION` findings remain **BLOCKED / R2 validation**.

## COMPAT-X-R1 — Contract Versioning & Classification

**START_HEAD:** `a1e9691b57abaa164e51542337f982f55240c8c2`

**Purpose:** Define one cross-platform **evolution policy** (qualification layer) connecting P0 compatibility surfaces to version obligation, identity scheme, change taxonomy, and domain version owners — without a universal runtime registry.

**Semantic policy SSOT:** `docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md`

**Qualification enforcement:** `tests/qualification/compat_x/` (`_compat_x_versioning_policy.py` mirrors canonical policy mechanically)

**Classification SSOT:** `tests/qualification/compat_x/_compat_x_versioning_classification.py` (`COMPAT_X_R1_CLASSIFICATIONS` — 1:1 with `COMPAT_X_INVENTORY`)

**Gates:** `tests/qualification/compat_x/_compat_x_versioning_gates.py`, `tests/qualification/compat_x/test_compat_x_versioning_r1_gates.py`

### Policy model (normative)

- Cross-platform evolution rules: **COMPAT-X** (`_compat_x_versioning_policy.py`).
- Individual version truth: domain owners (`CONTRACT_SCHEMA_REGISTRY`, `RUNTIME_SCHEMA_REGISTRY`, event payload registry, plugin manifest owner, etc.).
- **No** `GLOBAL_SCHEMA_VERSION_REGISTRY`; registries are not merged.

Canonical rules (see `PLATFORM_POLICY_CANON` in policy module):

- For a compatibility-relevant surface, the owning domain controls version identity; COMPAT-X defines evolution rules only.
- Breaking semantic/structural change ⇒ new contract/schema version.
- Unknown compatibility impact ⇒ fail closed.
- Old-version acceptance ⇒ explicit reader/migration/compatibility policy only.
- Additive compatibility is family-policy/evidence based (not global).

### Taxonomies

| Version obligation | Meaning |
| --- | --- |
| `EXPLICIT_VERSION_REQUIRED` | Surface must carry or inherit explicit schema/contract version identity |
| `VERSION_INHERITED_FROM_CANONICAL_ENVELOPE` | Version owned by enclosing canonical envelope |

**SemanticSurfaceIdentity vs VersionIdentity (R1-R3):** `SemanticSurfaceIdentity` identifies *which contract/surface* (e.g. `class.field:<module>:<class>:<field>`). `VersionIdentity` / `current_version` identifies *which generation* of that surface; it is an attribute of semantic identity, never a substitute for it.
| `INTERNAL_NON_VERSIONED_ALLOWED` | Internal/adapter surface — no canonical version authority |
| `NOT_APPLICABLE` | Business/concurrency/deployment version fields ≠ schema evolution |

| Version identity scheme | Examples |
| --- | --- |
| `SCHEMA_ID_GENERATION` | `agent_run.v1`, `runtime_event.v1` |
| `INTEGER_GENERATION` | `schema_version = 1` |
| `SEMANTIC_VERSION` | SemVer where domain uses it |
| `EXTERNALLY_DEFINED_VERSION` | Consumed but not platform-owned scheme |

| Change class | R1 consequence |
| --- | --- |
| `REPRESENTATION_PRESERVING` | No bump required |
| `ADDITIVE_BACKWARD_COMPATIBLE` | Bump optional when family policy allows |
| `BREAKING_STRUCTURAL` / `BREAKING_SEMANTIC` / `REMOVAL_OR_RENAME` | New version identity required |
| `UNKNOWN` | Qualification fails closed |

### Owner model

```text
Cross-platform evolution rule → COMPAT-X versioning policy (qualification)
Individual version truth       → domain semantic owner (registries / module owners)
```

`versioning_policy` owner matrix row: **CURRENT_CONFIRMED_OWNER** @ `docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md` (composition/enforcement: `tests/qualification/compat_x/`).

### Inventory reconciliation

```text
P0 semantic surfaces (1879) = R1 classified (1879) + NOT_APPLICABLE obligations + explicit R2/R3/R4/R5 deferrals
unclassified version-policy obligations = 0
```

### Blockers preserved for later stages

- **5** `PERSISTED_SCHEMA_WITHOUT_VERSION` — R1 obligation `EXPLICIT_VERSION_REQUIRED`, compliance `BLOCKED_R2_VALIDATION`.
- FRZ-CMP-03..07 remain **BLOCKED** (R2–R5).

### Tenant audit (R1 local)

**Verdict:** **PASS** — `test_cx_r1_tenant_audit_reuse_extcomp` reuses INT-EXTCOMP `test_cert_22_resolver_tenant_isolation`.

### R1 tests

```text
uv run pytest -p no:xdist tests/qualification/compat_x/
uv run python scripts/maintenance/check_contract_schema_versions.py
uv run pytest -p no:xdist tests/unit/runtime/schema/test_schema_registry_b07.py
uv run pyright tests/qualification/compat_x
```

`scripts/maintenance/check_contract_schema_versions.py` remains the **domain-specific** CONTRACT_SCHEMA_REGISTRY gate (option A); R1 adds separate COMPAT-X qualification gates.

### Architecture alignment

§40.11 (`AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`) contract-family table remains authoritative for ACP; COMPAT-X generalizes cross-surface evolution rules without contradicting per-family version schemes (no fake global SemVer).

## Post-Step Enterprise Discovery (COMPAT-X-P0 closure)

```text
New current-parent blockers:
FRZ-CMP-02..07 as already discovered by accepted P0.

New future mandatory debt:
none beyond accepted COMPAT-X remediation set.

New candidate roadmap stages:
none; R1–R5 are COMPAT-X children.

FRZ coverage gaps:
FRZ-CMP-02..07 require remediation.
FRZ-CMP-01/08 have accepted P0 candidate evidence but remain OPEN until parent closure.

New ownership / boundary / authority concerns:
none newly discovered by P0-R3.

Roadmap amendment required:
NO
```

## COMPAT-X-R1-R1 — Version Policy Authority & Evidence Integrity Closure

**Rejected baseline (do not use):** `1b107dc8e87acf7d9d547ab694885beacbfd7197`

**Independent exact-SHA audit rejection:** `1e79d2a72be8095515f0e6ffa8e59dd94eb5c7f6` — public/stable R1 accounting used `VERSION_REQUIRED_BUT_MISSING` only and could treat `PUBLIC_STABLE` + missing resolved identity as R1-success when deferral was `BLOCKED_LATER_STAGE` (e.g. qualification false negative on `ProofReceipt.schema_version` while production identity exists via `PROOF_RECEIPT_SCHEMA_VERSION`).

**Purpose:** Canonical semantic policy authority outside qualification tests; typed version authority disposition; obligation vs compliance; additive family evidence; shim canonical-authority gates; FRZ-CMP-02 PASS CANDIDATE scope. **production delta = 0**. **Superseded by COMPAT-X-R1-R2.**

## COMPAT-X-R1-R2 — Public Version Compliance & Static Version Resolution Closure

**START_HEAD:** `1d558bc621bc41b0fe1f853f371c51ea004916b7`

**Root cause (qualification, not production):** `tests/qualification/compat_x/_compat_x_ast_signals.py` extracted class-field versions from literal nodes only; `schema_version = PROOF_RECEIPT_SCHEMA_VERSION` (`ast.Name`) stayed `unknown` despite same-module `PROOF_RECEIPT_SCHEMA_VERSION = "intergrax.proof_receipt.v1"`.

**Static resolution (AST-only, no import/eval):** `build_module_version_constant_map` → resolve class defaults via same-module constants and `Literal[…]` annotations; annotation/default conflict → fail closed (`static_version_conflict`); dynamic expressions stay unresolved.

**ProofReceipt production evidence (unchanged):** `intergrax/proofs/receipts/contracts.py` — `PROOF_RECEIPT_SCHEMA_VERSION` + `ProofReceipt.schema_version` → `intergrax.proof_receipt.v1`; discovery `version_source` continuity `PROOF_RECEIPT_SCHEMA_VERSION→ProofReceipt.schema_version`; classification `EXPLICIT_VERSION_REQUIRED` / `VERSION_PRESENT_AND_OWNED` / `SCHEMA_ID_GENERATION`.

**Public/stable R1 identity gate:** `public_stable_noncompliant_version_identity_count` requires `VERSION_PRESENT_AND_OWNED` or `VERSION_INHERITED_AND_RESOLVED` for every `PUBLIC_STABLE` surface (later-stage deferral does not waive missing identity).

**Closed-world reconciliation (post R1-R2 implementation, pre-audit):**

| Metric | Before R1-R2 (rejected R1-R1 audit) | After R1-R2 implementation |
|--------|-------------------------------------|----------------------------|
| Total P0/R1 surfaces | 1870 | 1870 |
| PUBLIC_STABLE surfaces | 10 | 10 |
| PUBLIC_STABLE compliant identities | 9 | 10 |
| PUBLIC_STABLE noncompliant identities | 1 (`ProofReceipt` false negative) | 0 |
| `static_version_reference_conflict_count` | — | 0 |
| Five `PERSISTED_SCHEMA_WITHOUT_VERSION` | 5 preserved | 5 preserved (R2) |

**Adversarial probes:** `test_compat_x_ast_static_version_resolution.py` (A–G); `test_compat_x_versioning_r1_gates.py` (H–M including production `ProofReceipt` path).

**FRZ-CMP-02:** PASS CANDIDATE pending independent exact-SHA audit; checklist row remains **OPEN**.

**Independent exact-SHA audit rejection @ `bc6b46b7c300d1db12ba664d327b339a75762532`:** static resolution correct, but semantic identity became version-value-derived (`schema.literal:<version>`) and collapsed 9 surfaces vs accepted P0 (`1879` → `1870`); inherited-version adversarial coverage incomplete (`pytest.skip` on mandatory probe).

## COMPAT-X-R1-R3 — Semantic Surface Identity Preservation & Inherited-Version Gate Closure

**Rejected baseline (do not use):** `bc6b46b7c300d1db12ba664d327b339a75762532`

**Fix:** class version fields keep contract-specific `class.field:<module_path>:<class_name>:<field_name>` semantic identity; resolved version lives in `current_version` + `version_source` only (static resolution preserved).

**ProofReceipt:** `semantic_identity` = `class.field:intergrax/proofs/receipts/contracts.py:ProofReceipt:schema_version`; `current_version` = `intergrax.proof_receipt.v1`; `version_source` = `PROOF_RECEIPT_SCHEMA_VERSION→ProofReceipt.schema_version`.

**Inherited-version gates:** synthetic `PUBLIC_STABLE` + `COMPATIBILITY_ADAPTER` probes — resolved canonical envelope owner/version ⇒ PASS; unresolved envelope authority ⇒ FAIL (no inventory skip).

**Closed-world reconciliation (accepted P0 vs R1-R3 corrected inventory):**

| Metric | Accepted P0 @ `0612c2a7…` | Rejected R1-R2 @ `bc6b46b7…` | R1-R3 corrected |
|--------|---------------------------|------------------------------|-----------------|
| raw candidates | 7266 | 7266 | 7266 |
| semantic surfaces | 1879 | 1870 | 1977 |
| exclusions | 5289 | 5289 | 5289 |
| unclassified | 0 | 0 | 0 |
| cross-contract `schema.literal` merge | 98 (legacy P0) | 107 (R1-R2 regression) | 0 |
| split vs accepted P0 | — | — | +98 (contract-specific identities) |
| R1-R2 regression vs P0 | — | −9 | restored |

**Adversarial probes:** `test_compat_x_versioning_r1_r3_identity_gates.py` (A–D + reconciliation); `test_compat_x_versioning_r1_gates.py` (inherited K/L synthetic).

**FRZ-CMP-02:** PASS CANDIDATE pending independent exact-SHA audit; checklist row remains **OPEN**.

### Authority graph (post R1-R1)

```text
canonical architecture compatibility policy → semantic owner
qualification code (tests/qualification/compat_x) → enforcement only
domain registries → current version authority
compatibility shim → NO canonical current-version authority (unless explicit canonical envelope inheritance)
```

## Program status

```text
COMPAT-X-P0-R1 = REJECTED / superseded
COMPAT-X-P0-R2 = REJECTED / superseded by accepted R3
COMPAT-X-P0-R3 = CLOSED / independently accepted
COMPAT-X-P0   = CLOSED / independently accepted
COMPAT-X-R1-R1 = REJECTED / superseded by R1-R2 @ 1e79d2a72be8095515f0e6ffa8e59dd94eb5c7f6
COMPAT-X-R1-R2 = REJECTED / superseded by R1-R3 @ bc6b46b7c300d1db12ba664d327b339a75762532
COMPAT-X-R1-R3 = READY FOR AUDIT
COMPAT-X-R1   = BLOCKED ON R1-R3 AUDIT
COMPAT-X      = OPEN / CURRENT — BLOCKED ON R1 AUDIT + R2–R5 REMEDIATION
COMPAT-X-R2   = NOT ENTERED
TENANT-X      = NOT ENTERED
```

**Accepted P0 evidence HEAD:** `0612c2a7a263b22a6b72ed5476e0bf0a52cdfea0` · **production delta = 0** · **new global FRZ PASS = 0** · **new FRZ-TEN PASS = 0**.

**Next mandatory child:** **COMPAT-X-R2** — Persisted Schema Migration (**not entered**).
