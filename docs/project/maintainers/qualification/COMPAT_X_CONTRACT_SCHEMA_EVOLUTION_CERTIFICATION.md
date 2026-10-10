# COMPAT-X — Contract, Schema & Evolution Certification (P0-R2 baseline)

**Status:** COMPAT-X-P0-R2 — **READY FOR AUDIT** (Cursor implementation; not COMPAT-X parent closure)

**Branch:** `development`

**START_HEAD (P0-R2 session):** `237a7341a9be0e017e23dfb7ed2efe05543fda53`

**Bookkeeping tip (pre-R2):** `237a7341a9be0e017e23dfb7ed2efe05543fda53`

**Rejected P0-R1 implementation (do not use as closure baseline):** `065a0dcd875f8b985792e70f4e1f1b20bd8de094`

**P0-R2 reason:** Independent audit rejected weak adversarial gates (tautological version-removal test, narrow parallel-authority classifier). R2 strengthens qualification/discovery only (**production delta = 0**).

**Production delta:** `0`

## Purpose

Mechanical closed-world inventory of compatibility-relevant platform surfaces on current HEAD with **candidates = semantic surfaces + evidence-backed exclusions**. Establishes evidence for **FRZ-CMP-01..08** *candidate* state only. Does **not** close **COMPAT-X** or promote global **FRZ-CMP-* = PASS**.

## Inventory summary (mechanical)

| Metric | Count |
| --- | ---: |
| Raw discovered signals (`DISCOVERED_CANDIDATES`) | 7255 |
| Semantic compatibility surfaces (`CLASSIFIED_COMPAT_SURFACES`) | 1868 |
| Evidence-backed exclusions (`EXCLUSIONS_WITH_EVIDENCE`) | 5289 |
| Unclassified candidates | 0 |
| Public/stable facet (inventory) | (see gate `test_cx_p0_r1_inventory_counts_reportable`) |
| Migration mechanisms (discovered) | 157 |
| Compatibility shim candidates (discovered) | 463 |
| Parallel authority (production inventory) | 0 |
| Registry version conflicts | 0 |

**SSOT:** `tests/qualification/compat_x/_compat_x_inventory.py` (`COMPAT_X_INVENTORY`)

**Discovery / parity:** `tests/qualification/compat_x/_compat_x_closed_world.py` (`build_closed_world_report`)

**Gate:** `tests/qualification/compat_x/test_compat_x_inventory_gates.py` — parity + adversarial suite (P0-R2 version-removal mutation probe; bounded AST parallel-authority probes)

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
| `compat.shim` | 463 |
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
| Versioning policy | FRAGMENTED_UNOWNED |
| Event schema evolution | CANDIDATE_OWNER_REMEDIATION_REQUIRED |
| Event evolution policy | FRAGMENTED_UNOWNED |
| Plugin/provider compatibility policy | FRAGMENTED_UNOWNED |
| Deprecation/removal | FRAGMENTED_UNOWNED |
| Compatibility adapters (langchain bridge) | CURRENT_CONFIRMED_OWNER (translation-only evidence) |

## FRZ-CMP candidate state (P0-R2)

| Criterion | Candidate | Blocker IDs |
| --- | --- | --- |
| FRZ-CMP-01 | **PASS CANDIDATE** | — (closed-world parity + semantic inventory) |
| FRZ-CMP-02 | **BLOCKED** | CMP-P0-B02 → COMPAT-X-R1 |
| FRZ-CMP-03 | **BLOCKED** | CMP-P0-B03 (+ 5 `PERSISTED_SCHEMA_WITHOUT_VERSION` — **CHILD TASK REQUIRED**) → COMPAT-X-R2 |
| FRZ-CMP-04 | **BLOCKED** | CMP-P0-B04 → COMPAT-X-R3 |
| FRZ-CMP-05 | **BLOCKED** | CMP-P0-B05 → COMPAT-X-R4 |
| FRZ-CMP-06 | **BLOCKED** | CMP-P0-B06 → COMPAT-X-R2 |
| FRZ-CMP-07 | **BLOCKED** | CMP-P0-B07 → COMPAT-X-R5 |
| FRZ-CMP-08 | **PASS CANDIDATE** | — (0 `PARALLEL_AUTHORITY` production; AST bounded classifier + multi-shape synthetic probes) |

## Proposed remediation grouping (re-derived from corrected discovery)

1. **COMPAT-X-R1** — Contract Versioning & Classification (FRZ-CMP-02)
2. **COMPAT-X-R2** — Persisted Schema Migration (FRZ-CMP-03, FRZ-CMP-06)
3. **COMPAT-X-R3** — Event Evolution (FRZ-CMP-04)
4. **COMPAT-X-R4** — Plugin/Provider Compatibility (FRZ-CMP-05)
5. **COMPAT-X-R5** — Deprecation / Compatibility Shim Closure (FRZ-CMP-07, FRZ-CMP-08 hardening)

## Tenant isolation audit (local COMPAT-X-P0-R2)

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

## Unresolved findings (inventory blockers, not remediated in P0-R2)

- **5** `PERSISTED_SCHEMA_WITHOUT_VERSION` surfaces (table above) — **CHILD TASK REQUIRED** for COMPAT-X-R2 validation.
- All versioned surfaces still carry `VERSIONED_POLICY_MISSING` until **COMPAT-X-R1**.

**COMPAT-X parent:** **OPEN** — blocked on P0-R2 independent audit, then remediation waves.

**Roadmap:** `COMPAT-X-P0-R1` = **REJECTED / superseded by R2**; `COMPAT-X-P0` = **BLOCKED ON P0-R2 AUDIT**; `COMPAT-X` = **BLOCKED ON P0 CLOSURE**; `TENANT-X` = **NOT ENTERED**.
