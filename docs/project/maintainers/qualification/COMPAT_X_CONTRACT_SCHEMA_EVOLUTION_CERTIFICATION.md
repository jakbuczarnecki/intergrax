# COMPAT-X — Contract, Schema & Evolution Certification (P0-R1 baseline)

**Status:** COMPAT-X-P0-R1 — **READY FOR AUDIT** (Cursor implementation; not COMPAT-X parent closure)

**Branch:** `development`

**START_HEAD (P0-R1 session):** `88b2844f565f1869e61b3bb97305b623eb10f295`

**IMPLEMENTATION_COMMIT (P0-R1 Cursor):** `3c8f7ba76d1c2600fcb7a1891b4cc9223efa23f8`

**Rejected / incomplete P0 evidence (do not use as closure baseline):** `cea6393775c8b92a2cefb165fc9930511c776169`

**P0-R1 reason:** Independent audit rejected P0 closed-world proof (AST-only constants, hard-coded migration/shim lists, tautological adversarial tests, owner matrix overstated). R1 repairs qualification/discovery only (**production delta = 0**).

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

**Gate:** `tests/qualification/compat_x/test_compat_x_inventory_gates.py` — parity + adversarial suite

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

## FRZ-CMP candidate state (P0-R1)

| Criterion | Candidate | Blocker IDs |
| --- | --- | --- |
| FRZ-CMP-01 | **PASS CANDIDATE** | — (closed-world parity + semantic inventory) |
| FRZ-CMP-02 | **BLOCKED** | CMP-P0-B02 → COMPAT-X-R1 |
| FRZ-CMP-03 | **BLOCKED** | CMP-P0-B03 (+ 5 `PERSISTED_SCHEMA_WITHOUT_VERSION` heuristic findings — **CHILD TASK REQUIRED** for validation) → COMPAT-X-R2 |
| FRZ-CMP-04 | **BLOCKED** | CMP-P0-B04 → COMPAT-X-R3 |
| FRZ-CMP-05 | **BLOCKED** | CMP-P0-B05 → COMPAT-X-R4 |
| FRZ-CMP-06 | **BLOCKED** | CMP-P0-B06 → COMPAT-X-R2 |
| FRZ-CMP-07 | **BLOCKED** | CMP-P0-B07 → COMPAT-X-R5 |
| FRZ-CMP-08 | **PASS CANDIDATE** | — (0 `PARALLEL_AUTHORITY` in production inventory; synthetic classifier probe) |

## Proposed remediation grouping (re-derived from corrected discovery)

1. **COMPAT-X-R1** — Contract Versioning & Classification (FRZ-CMP-02)
2. **COMPAT-X-R2** — Persisted Schema Migration (FRZ-CMP-03, FRZ-CMP-06)
3. **COMPAT-X-R3** — Event Evolution (FRZ-CMP-04)
4. **COMPAT-X-R4** — Plugin/Provider Compatibility (FRZ-CMP-05)
5. **COMPAT-X-R5** — Deprecation / Compatibility Shim Closure (FRZ-CMP-07, FRZ-CMP-08 hardening)

## Tenant isolation audit (local COMPAT-X-P0-R1)

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

P0-R1 session: **22** compat_x tests **PASS**; related registry + extcomp **34** **PASS**; pyright compat_x **0** errors.

## Unresolved findings (inventory blockers, not remediated in P0-R1)

- **5** `PERSISTED_SCHEMA_WITHOUT_VERSION` surfaces from mechanical wire/persist heuristic (may include false positives — requires COMPAT-X-R2 validation).
- All surfaces still carry `VERSIONED_POLICY_MISSING` until **COMPAT-X-R1**.

**COMPAT-X parent:** **OPEN** — blocked on P0-R1 independent audit, then remediation waves.
