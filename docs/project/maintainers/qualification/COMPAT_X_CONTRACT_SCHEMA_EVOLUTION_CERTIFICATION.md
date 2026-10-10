# COMPAT-X — Contract, Schema & Evolution Certification (P0 baseline)

**Status:** COMPAT-X-P0 — **READY FOR AUDIT** (Cursor implementation; not COMPAT-X parent closure)

**Branch:** `development`

**START_HEAD:** `65fe8d4255136a67fb14cbacfa1b436be71a11ba`

**Production delta:** `0`

## Purpose

Mechanical closed-world inventory of compatibility-relevant platform surfaces on current HEAD. Establishes evidence for **FRZ-CMP-01..08** *candidate* state only. Does **not** close **COMPAT-X** or promote global **FRZ-CMP-* = PASS**.

## Inventory summary (mechanical)

| Metric | Count |
| --- | ---: |
| Discovered compatibility surfaces | 508 |
| Classified inventory surfaces | 508 |
| Unclassified (`EvolutionState.UNCLASSIFIED`) | 0 |
| Public/stable facet | 9 |
| Persisted schema facet | 17 |
| Event schema facet | 38 |
| Plugin/provider contract facet | 16 |
| Distribution package contract facet | 62 |
| Compatibility shim facet | 1 |

**SSOT:** `tests/qualification/compat_x/_compat_x_inventory.py` (`COMPAT_X_INVENTORY`)

**Discovery:** `tests/qualification/compat_x/_compat_x_discovery.py`

**Gate:** `tests/qualification/compat_x/test_compat_x_inventory_gates.py` — `discovered == classified`

## Canonical registries (complementary owners)

| Registry | Path | Responsibility |
| --- | --- | --- |
| `CONTRACT_SCHEMA_REGISTRY` | `intergrax/contracts/migrations/registry.py` | Tier-0 **public contract** schema ids for agent run, checkpoints, side effects, org policy envelopes |
| `RUNTIME_SCHEMA_REGISTRY` | `intergrax/runtime/schema/registry.py` | **Runtime persisted/wire** schema bundle ids + `validate_schema_version` |
| Event payload registry | `intergrax/runtime/events/payload_registry.py` | **Runtime event payload** `payload_schema_id` families |

**Architecture note (P0):** registries are **complementar**, not duplicates. No merge proposed in P0.

## Owner matrix

See `tests/qualification/compat_x/_compat_x_owner_discovery.py` (`COMPAT_X_OWNER_MATRIX`).

| Concern | Semantic owner |
| --- | --- |
| Public contract identity/version | `intergrax/contracts/migrations/registry.py` |
| Runtime schema registry | `intergrax/runtime/schema/registry.py` |
| Persisted schema migration | Split: contracts registry + local format migrations (**COMPAT-X-R2**) |
| Event schema/evolution | `intergrax/runtime/events/payload_registry.py` |
| Platform plugin manifest | `intergrax/core/plugins/package_contract.py` |
| Provider contract compatibility | `intergrax/integrations/registry/catalog.py` |
| External contract semantic assessment | `intergrax/integrations/contracts/external_contract_compatibility.py` (advisory; not global version registry) |
| Deprecation/removal | **UNOWNED — COMPAT-X-R5** |
| Compatibility adapters | `intergrax/compat/langchain/documents.py` |

## FRZ-CMP candidate state (P0)

| Criterion | Candidate | Blocker IDs |
| --- | --- | --- |
| FRZ-CMP-01 | **PASS CANDIDATE** | — |
| FRZ-CMP-02 | **BLOCKED** | CMP-P0-B02 |
| FRZ-CMP-03 | **BLOCKED** | CMP-P0-B03 |
| FRZ-CMP-04 | **BLOCKED** | CMP-P0-B04 |
| FRZ-CMP-05 | **BLOCKED** | CMP-P0-B05 |
| FRZ-CMP-06 | **BLOCKED** | CMP-P0-B06 |
| FRZ-CMP-07 | **BLOCKED** | CMP-P0-B07 |
| FRZ-CMP-08 | **PASS CANDIDATE** | — (shim inventory only; parent closure still requires full COMPAT-X) |

## Proposed remediation grouping

1. **COMPAT-X-R1** — Contract Versioning & Classification (FRZ-CMP-02)
2. **COMPAT-X-R2** — Persisted Schema Migration (FRZ-CMP-03, FRZ-CMP-06)
3. **COMPAT-X-R3** — Event Evolution (FRZ-CMP-04)
4. **COMPAT-X-R4** — Plugin/Provider Compatibility (FRZ-CMP-05)
5. **COMPAT-X-R5** — Deprecation / Compatibility Shim Closure (FRZ-CMP-07, FRZ-CMP-08 hardening)

## Tenant isolation audit (local COMPAT-X-P0)

**Verdict:** **PASS** (local scope; not global **TENANT-X**)

- **INT-EXTCOMP-X** invariant preserved: `expectation.tenant == evidence.tenant == resolver-key.tenant == assessment.tenant` (reuse `tests/qualification/external_contract_compatibility/`).
- Inventory marks `mechanism.external_contract_compatibility` as tenant-relevant.
- Adversarial probe: LangChain compat bridge rejects invalid `schema_version` without silent default tenant widening (`test_cx_p0_adversarial_09_*`).
- **Global FRZ-TEN-* promotion:** `0`

## Test evidence

```text
uv run pytest -p no:xdist tests/qualification/compat_x/
uv run pytest -p no:xdist tests/unit/runtime/schema/test_schema_registry_b07.py
uv run pytest -p no:xdist tests/qualification/external_contract_compatibility/test_external_contract_compatibility_certification.py
uv run pyright tests/qualification/compat_x
```

P0 session: **18** compat_x tests **PASS**; related registry + extcomp **34** **PASS**; pyright compat_x **0** errors.

## Enterprise audit matrix (P0 scope)

| Item | Result |
| --- | --- |
| Contracts over implementations | PASS (inventory cites contract owners) |
| Hard boundaries | PASS (no production code changes) |
| Semantic ownership | BLOCKED (deprecation/migration partially unowned) |
| Composition ownership | PASS (matrix documents sanctioned paths) |
| Strong typing | PASS (typed inventory records; no dict pseudo-contract) |
| Pluginability | PASS (distinct plugin vs external compat owners) |
| Replaceability | N/A — WITH EVIDENCE (P0 discovery only) |
| Duplicate mechanisms | PASS (CONTRACT vs RUNTIME registry classified complementary) |
| Compatibility shims | PASS CANDIDATE (1 shim, no parallel authority in inventory) |
| Version authority | BLOCKED (policy missing — R1) |
| Migrations | BLOCKED (R2) |
| Event evolution | BLOCKED (R3) |
| Persisted schema evolution | BLOCKED (R2/R3) |
| Plugin/provider compatibility | BLOCKED (R4) |
| Governance boundary | PASS (no governance change) |
| Execution authority | PASS (delta 0) |
| Tenant-local invariant | PASS |
| Evidence/trace continuity | PASS (reuse TRACE-X / INT-EXTCOMP evidence) |
| Fail-closed unknown versions | PASS (probes on runtime + event payload) |
| Regression protection | PASS (mechanical gates) |

**COMPAT-X parent:** remains **OPEN** — P0 is inventory baseline only.
