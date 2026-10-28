# EBH-5 — SQLite mandatory annex

**AUDIT_BASELINE_SHA:** `83a376b3e1b312e2e4f88da548c99c328f3589f7`

**Touchpoint SSOT:** [`_sqlite_touchpoint_registry.py`](../../../tests/qualification/ebh_5/_sqlite_touchpoint_registry.py) — **38** paths, closed-world vs `discover_sqlite_connect_paths_under_intergrax()`.

## Summary classification

| Group | RPL-H | Notes |
| --- | --- | --- |
| Integration `relational_store/sqlite` | REPLACEABLE — QUALIFIED | EBH-5 L3 `test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile` |
| Runtime persistence SQLite impls | FIXED SUBSTRATE — RATIFIED | Checkpoint, human, idempotency, trace, agent stores — Tier-3 composition |
| `single_node_production_profile_v1` triad | FIXED SUBSTRATE — RATIFIED | GCA/PIS/AD paths under DATA_DIR |
| Lab adaptive signal/proposal | INTERNAL | Lab factory scoped paths (GAP-01 remediated) |
| `distributed.sqlite_kv` | REPLACEABLE — NOT YET QUALIFIED | Alternate via `KEY_VALUE_CACHE` integration — L3 debt |

## Per-touchpoint checklist (8 questions)

Each inventoried SQLite concern in SSOT answers:

1. Intentional adapter — **yes** for integration sqlite; **yes** for runtime port implementations.
2. Canonical contract — **yes** where `VendorBackendConcernRecord.canonical_contract_or_port` is set.
3. Alternate via sanctioned composition — **integration relational yes**; runtime ports **no** (fixed substrate).
4. Core/scenario change to switch — **config-only** (integration) or **adapter-only / BLOCKER** (fixed).
5. Guarantees without SQLite-only semantics — expressed on port contracts (STATE-X evidence); not re-proven in EBH-5 L4.
6. Configuration-driven — integration **yes**; profile v1 **FIXED paths**.
7. Ratified if fixed — see [`EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md`](EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md).
8. DIST-SCALE-X multi-process — **scoped acceptance** @ `df485a96…`; WL-7 **14400 s NOT PROVEN** (deferred non-blocking).

**Working SQLite alone is not replaceability evidence** — qualified rows require matrix + behavioral proofs.
