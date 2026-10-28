# EBH-5 — SQLite mandatory annex

**Evidence SHA:** `5c9b66db632976ada039d406474530d7c6cb5f86` (refresh at closure)

**Discovery:** `discover_sqlite_connect_paths_under_intergrax()` in [`_closed_world_discovery.py`](../../../tests/qualification/ebh_5/_closed_world_discovery.py).

## Summary classification

| Group | RPL-H | Notes |
| --- | --- | --- |
| Integration `relational_store/sqlite` | REPLACEABLE — QUALIFIED | INT-CONFIG-REAL-X + sanctioned `IntegrationProfile` |
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
