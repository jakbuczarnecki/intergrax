# DIST-SCALE-X P4-A — Physical run provenance

## Disposition

**DIST-SCALE-X P4-A — READY FOR FINAL INDEPENDENT AUDIT** (WL-5 gate seq corrected @ `f1389fdaa4d69fb343dfc42e2040c2f8ebb2c39b`; F/G/H evidence reused from prior batch @ `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845`; parent DIST-SCALE-X remains **OPEN**).

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA (P4-A initial)** | `4d2cbd40eb503d1eb418b7d82a884b29083ca5e9` |
| **IMPLEMENTATION_SHA (post-batch fixes)** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **IMPLEMENTATION_SHA (WL-5 gate seq fix)** | `f1389fdaa4d69fb343dfc42e2040c2f8ebb2c39b` |
| **EXECUTION_SHA (authoritative WL-5 physical)** | `f1389fdaa4d69fb343dfc42e2040c2f8ebb2c39b` |
| **EXECUTION_SHA (prior batch F/G/H/C02–C05)** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **PUBLICATION_SHA** | `8977df72d00a5c11f63e7888b58fcc12fdf33d29` |

Intermediate commits on `development`: `b965f05a5`, `e3519de78`, `9f0c53e2c36e3549de9776fbeef44e9c14bc60c2` (pre-correction HEAD).

## Physical run (prior full batch)

| Field | Value |
| --- | --- |
| **run_id** | `p4-a-20261027-094853` |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p4_qualification_batch.py -q --tb=short` |
| **Log** | `.tmp/proof/dist-scale-x/p4-a-20261027-094853/run.log` |
| **Outer batch** | 1 passed (~392.7s) |
| **Inner L3/L4 selection** | 10 passed (`p4_physical_faults`, `p4_physical_crash`, `p4_physical_restart`, `wl_05_persistence_heavy`) |
| **Preflight L1/L2** | 28 passed (structural gates, same resume session) |

Failed attempt (diagnostic): `p4-a-20261027-091225` @ `4d2cbd40…` — task_id mint, F-STORAGE double-unpause, F03 inspect key.

## WL-5 gate seq correction (C01 re-qualification)

**Root cause:** `_offer_phase()` reset `seq=0` on each invocation; SQLite `INSERT OR IGNORE` dropped measurement offers that collided with warmup seq keys.

**Fix:** `Wl5RateGate.offer_next()` — monotonic seq across warmup + measure; orchestrator uses `offer_next()` in `_offer_phase()`.

| Field | Value |
| --- | --- |
| **run_id** | `wl05-p4-3e570bc693b6` |
| **proof log** | `.tmp/proof/dist-scale-x/p4-a-wl5-correct-20261027-101520/run.log` |
| **manifest** | `.tmp/proof/dist-scale-x/wl05-p4-3e570bc693b6/dist_scale_x_evidence_manifest.json` |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/wl_05_persistence_heavy.py::test_wl05_persistence_heavy_baseline -q --tb=short` |
| **Duration** | ~271.3s (60s warmup + 120s measure envelope) |
| **Workers** | OS PIDs `4428`, `14868`; tenants `tenant-a`, `tenant-b`; writers=2 |
| **warmup offered / completed (writes+reads)** | 2400 / 549 (479 writes, 70 reads) |
| **measure offered / completed (writes+reads)** | 4800 / 861 (746 writes, 115 reads) |
| **measure read share** | 0.134 (target 0.10 ±0.05) |
| **errors** | warmup 0, measure 0 |
| **correctness violations** | none |
| **Regression** | `test_p4_wl5_gate_phase_seq.py` — PASS |

Prior C01 @ `f0c49fd3…` **invalidated** for WL-5 offer accounting only; not re-run: C03 (G), C05 (F), F/STORAGE scenarios (shared contracts unchanged).

## Corrective batch scope (C01–C05)

| ID | Status |
| --- | --- |
| **C01** WL-5 full 60s+120s, gate, N=2 OS workers, F01–F09 op | **PASS** @ `f1389fdaa4d69fb343dfc42e2040c2f8ebb2c39b` (re-qual after seq fix) |
| **C02** F01/F03/F07/F09 in `p4_persistence_op` | **Delivered** |
| **C03** G crash matrix | **PASS** @ `f0c49fd3…` — **reused** |
| **C04** PG contract unit tests | **Delivered** (unit layer; not re-run in this physical batch) |
| **C05** F-EVENT qual test; F-STORAGE docker pause | **PASS** @ `f0c49fd3…` — **reused** |

## Infra note

Physical tests require **psycopg handshake** to qual DSN (`INTERGRAX_POSTGRESQL_DSN` or localhost:5434) and **Docker Desktop** for `intergrax-postgresql` pause/unpause (F-STORAGE).

## P4 / parent status

- **P4-A:** WL-5 **PASS** @ `f1389fdaa4d69fb343dfc42e2040c2f8ebb2c39b`; F/G/H **reused** @ `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845`.
- **DIST-SCALE-X parent:** **OPEN**
