# DIST-SCALE-X P4-A — Physical run provenance

## Disposition

**DIST-SCALE-X P4-A — READY FOR INDEPENDENT AUDIT** (scoped: P4 L3/L4 batch @ `EXECUTION_SHA`; parent DIST-SCALE-X remains **OPEN**).

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA (P4-A initial)** | `4d2cbd40eb503d1eb418b7d82a884b29083ca5e9` |
| **IMPLEMENTATION_SHA (post-batch fixes)** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **EXECUTION_SHA (authoritative physical)** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **PUBLICATION_SHA** | `d62ba2070338735137b315f26c6019197578e276` |

Intermediate fix commits on `development` (unpushed): `b965f05a5`, `e3519de78`.

## Physical run

| Field | Value |
| --- | --- |
| **run_id** | `p4-a-20261027-094853` |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p4_qualification_batch.py -q --tb=short` |
| **Log** | `.tmp/proof/dist-scale-x/p4-a-20261027-094853/run.log` |
| **Outer batch** | 1 passed (~392.7s) |
| **Inner L3/L4 selection** | 10 passed (`p4_physical_faults`, `p4_physical_crash`, `p4_physical_restart`, `wl_05_persistence_heavy`) |
| **Preflight L1/L2** | 28 passed (structural gates, same resume session) |

Failed attempt (diagnostic): `p4-a-20261027-091225` @ `4d2cbd40…` — task_id mint, F-STORAGE double-unpause, F03 inspect key.

## Corrective batch scope (C01–C05)

| ID | Status |
| --- | --- |
| **C01** WL-5 full 60s+120s, gate, N=2 OS workers, F01–F09 op | **PASS** @ `EXECUTION_SHA` |
| **C02** F01/F03/F07/F09 in `p4_persistence_op` | **Delivered** |
| **C03** G crash matrix | **PASS** — staged F01→F09 + post-kill PG inspect |
| **C04** PG contract unit tests | **Delivered** (unit layer; not re-run in this physical batch) |
| **C05** F-EVENT qual test; F-STORAGE docker pause | **PASS** — injected F07; F-STORAGE writes under `docker pause` |

## Infra note

Physical tests require **psycopg handshake** to qual DSN (`INTERGRAX_POSTGRESQL_DSN` or localhost:5434) and **Docker Desktop** for `intergrax-postgresql` pause/unpause (F-STORAGE).

## P4 / parent status

- **P4-A:** physical batch **PASS** @ `EXECUTION_SHA`; evidence publication on `development` (no push unless operator requests).
- **DIST-SCALE-X parent:** **OPEN**
