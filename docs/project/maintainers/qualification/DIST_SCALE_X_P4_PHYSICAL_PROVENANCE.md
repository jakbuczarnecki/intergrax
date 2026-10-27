# DIST-SCALE-X P4-A — Physical run provenance

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA (P4-A initial)** | `04329842a0ffc3f62eb928dbf85a49919cf5fb10` |
| **PUBLICATION_SHA (initial evidence)** | `bf3b21df115a271e2a4332b81644d3dca54d4bab` |
| **IMPLEMENTATION_SHA (Protocol 03→04 correction)** | `eade40553` (full: `eade40553…` @ `git rev-parse eade40553`) |
| **IMPLEMENTATION_SHA (final differential correction)** | **UNCOMMITTED** @ working tree (post-`fbbfa381`) |
| **EXECUTION_SHA (authoritative physical)** | **PENDING** — qual PG + Docker required |
| **PUBLICATION_SHA** | **PENDING** — no push until EXECUTION_SHA recorded |

## Preflight (structural, no PG)

| Command | Result |
| --- | --- |
| L1/L2 catalog + evidence gates | 28 passed @ correction session |
| P4 batch (`test_dist_scale_x_p4_qualification_batch`) | Requires live PostgreSQL |

## Corrective batch scope (C01–C05)

| ID | Status |
| --- | --- |
| **C01** WL-5 full 60s+120s, gate, N=2 OS workers, F01–F09 op | **Code complete**; authoritative PASS **pending** PG re-run |
| **C02** F01/F03/F07/F09 in `p4_persistence_op` | **Delivered** |
| **C03** G crash matrix | **Code complete** — staged F01→F09 + post-kill PG inspect (`p4_staged_persistence`, `p4_pg_state_inspection`); **PASS pending** live PG |
| **C04** PG contract unit tests | **Delivered** (`test_postgresql_task_checkpoint_store_contract.py`, F03 tests) |
| **C05** F-EVENT qual test; F-STORAGE docker pause | **Code complete** — F-EVENT injected; F-STORAGE **fail** if no write errors (no skip-as-PASS); **PASS pending** Docker+PG |

## Production delta (correction)

- Concurrent DDL: `pg_advisory_xact_lock` in F01/F03 `_ensure_schema`.
- Qual: `prime_p4_qualification_stores`, `Wl5RateGate`, WL-5 worker driver, schema race handling.

## Infra note

Physical tests require **psycopg handshake** to qual DSN (`INTERGRAX_POSTGRESQL_DSN` or localhost:5434) and, for F-STORAGE, **Docker Desktop** for `intergrax-postgresql` pause/unpause.

## P4 / parent status

- **P4-A:** correction landed; **EXECUTION_SHA** not recorded on PG-down workstation.
- **DIST-SCALE-X parent:** **OPEN**
