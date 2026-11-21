# EQO-X-05 — TENANT-X drift read (C-03 + WB02 cross-check)

**Finding:** EQOA-A3-012 (HIGH, `REQUALIFY`)  
**Owner (disposition):** TENANT-X / PDC-X D2  
**Task:** link Campaign C drift row with WB02 tenant slice; closure remains **PT-Q07** post-FCS (no duplicate recertification).

## EQO-C-03 layer row (TENANT-X)

From `EQO_C_DRIFT.jsonl` @ branch tip:

| Field | Value |
|-------|-------|
| `layer_id` | TENANT-X |
| `historical_sha_source` | roadmap ledger (read-only) |
| `diff_a_baseline_train_pin` | `files_changed=0 product_changed=0` |
| `diff_a_baseline_doc_pin` | `files_changed=0 product_changed=0` |
| `drift_class` | `NO_PRODUCT_DELTA` |

**Note:** Mechanical surface diff vs `A_BASELINE` (`258bbf6`) shows no product delta on either pin; this does **not** resolve A3-012 certification-drift claim (historical cert SHA vs later tenant-bearing paths — see dossier A5-2).

## WB02 probe cross-reference

Pre-FCS WB02 execution on `TRAIN_PIN` is recorded in:

- `eqo_closure/runs/wb02/TENANT_ISOLATION_EVIDENCE.md`
- `eqo_closure/runs/wb02/pytest_wb02_exe.log` (66 passed, 1 skipped)

WB02 does **not** own A3-012 (A3-012 is X-05 / REQUALIFY, not in the 51 Campaign-B IDs). WB02 interim dispositions for tenant findings remain `ACCEPTED_SCOPED_LIMITATION` / `BLOCKED` as documented there.

## Dossier anchor (read-only)

Campaign A title: TENANT-X global tenant-isolation PASS not maintained after certification; drift gate pinned to historical SHA with open untenanted-registry paths (`campaign_a/shards/A5/A5-2.findings.jsonl`).

## Disposition (interim)

`NOT_CLOSED` — C-03 row + WB02 slice linked; full tenant requalification owned by **PT-Q07**. EQO does not assert INDEPENDENTLY ACCEPTED.

**Audit:** session E pending.