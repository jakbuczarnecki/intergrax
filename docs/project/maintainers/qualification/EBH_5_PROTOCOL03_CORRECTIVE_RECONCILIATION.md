# EBH-5 — Protocol 03 corrective batch reconciliation

| Field | Value |
| --- | --- |
| **Disposition** | EBH-5 — CORRECTIONS REQUIRED / PARENT OPEN |
| **AUDIT_BASELINE_SHA** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **Corrective status** | **READY FOR INDEPENDENT RE-AUDIT** (post-commit publication SHA) |
| **Parent CLOSED** | **NO** |

See successor dossier: [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md).

## Corrective items (EBH5-A01..A05)

| ID | Disposition |
| --- | --- |
| **A01** | `REPLACEABLE — QUALIFIED` requires resolvable pytest node under `tests/qualification/ebh_5/` bound to concern via `_behavioral_proof_registry.py` + `_proof_validation.py`; negative gates in `test_ebh_5_proof_integrity_gates.py`. |
| **A02** | Five concerns re-bound to L3 tests in `test_ebh_5_behavioral_proofs.py` (incl. new `test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service`); historical INT-CONFIG / HARNESS citations moved to supplemental notes only. |
| **A03** | All **38** `sqlite3.connect` touchpoints in closed-world `_sqlite_touchpoint_registry.py`; `test_ebh5_q09_sqlite_touchpoints_fully_accounted`. |
| **A04** | `AUDIT_BASELINE_SHA`, `proof_execution_sha()` = current HEAD at run time; SHA format vs git object distinguished in proof gates. |
| **A05** | Inventory/matrix **5** QUALIFIED / **51** NOT YET (matrix rows); inventory **69** concerns; dossier counts aligned; pytest expectation updated. |

## Counts (SSOT @ corrective implementation)

| Metric | Count |
| --- | --- |
| Inventory concerns | 69 |
| Matrix rows (extensibility promised) | 56 |
| `REPLACEABLE — QUALIFIED` | 5 |
| `REPLACEABLE — NOT YET QUALIFIED` (matrix) | 51 |
| SQLite touchpoints accounted | 38 |

## Verification

```powershell
uv run pytest tests/qualification/ebh_5/ -q
```

**Expected after corrective batch:** 46 passed, 1 skipped (reachability graph module optional).

## SAC / FRZ

- SAC proposal continuity unchanged — no competing SAC.
- **FRZ / ECAP:** evidence hardening only; **no** new global FRZ PASS; **no** parent CLOSED.
