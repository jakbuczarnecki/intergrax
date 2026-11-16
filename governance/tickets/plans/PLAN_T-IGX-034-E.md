## PLAN `T-IGX-034-E` — PRE-ENT Wave 1 REPLAY mechanical gates

Campaign: PRE-ENT predecessor certification closure. Reconcile ROADMAP-REPLAY-X mechanical exit gates on `development` after historical product batches 030-B/C, 032-A, 033-A landed post DUP-X.

### 1. Problem

`test_roadmap_replay_x_mechanical_exit_gates.py` RED @ HEAD: `historical_evidence_invalidated_by_later_changes` = 2 (`unscoped_prod_delta:intergrax` + ECAP vs mechanical reconciliation). Semantic cause: 84 prod paths since DUP-X absent from `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` — not `.claude` tree drift.

### 2. Scope (qualification + governance only)

| Area | Action |
| --- | --- |
| `tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py` | SSOT path batches + dispositions @ `4a932d0c` |
| `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` | Merge L2 CTRL-X + PRE-ENT L3 rows |
| `docs/.../PRE_ENT_REPLAY_L3_PROD_DELTA.md` | Operator-readable batch map |
| `ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md` | Unscoped empty @ PRE-ENT head |
| `ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md` | C2 scoped delta narrative; counters unchanged @ 0 |

No FRZ row promotion. No parent `CLOSED` / PASS claims.

### 3. Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

### 4. Rollback

Revert disposition modules + PRE-ENT qual docs; restore prior `_prod_delta_disposition.py` single-path L2 scope.
