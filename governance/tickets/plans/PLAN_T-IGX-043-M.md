# PLAN T-IGX-043-M — ENT-AUDIT Part E scaffold (ARCH-FREEZE development tor)

Status: **draft scaffold rev 1** (2026-11-21, autonomy fala 2). SSOT pointer: sibling **T-IGX-043-I** on `arch-freeze/rw-train` (Parts A–D); this ticket is the **ABSENT** Part E recovery named in `EQO_ARTIFACT_RECOVERY_REGISTRY.jsonl`.

## 1. Purpose

Close the ENT-AUDIT-X wave-diff and manifest-consumer gap so Campaign A ENT findings (A6-003..A6-008 family) can complete on the development tor without waiting for train merge.

## 2. Prerequisites

- **T-IGX-043-I** Parts A–D concepts available on train (read-only reference); no requirement to merge train.
- **T-IGX-043-B** `wiring_posture.py` on `development` (FC-2 dependency for unrelated tickets; 043-M does not edit it).
- FC-3 regression batch 78 green for scope in `regression_scope.txt` (fala 2 follow-up).
- RW-PLAN-AUDIT **BLOCKED** — plan does not claim SCOPE closure.

## 3. Write set (indicative — detail in rev 2)

| Area | Paths (indicative) |
|------|---------------------|
| Wave-diff / impact map | `intergrax/platform/enterprise_audit/changed_since.py`, `data/changed_since_impact_map.v1.json` (regenerated) |
| Manifest consumers | modules named by Part E split on train (TBD node ids) |
| Scripts | `scripts/enterprise_audit/build_impact_map.py`, `run_enterprise_audit.py` (Ref trailers) |
| Tests | `tests/qualification/ent_audit_x/unit/` (new / extended; node ids TBD) |

## 4. Test contract (placeholder)

- Command 1: ENT-AUDIT Part E unit package (name TBD in rev 2).
- Command 2: regression slice from `regression_nodes.txt` or EQO split2 list (architect copies exact node ids).
- `uv run python -m scripts.governance.igx_gov ticket` before push.

## 5. Out of scope

Check manifest rewrite, IIAC baseline promotion, train merge, FRZ promotion.

## 6. Next steps (operator / architect)

1. Copy Part E section from train plan `PLAN_T-IGX-043-I.md` (when published on development) into rev 2 with file-level diffs.
2. Record plan digest; `plan_pending` → operator/delegate approval.
3. Implement in worktree `af-dev-043m` after FC-3 regression fala 2 is green.