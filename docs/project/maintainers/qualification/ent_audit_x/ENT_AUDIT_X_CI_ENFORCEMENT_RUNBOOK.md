# ENT-AUDIT-X — CI enforcement runbook (OD-ENT-02)

**Blocker:** `BLK-CI-ENFORCE` — `development` branch protection not configured (gh API 404 @ planning wave).

## Required checks (target)

| Check | Workflow |
| --- | --- |
| Enterprise audit FAST (real) | `enterprise-audit.yml` job `ent-audit-fast` — pytest L1/L2/L3 + CLI FAST without `--dry-run` |
| Enterprise audit nightly | `enterprise-audit-nightly.yml` — qual suite + **FULL/FREEZE real** (dry-run **not** qualification) |
| Unit regression (existing) | `unit-tests.yml` |

## Operator steps

1. GitHub → Settings → Branches → protect `development`.
2. Require status check **Enterprise audit (ENT-AUDIT-X) / ent-audit-fast**.
3. Record completion in operator record; update FRZ-AUD-10 only after Protocol 03 B6.

**Honest status until OD-ENT-02:** FRZ-AUD-10 enforceability **BLOCKED** — workflow exists; branch protection **MISSING**.
