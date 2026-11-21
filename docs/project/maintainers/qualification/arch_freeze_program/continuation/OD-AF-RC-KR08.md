# OD-AF-RC-KR08 — merge mandate for red `governance-fast` on the RW train

**Status:** `PENDING_OPERATOR` (not decided in RC-1 on train package).

**Question (plan section F):** May the orchestrator merge train PRs while `governance-fast` is red, when the failure is exactly the ACT-N11 stale-audit class (KR-08), before FCS?

**Program recommendation:** TAK, narrowly — train only; not `development`; not FCS/M3; no ruleset bypass; stale ACT-N11 on product paths touched since the last audit SHA only; closure remains PT-Q03.

**RC-1 action:** this file records the open decision on the **train admission** package (`af2/rc1-continuation` @ `36c470249`).

## Tor `development` — cross-reference (G4)

Operator **OD-AF-AUTO-01** w [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) rejestruje **TAK** (wąsko) dla kolejnych **product** PR na train przy ACT-N11 — **nie** dla samych docs. Merge PR **#699** (2026-11-21) przy czerwonym `governance-fast` jest **faktem operatora/automation** — [`CHECKPOINT.md`](CHECKPOINT.md), [`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md); to **nie** zastępuje formalnego podpisu tego rekordu na trainie ani RB-2 na `development`.

**Consequence if NO (train):** RB-2 / PT-Q03 must green `governance-fast` before merge of #680 and downstream RC-2 work that depends on FC1.

**Consequence if TAK (train):** merge only after digest review and delegated ticket rules; other reds still block.