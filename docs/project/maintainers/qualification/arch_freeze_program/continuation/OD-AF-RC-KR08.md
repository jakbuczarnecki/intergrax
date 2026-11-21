# OD-AF-RC-KR08 — merge mandate for red `governance-fast` on the RW train

**Status:** `PENDING_OPERATOR` (not decided in RC-1).

**Question (plan section F):** May the orchestrator merge train PRs while `governance-fast` is red, when the failure is exactly the ACT-N11 stale-audit class (KR-08), before FCS?

**Program recommendation:** TAK, narrowly — train only; not `development`; not FCS/M3; no ruleset bypass; stale ACT-N11 on product paths touched since the last audit SHA only; closure remains PT-Q03.

**Evidence cited in plan:**

- Ruleset on `development`: no required status checks (2026-11-20 API read).
- PR #680 (`af2/pt-fc1`) carries red `governance-fast` (KR-08).
- Standing record 2 does not grant merge despite red CI.

**RC-1 action:** this file records the open decision. No merge of #680 or any other red-governance train PR was performed. Orchestrator STOP until the Operator answers.

**Consequence if NO:** RB-2 / PT-Q03 must green `governance-fast` before merge of #680 and downstream RC-2 work that depends on FC1.

**Consequence if TAK:** merge #680 only after digest review and delegated ticket rules; other reds still block.