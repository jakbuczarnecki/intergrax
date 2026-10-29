# ROADMAP-REPLAY-X — ECAP current-HEAD audit report (§14)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **Parent status** | **READY FOR INDEPENDENT AUDIT** — **NOT CLOSED** |

## C1 — Closed-world inventory

[`ROADMAP_REPLAY_X_CLOSED_WORLD_INVENTORY.md`](ROADMAP_REPLAY_X_CLOSED_WORLD_INVENTORY.md) — Tier 0–3; unclassified **0** in replay scope.

## C2 — Ownership / duplicate authority

Prod Δ empty since DUP-X; DUP-X + EBH-6 SSOT reused; NPSC ten-module disposition recorded.

## C3 — VBN taxonomy (31 domains)

Seven replay artifacts @ normative schema (§2.1–2.7) + EBH-5/6 historical SHA citations.

## C4 — Drift vs accepted baselines

[`ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md`](ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md) — no parent reopen from prod.

## C5 — Qualification replay plan

[`ROADMAP_REPLAY_X_QUALIFICATION_MANIFEST.md`](ROADMAP_REPLAY_X_QUALIFICATION_MANIFEST.md) — **L2/L3 OPEN** until `TEST_EXECUTION_SHA` on committed tree.

## Six counters (mechanical @ dirty worktree)

Computed by `tests/qualification/roadmap_replay_x/_closure_counters.py` — **B6 must recompute**.

| Counter | Value | Denominator / source |
| --- | ---: | --- |
| reopened_historical_stages | 0 | §5 category A rows + reopen grep |
| unresolved_replay_blockers | 0 | ECAP register (implementer) |
| roadmap_drift | 0 | `_roadmap_drift.py` |
| historical_evidence_invalidated_by_later_changes | 0 | prod git diff since DUP-X |
| unclassified_replay_findings | 0 | all §5 rows carry rule_id |
| vendor_neutrality_replay_failures | *PROHIBITED OPEN* | EBH-6 register scan |

**Note:** `unclassified_replay_findings` uses category **I** ledger rows as proxy until ECAP findings register populated @ B6.

## External gates (OPEN)

| Gate | Status |
| --- | --- |
| B6 Protocol 03 | **OPEN** |
| Operator OD | **OPEN** |
| L2 parent replay @ TEST_EXECUTION_SHA | **OPEN** |
| L3 ECAP bundle @ TEST_EXECUTION_SHA | **OPEN** |
| EBH-7 | **NOT ENTERED** |
