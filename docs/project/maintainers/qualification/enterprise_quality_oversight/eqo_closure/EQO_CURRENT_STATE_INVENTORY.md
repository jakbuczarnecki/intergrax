# EQO current state inventory

> **RECONCILED 2026-11-27 (W2.1-D, OP-EC-01 OD-W2-DOCS continuation).**
> - Current: development == origin/development == 3190a4487; Wave 2 merges 79f52a89f, a3c6a9d43, f52eb70e8, b7002e24f, plus follow-up commits 7d2cc3d28 (ticket Lifecycle formatting) and 3190a4487 (INTELLIGENCE-X R2 docs, PROPOSED, not Wave 2 scope) are pushed.
> - PR #680 was MERGED 2026-11-21 into arch-freeze/rw-train as 65ef1f196, now an ancestor of development; this supersedes the lines below that place #680 only on rw-train or as an open draft. C16/C21 statuses are unchanged.
> - 8e146e21c (PR #715) is an ancestor of development; AD-AF-DEVTRAIN D-1 (MEP-AF2_v2.1 Appendix V6, signed 2026-11-26) makes development the train, so there is no separate train and any text below saying 'merge 8e146e21 into development forbidden', 'train only' or '164 commits ahead' is historical, not violations.
> - Pins recorded below (8cee3c5a9, 28c27cb89 and 258bbf6ab) are historical; the current development pin is 3190a4487.
> - CI at 7d2cc3d28 is red (0 new Wave 2 regressions; downstream jobs never ran): see `orchestration/EQO_W21_CI_DISPOSITION.md`. No G2/R/FCS/M3, not closed, no certification claim.
> - Body below is unchanged historical record.

**Snapshot UTC:** 2026-11-20T17:57:06Z

**Pins:** DOC `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` · TRAIN `28c27cb896d02e0185b9fa02047411fa152f6469` · Campaign A `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`

## Program verdict

- Campaign A dossier: **PASS_SCOPED** at baseline (not platform certification).
- Campaign B/C/D: no published dossier; execution artifacts in `eqo_closure/`.
- Product remediation: on **train only** (164 commits ahead of development).
- Open PR: **#680** (T-IGX-043-A, draft).

## Counts (mechanical)

| Object | Value |
|--------|------:|
| OPEN findings | 239 |
| CAMPAIGN_B handoff | 51 |
| CAMPAIGN_C handoff | 2 |
| FRZ OPEN (checklist column) | 109 / 225 |
| EDER rows | 1073 INGESTED |
| Known-red DRAFT | 28 |

See `EQO_PIN_BIND.json`, `EQO_ARTIFACT_RECOVERY_REGISTRY.jsonl`, and sibling plan/WBS files.
