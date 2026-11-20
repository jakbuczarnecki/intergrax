# ARCH-FREEZE continuation checkpoint — RC-0

Date: 2026-11-20. Package: RC-0 evidence and baseline only.

This file records what was read and what was not claimed. It does not declare G2, FCS, M3, or ARCH_FREEZE_CLOSED. It does not grant OD-AF-RC-KR08. It does not merge PR #680.

## Head

| Item | Value |
| --- | --- |
| Worktree | `D:/Projekty/_wt/intergrax/af-rc0` |
| Branch | `af2/rc0-continuation` |
| HEAD at dossier time | `28c27cb896d02e0185b9fa02047411fa152f6469` |
| `origin/arch-freeze/rw-train` | same SHA, re-verified after fetch |
| Base commit subject | Merge pull request #679 from jakbuczarnecki/af2/rw-standing-2 |

The local checkout `D:/Projekty/intergrax` was not used as the work tree.

## Tasks in this package

- Continuation checkpoint and evidence dossier under this directory.
- Secret scan of the two untracked supply-ab probe tests. Verdicts and hashes are in `RC0_EVIDENCE.md`. The probe files and their logs were not copied here.
- Read-only disposition of PT-E03, T-IGX-051-E Part D / EX-D1-07, and T-IGX-043-F.
- Search for published evidence of T-AF2-CLOSED-WORLD, T-AF2-BND-CLOSURE, T-AF2-TYP-MEASURE, PT-V02..PT-V06, and T-AF2-RW-PLAN-AUDIT.
- EDER grouping of the three named packages. No new tickets. Register not edited.
- One KR-18 node replay. Not a PT-Q02 verdict.

## Pull requests touched by the reading

| PR | What the reading found |
| --- | --- |
| #674 | Merged at `e7dd2f0bc`. Part D / EX-D1-07 is not in that diff. |
| #680 | Not merged. Not reviewed. Not modified. |
| #679 | Already the train head. Not modified by this package. |

## Findings

See `RC0_EVIDENCE.md` and `EDER_CLASSIFICATION.json`.

- PT-E03: `NOT_MERGED`.
- T-IGX-051-E Part D / EX-D1-07: `NOT_IN_PR_674_DIFF` (held control leg only).
- T-IGX-043-F: `NOT_IMPLEMENTED_ON_TRAIN`.
- CLOSED-WORLD, BND-CLOSURE, TYP-MEASURE, PT-V02..PT-V06, RW-PLAN-AUDIT: `BLOCKED_EVIDENCE`. Replay commands are in the evidence file. None of those runs was executed here, and none is PASS.
- KR-18 node: one local replay, 1 passed. Not independent acceptance.
- EDER: 1073 rows. CLASS-1 = 61, CLASS-UNKNOWN = 489, OWNER-1 = 45. Product `FIX_PENDING_AFTER_PDC` stays on the existing W3 tickets named in the continuation plan (19 BLOCKS, 6 CONDITIONS).

## Decisions

No operator decision was taken. OD-AF-RC-KR08 remains unanswered. No owner was written onto an EDER row. No CODE_FIX row was moved to a new ticket.

## Next package

RC-1 (prerequisites and admission). Not started. Entry is this RC-0 exit record. RW-PLAN-AUDIT stays an independent session. G01 and G02 stay open. The TYP measurement stays blocked on the command in the evidence file.

## STOP

Product edits, test weakening, new EDER tickets, merge of #680, implementation PRs, edits of the roadmap, the FRZ checklist, or workflows, and any declaration of G2, FCS, M3, or ARCH_FREEZE_CLOSED.

## Recovery

If a later session must rerun a blocked measurement, use the command in `RC0_EVIDENCE.md` and publish the log only after a secret scan. A product change after a future G2 is a new G2 and a new PT-Q02, not a patch on this SHA. This checkpoint lives only under `docs/project/maintainers/qualification/arch_freeze_program/continuation/` and does not change a product path.
