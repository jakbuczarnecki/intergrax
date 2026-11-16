# Historical independent audit register

> **Update 2026-11-08 (after PR 416)** - audits, reviews and designated-session decisions added after PR 408 (reports under `audits/`, decisions under `decisions/`): `rev-w0` ACCEPT_WITH_CHANGES; `q11` read-only investigation (F-015 code defect confirmed); `rev-w1` ACCEPT_WITH_CHANGES; `rev-plan` and `rev-plan2` ACCEPT_WITH_CHANGES (plan of T-IGX-031-A and Amendment A1); `aud-tests` PASS_SCOPED at `b28398026`; `aud-s12` PASS_SCOPED at `47c29b4c3`. Designated certifying sessions (process-enforced, never the implementer or auditor of their stage): `cert-w0`, `cert-w1`, `cert-031a`, `cert-031a2`. See `HISTORICAL_CERTIFICATION_CAMPAIGN_ADDENDUM_2026-11-08.md`.

Generated 2026-11-08T11:20:52Z. Claude session ids are transcript file names (the host does not expose them); `UNKNOWN` means not recoverable.

## 1. Independent audits, reviews and approvals performed in this campaign
| name | session id | role | scope | verdict | report |
|---|---|---|---|---|---|
| rev-docs | 2fa420b1-3289-4211-bdde-8f7a59068b02 | independent non-author reviewer | PR 405 / branch hist-cert/docs-patches (5 commits) | ACCEPT (after 1 correction commit; found a stage-label flip and an absent quotation) | audits/rev-docs/REVIEW.md |
| rev-rules | 36de2835-33cd-4117-a4a4-0f201f898370 | independent non-author reviewer | PR 406 / branch hist-cert/rules-update (3 commits) | ACCEPT (after nine required edits and two follow-up defects; found a false tooling statement) | audits/rev-rules/REVIEW.md |
| aud-030a | cdb0c03b-866b-433c-b899-a8d0ccbcc05f | independent auditor (not implementer, planner or approver) | T-IGX-030-A implementation SHA 683f955727fb (tests only) | PASS_SCOPED (conditions C-1/C-2 only; does not close TOOL-LIFE-X; OD-TLX-A/C authority not verifiable from the repo) | audits/aud-030a/AUDIT_REPORT.md |
| aud-final | 9b703bc3-6a1e-4ac8-9a52-dc662c0f5a1c | independent integration auditor of this package (mission phase P3) | campaign final package, commits 2a7aec93d -> 6c973c93a -> 63ff33812 (documentation only) | ACCEPT_WITH_CHANGES at 2a7aec93d (13 findings: 1 HIGH, 8 MEDIUM, 4 LOW), all applied; re-read at 6c973c93a: ACCEPT_WITH_CHANGES with one mechanical numeric edit (R-1, dashboard counters), applied in 63ff33812 and checked by git diff; the auditor stated the package is ACCEPT once R-1 is applied. Covers fidelity of the consolidation, NOT the correctness of the 85 worker judgements | audits/aud-final/AUDIT_REPORT.md |
| cert-030a | 929ea8c8-4988-4fc1-bcc6-6e66b265b0ac | designated certifying session (plan approval only, IIAC Amendment 01) | Plan of T-IGX-030-A, digest c6d1dca6... | APPROVE (plan only) | audits/cert-030a/CERT_DECISION.md |
| hist-p8-final-aud | a91e4f28-6c3d-4b8e-a0f5-2e7d9c1b4f63 | independent final campaign auditor (P8-AUD-FINAL) | Dossier at `b5c2481d7` / pin `878e2e384` lineage | PASS (dossier; not campaign CLOSED_VERIFIED) | audits/final-p8/AUDIT_REPORT.md |
| hist-p8-cert | process_enforced | designated certifying session (P8-CERT) | Terminal label after P8-AUD-FINAL | ACCEPT `HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS` | audits/cert-p8/CERT_DECISION.md |
| hist-final-scope | `f7c2a8e1-4b9d-4e6f-9a3c-2d8e5f1a0b7c` | independent final HIST-scope auditor (post batch; not final-closure implementer) | Full HIST end state @ `087bd1769` lineage + final-closure batch docs | PASS (scope closure honest; not `CLOSED_VERIFIED`) | audits/final-hist-scope/AUDIT_REPORT.md |

## 2. Implementer / planner sessions (never auditors of their own work)
| name | session id | role |
|---|---|---|
| impl-tlx | 4e77efc7-e2c5-4334-9974-392ee06497ca | planner and implementer of T-IGX-030-A (and plans of 030-B / 030-C) |
| HIST-01 | 8cca8db3-455a-4d8d-9c63-d356410c6ef8 | campaign worker HIST-01 (inventory and candidate evidence; no PASS issued) |
| HIST-02 | f13fa3c1-138a-4d38-9ff4-19244367cc49 | campaign worker HIST-02 (inventory and candidate evidence; no PASS issued) |
| HIST-03 | 5a9e8ba9-967f-49ab-bccd-cf3651c93abc | campaign worker HIST-03 (inventory and candidate evidence; no PASS issued) |
| HIST-04 | 303eeffe-0910-416b-b572-c2f1f67d1240 | campaign worker HIST-04 (inventory and candidate evidence; no PASS issued) |

## 3. What has NOT been independently audited
- Every one of the 85 per-ID dispositions (they are worker findings; no independent auditor has reviewed them).
- The worker findings registers (62 findings), including the theses about the EBH-4 R1 gate (F-013) and the PROD-Q P2-C defect (F-015).
- The HIST-04 governance and authority analysis (IIAC, Amendment 01, EVOL-X).
- The exact-SHA replays of HIST-02 and HIST-03 (they ran on a pinned checkout but were executed by the workers; see the overlap caveat in HIST03 / registry events).
- The campaign package itself (this document set): audit status is recorded in the final report, section 10.

## 4. Role separation observed
Implementer != auditor != certifier held for T-IGX-030-A: impl-tlx implemented; cert-030a approved the plan; aud-030a audited. No worker approved its own work. The delegation record for cert-030a is a transcription of the Operator's chat instruction (not signed by the Operator; assurance `process_enforced`).
