# Historical independent audit register
Generated 2026-11-08T10:56:32Z. Claude session ids are transcript file names (the host does not expose them); `UNKNOWN` means not recoverable.

## 1. Independent audits, reviews and approvals performed in this campaign
| name | session id | role | scope | verdict | report |
|---|---|---|---|---|---|
| rev-docs | 2fa420b1-3289-4211-bdde-8f7a59068b02 | independent non-author reviewer | PR 405 / branch hist-cert/docs-patches (5 commits) | ACCEPT (after 1 correction commit; found a stage-label flip and an absent quotation) | audits/rev-docs/REVIEW.md |
| rev-rules | 36de2835-33cd-4117-a4a4-0f201f898370 | independent non-author reviewer | PR 406 / branch hist-cert/rules-update (3 commits) | ACCEPT (after nine required edits and two follow-up defects; found a false tooling statement) | audits/rev-rules/REVIEW.md |
| aud-030a | cdb0c03b-866b-433c-b899-a8d0ccbcc05f | independent auditor (not implementer, planner or approver) | T-IGX-030-A implementation SHA 683f955727fb (tests only) | PASS_SCOPED (conditions C-1/C-2 only; does not close TOOL-LIFE-X; OD-TLX-A/C authority not verifiable from the repo) | audits/aud-030a/AUDIT_REPORT.md |
| cert-030a | 929ea8c8-4988-4fc1-bcc6-6e66b265b0ac | designated certifying session (plan approval only, IIAC Amendment 01) | Plan of T-IGX-030-A, digest c6d1dca6... | APPROVE (plan only) | audits/cert-030a/CERT_DECISION.md |

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
