# HISTORICAL CERTIFICATION CAMPAIGN DASHBOARD

_generated 2026-11-08T19:05:22+00:00 from records; start SHA `ba07ebaa3`_

## Counters (derived)

| # | Counter | Value |
|---|---|---|
| 1 | Total historical requirements (baseline) | 85 |
| 2 | Verified independent closures (CLOSED_VERIFIED) | 0 |
| 3 | Valid scope-limited acceptances (PASS_SCOPED) | 0 |
| 4 | Open architecture blockers (IMPLEMENTATION_REQUIRED+QUALIFICATION_REQUIRED) | 52 |
| 5 | Valid Path B live deferrals | 2 |
| 6 | Rows in state AUDIT_REQUIRED (not all are audit-only; see the readiness handoff, section 2) | 29 |
| 7 | Outstanding operator decisions | 3 |
| 8 | Findings requiring implementation | 4 |
| 9 | SSOT discrepancies | 13 |
| 10 | Unverified legacy evidence (rows in state UNVERIFIED + unmapped rows; UNVERIFIED findings are listed in the registers) | 0 |

FRZ rows with a per-ID record: 85 / 85  (unmapped 0)

## Per-state

| state | rows |
|---|---|
| INVENTORIED | 0 |
| IMPLEMENTATION_REQUIRED | 1 |
| QUALIFICATION_REQUIRED | 51 |
| AUDIT_REQUIRED | 29 |
| AUTHORITY_REQUIRED | 2 |
| READY_FOR_CERTIFIER | 0 |
| PASS_SCOPED | 0 |
| CLOSED_VERIFIED | 0 |
| PATH_B_DEFERRED | 2 |
| BLOCKED | 0 |
| UNVERIFIED | 0 |

## Workers

| id | hc terminal | session | phase | state | blockers | needs_master | updated |
|---|---|---|---|---|---|---|---|
| HIST-01 | t7 | `8cca8db3` | DONE - all six deliverables written; closure precisely blocked (see blockers / handoff) | AUDIT_REQUIRED | 3 | 5 | 2026-11-08T07:27:17Z |
| HIST-02 | t12 | `f13fa3c1` | P11-replay-executed-once-at-pin; deliverables updated | IMPLEMENTATION_REQUIRED | 3 | 4 | 2026-11-08T07:47:19Z |
| HIST-03 | t13 | `5a9e8ba9` | COMPLETE - inventory, per-ID dispositions, one combined run + 4 isolated reruns on shared pin; no product code changed, no commit | BLOCKED | 3 | 5 | 2026-11-08T09:26:15Z |
| HIST-04 | t14 | `303eeffe` | DONE - all six deliverables + early note + candidate patch written; worktree untouched (git status clean) | AUTHORITY_REQUIRED | 3 | 7 | 2026-11-08T10:04:36Z |

## Independent audit sessions

| id | role | verdict |
|---|---|---|
| cert-030a | designated certifying session (plan approval, not an implementation audit) | APPROVE (plan only) |
| rev-docs | independent non-author doc reviewer | ACCEPT (final, at 6d1434796, after the correction commit); report audits/rev-docs/REVIEW.md |
| rev-rules | independent non-author policy reviewer | ACCEPT (final, at 937c39820, after E-1..E-9 and D-1, D-2); report audits/rev-rules/REVIEW.md |
| aud-030a | independent auditor of T-IGX-030-A implementation (not implementer, not plan approver) | PASS_SCOPED (T-IGX-030-A tests-only at 683f955727fb; covers C-1/C-2 stale-test reconciliation + F3 cancel-test stabilisation only; does NOT close TOOL-LIFE-X; promotes no FRZ row); report audits/aud-030a/AUDIT_REPORT.md |
| aud-final | independent integration auditor of the final package (P3) | ACCEPT_WITH_CHANGES at 2a7aec93d (1 HIGH, 8 MEDIUM, 4 LOW); corrections applied in the next commit; text-only re-read requested; report audits/aud-final/AUDIT_REPORT.md |
| rev-w0 | independent non-author reviewer of hist-cert/cert-w0 | ACCEPT_WITH_CHANGES (F-1,F-2,F-3,F-5,F-8 applied/recorded); report audits/rev-w0 |
| q11 | read-only investigator (DR-6 / Q-11) | F-015 CONFIRMED code defect (d1040f485); owner SKILL-X, KV authoritative (recommendation only) |
| rev-w1 | independent non-author reviewer of hist-cert/cert-w1 | ACCEPT_WITH_CHANGES (F-1..F-7 applied in erratum); audits/rev-w1 |
| rev-plan | independent reviewer of PLAN_T-IGX-031-A | ACCEPT_WITH_CHANGES (findings 1-9 applied in 9318bf1bb) |
| aud-tests | independent auditor of T-IGX-031-A | PASS_SCOPED @ b28398026 (27-path: 6 failed = F-015 only, 0 already registered; exit code uncaptured: pytest process hung after summary, PID tree 59260) |
| rev-plan2 | independent reviewer of Amendment A1 (S12) | ACCEPT_WITH_CHANGES (F1-F5 low); 27-path rerun not needed |
| aud-s12 | delta auditor of S12 | PASS_SCOPED @ 47c29b4c3 (S12 AST equivalent; ruff clean; steps 1-5 green; R8/R9 open) |

## Publication

- status: ALL_CAMPAIGN_PRS_MERGED (405, 406, 407, 408, 410, 413, 414, 415, 416); refresh PR pending
- origin/development after merge: 51cb5f336
- post-merge CI: Regression gate success on 51cb5f336 (PR 416 merge); earlier: success on 62d1c6158 incl. ENT-AUDIT-X
- PR 405: merge commit 41908e3fb, head 6d1434796
- PR 406: merge commit feee82b2e, head 937c39820
- PR 407: merge commit 4e48eba82, head 90fae38aa
- PR 408: merge commit 991a3e410, head e799bea60
- PR 410: merge commit c169852bf, head de47bd2fe
- PR 413: merge commit 70cceae21, head 98f49d497
- PR 414: merge commit 6734971cf, head 14ff99875
- PR 415: merge commit af323c005, head b4afbe69a
- PR 416: merge commit 51cb5f336, head 705bedc54
