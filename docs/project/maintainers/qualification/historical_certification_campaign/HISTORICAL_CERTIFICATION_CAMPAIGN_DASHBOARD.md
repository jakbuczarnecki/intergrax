# HISTORICAL CERTIFICATION CAMPAIGN DASHBOARD

_generated 2026-11-08T10:57:36+00:00 from records; start SHA `ba07ebaa3`_

## Counters (derived)

| # | Counter | Value |
|---|---|---|
| 1 | Total historical requirements (baseline) | 85 |
| 2 | Verified independent closures (CLOSED_VERIFIED) | 0 |
| 3 | Valid scope-limited acceptances (PASS_SCOPED) | 0 |
| 4 | Open architecture blockers (IMPLEMENTATION_REQUIRED+QUALIFICATION_REQUIRED) | 51 |
| 5 | Valid Path B live deferrals | 3 |
| 6 | Outstanding audit requirements | 29 |
| 7 | Outstanding operator decisions | 7 |
| 8 | Findings requiring implementation | 4 |
| 9 | SSOT discrepancies | 13 |
| 10 | Unverified legacy evidence (UNVERIFIED rows + unmapped) | 0 |

FRZ rows with a per-ID record: 85 / 85  (unmapped 0)

## Per-state

| state | rows |
|---|---|
| INVENTORIED | 0 |
| IMPLEMENTATION_REQUIRED | 1 |
| QUALIFICATION_REQUIRED | 50 |
| AUDIT_REQUIRED | 29 |
| AUTHORITY_REQUIRED | 2 |
| READY_FOR_CERTIFIER | 0 |
| PASS_SCOPED | 0 |
| CLOSED_VERIFIED | 0 |
| PATH_B_DEFERRED | 3 |
| BLOCKED | 0 |
| UNVERIFIED | 0 |

## Workers

| id | hc terminal | session | phase | state | blockers | needs_master | updated |
|---|---|---|---|---|---|---|---|
| HIST-01 | t7 | `8cca8db3` | DONE - all six deliverables written; closure precisely blocked (see blockers / handoff) | AUDIT_REQUIRED | 3 | 5 | 2026-11-08T10:05:00Z |
| HIST-02 | t12 | `f13fa3c1` | P11-replay-executed-once-at-pin; deliverables updated | IMPLEMENTATION_REQUIRED | 3 | 4 | 2026-11-08T09:10:00Z |
| HIST-03 | t13 | `5a9e8ba9` | COMPLETE - inventory, per-ID dispositions, one combined run + 4 isolated reruns on shared pin; no product code changed, no commit | BLOCKED | 3 | 5 | 2026-11-08T09:25:52Z |
| HIST-04 | t14 | `303eeffe` | DONE - all six deliverables + early note + candidate patch written; worktree untouched (git status clean) | AUTHORITY_REQUIRED | 3 | 7 | 2026-11-08T13:00:00Z |

## Independent audit sessions

[{'id': 'cert-030a', 'role': 'designated certifying session (plan approval, not an implementation audit)', 'terminal': 't1f', 'verdict': 'APPROVE (plan only)', 'assurance': 'process_enforced'}, {'id': 'rev-docs', 'role': 'independent non-author doc reviewer', 'terminal': 't1h', 'scope': 'hist-cert/docs-patches', 'verdict': 'ACCEPT (final) - all 5 commits at 6d1434796 after the correction; report reviews/rev-docs/REVIEW.md'}, {'id': 'rev-rules', 'role': 'independent non-author policy reviewer', 'terminal': 't1j', 'scope': 'hist-cert/rules-update', 'verdict': 'ACCEPT_WITH_CHANGES (E-1..E-9); report reviews/rev-rules/REVIEW.md'}, {'id': 'aud-030a', 'role': 'independent auditor of T-IGX-030-A implementation (not implementer, not plan approver)', 'terminal': 't1k', 'pinned_checkout': 'D:/Projekty/_wt/intergrax/aud-030a @ 683f95572', 'verdict': 'PASS_SCOPED (T-IGX-030-A tests-only at 683f955727fb; covers C-1/C-2 stale-test reconciliation + F3 cancel-test stabilisation only; does NOT close TOOL-LIFE-X; promotes no FRZ row); report audits/aud-030a/AUDIT_REPORT.md'}]

## Publication
{'status': 'MERGED_BY_OPERATOR', 'merged_at': '2026-11-08T10:43Z (11:43 local)', 'origin_development_after': '41908e3fb', 'pull_requests': {'405': {'merge_commit': '41908e3fb', 'head': '6d1434796', 'ci_on_head': '6 pass, 11 skipping (after re-running jobs that failed on Install uv HTTP 504)'}, '406': {'merge_commit': 'feee82b2e', 'head': '937c39820', 'ci_on_head': '6 pass, 11 skipping'}, '407': {'merge_commit': '4e48eba82', 'head': '90fae38aa', 'ci_on_head': '6 pass, 11 skipping'}}, 'verified': 'each PR head == pushed tip; all tips are ancestors of origin/development; ACT-N11: 0 product-path changes vs ae467c8c4; 0 of the six MATERIAL_GOVERNANCE_PATHS changed; 21 files changed by the three PRs', 'post_merge_ci': 'Regression gate runs on development (4e48eba82, feee82b2e) were pending/in_progress at the last check - NOT yet confirmed green', 'ci_failures_before_rerun': "all caused by 'Unexpected HTTP response: 504' in the uv install step (infrastructure), none by tests or verifications"}
