# SIGNED_DECISIONS — SIGN-3_CTENT (session 17d6a6fa-e695-41e5-854c-3bed2299f663)

Delegation: DELEG-CERTCAMPAIGN-SIGN-2026-11-07, sha256 5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab (verified before any signing). Worktree HEAD b288ba756ef736f3d372832d8e684dbb19901ca4 = origin/development (fetched 2026-11-07).

## Evidence verified independently (both decisions)

| Claim | Result | How |
| --- | --- | --- |
| Plan digest = eb8707e1…b5a0 | CONFIRMED | `plan_digest.py` on `CT-ENT-PLAN/CT_ENT_PLAN_AND_RISKS.md` → identical |
| NF-01 validator red | CONFIRMED at HEAD b288ba756 | `validate_post_implementation_master_plan.py` exit 1: `disposition must declare READY, PLAN PARTIAL, or TECHNICAL PARTIAL constant` (log: `logs/validator_head.log`) |
| Cause: plan line 3 vs lib | CONFIRMED | plan line 3 = `INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED`; `master_plan_ssot_lib.py:90,92,94` only READY/PARTIAL/TECHNICAL PARTIAL; check at :433; READY guard at :581; plan changed in `477017f15` (2026-11-05) after the lib's last change |
| Red CI is this defect | CONFIRMED | `gh run view 42352868140 --log-failed` (ent-audit-fast, head dd07d15ff): `test_validator_passes_on_canonical_ssot` fails with exactly that disposition error; check-run `ent-audit-fast` = failure, `ent-audit-pr-gate` = success; the last 5 enterprise-audit runs on PRs all failed |
| GAP-RC-SSOT-DRIFT evidence false at base | CONFIRMED | plan line 163 `CLOSED … exit 0 @ HEAD` vs exit 1 above |
| NF-02 wrong check names | CONFIRMED | verifier lines 15-18 prefixed; real check-run names bare `ent-audit-fast`/`ent-audit-pr-gate` (GitHub API on dd07d15ff); workflow jobs at lines 30/70 have no `name:` |
| NF-03 ruleset-only protection | CONFIRMED live (read-only) | `rules/branches/development` → list of `deletion`, `non_fast_forward` (ruleset 20844442); classic `protection` → 404 "Branch not protected" |
| NF-02c docs repeat wrong strings | CONFIRMED | `git grep`: exactly the 4 docs named in the plan (packet lines 12-13, runbook:17, decision packet:71, master plan:168) + the verifier |
| Scope: no product code, no IIAC material path, no GitHub write | CONFIRMED | planned files are none of the 6 `MATERIAL_GOVERNANCE_PATHS` (`iiac_activation_gate.py:39-46`); no `intergrax/`/`agents/`/`applications/`; verifier stays `gh api` GET; ticket texts prohibit PUT/POST |
| Approval-gate hazard | CONFIRMED | `verify_plan_approval.py:17` regex `^Approved:\s*.+$` is presence-only; drafts contain no `Approved:` line |

### Ticket id fact (resolved)
`T-IGX-021-A` is TAKEN at origin/development (IIAC Amendment 01 specification ticket). `T-IGX-021-B` and `-C` are not in the tree but are RESERVED by that amendment (code ticket and independent audit). `git ls-tree origin/development governance/tickets/` highest = `T-IGX-021-A`; no `T-IGX-022*` in the tree, in any local `cert-wave/*` branch, or in campaign missions/signing dossiers. Assigned: **CT-ENT-01 = T-IGX-022-A**, **CT-ENT-02 = T-IGX-022-B**. Scaffold must re-check collisions.

Not done: no pytest run (not necessary: all claims verified by the validator run, CI log, and git/gh reads); did not exercise the proposed changes.

## Decision OD-CTENT01-PLAN-2026-11-07
- Outcome: **SIGNED_WITH_CONDITIONS**
- Exact text signed: `signed_text_OD-CTENT01-PLAN-2026-11-07.txt`, sha256 `e847ef1fefa17010d7e3bc7fda0bf6f3bfed01c92cfbaafdb2d799610b21585a`
- Option chosen: A with the "INTERNALLY CLOSED incompatible with validation errors" guard (recommended; my verification does not contradict it: the disposition constant is the only validator error, Option B would rewrite a published SSOT).
- Conditions: plan copied byte-unedited (digest binding); ticket-id re-check at scaffold; AUDITOR/CERTIFIER named by MASTER, distinct, before audit.

Signature lines (exact form):
```
Approved OD-CTENT01-PLAN-2026-11-07: operator-delegate @ 2026-11-07T18:20:10+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```
Plan-approval line for the ticket `T-IGX-022-A` (satisfies `verify_plan_approval.py`; bound to the digest below):
```
Approved: operator-delegate @ 2026-11-07T18:20:10+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab; decision OD-CTENT01-PLAN-2026-11-07)
Plan digest: eb8707e113cfdee3ed87cb4b1406da22e77e41c77e9c7ec1aa66162792b1b5a0
```

## Decision OD-CTENT02-PLAN-2026-11-07
- Outcome: **SIGNED_WITH_CONDITIONS**
- Exact text signed: `signed_text_OD-CTENT02-PLAN-2026-11-07.txt`, sha256 `1ac2ecb775057ad6bb975b980f7289e4c965cd7dfc33d792febbf064725600d3`
- Docs string correction and OD-ENT-02 packet re-issue confirmed; OD-ENT-02 live-vs-compensating NOT decided.
- Conditions: as above, plus the plan's binding implementer conditions (live verifier stays exit 1 until an admin acts; STOP on evaluate-mode rules; no GitHub write).

Signature lines (exact form):
```
Approved OD-CTENT02-PLAN-2026-11-07: operator-delegate @ 2026-11-07T18:20:10+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```
Plan-approval line for the ticket `T-IGX-022-B`:
```
Approved: operator-delegate @ 2026-11-07T18:20:10+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab; decision OD-CTENT02-PLAN-2026-11-07)
Plan digest: eb8707e113cfdee3ed87cb4b1406da22e77e41c77e9c7ec1aa66162792b1b5a0
```

## Notes for the publisher
- Copy lines verbatim; the ticket drafts' "## Approved … UNSIGNED" prose must be replaced by the plan-approval line, and `Lifecycle state` moved `plan_pending → plan_approved` by the integration owner.
- The draft tickets name T-IGX-021-A/B in their titles/text and in the plan §6; renumber in the tickets only. Do NOT edit the plan (digest).
- `verify_plan_approval.py` is presence-only; the `operator-delegate` form is process_enforced, not a cryptographic signature, and is not recognised by the EVOL-X closure preflight (Amendment 01 §7) — irrelevant to these plan approvals.
- Residual interpretation risk R1 (plan §7): the new accepted disposition string must not be cited as formal ENT-AUDIT-X closure.
