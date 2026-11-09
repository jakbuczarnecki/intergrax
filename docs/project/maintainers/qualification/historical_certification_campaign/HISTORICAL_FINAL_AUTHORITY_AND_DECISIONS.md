# Historical certification — final authority and decisions (Phase 1 baseline)

| Field | Value |
|---|---|
| START_HEAD | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` (`origin/development`, verified 2026-11-08) |
| Worktree baseline | `D:/Projekty/_wt/intergrax/hist-base` @ above SHA |
| Branch | `hist-cert/plan-reviews` |
| Master plan (repo copy) | [`HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md`](HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md) |
| Cursor approval source | `c:\Users\XPS\.cursor\plans\hist_closure_master_plan_6d5c72f5.plan.md` (Operator-approved plan mode artifact) |

## Baseline reconfirmation (P1-BASE)

| Check | Result |
|---|---|
| Remote `development` | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Local `D:/Projekty/intergrax` | Untouched (stuck at `15347255`, 64 behind) |
| Plan digest 030-B | Match `8c650422…3378` |
| Plan digest 030-C | Match `55d4c9e6…0103` |
| Plan digest 032-A | Match `2628d1a8…7446` |
| Plan digest 033-A | Match `ae167672…3775` |

### Defect sites at START_HEAD (read-only)

| ID | Location | Observation |
|---|---|---|
| HIST02-F-013 | `tool_operation_recovery.py:22` | Imports Nexus `error_codes` |
| H01-F015 | `invoker.py` `dep_boundary is None` | `pool.submit` not guarded with `mark_failed` on failure |
| HIST03-F-015 | `skill_execution_pinning_wiring.py:35-38` | Forwards both stores to strict leaf |
| HIST02-F-001 CLI | `intergrax/cli/external_work.py` | Tier-0 → Tier-3 imports / `importlib` |
| HIST-F-BND-03B (tracked) | `product_manifest_registry.py` | Tier-0 dynamic Tier-3 manifest imports; **not** in 033-A scope |

## Decision matrix (adopted technical choices — plan section A)

| Question | Existing record | Adopted answer | Rejected alternative | Risk | Authority | W1-PLAN covered? | Blocks | Depends on |
|---|---|---|---|---|---|---|---|---|
| 032-A pin precedence | Q-11 / HIST03-F-015 | **A2**: resolver KV-first; doc only if no KV | A1 leaf accepts both; B host-only; C drop doc from composition | Durability residual `HIST-F-DUR-01` | Master plan §A.1 | Step 3 names owner SKILL-X (sequence only) | P3-C3 | P1-SIGN 032-A |
| 032 step-6 blast | Plan text “five” vs paths | **Four named test files** in plan §7 step 6 | Directory-wide pytest | Unmeasured reds reported not fixed | Master plan §A.1 | Partial | P3-C3 evidence | P4 audit |
| 033-A CLI seam | HIST02 F-002 | **(b2)** command-set + Tier-3 launcher | Tier-0 stub commands (a) | BND-03 stays OPEN (`HIST-F-BND-03B`) | Master plan §A.2 | DR paths | P3-C4 | P1-SIGN 033-A |
| 030-B/C scope | DR-5 / DR-7 | Keep written plan scope | Cherry-pick old 024-A ids | ACT-N11 red window | Master plan §A.3 | Yes | P3-C1/C2 | P1-SIGN each |
| 021-B/C | DR-2 | **Separate stage**; not in product batch | Delegate activation grammar | Q-12 / F-H4-03 OPEN | Master plan §A.4 | Yes | P5+ | Operator |
| Merge grant | — | Merge commit on own `hist-cert/*` PRs after audit+CI | Squash/rebase | SHA delegation reachability | Master plan §A.5 | — | P5/P8 | P2-WAIT, audit |
| ENT consent / IIAC sign | IIAC in force | **Not granted** by plan approval | Assume ENT agreement | Global ACT-N11 on merge | Master plan §A.6 | — | P2-WAIT, P5-SIGN | Operator, ENT |

## Explicit non-decisions (still OPEN)

- ENT-AUDIT-X **shared re-activation consent** — still absent (Operator P2 supersession only; see agreement successor section).
- **Operator IIAC activation** line — unsigned; agents must not set `IN_FORCE` on a new record by hand.
- **T-IGX-021-B/C** — not started; no material-path edits authorized.
- Residuals **HIST-F-*** (`BND-03B`, `DUR-01`, `OLLAMA`, `SECSTORE`, `REGISO`, `NOCI`, `HANG`, etc.) — tracked; no implementation authority in Phase 1.
- **T-IGX-030-A / 031-A** lifecycle — remain `ready_for_audit` unless a certifier session advances per policy (optional P1-031-LIFECYCLE not executed here).
- **P1-SIGN** — BLOCKED; see [`audits/P1_SIGN_HANDOFF.md`](audits/P1_SIGN_HANDOFF.md).

## Phase 1 review verdicts

| Ticket | Review path | Verdict |
|---|---|---|
| T-IGX-030-B | `audits/plan-030b/PLAN_REVIEW.md` | PASS |
| T-IGX-030-C | `audits/plan-030c/PLAN_REVIEW.md` | PASS |
| T-IGX-032-A | `audits/plan-032a-r2/PLAN_REVIEW.md` | PASS |
| T-IGX-033-A | `audits/plan-033a-r2/PLAN_REVIEW.md` | PASS |

## Blocking effect summary

| Gate | State |
|---|---|
| P1-SIGN | BLOCKED — no ticket `Approved:` lines |
| P3 product lanes | BLOCKED per ticket until respective P1-SIGN |
| P2-WAIT (development merge) | **Operator supersession 2026-11-09** (`DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md`); ENT consent still absent |
| P5-SIGN-ACT | **BLOCKED** — literal Operator IIAC activation line |
| P5-VERIFY | **BLOCKED** — ACT-N11 red on product merge tree |
| Expected terminal campaign label | `HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS` unless external gates clear |
