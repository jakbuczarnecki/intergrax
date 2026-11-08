# HIST-01 acceptance handoff to MASTER

State: HIST-01 deliverables complete; closure is **precisely blocked** on items below. This is NOT an audit handoff for an implemented batch (HIST-01 implemented nothing, so no `handoffs/HIST01_*_AUDIT_HANDOFF.md` was created). Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`.

## A. Session requests (mission item 15)
| # | Request | Why | Constraint |
|---|---|---|---|
| N5 | **Implementer lane**: integrate the unpublished test-only correctives `0adb73380` (C-1) and `10089121e` (C-2) and the product fix `29c387879` with its red-first test `98b4bf514` (T-IGX-024-A) into a publishable branch | OD-ROUTE-DS3 re-audit precondition; H01-F002, F003, F014, F015 | Rename the TLX C-1 ticket: `T-IGX-023-A` collides with the published EVOL-X W7-06 ticket (PR #398). Commit trailer `Ref: governance/tickets/<id>.md` for the product-path commit |
| N6 | **Fresh independent auditor** (not HIST-01, not the cert-wave implementers) on the pinned post-correction SHA | DS-3 re-audit | Replay in this order, one `uv run` each, heavy lock: Batch B L2 incl. `test_tools_side_effect_safety.py`; F8 12-module inventory (expect 67); L3/TL11 17 modules with `--extra mcp`; F3 module as a whole; TL14 gates; plus a review of the 5df074fcb invoker delta against TL1-TL14 |
| N4 | **Exact-SHA clean-checkout run** of the QUAL-X Dimension A manifest (`qual_x_p5_dimension_a_pytest_module_targets()`, 35 modules + BND-03 + W2-C, about 18 min) on a pinned current SHA | FRZ-REG-07/10 and the replay rows; H01-F009 | Needs Operator budget (C10/C15 freeze: broad pytest). Not HIST-01's mutable worktree |
| N7 | Optional: a QUAL-X corrective for `scripts/maintenance/check_architecture_boundary_chaos.py` (F-02) by a separate implementer, audited by a different session | H01-F007 | Either fix or remove from the Q01 inventory with a reason |

## B. Authority blockers - executable next-action dossier
1. **TOOL-LIFE-X roadmap label (H01-F001).** Decision owner: operator/certifier. After N6 returns:
   - if the re-audit passes C-1/C-2/F8/TL14 and the OD-TLX-B narrowing is ratified, a certifier decision may keep `CLOSED - INDEPENDENTLY ACCEPTED` and cite the re-audit (not the unevidenced external review);
   - otherwise relabel as `CONDITIONAL`. Until then do not cite TOOL-LIFE-X as closed in any downstream claim (DIST-SCALE-X entry and ARCH-FREEZE intake depend on it by DS-5 text).
   Candidate wording, proposal only, unsigned: "TOOL-LIFE-X: independent audit AUD-TOOLLIFE @ 1548640708c850b3fbfca3f2db0e5ecfee4274dd CONDITIONAL (C-1, C-2); re-audit authorised by OD-ROUTE-DS3 after correctives; no external review artefact exists."
2. **Scope of promotion for FRZ-REG rows (H01-F013).** Options for the operator:
   - (a) REG rows promote only on the ENT-AUDIT-X Population E rollup (slowest, strictest);
   - (b) REG rows promote on Populations A-D evidence at a current-HEAD pinned SHA with a stated residual for Population E (HIST-01 recommends this for REG-07 and REG-10 only, after N4);
   - (c) leave all REG OPEN until ARCH-FREEZE.
   HIST-01 recommends (b) for REG-07/10, (a) for REG-01/04/05, and keeps REG-08 deferred (PROD-Q). No agent may decide this.
3. **Amendment 01 contradiction** (header IN FORCE vs body "Not in force"): HIST-04/operator. HIST-01 relies on HIST-04's "effective as signed" position and carries the residual.

## C. Candidate documentation patches (proposals; HIST-01 edited no published file)
| Patch | Target | Change |
|---|---|---|
| P-H01-1 | `TOOL_LIFE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md` s1, row "P0 closed-world" | replace `76a98cb3f3e4576ab694f03a5b3e270eb7694f9e` by the full SHA of `f5c933097` (resolve with `git rev-parse f5c933097`) |
| P-H01-2 | roadmap L482 and L2059 | append the AUD-TOOLLIFE CONDITIONAL verdict and the DS-3 re-audit status; label per decision B1 |
| P-H01-3 | `QUAL_X_..._INVENTORY.md` L1048 and body | "32 modules" -> 35 (P5 manifest at d9ea0d6e8); mark stale `READY FOR AUDIT` statuses as historical (F-04, F-06) |
| P-H01-4 | `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` s2 pointer table | correct the T-IGX-023-A/B references after the rename in N5 |

## D. What stays OPEN and why
FRZ-REG-01..10 (all). SKILL-X unconditional status (depends on DS-5 staying effective). TOOL-LIFE-X closure. Unchecked at HEAD: F8 inventory, Batch A-E L2, L3/TL11, Docker PostgreSQL L3, QUAL-X Dimension A manifest, SKILL-X EC4 determinism tests outside the inventory.

## E. Hygiene
Locks released (`locks/` empty after run2). `.venv` was created inside the worktree by `uv` (ignored by git; verify with `git status`). No commits on the branch. No secrets in logs.
