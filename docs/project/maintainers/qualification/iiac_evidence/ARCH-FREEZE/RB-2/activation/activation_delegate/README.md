# ARCH-FREEZE RB-2 activation delegate — PREP only

Written by the activation **prep** session (not certifier, not auditor, not implementer). No pytest beyond logged verifier calls below. Run from the worktree root; set `PYTHONPATH` to the worktree when invoking `scripts/governance/*.py` from pytest in a sibling checkout (same pattern as P8-04 `activation_delegate/README.md`).

| Log | What | Expected @ prep pin |
| --- | --- | --- |
| `rb2_activation_prep_01_precheck` | `precheck_rb2_activation_prep.py` | exit 0 when delta and draft manifest pins match |
| `rb2_activation_prep_02_confirm` | `confirm_rb2_activation_prep.py` | exit 0 when prep confirmations hold (prerequisites remain BLOCKED) |
| (future) `act_*` | `verify_iiac_policy_activation`, role separation, record phase-1/2 | **not run** until DP-08 delegate session |

Blockers before a real delegate session: PT-Q02 **PASS**, certifier manifest **PASS** (not DRAFT), `DELEGATION-*` under DP-08, OD-AF-12 activation line at G2, FCS merge commit.