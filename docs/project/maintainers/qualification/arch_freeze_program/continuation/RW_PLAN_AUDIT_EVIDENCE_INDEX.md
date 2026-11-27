# T-AF2-RW-PLAN-AUDIT — evidence index (AR-12-07)

> **Obowiązujący stan (AF-W2-CLOSURE-02, development @ c94745b11b35a3be55e4e9cb6502befc60262e26):** RED_FIRST_EVIDENCE=INCOMPLETE; inventory **39** rows — **RED_AT_PREFIX=21**, **EVIDENCE_MISSING=12** (OP-B / OD-AF-CLOSE-B2), **RED_FIRST_OUT=5**, **RED_FIRST_LATER_WAVE=1**; **RW_PLAN_AUDIT=NOT_PERFORMED**; **SCOPE=OPEN**.
> **Historyczne (superseded):** wpis COMPLETE (T-IGX-078-A @ 646e63953); starszy fragment poniżej o **36×** EVIDENCE_MISSING i uznanych DEBT-02/03 @ RED_AT_PREFIX — **nie obowiązuje** po GA.2R + OP-B.

| Artifact | Role |
| --- | --- |
| docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json | T-AF2-RW-PLAN machine output |
| docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md | Human render |
| docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json | Generator bindings (OP-B progressive metadata) |
| docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json | Known-red FIX planning mirror |
| docs/project/maintainers/qualification/arch_freeze_program/REV_MASTER_FORMAL_CLOSEOUT_RECORD.md | REV-MASTER predecessor |
| docs/project/maintainers/qualification/arch_freeze_program/continuation/CAMPAIGN_B_MISSING_CASE_MAP.md | PT-B01/B02 diagnostic matrix (28/28) |
| scripts/governance/verify_rw_plan_phase_check3.py | Mechanical phase check 3 (+ OP-B progressive structure) |

**Auditor session SHA (historical RM-03):** ebe45c26930a9291a3df2d6a75028c35c036cee2 (2026-11-24). **No implementer audit PASS attached.**

## PT-Q02 train CI factual replay (2026-11-24) — historical

| Artifact | Role |
| --- | --- |
| .tmp/session/arch-freeze/pt-q02_gh_pr_checks_811.log | gh pr checks 811 @ merge **876af7e5d** |
| .tmp/session/arch-freeze/pt-q02_gh_pr_checks_812.log | gh pr checks 812 @ train tip **e9bebe0ac** |
| .tmp/session/arch-freeze/pt-q02_pin_verify_2026-11-24.log | git rev-parse origin/development + origin/arch-freeze/rw-train |
| [TRAIN_CI_REPLAY_PLAYBOOK.md](TRAIN_CI_REPLAY_PLAYBOOK.md) Runda 3 | Cited job URLs for ci-smoke /
ag-gate / governance-fast |

**Not** RED_FIRST_EVIDENCE and not green governance-fast.

## RED_FIRST_EVIDENCE (INCOMPLETE)

Mechanical rule (
alidate_red_first_evidence): **RED_FIRST_EVIDENCE=INCOMPLETE** until every owed row records accepted RED_AT_PREFIX replay. **12** OP-B rows remain **EVIDENCE_MISSING** (progressive delivery allowed; structure retains OD-AF-CLOSE-B2 metadata). **FRZ-DEBT-02** and **FRZ-DEBT-03** stay **EVIDENCE_MISSING** with **seeded-red** obligation (not satisfied by mechanical PASS). **SCOPE not closed.**
