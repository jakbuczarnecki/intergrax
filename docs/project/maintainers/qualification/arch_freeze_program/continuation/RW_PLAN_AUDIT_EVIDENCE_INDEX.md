# T-AF2-RW-PLAN-AUDIT — evidence index (AR-12-07)

> **Obowiązujący stan (AF-W2-SCOPE-CLOSE-01):** PLAN_STRUCTURE=PASS; **RED_FIRST_EVIDENCE=INCOMPLETE**; inventory **39** — **RED_AT_PREFIX=21**, **EVIDENCE_MISSING=12**, **RED_FIRST_OUT=5**, **RED_FIRST_LATER_WAVE=1**; **PT-B01/PT-B02=PREDECESSOR_CONTRACT_SATISFIED** (diagnostic); **RW_PLAN_AUDIT=PASS** (AF-W2-INDEPENDENT-AUDIT-02 @ `73835a889fcebe4393113abd25d9cf9aeceda7c1`); **SCOPE=CLOSED**; mechanical checker `RW_PLAN_AUDIT=NOT_PERFORMED`.
> **Historyczne (superseded):** AF-W2-FINAL-01 NOT_PERFORMED banner; T-IGX-078-A COMPLETE @ 646e63953; RM-03 structural @ `ebe45c269` — lineage only.

| Artifact | Role |
| --- | --- |
| docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json | T-AF2-RW-PLAN machine output |
| docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md | Human render |
| docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json | Generator bindings (OP-B progressive metadata) |
| docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json | Known-red FIX planning mirror |
| docs/project/maintainers/qualification/arch_freeze_program/REV_MASTER_FORMAL_CLOSEOUT_RECORD.md | REV-MASTER predecessor |
| docs/project/maintainers/qualification/arch_freeze_program/continuation/CAMPAIGN_B_MISSING_CASE_MAP.md | PT-B01/B02 diagnostic matrix (28/28) |
| scripts/governance/verify_rw_plan_phase_check3.py | Mechanical phase check 3 (+ OP-B progressive structure) |

| Independent audit round 2 | [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) AF-W2-INDEPENDENT-AUDIT-02 |
| Audited immutable SHA | `73835a889fcebe4393113abd25d9cf9aeceda7c1` |
| SCOPE formal closeout | [`SCOPE_FORMAL_CLOSEOUT_RECORD.md`](../SCOPE_FORMAL_CLOSEOUT_RECORD.md) |
| SCOPE predecessor index | [`SCOPE_PREDECESSOR_INDEX.md`](SCOPE_PREDECESSOR_INDEX.md) |

**Auditor session SHA (historical RM-03, superseded):** ebe45c26930a9291a3df2d6a75028c35c036cee2 (2026-11-24). Round-2 PASS is independent only; implementer publication does not re-sign the verdict.

## PT-Q02 train CI factual replay (2026-11-24) — historical

| Artifact | Role |
| --- | --- |
| .tmp/session/arch-freeze/pt-q02_gh_pr_checks_811.log | gh pr checks 811 @ merge **876af7e5d** |
| .tmp/session/arch-freeze/pt-q02_gh_pr_checks_812.log | gh pr checks 812 @ train tip **e9bebe0ac** |
| .tmp/session/arch-freeze/pt-q02_pin_verify_2026-11-24.log | git rev-parse origin/development + origin/arch-freeze/rw-train |
| [TRAIN_CI_REPLAY_PLAYBOOK.md](TRAIN_CI_REPLAY_PLAYBOOK.md) Runda 3 | Cited job URLs for ci-smoke / rag-gate / governance-fast |

**Not** RED_FIRST_EVIDENCE and not green governance-fast.

## RED_FIRST_EVIDENCE (INCOMPLETE)

**OD-AF-CLOSE-B6=DECIDED** (variant A). **RED_FIRST_EVIDENCE=INCOMPLETE**; **12** OP-B rows **EVIDENCE_MISSING**; **OP_B_PENDING=12**. **SCOPE=CLOSED** with OP-B obligations preserved ([`SCOPE_FORMAL_CLOSEOUT_RECORD.md`](../SCOPE_FORMAL_CLOSEOUT_RECORD.md)). Validator does not claim independent historical replay.
