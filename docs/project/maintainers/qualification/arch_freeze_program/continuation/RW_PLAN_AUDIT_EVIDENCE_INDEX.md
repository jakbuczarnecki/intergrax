# T-AF2-RW-PLAN-AUDIT — evidence index (AR-12-07)

| Artifact | Role |
| --- | --- |
| `docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json` | T-AF2-RW-PLAN machine output |
| `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md` | Human render |
| `docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json` | Known-red FIX planning mirror |
| `docs/project/maintainers/qualification/arch_freeze_program/REV_MASTER_FORMAL_CLOSEOUT_RECORD.md` | REV-MASTER predecessor |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md` | PT-B01/B02 SCOPE input |
| `scripts/governance/verify_rw_plan_phase_check3.py` | Mechanical phase check 3 |

**SHA:** `ebe45c26930a9291a3df2d6a75028c35c036cee2` (auditor session 2026-11-24). **No implementer verdict attached.**

## RED_FIRST_EVIDENCE (still INCOMPLETE)

Mechanical rule (`scripts/governance/verify_rw_plan_phase_check3.py`, `validate_red_first_evidence`): every `code_fix_inventory` row in `rw_plan.v1.json` with `red_first_status: EVIDENCE_MISSING` needs a tracked `RED_AT_PREFIX` replay with a 40-character `red_first_prefix_sha` and a non-empty `red_first_evidence_ref` before `RED_FIRST_EVIDENCE` can complete. All 20 FIX rows remain `EVIDENCE_MISSING` at `c9ce6175f` (no row greened in this pass). Missing input: wave red-first session logs (typically under `.tmp/session/`, not product-path commits). **SCOPE not closed.**

