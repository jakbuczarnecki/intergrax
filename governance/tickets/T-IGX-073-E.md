# T-IGX-073-E — W2-RECOVERY-01: RW-PLAN, phase-check-3, PT-V04, Campaign B alpha

## Scope

Architecture mandate **W2-RECOVERY-01** on `development` @ `8aaa178a5d45736ed51efe7c03b638ce5e253e37`:

- Canonical **T-AF2-RW-PLAN** (`RW_PLAN.md` + machine `rw_plan.v1.json`).
- Strict **phase-check-3** validator (`verify_rw_plan_phase_check3.py`).
- **PT-V04** S1 baseline runner; repair pytest collection (`pytest-asyncio` vs pytest 9).
- **Campaign B** B-1/B-2 alpha evidence runner + manifest (diagnostic, non-certifying).

Does **not** set `SCOPE`, `G2`, `RW-PLAN-AUDIT` PASS, or `REV-MASTER` formal PASS.

## Approved

Approved: W2-RECOVERY-01 architecture mandate @ 2026-11-24

## Test contract

```text
python scripts/governance/build_rw_plan.py
python scripts/governance/verify_rw_plan_phase_check3.py
python scripts/governance/verify_arch_freeze_plan_traceability.py
python -m pytest tests/unit/governance/test_verify_rw_plan_phase_check3.py --collect-only -q
python scripts/qualification/run_pt_v04_s1_baseline.py
python scripts/qualification/run_campaign_b_w2_alpha.py --phase b1
python scripts/qualification/run_campaign_b_w2_alpha.py --phase b2
```

Evidence logs: `.tmp/session/w2-recovery-01/`

## Lifecycle state

in_progress