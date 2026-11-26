# Campaign B alpha — manifest slots (V5)

Diagnostic mapping for **T-IGX-073-E** (V5). Alpha baseline **B**: `7960ae7f6c82da60ca9acdc0ceebf63810002a42`. Session logs under `.tmp/session/arch-freeze/campaign-b-v5/` (not committed).

## Defined slots (manifest @ candidate branch)

**Coverage:** B-1 **19 / 19** manifest rows defined; alpha **`PARTIAL_DIAGNOSTIC`** (**8 PASS**, **3 NOT_EXECUTABLE**, **8 FAIL**; **11/19** incomplete @ B). B-2 **9 / 9** defined; alpha **`PARTIAL_DIAGNOSTIC`** (**5 PASS**, **1 NOT_EXECUTABLE**, runner **3 FAIL**; **4/9** incomplete @ B; S03 node **127 passed** per stdout — see [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md).

### B-1 recovered / overlay (R04)

| case_id | AB | Tracked vendor | B overlay command |
| --- | --- | --- | --- |
| R04 | AB-06 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_ten.py` | `python -m pytest .tmp/campaign-b-overlay/probes/pp_ten.py -q --tb=line` (copy from tracked tree into B worktree `.tmp/campaign-b-overlay/`) |

### B-2 recovered / overlay (R24, R25)

| case_id | AB | Tracked vendor | B overlay command |
| --- | --- | --- | --- |
| R24 | AB-10 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_nt19.py` | `python -m pytest .tmp/campaign-b-overlay/probes/pp_nt19.py -q --tb=line` |
| R25 | AB-10 | `scripts/qualification/campaign_b_w2_alpha/scripts/r25_impact12.py` | `python -I .tmp/campaign-b-overlay/scripts/r25_impact12.py .tmp/campaign-b-overlay/r25-out` |

Equivalence detail: [`CAMPAIGN_B_V5_EQUIVALENCE.md`](CAMPAIGN_B_V5_EQUIVALENCE.md). Execution summary: [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md).

## Undefined slots

None at V5 close-out (all 27 planned manifest cases have in-manifest commands).

## Not present at baseline B tree (overlay / train-only nodes)

These manifest rows point at paths that do not exist in the product tree at **B**; the runner records `NOT_EXECUTABLE_AT_BASELINE` unless a future overlay copies train-era governance scripts into `.tmp/`:

| case_id | Missing at B | Overlay / successor |
| --- | --- | --- |
| R01a | `scripts/governance/verify_rw_plan_phase_check3.py` | Train manifest case; execute @ RW-PLAN recovery SHA only |
| R02 | `tests/unit/governance/test_arch_freeze_plan_traceability.py` | Same |
| R05 | `scripts/governance/lint_iiac_closure_regex.py` | Same |
| R20 | `scripts/governance/verify_rw_plan_phase_check3.py` | Same |