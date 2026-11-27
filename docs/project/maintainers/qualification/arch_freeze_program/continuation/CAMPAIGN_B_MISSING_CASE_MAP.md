# Campaign B alpha — diagnostic case matrix (V5 + AF-W2-CLOSURE-02)

**Manifest:** `scripts/qualification/campaign_b_w2_manifest.v1.json` — **28** defined cases (B-1 **19**, B-2 **9**; R01 split → R01a/R01b).
**Historical baseline B:** `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.
**HEAD diagnostic SHA (supplement):** `c94745b11b35a3be55e4e9cb6502befc60262e26` (`development`).
**Session logs:** `.tmp/session/af-w2-closure-02/` (not committed except doc citations).
**Classification:** `PARTIAL_DIAGNOSTIC` — diagnostic completeness ≠ product certification. **PT-B01 / PT-B02 not formally CLOSED.**

Recoveries (R06, R10) and targeted closure (R04, R24, S03): see [`CAMPAIGN_B_V5_R06_R10.md`](CAMPAIGN_B_V5_R06_R10.md), [`CAMPAIGN_B_V5_R10_REMAINDER.md`](CAMPAIGN_B_V5_R10_REMAINDER.md), [`CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md). **R10 is HISTORICAL_ALPHA recovery evidence, not HEAD diagnostic.**

## Full matrix (28 / 28)

| Case ID | AB contract | Historical execution SHA | Historical outcome | Recovery outcome | HEAD_DIAGNOSTIC outcome | Evidence reference | Defect classification | Remediation / qualification owner | Diagnostic completeness | Remaining obligation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R00 | AB-05 | 7960ae7f | PASS (exit 0) | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R01a | AB-05 | 7960ae7f | NOT_EXECUTABLE_AT_BASELINE | — | PASS (verify_rw_plan exit 0 @ c94745b) | HEAD: .tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R01a_R20_stdout.txt | missing_tool@B | T-AF2-RW-PLAN / W3 | COMPLETE | HEAD diagnostic only |
| R01b | AB-05 | 7960ae7f | FAIL (exit 1) | — | — | CAMPAIGN_B_V5_ALPHA.md | registry_bypass | PT-G01 / registry | COMPLETE | RW CODE_FIX |
| R02 | AB-05 | 7960ae7f | NOT_EXECUTABLE_AT_BASELINE | — | PASS (pytest exit 0) | HEAD: .tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R02_stdout.txt | missing_tool@B | T-AF2-RW-PLAN-AUDIT | COMPLETE | HEAD diagnostic only |
| R03 | AB-06 | 7960ae7f | PASS (exit 0) | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R04 | AB-06 | 7960ae7f | FAIL (7 failed, 1 skip) | 8 failed, 5 passed (langchain-core) | — | CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md | tenant_probe_fail | PT-TEN / TENANT-X | COMPLETE | Historical defects open |
| R05 | AB-07 | 7960ae7f | NOT_EXECUTABLE_AT_BASELINE | — | PASS (lint exit 0) | HEAD: .tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R05_stdout.txt | missing_tool@B | IIAC closure | COMPLETE | HEAD diagnostic only |
| R06 | AB-08 | 7960ae7f | FAIL (collection exit 2) | PASS 92/92 exit 0 | — | CAMPAIGN_B_V5_R06_R10.md | missing_extra → recovered | MCP / fastmcp | COMPLETE | Product defects elsewhere |
| R07 | AB-07 | 7960ae7f | FAIL (exit 1) | — | — | CAMPAIGN_B_V5_ALPHA.md | closed_world_parity | TRACE-X / impact-map | COMPLETE | RW remediation |
| R08 | AB-07 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R09 | AB-09 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R10 | AB-08 | 7960ae7f | FAIL (collection exit 2) | PASS 48 total (5+43) | — | CAMPAIGN_B_V5_R06_R10.md; CAMPAIGN_B_V5_R10_REMAINDER.md | missing_extra → recovered | MCP F3 L3 | COMPLETE | Not HEAD evidence |
| R11 | AB-09 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R12 | AB-07 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R13 | AB-07 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R14 | AB-09 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R15 | AB-09 | 7960ae7f | FAIL (exit 1) | — | — | CAMPAIGN_B_V5_ALPHA.md | replay_closure_counters | ROADMAP-REPLAY-X | COMPLETE | RW remediation |
| R16 | AB-09 | 7960ae7f | FAIL (8 nodes) | — | — | CAMPAIGN_B_V5_ALPHA.md | trace_x_qualification | TRACE-X | COMPLETE | RW remediation |
| R17 | AB-11 | 7960ae7f | FAIL (STALE) | — | — | CAMPAIGN_B_V5_ALPHA.md | ebh7_matrix_freshness | EBH-7 | COMPLETE | Matrix refresh track |
| R20 | AB-10 | 7960ae7f | NOT_EXECUTABLE_AT_BASELINE | — | PASS (shared with R01a) | HEAD: .tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R01a_R20_stdout.txt | missing_tool@B | T-AF2-RW-PLAN | COMPLETE | HEAD diagnostic only |
| R21 | AB-10 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R22 | AB-10 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R23 | AB-10 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| R24 | AB-10 | 7960ae7f | PASS runner (6p, 4s) | 9 passed, 1 failed | — | CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md | nt19_diagnostic_fail | NT-19 / adapter budget | COMPLETE | One historical FAIL remains |
| R25 | AB-10 | 7960ae7f | FAIL (9/12 lanes) | — | — | CAMPAIGN_B_V5_ALPHA.md | impact_map_unmapped | EVOL-X / CHANGED-SINCE | COMPLETE | RW remediation |
| S01 | AB-10 | 7960ae7f | FAIL (contract→impl) | — | — | CAMPAIGN_B_V5_ALPHA.md | contract_surface_edge | EBH-2D / tier boundary | COMPLETE | RW remediation |
| S02 | AB-10 | 7960ae7f | PASS | — | — | CAMPAIGN_B_V5_ALPHA.md | — | — | COMPLETE | — |
| S03 | AB-10 | 7960ae7f | Runner kill (node 127 passed) | 127 passed exit 0 | — | CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md | interrupted_runner | PT-B02 / harness_01 | COMPLETE | Node complete; runner kill historical |
## Overlay commands (unchanged @ B)

| case_id | AB | Tracked vendor | B overlay command |
| --- | --- | --- | --- |
| R04 | AB-06 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_ten.py` | `python -m pytest .tmp/campaign-b-overlay/probes/pp_ten.py -q --tb=line` |
| R24 | AB-10 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_nt19.py` | `python -m pytest .tmp/campaign-b-overlay/probes/pp_nt19.py -q --tb=line` |
| R25 | AB-10 | `scripts/qualification/campaign_b_w2_alpha/scripts/r25_impact12.py` | `python -I .tmp/campaign-b-overlay/scripts/r25_impact12.py .tmp/campaign-b-overlay/r25-out` |

**Coverage summary:** B-1 **19/19** accounted; B-2 **9/9** accounted. Historical defects remain mapped per [`CAMPAIGN_B_V5_DEFECT_MAP.md`](CAMPAIGN_B_V5_DEFECT_MAP.md); baseline B tree **unchanged**. **AF-W2-CLOSURE-03:** R04/R24/S03 digests in [`CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md); does not imply green baseline @ B.
