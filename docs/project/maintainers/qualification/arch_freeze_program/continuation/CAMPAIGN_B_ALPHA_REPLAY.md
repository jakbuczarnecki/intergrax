# Campaign B alpha — HEAD diagnostic replay (P1)

**Execution SHA:** `3991186b1e388efef28e6a1be09aabc32e7fc04d`  
**Branch / worktree:** `af-fc/b-alpha` @ `D:/Projekty/_wt/intergrax/af-fc-b1`  
**Recorded:** 2026-11-25  
**Context:** `HEAD_DIAGNOSTIC` (read-only manifest cases; non-certifying)  
**Runner:** `scripts/qualification/run_campaign_b_w2_alpha.py` (`--instance-id af-fc-b1`)  
**Scratch logs:** `D:/Projekty/_wt/intergrax/_scratch/af-fc-b1/` (runner + S03 pytest stdout)

## Prior record check

No in-repo `CAMPAIGN_B_ALPHA_REPLAY.md` or equivalent verdict existed for this SHA family at `3991186b1`. Package B pin [`CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md`](CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md) (`51edb2a7c`) is an ancestor record only; this replay supersedes it for the current head.

## PT-B01 (phase `b1`)

| Metric | Count |
| --- | ---: |
| Manifest defined cases | 9 |
| PASS | 9 |
| FAIL | 0 |
| Incomplete / not executed | 0 |
| Runner exit code | 0 |

Evidence JSON: `.tmp/session/w2-recovery-03/campaign-b/b1/HEAD_DIAGNOSTIC/3991186b1e38/af-fc-b1/RESULTS_B1.json`

## PT-B02 (phase `b2`)

| Metric | Count |
| --- | ---: |
| Manifest defined cases | 7 |
| PASS | 6 |
| FAIL | 1 (`S03`) |
| Incomplete / not executed | 0 |
| Runner exit code | 1 |

Evidence JSON: `.tmp/session/w2-recovery-03/campaign-b/b2/HEAD_DIAGNOSTIC/3991186b1e38/af-fc-b1/RESULTS_B2.json`

### S03 (`tests/qualification/harness_01`) pytest sub-counts

Command stopped at `--maxfail=3` (manifest default).

| Pytest outcome | Count |
| --- | ---: |
| passed | 47 |
| failed | 1 |
| errors | 2 |

Failing / errored nodes at this head:

- `FAILED` `test_harness_01_shared_host_nexus_importers_are_w6_debt_not_final_legal`
- `ERROR` `test_harness_01_public_authoring_surfaces_do_not_reference_nexus_as_import_api` (setup: local plugin install; venv missing `pip`)
- `ERROR` `test_harness_01_public_authoring_negative_synthetic_nexus_import_fails` (same setup)

## Aggregate (manifest cases)

| Owner | PASS | FAIL | Total defined |
| --- | ---: | ---: | ---: |
| PT-B01 | 9 | 0 | 9 |
| PT-B02 | 6 | 1 | 7 |
| **Combined** | **15** | **1** | **16** |

## Scope defect handoff (ID-only)

S03 non-green sub-outcomes at this head remain consistent with three PT-B02 / S03 defect slots (`DEF-CB-B2-S03-01`..`03`) in [`CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md`](CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md); IDs are not remapped here.

## Disposition

Diagnostic only. Does not set SCOPE, omega, G2, or FCS. Feeds SCOPE predecessors PT-B01 and PT-B02 evidence at `3991186b1`.