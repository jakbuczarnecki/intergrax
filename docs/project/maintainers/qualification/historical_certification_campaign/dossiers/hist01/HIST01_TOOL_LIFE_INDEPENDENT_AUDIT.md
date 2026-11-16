# HIST-01 - TOOL-LIFE-X historical closure publication: independent review

## 0. Role statement
HIST-01 is a **reviewer of a historical publication it did not implement** (TOOL-LIFE-X closure, 2026-10-26). HIST-01 authored **no corrective**. This review is a SECOND look after AUD-TOOLLIFE (aud_toollife, verdict CONDITIONAL / chain gate STOP); it does not replace the re-audit that OD-ROUTE-DS3 reserves for a fresh session after the correctives land. It issues **no PASS and no acceptance**. Its runs happened on a mutable worktree (clean at `ba07ebaa3`), so they are not exact-SHA evidence.

AUD-TOOLLIFE files were used as LEADS (local, unpublished); every claim below was re-checked against the repo or a run, unless marked LEAD.

## 1. Object
| Item | Value | Check |
|---|---|---|
| Acceptance baseline = implementation/test SHA | `1548640708c850b3fbfca3f2db0e5ecfee4274dd` | exists; ancestor of HEAD |
| Publication commit | `35344557d3ce67a631f82df96c86b79b92b4db93` | descendant of baseline; ancestor of HEAD |
| Publication delta | 3 files, all `.md` (roadmap, reconciliation, F8 evidence doc) | `git diff --name-only` baseline..publication: no code or test file |
| FRZ checklist delta `76a98cb3f..35344557d` | empty | `git diff --stat` empty: no FRZ promotion |
| P0 SHA cited in the reconciliation | `76a98cb3f` | **WRONG**: that is the SKILL-X publication commit; the P0 record is `f5c933097` (16:56:22, five minutes later). Documentation slip, H01-F006 |
| Batch C commit that introduced ABANDONED | `a5f813a2c` (2026-10-24) | ancestor of baseline |
| Stage production delta P0..baseline | 102 files under `intergrax/` | large for a "certification" stage; wave approvals are text lines (AUD F-06, LEAD) |

## 2. Findings (all in HIST01_FINDINGS_AND_RESIDUALS.json)
1. **Label vs audit (H01-F001).** Roadmap L482/L2059 show `CLOSED - INDEPENDENTLY ACCEPTED`, plus "publication READY FOR INDEPENDENT PUBLICATION AUDIT". An independent audit was done and returned CONDITIONAL / STOP. The roadmap does not mention it. OD-ROUTE-DS3 deliberately left the label undecided. The "external independent review (ChatGPT)" has no artifact (reconciliation s0 and s7; AUD F-03). HIST-01 found no review text either.
2. **C-1 reproduced red at HEAD (H01-F002).** Four tests in `test_tools_side_effect_safety.py` fail: ABANDONED returned where TIMEOUT / VALIDATION_ERROR / the pool-shutdown message is asserted. Log `hist01/run1_tlx_c1_c2_head.log`. Same four as the audit, so the later invoker rewrite did not change them.
3. **C-2 reproduced red at HEAD (H01-F003).** `test_f3_streamable_http_parent_cancel_during_catalog_child` fails in module order (same run).
4. **Invoker rewritten after acceptance (H01-F004).** `5df074fcb` (EBH-7 W1) changed `invoker.py` by +182/-211 after the baseline. Of the accepted TL1-TL14 only parts were replayed at HEAD (below).
5. **Real implementation defect candidate (H01-F015).** In `RuntimeToolInvoker._execute_once`, branch `dep_boundary is None`, `ext_op.mark_running()` is followed by an unguarded `pool.submit(...)` (L1913-1915 at HEAD); the sibling branch is guarded. A submit failure (always after `close()`) would leave the durable record RUNNING. Confirmed by reading the code; the runtime reproduction is the cert-wave implementer's claim (T-IGX-024-A), not HIST-01's.
6. **Correctives exist but are unpublished and one ID collides (H01-F005, F014).** T-IGX-023-A (TLX C-1) and T-IGX-023-B (C-2) are on local `cert-wave/*` branches; T-IGX-023-A collides with the published EVOL-X ticket of the same name.
7. **Exit criteria 7/8 narrowed outside the roadmap.** OD-EC7 appears 0 times in the roadmap and the FRZ checklist (grep). Criterion 7 (stalled detection) was reduced to DEADLINE_ONLY and 8 (worker crash recovery) to the qualification topology with production activation off (`OD-R4-G-A07-PROD-ACTIVATE` OPEN). These are bounded and declared in the closure record, but only there. OD-TLX-B in the delegated chain is about this narrowing; HIST-01 did not read its signed text (it lives in the gitignored SIGN-4 dossier).

## 3. Review of the unpublished C-1 corrective (read-only, `git show 0adb73380`)
Test-only. It changes the two timeout/validation-error tests to assert ABANDONED, and the two post-`close()` tests to assert ABANDONED / `"abandoned"` / `executor.calls == 1` / ledger UNCERTAIN. It keeps the ledger, replay and effect-count assertions, drops only the `ThreadPoolExecutor` message substring, and adds a post-close replay-blocked negative test. Import-order and format hunks are noted as forced by the pre-commit gate. HIST-01's reading: no obvious weakening of the invariant (no blind replay, no repeat effect); the semantic choice (a never-started post-close call is ABANDONED) was taken by OD-TLX-C Option A, which HIST-01 cannot verify from the repo. Verdict on the corrective: **plausible, unaudited**. The C-2 corrective (event-driven cancel instead of a 60 ms timer, plus two named negatives) was identified from its commit message only, not reviewed line by line.

## 4. Coverage of accepted behaviour at HEAD (mission item 12)
| Run | Command scope | Result |
|---|---|---|
| run1 | `test_tools_side_effect_safety.py` + F3 streamable-HTTP module, `--extra mcp` | 5 failed, 34 passed (the 4 C-1 tests + the C-2 test) |
| run2 | 7 SKILL-X modules + 2 TL14 architecture modules (`test_platform_execution_unification_u5_final_zero_bypass.py`, `test_agent_lifecycle_bypass_ast.py`) | 135 passed |
| not run | F8 67-test inventory, Batch A-E L2 union, L3/TL11 17-module set, Docker PostgreSQL L3 (C10/C15 freeze) | UNVERIFIED at HEAD |

So TL14 (no lifecycle bypass) has green gates at HEAD (if those are the intended modules; the mapping is HIST-01's reading). TL1-TL13 are **not** re-proven at HEAD. The historical 67/67 is not a substitute.

## 5. Disposition
- Stage label: **not decided by HIST-01** (AUTHORITY_REQUIRED). Evidence says "CONDITIONAL" is the supportable reading; "CLOSED - INDEPENDENTLY ACCEPTED" is not supported by a published independent review.
- Open implementation defects: H01-F015 (fix exists, unpublished). Open qualification defects: C-1, C-2 (correctives exist, unpublished).
- Required sequence (OD-ROUTE-DS3): publish correctives under unique IDs -> pin an exact post-correction SHA -> fresh independent auditor replays Batch B L2, F8 inventory, L3/TL11 (`--extra mcp`), TL14 gates, and the HEAD invoker delta -> certifier decision on the roadmap label and on OD-EC7 / OD-TLX-B narrowing.
- An independent publication audit of the 2026-10-26 closure as published exists only in the form of AUD-TOOLLIFE. The historical test count remains a historical count.
