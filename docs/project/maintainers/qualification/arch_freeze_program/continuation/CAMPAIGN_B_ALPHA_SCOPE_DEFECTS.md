# Campaign B alpha — SCOPE defect index (Package B)

> **Historical (pre–AF-W2-CLOSURE-04):** the defect table below recorded alpha HEAD diagnostics from Package B (`51edb2a7c3399cc03f82173b255602efcb208ecf`). It is **not** the current Campaign B accounting source.

## Current accounting (AF-W2-CLOSURE-04)

Canonical cross-links:

- [`CAMPAIGN_B_MISSING_CASE_MAP.md`](CAMPAIGN_B_MISSING_CASE_MAP.md) — **28/28** defined cases (B-1 **19/19**, B-2 **9/9**).
- [`CAMPAIGN_B_V5_DEFECT_MAP.md`](CAMPAIGN_B_V5_DEFECT_MAP.md) — historical vs recovery vs HEAD diagnostic disposition.
- [`CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md) — R04/R24/S03 targeted measurements; HEAD diagnostic supplements.

| Scope | Count | Note |
| --- | ---: | --- |
| B-1 cases | 19/19 | Matrix complete @ baseline **B** `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| B-2 cases | 9/9 | Matrix complete @ baseline **B** |
| **Total** | **28/28** | Diagnostic completeness ≠ PT-B01/PT-B02 formal CLOSED |
| Historical defect rows | 9 | Remain open product/qualification tracks @ **B** (not re-measured here) |
| R06 / R10 | Recovered | Extra-install reruns documented (not HEAD diagnostic) |
| S03 nodes | 127 PASS, exit 0 | [`CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md) |
| NOT_EXECUTABLE @ **B** | 4 | R01a, R02, R05, R20 — **HEAD_DIAGNOSTIC** supplementary only |

**HEAD diagnostic execution SHA:** `30d05b0cf9972b02de3f14d741bd3fbdc9d3631c` (`origin/development` @ AF-W2-CLOSURE-04 preflight).

| Case | HEAD outcome | Execution / log metadata |
| --- | --- | --- |
| R01a / R20 | PASS — `verify_rw_plan_phase_check3.py` exit 0 | Prior session log cited in missing-case map: `.tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R01a_R20_stdout.txt` (not in git; log content not re-verified in CLOSURE-04) |
| R02 | PASS — traceability pytest exit 0 | `.tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R02_stdout.txt` (not in git; unavailable for digest replay) |
| R05 | PASS — IIAC lint exit 0 | `.tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R05_stdout.txt` (not in git; unavailable for digest replay) |

Historical baseline **B** rows are unchanged. HEAD results do not back-fill **B**.

Does not set SCOPE, omega, G2, or FCS.

---

## Historical defect IDs (alpha FAIL @ legacy HEAD diagnostic)

**Pin:** `51edb2a7c3399cc03f82173b255602efcb208ecf` (execution @ `deea67621120` after REV-MASTER commit in the same worktree)
**Logs:** `.tmp/session/arch-freeze/campaign-b-alpha/pkg-b-51edb2a/` (not tracked)
**Nodes:** PT-B01 (B-1), PT-B02 (B-2 alpha)
**Nature:** ID-only SCOPE input. No reproduction steps. Non-certifying diagnostic.

| ID | Campaign case | Owner node |
| --- | --- | --- |
| DEF-CB-B2-S03-01 | PT-B02 / S03 | PT-WF |
| DEF-CB-B2-S03-02 | PT-B02 / S03 | PT-WF |
| DEF-CB-B2-S03-03 | PT-B02 / S03 | PT-WF |

B-1 defined cases: **0** defects (`incomplete=0/9`).

## Package C consumer index (historical)

Handoff defect IDs (ID-only; no reproduction steps): `DEF-CB-B2-S03-01`, `DEF-CB-B2-S03-02`, `DEF-CB-B2-S03-03`. Owner node PT-WF. Logs under `.tmp/session/arch-freeze/campaign-b-alpha/pkg-b-51edb2a/`.

## Historical disposition

Does not set SCOPE, omega, or FCS. Fed SCOPE predecessor PT-B02 alpha evidence only.

**Package C (train @ `e9bebe0ac`):** [`DEF_CB_B2_S03_PT_WF_DISPOSITION.md`](DEF_CB_B2_S03_PT_WF_DISPOSITION.md) — DEF-CB-B2-S03-01..03 do **not** activate PT-WF E2–E4 product paths.
