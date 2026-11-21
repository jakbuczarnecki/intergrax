# AUDIT REPORT — AUD-QUALXP0 (QUAL-X-P0 standalone verification) — AUDITOR RECORD

Auditor `aud_qualxp0` (fresh session; not aud_qualx, not STALE-ROWS/REVIEW, not a QUAL-X implementer). Date 2026-11-07.
Authority: Operator's independent-audit instruction only. NOT a closure signature, acceptance decision, OD record or FRZ promotion.

## Verdict: PASS_SCOPED (chain_gate: PROCEED)

Scope audited = the P0 commit's content: (a) the Q01–Q12 closed-world inventory and its bounded P1–P5 packages, (b) the bundled Path B predecessor reconciliation of the SKILL-X / TOOL-LIFE-X / DIST-SCALE-X rows, (c) its non-closure / no-FRZ-promotion claims.
Bounded exclusions: P0's evidence-strength labels for Q01/Q03 are NOT accepted as true at the P0 tree (3 cited gates are red there — F-00); Q05 `test_npsc*`, AW-7C PHYSQ, Q09/Q10 historical batches and the full `tests/qualification/governance/**` / `tenant_x/**` / `state_x/**` / `trace_x/**` trees were not replayed by me.

## 1. What QUAL-X-P0 is
- Roadmap row `QUAL-X-P0` (L479, `[ ] READY FOR AUDIT`) + ledger row (L2055, "Independent audit: pending on P0 commit SHA"); inventory doc `docs/project/maintainers/qualification/QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md` (179 lines at P0).
- One commit: `a70b68cf947b77de58bce8df0863d3114991cbd9` (2026-10-22 15:30:23; parent = START_HEAD `497af3cdbc9d8102b95975fda362a91e25ff2b29`, 15:14:15). Touches 2 docs only (roadmap +14/-6, inventory +179); production delta 0; "pytest @ P0 = 0 (static inventory)". It therefore has NO mandatory tests of its own.
- Claims: inventory Q01–Q12 with evidence-strength classes; packages QUAL-X-P1..P5; Path B reconciliation of 3 stage rows; QUAL-X parent stays OPEN (at the time); FRZ-REG-01..10 stay OPEN; P0 itself = READY FOR AUDIT only.
- SHAs: a70b68cf9, 497af3cdb and the 9 historical SHAs the matrix cites (bd2f7d78, 33490b6a, 765fabe, 8b3db676, 5ccb72fc, f38e455, bd54941d, 3e1c82f2, 0155580a) resolve fully and are ancestors of base 2f6664d53. Between P0 and base there are 14 QUAL-X commits, all by one git identity.

## 2. Independent verification
- **Static (scratch detached worktree @ a70b68cf9):** all cited gate/test/script/dir paths resolve except one wrong path (F-03). `tests/qualification` = 552 files (matches "~552"); 366 `tests/**/*gate*.py` basenames (matches "~367"). `test_prod_q_p0_*` = 8 files; `gov_x2/test_gov_x2_qualification_batch.py` exists.
- **Path B reconciliation (diff of the commit):** SKILL-X / TOOL-LIFE-X / DIST-SCALE-X "Depends on" changed from "all prior mandatory stages incl. PROD-Q" to "predecessors CLOSED per §2.0.3; PROD-Q parent CLOSED not required for Dimension A entry; live slice mandatory for production use only". Compared with roadmap §2.0.3 (L240+): consistent. QUAL-X row PLANNED/MANDATORY → OPEN/IN PROGRESS. No stage marked CLOSED; no FRZ status changed (FRZ-REG-06/07/10 OPEN at P0 and at base). **Verified.**
- **Replay** (ONE pytest, `-p no:xdist`, heavy-test lock held 15:36–15:41, scratch venv via `uv sync --frozen`) at exact a70b68cf9, 15 modules = Q01 (ebh_3, ebh_2a, ebh_2i, ebh_4_r1, hardening_3), Q03 (ebh_2f_r2), Q04 (gov_x2 batch), Q02 (8 prod_q_p0 modules): **207 collected → 202 passed, 3 FAILED, 2 skipped (317.7 s).** Log `logs/pytest-p0.log`.
  - Skips: 2 platform-conditional tests in prod_q_p0_integrity (POSIX-only / Windows without job-object) — environment, not defects.
  - **FAILED (genuine):** `test_ebh_2a_public_contract_boundary_gate::test_gate_passes_on_repository_with_registered_debt`, `test_ebh_2i_final_rescan_gate::test_ebh_2i_public_contract_boundary_gate_passes`, `test_hardening_3_layer_boundary_gate::test_hardening_3_contracts_do_not_import_runtime_except_allowlist`. Cause: `intergrax/contracts/governed_external_work_create_follow_on.py:29` and `…_provider_invocation_identity.py:7` import `intergrax.runtime.attestation.canonical_json`; `intergrax/contracts/tools/qualified_tool_invocation.py:17` imports `intergrax.tools.invocation_wiring`. The tree at P0 has real contracts→runtime / foreign-domain coupling.
  - Remediated afterwards: `9b9a85a98` (15:51, "correct tool invocation public boundary") and `b74d6383b` (HARDENING-3 SSOT correction). The same three modules are in AUD-QUALX's 630/630 green run at d9ea0d6e8.
  - Scripts (plain python, no pytest): `check_context_tier0_import_boundary.py` exit 0; `check_architecture_boundary_chaos.py` **exit 1** (F-02: calls `scripts/<child>.py`; children were moved to `scripts/maintenance/` by e182e59fd; file unchanged through base).

## 3. Overlap with AUD-QUALX (d9ea0d6e8, 630 passed) — precise
- Covered by AUD-QUALX at d9ea (a different SHA, not P0's): the 4 EBH gates + hardening_3 + ebh_2f_r2 (Q01/Q03 — exactly the ones red at P0, green at d9ea), harness_01 gates (Q05/Q08/Q11), 4 tenant_x modules + P2 negative matrix (Q04/Q08), 3 state_x + 2 trace_x modules (Q06), qual_x P3/P4/P5/FINAL (realizing packages P3–P5).
- NOT covered by AUD-QUALX, replayed by me: Q02 prod_q_p0 (8 modules, pass at P0), Q04 gov_x2 batch (pass), tier0 script.
- NOT covered by anyone: chaos script (broken), `test_npsc*` (Q05), AW-7C PHYSQ (Q08/Q10), bulk of governance/tenant_x/state_x/trace_x trees, FRZ-REG-06/07 historical batches.
- AUD-QUALX states it did not audit P0's prose/matrix line by line. P0 is therefore **not** absorbed by it — only P0's downstream outcome (P1–P5 + FINAL on Dimension A) is covered there.

## 4. Is P0 satisfied?
Yes for what it is — a docs-only, non-closing inventory plus normative predecessor reconciliation: scope, SHAs, counts, non-closure claims and Path B text verify, and its identified gaps (no current-HEAD gate rollup, FRZ-REG-10 undefined, env classification) are corroborated by the replay (the red gates live-demonstrate its Q01/Q07 gap) and were addressed by P1–P5.
Not satisfied as an *accurate current-HEAD evidence statement*: the inventory presents Q01/Q03 as DIRECT while 3 cited gates were red on the tree it inventoried (F-00), cites a broken script (F-02) and a wrong path (F-03). Its own process gate (independent audit before P1) was never met (F-01).

## 5. What remains for the roadmap row to be honestly closed
1. Operator decision: record this audit as the P0 independent verification (not absorption), accepting the PASS_SCOPED bounds (signature package provided, UNSIGNED).
2. Authorized publisher (not me): update roadmap L479 + ledger L2055 + inventory header citing the auditor record; add an erratum for F-00 / F-02 / F-03 and acknowledge F-01.
3. Separately (product, outside this audit): fix `check_architecture_boundary_chaos.py` paths or drop it from Q01.
No FRZ-REG status change is implied. AUD-QUALX's Population A acceptance is independent of these edits.

## Findings
See FINDINGS.json: F-00 MEDIUM, F-01 MEDIUM, F-02 MEDIUM, F-03 LOW, F-04 LOW, F-05/F-06 INFO. None blocking.
