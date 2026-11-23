# PT-FRZ-RR (W0) — plan: write the independent re-read of WP-4 FRZ classifications

## Context
WP-4 (3 shards) classified the 109 OPEN FRZ criteria; nobody re-read them. T-AF2-FRZ-MATRIX and PT-D10 depend on them. This session (independent reviewer, read-only, no implementation context) re-read the 14 LOW-confidence OPEN rows and the 28 `path_b_hides_arch_obligation=true` rows against the checklist on `origin/development` (`7960ae7f6`, checklist last changed `3726c1da3`; 225 rows / 109 OPEN, line numbers identical to WP-4's `checklist_line`), plus DP-34/35 and OD-AF-18. All reading is done; the only remaining step is writing the two output files.

## Steps (after approval)
1. Write `D:/Projekty/intergrax/.tmp/session/arch-freeze/frz-rr/FRZ_RR.md` with the content below (gitignored `.tmp/`; no tracked edit).
2. Write a 2-line hand-over note `FRZ_RR_NOTE.md` in the shared notes folder `C:/Users/XPS/AppData/Roaming/Claude/side-session-notes/local_eaf9ef72-d1c1-44d0-acbc-fac05f02edf7/` (verdict + path).
3. Reply in Polish, at most 10 lines.

No pytest, no uv, no git writes. Verification: re-open FRZ_RR.md and count the table rows (41 primary + 9 supplementary + 2 L66).

---

## FRZ_RR.md content (draft)

# PT-FRZ-RR — independent re-read of WP-4 FRZ classifications

- Role: independent reviewer (read-only, no implementation context). Node PT-FRZ-RR (W0), MEP-AF2 v2.1.
- Input: `.tmp/session/eqo-a-remediation/WP-4/` (`rows.jsonl` 209 rows, `PLAN.md`, `findings_rows.jsonl`, `OPEN_QUESTIONS.md`, shard notes).
- Checklist: `git show origin/development:docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` at `origin/development` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (last checklist change `3726c1da3`). 225 rows, 109 OPEN. All 109 OPEN rows are in `rows.jsonl`; status and `checklist_line` match for every row (0 mismatches).
- Decisions: DP-34/DP-35/DP-36/DP-37/DP-39/DP-40 in `.tmp/session/eqo-a-triage/OPERATOR_DECISIONS_REMEDIATION_WAVE1.md@75`; OD-AF-18 in `.tmp/session/arch-freeze/OPERATOR_PACKET_1.md@18`.
- Notation: `CL@n` = checklist line n at the SHA above; `PDC@n` = `docs/project/maintainers/qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md@origin/development:n`; `RM@n` = `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md@origin/development:n`. Class is written as `class / impact`.
- Nothing was executed; no FRZ status is changed; this is not a G6/IIAC sign-off.

## Verdict: FRZ_RR_DIFFERENCES

- WP-4 itself moves **no** architecture obligation into Path B; its classification is conservative. 33 of the 41 primary rows agree.
- 8 row-level classification differences: REG-04, DOC-06, FRZ-05, FRZ-06, FRZ-07, FRZ-08 (impact raised to BLOCKS), SEC-05 and SEC-06 (WP-4 predates OD-AF-18; live part is a `NOT_CLAIMED` exclusion, not PROD-Q-P1 / Path B).
- Flag-set difference: the brief's "28 would-be-hidden rows" (`path_b_hides_arch_obligation=true`) is **not** the DP-34 set of "28 rows with `path_b=B` and no L66 basis". They overlap in 18 rows only. 10 B-flagged architecture rows carry `hides=false`. See section C. The matrix must protect the union (38 rows).
- 9 decision-consistency flags (section E), of which 4 concern texts outside WP-4 that frame architecture or live obligations as Path B / N/A without a decision.

## A. 14 LOW-confidence OPEN rows

| FRZ_ID | WP-4 class | your class | agree? | reason | evidence |
|---|---|---|---|---|---|
| FRZ-CTR-01 | architecture / BLOCKS | architecture / BLOCKS | YES (confidence → MEDIUM) | The PDC mapping WP-4 could not verify exists: PDC-D3-R006 (Ollama adapters lack `stream_with_tools`) is PARTIAL / IMPLEMENTATION_COMPLETION and is mapped to FRZ-CTR-01. All row evidence is scoped ("not global PASS"). | CL@91; PDC@68 |
| FRZ-CTR-03 | architecture / CONDITIONS | architecture / CONDITIONS | YES (stays LOW) | No defect is cited. The row is not referenced by PDC, so the only vehicle is the re-audit plus a census. It escalates to BLOCKS if the census finds `dict[str, Any]` pseudo-contracts. No Path B exposure. | CL@93; PDC_FRZ_COVERAGE_MATRIX.md@14 |
| FRZ-CTR-05 | architecture / CONDITIONS | architecture / CONDITIONS | YES | The cohesion criterion needs a stated rule (DOC) before audit. PDC-D1-R005 is PRESENT. No Path B exposure. | CL@95; PDC@28 |
| FRZ-TYP-02 | architecture / CONDITIONS | architecture / CONDITIONS | YES (stays LOW) | All evidence is scoped. The row is not referenced by PDC. Neighbouring object-typing debt Q2-D2 (TYP-01/03) shows boundary typing is not clean, so it escalates to BLOCKS if the census finds semantic `Any`. | CL@98, CL@97; PDC_FRZ_COVERAGE_MATRIX.md@34 |
| FRZ-TYP-06 | architecture / CONDITIONS | architecture / CONDITIONS | YES (stays LOW) | Only one AST gate is cited (`resolve_typed.py`), with no platform-wide suppression gate. Closable by a suppression audit on existing code; escalates if masking is found. | CL@102 |
| FRZ-PLG-01 | architecture / BLOCKS | architecture / BLOCKS | YES (confidence → MEDIUM) | PDC-D3-R002 records an EBH-5 replay RED (239p/4f) mapped to PLG-01, and PDC-D5-R005 (evaluators behind one contract) is PARTIAL / INTEGRATION_CORRECTION. | CL@103; PDC@64, PDC@109 |
| FRZ-SEC-08 | architecture / CONDITIONS | architecture / CONDITIONS | YES on class; NO on `hides=false` | The class is right, but the row has ReplayX `path_b=B` with no L66 basis and closing stage "CTRL-X, PROD-Q". A PROD-Q deferral would hide the propagation qualification, so `hides` must be true (see C). | CL@225, CL@66 |
| FRZ-PRD-07 | architecture / UNKNOWN | architecture / UNKNOWN | YES | Offline-provable strict-vs-lab separation; the replay decides the impact. Under DP-34 the row needs an architecture vehicle; "STAGE:PROD-Q" (CLOSED SCOPED) must not be its closer. | CL@257, CL@66 |
| FRZ-REG-04 | architecture / UNKNOWN | architecture / BLOCKS | NO | WP-4's own rule decides it: ENT-A-REG-01 ("Stale allowlists = finding") is implemented by `test_replay_x_frz_row_count`, which counts FRZ rows and does not inspect allowlists. No allowlist inventory exists, and the row has no evidence, so a new inventory or gate is needed. | CL@266; ent_audit_x/ANNEX_C_CONTROL_MATRIX.md@102; ent_audit_x/ANNEX_G2_CONTROL_GATE_MAPPING.md@60 |
| FRZ-DOC-06 | architecture / CONDITIONS | architecture / BLOCKS | NO | No composition-manifest artefact exists on `origin/development`; it is only named in the roadmap and checklist. A manifest that does not exist cannot be shown "current". This depends on the FRZ-08 artefact (DP-40). | CL@296; RM@499, RM@1784 |
| FRZ-FRZ-05 | architecture / UNKNOWN | architecture / BLOCKS | NO | The artefact and its generator are absent (only named in RM@499). DP-40 orders "define artefact, generator and owner for FRZ-FRZ-03/05..09", so it is new work, not a snapshot. | CL@307; RM@499; DP-40 |
| FRZ-FRZ-06 | architecture / UNKNOWN | architecture / BLOCKS | NO | Same as FRZ-05. The layer manifest also cannot be frozen cleanly while the Tier-2 bypass gate fails (EQOA-A4-004). | CL@308; RM@499; DP-40 |
| FRZ-FRZ-07 | architecture / UNKNOWN | architecture / BLOCKS | NO | Same as FRZ-05. The only related gate, ENT-A-DOC-02 (`check_manifest_annex_c_alignment.py`), checks Annex C manifest alignment; nothing defines it as the FRZ ownership manifest. | CL@309; ent_audit_x/ANNEX_C_CONTROL_MATRIX.md@106; DP-40 |
| FRZ-FRZ-08 | architecture / UNKNOWN | architecture / BLOCKS | NO | Same as FRZ-05, plus unresolved composition findings A4-015 and A4-037. | CL@310; DP-40 |

## B. 28 would-be-hidden-obligation rows (`path_b_hides_arch_obligation=true`; FRZ-PRD-07 is in A)

| FRZ_ID | WP-4 class | your class | agree? | reason | evidence |
|---|---|---|---|---|---|
| FRZ-TYP-01 | architecture / BLOCKS | architecture / BLOCKS | YES (track stale) | Q2-D2 debt; "Path B not triggered" is correct. DP-36 has decided remediation (product path, after the PDC window), so the track becomes CODE_FIX rather than DECISION. | CL@97; DP-36 |
| FRZ-TYP-03 | architecture / BLOCKS | architecture / BLOCKS | YES (track stale) | Same as TYP-01. | CL@99; DP-36 |
| FRZ-PLG-08 | architecture / BLOCKS | architecture / BLOCKS | YES | No evidence in the row. UPGRADE is ABSENT, RETIRE is PARTIAL, and skill lifecycle is PARTIAL (OD-PDC-SCOPE-01 A: implement). The dossier label "N/A-with-evidence (Path B)" is invalid. DP-37 disposes of it. | CL@110; PDC@34, PDC@37, PDC@41; DP-37 |
| FRZ-SEC-01 | mixed / BLOCKS | mixed / BLOCKS | YES | A7-019 chain, so a DP-34 A/B split applies. The architecture sub-row holds F02/R2/GCA SQLite gates and the TENANT-X mapping. The live sub-row holds only the multi-instance trust anchor (Decision-03; DP-39 splits D03-A). | CL@218; DP-34, DP-39 |
| FRZ-SEC-04 | mixed / BLOCKS | mixed / BLOCKS | YES | L66 covers the live part only (DP-35). The split is not materialised. PDC-D6-R026 "Production auth-bypass live proof" = N/A_WITH_EVIDENCE must not be read as SEC-04 live part N/A (see E4). | CL@221, CL@66; PDC@124, PDC@148; DP-35 |
| FRZ-SEC-05 | mixed / BLOCKS (DECISION; live part → PROD-Q-P1; vehicle "PROD-Q-PATH-B only if L66 amended") | architecture / CONDITIONS, plus a `NOT_CLAIMED` exclusion of E2B/Modal/Daytona as production sandbox providers | NO | OD-AF-18 settled the pending scope ruling. The live part is a frozen-scope exclusion, not PROD-Q-P1 or Path B. The architecture part (AW-7C PHYSQ reference-substrate replay at the freeze SHA; PDC-D6-R010 PRESENT / requalification) is a re-run on existing implementation. The replay needs a qualified Linux environment, so an explicit authorisation is needed under C10/C15. The row text "PROD-Q still required" needs a DOC_PATCH. | CL@222; PDC@132; OPERATOR_PACKET_1.md@18 |
| FRZ-SEC-06 | mixed / BLOCKS (DECISION; live part → PROD-Q-P1) | architecture / BLOCKS, plus a `NOT_CLAIMED` exclusion of the provider part | PARTIAL (impact yes, class and track no) | OD-AF-18 covers only provider production qualification. The plugin trust model stays mandatory: PDC-D6-R016 (plugin admission is structural only, no signature or provenance) is PARTIAL / IMPLEMENTATION_COMPLETION (OD-PDC-SCOPE-07 A: implement), and attestation D1-R003 and revocation D1-R013 need adversarial coverage. The exclusion must not be worded to cover plugin trust. | CL@223; PDC@138, PDC@26, PDC@36; OPERATOR_PACKET_1.md@18 |
| FRZ-SEC-09 | architecture / BLOCKS | architecture / BLOCKS | YES | Ownerless (DP-48). Under the DP-34 split, a live sub-row may hold only the operator/infra retention configuration; platform-owned retention enforcement stays architecture. | CL@226; DP-34 |
| FRZ-REL-03 | mixed / BLOCKS | mixed / BLOCKS | YES | IN-SCOPE BLOCKER and HTTP re-entry NOT PROVEN in the row. The live sub-row holds only the multi-instance first-writer PIS. | CL@230; DP-34, DP-39 |
| FRZ-REL-05 | mixed / BLOCKS | mixed / BLOCKS | YES | The live sub-row holds only provider-native termination; W4-C/IG-F01 semantics stay architecture. | CL@232; DP-34 |
| FRZ-REL-08 | architecture / CONDITIONS | architecture / CONDITIONS | YES | In the DP-34 14-row chain. Any live sub-row may hold only load capacity (PRD-11), never admission semantics. | CL@235; DP-34 |
| FRZ-REL-09 | architecture / BLOCKS | architecture / BLOCKS | YES | A4-036 (in-process queue downgrade) contradicts explicit queue semantics. | CL@236 |
| FRZ-REL-10 | architecture / BLOCKS | architecture / BLOCKS | YES | A4-035 (in-memory scaler substitution) contradicts fail-closed behaviour. | CL@237 |
| FRZ-REL-11 | architecture / CONDITIONS | architecture / CONDITIONS | YES | Bounded retry is offline-provable with fakes. | CL@238 |
| FRZ-PRD-03 | mixed / BLOCKS | mixed / BLOCKS | YES | DP-35 live-part-only; the A/B split must be materialised. | CL@253, CL@66; DP-35 |
| FRZ-PRD-04 | mixed / BLOCKS | mixed / BLOCKS | YES | DP-34 chain. The live sub-row holds only the live HTTP/GCA reproduction; lifecycle ordering and A4-040 stay architecture. | CL@254; DP-34 |
| FRZ-PRD-05 | architecture / BLOCKS | architecture / BLOCKS | YES | DIM16 S2b debt. `rows.jsonl` tags it EQOA-A7-019, but it is not in the 14-row chain. It must not get a DP-34 live sub-row. | CL@255 |
| FRZ-PRD-06 | architecture / CONDITIONS | architecture / CONDITIONS | YES | Under DP-34 every PRD row needs a PDC-X or ENT-AUDIT-X Population E vehicle. The WP-4 vehicle is ENT-A-PRD-01..03; that it belongs to Population E must be confirmed. | CL@256; DP-34 |
| FRZ-PRD-08 | mixed / BLOCKS | mixed / BLOCKS | YES | DP-35 live-part-only; IN-SCOPE BLOCKER; D2 strict graph required. | CL@258, CL@66; DP-35 |
| FRZ-PRD-09 | architecture / BLOCKS | architecture / BLOCKS | YES | The entrypoint census was explicitly excluded in P3 (A4-040). | CL@259 |
| FRZ-PRD-10 | architecture / CONDITIONS | architecture / CONDITIONS | YES | D4 evidence exists; closable by audit. | CL@260 |
| FRZ-PRD-11 | mixed / BLOCKS | mixed / BLOCKS | YES (vehicle wording no) | The WP-4 vehicle "PROD-Q-PATH-B (only if L66 is amended)" conflicts with DP-34, under which L66 is the only authority. The DP-34 vehicle is the A/B split, with a live sub-row for load-level capacity only. | CL@261, CL@66; DP-34 |
| FRZ-PRD-12 | mixed / BLOCKS | mixed / BLOCKS | YES (vehicle wording no) | Same as PRD-11. The live sub-row holds only the live-environment runbook/DR; recovery-responsibility documentation stays architecture. | CL@262; DP-34 |
| FRZ-REG-08 | architecture / CONDITIONS | architecture / CONDITIONS | YES | In the DP-34 chain. Any live sub-row holds only the live HTTP/GCA `build_r1_sqlite_disposition`; the classification proof stays architecture. | CL@270; DP-34 |
| FRZ-DEBT-05 | architecture / BLOCKS | architecture / BLOCKS | YES | PDC-D1-R019 is DEFERRED_OUTSIDE_FREEZE_SCOPE under an authorised exclusion (OD-PDC-SCOPE-03 A). That is a decided exclusion, not Path B; the other DEFERRED/ABSENT items need DP-37 disposition. | CL@301; PDC@42; DP-37 |
| FRZ-FRZ-02 | architecture / BLOCKS | architecture / BLOCKS | YES (scope wider) | The row note "live-scoped `FRZ-PRD-*`" is wider than L66. The same wide wording is also in roadmap L499, which WP-4 did not flag. The DOC_PATCH must cover both. | CL@304, CL@66; RM@499 |
| FRZ-TEN-06 | architecture / CONDITIONS (PASS row, HIST-REQUAL) | architecture / CONDITIONS | YES | The row text has no live carve-out; PROD-Q must not defer the revalidation of a PASS row. | CL@331 |

## C. B-flagged rows with no L66 basis that WP-4 marks `hides=false` (supplementary; FRZ-SEC-08 is in A)

All of them name "CTRL-X, PROD-Q" as closing stage in the checklist and carry ReplayX `path_b=B`, which DP-34 rules non-authoritative. A whole-row move per that flag would hide the architecture obligation, so `path_b_hides_arch_obligation` must be **true**. The classes themselves agree.

| FRZ_ID | WP-4 class | your class | agree? | reason | evidence |
|---|---|---|---|---|---|
| FRZ-SEC-02 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Late secret resolution is a contract property; the row has no evidence. | CL@219 |
| FRZ-SEC-03 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Static secret gate; becomes BLOCKS if the d6 gate is not platform-wide. | CL@220 |
| FRZ-SEC-07 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO; decision needed | WP-4 proposes closing the "Not established" bounds (SaaS provider correctness, IAM, hostile kernel escape, DNS rebinding) as limitations or N/A. That is a scope exclusion without a decision; OD-AF-18 covers only E2B/Modal/Daytona (see E6). | CL@224 |
| FRZ-SEC-10 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | N/A is allowed only with evidence that the platform is not the owner. | CL@227 |
| FRZ-REL-01 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Timeout ownership; TOOL-LIFE-X evidence is unmapped. | CL@228 |
| FRZ-REL-02 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Retry ownership. | CL@229 |
| FRZ-REL-04 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Cancellation (HARNESS-W4). | CL@231 |
| FRZ-REL-06 | architecture / CONDITIONS | architecture / CONDITIONS | class YES; hides NO | Typed provider failure; PDC-D3-R006 is PARTIAL. | CL@233; PDC@68 |
| FRZ-REL-07 | architecture / BLOCKS | architecture / BLOCKS | class YES; hides NO | Silent backend substitution A4-035/A4-036. | CL@234 |

Set arithmetic (from `rows.jsonl`): B-flag rows = 33. B without L66 = 28 (REL 11, SEC 9, PRD 8). `hides=true` = 28. Overlap = 18. B-only = the 10 rows above incl. SEC-08. hides-only = TYP-01, TYP-03, PLG-08, SEC-04, PRD-03, PRD-08, REG-08, DEBT-05, FRZ-02, TEN-06. **Path-B-protected set for T-AF2-FRZ-MATRIX = union = 38 rows.**

## D. L66 rows vs DP-35

| FRZ_ID | WP-4 class | your class | agree? | reason | evidence |
|---|---|---|---|---|---|
| FRZ-PRD-01 | live / NONE (FULL; track ACCEPTED_LIMITATION) | live / NONE | YES (track wording no) | DP-35 keeps the row OPEN under PROD-Q with a no-production-authorisation note. `next_track=ACCEPTED_LIMITATION` must not become "PASS + linked limitation" (OD-AF-08 path). PDC-D3-R021 = N/A_WITH_EVIDENCE mapped to PRD-01 must not promote the row to N/A (see E4). | CL@251, CL@66; PDC@83; DP-35 |
| FRZ-PRD-02 | mixed / CONDITIONS (LIVE_PART_ONLY) | mixed / CONDITIONS | YES | DP-35 says live-part-only. L66 lists the row whole, so the A/B split must be materialised. EBH-5/6 replaceability stays architecture (RPL rows). | CL@252, CL@66; DP-35 |

## E. Decision-consistency flags (architecture obligation → Path B / NOT_CLAIMED / N/A without decision)

1. **SEC-05: PDC text vs OD-AF-18.** PDC-D6-R010 says "provider live = Path B". OD-AF-18 makes it a `NOT_CLAIMED` frozen-scope exclusion, and SEC-05 is not on L66. PDC@132; OPERATOR_PACKET_1.md@18.
2. **SEC-05/06 in DP-34 vs OD-AF-18.** Both rows are in the A7-019 chain, where DP-34 lets the live sub-row "stay OPEN under PROD-Q". For these two rows, OD-AF-18 (the specific and later decision) makes the provider part `NOT_CLAIMED`. The WP-4 `owner` (PROD-Q-P1) and `closure_vehicle` ("PROD-Q-PATH-B only if L66 amended") are stale. The checklist has no `NOT_CLAIMED` text yet (grep: 0 hits).
3. **RPL-02 (outside the sample).** PDC-D3-R017 says "live parity = Path B" for FRZ-RPL-02, which is not on L66. PDC@79.
4. **N/A_WITH_EVIDENCE on live parts.** PDC-D3-R021 (→ PRD-01) and PDC-D6-R026 (→ SEC-04 live auth-bypass proof) are PDC-scope N/A. They must not propagate as FRZ N/A: DP-35 keeps PRD-01 and the live part of SEC-04 OPEN under PROD-Q. PDC@83, PDC@148.
5. **Wide Path B wording.** "live-scoped `FRZ-PRD-*` may remain OPEN" appears in FRZ-FRZ-02 and in roadmap L499. Read literally, it parks PRD-04..07 and PRD-09..12 without a decision. CL@304; RM@499.
6. **SEC-07 "Not established" bounds.** Closing them as limitation or N/A is an exclusion without a decision. CL@224.
7. **Vehicle "PROD-Q-PATH-B (only if L66 is amended)"** on SEC-05, SEC-06, PRD-11 and PRD-12 contradicts DP-34 ("L66 is the only Path B authority"). Replace it with the DP-34 A/B split vehicle for PRD-11/12, and with OD-AF-18 for SEC-05/06.
8. **A7-019 chain count.** `rows.jsonl` tags 15 rows with EQOA-A7-019 (PRD-05 extra); WP-4 and DP-34 count 14. PRD-05 is architecture and must not get a live sub-row.
9. **Stale tracks after decisions.** TYP-01/03 (DP-36: remediate), PLG-08 and DEBT-05 (DP-37), and SEC-05/06 (OD-AF-18) still show `next_track=DECISION`.

## Limits
- Static review only; no gate, replay or pytest was run. "Artefact absent" for the manifests rests on `git grep` of `docs/`, `scripts/` and `governance/` at `origin/development`.
- PDC matrix lines are from the generated document at PDC audit head `1ebd2ab4c2e6`, as published on `origin/development`; later PDC repair state was not checked.
- DP-34's 14-row chain was reconstructed from `findings_rows.jsonl` (EQOA-A7-019) and `OPEN_QUESTIONS.md` OQ-02.
