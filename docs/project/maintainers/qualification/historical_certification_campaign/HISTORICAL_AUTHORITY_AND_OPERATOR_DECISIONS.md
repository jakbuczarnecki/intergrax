# OPERATOR DECISION REQUEST - HIST-01..04 campaign (consolidated)

> **Status update 2026-11-08 (after PR 416)** - all of DR-1, 2, 3, 4, 6, 8 and 9 now have an answer on record (scope rulings with known limits), certified by designated sessions in the Operator's name (`decisions/OPERATOR_DECISION_RECORDS_W0.md` and `_W1.md`; explicit Operator confirmation in `delegations/DELEG-HIST-W0-CONFIRMATION-2026-11-08.md`). The summary table and sections below show the state at PR 408 and are not rewritten; see `HISTORICAL_CERTIFICATION_CAMPAIGN_ADDENDUM_2026-11-08.md` section 2 for the answers and their known limits.

Findings baseline: `ba07ebaa34d04633bd60510f68c18346b1ef160d`. Current `origin/development`: `41908e3fb` (includes the Operator-merged ENT-AUDIT-X PRs #399-#404 and the campaign PRs #405 documentation patches, #406 Amendment 01 revision, #407 T-IGX-030-A tests only; the FRZ checklist has no status changes: 120 OPEN / 105 PASS, all 85 historical rows OPEN).
State: all four workstreams finished their audit phase; T-IGX-030-A is independently audited (PASS_SCOPED) and merged; the product batch (T-IGX-030-B / T-IGX-030-C) is HELD; the decisions marked "open" below are the remaining items that need the Operator.
Sources: worker dossiers (published under `dossiers/hist01..04/`), the local registry `campaign_registry.json` (identifiers `OD-*`), logs in `hist02/`, `hist03/`.
Language: this document is English-only (project rule: repository artifacts are English; operator chat uses the session language).

## 0. Scope, evidence levels, and the signing rule

- This is the single consolidated request. Twelve decisions (DR-1..DR-12) in five blocks. Each can be closed with one sentence (section 8).
- **Nothing here promotes any FRZ row.** 85 rows have a disposition and all stay OPEN; independent closures: 0.
- **Out of scope:** ENT-AUDIT-X (a separate session), FRZ-AUD, ARCH-FREEZE R1-R6, production activation, live PROD-Q integrations.
- Evidence levels: "MASTER-verified" = checked by MASTER directly in git or in a file; "worker thesis" = an analysis result from a worker, not yet independently audited.

### 0.1 Signing rule (CORRECTED on 2026-11-08 after the Operator's objection)
An earlier version of this document said the Operator "must write personally" the `Approved ...: operator @ ...` records (DR-1, DR-2) and the ticket `Approved:` line. That was wrong as a rule.
- The Operator's own words (IIAC Amendment 01, R-1..R-6) are: the Operator may commission a **new, dedicated, non-implementing session** to sign and certify in the Operator's name; the commission itself transfers the scope (R-6). Conditions: R-2 (new, dedicated, not a continuation, not an implementation session) and R-3 (implementer != auditor != certifier).
- The "personal signature" requirements come from the safeguards **S-1 and S-3, which an earlier agent session added**. The document itself calls them "agent-proposed" and "strikeable by the operator"; they were never confirmed. `governance/tickets/README.md` says `Approved: <operator> @ <ISO8601>` without "personally".
- Therefore wherever this document says a record must be "written personally" (DR-1, DR-2, DR-3, DR-4, DR-9), read: **to be decided by the Operator or by a session the Operator designates**. The rule update was applied by PR 406 (merged by the Operator; reviewed by rev-rules, final verdict ACCEPT). Whether Amendment 01 is in force as policy is still DR-2.
- What remains is record honesty, not a rule: a delegate's line must not read as an Operator-written line, and the signing session must not be the implementer. Assurance of any delegated record is `process_enforced` (no cryptographic identity proof).
- First use: plan approval of T-IGX-030-A by the session `cert-030a` (`delegations/DELEG-HIST-030A-2026-11-08.md`, transcribed by MASTER from the Operator's chat instruction; not signed by the Operator).

## 1. Summary table

| DR | Topic | Concerns | Recommendation | Consequence of deferral | Status |
|---|---|---|---|---|---|
| **A. Authority** | | | | | |
| DR-1 | Ratification of delegated EVOL-X signatures | EVOL-X, FRZ-EVOL-01..10 (10 x PASS) | ratify per candidate C1 | EVOL-X closure stays `AUTHORITY_GAP` | open |
| DR-2 | Status of IIAC Amendment 01 | all delegated signatures | strike S-1/S-3 + make text consistent (now) + T-IGX-021-B/C as a separate stage | delegated decisions keep an ambiguous basis | open (revision merged in PR 406; status decision pending) |
| **B. Labels and promotion scope** | | | | | |
| DR-3 | TOOL-LIFE-X roadmap label (L482) | TOOL-LIFE-X, DIST-SCALE-X, ARCH-FREEZE intake | option A now, B after re-audit | roadmap says "CLOSED - INDEPENDENTLY ACCEPTED" against the signed DS-3 | open |
| DR-4 | Promotion scope for FRZ-REG | REG-01..10 | HIST-01 split: (b) for REG-07/10, (a) for REG-01/04/05 | no promotion rule | open |
| **C. Code and test changes (tickets)** | | | | | |
| DR-5 | Red EBH-4 R1 gate (F-013) | BND-01..04 | option 1 | architecture gate red on the base | **ANSWERED: 1** (product part HELD, see DR-12; ticket T-IGX-030-B plan_pending) |
| DR-6 | Red PROD-Q P2-C tests (F-015) | PRD/SEC/REL rows citing P2-C | root-cause first, then ticket | P2-C evidence stays invalid | open |
| DR-7 | Integrate 3 unpublished TLX correctives | TOOL-LIFE-X, REG-02/03, SEC/REL | option 1 under new IDs with audit | TOOL-LIFE-X stays CONDITIONAL | **ANSWERED: 1** (test-only part merged: PR 407; product part HELD as T-IGX-030-C) |
| **D. Scope rulings, exceptions, budget** | | | | | |
| DR-8 | Tier-0 -> Tier-3 exception (F-001) | BND-03 | move the demo composition to Tier-3 | unrecorded exception to a hard boundary | open |
| DR-9 | PROD-Q/SEC scope rulings (P-03, P-04, SEC-05, D03-A, waivers) | SEC-05/06, PRD-04, SEC-01, REL-03/05, PRD-08 | waive nothing; stricter reading | rows stay `AUTHORITY_REQUIRED` | open |
| DR-10 | Budget for QUAL-X Dimension A replay (C10/C15) | REG-07, REG-10 + replay rows | approve one bounded run | REG-07/10 cannot reach `CLOSED_VERIFIED` | **ANSWERED: yes** (run on the final integrated SHA) |
| **E. Publication** | | | | | |
| DR-11 | Publication of doc patches and unpublished signed records | roadmap, checklist, SIGN-6..10 | patches after review; records after DR-1/DR-2 | 13 SSOT discrepancies remain | **ANSWERED: A** (documentation patches merged: PR 405; signed-records part held) |
| **F. Sequencing** | | | | | |
| DR-12 | Product-path edits vs the IIAC ACT-N11 gate | F-013 fix, commit 29c387879 | B: tests now, product edits batched with the next IIAC re-activation | - | **ANSWERED: B** (C4 included with C3; tests-only part merged: PR 407) |

## 2. Block A - Authority

### DR-1 - Ratification of delegated EVOL-X signatures (`OD-HIST04-A`)
**Evidence (HIST-04 thesis, to be verified by an auditor):** five lines in the plain `operator @` form (P3, PRE-P3, B6-ACCEPT, OD-EVOL-12, OD-EVOLX-08) were written by operator-designated signing sessions. The campaign delegations (`DELEG-CERTCAMPAIGN-SIGN-2026-11-07`, `DELEG-2026-11-08`) **exclude** EVOL-X; the per-decision records carry no operator signature line and no session id. The B6 gate accepts only the plain operator form and rejects the delegate form, so delegates had to use the plain form for it to pass. HIST-04 conclusion: `AUTHORITY_GAP`. MASTER-verified: `FRZ-EVOL` has 10 rows at `PASS` on the base, and recent commits describe the W7-05 closure as formal.
**After the merges of PRs #399-#403:** unchanged. The new code does not touch EVOL-X records or its preflight. According to HIST-04, the EVOL-X delegations (no operator signature line, no designated session id) would not meet S-1 as it is now enforced for OD-ENT-01 - a comparison of standards, not a finding that the EVOL-X lines are invalid.
**Options:** (1) ratification per candidate C1 (`HIST04_CANONICAL_PATCH_PROPOSALS.md`), effective from the original timestamps; risk: retroactive effect, needs your assessment of the evidence (HIST04 section 3). (2) Re-signing of those lines (by you or by a designated session); cleanest provenance, may require re-binding the closure SHAs (not assessed - to be assessed by HIST-04 or an auditor). (3) Decline ratification - EVOL-X formal closure and FRZ-EVOL promotion must then be re-assessed (HIST-04 does not recommend declining).
**Recommendation:** (1). **Deferral:** EVOL-X closure keeps no valid authority; 10 FRZ-EVOL rows rest on the same chain. **Audit after the decision:** independent document review (a session that is not the author); no IIAC change needed. Note: ENT-AUDIT-X is a separate session; its handoff is not altered by MASTER - coordinate timing.

### DR-2 - IIAC Amendment 01 status (`OD-HIST04-B`)
**Evidence (HIST-04):** the header says "IN FORCE AS OPERATOR-ADOPTED POLICY TEXT"; sections 5.3 and 6 say "PROPOSED - NOT IN FORCE"; commit `d82f8b722` changed only the Status and Applies lines. Audit T-IGX-021-C was waived on your instruction, implementation T-IGX-021-B was not done. Safeguards S-1..S-3 were proposed by an agent and were never confirmed or struck (OQ-1..4 open). HIST-04 conclusion: the mechanism **can** permit false acceptance or authority expansion (the identity of "the operator" in a chat cannot be proven; every commit in the chain has one git identity).
**Update after PRs #399-#403 (HIST-04 delta addendum, worker thesis, code not executed):** PR #401 added an `operator-delegate` path for **OD-ENT-01 only** (since moved to `scripts/enterprise_audit/iiac_baseline_promotion.py` by #403). It mechanically enforces: IIAC policy IN_FORCE, a delegation published in an ancestor commit, an operator signature line on the delegation, matching decision id / policy / record path / session id, exclusion of listed implementer sessions, and a time window. It still cannot prove authorship or session identity, and it is not T-IGX-021-B as specified (EVOL-X preflight, B6 format and the `verify_iiac_*` role checks are untouched). Conclusions on Amendment 01 are unchanged; the ENT track itself records the same inconsistency. New risk: a decision-specific grammar (`OD-ENT-\d{2}`) next to the general one - a future policy should define one.
**Options:** (a) an Operator note: Amendment 01 is in force as a process-enforced policy for certifications started after `d82f8b722`, with S-1/S-3 struck (S-2 kept or struck); (b) schedule T-IGX-021-B/C and a new audited policy version (touches material paths; voids the activation; needs an independent audit); (c) (a) now + (b) as a separate stage.
**Recommendation:** (c). The revision text for (a) was merged in PR 406 (it touches none of the six `MATERIAL_GOVERNANCE_PATHS`). **Deferral:** delegated signatures stay `process_enforced` on an ambiguous basis.

## 3. Block B - Labels and promotion scope

### DR-3 - TOOL-LIFE-X roadmap label, L482 (`OD-HIST04-C`)
**Evidence:** the current label "CLOSED - INDEPENDENTLY ACCEPTED" conflicts with the signed DS-3 (AUD-TOOLLIFE: CONDITIONAL / STOP @ `1548640708...`; C-1: four red tests, C-2: the F3 cancel test). Additionally (HIST-02, MASTER-verified replay): the EBH-4 R1 gate is red because of TOOL-LIFE-X code (DR-5). DIST-SCALE-X and the ARCH-FREEZE intake depend on this label (DS-5 text).
**Options:** A - an annotation without a status change ("... NOT supported by an independent PASS and must not be cited as current-HEAD proof"); B - change the status to `[ ] CONDITIONAL - AUDIT OPEN`; C - leave unchanged (inconsistent with the signed DS-3).
**Recommendation:** A now; after the re-audit decide on keeping or B. **Deferral:** a false "closed" stays in the roadmap. **Audit:** review of the patch by a non-author session.

### DR-4 - Promotion scope for FRZ-REG (`H01-F013`, `OD-HIST01-B`)
**Evidence:** QUAL-X Population A is closed only for its own scope (SHA `d9ea0d6e8`); 7 REG rows need a replay on current HEAD.
**Options:** (a) REG rows promote only after the ENT-AUDIT-X Population E rollup (strictest); (b) after Population A-D evidence at a pinned current SHA, with a stated Population E residual; (c) all REG rows stay OPEN until ARCH-FREEZE. HIST-01 recommends (b) for REG-07/10, (a) for REG-01/04/05, and keeps REG-08 deferred to PROD-Q.
**Recommendation:** the HIST-01 split, with (b) run only after DR-10. **Deferral:** no promotion rule. **Audit:** independent and mandatory for any promotion.

## 4. Block C - Code and test changes (tickets)

### DR-5 - Red EBH-4 R1 gate (`OD-HIST02-B`, F-013, MAJOR) - ANSWERED: option 1
**Evidence (MASTER verified the run result and the history):** `test_ebh_4_r1_production_nexus_imports_outside_ee_are_zero` fails on `ba07ebaa3`: `intergrax/runtime/tools/tool_operation_recovery.py:22` imports `runtime.nexus`; the file comes from TOOL-LIFE-X commits (`ce57be26c`, `beb46a447`, `a5f813a2c`). Probable cause (worker thesis): the `RuntimeErrorCode` enum lives in Nexus instead of the contract layer. The accepted EBH-6/EBH-7 runs did not execute this gate.
**Answer:** option 1 - ticket: move/alias the enum into the contract layer, gate unweakened, independent audit. **Execution:** ticket `T-IGX-030-B`, HELD by DR-12 until the IIAC re-activation batch.

### DR-6 - Red PROD-Q P2-C tests (`OD-HIST03-B`, F-015, HIGH)
**Evidence:** `test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` - 6 of 7 tests fail **even in isolation** (`ValueError: wire_skill_execution_pinning_store accepts kv_store or document_store, not both`). MASTER-verified: that message comes from `intergrax/skills/execution_pinning_persistence.py:492`, added by `d1040f485` (SKILL-X Batch C, 23 Oct, an ancestor of the base; the same commit added 16 lines in `applications/_shared/harness_host_runtime.py`). **Worker analysis (HIST-03 F-015, thesis, not independently audited, not verified by MASTER):** a code defect, not environment leakage - the `ValueError` itself shows that both stores are non-None at the call; the strict production composition supplies both, the host runtime forwards both, and the SKILL-X Batch C wiring rejects both. The first combined 27-path run had masked it with registry-pollution errors (MASTER's earlier attribution of an "active execution identity required" message to these tests was wrong: that message came from `test_sec_ent.py`). **Not verified:** whether the tests were green at the accepted PROD-Q P2-C SHA, and the worker's root-cause thesis.
**Options:** (1) approve an independent read-only verification of the worker's root-cause thesis first (`Q-11`), then a ticket with an assigned owner (SKILL-X/HIST-01 vs PROD-Q) and a design choice of which store is authoritative; (2) a ticket right away; (3) defer. **Recommendation:** (1) (cheap, no code change; confirms or refutes the thesis before a product change is designed). **Deferral:** P2-C evidence stays invalid for rows that cite it; the SKILL-X acceptance did not catch this change.

### DR-7 - Integration of the unpublished TLX correctives (`OD-INT-01`, `Q-06`) - ANSWERED: option 1
**Evidence:** three commits exist on no `origin/*` branch: `0adb73380` (tests; "reconciles four stale side-effect-safety tests with ABANDONED semantics" - risk of weakening a gate), `10089121e` (event-driven parent-cancel test), `29c387879` (**product code** `intergrax/runtime/nexus/tools/invoker.py`, +5/-1). ID collisions: `T-IGX-023-A` exists on the base as a different ticket; `T-IGX-024-A` is used by the ENT-AUDIT-X stream.
**Answer:** option 1 - mechanical integration under new unique IDs, then a fresh independent auditor with a gate-weakening review. The separate test-only batch (F-006, F-016) is NOT included. Also answered: the red-first test `98b4bf514` joins as C4 together with C3.
**Execution (DR-12 = B):** `T-IGX-030-A` = C1 + C2 (tests only) now; `T-IGX-030-C` = C3 + C4 HELD.

## 5. Block D - Scope rulings, exceptions, budget

### DR-8 - Tier-0 -> Tier-3 exception in BND-03 (`OD-HIST02-A`, F-001, MAJOR)
**Evidence:** `intergrax/cli/external_work.py:338` imports `applications.governed_contractor_application.host.collaborative_work_local_fixture` - this breaks the hard boundary in `AGENTS.md` ("intergrax/ MUST NOT import from ... applications/"). The repository treats it as a "ratified tier exception" but the only authority cited is an audit document; decision id: UNKNOWN.
**Options:** (1) record a named `ACCEPTED_LIMITATION` with a decision id; (2) a ticket to move the CLI demo composition to Tier-3 (a product change); (3) defer. **Recommendation:** (2) (the boundary is marked "never violate"); an exception only if moving is infeasible. The cost of moving has not been estimated. Note: any `intergrax/` edit is subject to DR-12. **Deferral:** an unrecorded exception to a hard rule.

### DR-9 - PROD-Q/SEC scope rulings (`OD-HIST03-A`)
- **P-03 (SEC-06):** split into (a) plugin-trust architecture and (b) per-provider live qualification (PROD-Q-P1, Dimension B), or add SEC-06 to the Path B list (L66). The row stays OPEN. Rec.: split; do not widen L66.
- **P-04 (PRD-04):** architecture- or live-scoped. Rec.: architecture-scoped (the stricter reading) until the startup evidence is executed; the worker gave no substantive recommendation.
- **SEC-05:** the evidence is the Linux reference substrate only; the meaning of the scope needs your ruling.
- **Decision-03 / D03-A** (Kubernetes, trust anchor / global PIS, Spanner + Azure witness): which parts are Dimension A and which are infrastructure/live - concerns SEC-01, REL-03/05, PRD-08.
- **Waivers** AUTH-08 HTTP / G-08 / IG-F01 / DIM16 S2b: mandatory or waived. Rec.: no waivers now; keep PARTIAL/CONDITIONAL records unrewritten.
**Deferral:** the listed rows stay `AUTHORITY_REQUIRED`. **Audit:** independent, after any checklist change.

### DR-10 - Budget for the QUAL-X Dimension A replay (`OD-HIST01-A`) - ANSWERED: yes
**Evidence:** HIST-01 (N4) needs a clean run on a pinned current SHA of the manifest `qual_x_p5_dimension_a_pytest_module_targets()` (35 modules + BND-03 + W2-C, ~18 min). The C10/C15 freeze requires your separate authorization for broad pytest.
**Answer:** approve one bounded run in a fresh independent auditor session (heavy lock, serial, `--extra dev-ci`, no `--basetemp` outside the repo; known: W5 registry pollution - run those files in separate processes). **Sequencing (MASTER):** run on the final integrated SHA after the product batch, so that the evidence concerns the corrected code; with DR-12 = B this is after the IIAC re-activation batch.

## 6. Block E - Publication

### DR-11 - Publication of patches and unpublished records (`OD-PUB-01`, `Q-01/Q-02/Q-08`) - ANSWERED: option A
**Scope:** (i) candidate documentation patches (none changes an FRZ status); (ii) already-signed unpublished records SIGN-6..10, the TRACE-X L395/L404 closures and copies of `DELEGATION_2026-11-07/08_MASTER.md` (patch 10 stays unapplied).
**Answer:** A - publish (i) after an independent non-author review; (ii) only after DR-1/DR-2 (unanswered, so held).
**Patch set prepared (all verified by MASTER with `git apply --check` on `492e361c0`; no FRZ status change):**

| Source | Files | Lines | Notes |
|---|---|---|---|
| hist04 `PATCH_H4_01_REBASED_64ce093.diff` | 2 | +2/-1 | the old diff stopped applying after PR #399; the L2065 change became redundant |
| hist02 BND-06 / PLG-06 | 1 | +2/-2 | checklist evidence cells; both rows stay OPEN |
| hist03 P-01, P-02, P-07 | 4 | +7/-5 | four OBS rows stay OPEN |
| hist01 P-H01-1 and part of P-H01-3 | 2 | +2/-1 | P-H01-1 SHA correction verified: the P0 record was added by `f5c9330973f7`; `76a98cb3f` is the SKILL-X publication commit and does not touch that file |

**Excluded on purpose:** anything that changes a stage label (DR-3), the REG scope (DR-4), needs DR-9, or edits signed delegated records.
**Done:** PR 405 (merged).

## 7. Block F - Sequencing

### DR-12 - Product-path edits vs the IIAC ACT-N11 gate (`OD-ACT-N11`) - ANSWERED: option B
**Evidence (MASTER verified in code):** `scripts/governance/iiac_activation_gate.py:571-576` reports "ACT-N11 product path changed after the audited SHA" when any `intergrax/`, `agents/` or `applications/` path differs from the audited SHA (`ae467c8c4`). ENT-AUDIT-X hit the same in `e02dc2b19` (18 governance-fast tests turned red) and moved its code out of `intergrax/`. It affects the F-013 fix and commit `29c387879`; it does not affect the test-only commits `0adb73380` and `10089121e`. The red-first test `98b4bf514` depends on `29c387879`.
**Options:** A - do everything now and accept an IIAC re-audit and re-activation; B - tests-only C1/C2 now (ticket `T-IGX-030-A`), and batch C3, C4 and the F-013 fix with the next IIAC re-activation that ENT-AUDIT-X closure needs anyway; C - leave the product alone.
**Answer:** B. **Consequence:** C-1/C-2 and the EBH-4 R1 gate stay red until the product batch; CI is not broken. Whether one re-activation covers both streams must be confirmed by the ENT-AUDIT-X stream - it is not assumed.

## 8. Status of answers and how to answer

The answers below are relayed by the MASTER orchestrator from the Operator's chat (English rendering); no Operator-signed record exists in the repository and the chat transcript is the primary record.

Received from the Operator (chat, 2026-11-08): `DR-5: 1 - DR-7: 1 - DR-10: yes - DR-11: A - DR-12: B - C4: yes (with C3)`. All other decisions are deferred and their rows stay OPEN.
To answer the rest, one line per decision is enough, for example `DR-1: 1 - DR-2: c - DR-3: A - DR-4: split - DR-6: 1 - DR-8: 2 - DR-9: rec.`; "rec." accepts the recommendation in this document.

| After the answer | Next MASTER step |
|---|---|
| DR-1, DR-2 | a designated session (or the Operator) writes the records; MASTER orders an independent review and binds the SHAs |
| DR-3, DR-11 | integrate the patch set after fetch/rebase, independent integration audit (P3), PR |
| DR-6, DR-8 | tickets for plan approval (by the Operator or a designated session); only then an implementer, then a fresh auditor |
| DR-9 | candidate checklist changes, then review |
| DR-10 | fresh auditor session after the product batch; the result goes back to HIST-01 / DR-4 |

## 9. Residual risks independent of the answers
- Worker results are not yet independently audited; the P2-C regression and the F-013 cause are theses.
- Per the checklist, a promotion needs an exact SHA, code, tests and an independent audit - nothing in this document replaces that.
- The heavy-lock incident (hist02 removed hist03's live lock) is recorded in the registry; hist03 results carry an overlap caveat.
- Delegated records are `process_enforced`: the delegate's identity cannot be proven (single git identity).
