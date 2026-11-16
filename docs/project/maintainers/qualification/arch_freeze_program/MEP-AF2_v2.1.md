# ARCH-FREEZE — Master Execution Plan (MEP-AF2) — v2.1 (merged with EQO-A phase 2)

**Version:** v2.1, 2026-11-16. v1 and v2 are kept at `.tmp/session/arch-freeze/MEP-AF2_v1.md` and `MEP-AF2_v2.md`.

**v2.1 (+PD3)** applies the PA-0 delta re-audits `PLAN_DELTA_AUDIT_V21.md` (PD3-01..06) and `.tmp/session/arch-freeze/pa0/PLAN_DELTA_AUDIT_V2.md` (PD2-01..PD2-16). Each fix is marked `[PD2-nn]`.

**Why v2:** the PA-0 independent plan audit (`.tmp/session/arch-freeze/pa0/PLAN_AUDIT.md`) returned `PLAN_AUDIT_FINDINGS` (1 BLOCKER, 6 MAJOR, 8 MINOR). The Operator then decided, on 2026-11-16 in this chat, to **merge** MEP-AF2 with the EQO-A remediation phase-2 plan into ONE plan. The decision is recorded as OD-AF-00. English rendering of the Operator's choice: "Merge into one".

**Planning HEAD:** `origin/development` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.

**Sources:** repo facts were read with `git show origin/development:<path>`, `gh api` (GET only) and `gh run`. The EQO-A phase-2 plan and its decisions are session artefacts under `.tmp/session/eqo-a-remediation/` and `.tmp/session/eqo-a-triage/`. They are not yet repo records; T-AF2-OD-RECORD publishes them.

**Planning verdict (section Q):** `PLAN_READY_FOR_OPERATOR_REVIEW`. It is conditional on the PA-0 delta re-audit of v2.

---

## 0. How the two plans merge (OD-AF-00)

| Layer | Source of truth | Content |
|---|---|---|
| Remediation and requalification backbone up to **ENTRY** (the Operator's L499 entry check) | EQO-A phase-2 master plan `WP-12/MASTER_PLAN.md`, `DAG.md`, `TICKETS_DRAFT.md`, `AUTHORIZATION_REQUESTS.md` | 90 nodes (37 tickets, `PT-*`), waves W0–W7, decisions DP-01..DP-68 (binding), authorizations AR-*/AB-* |
| ARCH-FREEZE certification and closure, and the holes WP-12 leaves open | **MEP-AF2 v2 (this file)** | PDC-X residual qualification (input of WP-12 node **DS-EXC**; answers WP12-Q11); plans and negative tests for the residual fixes and for the 35 unplanned FRZ CODE_FIX rows (answers WP12-Q12 and phase check 3); ARCH-FREEZE package reconciliation; freeze manifests; VBN dossier; post-freeze policy; FREEZE SHA lock; ECAP §2.C; VBN acceptance; FRZ promotion; ECAP §15 admission; closure; ECAP §16 handoff |
| Canonical ARCH-FREEZE package (WBS 54 tasks, matrix, ladder, registers) | `docs/project/maintainers/qualification/ARCH_FREEZE_*.md` | Updated by the orchestrator only. WP-12 nodes and MEP-AF2 tasks are mapped to WBS IDs (appendix Z). |

**Merge rules:**
- **DP-01..DP-68 bind this plan.** Where v1 differed, v2 follows the DP. Examples:
  - DP-13: inversion, no allowlist for the 12 modules;
  - DP-14: agents may import `integrations.contracts` leaves;
  - DP-11: one product wave;
  - DP-34/35: Path B per checklist L66 with an A/B split;
  - DP-42/62: DIST-SCALE-X stays BLOCKS until independent evidence exists, C10/C15 is kept;
  - DP-61: one EBH-7 pin as the last event.
- **Draft PRs already in flight stay in the plan.** #605 (DB-F1 = PT-D01, `T-IGX-042-A`), **#608 (DB-F2 = PT-D07, `T-IGX-042-B`)**, #606 (DB-D = PT-D05, `T-IGX-042-C`), #607 (DB-G = PT-D02, `T-IGX-042-D`). [PD2-12]
  - #606 edits `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` and `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`. It therefore **merges before** T-AF2-SSOT-RECONCILE-B and T-AF2-FRZ-MATRIX (both W2, hard edges). T-AF2-SSOT-RECONCILE-A (W0) does not touch those two files [PD3-01]. The orchestrator, which owns those registers, re-checks #606's hunks.
- **DAG precedence [PD2-05].** Where the WP-12 `DAG.md` and this plan conflict, **this plan wins**. Edge amendments to WP-12:
  - (a) remove PT-VBN from the hard prerequisites of PT-Q13;
  - (b) PT-VBN runs after T-B7-LOCK-FREEZE-SHA and T-B7-ECAP-2C;
  - (c) ENTRY waits for PT-VBN through T-B7-PROMOTE-RPL and COORD;
  - (d) wording: WP-12 "FCS = RB-2 audited SHA" becomes "RB-2 audits G2; FCS = merge commit of the train with a product tree identical to G2";
  - (e) PT-C01's new mapped ENT checks move into the RW train (PT-WD) [PD2-03];
  - (f) PT-D08 and PT-E05 depend on SCOPE, because they ride in the RW train (REV-MASTER RM-07);
  - (g) **SCOPE ← REV-MASTER, T-AF2-RW-PLAN and T-AF2-RW-PLAN-AUDIT** (REV-MASTER RM-02/RM-03);
  - (h) **M3 is in W6**: it is the EBH-7 pin, the last event of W6. W7 starts with LOCK = M3 (REV-MASTER RM-04).

  T-AF2-DAG-REGEN (W0) applies these in `WP-12/plan_data.py`, re-runs the **full** WP-12 pipeline (`reconcile.py`, `blockers.py`, `build_plan.py`, `gen_auth.py`, `gen_master.py`), and re-checks acyclicity, phase check 1 and ENTRY reachability before W0 exits (REV-MASTER RM-01).
- **The `T-IGX-042-*` series belongs to EQO-A.** New tickets use **`T-IGX-043-*`** (W0–W2), **`T-IGX-044-*`** (RW product train) and **`T-IGX-045-*`** (W4–W8).
- **One orchestrator.** This chat is the orchestrator session, with a dashboard (section K). There is no second orchestrator.

---

## Context

ARCH-FREEZE is the last mandatory stage before SCENARIO-FACTORY-X / SEF-X / SCENARIO-READINESS-X.

Its formal predecessors are CLOSED in the roadmap: EVOL-X, ENT-AUDIT-X, MCP-SERVER-X and PDC-X (RM@498, @1847; PRs #603/#604). PDC-X closed as a scoped Variant A: PASS_SCOPED, 60 declared exceptions, 13 PARTIAL rows, no FRZ promotion, EBH-7 L3 replay failed and was not re-run.

The EQO-A Campaign A audit (PR #589, 239 findings) and its phase-2 plan, prepared before PDC-X closed, already plan the remediation route to ENTRY.

This plan joins both into one executable program. It ends at `ARCH-FREEZE CLOSED — INDEPENDENTLY CERTIFIED — EVIDENCE-BACKED — EXACT-SHA`.

---

## A. Executive planning verdict

**At `7960ae7f6`, ARCH-FREEZE is NOT admissible.**

**Confirmed blockers** (verified at HEAD; EQO-A IDs in brackets):
1. **SYS-INV-01.** The gate `scripts/maintenance/check_intergrax_no_applications_imports.py` fails: 12 `intergrax/runtime/**` files import `intergrax.applications.*`. The gate is in no CI profile [A4-003 BLOCKER]. Fix: PT-WA (DP-13).
2. **Tier-2 registry-bypass gate fails.** 10 `agents/*` files import integrations/tools [A4-004]. Fix: PT-WA (DP-14).
3. **RM@499 independence is unmet** for TOOL-LIFE-X (CONDITIONAL/STOP), SKILL-X (conditional), DIST-SCALE-X (operator-scoped), **EBH-7** (B6 not passed; L3 replay FAILED) and **ROADMAP-REPLAY-X** (invalidation counter ≠ 0; 5 reds) [A2-002 BLOCKER, A2-001, A1-017, A5-007, A0-008]. Fix: PT-Q08, PT-Q09, PT-Q11, PT-Q05, PT-D03.
4. **No live merge enforcement on `development`.**
   - Ruleset 26771061 contains only `deletion` and `non_fast_forward`.
   - "Regression gate" is red on every recent push.
   - No CI ran after the rebind (docs-only pushes are `paths-ignore`d).
   - Fix: PT-S01..S06, PT-AUD-ENF12/ENFA (DP-64/67).
5. **PDC-X residuals reachable on production paths** cover tenant isolation, supply chain, prompt trust, streaming, drain and non-CAS recovery. Fix: section E (DS-EXC) and section G (RW train).
6. **FRZ.**
   - 109 rows OPEN: 104 fully architecture-scoped, 3 split (PRD-03, PRD-08, SEC-04) and 2 live (PRD-01/02). [PD2-14]
   - 116 PASS-type rows are historical and must be re-accepted.
   - 8 rows carry labels outside the allowed vocabulary.
   - DEBT-04 requires PARTIAL = 0, but 13 PDC rows are PARTIAL.
   - 35 BLOCKS/CODE_FIX rows have no plan in WP-12 (WP12-Q12).
7. **Missing artefacts:**
   - freeze manifests (FRZ-FRZ-05..08; DP-40);
   - independent VBN acceptance (FIND-VBN-PARTIAL);
   - FREEZE_CERTIFICATION_SHA.

**State of WP-12 itself:**
- Its own review verdict is ACCEPT_WITH_CONDITIONS; the delta review **REV-MASTER** is still open.
- Phase check 3 (negative tests for every CODE_FIX) **FAILS**.
- 53 of the 60 PDC exceptions are `NOT_IN_RW`.
- MEP-AF2 v2 closes these gaps through T-AF2-DS-EXC-PACKAGE, T-AF2-FRZ-MATRIX and T-AF2-RW-PLAN.
- REV-MASTER stays a hard prerequisite of SCOPE.

**Main risks:**
- R1: relocation ripple in PT-WA, which the transitive closure confirms.
- R2: ACT-N11 voids on any product change. Mitigations: train branch, development hold, RB-2 before qualification, merge commits only.
- R3: DIST-SCALE-X envelope under C10/C15 (DP-62).
- R4: feasibility of TYP-02..05 platform-wide.
- R5: false green from known-red classes, `--sync-wbs`, or squash merges.
- R6: an oversized single wave (DP-11). A late defect forces RB-3 (WP12-Q03).

---

## B. Verified baseline (2026-11-16)

| Item | Value |
|---|---|
| `origin/development` | `7960ae7f6` (PR #604); the local checkout `76516c4dd` is 44 behind |
| PDC-X | CLOSED: closure `feaa2a87d` (PR #603), SHA fill-in `399d194e5` (PR #604); global audit PASS_SCOPED @ `969cbd964`; OD-PDC-X-CLOSE signed in chat; the handoff still carries DRAFT headers, but its effective status is HANDOFF READY (SCOPED) and its limitations bind |
| IIAC | IN_FORCE; policy `IIAC-2026-11-05.1`; effective_from `9f05749bc`; corrective `969cbd964`; audit_execution `a73d15ab3`. **ACT-N11 is measured from `audit_execution_sha` `a73d15ab3`** (`iiac_activation_gate.py`); zero product changes since |
| EQO-A phase 2 | W0 decision sessions DS-1..DS-6 answered (DP-01..DP-68); DS-EXC, DS-Q, REV-MASTER and PT-FRZ-RR still open; W1/W2 docs batches in flight (#605, #606, #607, drafts) |
| Branch protection | none on `development` (404); ruleset "Protect development" has `deletion` + `non_fast_forward` and no bypass actors |
| CI on `development` | "Regression gate" failed on the last 12 runs (`governance-fast`: IIAC A-01 tests from before the rebind); no run at HEAD |
| FRZ | 225 / 26; PASS 108 + 8 variants; OPEN 109 |
| WBS | 54 tasks: COMPLETE 18, SUPERSEDED 1, BLOCKED_EXTERNAL 10, PLANNED 25 |

T-AF2-BASELINE-PIN re-verifies every row at its start pin `AUDIT_BASELINE_2`. Any drift goes to a post-baseline delta section.

---

## C. 54-task WBS reconciliation

Rules:
- IDs and history are kept.
- `--sync-wbs` / `--sync-matrix` are **forbidden**: they rewrite rows from hard-coded data (`verify_arch_freeze_plan_traceability.py`@236-1004).
- The **validator hard rules** stay unchanged (@120-207).

| TASK_ID | Historical | v2 qualification | WP-12 / MEP node and authority |
|---|---|---|---|
| T-R0-WAVE1-SSOT-RECONCILE | COMPLETE | COMPLETED_AND_VALID (historical) | succeeded by T-AF2-SSOT-RECONCILE-A + -B |
| T-R0-BASELINE-CLOSURE-HEAD | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | T-AF2-SSOT-RECONCILE-A (pre-admission pins `81829712`/`18a0f68`) |
| T-B0-001 | COMPLETE | SUPERSEDED_BY_CURRENT_AUTHORITY | `AUDIT_BASELINE_2` (T-AF2-BASELINE-PIN) |
| T-B0-SSOT-REFRESH | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | T-AF2-VALIDATOR-AMEND |
| T-B0-TRACE-VALIDATE | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | re-run after T-AF2-VALIDATOR-INDEP-REVIEW |
| T-B1-INV | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | T-AF2-CLOSED-WORLD |
| T-B1-PROD-REACH-SCAN | SUPERSEDED | SUPERSEDED (kept) | — |
| T-B1-ENGINE | COMPLETE | COMPLETED_AND_VALID | reused by T-AF2-CLOSED-WORLD |
| T-B1-MANDATORY-DISPOSITION | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | T-AF2-CLOSED-WORLD (21 `unknown:dynamic`) |
| T-B1-STRICT-QUALIFY | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | T-AF2-CLOSED-WORLD; PT-Q13 |
| T-B1-INDEP-AUDIT-PACKAGE | COMPLETE | COMPLETED_AND_VALID (historical) | — |
| T-B2-BND03-RECONCILE | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED + **TECHNICAL_BLOCKER** | PT-WA, PT-G01 |
| T-B2-VBN | COMPLETE | IMPLEMENTED_NOT_CERTIFIED | T-AF2-VBN-DOSSIER |
| T-B2-DEBT | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | EDER: PT-D02 / PT-R01..R03 (DP-46..52) |
| T-B3-DOC | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | PT-D01/D05/D07/D12 |
| T-B4-L1 / -L2 / -L3-FULL | COMPLETE | COMPLETED_REQUALIFICATION_REQUIRED | PT-Q13 at M3 |
| T-B5-EVOL-B6 | BLOCKED_EXTERNAL | SUPERSEDED_BY_CURRENT_AUTHORITY | EVOL-X B6 `1dd769331` (IIAC) |
| T-B5-EVOL-FRZ | BLOCKED_EXTERNAL | SUPERSEDED_BY_CURRENT_AUTHORITY | FRZ-EVOL rows were set PASS(-SCOPED) through IIAC (checklist @273-282). Stays a **pre-lock** validator ancestor (EVOL chain → … → LOCK), so it is never placed after ECAP-2C [PD2-01]. The M3 re-acceptance moves to T-AF2-REQUAL-HIST-PASS. |
| T-B5-EVOL-FORMAL-CLOSURE | BLOCKED_EXTERNAL | SUPERSEDED_BY_CURRENT_AUTHORITY | `be77b3a5f` |
| T-B6-OD-ENT-00 / T-B6-ENT-B6 | BLOCKED_EXTERNAL | SUPERSEDED_BY_CURRENT_AUTHORITY | ENT-AUDIT-X closed via IIAC |
| T-B6-OD-ENT-01 | BLOCKED_EXTERNAL | COMPLETED_AND_VALID | Baseline `480550d8` was promoted (manifest digest `0fe966e4…`). New promotions: T-AF2-ENT-BASELINE-PREFREEZE and T-B8-ECAP-16. |
| T-B6-OD-ENT-02 | BLOCKED_EXTERNAL | COMPLETED_AND_VALID (compensating mode) | Live required checks = PT-S01..S06 (DP-64/67, AR-45..52), with no edge into OD-ENT-08 |
| T-B6-FRZ-AUD | BLOCKED_EXTERNAL | SUPERSEDED_BY_CURRENT_AUTHORITY | FRZ-AUD rows were set PASS through IIAC (checklist @316-325). Stays a pre-lock validator ancestor [PD2-01]. The M3 re-acceptance (AUD-10 only after PT-AUD-ENFA) moves to T-AF2-REQUAL-HIST-PASS. |
| T-B6-OD-ENT-08 | COMPLETE @ W5-02 | COMPLETED_AND_VALID | Fix the OD register and audit matrix (both still say OPEN) |
| T-B7-LOCK-FREEZE-SHA | PLANNED | CURRENTLY_OPEN (W7, = M3) | — |
| T-B7-L3-FREEZE | PLANNED | CURRENTLY_OPEN (W7) | inside PT-Q13 |
| T-B7-VBN-HEAD-DOSSIER | PLANNED | CURRENTLY_OPEN (W2/W5) | = T-AF2-VBN-DOSSIER |
| T-B7-VBN-INDEPENDENT-ACCEPT | BLOCKED_EXTERNAL | CURRENTLY_OPEN (W7, after LOCK) | = PT-VBN, moved from W6 to W7 because the validator requires VBN-ACCEPT ← LOCK |
| T-B7-ECAP-2C | BLOCKED_EXTERNAL | CURRENTLY_OPEN (W7) | new independent session at M3 |
| T-B7-PFP-ACTIVATE | PLANNED | CURRENTLY_OPEN (W5) | before LOCK |
| T-B7-PFP-VERIFY | PLANNED | CURRENTLY_OPEN (W8) | — |
| T-B7-TEN-REVAL | PLANNED | PDC_RESIDUAL_REVIEW_REQUIRED (W6/W7) | PT-Q07 at FCS, plus a family replay at M3 in PT-Q13 |
| T-PATHB-EXCL | PLANNED | OPERATOR_DECISION_REQUIRED → decided by DP-34/35 (record only) | PT-D10 A/B split |
| T-B7-CMP-REVAL | PLANNED | CURRENTLY_OPEN (W7) | PT-Q12 / PT-Q13 |
| T-B7-PROMOTE-RPL | PLANNED | CURRENTLY_OPEN (W7, ← VBN-ACCEPT) | — |
| T-B7-PROMOTE-DEBT | PLANNED | **TECHNICAL_BLOCKER** until PARTIAL = 0 | RW train + EDER |
| T-B7-PROMOTE-DOC | PLANNED | CURRENTLY_OPEN (W7) | — |
| T-B7-PROMOTE-BND | PLANNED | **TECHNICAL_BLOCKER** | PT-WA, PT-G01 |
| T-B7-PROMOTE-CTR / OWN / PLG / PRD | PLANNED | CURRENTLY_OPEN (W7) | — |
| T-B7-PROMOTE-TYP | PLANNED | **EVIDENCE_BLOCKER** | T-AF2-TYP-MEASURE, OD-AF-11 |
| T-B7-PROMOTE-OBS / REG / REL / SEC | PLANNED | PDC_RESIDUAL_REVIEW_REQUIRED | RW train |
| T-B7-COORD-ARCH-FRZ | PLANNED | CURRENTLY_OPEN (W7 → ENTRY) | — |
| T-B7-FRZ-FRZ | PLANNED | CURRENTLY_OPEN (W8) | — |
| T-B8-ROADMAP | PLANNED | CURRENTLY_OPEN (W8) | — |
| T-B8-ECAP-16 | PLANNED | CURRENTLY_OPEN (W8, **after T-B8-ROADMAP**, validator @162-164) | §16 handoff only, written to `docs/project/maintainers/qualification/arch_freeze_program/ARCH_FREEZE_ECAP16_HANDOFF.md` and linked from the roadmap [PD3-05]. The post-freeze baseline promotion is the separate task T-AF2-POST-BASELINE [PD2-04]. |

**Counts (recomputed mechanically in T-AF2-SSOT-RECONCILE-A):**

| v2.1 status | Count |
|---|---|
| COMPLETED_AND_VALID | 6 |
| COMPLETED_REQUALIFICATION_REQUIRED | 12 |
| IMPLEMENTED_NOT_CERTIFIED | 1 |
| SUPERSEDED (historical) | 1 |
| SUPERSEDED_BY_CURRENT_AUTHORITY | 7 |
| OPERATOR_DECISION_REQUIRED (decided, record only) | 1 |
| CURRENTLY_OPEN, incl. blocker-tagged | 26 |
| **Total** | **54** |

**DAG changes** (implemented in T-AF2-VALIDATOR-AMEND; independent review in T-AF2-VALIDATOR-INDEP-REVIEW):
1. Keep every hard rule:
   - `OD-ENT-08 ← OD-ENT-02`, `PFP-ACTIVATE ← OD-ENT-08`, `LOCK ← PFP-ACTIVATE`;
   - `VBN-ACCEPT ← LOCK`, and `VBN-HEAD` must not depend on LOCK;
   - `PROMOTE-RPL ← VBN-ACCEPT`;
   - the EVOL chain;
   - `COORD` ← its 20 tasks, including EVOL-FRZ and FRZ-AUD.

   All of them are satisfied without rewiring.
2. Treat `SUPERSEDED_BY_CURRENT_AUTHORITY` as a satisfied dependency; it requires an `authority_ref`.
3. New edges. `T-B5-EVOL-FRZ` and `T-B6-FRZ-AUD` get **no** edge after ECAP-2C [PD2-01]:
   - `T-B7-LOCK-FREEZE-SHA` ← T-B7-PFP-ACTIVATE, T-AF2-ENT-BASELINE-PREFREEZE, T-AF2-ECAP-REVISION, T-AF2-MANIFESTS, PT-Q11 (EBH-7 pin = M3);
   - `T-B7-PROMOTE-*` (all 13) and `T-AF2-REQUAL-HIST-PASS` ← `T-B7-ECAP-2C`;
   - `T-B7-COORD-ARCH-FRZ` ← `T-AF2-REQUAL-HIST-PASS` [PD2-11];
   - `T-AF2-POST-BASELINE` ← PT-Q13, OD-AF-16b, OD-AF-17 [PD2-04];
   - `T-AF2-W8-INDEP-CONFIRM` ← `T-AF2-POST-BASELINE`, T-B7-PFP-VERIFY;
   - `T-B7-FRZ-FRZ` ← `T-AF2-W8-INDEP-CONFIRM`;
   - `T-B8-ROADMAP` ← `T-B7-FRZ-FRZ`;
   - `T-B8-ECAP-16` ← `T-B8-ROADMAP`;
   - the final `T-AF2-VBN-DOSSIER` (= T-B7-VBN-HEAD-DOSSIER) does **not** depend on LOCK (validator @199-200) [PD2-11];
   - `T-AF2-SSOT-RECONCILE-B` and `T-AF2-FRZ-MATRIX` (both W2) ← PT-D05 (#606 merged) [PD2-12, PD3-01];
   - `T-AF2-POST-BASELINE` ← `T-B7-COORD-ARCH-FRZ` (ENTRY): baseline promotion stays after ENTRY, as in WP-12 DS-ADM [PD3-06].
4. Computed counters:
   - WBS = 54 + |T-AF2-*| + |mapped PT-*|;
   - checklist PASS/OPEN come from the checklist, and the matrix status is cross-checked against it. **The matrix-vs-checklist cross-check is enabled only from T-AF2-SSOT-RECONCILE-B** (a matrix-header status flag), not in the W0 run [PD4-01];
   - `EXPECTED_QUAL_RUNS` is computed from the qualification plan (it is 16 today, @35/@1188);
   - task links accept `PT-` and `T-AF2-` IDs as well as `T-`, and the ID regex uses a **word boundary** (`\b(?:T|PT)-[A-Z0-9-]+`), so `PT-Q11` is never read as `T-Q11` (@1055) [PD2-07 residual];
   - FIND-PROD-REACH-DEFERRED severity = INFO;
   - **DAG self-test**: a cycle check, the 20 COORD prerequisites, every hard rule, and a test fixture proving that an edge `T-B5-EVOL-FRZ ← T-B7-ECAP-2C` is rejected [PD2-01].
5. OD-AF-01 supersedes "task count remains 54" (WBS@8; OD-PDC-ARCHF-01).

---

## D. 225-FRZ acceptance coverage

**PASS rule** (checklist @36-45; IIAC @38-41). A row is PASS only with all four of:
- the exact SHA = **M3** (product tree ≡ FCS);
- code or contract evidence at M3;
- gate or test evidence executed at M3, including a negative test where the criterion is a prohibition;
- an independent audit row.

The certifier promotes row by row. `N/A — WITH EVIDENCE` needs an evidence file. PARTIAL, SCOPED and COMPENSATING are never final statuses.

**Per-row planning record** (T-AF2-FRZ-MATRIX adds 17 columns to `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md`):

`FRZ_ID | meaning | semantic_owner | contract | production_paths | checklist_status | matrix_status | historical_evidence | evidence_valid_at_HEAD | tests | negative_tests | required_fix | required_certification | operator_decision | PASS_condition | NA_condition | path_B | freeze_impact`

T-AF2-FRZ-MATRIX also produces a **ticket-ready plan for each of the 35 BLOCKS/CODE_FIX rows** (WP12-Q12). Each plan names its scope, negative test, regression scope and sensitivity, and is assigned to an RW-train ticket.

| Family (n) | At HEAD | Route to PASS / N/A | Closing nodes |
|---|---|---|---|
| BND (6) | 6 OPEN | inversion (DP-13), agents (DP-14), HIST-F-BND-03B (DP-21), AST gates (DP-22), layer manifest | PT-WA, PT-G01, T-AF2-MANIFESTS, T-B7-PROMOTE-BND |
| OWN (5) | 5 OPEN | ownership manifest; resolver owner (DP-16); composition roots (DP-18); DUP-X replay | PT-WB, PT-G02, PT-Q05, T-B7-PROMOTE-OWN |
| CTR (6) | 6 OPEN | contract manifest (public/internal); projections (DP-17); typed provider errors | T-AF2-MANIFESTS, T-AF2-RW-RUNTIME, PT-G02, T-B7-PROMOTE-CTR |
| TYP (6) | 6 OPEN | DP-36 (TYP-01/03 remediated); TYP-02/04/05 measured, then fixed or ratchet-gated (OD-AF-11) | T-AF2-TYP-MEASURE, T-AF2-RW-TYP, T-B7-PROMOTE-TYP |
| PLG (8) | 8 OPEN | DP-37; plugin digest, attestation; PLG-02 N/A WITH EVIDENCE under SCOPE-03 | T-AF2-RW-SUPPLY, PT-G02, T-B7-PROMOTE-PLG |
| RPL (4) | 4 OPEN | DP-36 (RPL-04); VBN §2.3/2.6 | T-AF2-VBN-DOSSIER, PT-VBN, T-B7-PROMOTE-RPL |
| EXE (7), GOV (10), TRC (12), STA (8), REC (10), CFG (8), CTL (12), HRN (8) | PASS (historical; EXE 5, GOV 9, TRC 1, REC 8, CFG 1 HIST-REQUAL; GOV-X1 / HARNESS-W4/W7/W8 EV-INCONSISTENT per DP-09) | replay at M3 plus **independent row re-acceptance** | PT-Q06, PT-Q12, PT-Q13, **T-AF2-REQUAL-HIST-PASS** |
| CMP (8) | PASS (1 HIST-REQUAL) | G-24 order-dependent reds; CMP-REVAL | T-AF2-RED-CLASSES, T-B7-CMP-REVAL |
| OBS (7) | 7 OPEN | TRACE-X requalification; EX-L-13 schema; AUDIT_EVIDENCE producer | PT-Q06, T-AF2-RW-GATES, T-B7-PROMOTE-OBS |
| SEC (10) | 10 OPEN | tenant, supply chain, trust boundary, secrets, retention (section E); split for SEC-04, **SEC-05, SEC-06** (below) | T-AF2-RW-TENANT/SUPPLY/TRUST, PT-WF, T-B7-PROMOTE-SEC |
| REL (11) | 11 OPEN | timeouts, stream-open errors, drain, retry cap, saturation | T-AF2-RW-RUNTIME, T-AF2-RW-RECOVERY, PT-FC2, T-B7-PROMOTE-REL |
| PRD (12) | 12 OPEN | DP-34/35/39/41; PRD-01/02 live; PRD-03/08 split; DP-29 (PRD-11 split); rest architecture | PT-D10, PT-FC1..FC3, T-B7-PROMOTE-PRD |
| REG (10) | 10 OPEN | known-red classes, determinism, clean checkout, gates in CI, false-green guard | T-AF2-RED-CLASSES, PT-S06, PT-E02, T-B7-PROMOTE-REG |
| EVOL (10) | 4 PASS + 6 off-vocabulary | OD-AF-08 normalization; LIM-05 stays a limitation | T-AF2-REQUAL-HIST-PASS [PD2-01] |
| DOC (6) | 6 OPEN | docs batches; manifests | PT-D01/D05/D07/D12, T-AF2-MANIFESTS, T-B7-PROMOTE-DOC |
| DEBT (6) | 6 OPEN | EDER as the frozen debt register; PARTIAL = 0; DEFERRED disposed (DP-37, DP-51) | PT-D02, PT-R01..R03, T-B7-PROMOTE-DEBT |
| FRZ (13) | 12 OPEN, FRZ-11 SCOPED | DP-40 artefacts (FRZ-03/05..09); W7 and W8 tasks; **FRZ-11 re-acceptance and label normalization** [PD2-14] | T-AF2-MANIFESTS, T-B7-LOCK, T-AF2-REQUAL-HIST-PASS (FRZ-11), T-AF2-W8-INDEP-CONFIRM (01/02/09/10/13), T-B7-FRZ-FRZ |
| AUD (10) | 9 PASS, AUD-10 COMPENSATING | AUD-10 → PASS only after PT-AUD-ENFA (LIVE) | PT-S01..S06, T-AF2-REQUAL-HIST-PASS [PD2-01] |
| TEN (12) | 12 PASS (all HIST-REQUAL) | TENANT-X re-pin and delta gate (DP-59); requalification after the tenant fixes | PT-Q04, PT-Q07, T-B7-TEN-REVAL |

**Split matrix** (an appendix of the matrix; DP-34/35):

| Row | Dimension A (PASS required) | Dimension B / not claimed |
|---|---|---|
| PRD-01 | — | stays OPEN under PROD-Q, with a no-production-authorisation note |
| PRD-02 | — | live provider qualification (PROD-Q) |
| PRD-03 | unsupported configuration fails closed in STRICT; negative test | live operator profile (PROD-Q) |
| PRD-08 | production-only bypass = 0 (gate) | live environment verification (PROD-Q) |
| SEC-04 | auth bypass = 0 in code and composition | live identity provider (PROD-Q) |
| **SEC-05** | sandbox admission and isolation gates PASS at M3; logical egress deny-all PASS | physical egress and live isolation strength of E2B/Modal/Daytona: `NOT_CLAIMED` in the contract manifest, N/A WITH EVIDENCE for that part |
| **SEC-06** | provider/plugin trust admission (digest, attestation) PASS at M3 | live qualification of E2B/Modal/Daytona as production providers: `NOT_CLAIMED`, frozen-scope exclusion, N/A WITH EVIDENCE |

Path B never holds a tenant, fail-closed, contract or execution-protection obligation. Any wider transfer needs a new explicit Operator decision.

**SEC-05/06 are not Path B rows** (checklist @66). DP-34/35 cover only PRD-02/03/08 and SEC-04.
- Their `NOT_CLAIMED` part is a **frozen-scope exclusion** and needs its own Operator decision, **OD-AF-18** (packet 1). The alternative is to build and qualify those providers. [PD2-07]
- PT-D10 also rewrites the checklist SEC-06 text (@223, "remains OPEN (PROD-Q)") accordingly.

---

## E. PDC-X residual qualification (= content of WP-12 node DS-EXC)

**Deliverable:** T-AF2-DS-EXC-PACKAGE → `docs/project/maintainers/qualification/PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`, with the 13 required columns:

`EXCEPTION_ID | ORIGINAL_REQUIREMENT | AFFECTED_PRODUCT_PATH | ROOT_CAUSE | CURRENT_STATE | PRODUCTION_REACHABILITY | ARCH_FREEZE_IMPACT | REMEDIATION | DEFER_ALLOWED | REOPEN_CONDITION | OWNER | AUDITOR | FINAL_ACCEPTANCE`

The Operator decides DS-EXC on this package. Reachability is re-measured at HEAD by the closed-world engine. For E3/E4-class items, the reachability check runs first (DP-33).

**Policy:**
- Tenant, security, fail-closed, contract and execution-protection defects on a reachable path → `REMEDIATION_REQUIRED_BEFORE_FREEZE`, with `DEFER_ALLOWED = no`.
- Dormant (NC) items → `ACCEPTABLE_SCOPED_WITH_EVIDENCE` only with a non-composition gate that fails closed **and** a `NOT_COMPOSED` mark.
- Missing features follow DP-30: raise until built, plus an exception stated in the freeze documents.

| Disposition | IDs | RW ticket |
|---|---|---|
| REMEDIATION — tenant | EX-L-09: G-09 is fixed by PT-WF (DP-23); **G-12 install-over-occupied CAS is fixed here**. EX-D4-04, EX-D5-05 / PDC-D5-R015, EX-D8-03, EX-D8-04 (bound to principal/run), EX-L-02, F-D2-11 (tenant part). MCP R-P8-01: closes after the B-1 check X-A8-022 (WP12-Q16 A) and an independent audit. | T-AF2-RW-TENANT (+ PT-WF) |
| REMEDIATION — security / supply chain | EX-D1-01 (HIGH), EX-D1-02 (wire both launchers), **PDC-D1-R017** (production consumer of the lifecycle evidence sink), EX-D1-04 (revoked revision with no trusted target stops serving), EX-D6-03, **EX-D6-01** (tamper evidence: build it, or `NOT_CLAIMED` with SEC/AUD wording reviewed in W7), A-01 (trust hook before persist), fail-open set A-12 (F-D3-12, F-D6-09, F-D6-11, F-D5-12, F-D8-14), **F-D6-10 / PDC-D6-R024** (strict validator accepts weak or disabled settings), D6-R007 (F-D6-14, F-D6-03), D6-R010, F-D6-02, F-D6-04 | T-AF2-RW-SUPPLY |
| REMEDIATION — trust boundary | EX-D2-01, EX-D2-02 (default = most-restrictive class), EX-D2-05, EX-D2-07 (STRICT fail-closed), EX-L-05 and **PDC-D2-R004** (RAG red root cause; PARTIAL row) [PD2-10] | T-AF2-RW-TRUST |
| REMEDIATION — retention | EX-D8-01: compose retention plus an AUDIT_EVIDENCE producer. If the Operator chooses the alternative, publish an **explicit responsibility statement** (owner, scope, fail-closed gate) so that SEC-09 can be PASS. Withdrawal alone cannot make SEC-09 N/A. | T-AF2-RW-SUPPLY |
| REMEDIATION — runtime / LLM | EX-L-08, EX-D3-01, EX-D3-03 (typed error contract v1), EX-L-07, D3-R005, D3-R018, D3-R019 | T-AF2-RW-RUNTIME |
| REMEDIATION — background / recovery | EX-L-06, EX-D7-01 / PDC-D7-R014, EX-D4-01, EX-L-03 (CAS), EX-D1-07 (G-08), D7-R015, G-20 | T-AF2-RW-RECOVERY (+ PT-FC2 queue) |
| REMEDIATION — gates / contracts / tests | EX-D5-03, EX-L-01/11 + EX-D3-04 (canonical `get_steps(context)`), EX-L-13, D1-R015, D5-R002, D6-R015 | T-AF2-RW-GATES |
| REMEDIATION — test reds | EX-G-05 (roadmap_replay_x: DP-60) | PT-Q04 / PT-Q05 |
| REMEDIATION — docs | D1-R009, D7-R022, F-D3-16 | PT-D06 / PT-D07 |
| REQUALIFICATION | EX-G-06 (EBH-7: PT-Q11), EX-L-12 (verify T-IGX-041-P), EX-D6-04 (mcp extra), EX-D6-02 (if the production host composes no security controls → T-AF2-RW-SUPPLY), W-01/W-02 exit proof | PT-Q11, PT-Q12 |
| OPERATOR DECISION (scope / claim; recommended option first) | EX-D2-06 (STRICT forbids unscoped memory), EX-D4-02 (in-process supervision only; OS child watch `NOT_CLAIMED`), EX-D4-05 (flag moves to the application profile), EX-D5-01 (groundedness not claimed; D5-R005 reworded), EX-D7-03 (durable producers set an identity), EX-G-03 / PLG-02 (N/A WITH EVIDENCE), EX-L-04 (accept as dormant) | DS-EXC |
| ACCEPTABLE_SCOPED_WITH_EVIDENCE | EX-D1-03 (with an operator procedure), EX-D1-05, EX-D1-06, EX-D2-03, EX-D3-02, EX-D5-02, EX-D7-02 (DP-32), EX-D8-02, EX-L-10, EX-L-14 (fails closed; a live Neo4j run is only a PROD-Q note) | T-AF2-RW-GATES (reachability gates) |
| ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | EX-G-02, EX-G-07, EX-D2-04, EX-D5-04, EX-D4-03 (lab only; needs a PRD-07 separation gate) | T-AF2-MANIFESTS |
| TRANSFER_TO_LIVE_PROD_Q | EX-G-04, limited to the PDC rows it names (D3-R021, D6-R026, live parts of D3-R017/D6-R010) under PRD-02. SEC-05/06 are **not** transferred; they are handled by the split and OD-AF-18. [PD2-07] | — |
| Bounded claim | EX-G-01: D7-R012 → VBN §2.4 fixed substrate (SQLite single writer) with a multi-instance fail-closed guard; D6-R012 → see split SEC-05; D8-R015 → re-measured in W2, then fixed or bounded | T-AF2-VBN-DOSSIER |
| Other handoffs | R-P7-01..06 → disposition in W2 (DP-45: R-P7-04/06 are limitations); R-P7-02 → T-AF2-RED-CLASSES; F-P8-01, F-MCP-P5/P6-GOV, P8-PROC-01 → process lessons checked in PA-1 (section O); LIM-RESEARCH-STRICT-FACTORY-MEMORY → out of scope with evidence; F-W3-001 → T-AF2-RED-CLASSES; OPEN-BP-01 → PT-S01..S06 | — |
| PDC handoff §5.9 / §6 | dependency-bot / `uv.lock` batch → T-AF2-DEPS-BATCH (in the RW train); validator run at the closure SHA → T-B8-ROADMAP | — |

**DEBT-04 / DEBT-05:** each of the 13 PARTIAL rows ends fixed, N/A with evidence, or out of scope by OD. T-AF2-DS-EXC-PACKAGE carries them as their own table [PD2-10]:

| PARTIAL row | Node |
|---|---|
| D1-R009 | PT-D06 |
| D1-R015 | T-AF2-RW-GATES |
| D1-R017 | T-AF2-RW-SUPPLY |
| D2-R004 | T-AF2-RW-TRUST |
| D3-R005 | T-AF2-RW-RUNTIME |
| D3-R018 | T-AF2-RW-RUNTIME |
| D3-R019 | T-AF2-RW-RUNTIME + T-AF2-VBN-DOSSIER |
| D5-R002 | T-AF2-RW-GATES |
| D6-R007 | T-AF2-RW-SUPPLY |
| D6-R010 | T-AF2-RW-SUPPLY |
| D6-R015 | T-AF2-RW-GATES |
| D7-R015 | T-AF2-RW-RECOVERY |
| D7-R022 | PT-D06 |

**Every ID appears exactly once**, checked mechanically over the 60 EX IDs **and** the 13 PARTIAL IDs.

**WP12-Q02 answered** (the 3 CODE_FIX findings without a plan row) [PD2-15]:
- **A4-005** (dependency-ownership gate red: jsonschema, tzdata) → T-AF2-DEPS-BATCH;
- **A4-018** (direct `ollama` SDK use) → T-AF2-RW-RUNTIME;
- **A3-018** (STATE-X citations) → PT-G02 (its "STATE-X citation repair").

T-AF2-RW-PLAN gives each of them a negative test.

---

## F. Consolidated gap register (added to `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md` with links to EQO-A IDs and EDER)

| ID | Sev | Finding | Owner node |
|---|---|---|---|
| FIND-AF2-01 | BLOCKER | SYS-INV-01: 12 runtime → `intergrax.applications` imports; the gate is not in CI (A4-003) | PT-WA, PT-G01, PT-S06 |
| FIND-AF2-02 | HIGH | 10 agents import integrations/tools (A4-004) | PT-WA (DP-14) |
| FIND-AF2-03 | BLOCKER | RM@499 independence unmet: SKILL-X, TOOL-LIFE-X, DIST-SCALE-X, **EBH-7, ROADMAP-REPLAY-X** | PT-D03, PT-Q08, PT-Q09, PT-Q11, PT-Q05, T-AF2-RM499-ENTRY |
| FIND-AF2-04 | BLOCKER | no required checks on `development` | PT-S01..S06, PT-AUD-ENF12/ENFA |
| FIND-AF2-05 | HIGH | Regression gate red; no run at HEAD | T-AF2-CI-HEAD-RUN, PT-Q03 |
| FIND-AF2-06 | HIGH | 116 historical PASS rows and 18 stages without a current replay | PT-Q05..Q07, PT-Q12, PT-Q13, T-AF2-REQUAL-HIST-PASS |
| FIND-AF2-07 | MED | 8 off-vocabulary labels; checklist @47 stale | PT-D10, T-AF2-REQUAL-HIST-PASS |
| FIND-AF2-08 | MED | ARCH-FREEZE package drift; validator hard-codes | T-AF2-SSOT-RECONCILE-A/-B, T-AF2-VALIDATOR-AMEND |
| FIND-AF2-09..12 | HIGH | PDC residuals: tenant, supply chain, trust, runtime/recovery | section E tickets |
| FIND-AF2-13 | MED | dormant capabilities without a contract mark | T-AF2-MANIFESTS, T-AF2-RW-GATES |
| FIND-AF2-14 | MED | claimed features missing | DS-EXC |
| FIND-AF2-15 | HIGH | DEBT-04/05 vs PARTIAL and DEFERRED | EDER, RW train |
| FIND-AF2-16 | HIGH | known-red classes | T-AF2-RED-CLASSES |
| FIND-AF2-17 | MED | C10/C15 limits (DP-62) | T-AF2-VBN-DOSSIER, PT-Q09 |
| FIND-AF2-18 | MED | ACT-N11 fragility | section K |
| FIND-AF2-19 | MED → **CLOSED (FALSE)** | PR workflows may run `main`'s copy (`unit-tests.yml`@9-10). **W0 probe PR #609 (run 43858457516) showed that PR runs use the PR branch's workflow definition**; the header comment in `unit-tests.yml` is wrong (a doc fix goes to T-AF2-RED-CLASSES). T-AF2-MAIN-SYNC is not needed. | T-AF2-CI-HEAD-RUN (done) |
| FIND-AF2-20 | MED | freeze manifests missing | T-AF2-MANIFESTS (DP-40) |
| FIND-AF2-21 | — | parallel EQO-A track (PA0-01) | resolved by OD-AF-00 (merge) |
| FIND-AF2-22 | HIGH | TYP-02..05 feasibility | T-AF2-TYP-MEASURE, OD-AF-11 |
| FIND-AF2-23 | HIGH | WP-12 phase check 3 FAIL: 18 CODE_FIX without explicit negative tests | T-AF2-RW-PLAN |

Historical findings:
- **FIND-VBN-PARTIAL** → PT-VBN.
- **FIND-FORMAL-B6** → superseded (IIAC).
- **FIND-ACCEPTED-BASELINE** → T-AF2-ENT-BASELINE-PREFREEZE and T-B8-ECAP-16.

---

## G. Technical remediation: the RW product train (W3)

**Shape:** one train (DP-11) on branch `arch-freeze/rw-train`. It contains:
- the WP-12 product tickets: PT-WA, PT-WB, PT-FC1..FC4, PT-WD, PT-WE (conditional), PT-WF, PT-D08, PT-E05 (material path), PT-T01 (tests);
- the MEP-AF2 tickets listed below.

**T-AF2-RW-PLAN (W2) covers four things:**
- the file-scope overlap matrix for every train ticket;
- the merge order;
- one explicit negative test per CODE_FIX finding (closing phase check 3);
- the regression scope per ticket.

**Order:**
1. PT-WA alone, first (it has the largest blast radius).
2. PT-WB.
3. In parallel, up to 5 tickets with disjoint scopes.
4. Tickets that overlap a ticket already running are serialized. Known overlaps: `applications/_shared/harness_host_runtime.py` (PT-FC1, PT-WF, T-AF2-RW-TENANT) and `applications/_shared/*_wiring.py` (PT-FC2, PT-WB, T-AF2-RW-SUPPLY).

**PT-WA design** (DP-13/14/21/22, verified against the transitive closure computed in T-AF2-BND-CLOSURE):
- Tier-1 callers reach the 12 modules (`environment_orchestration_materialization.py`, `host_orchestration_loop_init_spec.py`, `runtime/task/nexus_worker_execution.py`). The fix is therefore **inversion**:
  - neutral Tier-0 contracts and ports in `intergrax/contracts/`;
  - Tier-3 builds the implementations;
  - Tier-1 applies them through the injected ports.
- There is no allowlist for these modules.
- Agents may import only the `integrations.contracts` leaf modules, and the registry-bypass gate is rewritten (DP-14).
- Estimated size: about 20 modules and about 30 test files.
- Negative tests:
  - an AST gate fixture that imports `intergrax.applications` from Tier-0/1 must fail;
  - so must a lazy function-level import.

**MEP-AF2 train tickets** (scopes are indicative; the binding scope comes from T-AF2-RW-PLAN):

| Ticket | Scope | Fix | Mandatory negative test |
|---|---|---|---|
| T-AF2-RW-TENANT | sandbox `hosted_session.py`, storage/workspace tools, `ToolWiringContext`, `ArtifactAccessContext`, `OnlineEvaluationObservation`, `admin_service.py` (install CAS), MCP host auth | per-tenant sandbox sessions; tenant-bound object keys; tenant in the tool context; artifact read bound to principal/run; `tenant_id` on the observation; CAS install; MCP without harness auth refuses a client-asserted tenant | cross-tenant read/write/install rejected; client-asserted tenant refused |
| T-AF2-RW-SUPPLY | `intergrax/agent_distribution/`, both launchers, `core/plugins/`, `runtime/security/`, `secret_safety.py`, `security_audit_trail.py`, `governance/quota.py`, strict-config validator, lifecycle evidence sink consumer, retention composition | attestation verified; revocation source and trust coordinator injected; trust hook before persist; digest check; fail-open set closed; strict validator refuses weak or disabled settings; secret-gate evasions closed; evidence-sink consumer; retention plus AUDIT_EVIDENCE producer | unsigned or tampered package refused; malformed quota → deny; weak strict setting refused |
| T-AF2-RW-TRUST | `llm/messages.py`, `graph_assembly.py`, `context/provider_lifecycle.py`, `context/planner.py` | trust classes; untrusted content never `role=system`; orphan `tool_call_id` fails closed in STRICT; most-restrictive egress default; ledger sizes the full message | an injection fixture cannot reach the system role |
| T-AF2-RW-RUNTIME | `intergrax/llm_adapters/` | stream-open errors pass through breaker/retry; timeout forwarded; typed error contract v1; served-model attribution; refusal-normalization matrix | a stream-open failure trips the breaker; the timeout is enforced |
| T-AF2-RW-RECOVERY | `background_tasks/`, `queueing/`, `document_store_task_queue.py`, `colocated_worker.py`, `runner.py`, `long_running/scheduler.py`, `durable_registry_projection_input_store.py` | drain controller; `wait_for_idle`; CAS recovery; full durable CAS; UNCERTAIN reconciler; capped retry | no double execution across restart or stop |
| T-AF2-RW-GATES | `scripts/gates/` promotion gate, the scenario `get_steps` contract, `runtime/events/payloads/canonical.py`, `qualification_runtime_trace.py`, eval digests, NOT_COMPOSED reachability gates | gate bound to stage plus ticket; one canonical signature; schema reconciled; digest enforced; non-composition gates | an unrelated ticket gets DENIED; composing a NOT_COMPOSED feature raises |
| T-AF2-RW-TYP | frozen contract surfaces | DP-36 (TYP-01/03, RPL-04); TYP-02/04/05 per OD-AF-11 | ratchet gate fails on a new `Any` in a frozen contract |
| T-AF2-DEPS-BATCH | `pyproject.toml`, `uv.lock` | held dependency-bot batch (PDC §5.9) | `uv sync --frozen` succeeds in CI |

The **RW head = G2** once every train ticket has merged into the train and the train CI is green, except for the mechanically derived expected-red set.

---

## H. Vendor/backend neutrality plan

| Step | Node | Content |
|---|---|---|
| 1 | T-AF2-VBN-DOSSIER (W2 draft, W5 final) | §2.1 (domains 1–31, `unclassified` = 0, 54 CONFIG-X concerns); §2.2 (PROHIBITED → RESOLVED/ALLOWED/ADR); §2.3; §2.4 fixed substrates (SQLite single writer; in-memory graph defaults); §2.5; §2.6 (full matrix, not a sample, EBH-7 level); **§3 SQLite annex (8 questions; Q8 multi-process validity is tied to the DIST-SCALE-X envelope, DP-62)**; D3-R019 |
| 2 | `T-B7-VBN-HEAD-DOSSIER` | the dossier file `ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md` |
| 3 | PT-Q13 at M3 | certifying runs for §2.1–2.6; §2.7 `prohibited_unresolved` = 0, `promised_unqualified` = 0 |
| 4 | **PT-VBN = T-B7-VBN-INDEPENDENT-ACCEPT** (W7, after LOCK) | a new independent session gives **7 per-artefact verdicts** with evidence SHA = M3, **independently ratifies each fixed substrate (§2.4)**, replays the 12 VBN checks with `ENT_AUDIT_X_SEMANTIC=1`, and signs §2.7. EBH-5 and ENT VBN 12/12 are inputs only. |

---

## I. Enterprise qualification plan

| Level | Content | When |
|---|---|---|
| L1 | all tier / registry / vendor gates, including SYS-INV-01 and the upward-import gate; `igx_gov ticket`; traceability validator; B1 strict | every PR; RW head; M3 |
| L2 | `tests/unit -m "gate and not no_ci"`; governance suite; touched domain suites | every PR (targeted); M3 (full) |
| L3 | pdc_x/xd, ent_audit_x, TENANT-X FINAL, EBH-7 P03 L3, EVOL P3 L3, ROADMAP-REPLAY-X, TRACE-X, STATE-X, DUP-X | W6 (PT-Q05..Q07, Q11, Q12) and M3 (PT-Q13) |
| FULL / FREEZE | `run_enterprise_audit.py run --mode FULL`; `--mode FREEZE --evidence-bundle` (certifying, clean tree) | M3 |
| CHANGED-SINCE | `--since-sha <M3>` | W8 corrective loop and terminal check |
| Adversarial | Campaign B-1/B-2 alpha at B, omega at FCS (PT-B01/B02/B1W/B2W, PT-BA); NT-01..16 red-first (PT-T01); the section G negative tests | W2, W3, W6 |
| Mutation | negative tests of T-AF2-RW-TENANT/SUPPLY (wave audit mutation check in PT-Q02) | W4 |
| Clean checkout | fresh-clone tests, autocrlf true/false/input | M3 |
| Exact-SHA validation | IIAC validators; ladder consistency; ancestry | W4, W7, W8 |

**CI execution premises:**
- Push CI runs only on `main` and `development` (`unit-tests.yml`@20-24).
- Dispatch runs at the tip of the given ref.
- A certifying FREEZE run needs a clean tree (`runner.py`@85).

Rules:
- Qualification dispatches use **pinned refs** `af2/pin-<sha>`. The orchestrator checks `run.head_sha` against the pin.
- FREEZE runs on a **detached clean checkout**.
- The expected-red set on the train comes from a W0 **probe PR**: a one-line throwaway product change, closed without merging. It is stored in `.tmp/session/arch-freeze/ci/expected_red_A01.json` and may only shrink.
- **FIND-AF2-19 is a blocking input to PT-S01** (the ruleset's required check names must exist in the workflow definition that PR runs use). T-AF2-CI-HEAD-RUN settles it in W0 with a **probe PR from a branch that changes a workflow**, which shows which definition a PR run actually uses. If `main`'s copy is used, **T-AF2-MAIN-SYNC runs in W1 before PT-S01**. It is an outward-facing merge to `main`, so it needs its own authorization (AR-AF2-01) and has its own ancestry rule. [PD2-08]

**ENT baseline chain** (`runner.py`@220-234; `iiac_baseline_promotion.py`@9-22, @260-294). New gates change the check-manifest digest, so certifying runs would raise `BASELINE:MANIFEST_DIGEST`. Therefore **T-AF2-ENT-BASELINE-PREFREEZE** (W5, before LOCK):
1. `--mode FREEZE --bootstrap-candidate` at the pre-pin tip;
2. an auditor session and a certifier session produce the **ENT-AUDIT-X IIAC manifest** (`stage_id=ENT-AUDIT-X`, verdict PASS/PASS_SCOPED, the same TES and evidence pins, three pairwise-distinct sessions);
3. the **Operator delegation file** is published under `iiac_evidence/ENT-AUDIT-X/operator_decisions/` (its SHA-256 equals the record minus its signature lines) **before** the record;
4. a non-certifying dry check with `verify_delegated_record`;
5. promotion through `iiac_baseline_promotion.py`.

**Rule: the ENT engine and the check manifest live under `intergrax/` [PD2-03].** `intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json` and `intergrax/platform/enterprise_audit/**` are ACT-N11 product paths. Therefore:
- every change to the ENT manifest or engine lands in the **RW train before G2** (PT-WD, including PT-C01's new mapped checks);
- gates added after FCS (PT-S05/S06, the manifest drift gate) live only in `scripts/maintenance`, CI or `tests/`, and are **not** registered as ENT manifest checks;
- T-AF2-ENT-BASELINE-PREFREEZE asserts `git diff --name-only G2 <tip> -- intergrax/ agents/ applications/` is empty before it starts.

**Post-M3 path allowlist (EBH-7 identity gate) [PD2-02].** After M3, only the paths accepted by `scripts/qualification/verify_ebh7_p03_immutable_evidence.py`@19-37 may change:
- the prefixes `docs/project/maintainers/qualification/` and `.github/`, although this plan freezes `.github/workflows` anyway;
- 9 exact files, including the roadmap.

Consequences:
- W7/W8 work is **ticket-less docs commits under `docs/project/maintainers/qualification/`**. Every `T-IGX-045-*` ticket file is created and finalized **before the pin**, and `governance/tickets/` is not edited after M3.
- The post-freeze ENT baseline promotion rewrites `tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json` (`baseline.py`@31), which is **not** allowlisted. Before the pin, inside PT-Q11's pre-pin batch, the Operator decides **OD-AF-17**:
  - **A (recommended):** add that exact file to `_EXECUTABLE_IDENTITY_ALLOWLIST_EXACT`. This is a gate-meaning change of the DP-60 type, so it needs its own independent review inside PT-AUD-DOC.
  - **B:** record the post-freeze baseline as a candidate file under `qualification/`, and write the accepted JSON only in the first post-freeze change (SCENARIO-FACTORY-X entry). In that case FRZ-FRZ-13 is met by the recorded candidate plus its IIAC chain.
- T-AF2-CI-HEAD-RUN adds a probe that runs `executable_identity_violations` against a simulated W8 diff. The probe **runs again in W6 on the pre-pin tip** after the OD-AF-17 change, to show that option A works [PD3-05].
- Under option B, T-AF2-W8-INDEP-CONFIRM must state explicitly that FRZ-FRZ-13 and RM@499 ("baseline SHA recorded") are met by the recorded candidate and its IIAC chain [PD3-05].
- A **ticket-less docs-only PR trial** under `qualification/` must pass the `ticket-governance` check (from PT-E02). It is **mandatory at PT-S05**, when the check becomes required, and is repeated in the W6 identity probe. The W1 trial in PT-S01 is optional [PD3-05, PD4-02].

The same delegated chain runs again in W8 as **T-AF2-POST-BASELINE** (OD-AF-16b). The baseline SHAs go into the ladder before T-B8-ROADMAP. [PD2-04]

**Known-red policy (T-AF2-RED-CLASSES):**
- Each class is FIX, ENV (proven to be environment-only, for example MAX_PATH by a short-path run) or OUT (by OD).
- No skip or xfail may hide a red.
- At M3 the expected-red set equals the registered ENV/OUT set exactly.

**Process budget:**
- one `pytest` process at a time across all sessions, using the shared lock `<worktree base>/_locks/pytest.lock` (EQO-A rule);
- broad suites run in GitHub Actions;
- logs go to `.tmp/session/arch-freeze/<task>/`.

---

## J. Merged execution DAG

**SHA ladder:**

| Rung | Meaning |
|---|---|
| `AUDIT_BASELINE_2` | W0 pin |
| `G2` | RW head = wave-audited SHA |
| `R` | RB-2 phase-2 commit; `git diff G2 R` = docs only |
| `FCS` | merge commit of the train into `development`; product tree ≡ G2 |
| `M3` | EBH-7 final pin = **FREEZE_CERTIFICATION_SHA = TEST_EXECUTION_SHA**; product tree ≡ FCS |
| `AUDIT_EXECUTION` | — |
| `EVIDENCE_PUBLICATION` | — |
| `CERTIFICATION_PUBLICATION` | — |
| `FORMAL_CLOSURE_PUBLICATION` | — |

| Wave | Nodes (WP-12 `PT-*` / `DS-*` plus MEP `T-AF2-*` / WBS) | Exit gate |
|---|---|---|
| **P0** | PA-0 delta re-audit of v2 (independent session) | no open plan finding ≥ MAJOR |
| **W0 — decisions and SSOT** | DS-1..DS-6 (DONE); **DS-EXC** (input: T-AF2-DS-EXC-PACKAGE); **DS-Q** (resolved as record-only per M2); PT-FRZ-RR; REV-MASTER (WP-12 delta review RV-01..06); AUTH-PRE. MEP: T-AF2-BASELINE-PIN, T-AF2-OD-RECORD (OD-REM-01..08, C1–C10, DP-01..68, OD-AF-*, verbatim English rendering, into `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`; also delivers DP-07), T-AF2-SSOT-RECONCILE-A, T-AF2-VALIDATOR-AMEND, T-AF2-CI-HEAD-RUN (+ FIND-AF2-19 workflow-source probe + EBH-7 identity probe), T-AF2-EXPECTED-RED-PROBE, T-AF2-DS-EXC-PACKAGE, **T-AF2-DAG-REGEN** (applies the section 0 amendments) [PD2-05], **T-AF2-PLAN-PUBLISH** [PD2-06] | Operator packet 1 answered; DS-EXC decided; REV-MASTER passed; DAG-REGEN acyclic with ENTRY reachable; validator exit 0 on the part-A state (matrix and gap register reconciled later, in SSOT-RECONCILE-B) [PD3-01] |
| **W1** | PT-D01 (#605), PT-E00, PT-E01, PT-E02, **T-AF2-MAIN-SYNC (only if FIND-AF2-19 is confirmed; before PT-S01)** [PD2-08], PT-S01 (trial), PT-S02 (Stage 1), PT-S07 (optional), PT-V02, PT-V05 | merged or waiting for an AR |
| **W2** | PT-D02 (#607), PT-R01..R03, PT-D03..D07 (#606 = PT-D05 before SSOT/FRZ-MATRIX; #608 = PT-D07), PT-D12, PT-G01, PT-G02, PT-E03, PT-E04, PT-V03, PT-V04, PT-V06, PT-B01, PT-B02, PT-S03, PT-AUD-ENF12. MEP: T-AF2-SSOT-RECONCILE-B, T-AF2-FRZ-MATRIX, T-AF2-CLOSED-WORLD, T-AF2-BND-CLOSURE, T-AF2-TYP-MEASURE, T-AF2-RM499-ENTRY, T-AF2-MANIFESTS (draft, DP-40), T-AF2-VBN-DOSSIER (draft), T-AF2-ECAP-REVISION (draft), T-AF2-RED-CLASSES, **T-AF2-RW-PLAN**, T-AF2-VALIDATOR-INDEP-REVIEW | **SCOPE** = DS-1..DS-4 + DS-EXC + S1 + B-1/B-2 alpha + REV-MASTER + T-AF2-RW-PLAN + T-AF2-RW-PLAN-AUDIT (hard edges in the regenerated DAG) |
| **W3 — RW train** | PT-WA → PT-WB → {PT-FC1..FC4, PT-WD, PT-WE (cond.), PT-WF, PT-D08, PT-E05, PT-T01, T-AF2-RW-TENANT, -SUPPLY, -TRUST, -RUNTIME, -RECOVERY, -GATES, -TYP, T-AF2-DEPS-BATCH} | RW head G2 frozen; train CI green except the exact expected-red set |
| **W4 — wave audit and RB-2** | **PT-Q02** (independent wave audit at G2, which is also the IIAC delta audit `a73d15ab3..G2`, verdict exactly PASS for RB-2) → **PT-Q03** RB-2 (certifier and activation-delegate sessions; IIAC-GOV-1 manifest; field map in L) → R → **T-AF2-RW-MERGE** (merge commit into `development`; FCS) → PT-S04 (Stage 3, after ≥ 20 green runs) | `verify_iiac_policy_activation.py --require-in-force` exit 0 at FCS; `governance-fast` green |
| **W5 — pre-pin preparation** | PT-Q04, PT-PB, PT-D09 (roadmap), PT-D10 (checklist; A/B split incl. SEC-05/06), PT-AUD-DOC, PT-D11, PT-S05, PT-S06, PT-AUD-ENFA. MEP: T-B7-PFP-ACTIVATE (rebaselines `INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md` and `..._ARCHITECTURE_GUARD_MATRIX.md`; contract versioning, deprecation, migration; requalification triggers; escalation), T-AF2-MANIFESTS (final, drift-gated), T-AF2-ECAP-REVISION (final; FRZ recount), T-AF2-VBN-DOSSIER (final), T-AF2-ENT-BASELINE-PREFREEZE | everything merged and reviewed before the pin |
| **W6 — requalification** | PT-B1W, PT-B2W, PT-BA, PT-C01, PT-Q05 (RRX counter 0), PT-Q06 (STATE-X, TRACE-X), PT-Q07 (TENANT-X, TEN-REVAL at FCS, MCP delta), PT-Q08 (TOOL-LIFE-X RA-1), PT-Q09 (DIST-SCALE-X scoped envelope), PT-Q10 (optional), PT-Q12 (family suites), PT-D13 → **PT-Q11** EBH-7 final pin plus independent B6 (last event) = **M3** | verdicts published; M3 recorded |
| **W7 — freeze certification at M3** | **T-B7-LOCK-FREEZE-SHA** (= M3) → **PT-Q13** (T-B7-L3-FREEZE: L1–L3, FULL, FREEZE certifying, clean checkout, FRZ-AUD-10 LIVE check, EDER populated) → **T-B7-ECAP-2C** (new independent session, ≠ the PT-Q02 auditor: C1–C5 full platform at M3, residual-qualification acceptance) → **PT-VBN** → certifier: **T-AF2-REQUAL-HIST-PASS** (EXE, GOV, TRC, STA, REC, CFG, CTL, HRN, CMP, TEN, **EVOL-01..10, FRZ-11, AUD-01..10** with OD-AF-08 normalization) [PD2-01, PD2-14], T-B7-TEN-REVAL, T-B7-CMP-REVAL, T-B7-PROMOTE-* (13) → T-B7-COORD-ARCH-FRZ → **ENTRY** (the Operator's L499 entry check). All commits are docs-only under `qualification/` and ticket-less (post-M3 allowlist) [PD2-02]. | architecture-scoped OPEN = 0, except the W8 set {FRZ-FRZ-01, -02, -09, -10, -13} |
| **W8 — admission and closure** | T-B7-PFP-VERIFY (CHANGED-SINCE M3); **T-AF2-POST-BASELINE** (FREEZE evidence bundle at M3 + delegated chain from section I; OD-AF-16b; file location per OD-AF-17) [PD2-04]; **T-AF2-W8-INDEP-CONFIRM** (new independent session: row verdicts for the W8 set plus the ECAP §15 12-item confirmation, before admission); T-B7-FRZ-FRZ (§15 admission dossier); **DS-ADM** (OD-ARCH-FREEZE-01 and EDER-D-0170/0172/0173); T-B8-ROADMAP (closure text, ledger, ladder, validator at the closure SHA); T-B8-ECAP-16 (§16 handoff to SFX / SEF-X / SCENARIO-READINESS-X); terminal consistency check (N13) | section N all true; `executable_identity_violations(M3, HEAD)` empty at every W8 commit |

**Critical path:** P0 → W0 (DS-EXC, REV-MASTER) → W2 (SCOPE, T-AF2-RW-PLAN) → W3 PT-WA → W3 train → W4 PT-Q02 → PT-Q03 → FCS → W5 PT-D09/D10/AUD-DOC → W6 PT-Q11 (M3) → W7 PT-Q13 → T-B7-ECAP-2C → COORD → ENTRY → W8 → CLOSED.

---

## K. Orchestration: this session as orchestrator with a dashboard

**Orchestrator.** This chat runs as role `orchestrator` (`role_registry.yaml`). It:
- keeps the dashboard;
- prepares ticket plans, briefs, scope files and SHAs;
- starts **one sub-session per executable step** (implementation ticket, run, campaign, audit);
- uses sub-agents only for read-only analysis that returns within a turn;
- edits shared registers itself, in dedicated docs PRs: WBS, matrix, gap register, OD register, ladder, EDER index, checklist and roadmap (the last two only through PT-D09/PT-D10).

**Dashboard.**
- A private Artifact page ("ARCH-FREEZE Orchestrator") with the `db` capability, one row per node.
- Columns: node | wave | kind | ticket | session | branch / PR | status | SHA | gate | blocker.
- Statuses: `PENDING`, `READY`, `RUNNING`, `REVIEW`, `MERGED`, `AUDITED`, `BLOCKED`, `DONE`.
- The orchestrator updates rows with ArtifactData after every session report. The dashboard is a view only; the repo registers stay the evidence.
- **Classification [PD2-16]:** the page is private. It names defects only by ID, title and node, with no exploit detail, reproduction steps, secrets or credential references. It is **not evidence**, and the repo registers remain authoritative.

**Session types:**

| Kind | Who runs it | Model | Worktree | Permission |
|---|---|---|---|---|
| Implementation ticket (DOC / TEST / PROD / MAT) | sub-session, `implementer` role | Sonnet | own worktree `<parent>/_wt/intergrax/<slug>` (slug ≤ 24) | normal |
| Review of a PR | sub-agent `reviewer` (read-only), or a sub-session for product PRs | Sonnet | shared read-only | read-only |
| Run / Campaign B (non-certifying) | sub-session | Sonnet | scratch worktree at the pinned SHA | read-only on tracked files |
| Independent audit / certifier / activation delegate | **new sub-session started only on the Operator's confirmation** (IIAC: no orchestrator context; brief = SHA + scope file + evidence index) | Opus | shared read-only | plan / read-only |
| Decision intake (DS-*) | this chat, with the Operator | — | — | — |

**Concurrency:**
- at most 6 live sub-sessions, and at most 5 implementers in the W3 train;
- one `pytest` process at a time (shared lock);
- file-scope intersections are computed before every launch, and overlapping tickets are serialized.

**Branches:**
- the RW train is `arch-freeze/rw-train`; train tickets use `af2/<slug>` → PR into the train;
- non-product W1/W2/W5/W6 tickets → PR into `development`.

**Development hold (OD-AF-02) [PD2-09]:**

| Period | Rule on `development` |
|---|---|
| **From now until FCS** (ACT-N11 has measured from `a73d15ab3` since PDC closure) | No change under `intergrax/`, `agents/` or `applications/`. No change to the IIAC protocol doc, the 5 IIAC scripts or the 6 material governance paths. Product changes exist only on `arch-freeze/rw-train`. |
| **From FCS until the end** | Nothing under `intergrax/`, `agents/` or `applications/`. **At no time after RB-2** may the IIAC protocol doc, the 5 IIAC scripts or the 6 material governance paths (`iiac_activation_gate.py`@40-49) change [PD3-02]. |
| **From FCS until M3** (pre-pin) | Allowed non-product edits: docs; `tests/` (respecting the EBH-7 allowlist ordering); `scripts/maintenance` and `scripts/governance` gates that are not ENT manifest checks and **not** IIAC material paths; `.github/workflows` (PT-S05/S06, drift gate); `governance/tickets/` (all W7/W8 tickets finalized here). |
| **After M3** | Only the EBH-7 post-pin allowlist, further restricted by this plan to docs and evidence under `docs/project/maintainers/qualification/` and the roadmap. |

The orchestrator re-checks `git diff --name-only` over the held paths before every wave. Any hit → STOP.

**Merges:**
- pinning PRs and the train merge use a merge commit only (the repo allows squash and rebase, so this is a rule the orchestrator enforces);
- compare `headRefOid` with the pushed SHA and wait for that head's CI before merging;
- set the upstream before the first push, then push with an explicit refspec.

**Tickets and approvals (PA0-14):**
- Per wave, the orchestrator drafts every ticket plus `PLAN_T-IGX-…md` and the plan digest.
- The Operator writes the `Approved:` lines for the whole wave in one approval packet, or names a recorded delegate under DP-07/DP-08.
- `verify_plan_approval.py` must pass before a sub-session starts.
- Every product commit carries the `Ref:` trailer. No AI attribution.

**IIAC text hygiene:** a lint for the `_CLOSURE` regex (`iiac_activation_gate.py`@35) runs on every record and OD activation draft (T-AF2-RED-CLASSES adds it as a tool).

---

## L. Independent certification plan

- **Roles:**
  - IMPLEMENTER: the orchestrator and the implementer sessions.
  - INDEPENDENT AUDITOR: a new session that never saw implementation context. PT-Q02, T-B7-ECAP-2C, PT-VBN and T-AF2-W8-INDEP-CONFIRM are done by **at least two distinct auditor sessions**. The ECAP-2C auditor ≠ the PT-Q02 auditor, without exception.
  - CERTIFIER: a new session that did not author the validators it uses.
  - ACTIVATION DELEGATE: a separate session.
  - OPERATOR: decides and may delegate signing (DP-07/DP-08). Delegations are recorded.
- **RB-2 (PT-Q03) field and ancestry map** (`iiac_activation_gate.py`@34, @258-285, @381-399, @486-499, @545-562):

  | Field / check | Requirement |
  |---|---|
  | manifest `stage_id` | `IIAC-GOV-1` |
  | manifest verdict | exactly `PASS` |
  | `audited_implementation_sha` | = `corrective_implementation_sha` = G2 |
  | manifest `audit_execution_sha` | = record `audit_execution_sha` |
  | ancestry | governance → corrective → audit → evidence → effective |
  | `certification_publication_sha` | an ancestor of `effective_from_sha`, and ≠ corrective |
  | handoff record | contains both session IDs |
  | operator decision file | AUTHORIZED, with the exact line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <ts>` |
  | publication | two phases (`effective_from_sha` ≠ HEAD) |
  | phase-1 blob | `effective_from_sha` left empty (@415-421); phase 2 sets it, and `policy_activation_sha` = `effective_from_sha` [PD2-13] |
  | `implementation_session_id` | = the orchestrator session. The handoff record lists **every implementer session ID** of the RW train, and none of them may equal the auditor, certifier or delegate session (@505-509, @545-562) [PD2-13] |

- **Auditor duties:**
  - replay at the pinned SHA; re-run the L1 gates;
  - spot-check L2/L3 with its own commands;
  - verify all four elements per FRZ row and each residual disposition against the code;
  - findings go to the orchestrator only.
- **Validators:**
  - `verify_iiac_audit_manifest.py --require-handoff-template`;
  - `verify_iiac_role_separation.py --strict-product-diff`;
  - `verify_iiac_policy_activation.py --require-in-force`.
- **Final binding:**
  - product evidence → FCS ≡ G2 tree;
  - execution and certification evidence → M3.

  Any verdict weaker than PASS on an architecture-scoped item blocks §15.

---

## M. Operator decision package (v2)

**M1. Decided; record only (T-AF2-OD-RECORD publishes them):**
- OD-AF-00 (merge, 2026-11-16);
- OD-REM-01..08 and C1–C10;
- **DP-01..DP-68**;
- OD-PDC-X-CLOSE;
- OD-ROUTE-DS3;
- OD-QUALX-ACCEPT-2026-11-07;
- IIAC-OD-PDC-X-P8-ACT-N11;
- Path B authorization of 2026-10-22, plus DP-34/35;
- OD-IC-01;
- OD-ENT-02 compensating mode;
- C10/C15 kept (DP-62);
- OD-PDC-ARCHF-01.

**Questions from v1 that are now answered by a DP:**
- OD-AF-03 → DP-13/14;
- OD-AF-04 → DP-42/62;
- OD-AF-05 → DP-62;
- OD-AF-09 → DP-34/35;
- OD-AF-13 → DP-64/67, with each ruleset stage under its own AR-45..52.

**M2. Superseded or executed (record update only):**
- OD-EVOL-11, OD-EVOL-FORMAL and OD-ENT-00 are superseded by IIAC. This resolves DS-Q items EDER-D-0167/0168/0169.
- OD-ENT-01 was executed.
- OD-ENT-02 is compensating; EDER-D-0171 → PT-S01..S06.
- OD-ENT-08 is COMPLETE.

**M3. Packet 1 (needed before W1 continues):**

| OD_ID | Question | Options | Recommended | Needed by |
|---|---|---|---|---|
| OD-AF-01 | Adopt MEP-AF2 v2 as the single program plan. This means: WBS extension (T-AF2-* plus mapped PT-*); validator amendment; supersede the R1–R6 content of `ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md` and the scope of `T-IGX-010-C`; ticket series T-IGX-043/044/045 | A adopt / B reject | **A** | W0 |
| OD-AF-02 | Execution authority: the orchestrator starts sub-sessions; worktrees; the `arch-freeze/rw-train`, `af2/*` and `af2/pin-*` branches; the orchestrator merges non-product PRs after green CI and reviewer PASS, and train PRs into the train; the development hold; merge commits only; per-wave ticket approval packets | A grant / B per-PR operator merge | **A** | W0 |
| DS-EXC | Decide the PDC-X residual package from section E. It contains the fail-closed fixes, the scope/claim items with recommended options, the dormant and out-of-scope items, and the bounded claims. | per item: recommended / alternative | **recommended per item** | SCOPE |
| OD-AF-06 | Delegate the certifying FREEZE run and the ENT bootstrap to an operator-designated session (nightly @36 says "Operator-only") | A delegate / B operator runs it | **A** | W5 / W7 |
| OD-AF-08 | Label normalization: the off-vocabulary labels become PASS with a linked limitation record only after the W7 independent row verdict; AUD-10 stays non-PASS until PT-AUD-ENFA | A only (no alternative outside the vocabulary) | **A** | W7 |
| OD-AF-16a | Pre-freeze ENT baseline promotion through the delegated path; the Operator publishes the delegation file in `iiac_evidence/ENT-AUDIT-X/operator_decisions/` | A delegate / B operator-origin `promote-baseline` | **A** | W5 |
| **OD-AF-18** [PD2-07] | Frozen-scope exclusion for FRZ-SEC-05/06: E2B, Modal and Daytona as production sandbox providers are `NOT_CLAIMED` in the contract manifest (N/A WITH EVIDENCE for that part; the architecture admission and isolation gates must PASS) | A exclude (recommended) / B build and qualify them before the freeze | **A** | SCOPE |
| **AR-AF2-01** [PD2-08] | Authorization for T-AF2-MAIN-SYNC (merge the workflow definitions to `main`), only if FIND-AF2-19 is confirmed | A authorize if confirmed / B do not | **A** | W1 |
| WP-12 open items | DP-54 (standalone ECAP executable: yes/no); WP12-Q01..Q17, of which v2.1 answers **Q02**, Q11, Q12 and Q16-A; AR/AB authorizations, requested wave by wave | as in `WP-12/OPEN_QUESTIONS.md` | — | per wave |

**OD-AF-17** [PD2-02] is asked in the pre-pin packet (W5/W6, before PT-Q11): where the post-freeze ENT baseline JSON lives under the EBH-7 identity allowlist. Options A/B are in section I; A is recommended.

**M4. Audit-dependent:**
- OD-AF-10: residuals left after the RW train;
- OD-AF-11: TYP-02..05 scope, from the measurement;
- OD-AF-12: the RB-2 activation line at G2 (written by the Operator or a recorded delegate).

**M5. Final:**
- OD-AF-14: M3 confirmation;
- **OD-ARCH-FREEZE-01** (§15 admission, inside DS-ADM);
- OD-AF-15: closure text and acceptance of the FRZ promotions;
- OD-AF-16b: post-freeze baseline promotion and the §16 handoff.

No agent signs any decision.

---

## N. Exact-SHA closure matrix (ARCH-FREEZE CLOSED iff all true)

| # | Condition | Evidence |
|---|---|---|
| N1 | Ladder records: AUDIT_BASELINE_2, G2, R, FCS, M3 (= FREEZE_CERTIFICATION = TEST_EXECUTION), AUDIT_EXECUTION, EVIDENCE_PUBLICATION, CERTIFICATION_PUBLICATION, the RB-2 phases, the pre- and post-freeze ENT baselines, FORMAL_CLOSURE_PUBLICATION | ladder + validator |
| N2 | Every ladder SHA is an ancestor of `development`. `git diff G2 R` is docs only. The product tree at FCS and M3 equals G2. After M3, only the EBH-7 post-pin allowlist applies, and `executable_identity_violations(M3, HEAD)` is empty at every later commit. The baseline JSON is allowed only through OD-AF-17 A. [PD2-02] | CHANGED-SINCE + EBH-7 identity gate |
| N3 | Predecessors CLOSED with RM@499 independence: SKILL-X, TOOL-LIFE-X, DIST-SCALE-X (scoped envelope), **EBH-7 (independent B6 at M3)**, **ROADMAP-REPLAY-X (counter 0, independent)**, DUP-X, EVOL-X, ENT-AUDIT-X, MCP-SERVER-X, PDC-X | roadmap + audit records |
| N4 | FRZ architecture-scoped OPEN = 0 and BLOCKED = 0; all TEN rows PASS; only Path B rows OPEN; counters 225/26 | checklist + validator |
| N5 | VBN §2.1–2.7 plus the SQLite annex accepted independently at M3 (7 per-artefact verdicts) | dossier |
| N6 | ECAP §2.C PASS at M3; FREEZE exit 0; BLOCKER = 0, unclassified = 0, expired debt = 0 | §14 report |
| N7 | EDER frozen: PARTIAL = 0, DEFERRED disposed, every N/A evidenced; residual qualification accepted | EDER + auditor rows |
| N8 | Manifests frozen and drift-gated | manifests |
| N9 | IIAC IN_FORCE bound to G2 (IIAC-GOV-1, PASS); validators exit 0; `_CLOSURE` lint clean | activation record |
| N10 | Required checks LIVE (PT-AUD-ENFA PASS); negative trial blocked | ruleset GET + runs |
| N11 | Post-freeze policy active and verified | PFP-VERIFY |
| N12 | OD-ARCH-FREEZE-01, OD-AF-15 and OD-AF-16b recorded | OD register |
| N13 | Post-publication consistency check at FORMAL_CLOSURE_PUBLICATION | terminal record |

**ECAP §15 → N map** (carried in the T-B7-FRZ-FRZ dossier and confirmed in T-AF2-W8-INDEP-CONFIRM):

| §15 item | N rows |
|---|---|
| 1 stages closed | N3 |
| 2 FRZ | N4 |
| 3 vendor hardcoding | N5 |
| 4 debt | N7 |
| 5 extensibility (RPL-H) | N5, N4 (RPL) |
| 6 **no unapproved shadow authority** | N4 (OWN-04), plus PT-Q05 DUP-X |
| 7 §2.C | N6 |
| 8 **ROADMAP-REPLAY-X** | N3 |
| 9 evidence continuity | N1, N2 |
| 10 EVOL controls | N3, N11 |
| 11 production and distributed semantics for the claimed scope | N3 (DIST-SCALE-X), N4 (PRD/REL) |
| 12 **scenario readiness remains a post-freeze gate** | closure text + §16 handoff |

**Invalidation:**
- **What voids M3:** any change after M3 under `intergrax/`, `agents/`, `applications/`, frozen `tests/` gates, `scripts/{governance,enterprise_audit,ci,maintenance,gates}`, `.github/workflows`, `pyproject.toml`, `uv.lock`, the IIAC doc or scripts, or the 6 material governance paths.
- **Recovery:** a product change goes back to W3/W4 (new RB). A test or gate change goes back to W6/W7 with a CHANGED-SINCE delta audit.

---

## O. STOP and recovery

**STOP** halts the affected stream. The orchestrator records `.tmp/session/arch-freeze/stop/<id>.md`, updates the dashboard and asks the Operator one consolidated question. Causes:
1. a BLOCKER in frozen scope left unresolved after its task;
2. a tenant or security residual with no allowed disposition;
3. VBN acceptance not credible;
4. an FRZ inconsistency (validator FAIL, counters ≠ 225/26);
5. a wrong exact SHA (ladder, ancestry or diff check fails);
6. an auditor independence breach;
7. IIAC not IN_FORCE on `development`;
8. ECAP §2.C not PASS;
9. required CI not green (beyond the exact expected-red set, which is only allowed on the train);
10. post-freeze enforcement not active;
11. a missing mandatory decision or authorization (DP, AR, packet);
12. a material change after M3 or a breach of the development hold;
13. any request to run `--sync-wbs`/`--sync-matrix`, add skip/xfail, or widen an allowlist without OD;
14. a squash-merged pinning PR, or a run whose `head_sha` is not the pin;
15. a Campaign B finding after SCOPE that needs a product change. This leads to RB-3 or a second wave by Operator decision (WP12-Q03).

**Re-entry:**
1. record the root cause;
2. fix on a ticket branch with a targeted test and a negative test;
3. run an independent delta re-audit (CHANGED-SINCE from the last audited SHA);
4. redo the affected ladder rungs (new G2/RB if the product changed; new M3 if tests or gates changed);
5. record the Operator's decision.

**Corrective loop:** `FINDING → ROOT CAUSE → FIX → TARGETED TEST → REAUDIT → ACCEPTANCE`. After 3 loops on one finding, escalate.

**PA-1 (process check, W2).** A read-only check that the PDC/MCP process findings (F-P8-01, F-MCP-P5/P6-GOV, P8-PROC-01, K-01..K-07) do not recur in the ARCH-FREEZE procedures. It produces a short report and feeds REV-MASTER.

---

## P. Handoff: orchestrator operating instruction (this session)

### P.1 Before W0 continues

1. The PA-0 delta re-audit of v2 runs in the same independent session, limited to the changed sections.
2. The Operator answers packet 1 (M3).
3. The orchestrator builds the dashboard and the full step table (appendix Z) and publishes it as an Artifact.

### P.2 Orchestrator loop (per wave)

```
1. Re-check the development hold; fetch origin; refresh the dashboard.
2. For the wave: list READY nodes (all hard prerequisites DONE, decision/AR present).
3. Draft tickets + PLAN_*.md + digests for the wave; send ONE approval packet to the Operator.
4. Compute file-scope intersections; launch ≤ cap sub-sessions on disjoint scopes
   (start_session, fresh context, brief = node row + ticket + scope + tests + STOP rules + common brief).
5. On each session report: verify branch/PR/head SHA/CI yourself; reviewer sub-agent on the diff;
   merge only per K rules; update registers in an orchestrator docs PR; update dashboard rows.
6. Audit/certifier/delegate nodes: prepare scope file + evidence index + SHAs; propose the session to the
   Operator (own card); never brief it with implementation context.
7. Run the wave exit gate; on failure apply section O.
NEVER: sign decisions; self-PASS FRZ; --sync-wbs/--sync-matrix; skip/xfail to hide reds; GitHub settings
outside an AR; product changes on development outside the train merge; parallel pytest; artifacts
outside .tmp/; AI attribution; start W7 before M3; start PT-Q03 without a PASS PT-Q02.
```

### P.3 Task contract

Every node row on the dashboard and in the WBS carries:

`TASK_ID | BATCH_ID | REQUIREMENT_IDS | FINDING_IDS | CANONICAL_OWNER | DEPENDENCIES | FILE_SCOPE | IMPLEMENTATION_ACTION | TEST_CONTRACT | NEGATIVE_TEST | EVIDENCE_OUTPUT | AUDITOR | CERTIFICATION_GATE | OPERATOR_GATE | ACCEPTANCE_CRITERIA | RISK_CLASS | PARALLELISM | STOP_RULE | DONE_DEFINITION`

Sources: WP-12 `TICKETS_DRAFT.md` for the PT-* nodes, and this file (sections E, G–I) for the T-AF2-* nodes.

---

## Q. Verdict

**`PLAN_READY_FOR_OPERATOR_REVIEW`**. PA-0 independent plan audit: **PLAN_AUDIT_PASS** (`.tmp/session/arch-freeze/pa0/PLAN_DELTA_AUDIT_V21_PD3.md`), after three rounds (PA0-01..15, PD2-01..16, PD3-01..06). PD4-01/02 were applied in place. This plan audit does not replace REV-MASTER or the Operator's packet-1 answers. ARCH-FREEZE itself remains NOT ADMISSIBLE until section N holds.

---

## Appendix Z. Authoritative new task list (T-AF2-*) and WBS mapping of WP-12 nodes

| TASK_ID | Wave | Depends on | Scope / output | Ticket | Kind |
|---|---|---|---|---|---|
| T-AF2-BASELINE-PIN | W0 | OD-AF-01 | `AUDIT_BASELINE_2` + re-verification of section B | — (orchestrator) | RUN |
| T-AF2-OD-RECORD | W0 | OD-AF-01 | OD register table with OD-REM, C1–C10, DP-01..68, OD-AF-*, OD-PDC (labelled English rendering) | T-IGX-043-A | DOC |
| T-AF2-SSOT-RECONCILE-A [PD3-01] | W0 | BASELINE-PIN | WBS (54-task requalification, counts), SHA ladder, pre-admission report, audit matrix, OD register pointer, Wave-1 header, PDC handoff note, roadmap gate notes "SATISFIED"; **excludes** the FRZ matrix and gap register | T-IGX-043-A | DOC |
| T-AF2-VALIDATOR-AMEND | W0 | SSOT-RECONCILE-A | `verify_arch_freeze_plan_traceability.py`: computed counters (matrix cross-check gated by the SSOT-RECONCILE-B header flag [PD4-01]), `PT-`/`T-AF2-` IDs, `EXPECTED_QUAL_RUNS` computed, hard rules kept | T-IGX-043-B | TEST |
| T-AF2-SSOT-RECONCILE-B [PD3-01] | W2 | PT-D05 (#606 merged), SSOT-RECONCILE-A | FRZ acceptance matrix status reconciliation (checklist 108+8/109) and the gap register (FIND-AF2-*), on top of the #606 hunks | T-IGX-043-A2 | DOC |
| T-AF2-VALIDATOR-INDEP-REVIEW | W2 | VALIDATOR-AMEND | independent reviewer session over the validator diff | — | AUD |
| T-AF2-CI-HEAD-RUN | W0 | BASELINE-PIN | dispatch Regression gate + ENT FAST at a pinned ref; verify FIND-AF2-19 | — | RUN |
| T-AF2-EXPECTED-RED-PROBE | W0 | CI-HEAD-RUN | probe PR (closed) → `expected_red_A01.json` | — | RUN |
| T-AF2-DS-EXC-PACKAGE | W0 | BASELINE-PIN | `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md` (section E) | T-IGX-043-C | DOC |
| T-AF2-FRZ-MATRIX | W2 | DS-4, PT-FRZ-RR | 225 rows × 17 fields; 35 ticket-ready CODE_FIX plans | T-IGX-043-D | DOC |
| T-AF2-CLOSED-WORLD | W2 | BASELINE-PIN | B1 engine re-run, ownership/shadow/composition scan | T-IGX-043-E | RUN |
| T-AF2-BND-CLOSURE | W2 | CLOSED-WORLD | transitive import closure for PT-WA | — | RUN |
| T-AF2-TYP-MEASURE | W2 | CLOSED-WORLD | Any/object/reflection/string-dispatch counts over frozen surfaces | — | RUN |
| T-AF2-RM499-ENTRY | W2 | SSOT-RECONCILE-A | each RM@499 condition → evidence or node; TRACE-X rows @401/@410; post-freeze PLANNED rows note | T-IGX-043-F | DOC |
| T-AF2-MANIFESTS | W2 draft / W5 final | CLOSED-WORLD, DP-40 | contract, layer, ownership and composition manifests, generated from the **G2 tree**, stored under `docs/project/maintainers/qualification/arch_freeze_program/manifests/` (non-product, post-pin allowlisted; never under `intergrax/`); drift gate in `scripts/maintenance` reads them there; NOT_COMPOSED/NOT_CLAIMED marks; frozen-scope exclusions [PD3-04] | T-IGX-043-G (draft), T-IGX-045-B (final) | DOC + TEST |
| T-AF2-VBN-DOSSIER | W2 / W5 | CLOSED-WORLD | VBN §2.1–2.6 + SQLite annex (= T-B7-VBN-HEAD-DOSSIER) | T-IGX-043-H, T-IGX-045-C | DOC |
| T-AF2-ECAP-REVISION | W2 / W5 | MANIFESTS | ECAP §16 revision for new contracts and regression classes; FRZ recount | T-IGX-043-I, T-IGX-045-D | DOC |
| T-AF2-RED-CLASSES | W2 | CI-HEAD-RUN | known-red register FIX/ENV/OUT + `_CLOSURE` lint tool + F-W3-001. Inputs from the W0 baseline dispatch (run 43858238803): gate-tests (25 modules fail collection, e.g. psycopg not installed), gate-governance-tier (7 × `pre_model governance identity unavailable`), gate-governance-wiring (CI secret env missing), diag-final-external-proof (otel collector), durable-diagnostics-external-proof (MongoDB not a ConditionalDocumentStore); the wrong `unit-tests.yml` header comment | T-IGX-043-J | TEST |
| T-AF2-RW-PLAN | W2 | FRZ-MATRIX, DS-EXC, BND-CLOSURE | train overlap matrix, order, negative test per CODE_FIX, regression scope | — (orchestrator) | DOC |
| T-AF2-RW-PLAN-AUDIT (REV-MASTER RM-03) | W2 | T-AF2-RW-PLAN, T-AF2-FRZ-MATRIX | independent session: recompute WP-12 phase check 3 with the strict criterion over the RW-PLAN and FRZ-MATRIX output; red-first rule (each negative test fails at a pre-fix SHA; "a gate already red" does not count); per-ticket regression scope mapped to each of the 18 CODE_FIX findings; owners for the 6 row-level audit-flag conflicts | — | AUD |
| T-AF2-DAG-REGEN [PD2-05] | W0 | OD-AF-01 | apply the section 0 edge amendments to `WP-12/plan_data.py`, re-run `build_plan.py`, verify acyclicity and ENTRY reachability | — (orchestrator; `.tmp` only) | RUN |
| T-AF2-PLAN-PUBLISH [PD2-06] | W0 | DAG-REGEN, OD-RECORD | publish (docs-only) the WP-12 `DAG.md`, node table, `TICKETS_DRAFT.md`, AR/AB list and a copy of this plan, plus `SOURCES.sha256` (a digest of every source file, this plan included), under `docs/project/maintainers/qualification/arch_freeze_program/`. The plan cites only the manifest path and the publishing commit SHA, never its own digest [PD3-03] | T-IGX-043-A | DOC |
| T-AF2-RW-TENANT | W3 | SCOPE, PT-WA; serialized with PT-FC1/PT-WF on `harness_host_runtime.py` | section G row; EX-L-09 (G-12), EX-D4-04, EX-D5-05, EX-D8-03/04, EX-L-02, F-D2-11, R-P8-01 | T-IGX-044-A | PROD |
| T-AF2-RW-SUPPLY | W3 | SCOPE, PT-WA; serialized with PT-FC2/PT-WB on `applications/_shared/*_wiring.py` | section G row; EX-D1-01/02/04, D1-R017, EX-D6-01/03, A-01, A-12, F-D6-10, D6-R007/R010, F-D6-02/04, EX-D8-01 | T-IGX-044-B | PROD |
| T-AF2-RW-TRUST | W3 | SCOPE, PT-WA | `llm/messages.py`, `graph_assembly.py`, `context/*`; EX-D2-01/02/05/07, EX-L-05, D2-R004 | T-IGX-044-C | PROD |
| T-AF2-RW-RUNTIME | W3 | SCOPE, PT-WA | `intergrax/llm_adapters/`; EX-L-08, EX-D3-01/03, EX-L-07, D3-R005/R018/R019, A4-018 | T-IGX-044-D | PROD |
| T-AF2-RW-RECOVERY | W3 | SCOPE, PT-WA; serialized with PT-FC2 on queue wiring | background/queueing/scheduler; EX-L-06, EX-D7-01, EX-D4-01, EX-L-03, EX-D1-07, D7-R015, G-20 | T-IGX-044-E | PROD |
| T-AF2-RW-GATES | W3 | SCOPE, PT-WA | `scripts/gates`, scenario contract, event payloads, eval digests, NOT_COMPOSED gates; EX-D5-03, EX-L-01/11/13, EX-D3-04, D1-R015, D5-R002, D6-R015 | T-IGX-044-F | PROD + NONPRODUCT |
| T-AF2-RW-TYP | W3 | SCOPE, PT-WA, T-AF2-TYP-MEASURE, OD-AF-11 | frozen contract surfaces; DP-36 | T-IGX-044-G | PROD |
| T-AF2-DEPS-BATCH | W3 | SCOPE | held dependency-bot / `uv.lock` batch; A4-005 dependency-ownership gate | T-IGX-044-H | NONPRODUCT |
| T-AF2-POST-BASELINE [PD2-04] | W8 | PT-Q13, **T-B7-COORD-ARCH-FRZ (ENTRY)** [PD3-06], OD-AF-16b, OD-AF-17 | FREEZE evidence bundle at M3, delegated ENT chain, baseline SHA in the ladder, file location per OD-AF-17 | — (ticket-less docs under `qualification/`; the JSON only under OD-AF-17 A) | MAT/DOC |
| T-AF2-RW-MERGE | W4 | PT-Q03 | train → `development` (merge commit) = FCS | — | MERGE |
| T-AF2-ENT-BASELINE-PREFREEZE | W5 | PT-S06, T-AF2-RED-CLASSES | delegated ENT baseline chain (section I) | T-IGX-045-A | MAT/DOC |
| T-AF2-MAIN-SYNC (conditional) [PD2-08] — **NOT REQUIRED: FIND-AF2-19 false (W0 probe)** | **W1**, before PT-S01 | FIND-AF2-19 confirmed by the W0 probe; AR-AF2-01 | `main` sync of the workflow definitions | T-IGX-043-K | NONPRODUCT |
| T-AF2-REQUAL-HIST-PASS | W7 | T-B7-ECAP-2C; → COORD | independent row verdicts at M3 for EXE/GOV/TRC/STA/REC/CFG/CTL/HRN/CMP/TEN, **EVOL-01..10, FRZ-11 and AUD-01..10** (OD-AF-08 normalization; AUD-10 only after PT-AUD-ENFA) [PD2-01, PD2-14] | — (ticket-less) | AUD |
| T-AF2-W8-INDEP-CONFIRM | W8 | T-AF2-POST-BASELINE, T-B7-PFP-VERIFY | row verdicts for FRZ-FRZ-01/02/09/10/13 + §15 12-item confirmation | — | AUD |

**WBS mapping of the WP-12 nodes:**

| WP-12 node | WBS task(s) |
|---|---|
| PT-VBN | T-B7-VBN-INDEPENDENT-ACCEPT |
| PT-Q13 | T-B7-L3-FREEZE (+ FULL/FREEZE) |
| PT-Q07 | T-B7-TEN-REVAL (part) |
| PT-Q12 | T-B7-CMP-REVAL (part) |
| PT-D10 | T-PATHB-EXCL (+ checklist edits) |
| PT-S01..S06 | follow-on of T-B6-OD-ENT-02 (live enforcement) |
| PT-D02 / PT-R01..R03 | T-B2-DEBT requalification |
| PT-WA / PT-G01 | T-B2-BND03-RECONCILE |
| DS-ADM | T-B7-FRZ-FRZ + T-B8-ROADMAP admission steps |

All other PT-* nodes are added to the WBS as their own rows under batch `EQOA-W<n>`.
