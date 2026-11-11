# ENT-AUDIT-X — IIAC closure matrix (status overlay)

**Authority and scope.** This file is a **status overlay**, not a requirements document. It records the current *status* of ENT-AUDIT-X closure obligations under IIAC policy `IIAC-2026-11-05.1` (IN_FORCE, activation `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`). It supersedes only the **status columns** of the plan-time matrices (Matrix A/B/D/F of the post-implementation master plan, Annex B/F/G/G2) and the closure-JSON prose. **Requirements stay in the canonical documents** referenced below; nothing here adds, weakens or waives a requirement. Machine state: `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json`. SHAs: `ENT_AUDIT_X_SHA_LADDER.md` (append-only). Campaign plan: `agent-env/plans/pasted-content-id-00e5-integrax-expressive-bubble.md`; ticket `governance/tickets/T-IGX-024-A.md`.

**ENT-AUDIT-X formal closure: CLOSED (IIAC PASS_SCOPED).** Headline: `ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`. **OD-ENT-08 RECORDED** · FRZ-FRZ-11 **PASS — SCOPED**. `FORMAL_CLOSURE_PUBLICATION_SHA` pinned W5 phase 2. **ARCH-FREEZE not CLOSED.** Limitations: OPEN-BP-01, F-W3-001, Path B FRZ-PRD OPEN.

## 1. Snapshot (W2 closure overlay facts, 2026-11-11)

| Item | Value |
|---|---|
| Last-wins TEST_EXECUTION_SHA | `32a9633440007eb4e8f00a825bbf2e399192b346` (W2 closure: certifying FULL 85/85/0 + normal FREEZE vs accepted baseline `480550d80…`; merge PR #442 CI run `42869847981` success) |
| Last-wins EVIDENCE_PUBLICATION_SHA | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` (B5 `qualification_at_32a963344…`; W2 normal FREEZE tree pub `bc4b5f61666a8fea638064f50809fff2089b1b8b`) |
| Historical R-FRESH TES / EVIDENCE | `b5c2481d79f955fa5f483a9288aee8de4581f953` / `63ff80819da5a7324257ea598e342810c1838199` (preserved) |
| Historical loop-1 TES / EVIDENCE | `62d1c615855a3eaf569777a10e342aacc889a128` / `163ef0c03b3cbead556499ee81cb99d640665dce` (preserved) |
| IIAC round 1 audit | CONDITIONAL — historical annex under `iiac_evidence/ENT-AUDIT-X/historical/` (not B6 acceptance) |
| IIAC round 2 audit | `INTERRUPTED_NO_VERDICT` (do not resume on `f9e3a932`) |
| `development` tip at campaign start (W0) | `ba07ebaa34d04633bd60510f68c18346b1ef160d` |
| Independent technical re-audit (not formal B6) | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` (yes) |
| IIAC activation | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` (yes) |
| EVOL-X formal closure / ENT handoff / IIAC B6 | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` / `7a4f279fd8c0986431941c77283d872ea5b0c1ba` / `1dd769331506149a1ee2f411c2bc62e85e5bbf25` (all yes) |
| Product-path delta TES → tip | none in `intergrax/`, `agents/`, `applications/`; tooling only in `scripts/enterprise_audit/` (3 files) and 2 ENT tests |
| ENT CI on `development` / PRs | `enterprise-audit.yml` PR runs and the 2026-11-07 nightly: `failure` (2 tests, see D-1). Last nightly success: run `42015310879` @ `162acea3…` |
| Unrelated red tests on `development` @ `ba07ebaa…` | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py`: `test_replay_x_prod_delta_empty_since_dup_x`, `test_replay_x_closure_counters_computed`, `test_replay_x_b6_r2_reconciliation_zero` fail identically with and without the W0 edits (checked 2026-11-08 by stashing the edits). Outside ENT-AUDIT-X scope (ROADMAP-REPLAY-X); not caused or fixed by this campaign |
| Branch protection `development` | none (`/protection` = 404). Ruleset `Protect development` (id 26771061): `deletion` + `non_fast_forward` only; no required checks; not equivalent to live protection |

## 2. Closure matrix

Statuses: `QUALIFIED` · `CLOSED` · `OPEN` · `BLOCKED` · `UNVERIFIED`. `Aud` = independent IIAC audit required; `OD` = operator decision required. Plan task ids refer to the campaign plan §7.

### 2.1 Previously qualified (re-verify only)

| ID | Requirement (canonical source) | Status | Evidence | Next |
|---|---|---|---|---|
| M-01 | Runner, registry, contracts, scheduling, isolation, sandbox, redaction, tenant, fail-closed (Annex A / master plan Matrix E) | QUALIFIED | L3 @ TES | replay at TES′ (W1-05), audit (W2) · Aud |
| M-02 | 85 acceptance checks; FULL and FREEZE 85/85 | QUALIFIED | B5 @ `30085953…` | replay at TES′ · Aud |
| M-03 | Matrix C — 69 of 75 records | QUALIFIED | master plan Matrix C | spot-check (W2) · Aud |
| M-04 | VBN domains 1–31, hardcoding, silent fallback | QUALIFIED | `vbn_domains.v1.json` | replay · Aud |
| M-05 | B5 tree integrity (`verify_b5_publication_tree`) | QUALIFIED | W8 findings 001/002/003/006 FIXED_WITH_EVIDENCE | re-run on TES′ tree |
| M-06 | EVOL-X predecessor formally closed | CLOSED | `be77b3a5…`, `7a4f279fd…`, D10 attestation PASS (process_enforced); FRZ-EVOL-01..10 promoted | OD-ENT-00 · OD |
| M-07 | IIAC IN_FORCE | CLOSED | activation record | re-check at each publication |

### 2.2 Implementation and tooling defects (the only code work)

| ID | Defect | Status | Lane |
|---|---|---|---|
| D-1 | `master_plan_ssot_lib.py` rejects the current master-plan disposition; two tests fail on every PR and the nightly | OPEN | A |
| D-2 | `baseline.py` promotion accepts only a human operator record; no IIAC/delegated path (GAP-RC-OD-TRUST; C001, C002, C003, C042, C057, C065) | OPEN | B |
| D-3 | `ent_audit_x_formal_closure_preflight.py`: reads a non-existent EVOL key; manifest path differs from the EVOL precedent; fails open on `*unset*` ladder cells; no Gate A/B split; guard knows only the old CLOSED string | OPEN | C |
| D-5 | OD-ENT-02 verifiers: compensating verifier checks marker strings only; live verifier ignores rulesets | OPEN | C |
| D-6 | Status JSON lacks `od_ent_02_mode` and IIAC fields; stale ladder row `EVOL B6_ACCEPTANCE_PUBLICATION_SHA = PENDING` (the preflight currently detects an unset B6 pin only through this row) | OPEN | C |
| D-7 | Dossier gate (`mechanical_readiness_check.py`) needs five substrings in `ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md` | UNBLOCKED, constraint | W1-06 |

### 2.3 Missing evidence, decisions and publications (no code)

| ID | Item | Status | Gate |
|---|---|---|---|
| E-1 | IIAC independent audit: session manifest, handoff record, report, findings, pass record | **SATISFIED** (W3 pub `fd5728955…` @ `dc778d12…`; prior dossier `c8e4a91f…` adopted) | W4 certifier |
| E-2 | Accepted-baseline manifest (OD-ENT-01) | **SATISFIED** | `ENTERPRISE_AUDIT_BASELINE.json` @ W3-02; independent replay W4 |
| E-3 | Certifying FREEZE vs the accepted baseline | **SATISFIED** | `freeze_normal_at_32a963344…` + W3 independent audit Mode A/B @ TES′ |
| E-4 | FRZ-AUD-01..10 per-row evidence and promotion | **SATISFIED** (W4-03 promotion @ checklist; `frz_aud_status=PASS_SCOPED`) | W5 · OD |
| E-5 | OD-ENT-02 compensating-control package refreshed and independently accepted | **SATISFIED (SCOPED)** | Certifier `SUFFICIENT_COMPENSATING_SCOPED` @ W4 pub `5f178174e…`; OPEN-BP-01 carried |
| E-6 | OD-ENT-00 (accept EVOL formal status) | RECORDED | W3-01 done; BLK-ROADMAP-ENT-DEP satisfied at record |
| E-7 | OD-ENT-08 (formal closure authorisation) | **RECORDED** @ `2026-11-11T07:20:00Z` | W5 · OD |
| P-1 | `B6_ACCEPTANCE_PUBLICATION_SHA` | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` (phase-1 evidence publication) | W2 adds FRZ/current-SHA evidence |
| P-2 | `ACCEPTED_BASELINE_SHA` | `480550d80…` | W3-02 — ladder pinned |
| P-3 | `FORMAL_CLOSURE_PUBLICATION_SHA` | unset | W5 |
| P-4 | Roadmap §5 headline, FRZ-FRZ-11, WBS rows, terminal dossier | **PUBLISHED** (W5-01); pin W5-02 | W6 handoff |

### 2.4 Matrix C (75 records)

**75/75 `QUALIFIED`.** Six operator-trust rows **C001, C002, C003, C042, C057, C065** closed via IIAC baseline promotion replay (39/39 unit tests) + accepted baseline `480550d80…`. GAP-RC-OD-TRUST: **CLOSED_SCOPED** @ W4-03. Source rows: master plan "Matrix C — 75 TaskID".

### 2.5 FRZ-AUD-01..10

All ten: **promoted** in `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` @ W4-03. Rows 01–09: `PASS`. Row 10: `PASS — COMPENSATING (SCOPED)`; **OPEN-BP-01** carried. Certifier publication SHA `5f178174e…`; promotion record pinned via ladder `W4_03_FRZ_PROMOTION_PUBLICATION_SHA`.

### 2.6 Operator decisions

| Decision | Status | Note |
|---|---|---|
| OD-ENT-00 | **RECORDED** | `iiac_evidence/ENT-AUDIT-X/operator_decisions/ENT_AUDIT_X_OD_ENT_00_DECISION_RECORD.md` @ signing session `ent-sign-w3-2026-11-09` |
| OD-ENT-01 | **RECORDED** | `iiac_evidence/ENT-AUDIT-X/operator_decisions/OD_ENT_01.json`; `iiac_baseline_promotion.py` promote @ TES′ B5 bundle |
| OD-ENT-02 | **RECORDED** (compensating) | `od_ent_02_mode=compensating`; auditor sufficiency verdict still required (W4-02) |
| OD-ENT-08 | **RECORDED** | `ENT_AUDIT_X_OD_ENT_08_DECISION_RECORD.md` · session `ent-sign-w5-od-08-2026-11-11` |
| OD-ENT-03 | SUPERSEDED | IIAC identifies sessions, not named auditors |
| O-1 (delegation record for the signing session) | **TRANSCRIBED** | operator `@` lines published; `DELEGATION_ENT_OD_ENT_01.md` binds OD-ENT-01 |
| OD-IC-01 | VALID, internal only | never satisfies a formal gate |

### 2.7 Open-BP-01

Live branch protection of `development` is unresolved debt (`OPEN-BP-01`, defined in `OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md` and `OD_ENT_02_THREAT_MODEL.md`). It remains open under the COMPENSATING route and is carried into the ARCH-FREEZE handoff.

## 3. Supersession map (status only, history preserved)

Documents that carry a one-line pointer to this overlay (their text is unchanged): `ENT_AUDIT_X_COMPLETE_FORMAL_CERTIFICATION_AND_CLOSURE_MASTER_PLAN.md` (Path A / named auditor), `ENT_AUDIT_X_OPERATOR_DECISION_AND_EXTERNAL_AUTHORITY_PACKET.md`, `ENT_AUDIT_X_R8_EXTERNAL_GATE_PACKET.md`, `ENT_AUDIT_X_ONE_PASS_AGENT_HANDOFF_B6.md`, `ENT_AUDIT_X_OPERATOR_RECORD_EVOL.md`, `ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md`, `ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md`, `ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md`, `../ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md`.

Superseded by this overlay (status columns only, not edited): master plan Matrix A (stale: cites TES `b8bb88c9…`), master plan footer and section 18, Annex G2 qualification column, Annex F GAP-COV rows, the GAP-RC "Closure = PARTIAL" cells that contradict "Reconcile = CLOSED".

## 4. Deliberately not edited

| Item | Reason |
|---|---|
| `ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md`, `ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md`, Annex G2, CI runbook, quickstart | parsed by the validator and ENT tests |
| `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` (L4, L91 still read `NOT_IN_FORCE`), other `MATERIAL_GOVERNANCE_PATHS` | editing flips `verify_iiac_policy_activation --require-in-force`; the activation record is authoritative |
| IIAC Amendment 01 (status text inconsistent: L3 vs L51/L121/L140) | operator's call; recorded in delegation record O-1 |
| `evol_x/*` (incl. reconciliation package draft, status JSON note) | EVOL-X is closed; history is not rewritten |
| `ARCH_FREEZE_*` documents with stale "EVOL B6 pending" text, WBS rows `T-B5-EVOL-*` / `T-B6-*` | ARCH-FREEZE territory; WBS rows are generated by `verify_arch_freeze_plan_traceability.py` and change in W6 together with their generator |
| Ladder row `EVOL B6_ACCEPTANCE_PUBLICATION_SHA = PENDING` | kept until lane C fixes the preflight, to avoid hiding an unset pin |

## 5. Update log

| Date | Step | Note |
|---|---|---|
| 2026-11-08 | W0-04 / W0-05 | overlay created; ladder rows appended (IIAC activation, EVOL pins, campaign baseline); status-vocabulary corrections in roadmap L490/L561/L2065/L2066, CAAP L152/L154, FRZ checklist L47 — no status promoted; ticket `T-IGX-024-A` |
| 2026-11-09 | W3 signing | O-1 transcription; OD-ENT-00/02 published; `od_ent_02_mode=compensating`; W3-02 baseline promotion blocked on validator; W5 OD-ENT-08 paused — session `ent-sign-w3-2026-11-09` |
| 2026-11-09 | W3-02 | OD-ENT-01 delegated record + `iiac_baseline_promotion` ACCEPTED_BASELINE_SHA `480550d80…` |
| 2026-11-10 | W1-01 | JSON `accepted_baseline_sha` aligned to ladder/on-disk baseline; overlay header §2.3 E-3/P-1 pins; roadmap §5 factual pin line (no CLOSED) — T-IGX-024-B |
| 2026-11-11 | W3→W4 reconcile | SSOT overlay: W2 TES/EVID pins, E-3 SATISFIED, CI run `42869847981` @ TES; F-W3-002/003; FRZ-AUD remains OPEN — T-IGX-024-B |
| 2026-11-11 | W4 Batch D | Certifier `PASS_SCOPED` @ `5f178174e…`; W5 operator packet; **STOP** before OD-ENT-08 — T-IGX-024-B |
| 2026-11-11 | W4-03 | FRZ-AUD-01..10 promoted; Matrix C 75/75; canonical IIAC manifest; Gate A green; od_ent_02_compensating.v1.json — T-IGX-024-B |
