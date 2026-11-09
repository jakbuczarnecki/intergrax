# IIAC independent audit — ENT-AUDIT-X (R-AUD / W2-02 session D, round 3)

Role: `independent-quality-auditor` | Policy: `IIAC-2026-11-05.1` (IN_FORCE) | Profile: `iiac-v1`  
Recovery plan: `ent-audit-x_recovery_closure` (2026-11-09) | Tickets: `T-IGX-024-A` (W0/W1), `T-IGX-024-B` (R-FRESH → R-PUB → R-AUD)  
Assurance level: `process_enforced` (sub-agent under orchestrator; session id below).  
Audit date (UTC): 2026-11-09. Worktree: `D:\Projekty\_wt\intergrax\ent-w2-audit3` @ `ee5470f543d0c1494a7dd5febc41e5ff968ade72`, `core.autocrlf=false`.  
Historical inputs (read-only): round 1 `.tmp/session/ent-w2-audit/` (CONDITIONAL @ `15347255`), round 2 `.tmp/session/ent-w2-audit2/` (INTERRUPTED, no verdict).

## 0. Verdict

**PASS_SCOPED.**

At `audited_implementation_sha` = `test_execution_sha` = `b5c2481d79f955fa5f483a9288aee8de4581f953` (R-FRESH TES′), with `evidence_publication_sha` = `63ff80819da5a7324257ea598e342810c1838199` and audit replay at `audit_execution_sha` = `ee5470f543d0c1494a7dd5febc41e5ff968ade72` (merge PR #421), the implementation reproduces under IIAC:

- Published B5 tree `qualification_at_b5c2481d…`: `verify_b5_publication_tree` → **0 errors** (Mode A).
- Bundled `qual_wave_report.json`: **PASS**; certifying FULL **85/0/85**; FREEZE bootstrap-candidate **85/0/85**; nightly stage cites run **42593448899** success @ TES′.
- `git diff b5c2481d..ee5470f5` on `intergrax/`, `agents/`, `applications/`: **empty** (scope lock).
- Loop-1 HIGH findings **F-01**, **F-02** and corrective MEDIUM **F-09..F-11** remain **FIXED_VERIFIED** (strict B5 on LF bytes; `promote-baseline` repo_root; preflight manifest bindings).
- No new HIGH or BLOCKER findings at TES′.

The scope is **not** unconditional PASS: findings **F-03..F-08** and **F-12..F-20** are **CARRIED** from round 1/2 (documented design/process limits, Matrix C evidence honesty, sandbox semantics, OD-ENT-01 trust, ADV stage strength, LIVE branch protection). Observational items **N-01**, **N-02**, **N-05..N-07** remain OPEN. **N-03** is reduced after R-PUB JSON sync; **N-04** is SUPERSEDED.

This report is **not** B6 publication, does **not** set `B6_ACCEPTANCE_PUBLICATION_SHA`, and does **not** close ENT-AUDIT-X. W2-04 certifier and publication under `docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/` await orchestrator approval.

**Auditor harness sub-agent id (Cursor):** `631309c1-023b-47e2-ac78-f7c104011033` (orchestrator may record canonical harness session id in the manifest).

### Finding counts (`findings.json`)

| Class | Count | Notes |
| --- | ---: | --- |
| BLOCKER | 0 | |
| HIGH | 0 | F-01, F-02 FIXED_VERIFIED |
| MEDIUM | 8 | F-03..F-08, F-12, F-13 CARRIED |
| LOW | 7 | F-14..F-18 CARRIED; N-01, N-02 OPEN |
| INFO | 5 | F-19, F-20 CARRIED; N-03 OPEN; N-04 SUPERSEDED; N-05..N-07 OPEN |

### Mandatory limitations (manifest)

1. **LIM-process:** assurance `process_enforced`; auditor is orchestrator sub-agent.
2. **LIM-carried:** F-03..F-08, F-12..F-20 as scoped acceptance limits until W3–W5 closure objects exist.
3. **LIM-OD-ENT-01:** human/delegated promote path trust is procedural (F-03); Matrix C rows C001,C002,C003,C042,C057,C065 stay NOT_QUALIFIED until W3-02.
4. **LIM-OPEN-BP-01:** `verify_ent_audit_branch_protection.py` **FAIL** on live `development` (ruleset without required ENT checks); CLOSURE uses COMPENSATING route (F-19).
5. **LIM-B6-unset:** no accepted baseline, no OD-ENT-08, no formal CLOSED headline.
6. **LIM-no-certifier:** W2-04/W2-05 not executed in this session.

## 1. Scope lock

| Pin | SHA | Ancestor of `ee5470f5` |
| --- | --- | --- |
| audited_implementation_sha / test_execution_sha | `b5c2481d79f955fa5f483a9288aee8de4581f953` | yes |
| evidence_publication_sha | `63ff80819da5a7324257ea598e342810c1838199` | yes; TES′ ancestor of EVID′ |
| audit_execution_sha | `ee5470f543d0c1494a7dd5febc41e5ff968ade72` | HEAD |
| IIAC activation | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` | yes |

Product delta `b5c2481d..ee5470f5`: **0** paths under `intergrax/`, `agents/`, `applications/`. R-PUB merged documentation/B5 only (PR #421).

## 2. Mode A — B5 evidence integrity

Tree: `docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff/qualification_at_b5c2481d79f955fa5f483a9288aee8de4581f953/`.

- `verify_b5_publication_tree(..., expected_qualification_sha=b5c2481d)` → **[]** @ `autocrlf=false` worktree.
- `test_b5_publication_verifier.py`: **12 passed** (strict ladder-pinned tree; no Linux skip on this tree).
- **F-01:** **FIXED_VERIFIED** for TES′ tree (`.gitattributes -text` on `qualification_at_*`).
- **F-08:** **CARRIED** — structural verifier does not bind semantic report fields; git commit identity of EVID′ is the practical control.

## 3. Mode B — CI and replay

| Run | ID | headSha | Conclusion | Role |
| --- | --- | --- | --- | --- |
| Nightly FULL (R-FRESH) | **42593448899** | `b5c2481d` | success | T-08 / plan §10 current-HEAD proof |
| ENT-AUDIT-X PR #421 | **42613400627** | `6f25e359` (PR tip) | success | R-PUB gate |
| ENT-AUDIT-X merge #421 push | **42614851524** | `ee5470f5` | success (operator); `gh` showed `in_progress` during audit | post-merge regression |

Local R-FRESH log: `.tmp/session/ent-r-fresh/qual_wave_report.json` — **PASS**, `pinned_sha` = `qualification_sha` = `b5c2481d`, `clean_tree: true`.

Published B5 `enterprise_audit_report.json`: `checks_executed` 85, `checks_skipped` 0, `status` PASS, `current_sha` `b5c2481d`.

## 4. Mode C — adversarial spot checks

Full round-1 probe grid was **not** re-executed (token/time); loop-1 fixes re-spotted:

- **F-02:** `test_promote_baseline_repo_root.py` **2 passed** @ audit HEAD.
- **F-09..F-11:** code path unchanged since loop-1 merge; round-2 evidence retained in `findings.json`.
- **F-03, F-06, F-07:** product paths unchanged; limitations **CARRIED** (OD-ENT-01 boolean record, sandbox allowlist, exit-0-on-skip).

## 5. Matrix C and closure honesty

- Master plan Matrix C: **75** rows = **69 QUALIFIED + 6 NOT_QUALIFIED** (C001,C002,C003,C042,C057,C065) — unchanged intent.
- **F-05**, **F-13:** **CARRIED** — wbs/overlay cells may cite historical `.tmp` paths; B5 @ TES′ is the durable execution proof for this audit.
- `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json` @ `ee5470f5` syncs TES′/EVID′ pins (**N-03** partially addressed).

## 6. OD-ENT-02 / LIVE vs COMPENSATING

- `verify_ent_audit_branch_protection.py` @ `ee5470f5`: **FAIL** (missing `ent-audit-fast`, `ent-audit-pr-gate` on live ruleset) — **F-19 CARRIED**, **OPEN-BP-01**.
- **F-10**, **F-11:** **FIXED_VERIFIED** in preflight/compensating wiring (loop 1); sufficiency verdict for COMPENSATING package is **out of scope** until W3/W5.

## 7. VBN and FRZ artefacts

- B5 `vbn_domains.v1.json`: 31 domains accounted in report (`vbn_domains_accounted`: 31). **F-15 CARRIED** (inventory-style mapping).
- **F-12 CARRIED** — FRZ coverage prefix heuristic; FRZ-AUD row promotion is W4.

## 8. Historical rounds

| Round | SHA | Outcome | Use in round 3 |
| --- | --- | --- | --- |
| 1 | `15347255` | CONDITIONAL (published historical annex) | F-01/F-02 basis; loop-1 scope |
| 2 | `f9e3a932` | INTERRUPTED | **Discarded** — placeholders; do not resume |
| 3 | `ee5470f5` | **PASS_SCOPED** (this report) | Authoritative for TES′ |

## 9. Assurance and write surface

Auditor edited **no** tracked product paths; output only under `.tmp/session/ent-w2-audit3/`. No PASS/CLOSED publication to `iiac_evidence/ENT-AUDIT-X/` in this session.

## 10. Certification handoff

Orchestrator may proceed to **W2-04** certifier when approved, carrying:

- `verdict`: PASS_SCOPED  
- `limitations`: LIM-01..LIM-06 above + verbatim F-03..F-20 summaries in `findings.json`  
- Digests: `iiac_audit_session_manifest.auditor_draft.json`  
- **Blockers for formal CLOSED (not for PASS_SCOPED B6 draft):** O-1 unsigned, `B6_ACCEPTANCE_PUBLICATION_SHA` unset, OPEN-BP-01, W5 operator review.
