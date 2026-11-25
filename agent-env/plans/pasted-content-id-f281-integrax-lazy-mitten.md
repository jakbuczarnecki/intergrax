# MCP-SERVER-X P6 — Master closeout plan (recovery, ACT-N11 activation, formal scoped closure)

**Ticket:** `T-IGX-035-G` · **Stage:** MCP-SERVER-X / P6 · **Policy:** `IIAC-2026-11-05.1`
**Plan date:** 2026-11-11 · **Mode:** PLAN (nothing executed, signed, merged or activated)
**Target:** `P6 PARTIAL_EXACT_BLOCKERS → P6 CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED → P7 READY`

## Context

PR #459 (P6 product: public developer API, L3 reference server, migration docs) was merged into `development`
@ `a695c9c83` at 2026-11-11T17:08:34Z while `governance-fast` was red and before the ACT-N11 P6 rebind.
Since then `development` itself is red: `verify_iiac_policy_activation --require-in-force` fails because three
ACT-N11 product paths changed after the active audit pin `8fff85e3b`. Recovery PR #460 (`mcp-p6-gov-closeout`
@ `4015878a2`) publishes G6 + IIIC evidence and a certifier preflight, but cannot turn green on its own: the
activation record still points at the P4 pin, and the operator OD drafted on the branch would be **rejected by
the gate** even if signed. This plan sets out one canonical, precedent-backed path (the P4 PR #455 pattern) to
make the record valid, get CI green without bypassing anything, close P6 (scoped) and hand off to P7.

---

## A. CURRENT STATE AUDIT

### A.1 Confirmed facts (verified from git / gh / verifiers on 2026-11-11)

| Fact | Value | Source |
|---|---|---|
| `origin/development` HEAD | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` (merge #459, parents `e48106740` + `9b8c19d75`) | `git rev-parse`, `git rev-list --parents` |
| `development` CI @ `a695c9c83` | Regression gate **failure** (push run) | `gh run list --branch development` |
| PR #459 | MERGED 17:08:34Z, head `9b8c19d75`, merged by repo owner | `gh pr view 459` |
| Tree `9b8c19d75` vs `a695c9c83` | identical (empty diff) | `git diff --stat` |
| `5ff8606b0..9b8c19d75` | docs + ticket only (no product path) | `git diff --name-only` |
| PR #460 | OPEN, head `4015878a2`, base `development`, MERGEABLE / UNSTABLE | `gh pr view 460` |
| PR #460 checks | ci-smoke ✅, ent-audit-fast ✅, ent-audit-pr-gate ✅, evol-x-b4-preflight ✅, governance-policy-pin ✅ (advisory, `continue-on-error`), **governance-fast ❌**; 11 jobs SKIPPED by design | statusCheckRollup |
| governance-fast failure | exactly the ACT-N11 product-path errors (3 paths) through `test_evol_x_real_repo_fresh_clone.py`, `test_iiac_activation_gate.py:256`, `test_iiac_negative_guards.py:281` | job log 130689726218 |
| Recovery diff `a695c9c83..4015878a2` | 28 files, all under `docs/…/qualification/` + ticket; **0 product paths**; 0 `MATERIAL_GOVERNANCE_PATHS` | `git diff --name-only`, `product_paths_changed_between` |
| Anchors | `e4810674`, `5ff8606b`, `a695c9c8`, `8fff85e3` ⊂ development; `2d3eceb6`, `276c6430`, `c4de894a`, `4015878a` branch-only | `merge-base --is-ancestor` |
| ACT-N11 delta `8fff85e3b..a695c9c83` | `applications/research_application/mcp/server.py`, `intergrax/applications/_shared/mcp_application_server.py`, `intergrax/applications/_shared/mcp_developer_api.py` | `product_paths_changed_between` |
| Cumulative inventory `11ac8ad85..a695c9c83` | **17 paths** = P4 16 + 1 new (`mcp_developer_api.py`); matches `audit_scope.json` exactly | recomputed |
| Product delta after `2d3eceb6c` / after `a695c9c83` | `[]` / `[]` | recomputed |
| `verify_iiac_audit_manifest.py <P6 manifest> --require-handoff-template` | PASS | run @ `4015878a2` |
| `verify_iiac_role_separation.py <P6 manifest> --strict-product-diff` | PASS | run @ `4015878a2` |
| `verify_iiac_policy_activation --require-in-force` | FAIL (3 × ACT-N11 product path) | run @ `4015878a2` |
| Active record | `IN_FORCE`, `audit_execution_sha=evidence_publication_sha=8fff85e3b`, `effective_from_sha=ce4b9f5ce`, OD `IIAC-OD-MCP-ACT-N11-2026-11-11` | `IIAC_POLICY_ACTIVATION_RECORD.json` |
| `development` ruleset | rules: `deletion`, `non_fast_forward` only — **no required status checks**, no bypass actors | `gh api …/rulesets/26771061` |
| Ticket lifecycle | `quality_gate`; checkpoint says closed BLOCKED | `T-IGX-035-G.md` |
| Working tree | clean except untracked `agent-env/plans/pasted-content-id-6543-…md` (must NOT be committed) | `git status` |

### A.2 Authority precedence (as documented; nothing new invented)

1. **Mechanical gate code** — `scripts/governance/iiac_activation_gate.py` (+ `verify_iiac_*`) is the fail-closed authority for IN_FORCE.
2. **IIAC protocol** — `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` §Policy activation (items 1–5: PASS manifest, OD file `IIAC_OD_ACTIVATION_*.md` with exact fields/line, two-phase SHA publication, `--require-in-force` exit 0) and `IIAC_CERTIFICATION_AGENT_PUBLICATION_CONTRACT.md` (certifier activates only with exit 0 + PASS manifest + authorized OD; EBH-7 closure-commit rule).
3. **IIAC Amendment 01** — the operator may sign personally **or** commission a new dedicated non-implementing signing session (R-1..R-6; S-2 plain form only if the commission names it).
4. **Enterprise authorities** — roadmap §MCP-SERVER-X rows (494, 2071), freeze checklist (no FRZ change in P6), ECAP.
5. **MCP SSOT** — `PLAN_T-IGX-035-A.md` (digest `5dd3c72e…`), ticket, INDEX/WBS/SHA ladder/matrix/gap register, ODs.
6. **Precedent** — P4 recovery PR #455 (evidence `8fff85e3b` → pin `816ad8613` → certifier `6b15e4398` → phase-1 `ce4b9f5ce` → phase-2 `f2ad8ea0b` → OD literal fix `485d0b06f` → merge commit `48181df7f`, green) and P5 closeout (#457 G6 closeout, #458 ODs, all four ODs signed together).

Naming: the only policy record is **`IIAC_POLICY_ACTIVATION_RECORD.json`** (owner: IIAC policy / certification-governance agent). "IIIC" in P6 file names is a stage label for the independent MCP audit; any `IIIC_POLICY_ACTIVATION_RECORD.json` reference is a naming error (no such file exists).

### A.3 Findings (this audit)

| ID | Severity | Finding | Effect |
|---|---|---|---|
| **PF-01** | BLOCKER | P6 OD `IIIC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` is not gate-compliant: filename prefix must be `IIAC_OD_ACTIVATION_` (ACT-N06); text lacks `Decision scope: IIAC policy activation` and `Audit report digest: c405c65f…` (ACT-N07); draft line `Approved IIIC policy activation … (MCP ACT-N11 P6 rebind): operator @ …` does not match `_SIGNATURE` (`^Approved IIAC policy activation IIAC-2026-11-05\.1: operator @ <ts>$`); status written in bold (P4 needed fix `485d0b06f`). | Signing the current draft would still fail activation. |
| **PF-02** | BLOCKER | No `IIAC_SESSION_HANDOFF_RECORD_*P6.md` exists; the auditor handoff is `IIIC_SESSION_HANDOFF_RECORD_…` and is rejected as `handoff_record_path` (ACT-N14). | Phase-1 needs a new IIAC activation handoff (P4 pattern). |
| **PF-03** | BLOCKER | No activation-binding manifest (`stage_id=IIAC-GOV-1`, `verdict=PASS`) for P6. The product manifest (`MCP-SERVER-X`, `PASS_SCOPED`) cannot be `session_b_manifest_path`. | Phase-1 needs `iiac_audit_session_manifest.IIAC-ACT-N11-MCP-P6-ACTIVATION.json` (P4 pattern). |
| PF-04 | MAJOR (doc) | OD/dossier/certifier say "cumulative 19 paths" / "16 + 3"; the actual inventory is **17** (2 of the 3 delta paths were already in P4). | Fix in new IIAC OD + activation handoff (erratum); auditor artifacts are not edited. |
| PF-05 | MAJOR (provenance) | G6 evidence, IIIC evidence, certifier report and pins were committed by one git identity within 138 s (18:17:34→18:19:52); certifier commit `276c64305` also edited auditor dossier/findings/G6 manifest (verified **pin-only**: PENDING→`2d3eceb6c`) plus ticket/OD/ladder. Independence is `process_enforced` attestation only. | W4 certifier = new dedicated session re-validating final bytes; operator confirms session provenance in the OD. |
| PF-06 | MINOR | G6 dossier says "~225 s" replay; log shows `97.57s`; auditor logs have no command/SHA header. | Residual note; P7 logs must carry command + SHA header. |
| PF-07 | STRUCTURAL | `development` ruleset requires no status checks → "mandatory CI" is policy-only; root cause of F-MCP-P5-GOV-001 and its recurrence F-MCP-P6-GOV-001. | Tracked under existing **OPEN-BP-01** (FRZ-AUD-10 compensating); visible in GOV-01; not changed in P6. |
| PF-08 | RISK | `development` stays red until #460 lands; any further product-path merge before then makes the `2d3eceb6c` audit pin stale. | Global STOP condition. |
| PF-09 | MAJOR (SSOT) | Ladder `P6_PUBLICATION_SHA` = product merge (conflicts with P4 convention `P4_MERGE_SHA` vs `P4_PUBLICATION_SHA`); rows for IIIC audit, activation, terminal closure are missing. | W6 ladder reconciliation. |
| PF-10 | MAJOR | `OD-MCP-P6-TICKET-01` and the P7 entry OD (precedent `OD-MCP-P5-P6-01`) do not exist as files; `OD-MCP-P6-GOV-01` is a stub. | W1 drafts them (unsigned). |
| PF-11 | MINOR | Roadmap row 2071 still shows the P5 state; row 494 says "Engine not started until OD-MCP-01..07" (stale). | 2071 updated in W6; 494 → residual for P8 roadmap close. |
| PF-12 | RISK | Squash/rebase merge of #460 would orphan `2d3eceb6c`/phase-1 SHAs from `development` ancestry. | Merge-commit method is mandatory (P4 #455 precedent). |

Historical findings preserved verbatim, never waived or back-dated: **F-MCP-P6-GOV-001** (CRITICAL — #459 merged before G6 publication, ACT-N11 rebind and green `governance-fast`), **F-MCP-P6-GOV-002** (HIGH — ticket stale at merge), **F-MCP-P5-MEM-001 / LIM-RESEARCH-STRICT-FACTORY-MEMORY** (LOW, cross-stage PROD-Q, OPEN non-blocking).

---

## B. COMPLETE P6 CLOSEOUT INVENTORY

Owner keys: **ORCH** orchestrator/publisher session · **CERT** new dedicated certification-governance session · **OP** operator (or R-6 designated signing session) · **TV** new independent terminal verifier (read-only role `independent-quality-auditor`).
Class keys: **MB** mandatory blocker · **OD** operator decision · **NR** non-blocking residual · **HF** historical finding (must stay visible) · **P7/P8** must not be pulled into P6.

| ID | Source | Current status | Required action | Owner | Dep. | Evidence | Acceptance gate | Class |
|---|---|---|---|---|---|---|---|---|
| OBL-01 | A.1 | baseline verified at plan time | re-verify at execution start | ORCH | — | W0 log | all W0 checks hold | MB |
| OBL-02 | PF-01 | OD non-compliant | publish gate-compliant unsigned `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_<date>.md`; mark IIIC_ draft SUPERSEDED | ORCH | 01 | file @ C_W1 | W2 dry-run: only operator-pending errors | MB |
| OBL-03 | Protocol §PA-3 | unsigned | sign activation OD (exact line + `Decision status: AUTHORIZED`) | OP | 02, 07 | signed OD | regex + text checks pass | OD |
| OBL-04 | PF-03 | missing | activation-binding manifest P6 | CERT | 03 | manifest @ C_P1 | `verify_iiac_audit_manifest` PASS | MB |
| OBL-05 | PF-02 | missing | IIAC activation handoff P6 (policy version, impl/auditor/certifier ids, 17-path erratum) | CERT | 03 | handoff @ C_P1 | gate ACT-N14/N16/N17 clean | MB |
| OBL-06 | Protocol §PA-4 | record @ P4 pin | phase-1 record (`effective_from_sha=""`) | CERT | 04, 05 | C_P1 | local gate: only the 4 expected phase-1 errors | MB |
| OBL-07 | Protocol §PA-4/5 | — | phase-2 (`effective_from_sha=policy_activation_sha=C_P1`) | CERT | 06 | C_P2 | `--require-in-force` exit 0 | MB |
| OBL-08 | PF-05 | attestation-only | certifier re-validation at final bytes (logs with command + SHA header) | CERT | 03 | `certifier_activation_*.log` @ C_P1 | all validators exit 0 | MB |
| OBL-09 | §8 | governance-fast red | full required-check matrix green on #460 head C_P2 | CERT/ORCH | 07 | gh checks | C.4 matrix | MB |
| OBL-10 | PF-12 | open PR | merge #460 with **merge commit** | OP | 09 | M_460 | first-parent on development | MB |
| OBL-11 | §8 | dev red | development push CI green @ M_460; fresh-clone `--require-in-force` exit 0 | ORCH | 10 | run id + log | exit 0 | MB |
| OBL-12 | F-GOV-001/002 | REQUESTED stub | full GOV-01 text (options, recommendation, OPEN-BP-01 cross-ref); sign | ORCH→OP | 01 | signed OD | option recorded | OD |
| OBL-13 | PF-10 | missing | `OD-MCP-P6-TICKET-01` draft → sign after OBL-11 | ORCH→OP | 11, 12 | signed OD | conditions E.3 met | OD |
| OBL-14 | PF-10 | missing | `OD-MCP-P6-P7-01` (P7 entry) draft → sign | ORCH→OP | 13 | signed OD | effective only after W7 PASS | OD |
| OBL-15 | ticket | `quality_gate` | → `closed` (scoped), checkpoint with all SHA roles | ORCH | 13 | ticket @ C_CL | igx_gov ticket PASS | MB |
| OBL-16 | P5 precedent | missing | `MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md` | ORCH | 13 | file | lists all P6 ODs + findings | MB |
| OBL-17 | PF-09 | stale | SHA ladder reconciliation (§D W6) | ORCH | 10 | ladder | every role distinct, ancestry checked | MB |
| OBL-18 | WBS/INDEX | "Recovery"/"In progress" | P6 → Closed (scoped); P7 → Ready (entry OD) | ORCH | 15 | docs | no CLOSED stage claim | MB |
| OBL-19 | matrix | MCP-ACC-01 "candidate SHA" | pin P6 scoped evidence (`5ff8606b0`/`9b8c19d75`), P7 exact package pending | ORCH | 15 | matrix | L2≠L3 preserved | MB |
| OBL-20 | gap register | REQ-MCP-DX-01 "addressed" | → "closed P6 (scoped) @ SHA"; residuals unchanged | ORCH | 15 | register | no residual silently closed | MB |
| OBL-21 | roadmap 2071 | P5 state | P6 CLOSED (scoped) + findings; stage IN PROGRESS; Next P7 | ORCH | 15 | roadmap diff | no FRZ / stage-close change | MB |
| OBL-22 | terminal dossier | "NOT CLOSED" | final P6 terminal dossier (SHA table, gates, findings, ODs, residuals) | ORCH | 15 | dossier | matches ladder | MB |
| OBL-23 | P7 handoff | draft | complete (§G); READY effective after W7 | ORCH | 15 | handoff | §G checklist | MB |
| OBL-24 | §9.11 | — | independent terminal verification of remote `development` @ M_CL | TV | closure merge | report | §F all PASS | MB |
| OBL-25 | — | — | publish terminal report + `P6_TERMINAL_CLOSURE_SHA`/`FINAL_REMOTE_HEAD` | TV + ORCH | 24 | M_TV | CI green, merged | MB |
| OBL-26 | F-MCP-P6-GOV-001 | recorded | keep in G6 dossier, IIIC manifest, terminal dossier, roadmap, closure record | ORCH | — | grep | present at M_TV | HF |
| OBL-27 | F-MCP-P6-GOV-002 | recorded | same | ORCH | — | grep | present | HF |
| OBL-28 | PF-07 | open | cross-ref OPEN-BP-01 in GOV-01 + terminal dossier; branch protection is an operator choice outside P6 (`scripts/dev/apply_github_branch_protection.ps1`) | ORCH | — | text | visible | NR |
| OBL-29 | PF-04/06 | doc discrepancies | erratum in IIAC OD + activation handoff + terminal dossier; auditor files untouched | ORCH/CERT | — | text | visible | NR |
| OBL-30 | LIM-RESEARCH-STRICT-FACTORY-MEMORY | OPEN | carry unchanged into P7 handoff (MemoryProvider residual = PROD-Q dependency) | ORCH | — | handoff | unchanged | NR |
| OBL-31 | PF-11 row 494 | stale | log as residual for P8 roadmap close | ORCH | — | gap/terminal dossier | listed | NR |
| OBL-32 | plan §P7 | — | MCP-ACC-01..20 exact-SHA package, L1/L2/L3, EVOL/ENT delta, clean checkout, independent P8 audit | — | P6 closed | — | not in P6 | P7/P8 |

Zero unclassified mandatory obligations: every MB/OD row has an owner, action and gate.

---

## C. RECOMMENDED GOVERNANCE RECOVERY PATH

**Chosen mechanism:** the canonical **two-phase activation inside the same evidence PR (#460)** — exactly the P4 PR #455 pattern. The evidence PR is *not* merged red. It becomes green only once the signed OD, the activation-binding manifest, the IIAC handoff and the two-phase record are on its head. No waiver, no override, no `--evidence-root` on the canonical record, no change to gate code.

### C.1 Why this is the only compliant path
- The gate checks `product_paths_changed_between(audit_execution_sha, HEAD)`. The only way to clear it without touching product code is to rebind `audit_execution_sha` to a commit that already contains all P6 product paths and has independent evidence → `2d3eceb6c` (evidence publication; product Δ after it = `[]`).
- Rebinding requires (protocol §PA 1–5): published PASS manifest + authorized `IIAC_OD_ACTIVATION_*` + two-phase SHAs + exit 0. All of these can live on #460 before merge, so `development` never receives a red merge.
- Alternatives rejected: merging #460 red (violates the mandatory gate and repeats GOV-001); operator waiver (no waiver mechanism in the protocol, and the gate cannot be waived); reverting #459 (a product-path change → new ACT-N11 delta and new audits; does not erase GOV-001).

### C.2 Phase-1 record values (all verified against gate logic)

| Field | Phase-1 value | Rule satisfied |
|---|---|---|
| `policy_version` | `IIAC-2026-11-05.1` (unchanged) | ACT-N16 |
| `status` | `IN_FORCE` | ACT-N01 |
| `effective_from_sha`, `policy_activation_sha` | `""` | ACT-N12 phase-1 blob |
| `governance_implementation_sha` | `2ae1014116d353bc7742985e62f2bccc2bfac798` (unchanged) | ancestry |
| `corrective_implementation_sha` | `5ff8606b089828b334ebac133518ba47c9f6f7ce` (P4 set this to the audited MCP product pin `27e19f1f`) | = binding manifest `audited_implementation_sha` (ACT-N09) |
| `audit_execution_sha`, `evidence_publication_sha` | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ancestry; product Δ after = [] |
| `certification_publication_sha` | `276c64305799fa4d7f1a25018cc1cccf061fbafd` (the P6 certifier publication, already bound in the product manifest) | ≠ corrective; ancestor of effective; = binding manifest |
| `session_b_verdict` | `PASS` | binding manifest verdict |
| `session_b_governance_audit_report_digest` / `audit_report_path` | `c405c65f…` / IIAC-GOV-1 re-audit dossier (unchanged; bytes identical @ `2d3eceb6c`) | ACT-N08/N11 |
| `session_b_manifest_path` / digest | `…/act_n11_mcp_p6/iiac_audit_session_manifest.IIAC-ACT-N11-MCP-P6-ACTIVATION.json` / sha256 of the committed bytes | ACT-N03/N08 |
| `audit_scope_path`, `test_matrix_path`, `findings_path` | unchanged (P4 pattern; digests copied into binding manifest and re-checked) | ACT-N13/N15 |
| `handoff_record_path` / `handoff_digest` | `…/act_n11_mcp_p6/IIAC_SESSION_HANDOFF_RECORD_ACT_N11_MCP_REAUDIT_P6.md` / sha256 | ACT-N14/N16/N17 |
| `operator_decision_path` / `_id` / `_line` / `_class` | `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_<date>.md` / `IIAC-OD-MCP-ACT-N11-P6-<date>` / the exact signed line / `OD-MCP-ACT-N11-P6` | ACT-N06/N07 |
| `notes` | "Phase 1 ACT-N11 MCP P6 rebind. Prior pin 8fff85e3b (OD IIAC-OD-MCP-ACT-N11-2026-11-11, phase-2 f2ad8ea0b) preserved in history." | lineage |

Binding manifest = copy of the P4 binding manifest structure with: implementation `T-IGX-035-G-MCP-P6-IMPLEMENTER`, auditor `b8d2e7f4-…` (IIIC auditor), certifier = new W4 CERT session id, `audited_implementation_sha=5ff8606b0`, `test_execution_sha=9b8c19d75`, `evidence/audit_execution_sha=2d3eceb6c`, `certification_publication_sha=276c64305`, the same four digests (re-computed and confirmed equal), `verdict: PASS`, limitations stating P6 PASS_SCOPED product manifest, the 17-path inventory, F-MCP-P6-GOV-001, no stage close / no FRZ. `product_session_manifest_path` → P6 product manifest.
OD text must not contain `EVOL-X|ENT-AUDIT-X|ARCH-FREEZE` within 80 characters of `CLOSED` (ACT-N18 regex).

### C.3 Expected intermediate statuses

| Point | `verify_iiac_policy_activation --require-in-force` | governance-fast |
|---|---|---|
| now / after W1 | FAIL: 3 × ACT-N11 product path (only) | red (same 3) |
| W2 dry-run (scratch, unsigned OD) | FAIL: only ACT-N06 pending, ACT-N07 no AUTHORIZED, ACT-N07 line absent/not authorized + phase-1 empty-SHA errors | n/a (not pushed) |
| C_P1 (phase-1, signed OD) | FAIL: exactly `missing activation field: effective_from_sha`, `…policy_activation_sha`, and the two "must be a 40-char lowercase git SHA" errors | not pushed alone |
| C_P2 (phase-2) | **PASS exit 0** | expected green |
| M_460, M_CL, M_TV | PASS exit 0 | green |

### C.4 CI acceptance matrix

| Check | Workflow | Required | Expected on C_P2 / M_460 / closure PR / terminal PR | Note |
|---|---|---|---|---|
| ci-smoke | Regression gate | yes | SUCCESS | |
| governance-fast | Regression gate | yes | SUCCESS (currently the only red) | the blocker |
| evol-x-b4-preflight | Regression gate | yes | SUCCESS | |
| ent-audit-fast | ENT-AUDIT-X | yes | SUCCESS | |
| ent-audit-pr-gate | ENT-AUDIT-X | yes (PR) | SUCCESS | |
| governance-policy-pin | Regression gate | advisory (`continue-on-error: true`) | SUCCESS expected; a failure is recorded, not ignored silently | |
| ci-enterprise, platform-plugin-windows-e2e, gate-*, diag-*, durable-diagnostics-*, diagnostics-full-qualification, ebh7-p03-l3-qualification, evol-x-p3-l3-qualification | Regression gate | conditional (dispatch / path) | SKIPPED by design — recorded as SKIPPED | not runnable under C10/C15 freeze without operator authorization |
| development push run | Regression gate + ENT-AUDIT | yes | SUCCESS @ M_460, M_CL, M_TV | proves the published state |

Mandatory status is policy-enforced (PF-07). A red mandatory check = STOP, never "understood and ignored".

### C.5 STOP / rollback conditions (global)
- `origin/development` gains any product-path commit before M_460 → STOP (pin stale; new IIIC delta audit needed).
- Any change to `MATERIAL_GOVERNANCE_PATHS` between `2d3eceb6c` and HEAD → STOP (ACT-N11 stale audit).
- Any edit of auditor-authored artifacts (G6/IIIC dossier, findings, scope, matrix, logs, product manifest) → STOP.
- Phase-1 local gate shows any error outside the expected 4 → STOP; fix only CERT-owned artifacts, never relax the gate.
- OD line does not fullmatch `_SIGNATURE` or status not plain `Decision status: AUTHORIZED` → STOP, back to OP.
- Any mandatory check red on C_P2 → STOP, diagnose; no merge.
- Merge method other than merge commit → do not merge.
- Rollback before merge: revert the CERT commits on the branch (record returns to the `8fff85e3b` pin); after merge: no rewrite of history — a new corrective PR only.

---

## D. MASTER IMPLEMENTATION PLAN W0–W7

Serial spine: **W0 → W1 → W2 → W3 → W4 → W5 → W6 → W7**. Safe parallelism: GOV-01 signing (W3) may run in parallel with W2; W6 SSOT drafting may start after M_460 while waiting for the TICKET-01 signature. Everything that touches the activation record, SSOT hubs or a PR head is serial. Sessions: ORCH (may be the existing recovery/publisher session, never the P6 implementer), CERT (new), TV (new). The P6 implementer `T-IGX-035-G-MCP-P6-IMPLEMENTER` and auditors `77e4a696…` / `b8d2e7f4…` take no part in W4/W7.

### W0 — Current-state verification and authority resolution · ORCH · read-only
- Input SHA: `4015878a2` (#460 head), `a695c9c83` (dev).
- Tasks: `git fetch origin`; confirm branch `mcp-p6-gov-closeout`; `origin/development` == `a695c9c83` (or only non-product descendants); `product_paths_changed_between(2d3eceb6c, origin/development)` == []; no `MATERIAL_GOVERNANCE_PATHS` diff; PR #460 head == local HEAD; record unchanged; re-run the three verifiers (expect PASS/PASS/FAIL-3); `gh pr checks 460`.
- Output: `.tmp/session/mcp-p6-closeout/w0_baseline.log`. No commit.
- STOP: any deviation from A.1.

### W1 — Evidence/manifest integrity and publication preparation · ORCH · PR #460 branch
- Read: P4 OD `IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md`, P5 OD files, this plan §E.
- Change (docs only, under `docs/project/maintainers/qualification/`):
  1. New `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_<date>.md`: the exact §E.1 text with `Decision status: PENDING` and the `________________` placeholder (unsigned).
  2. `IIIC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md`: prepend "SUPERSEDED (not gate-compliant: ACT-N06 filename, ACT-N07 scope/digest/line) → see IIAC_OD_…"; content otherwise kept.
  3. `operator_decisions/OD-MCP-P6-GOV-01.md`: full text §E.2 (unsigned). New `OD-MCP-P6-TICKET-01.md` (§E.3) and `OD-MCP-P6-P7-01.md` (§E.4), both REQUESTED.
  4. Pin-only verification note in the terminal dossier: diff `2d3eceb6c..276c64305` on auditor files is SHA pin-only (PF-05) and the 17-path erratum (PF-04).
- Not touched: activation record, auditor artifacts, product code, ladder (W6).
- Commands: `uv run python -m scripts.governance.igx_gov ticket`; the three verifiers; commit with `Ref: governance/tickets/T-IGX-035-G.md`; push (upstream already set).
- Output SHA: C_W1. Acceptance: verifiers still PASS/PASS/FAIL-3 only; CI unchanged except governance-fast still red with the same 3 errors.
- STOP: any new failing check.

### W2 — CI/ACT-N11 bootstrap dry-run · CERT · scratch worktree, never pushed
- Worktree: `git -C "$MAIN" worktree add --detach "$WT_BASE/p6-act-dryrun" C_W1` (CLAUDE.md worktree rule, outside the repo).
- Tasks: draft binding manifest, IIAC handoff, phase-1 record exactly per C.2 against the **unsigned** OD; commit locally; run `verify_iiac_policy_activation --require-in-force`.
- Acceptance: the error set equals the operator-pending set + phase-1 empty-SHA errors (C.3 row 2). This proves every non-operator binding before the operator is asked to sign.
- Output: `.tmp/session/mcp-p6-closeout/w2_dryrun.log`; then `git worktree remove` + `prune`. Nothing published. No fake signature or timestamp is ever written, even locally.
- STOP: any other error → fix the CERT drafts (or W1 OD text) and repeat.

### W3 — Operator governance authorization · OP (personally or an R-6 designated signing session)
- Preconditions: C_W1 published; W2 PASS reported.
- Sign `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_<date>.md`: set `Decision status: AUTHORIZED` (plain), replace the placeholder with the exact line, and confirm the session-provenance statement (PF-05). Sign GOV-01 (option per §E.2). The line and timestamp are written or dictated verbatim by the operator; an agent only transcribes with a provenance note (Amendment 01 §1.2). It never composes the timestamp.
- Output SHA: C_OD (signature commit on #460, `Ref:` trailer).
- STOP: line not matching the regex; a status other than `AUTHORIZED`; changed scope.

### W4 — Independent certifier and two-phase activation · CERT (new dedicated session, ≠ implementer/auditors)
- Input SHA: C_OD.
- Tasks: (a) re-run `verify_iiac_audit_manifest.py <P6 product manifest> --repo . --require-handoff-template`, `verify_iiac_role_separation.py <manifest> --strict-product-diff --repo .`, recompute product deltas and the four digests; save `certifier_activation_*.log` with a command + SHA header under `act_n11_mcp_p6/`. (b) Create the binding manifest + IIAC handoff (C.2), and run `verify_iiac_audit_manifest` on the binding manifest. (c) Phase-1 edit of `IIAC_POLICY_ACTIVATION_RECORD.json`; commit C_P1 `gov(mcp): phase-1 IIAC activation record for ACT-N11 MCP P6 rebind.`; local gate → expected 4 errors only. (d) Phase-2: set both SHAs to C_P1; commit C_P2; `verify_iiac_policy_activation --require-in-force` → **exit 0**. (e) One pytest run: `uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_evol_x_real_repo_fresh_clone.py -q` → exit 0. (f) `igx_gov ticket`; push.
- Output SHAs: C_P1 = **P6_ACTIVATION_SHA**, C_P2.
- STOP: any C.5 condition.

### W5 — Mandatory CI replay and compliant publication · ORCH + OP
- Verify the C.4 matrix on head C_P2 (`gh pr checks 460`, head OID == pushed C_P2 per memory rule "verify PR head before asking to merge").
- OP merges #460 with **merge commit** → M_460 = **P6_PUBLICATION_SHA**.
- Post-merge: development push run green @ M_460; in a fresh detached worktree at `origin/development`: `verify_iiac_policy_activation --require-in-force` exit 0; `merge-base --is-ancestor` for `2d3eceb6c`, `276c64305`, C_P1, C_P2 → M_460. Log: `.tmp/session/mcp-p6-closeout/w5_post_merge.log`.
- STOP: any red check; head mismatch; ancestry failure.

### W6 — P6 formal scoped closure and SSOT reconciliation · ORCH + OP · new closure PR
- Branch: a new branch from M_460 (e.g. `mcp-p6-closure`) — **requires explicit operator approval** (branch-safety rule); set upstream to `origin/development` before the first push (memory rule).
- OP signs `OD-MCP-P6-TICKET-01` (and `OD-MCP-P6-P7-01`, conditional wording, §E.4) after W5 is verified.
- Commit A (records only): signed ODs + `MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md`. Commit B (closure, EBH-7 rule: CLOSED claim not in the commit that first creates a pass record) = C_CL = **P6_FORMAL_CLOSURE_SHA**:
  - Ticket → `closed` (scoped); checkpoint lists every SHA role below.
  - SHA ladder: add `P6_MERGE_SHA=a695c9c83` (relabel; note the former `P6_PUBLICATION_SHA` value), `P6_PUBLICATION_SHA=M_460`, `P6_G6_AUDIT_SHA=9b8c19d75`, `P6_IIIC_AUDIT_SHA=2d3eceb6c` (ACT-N11 `audit_execution_sha`; replay tree `9b8c19d75`), `P6_EVIDENCE_PUBLICATION_SHA=2d3eceb6c`, `P6_IIIC_CERTIFICATION_PUBLICATION_SHA=276c64305`, `P6_ACTIVATION_SHA=C_P1`, `P6_ACTIVATION_PHASE2_SHA=C_P2`, `P6_FORMAL_CLOSURE_SHA=C_CL`, `P6_TERMINAL_CLOSURE_SHA=_pending W7_`, `FINAL_REMOTE_HEAD=_pending W7_`.
  - WBS (P6 Closed (scoped); P7 Ready — entry OD), INDEX, QUALIFICATION_MATRIX (MCP-ACC-01 row), GAP_REGISTER (REQ-MCP-DX-01 + residuals), P6 terminal dossier, P7 handoff (§G), roadmap row 2071 only.
- Commands: `igx_gov ticket`; the three verifiers (record unchanged → exit 0); push; C.4 matrix green; OP merges with merge commit → M_CL = **P6_TERMINAL_CLOSURE_SHA**.
- STOP: any product or activation-record change in this PR; any red check; any stage-CLOSED or FRZ text.

### W7 — Independent terminal verification and P7 handoff · TV (new) + ORCH
- TV, read-only, fresh worktree @ `origin/development` == M_CL: run every §F check; write `MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md` (verdict `P6_TERMINAL_VERIFIED` or exact failures).
- ORCH (separate commit, same PR): pin `P6_TERMINAL_CLOSURE_SHA=M_CL`, `FINAL_REMOTE_HEAD`, and flip the P7 handoff header to READY (effective per OD-MCP-P6-P7-01). Branch per operator approval (new or the W6 branch re-synced). CI green; OP merges → M_TV.
- STOP: any §F failure → P6 stays not closed in practice; open a corrective on the failing row only.

---

## E. OPERATOR DECISION PACK (drafted in W1; never signed by an agent)

Touchpoints: **T1 = W3** (E.1 + E.2) · **T2 = W6** after M_460 is verified (E.3 + E.4) · plus execution authorizations: merges of #460 / closure PR / terminal PR, and new-branch approval for W6/W7.

### E.1 `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_<signing-date>.md` (location: `docs/project/maintainers/qualification/`)
```text
# IIAC operator decision — MCP ACT-N11 product re-bind (P6 cumulative)

Decision ID: IIAC-OD-MCP-ACT-N11-P6-<signing-date>
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation (MCP P6 product delta vs `8fff85e3bfc5190929ff464b00fbe759dadcb5f8`)
Decision status: PENDING
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Product independent audit dossier digest (MCP ACT-N11 P6 re-audit): f4b339ab2695a1f162cf4085a8123ffe6c5e6c568990212111b4647548f21a66

________________

Scope: three-path P6 delta (one new path: intergrax/applications/_shared/mcp_developer_api.py) on the
P4 sixteen-path ACT-N11 inventory; cumulative 17 paths (erratum: earlier drafts said 19).
Independent audit: PASS_SCOPED (auditor b8d2e7f4-1a3c-4f9e-8d62-5c1a0b9e7f3d). Certifier preflight: PASS.
Evidence publication SHA: 2d3eceb6cf7e93dccd4d96919319a6c969b4444b.
Finding F-MCP-P6-GOV-001 preserved; this decision is not a retroactive approval of PR #459.
Session provenance (operator confirms): implementer, G6 auditor 77e4a696-…, IIIC auditor b8d2e7f4-…,
certifier c4e8a1f2-… and the activation certifier are separate sessions; none is the implementer.
Authorization applies only to this audited scope. It does not bypass governance checks, does not promote FRZ,
does not close the MCP stage and does not authorize P7.
```
At signature the operator replaces `Decision status: PENDING` → `Decision status: AUTHORIZED` and the placeholder → `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <ISO-8601>` (exact form; the timestamp is the operator's). Predecessors: C_W1 + W2 PASS.

### E.2 `OD-MCP-P6-GOV-01` (location: `mcp_server_x/operator_decisions/`)
Question: accept post-publication independent scoped acceptance (G6 `ACCEPT_SCOPED` @ `9b8c19d75`, IIIC `PASS_SCOPED`) while preserving F-MCP-P6-GOV-001/002 on record?
- **A (recommended):** `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING`. No retroactive pre-merge approval. F-MCP-P6-GOV-001 is recorded as a **recurrence** of F-MCP-P5-GOV-001: the prospective gate of OD-MCP-P5-GOV-01 was in force and did not prevent it, and the mechanical gap (no required status checks on the `development` ruleset) is cross-referenced to OPEN-BP-01. Compliant with the protocol: findings stay visible, nothing is waived, and the P5 post-publication precedent applies.
- **B:** revert the #459 product merge. Not recommended: a product-path change needs a new ACT-N11 delta and new audits, and it does not remove the finding.
Operator record: `Authorized: operator @ <ts>` / `Selected option: <A|B> — …`. Explicit non-actions as in P5. Predecessor: G6 + IIIC published (true now).

### E.3 `OD-MCP-P6-TICKET-01`
Question: authorize `T-IGX-035-G → closed (scoped)` with AC-P6-01..08 satisfied @ documented SHAs and F-MCP-P6-GOV-001/002 on record?
Conditions (must be true at signing): G6 `ACCEPT_SCOPED` published; IIAC record bound to `2d3eceb6c` with `--require-in-force` exit 0 on `origin/development` @ M_460; C.4 matrix green on M_460; E.1 + E.2 signed.
Operator record: `Authorized: operator @ <ts>` / `Ticket closed: T-IGX-035-G — scoped P6; F-MCP-P6-GOV-001/002 preserved; post-publication independent G6 + IIIC records retained`.

### E.4 `OD-MCP-P6-P7-01` (P7 entry; precedent OD-MCP-P5-P6-01)
Question: authorize planning/entry for MCP-P7-01..09 on `development` without implying stage CLOSED?
Operator record: `Authorized: operator @ <ts>` / `P7 entry: authorized — effective at the origin/development SHA where the P6 terminal verification PASS is published`. It may be signed at T2 (conditional) or deferred to after W7.

No other OD is required by the protocol. Branch protection (PF-07) is an operator settings choice, not a P6 decision.

---

## F. TERMINAL ACCEPTANCE MATRIX (mechanical; TV runs it @ M_CL)

| # | Requirement | Validator | Required |
|---|---|---|---|
| F-01 | `origin/development` == M_CL; first-parent contains M_460 | `git ls-remote`, `git log --first-parent` | equal |
| F-02 | IIAC IN_FORCE bound to P6 | `verify_iiac_policy_activation --require-in-force` | exit 0 |
| F-03 | P6 product manifest valid | `verify_iiac_audit_manifest.py … --require-handoff-template` | exit 0 |
| F-04 | role separation | `verify_iiac_role_separation.py … --strict-product-diff` | exit 0 |
| F-05 | no product Δ after the audit pin | `product_paths_changed_between(2d3eceb6c, HEAD)` | [] |
| F-06 | 17-path inventory == `audit_scope.json` | recompute `11ac8ad85..a695c9c83` | equal |
| F-07 | SHA ancestry chain | `merge-base --is-ancestor` e481→5ff8→9b8c→a695→2d3e→276c→C_P1→C_P2→M_460→C_CL→M_CL | all true |
| F-08 | record phase-1 blob @ C_P1 has empty effective, same binding keys | `git show C_P1:…RECORD.json` | match |
| F-09 | auditor artifacts unchanged since publication (except pins verified in W1) | `git diff 276c64305 HEAD -- <auditor files>` | empty |
| F-10 | CI on M_460 and M_CL push runs | `gh run list --branch development` | success |
| F-11 | ticket `closed` (scoped) + E.3 signed + all SHA roles in checkpoint | grep / igx_gov | present |
| F-12 | E.1, E.2, E.3 signed in the canonical form; no agent-composed timestamp | text checks | present |
| F-13 | F-MCP-P6-GOV-001/002 visible in G6 dossier, IIIC manifest, terminal dossier, roadmap 2071, closure record | grep | present |
| F-14 | ladder: every role in D/W6 distinct and consistent with git | manual + git | consistent |
| F-15 | no `MCP-SERVER-X` stage CLOSED, no FRZ status change, no P7 artifacts | `git diff a695c9c83 M_CL -- checklist, roadmap` + grep | none |
| F-16 | P7 handoff complete per §G | checklist | complete |
| F-17 | branch state at session end | `git branch --show-current`, `git symbolic-ref HEAD` | branch, not detached |

`P6 CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED` holds only when F-01..F-17 all PASS and the report is merged (M_TV). Then, and only then, `P7 READY`.

### Closure matrix

| OBLIGATION | CURRENT EVIDENCE | GAP | OWNER | ACTION | VALIDATOR | FINAL REQUIRED STATUS |
|---|---|---|---|---|---|---|
| G6 technical acceptance | dossier + `p6_g6_closeout/` @ `2d3eceb6c` | none (PF-06 minor) | — | preserve | F-09 | ACCEPT_SCOPED published |
| IIIC ACT-N11 audit | `act_n11_mcp_p6/` PASS_SCOPED | none | — | preserve | F-03/F-04/F-06 | PASS_SCOPED published |
| Certifier | preflight PASS @ `276c64305` | independence attestation-only (PF-05) | CERT | re-validate @ final bytes | W4 logs | PASS @ final bytes |
| Activation OD | non-compliant IIIC draft | PF-01 | ORCH→OP | E.1 | gate ACT-N06/N07 | AUTHORIZED, exact line |
| Binding manifest | missing | PF-03 | CERT | create | verify_iiac_audit_manifest | PASS |
| IIAC handoff | missing | PF-02 | CERT | create | gate ACT-N14/N17 | present, digest bound |
| Activation record | P4 pin `8fff85e3b` | rebind | CERT | phase-1/phase-2 | `--require-in-force` | exit 0, bound to `2d3eceb6c` |
| governance-fast | red | rebind | CERT | W4 | CI | green on C_P2, M_460, M_CL |
| Publication | #460 open | merge | OP | merge commit | F-01/F-07 | M_460 on development |
| GOV-01 | stub | PF-10 | ORCH→OP | E.2 | text | signed (option recorded) |
| TICKET-01 | missing | PF-10 | ORCH→OP | E.3 | text | signed |
| P7 entry OD | missing | PF-10 | ORCH→OP | E.4 | text | signed |
| Ticket | quality_gate | closure | ORCH | W6 | igx_gov, F-11 | closed (scoped) |
| SHA ladder | conflated/missing rows | PF-09 | ORCH | W6 | F-14 | all roles distinct |
| WBS / INDEX / matrix / gaps / roadmap 2071 / terminal dossier | stale | SSOT | ORCH | W6 | F-15/F-16 | reconciled |
| Historical findings | recorded | — | ORCH | keep | F-13 | visible |
| Terminal verification | none | — | TV | W7 | §F | P6_TERMINAL_VERIFIED, merged |
| P7 handoff | draft | §G | ORCH | W6/W7 | F-16 | READY (after W7) |

---

## G. P7 HANDOFF (content required in `MCP_SERVER_X_P7_HANDOFF.md` at W6/W7)

- **Entry:** OD-MCP-P6-P7-01; base = `origin/development` @ M_TV; P7 ticket scaffold (next `T-IGX-035-H`) is P7 work, not P6.
- **P6 exit artifacts:** developer guide, migration matrix, G6 dossier, IIIC P6 package, terminal dossier, terminal verification report, closure record, ladder rows (all SHA roles above).
- **Backlog:** MCP-P7-01..09 (`PLAN_T-IGX-035-A.md` §MCP-P7): L1 unit (P7-01), L2 in-process labelled "not physical transport" (P7-02), L3 physical HTTP loopback discovery/schema/invoke/deny (P7-03), announced protocol subset matrix (P7-04), failure injection (P7-05), MCP + TOOL-LIFE regression (P7-06), EVOL change-impact + ENT Mode A delta (P7-07), clean-checkout replay (P7-08), exact-SHA evidence package with `TEST_EXECUTION_SHA` (P7-09).
- **Matrix:** every MCP-ACC-01..20 needs a test ID or a justified N/A at one exact SHA; P5/P6 scoped rows (ACC-01, 02, 09–15, 17) must be requalified at the P7 SHA, not inherited; L2 ≠ L3 kept apart.
- **Regression and adversarial:** P5-07 adversarial pack + P6 batch + `test_reference_apps_mcp`, one pytest per verification step (process budget).
- **Evidence rules:** logs carry the command + SHA header (PF-06 lesson); byte-preserving copies under `iiac_evidence/MCP-SERVER-X/`.
- **Cross-stage dependencies:** LIM-RESEARCH-STRICT-FACTORY-MEMORY (PROD-Q, OPEN, non-blocking, no fabricated vendor qual); OPEN-BP-01 (branch protection); roadmap row 494 staleness (P8); C10/C15 freeze (no Docker, no new vendors, diagnostic pytest needs operator authorization).
- **Governance:** any P7 product-path change triggers ACT-N11 → independent IIIC delta audit + two-phase rebind **before** merge (prospective gate of GOV-01); no merge with red mandatory checks.
- **P8 prerequisites:** frozen `IMPLEMENTATION_SHA` (P8-01), independent stage audit, IIIC certification, B6, roadmap stage row, ARCH-FREEZE handoff. No P7 claim of stage CLOSED / FRZ / IIIC terminal.

---

## Verification (how execution proves this plan)

- After W1: three verifiers = PASS / PASS / FAIL (3 ACT-N11 errors only); PR checks unchanged except governance-fast red with the same signature.
- After W2: the dry-run error set equals the operator-pending set (logged).
- After W4: `verify_iiac_policy_activation --require-in-force` exit 0 locally; the single governance pytest command exits 0; the C.4 matrix is green on C_P2.
- After W5/W6/W7: development push runs green; §F F-01..F-17 PASS in the TV report.

---

## Disposition

**`PLAN_READY_FOR_EXECUTION`**

**First executable batch:** W0 + W1 (ORCH, on `mcp-p6-gov-closeout` / PR #460): verify the baseline, then publish the gate-compliant **unsigned** IIAC OD, mark the IIIC draft superseded, and draft GOV-01 / TICKET-01 / P7-01. Docs only, one commit, `Ref:` trailer, push.
**STOP boundaries of the first batch:** no edit of `IIAC_POLICY_ACTIVATION_RECORD.json`; no edit of auditor artifacts, product paths or the ladder; no signature line, `AUTHORIZED` status or timestamp; no merge; no new branch; do not commit the untracked `agent-env/plans/pasted-content-id-6543-…md`. End the batch by reporting C_W1 and the unchanged CI state, then hand W2 to a new CERT session.
