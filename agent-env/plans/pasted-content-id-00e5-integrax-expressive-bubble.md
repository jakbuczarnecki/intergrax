# ENT-AUDIT-X — Complete Enterprise Certification & Formal Closure: Master Plan

> Plan date 2026-11-08 · Mode: PLAN ONLY · Target status (exact IIAC vocabulary, `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` L22-30):
> **`ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`**
> Scope qualifiers (SCOPED, COMPENSATING, process_enforced) go in the IIAC manifest `limitations` and the pass-record body, never in the roadmap headline.
> Operator decisions taken for this plan: **OD-ENT-02 = COMPENSATING primary** (LIVE kept as a pre-wired alternative); **OD-ENT-00/01/08 signed by operator-delegated signing sessions** (IIAC Amendment 01, `process_enforced`).

## 0. Context

ENT-AUDIT-X is technically complete and internally closed (`INTERNALLY_CLOSED_AUTOMATED_INTERNAL_AUDIT_ACCEPTED`), but formal closure under the now IN_FORCE IIAC policy has not happened. The blockers are not product defects. They are:

1. an independent IIAC audit that has never been run;
2. an IIAC-unaware baseline-promotion trust path and an unusable closure preflight;
3. stale CI (2 red tests);
4. unsigned operator decisions;
5. unpublished ladder pins.

This plan closes them in the minimum number of waves, reopens nothing that is already qualified, and creates no second SSOT. It stops at a formal ARCH-FREEZE handoff and does not execute ARCH-FREEZE.

## 1. Executive AS-IS audit (verified 2026-11-08)

**Baseline**

| Fact | Verified value |
|---|---|
| Supplied HEAD `ba07ebaa34d04633bd60510f68c18346b1ef160d` | = `origin/development` tip (`git ls-remote`) |
| Local `development` | `65ccf681a`, clean, **14 commits behind**. Not an ancestor relation problem: remote is strictly ahead (GitHub compare: `ahead_by 14, behind_by 0`) |
| The 14 commits | Roadmap/FRZ/CAAP/IIAC-matrix patches, EVOL-X W7-06 handoff (`7a4f279fd`, `360a4ed91`), D10 attestation, tickets T-IGX-022-A/B, 023-A, delegated decisions 2026-11-07, 2 EVOL-X test pins. **None touch `ent_audit_x/`, `intergrax/`, `agents/`, `applications/` or the ENT workflows** |
| TES `89c76eb7…`, B5 `30085953…`, re-audit `0e538269…` | All three are commits and ancestors of local HEAD |
| Change delta TES→HEAD | 155 commits, 590 files, **0 files in `intergrax/ agents/ applications/`**. ENT-relevant tooling: `scripts/enterprise_audit/{mechanical_readiness_check,reconcile_c075_s2,wbs_reconciliation_lib}.py`, tests `test_evidence_publication_strict.py` (mod), `test_sandbox_redaction_sensitive_markers.py` (new). `enterprise-audit*.yml` unchanged. `unit-tests.yml` gained a regression gate |

**Programme state**

- IIAC `IIAC-2026-11-05.1` is IN_FORCE (activation `f8c29fb8…`, ancestor of HEAD).
- EVOL-X is formally CLOSED (`be77b3a5…`, IIAC B6 `1dd76933…` PASS_SCOPED with 12 limitations, ENT handoff `7a4f279fd…`, D10 attestation PASS dated 2026-11-08). The D10 verifier's independence is "not provable from git".
- ENT: internal closure only. Formal B6 `PENDING_AUTHORITY`. `B6_ACCEPTANCE_PUBLICATION_SHA`, `ACCEPTED_BASELINE_SHA`, `FORMAL_CLOSURE_PUBLICATION_SHA` are unset. Matrix C 69/75 (six NOT_QUALIFIED). FRZ-AUD-01..10 OPEN. Gap register: only GAP-RC-OD-TRUST OPEN among 33 GAP-RC.

**CI (read from actual runs, not assumed)**

- `enterprise-audit.yml` PR runs: all latest 8 = `failure`. Run 42452693215 (head `360a4ed91`): `ent-audit-fast` failed at "L1/L2/L3 qualification" with 2 failed / 182 passed / **94 skipped**. `ent-audit-pr-gate` succeeded.
- Nightly: 2026-11-07 scheduled run on `development` failed identically (2 failed). Last successes: dispatch run 42015310879 (head `162acea3`) and scheduled run on `main` 2026-11-06.
- Root cause (the same in every failing run): `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py::{test_validator_passes_on_canonical_ssot, test_validator_cli_passes}`. The validator constants `scripts/enterprise_audit/master_plan_ssot_lib.py:90-94` and the check at `:433` don't know the post-`477017f15` disposition "INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED". **Closure PRs cannot go green until this is fixed.**
- Branch protection: `GET …/branches/development/protection` → 404. Rulesets: `Protect development` (id 26771061, created 2026-11-07 20:59) = `deletion` + `non_fast_forward` only, no required checks, no PR rule, `bypass_actors: []`. `Protect main` has ci-smoke/governance-* checks only.

## 2. Canonical authority & SSOT reconciliation

**Authority order for ENT closure** (higher wins):

1. IIAC activation record, protocol, `iiac-v1` manifest schema.
2. Active operator decisions (delegated record 2026-11-07, `IIAC_OD_ACTIVATION_*`).
3. Roadmap §5 and the FRZ checklist (patched to IN_FORCE at `5d74453af`).
4. `ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md` — technical execution SSOT, Matrices C/D/E. Its footer, §18 and Matrix A are stale (cite TES `b8bb88c9`).
5. Exact-SHA evidence (ladder, status JSON, `b6_independent_handoff/qualification_at_89c76eb7…`).
6. Everything else is historical evidence, never rewritten, only annotated.

**Superseded, not reopened** (annotate with a one-line "SUPERSEDED BY IIAC" overlay pointer, no history rewrite):

- Protocol 03 as a closure gate. Its semantics (Mode A/B/C, exact SHA, adversarial) are preserved inside IIAC.
- `COMPLETE_FORMAL_CERTIFICATION…MASTER_PLAN` §1.2/L52/L66/L138 ("Path A, ECAP-P4 NOT IN FORCE, named auditor").
- `ANNEX_B` auditor and promotion-gate columns.
- `OPERATOR_DECISION_AND_EXTERNAL_AUTHORITY_PACKET`, `R8_EXTERNAL_GATE_PACKET`, `ONE_PASS_AGENT_HANDOFF_B6`.
- Matrix A.
- R6/R7 TES/EVID SHAs.
- OD-IC-01 stays "internal only" and never satisfies a formal gate.

**Do-not-touch list**

- `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` L4/L91 still say `NOT_IN_FORCE`. The protocol is one of six `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-49`). Editing it flips `verify_iiac_policy_activation --require-in-force` to fail. Leave it; the activation record is authoritative. Optional operator decision OD-X1 (a policy-version patch) is non-blocking.
- EVOL-X artifacts, including `evol_x_closure_publication_preflight.py` (pinned to the B-4 acceptance, blob `356dc59a…`), the EVOL status JSON (fix the ENT reader, not the EVOL writer), and all EVOL history.
- Ladder history (append-only). The two same-role rows (TES, EVIDENCE) are kept.

**The one new file** (not a competing SSOT): `ent_audit_x/ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md` (task W0-04). It is a *status overlay* that supersedes only the stale status columns of Matrix A/B/D/F, Annex B/G2 and the closure JSON prose. Requirements remain by reference to the canonical documents. Machine state stays in `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json` (extended); SHAs stay in `ENT_AUDIT_X_SHA_LADDER.md`.

## 3. Requirement inventory (coverage verdicts)

| Area | Verdict | Basis |
|---|---|---|
| A. Architecture (runner, registry, contracts, scheduling, sandbox, redaction, tenant, fail-closed) | QUALIFIED at TES, no rebuild | `intergrax/platform/enterprise_audit/` 20 modules, 2691 lines; manifest-driven registry; `sandbox.py` path-escape + redaction; ADV-01..31 suite |
| B. Coverage: 85 checks (63 arch + 13 TEN + 9 VBN), 75 Matrix C, VBN 1-31 | QUALIFIED at TES; re-confirm at TES′ | `manifest.py:62`, `evidence.py:73`, L3 FULL/FREEZE 85/85; `vbn_domains.v1.json` `domains_accounted=31` |
| C. Modes & reporting | QUALIFIED functionally; **certifying FREEZE never run against an accepted baseline** | baseline promotion unimplemented for IIAC |
| D. Qualification & CI | **Defect D-1 (red CI); 94 CI skips unexplained** | see §1 |
| E. Governance & certification | **Never executed**: no IIAC audit, manifest, pass record, delegation for ENT, ODs | |
| F. Final acceptance (FRZ-AUD-01..10) | all OPEN | §6 |

**New findings**

- **Matrix C lives in the master plan** (L53-132, "Matrix C — 75 TaskID"), not in Annex C. Annex C is the 85-check control matrix. The task statement's "Matrix C 69/6" refers to the master-plan matrix.
- `OPEN-BP-01` is defined in `OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md:7` and `OD_ENT_02_THREAT_MODEL.md:5`, but **not** in `ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md`, as the package claims. It is also absent from the roadmap, the FRZ checklist and the WBS.
- `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json` has no `od_ent_02_mode` and no IIAC fields.
- The ladder lacks rows for the IIAC activation `f8c29fb8`, `4d54dfa9e`, `116fb6a2a` and the EVOL formal-closure and handoff pins.

## 4. Single consolidated closure matrix

Columns carried in the ledger: `TASK_ID · CANONICAL_REQUIREMENT · CURRENT_STATUS · EVIDENCE_SOURCE · EVIDENCE_SHA · GAP · IMPLEMENTATION_PATH · REQUIRED_ACTION · OWNER_ROLE · DEPENDENCIES · TESTS · ACCEPTANCE_CRITERIA · INDEPENDENT_AUDIT_REQUIRED · OPERATOR_DECISION_REQUIRED · PARALLELIZATION_ELIGIBILITY · FINAL_DISPOSITION`.
Below: compact working form; `Par` = parallel-eligible; `Aud` = independent audit required; `OD` = operator decision.

**Class 1 — fully qualified, re-verify only (no new tasks)**

| ID | Requirement | Status | Evidence @SHA | Action | WBS | Aud |
|---|---|---|---|---|---|---|
| M-01 | Runner/registry/contracts/isolation/fail-closed | QUALIFIED | L3 @`89c76eb7` | replay at TES′ + auditor Mode A/B/C | W1-05, W2 | yes |
| M-02 | 85 checks, FULL/FREEZE 85/85 | QUALIFIED | B5 @`30085953` | FULL+FREEZE-candidate at TES′ | W1-05 | yes |
| M-03 | Matrix C 69 records | QUALIFIED | master plan Matrix C | auditor spot-check + validator | W2 | yes |
| M-04 | VBN 1-31, hardcoding, silent fallback | QUALIFIED | `vbn_domains.v1.json` | replay | W2 | yes |
| M-05 | B5 tree integrity | QUALIFIED (W8-001/002/003/006 FIXED_WITH_EVIDENCE) | `verify_b5_publication_tree` | re-run on TES′ tree | W1-06 | yes |
| M-06 | EVOL-X predecessor | CLOSED | `be77b3a5`, `7a4f279fd`, D10 | validate only; OD-ENT-00 | W0-03, W3-01 | no |
| M-07 | IIAC IN_FORCE | CLOSED | `f8c29fb8` | gate re-run at each publication | all | no |

**Class 3 — real implementation defects (the only code work)**

| ID | Defect | Evidence | Action | Par |
|---|---|---|---|---|
| D-1 | Validator rejects the current disposition → CI red on every PR and the nightly | `master_plan_ssot_lib.py:90-94,433`; runs 42452693215, 42359724126 | Add the current internal-closure disposition (and the future IIAC-closed one) as accepted constants. Don't weaken the other checks. Add a negative test that an unknown disposition is still rejected | lane A |
| D-2 | **Baseline promotion is human-only**: `baseline.py:176-246` requires `operator_origin:true`, `record_origin ∉ {agent,cursor,automation}`, a record in `operator_records/`, a ladder B6 pin. A delegated signing session (an agent) can never satisfy it. This is the root of GAP-RC-OD-TRUST and C001/C002/C003/C042/C057/C065 (`IIAC_REQUIREMENT_MIGRATION_MATRIX.md:24-33`) | | Add an IIAC-aware path: an accepted-baseline manifest plus auditor replay, or a delegated `operator-delegate` record with a `Delegation:` line resolving to a published delegation. Keep the human path. Negative tests: forged delegate, agent origin with no delegation, missing B6 pin, wrong TES/evidence SHA, expired delegation, non-ancestor SHA | lane B |
| D-3 | **Closure preflight unusable**: `scripts/qualification/ent_audit_x_formal_closure_preflight.py` (a) reads the EVOL key `formal_closure_status`, which does not exist (`N-10` always fails; real keys: `parent_evol_x_status`, `protocol_03=SUPERSEDED_BY_IIAC_B6_ACCEPTANCE`, `w7_06_handoff_publication_sha`); (b) looks for the manifest at `ent_audit_x/*iiac*manifest*.json`, but the EVOL precedent keeps it at `iiac_evidence/<STAGE>/certification/iiac_audit_session_manifest.<STAGE>.json` (the pass record sits in the stage dir, which matches); (c) **fails open on unset pins**: `_ladder_sha_unset` (L33-41) matches only a bare `unset`/`null`/`PENDING`/`—` cell, not the ladder's `*unset*`, so a row without the word "PENDING" is not detected as unset (verified in W0-06 by direct read); (d) rejects the old CLOSED string but not the IIAC one; (e) requires `FORMAL_CLOSURE_PUBLICATION_SHA` set, but that is only set in the post-closure pin commit, a chicken-and-egg problem | | Split into **Gate A** (pre-closure: everything except the roadmap CLOSED headline and `FORMAL_CLOSURE_PUBLICATION_SHA`) and **Gate B** (post-pin: all), mirroring the EVOL two-gate preflight. Read the real EVOL keys plus the IIAC B6 record and ladder pins. Accept the IIAC headline string in the guard. Fix `*unset*` parsing. Require `od_ent_02_mode ∈ {live,compensating}` and, for compensating, a real verifier PASS | lane C |
| D-5 | OD-ENT-02 verifiers: `verify_od_ent_02_compensating_control.py` only greps marker strings ("PASS (package draft)"); `verify_ent_audit_branch_protection.py` reads only `/protection`, not rulesets | | The compensating verifier must check real, live evidence: ruleset API payload, required ENT/regression workflow run IDs on the publication SHA, Gate-A/B wiring and the OPEN-BP-01 register entry. The live verifier must also read rulesets. Neither may print PASS from marker text alone | lane C |
| D-6 | State/ladder gaps: no `od_ent_02_mode`/IIAC fields in the status JSON; ladder rows missing. **Parser concern closed in W0-06:** `parse_sha_ladder_table` (`baseline.py:139`) matches the first cell exactly, so suffixed historical rows ("TEST_EXECUTION_SHA (historical)") are ignored. Exactly one row each of `TEST_EXECUTION_SHA` (L42) and `EVIDENCE_PUBLICATION_SHA` (L41) exists, so there is no ambiguity today. **Constraint:** W1-06 must append TES′/EVIDENCE′ as *new* exact-name rows *below* the current ones (last wins) and re-label the old ones with a suffix. A test must pin this | | Extend the JSON additively. Append ladder rows under the constraint above. Make D-3's preflight reuse `parse_sha_ladder_table` instead of its own substring matcher | lane C |
| D-7 | `mechanical_readiness_check.py:369-370` requires a "Protocol 03 dossier" with the substrings "no pass declared" and "## External blockers (GAP-B)". The test `test_mechanical_readiness.py:21-22` pins `formally_certified is False` | | **Verdict (W0-06): not blocking today, but it constrains later edits.** `_verify_dossier_complete` (L368-384) requires in `ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md`: the TES pin, an `Audited IMPLEMENTATION_SHA \`<40hex>\`` that is an ancestor of HEAD, `b6_independent_handoff`, `## External blockers (GAP-B)` and the text "no pass declared". `ent-audit-pr-gate` runs this via `test_mechanical_readiness.py`. So: after W1-06 the dossier TES pin and the Audited IMPLEMENTATION_SHA must be moved to TES′; the five substrings must survive every later edit (record the IIAC pass in a separate section or file, never by deleting "no pass declared"). The check also needs a green CI at HEAD or a "doc-only HEAD" proof | W1-06, W5 |

**Class 4/5/6 — missing evidence, decisions, publications (no code)**

| ID | Item | Owner | Dep | OD | Aud |
|---|---|---|---|---|---|
| E-1 | IIAC audit session manifest + handoff record + audit report + findings + `*_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` | auditor → certifier | W1 done | no | self |
| E-2 | Accepted-baseline manifest (OD-ENT-01) → `ACCEPTED_BASELINE_SHA` | certifier + signer | E-1, D-2 | OD-ENT-01 | auditor replay |
| E-3 | Certifying FREEZE run vs the accepted baseline, with a fresh bundle | implementer(run)/auditor(verify) | E-2 | no | yes |
| E-4 | FRZ-AUD-01..10 per-row evidence + promotions | auditor evaluates, certifier publishes | E-3 | no | yes |
| E-5 | OD-ENT-02 compensating package refreshed + independently certified | implementer drafts, auditor accepts | D-5 | OD-ENT-02 | yes |
| E-6 | OD-ENT-00 (accept EVOL formal status) | signer | M-06 | OD-ENT-00 | no |
| E-7 | OD-ENT-08 (formal closure authorisation) | signer | E-1..E-5, Gate A | OD-ENT-08 | no |
| P-1 | `B6_ACCEPTANCE_PUBLICATION_SHA` | certifier | E-1 | | |
| P-2 | `ACCEPTED_BASELINE_SHA` | certifier | E-2 | | |
| P-3 | `FORMAL_CLOSURE_PUBLICATION_SHA` | certifier | E-7, Gate A | | |
| P-4 | roadmap §5 / ledger, FRZ checklist, WBS (via the generator script), ladder, status JSON, terminal dossier | certifier | P-3 | | |

**Class 7 — superseded (disposition SUPERSEDED, annotate only)** is in §2.

## 5. Architecture & qualification findings

1. Product code has been frozen since TES. **No product defects found**; the work is trust-path, gate and CI repair.
2. D-2 is the only change under `intergrax/platform/enterprise_audit/` (`baseline.py`), so it **forces a new TES′** (the audited implementation SHA changes). D-1/D-3/D-5/D-6 change `scripts/` and `tests/` only, but they are inside the qualified surface (`scripts/enterprise_audit/`).
3. The 94 CI skips need explanation. W1-05 and the auditor must classify them (environment-gated vs. silently dead). `test_ent_audit_cannot_exit_clean_with_skipped_mandatory.py` exists and must stay green.
4. The "latest CI PASS" is unknown: the last clean nightly (42015310879) is at `162acea3`, 169 commits before the remote tip. W1 produces new CI proof on TES′ (nightly `workflow_dispatch` FULL, the same pattern as run 42015310879), and the plan reads the actual run, never a prior claim.
5. Amendment 01 contradicts itself (L3 "in force as operator-adopted policy text 2026-11-07, not audited, not mechanically enforced"; L51/121/140 and the template "proposed — not in force"). No validator recognises `operator-delegate`. Mitigation: the operator explicitly confirms the status in the delegation record (O-1). The closure manifest records it as a `process_enforced` limitation.
6. The role registry has no certification-agent or signer role. W0 checks whether `verify_iiac_role_separation.py` needs a registry change. If not, roles are mapped in the handoff records (the operator attests them), as EVOL-X did.
7. Not verified: Mode C semantics (confirm in ECAP text at W0); whether `ent_audit_x_formal_closure_preflight.py` is a `MATERIAL_GOVERNANCE_PATH` (W0 check, since it drives the D-3 process); the full contents of Session B/C instructions.

## 6. FRZ-AUD-01..10 acceptance matrix

All rows: status **OPEN** today. Implementation exists at TES; the missing artifact is IIAC per-row evidence + certifier promotion. PASS needs: positive test + negative test green at the evidence SHA, cited in the manifest, an auditor row verdict, and a certifier ladder/checklist update in a dedicated commit. There is no mass promotion.

| FRZ-AUD | Requirement | Positive / negative test (Annex B) | Evidence artifact | Extra dependency |
|---|---|---|---|---|
| 01 | Runner + CLI | `…runner_exists` / `…runner_missing_neg` | L2 log, manifest | — |
| 02 | Modular `AuditCheck` registry | `…registry_contract` / `…monolith_script_neg` | registry unit output | — |
| 03 | FRZ/invariant coverage matrix | `…frz_coverage_complete` / `…unmapped_frz_neg` | `frz_coverage.v1.json` | — |
| 04 | JSON+MD reports, schemas | `…report_schemas` / `…report_schema_drift_neg` | report artifacts | — |
| 05 | Exact SHA attribution | `…manifest_sha_fields` / `…stale_sha_neg` | `evidence_manifest.v1.json` | — |
| 06 | Baseline vs current | `…baseline_compare` / `…non_ancestor_baseline_neg` | `baseline_comparison.v1.json` | **accepted baseline (E-2)** |
| 07 | NEW/RESOLVED/REGRESSED/UNCHANGED | `…drift_semantics` / `…invalid_drift_class_neg` | drift section | E-3 |
| 08 | Exit codes 0–3 | `…exit_code_contract` / `…false_pass_neg` | CLI transcript | — |
| 09 | FAST/FULL/FREEZE/CHANGED-SINCE | `…mode_policy_matrix` / `…partial_labeled_full_neg` | mode manifests | E-3 (certifying FREEZE) |
| 10 | Post-freeze enforcement | `…ci_policy_documented` + workflow run / `…branch_protection_required_neg` | **LIVE**: `verify_ent_audit_branch_protection.py` exit 0 + `gh api …/protection`. **COMPENSATING**: refreshed package + Session D acceptance + `verify_od_ent_02_compensating_control.py` real PASS | OD-ENT-02 |

Cross-stage: FRZ-FRZ-11 ("ENT independently CLOSED") is promoted at W5. FRZ-FRZ-12/13 and the other ARCH-FREEZE rows stay for ARCH-FREEZE. FRZ-EVOL-* status must be read at W0-03 (unknown whether all were promoted at EVOL closure).
Annex B test names are abbreviated in the source, so the exact IDs come from `tests/qualification/ent_audit_x/acceptance/` at W0.

## 7. WBS (W0 → W6) and dependency DAG

Convention: ticket-first. Next ticket IDs are **T-IGX-024-A…** (confirm against `governance/tickets/` after pulling; remote may be newer). Each product-path commit carries `Ref: governance/tickets/T-IGX-0xx-x.md`. Quality gate before every push: `uv run python -m scripts.governance.igx_gov ticket`. Plan approval per `igx-devos-plan-ticket-scope` (operator "Approved" line).

### W0 — AS-IS reconciliation (orchestrator, read-mostly)

| Task | DONE criteria |
|---|---|
| W0-01 Sync | `git fetch origin development`, fast-forward local `development` to the actual tip (record `git rev-parse HEAD`). Re-run the supplied-HEAD ancestry and `git status` check. Record any commits after `ba07ebaa` and re-classify the delta. Stay on the named branch (git-branch-safety rule) |
| W0-02 Lineage | `merge-base --is-ancestor` for TES, B5, re-audit, `f8c29fb8`, `be77b3a5`, `7a4f279fd`, `1dd76933` against the new HEAD: all pass or STOP |
| W0-03 Predecessor + FRZ-EVOL | Read the remote EVOL handoff, D10, `FRZ-EVOL-*` rows and `ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md`. Record any EVOL residual that blocks ENT (none expected) |
| W0-04 Closure matrix | Commit the status overlay (§2) with this plan's §4 content; the ladder gets the missing IIAC/EVOL rows (append-only) |
| W0-05 Stale-text batch | One docs-only commit, "patches 15+" style: SUPERSEDED-BY-IIAC pointers on the files listed in §2, roadmap L490/561/2065/2066 factual fixes that don't promote status, CAAP L152/154, FRZ checklist L47, WBS T-B5-EVOL statuses. **Excludes** the IIAC protocol and any `MATERIAL_GOVERNANCE_PATH` |
| W0-06 Probes | Read the `--help` of `run_enterprise_audit.py`, `iiac_activation_gate.py` material paths, `verify_iiac_*`, ECAP Mode A/B/C text, the EVOL precedent layout for IIAC artifacts (`iiac_evidence/EVOL-X/`), `parse_sha_ladder_table` last-wins behaviour, D-7 verdict, the role-separation verifier's registry needs |
| W0-07 Tickets | T-IGX-024-A (W1 corrective) scaffolded and approved; branch/PR names listed for operator approval |

### W1 — Technical readiness (implementer; lanes in isolated worktrees)

Worktrees: `<parent>/_wt/intergrax/<slug>` with slugs `ent-w1-a`, `ent-w1-b`, `ent-w1-c`. Scratch goes under `_wt/intergrax/_scratch/<slug>`. Branch creation needs operator approval (branch-safety), which the plan approval covers. Merge order A → B → C by PR.

| Task | Lane | Files owned (exclusive) |
|---|---|---|
| W1-01 D-1 validator fix + tests | A | `scripts/enterprise_audit/master_plan_ssot_lib.py`, `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` |
| W1-02 D-2 IIAC-aware baseline trust path + negative tests | B | `intergrax/platform/enterprise_audit/baseline.py`, `tests/qualification/ent_audit_x/unit/test_baseline_promotion.py` (+ new IIAC test file) |
| W1-03 D-3/D-5/D-6 preflight Gate A/B, verifiers, status JSON, ladder parser | C | `scripts/qualification/ent_audit_x_formal_closure_preflight.py`, `scripts/governance/verify_ent_audit_branch_protection.py`, `verify_od_ent_02_compensating_control.py`, their tests |
| W1-04 Merge | orchestrator | PRs A→B→C. Required `ent-audit-fast` and `ent-audit-pr-gate` must be **green on each PR** (read the run) |
| W1-05 Requalification at **TES′** (the clean merged tip) | single runner | See §13. L1/L2/L3 + acceptance, FULL 85/85 with `--evidence-bundle`, FREEZE `--bootstrap-candidate` (writes only a candidate baseline), `mechanical_readiness_check.py`, `check_manifest_annex_c_alignment.py`, validator CLI, skip classification, nightly `workflow_dispatch` FULL on TES′ |
| W1-06 Evidence publication | implementer | `publish_b5_evidence_bundle.py --qualification-sha <TES′>` → `qualification_at_<TES′>/`; `verify_b5_publication_tree` PASS; ladder TES′/EVIDENCE′ rows (append-only; the old rows are kept as historical) |

**DONE for W1:** the implementer hands over (TES′, EVIDENCE′, CI run IDs, candidate baseline, change delta TES→TES′ classified). The implementer declares **no** PASS.

### W2 — Independent IIAC certification (two separate sessions)

- **W2-01 Handoff record** (operator attests: new chat, registry roles, distinct session IDs, scoped write surface). Template `IIAC_SESSION_HANDOFF_RECORD.template.md`.
- **W2-02 Session D, auditor** — protocol in §10. Output: audit report, findings, the `iiac_audit_session_manifest.v1` draft, the digests. Written under `.tmp/session/ent-w2-audit/`; the auditor does not commit and edits no product path.
- **W2-03 Findings loop** — if findings exist: implementer corrective (W1 pattern), requalify the delta, a **new** auditor session re-audits (or a delta-audit like T-IGX-022-B). Max 2 loops, then REJECT procedure (§15).
- **W2-04 Certifier session** (not the implementer or the auditor; R-4 allows it to be the signer too): runs `verify_iiac_audit_manifest.py <manifest> --require-handoff-template`, `verify_iiac_role_separation.py`, `verify_iiac_policy_activation --require-in-force`. Publishes **phase 1**: the pass record + manifest + certification record + `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json` (`formal_b6_status=ACCEPTED_VIA_IIAC…`) with **no** CLOSED text, per the EBH-7 rule (publication-contract L21).
- **W2-05 Phase 2 pin** — commit that sets ladder `B6_ACCEPTANCE_PUBLICATION_SHA` = the phase-1 SHA (non-self-referential, the EVOL pattern).

### W3 — Baseline, authority, enforcement

| Task | DONE |
|---|---|
| W3-00 Delegation (**operator O-1**) | Operator signs a delegation record (template `IIAC_OPERATOR_SIGNING_DELEGATION.template.md`) naming a new dedicated signing session and scoping it to OD-ENT-00/01/02/08 and expiry. The delegation, policy activation and amendments stay non-delegable (S-3) |
| W3-01 OD-ENT-00 | The signing session records acceptance of EVOL formal status against `be77b3a5`/`7a4f279fd`/D10. Line `Approved OD-ENT-00: operator-delegate @ <ISO-8601>` + `Delegation: <path>@<sha>`. **Never** fabricate timestamps or IDs |
| W3-02 OD-ENT-01 + baseline | The signing session records OD-ENT-01. The certifier runs `promote-baseline` via the D-2 path from the closed TES′ bundle. The auditor replays the manifest (Mode A + B). `ACCEPTED_BASELINE_SHA` published (P-2). C001/C002/C003/C042/C057/C065 move to QUALIFIED with cited evidence; GAP-RC-OD-TRUST closes |
| W3-03 OD-ENT-02 compensating | Package + threat model refreshed per §12. The auditor issues a sufficiency verdict (Session D step 2). The signing session records OD-ENT-02 with `od_ent_02_mode=compensating`. OPEN-BP-01 is registered in the handoff, the roadmap OD register and the threat model |
| W3-04 (conditional) LIVE | If a repo admin applies protection before W5, flip to `live`. See §12 |

### W4 — Formal FREEZE and FRZ-AUD

- **W4-01** Certifying `FREEZE` run on a clean tree vs the accepted baseline (no `--bootstrap-candidate`). Freshness assertion: `git diff --stat TES′..HEAD -- intergrax agents applications scripts/enterprise_audit .github/workflows/enterprise-audit*.yml` is empty. If not, requalify (§13). New bundle published with the manifest split (`qualification_sha` ≠ `audit_execution_sha`).
- **W4-02** The auditor (Session D) evaluates the FRZ-AUD-01..10 rows individually (§6), issues row verdicts, and records limitations.
- **W4-03** The certifier promotes each row in dedicated commits (checklist rows plus ladder), with citations to manifest evidence. FRZ-AUD-10 reads `PASS — COMPENSATING (SCOPED)` (or `PASS — LIVE`). Preserve exclusions.

### W5 — Formal closure

1. **Gate A** (`ent_audit_x_formal_closure_preflight.py --gate A`) green. Resolves the contradiction between the historical (Protocol 03) and IIAC logic via D-3.
2. The signing session records **OD-ENT-08**.
3. Certifier **closure phase 1** (one commit): roadmap §5 ENT row headline set to the exact target string; the FRZ-FRZ-11 promotion; the status JSON (`formal_closure_campaign_status=CLOSED_IIAC…`; the historical `FAIL_CLOSED_EXTERNAL_AUTHORITY_PENDING` kept in a `superseded_history` field); the WBS via `verify_arch_freeze_plan_traceability.py` generator (the T-B6-* rows → COMPLETE); terminal dossier. It must not be the commit that first creates the pass record (that was W2-04).
4. Push through a PR. Required checks are green on the PR head (read the actual runs).
5. **Closure phase 2**: pin `FORMAL_CLOSURE_PUBLICATION_SHA` = the phase-1 merge SHA in the ladder. **Gate B** green.
6. Verify: `git ls-remote origin development` equals the local tip, the artifacts exist on the remote, and the roadmap headline matches character for character. Session-end branch-safety check.

### W6 — ARCH-FREEZE handoff (docs-only)

Update `ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md` in place (it is the handoff contract; it must also gain the OPEN-BP-01 entry the package already claims). Contents in §17. Add the `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` ENT-side updates through its generator. **Do not execute ARCH-FREEZE.**

### DAG

```text
W0-01 → W0-02 → {W0-03, W0-06} → W0-04 → W0-05 → W0-07
W0-07 → { W1-01(A), W1-02(B), W1-03(C) }  [parallel, disjoint files]
{A,B,C} → W1-04 (A→B→C PRs) → W1-05 → W1-06
W1-06 → W2-01 → W2-02 → (W2-03 loop?) → W2-04 → W2-05
W3-00(O-1) ∥ W2-*  (operator action, not blocked by audit)
W2-05 → W3-02 ;  W3-00 → W3-01 ;  W3-03 (needs D-5 + W2-02 verdict) ∥ W3-02
{W3-01, W3-02, W3-03} → W4-01 → W4-02 → W4-03 → W5 (Gate A → OD-ENT-08 → P1 → PR → P2 → Gate B) → W6
Critical path: W0 → W1(B lane + requal) → W2 audit → W3-02 baseline → W4 FREEZE → W5.
```

## 8. Parallelism & role-ownership matrix

| Session / role (registry role) | Owns | Forbidden |
|---|---|---|
| S0 Orchestrator (`orchestrator`) | DAG, ledger, status overlay, W0, W6 | product edits, any PASS |
| S1 Implementer (`implementer`, 3 worktree lanes A/B/C; verification serialised by one runner, per test-process-budget) | W1 | B6 PASS, FRZ promotion, self-CLOSED, any edit during an audit |
| S2 Independent IIAC auditor (`independent-quality-auditor`), new chat | W2-02, W4-02 | edits to product paths between `audited_implementation_sha` and `audit_execution_sha`, accepting its own commits |
| S3 Certification/governance agent (mapped in the handoff; no registry role exists), new chat | W2-04/05, P-1..P-4, W4-03, W5 | replacing the audit with a checklist, certifying validators it authored without review, overriding fail-closed |
| S4 Operator-designated signing session, new, dedicated (R-2) | OD-ENT-00/01/02/08 records | any other action; fabricated signatures |
| Operator (human) | O-1 delegation, plan approval, branch approval, optional OD-X1, GitHub admin if LIVE | — |

No agent certifies its own implementation. The signer and certifier may coincide (R-4). The implementer and auditor may never.

## 9. Operator decision inventory

| ID | Decision | Signer | Gate | Status |
|---|---|---|---|---|
| O-1 | Delegation record for the ENT signing session (confirm Amendment 01 status; scope; expiry) | **operator personally** (non-delegable) | before W3 | NEW |
| OD-ENT-00 | Accept EVOL formal status | delegated session | W3 | OPEN |
| OD-ENT-01 | Accepted baseline | delegated session | W3 (after B6 pin) | OPEN |
| OD-ENT-02 | Mode = compensating (primary); live alternative | delegated session | W3 | OPEN |
| OD-ENT-08 | Authorise formal closure | delegated session | W5 | OPEN |
| OD-ENT-03 | Auditor appointment | superseded by IIAC (session manifest) | — | SUPERSEDED |
| OD-X1 | (optional) a policy-version patch of the stale IIAC protocol L4/L91 | operator | none | non-blocking |
| Branch/PR approvals | feature branches for lanes A/B/C and phases | operator via plan approval | W1 | requested at plan exit |

The OD-EVOL-11 "auditor_id NOT RECORDED" item is not relevant to ENT (IIAC identifies sessions, not named people).

## 10. IIAC independent audit protocol for ENT (Session D)

Preconditions: handoff record; EVOL IIAC CLOSED manifest (`1dd76933…`); TES′, EVIDENCE′, candidate-baseline SHAs all ancestors of the audited HEAD; a new chat in role `independent-quality-auditor`.

1. **Scope lock.** Record `audited_implementation_sha` = TES′, `audit_execution_sha` = the HEAD at audit start. Assert the product-path diff between them is empty.
2. **Mode A (integrity).** Digests of every bundle file against `evidence_manifest.v1.json`. `verify_b5_publication_tree`. Schema validation of all six schemas. Reject if any digest mismatches.
3. **Mode B (replay).** Replay the retained CI/JUnit and sandbox evidence. Read the actual GitHub runs (`gh run view`) for the ENT workflows on the TES′ head. Do not accept a prior PASS claim.
4. **Mode C (adversarial).** Per the EVOL-X IIAC precedent (`IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md` §5.5): Mode A tamper cases, Mode B replay tampers and CLI negatives must all be rejected. For ENT also ADV-01..31 and `test_ent_audit_cannot_exit_clean_with_skipped_mandatory`. **Separately** (not Mode C): a clean-checkout execution of FULL 85/85 and FREEZE-candidate at the audited SHA, with classification of all skips. ECAP itself does not define the modes; the definitions come from the EVOL precedent and the ENT re-audit draft (Mode A = B5 digest + publication-index reconciliation; Mode B = CI JUnit / workflow conclusions + nightly FULL).
5. **Coverage.** Reconcile 85 manifest checks ↔ 85 acceptance wrappers ↔ Annex C; Matrix C 75 ↔ master plan validator; VBN 31 domains; GAP-COV.
6. **Adversarial / fail-closed.** Forged and expired delegation; agent-origin record; stale/non-ancestor SHA; tampered bundle; missing mandatory check; secret leakage into logs/argv/bundles; path escape; a tenant bleed probe. All must fail closed.
7. **Trust chain.** L0–L7 (`ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md`) re-expressed for IIAC; the B6 package draft §1–19 mapped.
8. **Neighbour surfaces.** Preflight, `igx_gov`, the workflows, `unit-tests.yml` regression gate, ladder parser.
9. **OD-ENT-02 route sufficiency** (W3-03) and **per-row FRZ-AUD verdicts** (W4-02).
10. **Verdict.** PASS / PASS_SCOPED / CONDITIONAL / FAIL, with `limitations`, `findings_digest`, `audit_report_digest`, `audit_scope_manifest_digest`, `test_matrix_digest`. The assurance level stays `process_enforced`. Declare it honestly; no cryptographic claim.

A technical recommendation alone (like `0e538269…`) is not formal B6.

## 11. Exact acceptance test matrix

Exact flags are confirmed from `--help` at W0-06. One `uv run pytest` per verification step (test-process-budget). Logs go under `.tmp/session/<task>/`.

| # | Check | Command (shape) | PASS |
|---|---|---|---|
| T-01 | ENT L1/L2/L3 | `uv run pytest tests/qualification/ent_audit_x/unit tests/qualification/ent_audit_x/integration tests/qualification/ent_audit_x/l3 -q` | 0 failed; skips enumerated and justified |
| T-02 | Validator | `uv run python scripts/enterprise_audit/validate_post_implementation_master_plan.py` | exit 0 (was failing) |
| T-03 | FULL | `run_enterprise_audit.py run --mode FULL --output-dir … --evidence-bundle` | exit 0, 85/85 |
| T-04 | FREEZE candidate | `… --mode FREEZE --bootstrap-candidate` | exit 0, candidate only |
| T-05 | Certifying FREEZE | `… --mode FREEZE` vs the accepted baseline | exit 0, drift classes valid |
| T-06 | Mechanical readiness | `mechanical_readiness_check.py` | exit 0; `formally_certified:false` stays |
| T-07 | Annex C alignment | `check_manifest_annex_c_alignment.py` | exit 0 |
| T-08 | B5 tree | `verify_b5_publication_tree` on `qualification_at_<TES′>` | OK |
| T-09 | IIAC activation | `verify_iiac_policy_activation --require-in-force` | IN_FORCE |
| T-10 | IIAC manifest/roles | `verify_iiac_audit_manifest.py <m> --require-handoff-template`; `verify_iiac_role_separation.py` | exit 0 |
| T-11 | Baseline negatives (D-2) | the new unit tests | every forgery rejected |
| T-12 | Preflight | `ent_audit_x_formal_closure_preflight.py` Gate A, Gate B | A green before W5-3; B green after W5-5 |
| T-13 | OD-ENT-02 | compensating verifier (real) or live verifier | per route |
| T-14 | Traceability | `verify_arch_freeze_plan_traceability.py` | exit 0 after the WBS update |
| T-15 | Quality gate | `uv run python -m scripts.governance.igx_gov ticket` | exit 0 |
| T-16 | CI | `gh run view` of `enterprise-audit.yml` (PR head) and nightly `workflow_dispatch` FULL | `success`, head SHA recorded |
| T-17 | Remote | `git ls-remote origin development` = local tip; artifacts on the remote | match |

## 12. LIVE vs COMPENSATING decision matrix (OD-ENT-02 / FRZ-AUD-10)

| | LIVE | COMPENSATING (**primary**) |
|---|---|---|
| Definition | Required checks `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast` and `/ent-audit-pr-gate` (P0), nightly and `unit-tests.yml` (P1) enforced on `development` | A scoped, independently certified control set. **No equivalence claim.** OPEN-BP-01 remains |
| Today's state | Not met (404; the ruleset has deletion + non-ff only) | Package + threat model exist as drafts. Verifier is marker-only (D-5) |
| Who acts | A repo admin (GitHub) | Implementer drafts, auditor accepts, signing session records OD-ENT-02 |
| Evidence | `gh api …/protection` (+ rulesets) payload, the verifier exit 0 | Rulesets payload (C1), PR+green-run evidence per publication SHA (C2), Gate A/B in the publication process (C3), nightly FULL/FREEZE (C4), ladder append-only + parser tests (C5), OPEN-BP-01 register (C6). *(Final control list is the one in `OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md`, refreshed at W3-03.)* |
| FRZ-AUD-10 disposition | `PASS — LIVE` | `PASS — COMPENSATING (SCOPED)` |
| Debt | none | OPEN-BP-01 → ARCH-FREEZE handoff, with an owner and a trigger (admin enables protection) |
| Switch rule | If protection becomes live before W5, re-run the live verifier, record `od_ent_02_mode=live`, and the auditor re-evaluates row 10 only | Switching back is disallowed once published |

Whichever route is chosen, the "Protect development" ruleset is listed as a real control and also as an explicit non-equivalent (it blocks force-push/deletion, not unchecked merges). The T-IGX-022-A claim that "`development` is protected" is corrected in the handoff.

## 13. Risk-based requalification strategy

| Change since the last evidence | Re-run |
|---|---|
| Docs, ladder, roadmap, FRZ rows, dossiers | none beyond the validators (T-02/T-07/T-14) |
| `scripts/governance`, `scripts/qualification` (preflight/verifiers) | their unit tests + T-12/T-13; no FULL |
| `scripts/enterprise_audit/*` or `tests/qualification/ent_audit_x/**` | T-01 + T-02 + T-07 + T-06 |
| `intergrax/platform/enterprise_audit/**` (D-2) | T-01, T-03, T-04 (full requalification → TES′) |
| Workflows `enterprise-audit*.yml` | T-16 nightly dispatch |
| Any code in `intergrax/ agents/ applications/` outside the ENT package | STOP, orchestrator classifies, broad requal |

W1-05 reruns FULL/FREEZE-candidate once, at TES′ (D-2 forces it). After TES′, every later commit must keep the freshness assertion (§7 W4-01) or trigger a requalification. Don't lower thresholds. Don't touch functional code to satisfy docs.

## 14. Publication and SHA ladder strategy

Append-only ladder; new rows: IIAC activation `f8c29fb8`, `4d54dfa9e`, `116fb6a2a`, EVOL `be77b3a5`/`7a4f279fd`/`1dd76933`, TES′, EVIDENCE′, AUDIT_EXECUTION, IIAC_CERT, B6_ACCEPTANCE, ACCEPTED_BASELINE, FORMAL_CLOSURE. Each pin is a separate commit that cites the SHA of an earlier commit (non-self-referential).

```text
P0 W1 merge PRs → TES′                         (code)
P1 evidence publication → EVIDENCE′            (docs, W1-06)
P2 IIAC phase 1 (record+manifest+certification, no CLOSED) → IIAC_CERT_SHA
P3 pin B6_ACCEPTANCE_PUBLICATION_SHA            (EVOL pattern)
P4 OD-ENT-00/01/02 records; baseline promotion → pin ACCEPTED_BASELINE_SHA
P5 certifying FREEZE bundle                     (docs/evidence)
P6 FRZ-AUD row promotions                       (dedicated commits)
P7 Gate A → OD-ENT-08 → closure phase 1 (roadmap headline, FRZ-FRZ-11, status, WBS, dossier)
P8 pin FORMAL_CLOSURE_PUBLICATION_SHA → Gate B
P9 handoff update (W6)
```

Commit trailer `Ref: governance/tickets/T-IGX-024-x.md` on commits that touch `intergrax/ agents/ applications/ scripts/`. Commits and pushes are mandatory at each publication step. The PR flow follows the repo precedent (merged PRs #395–#398). Both IIAC and EVOL gates hard-code the pass-record-before-CLOSED ordering.

## 15. Rollback, REJECT and corrective-batch procedure

- **Auditor FAIL/CONDITIONAL:** write findings into `.tmp/session/…`, publish nothing, open a corrective ticket, fix by the minimal diff in S1, requalify only the delta (§13), run a new auditor session. Maximum 2 loops; then stop and report to the operator with the findings table.
- **Gate A/B fail:** treat as a blocker. Never edit the guard to pass; fix the underlying cause (or file a defect).
- **Published error:** correct with a new append-only commit (a ladder `SUPERSEDED` row). Don't rewrite history or force-push (the ruleset blocks it anyway). If a premature CLOSED string reaches `development`, revert it by a PR revert commit and record the incident in the ledger.
- **Signing/delegation failure** (expired, unknown session, missing `Delegation:` line): STOP and ask the operator. Do not mint a record.
- **Test pins break on promotion:** the corrective batch updates the tests in the same PR as the status change (the T-IGX-022-A precedent), by branching on state, not by deleting the assertion.
- **Session end:** `git branch --show-current` and `git symbolic-ref HEAD` must succeed; restore `development` if detached; remove own worktrees (`git worktree remove`), not other people's.

## 16. Final closure preflight checklist (Gate A → B)

- [ ] `origin/development` tip recorded; all lineage SHAs are ancestors (W0-02).
- [ ] `verify_iiac_policy_activation --require-in-force` → IN_FORCE `IIAC-2026-11-05.1`.
- [ ] EVOL-X: ladder `FORMAL_CLOSURE_PUBLICATION_SHA=be77b3a5…`, handoff `7a4f279fd…`, `protocol_03=SUPERSEDED_BY_IIAC_B6_ACCEPTANCE`; OD-ENT-00 recorded.
- [ ] IIAC manifest valid (`iiac-v1`, 20 required fields), distinct sessions, handoff record present, verdict PASS/PASS_SCOPED, limitations enumerated.
- [ ] Pass record exists, was published in an earlier commit than the CLOSED text; `B6_ACCEPTANCE_PUBLICATION_SHA` pinned.
- [ ] `ACCEPTED_BASELINE_SHA` pinned; OD-ENT-01 recorded; certifying FREEZE run green vs it; freshness assertion true.
- [ ] Matrix C 75/75 QUALIFIED with cited evidence (the six closed); GAP-RC-OD-TRUST CLOSED; no other GAP-RC regresses.
- [ ] FRZ-AUD-01..10 none OPEN; row 10 has an explicit route disposition; OD-ENT-02 recorded; OPEN-BP-01 registered if compensating.
- [ ] CI: PR head `ent-audit-fast` + `ent-audit-pr-gate` `success`; nightly FULL on TES′ `success` (run IDs recorded).
- [ ] OD-ENT-08 recorded; the roadmap does **not** yet contain the CLOSED headline (Gate A) / does contain it exactly (Gate B).
- [ ] Gate B: `FORMAL_CLOSURE_PUBLICATION_SHA` pinned; remote tip equals the local tip.

## 17. ARCH-FREEZE handoff contract (W6)

The handoff doc (`ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md`) must contain, each as an exact SHA or path:

- TES′, EVIDENCE′, the prior TES/B5/re-audit (`89c76eb7`, `30085953`, `0e538269`) as history.
- The IIAC manifest, the audit report, the findings and digests, and the `audit_execution_sha`.
- The B6 acceptance, the accepted baseline, the certifying FREEZE evidence bundle.
- The FRZ-AUD-01..10 matrix with per-row verdicts and limitations.
- Records for OD-ENT-00/01/02/08, the delegation record, and the assurance level `process_enforced`.
- The enforcement disposition: route (`compensating`), the control list, **OPEN-BP-01** (owner, trigger, risk, residual).
- Explicit non-claims: no equivalence to live protection; PROD-Q live, M-09H NOT_MEASURED and vendor live qualification unchanged; FRZ-EVOL-07 qualification-layer only; ENT does not close ARCH-FREEZE.
- The EVOL predecessor pins and the D10 attestation reference.
- The `FORMAL_CLOSURE_PUBLICATION_SHA` and the final remote tip.

## 18. Exact instructions for the subsequent autonomous sessions

Common preamble for every session: use `agent-env/rules/*` always-on rules; state the preflight (read/edit/tests); stay on `development` or the approved branch; one `uv` process; artifacts only under `.tmp/session/<task>/`; commit trailer `Ref:`; never fabricate SHAs, timestamps, signatures, session IDs, or test results; if the read scope grows beyond the list, STOP and ask.

- **S0 Orchestrator (this session after approval):** execute W0 only; commit W0-04/05 (docs); report the matrix; scaffold T-IGX-024-A; hand W1 to S1. Maintain the ledger (the overlay + status JSON).
- **S1 Implementer (`/intergrax-micro-implement` or `/intergrax-hep-step` per lane):** read scope = the lane's file list in §7 plus `baseline.py`/`ent_audit_x_formal_closure_preflight.py` and tests. Implement D-1/D-2/D-3/D-5/D-6 exactly as in §4. Minimal diffs. Positive and negative tests per defect. Serial verification. Open PRs A→B→C, read the CI runs, then requalify at TES′ and publish evidence (W1-05/06). Don't declare PASS. Hand over the SHAs.
- **S2 Auditor (`/intergrax-independent-audit`, new chat):** execute §10 only. Read-only on product paths. Output to `.tmp/session/ent-w2-audit/`. Ask for the handoff record first. Report PASS/PASS_SCOPED/CONDITIONAL/FAIL with findings.
- **S3 Certifier (new chat):** validate the manifest and role separation; publish phase 1; pin; later publish P4–P8 in order, gating each on the preceding checks. Never combine pass-record creation and the CLOSED text in one commit.
- **S4 Signing session (new, dedicated):** act only after the operator's delegation record exists and cites its path@sha. Write the four `Approved … operator-delegate` lines with real timestamps; nothing else.
- **Operator checkpoints:** approve this plan and the branches; sign O-1; authorise S4's start; (optional) enable live protection, which flips OD-ENT-02 to LIVE; review the terminal dossier.

## 19. Completeness self-audit

| Requirement | Covered by |
|---|---|
| 75 Matrix C | 69 by evidence (M-03, re-checked in W2); 6 by W3-02 (D-2 + E-2) |
| 85 checks | M-02, W1-05, W2 step 5 |
| FRZ-AUD-01..10 | §6, W4 |
| Gaps | GAP-RC-OD-TRUST W3-02; BLK-CI-ENFORCE W3-03/04; BLK-ROADMAP-ENT-DEP OD-ENT-00; BLK-ENT-B6 W2; BLK-EVOL-B6/FRZ/HANDOFF closed by the EVOL closure (confirm in W0-03, which also reads the FRZ-EVOL rows); GAP-P-04 closed by `7a4f279fd`; GAP-P-05 W3-03; GAP-COV by the 85 qualification |
| Operator decisions | §9 |
| No history reopened | Class 1 and 7 receive no new tasks; EVOL untouched |
| EVOL not repeated | M-06/W3-01 validate only |
| No competing SSOT | single status overlay with a declared authority (§2) |
| IIAC replaces human-only gates | §2, D-3, §10 |
| Both OD-ENT-02 routes | §12, W3-03/04 |
| Independent mechanism per gate | B6 (W2), baseline (replay), FRZ (W4-02), OD-ENT-02 (W3-03), closure (Gate A/B + certifier) |
| Attainable without forged authority | the only unforgeable inputs are O-1 and the real sessions; every OD is a real delegated line |
| Handoff exact-SHA completeness | §17 |

**Genuinely open / unverified (owner → resolution):**

- ~~Mode C semantics, role-separation registry needs, D-7, ENT preflight as a material path, FRZ-EVOL promotion~~ — **resolved in W0-03/W0-06, see §21.**
- Amendment 01's unaudited status: it is not mechanically enforced and independence is not provable from git. It is accepted as a declared `process_enforced` limitation. The operator confirms in O-1; a stronger mechanism is outside ENT scope.
- The 94 CI skips (S1/S2 → W1-05, §10 step 4).
- The ticket-ID collision pattern seen in the delegated record (T-IGX-023-A is used for both an EVOL ticket and a TLX reference): confirm the numbering at W0-07.
- The final length of the OD-ENT-02 control list is set by the package, not by this plan.

## 20. Verification of this plan (end to end)

After approval, W0 executes first and its outputs are the check on this plan:

1. `git rev-parse HEAD` equals `git ls-remote origin development` after the fast-forward, and the ancestry commands pass.
2. The status overlay reproduces the §4 statuses from the repository (any divergence is reported before W1 starts).
3. W1 exit: T-01…T-08 green at TES′, with CI run IDs recorded.
4. W2 exit: T-09/T-10 green on the published manifest.
5. W5 exit: T-12 Gate B, T-14, T-16 and T-17 green, and the roadmap headline matches the target string exactly.

## 21. W0 execution log (2026-11-08)

**W0-01/02 done.** Local `development` fast-forwarded to `ba07ebaa34d04633bd60510f68c18346b1ef160d` (= `origin/development`). TES `89c76eb7`, B5 `30085953`, re-audit `0e538269`, IIAC activation `f8c29fb8`, EVOL closure `be77b3a5`, EVOL handoff `7a4f279fd`, EVOL B6 `1dd76933` are all ancestors of HEAD. Highest ticket: `T-IGX-023-A`, so `T-IGX-024-A` is free.

**W0-03/06 findings (read-only):**

1. **FRZ-EVOL-01..10 are already promoted** in the FRZ checklist (L273-282): 02-05 `PASS`; 01, 06, 08, 09, 10 `PASS — SCOPED`; 07 `PASS (EVOL-X SCOPED)`. The note "FRZ-EVOL OPEN until EVOL B6" is stale. No EVOL residual blocks ENT. The limitations LIM-01..07 must be carried into the ENT handoff.
2. **`MATERIAL_GOVERNANCE_PATHS` has exactly six entries** (`iiac_activation_gate.py:40-49`): `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py` and the IIAC protocol. **`ent_audit_x_formal_closure_preflight.py` and the two OD-ENT-02 verifiers are not material**, so lane C needs no activation pathway. Lanes must not touch the six.
3. The `_CLOSURE` regex (`iiac_activation_gate.py:35`) is applied only to the activation record text (L198), so it does not interact with the roadmap closure headline.
4. **`verify_iiac_role_separation.py` needs no registry**: it checks session inequality and the product-path diff between `audited_implementation_sha` and `audit_execution_sha`. It only *warns* unless `--strict-product-diff` is passed, so **the ENT audit and certifier must always pass `--strict-product-diff`** (added to T-10). No `role_registry.yaml` change is needed.
5. **IIAC artifact layout (EVOL precedent), adopt for ENT:** `iiac_evidence/ENT-AUDIT-X/{audit,certification,b6_acceptance,operator_decisions}/`, the manifest at `certification/iiac_audit_session_manifest.ENT-AUDIT-X.json`, the pass record at `ent_audit_x/ENT_AUDIT_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` (plus a copy in `audit/`). The existing preflight already expects that pass-record path; only its manifest glob is wrong (D-3b).
6. **D-3(c) is a fail-open defect**, not a parsing nit (see §4). Lane C needs a negative test: a ladder with `*unset*` pins must make the preflight fail.
7. **D-6 parser concern closed** and **D-7 verdict recorded** (see §4).
8. **Mode C = adversarial** (see §10 step 4).
9. `run_enterprise_audit.py` has subcommands `run` (flags incl. `--dry-run`, `--non-certifying`, `--bootstrap-candidate`, `--evidence-bundle`, `--since-sha`, `--check-id`) and `promote-baseline` (`--od-record`, `--baseline-dir`, …). The exact flags are in L102-160 of the script; the `--help` output has not been executed (no process started in W0).

**Still open after W0-03/06:** (a) W0-04/05 commits and the T-IGX-024-A ticket (awaiting the operator "Approved" line); (b) a stale-text patch list limited to non-material files; (c) the 94 CI skips (W1-05).
