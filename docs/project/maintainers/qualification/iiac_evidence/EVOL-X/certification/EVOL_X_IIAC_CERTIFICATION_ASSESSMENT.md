# EVOL-X — IIAC certifier assessment

> **Update after operator authorization (P1/P2 only, 2026-11-05):** P1 published as `4dee0050d1ab2d353c982da23cc82439bd79a53a` (committed blobs re-verified: 51/51 + pass record). B-1/B-2/B-3 are resolved by the operator's approval (implementer id handled as an approved historical exception — not invented; certifier commit authority from the operator, not the registry). The completed manifest passes `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff`. **B-4 and B-5 remain open**, so B6 acceptance, FRZ promotion and formal closure are NOT performed. The sections below are the original certifier reasoning; where they say "held/not published" for P1/P2, read the record `EVOL_X_IIAC_CERTIFICATION_RECORD.json` for the final state.

**Role:** `certification-governance-agent` · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE` @ `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`)
**Baseline HEAD:** `06a459afbab3aa8cea89e4118865d55e7b537a34` (= `origin/development`) · **Date:** 2026-11-05
**Independent auditor verdict (unchanged):** `PASS_SCOPED` — not upgraded, not rewritten.
**Certifier outcome:** `EVOL-X — INDEPENDENT AUDIT PASS_SCOPED — CERTIFICATION PARTIAL / EXACT BLOCKERS`

Status of this file: certifier working record in the working tree. Nothing here is committed, nothing is a closure claim, no FRZ row, ladder row, roadmap row or operator-register row was changed.

## 1. Integrity of the audit package (verified, read-only)

| Check | Result |
|---|---|
| `EVIDENCE_INDEX.json` vs bytes on disk | 51 indexed files, 0 digest/size mismatches, 0 unindexed, 0 missing |
| Digests quoted in handoff / manifest draft | dossier `88f9f427…77c6`, scope `24d6bcb1…a9ed`, test matrix `4ec65768…7d3e`, findings `d0e7b574…e95f`, FRZ matrix `bc9f6793…a6c1`, Part VII `c1bb37b4…dbad` — all equal the index |
| Pass record, two copies (`evol_x/` and `iiac_evidence/EVOL-X/audit/`) | byte-identical, sha256 `d9d17505…4cca`, 0 CRLF |
| Git ancestry | PLAN → TES → FINAL → activation → HEAD all hold; TES, FINAL, activation, internal-B6 `ee2c89a0`, IIAC-GOV-1 `a5fb6cb5` all ancestors of HEAD |
| Tree binding | MANIFEST `parent_test_execution_tree` `3244d4ea…` = `git rev-parse TES^{tree}` |
| Drift | TES→FINAL: 0 product/test/testing_support files. FINAL→HEAD: 0 bundle changes. TES→HEAD: 0 EVOL-X-scope changes, 34 product files (ENT-AUDIT-X work, outside audited tree) |
| `FRZ_EVOL_CANONICAL_REQUIREMENTS` vs auditor matrix labels | 0 problems |

Not re-run (by design): the 290-test replay. No implementation change and no finding invalidates the accepted replay (TES/FINAL unchanged, EVOL-X scope unchanged to HEAD). I read the auditor's probe sources/logs; I did not re-execute the probes.

## 2. Findings — certifier's independent disposition

| ID | Auditor | Certifier determination |
|---|---|---|
| F-C-01 | LOW | **Confirmed by reading source.** In `p03.verify_mode_b` the filter uses `replay_outcomes.get(n)` but the message f-string dereferences `replay_outcomes[n]`; a node missing from the replay JUnit raises `KeyError`, which the CLI does not catch (only `OSError/CalledProcessError/ValueError`), giving a traceback and exit 1. **Fail-closed holds** (no pass path; non-zero exit; the 8 plan Part V-I negatives, which concern canonical outcomes, exit 2 and pass in the 52-test integrity suite). It is **not** graceful validation: diagnosis is lost and the exit code is 1, not the documented 2. Must be reported as "rejection by unhandled exception". Not a blocker for the scoped verdict; correction needs a new audited SHA. |
| F-C-02 | LOW | **Documentation-conformance gap, not missing mandatory certification evidence.** Plan Part II-B says to document the PLATFORM exclusion in the FRZ-EVOL-06 footnote; the FINAL row's footnote is empty. But FRZ-EVOL-06 DONE = "no PLATFORM claim" (Part VII-B), the Mode A footnote-required set is 07/08/10 only, and the scope is asserted in a behavioral test, the pass record and the matrix. Bundle is digest-frozen (cannot be edited without a new FINAL). Compensating control: the verbatim LIM-01 footnote MUST appear in any FRZ-EVOL-06 text. Row stays SCOPED. |
| F-C-03 | MEDIUM | **Not silently downgraded. Does not invalidate the accepted scope; does invalidate any wider claim.** See §2.1. |
| F-C-04 | LOW | Confirmed as stated: M-03 shows internal consistency on self-surface arcs under O-X, not discrimination. LIM-04 travels with FRZ-EVOL-06; no widening of M-03. |
| F-C-05 | MEDIUM | Preserved. OD-EVOL-13 (`OPERATOR APPROVED`) authorizes `AUTOMATION_FIRST_SCOPED` with M-09H `NOT_MEASURED`. M-09A is tool-side attribution only; never convertible into human SLA PASS. FRZ-EVOL-08 text is change-time diagnostics only. |
| F-C-06 | MEDIUM | Preserved. OD-EVOL-05b Option 1 (`SIGNED / APPROVED`): qualification layer only; `ad_layer_alone_succeeds=True` means production `ActivationService.rollback` is not gated. Any production-enforcement claim is forbidden (Part X). |
| F-C-07 | LOW | Accepted as LIM-06; data-file exemption stays disclosed. |
| F-C-08 | PROC | Confirmed and **escalated to blocker B-4** (§4). Strict role-separation fails with `audit_execution_sha = HEAD`; FINAL is the valid precedent-consistent value. |
| F-C-09 | PROC | **Open; blocker B-1/B-2.** I did not invent identities or sign for the operator. |
| F-C-10 | INFO | Accepted (disclosed auditor tooling error, superseded probe). |
| F-C-11 | INFO | Out of scope; ARCH-FREEZE must not infer a clean platform collect. |

### 2.1 F-C-03 — test-ownership omission (4,161 test modules, 2,803 suites)

What the claims actually say:
- FRZ-EVOL-01 literal: *semantic change-impact assessment exists for in-scope platform surfaces*. DONE: behavioral PASS. Production-consumer detection was verified complete in the sample (0 of 75,380 closure elements missed in 60 real changes).
- FRZ-EVOL-06 literal: *regression test selection from impact graph with mandatory safety suites*, under policy `EVOL_CONSERVATIVE_FULL` = T1 ∪ T2 ∪ T3 ∪ Adj. Mandatory suites are always included; unknown impact → conservative.
- Nowhere in the plan, checklist rows or metric set (M-01..M-11) claims complete enumeration of every impacted test in the platform. `TESTED_BY` is listed as a mandatory REL, and F-P1-02 (under-selection) is closed by the oracle probe, not by a completeness theorem.

Why the omission does not invalidate the accepted scope:
- The auditor's under-selection oracle is restricted to the EVOL selection universe (`mandatory ∪ pop_a ∪ evol_manifest ∪ adjacent`) and uses the AST reverse closure *including* helper modules: 0 under-selection over 56 stratified real paths + 3 unknown paths conservative. The sampled omitted suites are platform tests (`tests.integration.*`, `tests.conformance.*`), i.e. the T4 population that EVOL-X explicitly does not claim (Part II-B).

What it does invalidate / residual risk (explicit):
1. The probe log does **not** partition the 4,161 omitted modules by membership in the EVOL universe; the "0 under-selection" statement is a 56-path sample, not an exhaustive or structural proof. Direct-import-only `TESTED_BY` can structurally under-select a T2/T3 test reached via a helper/conftest. Therefore: no unrestricted PASS on FRZ-EVOL-01/06, LIM-03 mandatory in both rows.
2. Any statement that EVOL-X provides complete test-impact coverage, or that its selection is safe for T4/platform-wide narrowing, is **unsupported** and must be struck. ENT-AUDIT-X / SCENARIO-READINESS-X must not rely on EVOL-X test ownership (or must record explicit non-reliance — the auditor's own closure criterion).
3. Classification kept: MEDIUM / TRACKED_FREEZE_DEBT, "blocker for any claim wider than EVOL_CONSERVATIVE_FULL", not a blocker for the scoped verdict.

## 3. FRZ-EVOL-01..10 criterion matrix (inputs, no promotion)

All ten checklist rows remain `OPEN`. Promotion text may only be written in the W7-05 formal closure commit and only if every §4 blocker is cleared.

| FRZ | Normative requirement | Independent evidence | SHA | Accepted scope | Limitations | Certifier disposition if gates clear | Operator decision | Remaining blocker |
|---|---|---|---|---|---|---|---|---|
| 01 | semantic change-impact assessment exists for in-scope platform surfaces | 11/11 tests, M-01 40/40, AST-closure probe | TES/FINAL | production-consumer impact, in-scope surfaces | LIM-03 | PASS — SCOPED | none beyond closure authority | B-1, B-4 |
| 02 | graph coverage meets M-02 threshold | 8/8, M-02 11528/11528, M-02C 0/11528 | TES/FINAL | as stated | — | PASS | — | B-1, B-4 |
| 03 | blast radius explainable per scenario/app | 6/6, S1 p+n, M-06 0/1 | TES/FINAL | as stated | — | PASS | — | B-1, B-4 |
| 04 | contract/schema evolution detects consumers (COMPAT applied) | 19/19, S2/S5 p+n, M-07 4/4 | TES/FINAL | as stated | — | PASS | — | B-1, B-4 |
| 05 | workflow evolution rules explicit | 17/17, S3 p+n, WM 128 enumerated | TES/FINAL | as stated | — | PASS | — | B-1, B-4 |
| 06 | test selection from impact graph with mandatory safety suites | 30/30, S6 p+n, M-03 2082/2082, M-04 8/8, M-05 2/2, 56-path probe | TES/FINAL | `EVOL_CONSERVATIVE_FULL` only | LIM-01, LIM-03, LIM-04; F-C-02 footnote carried verbatim: *EVOL_CONSERVATIVE_FULL scope only; not PLATFORM_CONSERVATIVE_FULL* | PASS — SCOPED | OD-EVOL-10 (signed) | B-1, B-4 |
| 07 | deployment/rollback qualified or unsafe refusal explicit | 19/19, S7 p+n, M-08 2/2, DM 384 enumerated | TES/FINAL | qualification layer only; Part X W7-03 determination by auditor accepted as sound | LIM-05; *production ActivationService.rollback durable-state enforcement is NOT claimed (OD-EVOL-05b Option 1)* | PASS (EVOL-X scoped) | OD-EVOL-05b (signed) | B-1 (auditor acts as W7-03 delegate only if handoff attested), B-4 |
| 08 | diagnostics: what changed / failed / which deploy | 17/17, M-09A 3/3, M-09T 3/3, M-09H NOT_MEASURED | TES/FINAL | change-time diagnostics | LIM-02; *incident/deploy correlation owned by ENT-AUDIT-X; M-09H not measured (OD-EVOL-13)* | PASS — SCOPED (Option A text) | OD-EVOL-13 (approved); OD-EVOL-12 → certifier gate (see §4 B-5) | B-1, B-4, B-5 |
| 09 | extensibility without core modification (M-10) | 11/11, S8/S4 p+n, M-10 0/2 | TES/FINAL | 9-surface frozen-core registry | LIM-06 | PASS — SCOPED | — | B-1, B-4 |
| 10 | post-freeze triggers and freshness enforced | 4/4, M-11 9/9 | TES/FINAL | executable T1..T9 + freshness counter | LIM-07; *scenario-wave audit owned by ENT-AUDIT-X* | PASS — SCOPED (Option A text) | OD-EVOL-12 → certifier gate | B-1, B-4, B-5 |

No row is unsupported at the evidence level; every row is blocked at the authority/procedure level (§4). No global FRZ PASS, no PROD-Q claim, no production-rollback claim.

## 4. Authority reconciliation and exact blockers

Current normative state: IIAC `IN_FORCE` (`--require-in-force` exit 0). IIAC-OD-01 scope: internal independent agent certification authority for EVOL-X and ENT-AUDIT-X; it does not close EVOL-X. `ECAP-P4-EVOL-SCOPE-AUTO-B6` is superseded/not in force. Internal B6 / OD-IC-01 / Protocol 03 automated record are historical evidence only.

| Item | Disposition |
|---|---|
| OD-EVOL-11 (named human auditor) | Replaced by `auditor_session_id` + operator-attested handoff (migration matrix). **Not satisfied until the handoff is attested** (B-1). |
| OD-EVOL-12 (FRZ-08/10 wording) | Register row still `PENDING`. Matrix maps "operator FRZ promotion" → certifier per-row gate; the matrix does not name OD-EVOL-12. Evidence for rows 08/10 supports the Option A scoped text. Needs operator confirmation that IIAC-OD-01 covers it, or a signed row (B-5). I did not edit the register. |
| OD-EVOLX-08 (parent closure) | Register row still `PENDING`. Matrix names only OD-ENT-08 (ENT); EVOL mapping is by analogy in the auditor dossier. Same handling (B-5). |
| FRZ-EVOL-01..10 | Evidence-level eligible as in §3; all `OPEN`; no promotion. |
| B6 independent acceptance (P3) | **Not issued**: IIAC requirements not all satisfied (B-1, B-2). `B6_ACCEPTANCE_PUBLICATION_SHA` stays `PENDING`. |
| W7-05 formal closure / W7-06 ENT-AUDIT-X handoff | **Not performed** — depend on P3 and a passing closure preflight. |

**Blockers (exact):**

- **B-1 Operator handoff attestation.** Mandatory per IIAC protocol. The auditor's draft record has every operator checkbox and the signature line empty; I prepared a certifier handoff record (unsigned) and did not tick or sign anything. (Note the IIAC-GOV-1 precedent also left the signature line pending; I do not treat that as permission to skip it.)
- **B-2 Manifest cannot be completed truthfully.** `implementation_session_id` is not recorded anywhere in the repository (operator must supply). `certification_publication_sha` can only exist after a publication commit and must then be bound in a separate commit (non-self-referential). Validators currently FAIL on the unmodified draft (`PENDING_CERTIFIER`, `certification_publication_sha` not 40-hex).
- **B-3 Role registry gap.** `governance/agent-control-plane/role_registry.yaml` has no `certification-governance-agent` entry (default role `orchestrator` has `may_commit: false`). I cannot confirm that this session is permitted to commit. Hence no commit/push was made. Operator must confirm authorization or add the role through the normal ticket path.
- **B-4 Closure preflight enforces superseded gates and cannot pass on valid IIAC evidence.** `scripts/qualification/evol_x_closure_publication_preflight.py` (run at HEAD, exit 1) still requires: EVOL IIAC manifest JSON in `evol_x/`, `b6_acceptance_publication_sha` in `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json`, ladder `B6_ACCEPTANCE_PUBLICATION_SHA`, OD-EVOL-12/OD-EVOLX-08 not `PENDING`, OD-EVOL-11 auditor named; and it accepts the pass record/manifest by filename glob only (no content validation). Per the protocol I did **not** bypass or patch it, and I will not certify my own gate change. **Required: independent corrective implementation** under a new ticket (implementer role, new audited SHA) that (a) validates the IIAC manifest content via `verify_iiac_audit_manifest`/`verify_iiac_role_separation --strict-product-diff`, (b) maps OD-EVOL-11/12/OD-EVOLX-08 and B6 SHA to the IIAC gates, (c) keeps legacy checks for non-IIAC mode; then an independent re-audit of that change. Its closure criterion: preflight PASS only when the manifest validates, not on file presence.
- **B-5 OD-EVOL-12 / OD-EVOLX-08 authority wording.** See table above; operator confirmation or signed rows required before FRZ-08/10 and parent closure text.

## 5. Publication sequence status

| Step | Status |
|---|---|
| P1 independent audit publication (byte-preserved) | **Held.** Auditor precondition #1 (operator-attested handoff) and B-3 unmet. Bytes verified unchanged; ready to publish unmodified. Publication SHA: none. |
| P2 certification record (`PASS_SCOPED`) | **Drafted, not published** (`EVOL_X_IIAC_CERTIFICATION_RECORD.draft.json`, `iiac_audit_session_manifest.EVOL-X.certifier_draft.json`). `certification_publication_sha` unbound. |
| P3 formal B6 acceptance | **Not issued** (B-1, B-2, B-4). |
| P4 FRZ + ODs | **No change**; all rows OPEN; ODs untouched. |
| P5 formal closure / ladder / roadmap / W7-06 | **Not performed.** |

### Ready-to-run sequence once B-1..B-5 are cleared
1. Operator signs `IIAC_SESSION_HANDOFF_RECORD_EVOL_X_CERTIFICATION.md` and supplies `implementation_session_id`; authorizes certifier commits (B-3).
2. Corrective preflight ticket merged and independently re-audited (B-4).
3. Commit A (P1): the unchanged 51-file audit directory + pass record under `evol_x/` (Ref ticket). Re-verify all digests from the commit blobs. No closure claim in this commit (EBH-7).
4. Commit B (P2): complete manifest + `EVOL_X_IIAC_CERTIFICATION_RECORD.json` with `certification_publication_sha` = Commit A; run both validators and the corrected preflight.
5. Commit C (P3): `B6_ACCEPTANCE_PUBLICATION_SHA` on ladder + governance JSON. Commit D (P5): per-row FRZ text from §3, roadmap, `FORMAL_CLOSURE_PUBLICATION_SHA`, W7-06 handoff.

## 6. Validator and preflight runs (this session, HEAD `06a459afb`)

See `validator_runs_certifier.txt`.
