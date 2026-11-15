# MCP-SERVER-X RC — certifier preflight

| Field | Value |
| --- | --- |
| **Campaign** | MCP-RC corrective campaign (residuals R-P8-01, R-P7-01..06 of the closed MCP-SERVER-X stage); tickets `T-IGX-038-A..F` |
| **Role** | Certifier, **preflight** (precedent: `MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`, row "Certifier": preflight vs activation certifier). IMPLEMENTER != AUDITOR != CERTIFIER. This session implemented and audited nothing |
| **Session** | `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` (designation; distinct from the implementer and both auditor sessions), branch `mcp-rc-certifier`, base = `mcp-rc` head |
| **Date** | 2026-11-13 |
| **Verdict** | **`CERTIFIER_PREFLIGHT_PASS_SCOPED`** (scope limits in section 3) |
| **Machine-readable** | `iiac_evidence/PDC-X/MCP-RC/certifier/certifier_preflight_result.json`; logs, scripts and reports under the same folder |

This preflight is **not** the activation certification, **not** an IIAC rebind, **not** stage closure and **not** a production-readiness claim. FRZ counter stays 225; no FRZ status changed; no P8 certification history touched.

## 1. Pin chain (derived from git, not from the dossiers)

| Role | SHA | Verified |
| --- | --- | --- |
| Base (`development` merge of PR #493) | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` | ancestor of `mcp-rc` head |
| First-audit audited / test SHA | `93a089d3cca941592f1ad4f65435333c86da82e5` | on `mcp-rc` and `mcp-rc-audit` |
| First-audit execution base (`audit_execution_sha`) | `f3078b293ecee402583b3d6ad9ef5aaeff0b2d1d` | `93a089d3c..f3078b293` touches 5 docs/evidence files, no product, test or script path |
| First-audit branch head | `4a459ed0be798ca13e292920ba081874244ee58e` (`mcp-rc-audit`, verdict PARTIAL) | adds only `docs/` |
| Last product change | `b0e19bf61b346e6941b2e190b703c6922e708b34` | `git diff --name-only b0e19bf61 1b5f26f24 -- intergrax applications agents` is **empty** (0 paths) |
| Re-audit execution base | `f1dde5a3e19924b82e902be18f11a21b64b822ba` | `b0e19bf61..f1dde5a3e` touches 4 docs/evidence files, 0 under `intergrax/ applications/ agents/ tests/ scripts/` |
| Re-audit branch head | `efa74c54536567d138b27a2d96dbb15b35f103c2` (`mcp-rc-reaudit`, PASS_SCOPED) | on top of `f1dde5a3e`, adds only `docs/` |
| Tests and harness after the audits | `b98dc77196cee10a9f06323bd47c4b69d0f1baa6` | the only non-doc change after `b0e19bf61` (see below) |
| Record head certified here | `1b5f26f249beaafe007ffff2f1e396ae393a07dc` | `mcp-rc` tip; this preflight's checkout `HEAD` before my commit |

Changed after the audited product SHA `b0e19bf61` (10 paths, all non-product): two mutation reports and two contract logs under `iiac_evidence/MCP-SERVER-X/rc_qualification/`, `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md`, `MCP_SERVER_X_RC_QUALIFICATION_PACKAGE.md`, `scripts/qualification/mcp_rc_mutation.py` (mutant M18 only, +8 lines), `tests/unit/applications/test_mcp_rc_audit_corrections.py` (+24 lines, one test: RA-01).

Re-audit scope: `reaudit/audit_scope.json` states `audit_execution_base_sha = f1dde5a3e`, `audited_implementation_sha = test_execution_sha = b0e19bf61`, `base_sha = 6a6279a49`, `first_audit_implementation_sha = 93a089d3c`; the 23 combined product paths equal `git diff --name-only 6a6279a49 1b5f26f24 -- intergrax applications agents` (8 facades + 15 `intergrax/` paths) and the 23 `ACT-N11` paths reported by the activation gate (section 2, row 16). The first audit's scope file carries the same base and audited SHA.

## 2. Checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Pin chain derived from git (section 1) | PASS | `logs/guardrails_git_checks.txt` |
| 2 | No product diff `b0e19bf61..1b5f26f24` | PASS | 0 paths |
| 3 | Audit execution bases docs-only; re-audit `audit_scope.json` matches git | PASS | section 1 |
| 4 | Sessions distinct | PASS | audit 1 `MCP-RC-G6-AUDITOR-90803ebc-…`, re-audit `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-…`, implementer `T-IGX-038-A-F-MCP-RC-IMPLEMENTER`, certifier (this) differs from all three; `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff` PASS on both manifests (`logs/certifier_verify_iiac_manifests.log`). The check is a `process_enforced` attestation: all commits of the campaign carry one git identity, so independence cannot be derived from authorship |
| 5 | Every non-merge commit `6a6279a49..1b5f26f24` (22) has `Ref: governance/tickets/T-IGX-038-[A-F].md` | PASS | `logs/guardrails_git_checks.txt` |
| 6 | Six plan digests equal `scripts/governance/plan_digest.py` on `governance/tickets/plans/PLAN_T-IGX-038-A..F.md` | PASS | A `94b40ae2…`, B `a68e5c97…`, C `fa0f4440…`, D `c09933e0…`, E `fab8bc46…`, F `4dcc7adc…` all MATCH; each plan's last change precedes its approval (A plan 18:23 / approval 18:24, B/D/E/F plan 18:17 / approval 18:21, C plan 06:30 / approval 06:31) |
| 7 | Delegate approvals inside `DELEGATION-MCP-RC-1` | PASS | the three approval commits (`503d32e5e`, `ee719daa6`, `ed6feddaa`) change only the `Approved:`/`Delegation:` lines of tickets A..F; every approval cites `DELEGATION-MCP-RC-1.md@4ef1e0935`, which exists; all approvals precede the first product commit `a45d0df1b`; no other ticket, no IIAC activation line, no merge or ruleset touched |
| 8 | Mandatory contract, one pytest process, `PYTHONPATH` = my checkout, list passed from Python | PASS | **481 passed, 1 xfailed, 0 failed, exit 0, 665.16 s**, 39 files, imports resolved to my checkout; equal to the claimed 481 / 1. The single xfail is `test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]` (`strict=True`, TRANSFERRED). `logs/rc_contract_certifier_at_1b5f26f24.log` (one aborted earlier attempt, exit 4 before any test ran, was caused by my own `-p no:cacheprovider` flag clashing with the repo `cache_dir` setting; kept as `logs/aborted_attempt0_cache_dir_config_error.log`, not evidence) |
| 9 | Mutation harness reports match harness | PASS | harness `scripts/qualification/mcp_rc_mutation.py` defines 18 mutants (M1..M18); stored reports: 17 at `b0e19bf61` plus M18 at `b98dc7719` (harness diff after `b0e19bf61` = M18 only). My own full run at `1b5f26f24`: **18 of 18 killed** (`mutation/certifier_mutation_report_18_at_1b5f26f24.*`). Kill quality: the stored tail of M4 and M5 shows only teardown noise, so I re-ran both with full output: M5 fails on `assert 'outcome=cancelled' in '…outcome=outcome_unknown'`; M4 fails by `CancelledError`/`TimeoutError` (the governed cancel never arrives, the test's own wait expires), a behavioural kill but not an assertion text (`logs/m4m5_full.log`) |
| 10 | Audit manifest digests (`audit_report_digest`, `audit_scope_manifest_digest`, `test_matrix_digest`, `findings_digest`) | PASS | all 8 digests (two manifests) equal SHA-256 of the files as published on `mcp-rc-audit` / `mcp-rc-reaudit` (`logs/audit_manifest_digest_check.json`) |
| 11 | Findings dispositions | PASS_SCOPED | section 4 |
| 12 | Guardrails | PASS | `git diff --name-only 6a6279a49 1b5f26f24` matches none of: activation record, FRZ/freeze files, P8 files, `.github`, `scripts/governance`, role registry; no `D`/`R` entries (41 A, 40 M); roadmap and ARCH-FREEZE documents unchanged (0 diff lines); FRZ counter 225 unchanged; no ruleset change possible from the diff |
| 13 | No new MCP protocol capability | PASS | the same five tools are registered before and after (`list_agents`, `run_agent`, `list_catalog_tools`, `describe_catalog_tool`, `invoke_catalog_tool`); no added tool, resource, prompt, sampling or transport registration in the product diff; wire contract v1.1 adds error codes only (declared in `ADR_MCP_RC_01`) |
| 14 | No deleted, skipped or loosened test used to obtain PASS | PASS | across the 21 modified test/fixture files `def test_` 211 → 214, `assert` 431 → 431, `pytest.raises` 74 → 76, skip markers 2 → 2, `xfail` 0 → 1 (only the strict transferred case), no deselect. Four files show fewer asserts or raises; all four were read: assertions replaced by contract-driven stronger forms (untyped payload to `MCP_RESULT_BOUND_EXCEEDED` tool error, tautological `inspect.getsource` replaced by a behavioural check, `pytest.raises(Exception)` replaced by an HTTP 401, source-text scan replaced by AST plus clean-interpreter checks). Only one wait budget was widened: TL11 and L3-R1 subprocess start 15 s to 60 s (RA-07, disclosed) |
| 15 | Pin-only edits in auditor-authored dossiers | NONE MADE | see section 5 |
| 16 | `verify_iiac_policy_activation --require-in-force` at `1b5f26f24` | **FAIL, expected** | 23 × `ACT-N11 product path changed after the audited SHA` — exactly the 23 product paths; this is the by-design trigger until the PDC-P7-00 two-phase rebind (`logs/verify_iiac_policy_activation_at_1b5f26f24.log`) |

## 3. Verdict and scope limits

`CERTIFIER_PREFLIGHT_PASS_SCOPED`. The pin chain, independence attestations, ticket and delegation chain, test contract, mutation harness and guardrails hold on my own runs. PASS applies **only** under these limits:

1. **Audit coverage of the last test change.** The re-audit covers product SHA `b0e19bf61`. The RA-01 test and mutant M18 (`b98dc7719`) were written by the implementer after the audit; no auditor reviewed them. I verified them mechanically (test read in full, green on the product, M18 killed, contract 481/1) — a certifier check, not an audit.
2. **Stale audit-status text in implementer documents.** `MCP_SERVER_X_RC_QUALIFICATION_PACKAGE.md` ("Limits" section) still says the corrective-loop changes "have **not** been audited", and the status field of `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md` says "NOT AUDITED"; the delta re-audit (PASS_SCOPED at `b0e19bf61`) exists. Not edited by me (outside this role); to be corrected additively by the document owner.
3. **Open ownership conditions** AF-05 and AF-16 (section 4) are not resolved; the re-audit made a named owner/ticket a condition.
4. **Evidence is not yet integrated.** The audit dossiers, manifests and evidence live only on `mcp-rc-audit` and `mcp-rc-reaudit`; they are not on `mcp-rc` or the future PR head. Both manifests still contain placeholders (section 5).
5. **Not covered by any audit or by me:** CI contexts on a PR head, G-DOC-01..12 (P8 documentation defects), shared harness modules (G-ID-05), production host cancellation after the bound, real cross-loop ingress and real catalog child execution, real product factory composition (AF-05).
6. **Run environment:** shared Windows workstation under concurrent load (20 or more python processes during my runs); the contract passed without a start-budget failure; no Docker, vendors or production configuration used.
7. **Drift:** `origin/development` is 97 commits ahead of the base (`f319c428a`), touches none of the 23 MCP product paths and did not change the activation record; the head to be activated must be evaluated after integration, not at `1b5f26f24`.

## 4. Findings disposition (AF-01..AF-16, RA-01..RA-07)

Every finding has a recorded disposition in `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md` (sections "Audit follow-up" and "Delta re-audit follow-up"); AF-11..AF-14, AF-15/AF-16 and RA-05..RA-07 are recorded as groups. No claimed-closed finding lacks evidence: AF-01, AF-02, AF-03, AF-04, AF-10 and RA-01 are closed by code or tests and by the re-audit's judgement (CLOSED / CLOSED_IN_CODE); my own run adds M9..M18 killed and the contract green.

Still open or accepted-scoped (verbatim dispositions):

| ID | Disposition (verbatim) |
| --- | --- |
| AF-05 | "**ACCEPTED_SCOPED**: real product-factory composition (strict local-workspace app) stays unverified for MCP opt-in; restoration is owned by the strict-product test-support/scaffold owner together with the transferred `[product]` xfail. Not closed by this campaign" — re-audit: "ACCEPT_SCOPED … condition open: named owner/ticket" |
| AF-16 | "INFO, accepted; the `[product]` xfail owner/ticket must be named by the operator before certification" — re-audit: "ACCEPT_SCOPED … TRANSFERRED; condition open: named owner/ticket" |
| AF-06 | "**Documented**: the effective bound is `max_call_seconds` plus the host drain; … Not changed (the drain is the existing transport constant)" |
| AF-07 | "**Wording amended**: … It never creates a deadline authority, never reports success, and never changes the canonical run budget" |
| AF-08 | "**Documented**: do not use `LOCAL_DEV_EXPLICIT` behind a same-host proxy; the guard is a development convenience, not an access control" |
| AF-09 | "**Documented**: the label is evidence for audit and governance, not a gate; … prefer `execute_mcp_agent_task_with_resolved_identity`" |
| AF-11..AF-14 | "**Accepted as declared** (reference server defaults to LOCAL_DEV in fixtures; factories expose no `identity_mode` by design; `max_call_seconds` is a composition parameter not yet threaded through facades; `initialize`/`tools/list` unauthenticated on API-key hosts is pre-existing protocol behaviour)" |
| AF-15 | accepted (INFO) |
| RA-02 | "**Open, accepted as declared** (a host-authored `ToolError` is the host's own wire contract; no product change made)" |
| RA-03 | "**Open, accepted as declared** (the opt-out is a composition parameter of the engine; no in-repo host needs it)" |
| RA-04 | "**Open, accepted as declared** (wrong label only; no success is ever claimed)" |
| RA-05..RA-07 | "Accepted as declared" (INFO) |
| Other accepted-scoped | G-ID-05 (shared harness module, TRANSFERRED), G-DL-02, G-DL-03, G-DL-04, G-PR-01..04 (all `ACCEPTED_SCOPED` in the corrections record) |

Certifier observations (not new findings): ADR decision 6 still reads "raw exception text never reaches the wire" while RA-02 records that a host-authored `ToolError` keeps its text (the re-audit recommended documenting the exception; the ADR was not amended, only the corrections record carries it); `OD-MCP-RC-02` is recorded as an operator chat answer, "not a signed `Authorized:` line".

## 5. Pin-only edits

**None.** Both auditor manifests carry placeholders: `certifier_session_id = PENDING-CERTIFIER-MCP-RC`, `evidence_publication_sha` and `certification_publication_sha` = 40 zeros. I made no pin-only edit because (a) the manifests are not on my branch and I may not push to the audit branches, and (b) no evidence publication SHA exists yet: it is the SHA at which the audit evidence is integrated into the branch to be certified. Whoever integrates the audit evidence replaces the placeholders in a pin-only commit and lists it (P6 precedent: diff `2d3eceb6c..276c64305`).

## 6. What this preflight does not do

Not the activation certification; no edit of `IIAC_POLICY_ACTIVATION_RECORD.json`; no rebind; no `Approved IIAC policy activation` line; no merge, no PR, no push to `mcp-rc`, `mcp-rc-audit`, `mcp-rc-reaudit` or `development`; no stage closure; no FRZ promotion (225); no production-readiness claim; no check of GitHub CI contexts.

## 7. Remaining steps (orchestrator and operator)

1. **Integrate the audit evidence** (both dossiers, manifests, findings, scope, matrices, logs) onto the branch that becomes the PR head; replace the placeholders in a listed pin-only commit; correct the stale "not audited" statements (section 3, point 2) additively.
2. **(a) Activation line** `Approved IIAC policy activation …` by the operator or a newly commissioned delegate (`DELEGATION-MCP-RC-1` expressly excludes it).
3. **(b) Activation certifier** in a separate new session: two-phase update of `IIAC_POLICY_ACTIVATION_RECORD.json` inside the PDC-P7-00 window; `verify_iiac_policy_activation --require-in-force` must then exit 0.
4. **(c) Merge** only after the five CI contexts (`ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate`) are green on the pinned PR head (compare `headRefOid` with the pushed SHA).
5. **(d) Terminal verification** in a further separate session after the merge.
6. **Operator-owned items:** named owner/ticket for the transferred `[product]` scaffold xfail (AF-05, AF-16); decision D-10 about the closed PRs #475, #476, #478; owner for the P8 documentation corrections (G-DOC-01..12, which the corrections record deliberately does not apply to the P8 originals); optional ruling on RA-02/RA-03/RA-04 hardening.
