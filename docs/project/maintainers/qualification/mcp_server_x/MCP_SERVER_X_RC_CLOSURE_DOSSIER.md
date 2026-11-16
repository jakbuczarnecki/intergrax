# MCP-SERVER-X RC — closure dossier (MCP-RC, terminal verification on `development`)

| Field | Value |
| --- | --- |
| **Date** | 2026-11-15 |
| **Ticket** | `governance/tickets/T-IGX-038-F.md` (campaign tickets `T-IGX-038-A..F`) |
| **Verified tree** | `development` at `61531828f5728acc8f2ed357b4b835d7f5dc4ebc` (merge of PR #554 `mcp-rc`, head `b4d2a3c17d872a17e4fc49c70252d34138416094`) |
| **Author role** | Terminal verifier and dossier author. This session is not the implementer, not an auditor (G6 addendum 4), not the certifier, not the signer and not the activation certifier. |
| **Nature** | A record of verification and of open items. It is **not** an audit, **not** a certification, **not** an operator closure decision and adds **no** verdict to any manifest. |

## 1. Scope delivered

The MCP-RC campaign (MCP Enterprise Residual Remediation) changed 23 product paths (`intergrax/` and `applications/`), all on `development` at the verified tree:

| Ticket | Item | Content |
| --- | --- | --- |
| `T-IGX-038-A` | MRC-1 | identity and tenant admission, fail-closed MCP identity |
| `T-IGX-038-B` | MRC-2 | one bounded-result choke point and canonical MCP wire-error contract |
| `T-IGX-038-C` | MRC-3 | server-side deadline, cancellation, admission and resilience |
| `T-IGX-038-D` | MRC-4 | MCP invocation correlation id |
| `T-IGX-038-E` | MRC-5 | R-P7-02 test repair, behavioural tests and mutation qualification |
| `T-IGX-038-F` | MRC-6 | protocol classification, compatibility record, corrective documentation |

The first independent audit (verdict `PARTIAL`, findings AF-01..AF-16) led to a corrective loop (AF-01, AF-02, AF-03, AF-04, AF-10 closed per the delta re-audit). The product code of the 23 paths is byte-identical to the implementer's `b0e19bf61` (checked here, section 3).

## 2. Evidence index

| Evidence | Verdict / result | Location |
| --- | --- | --- |
| Independent audit (G6), first | `PARTIAL` at `93a089d3c` (superseded) | `MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md` |
| Independent delta re-audit (G6) | `PASS_SCOPED` (delta and combined), 0 HIGH / 0 MEDIUM / 4 LOW / 3 INFO new | `MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md`, `iiac_evidence/PDC-X/MCP-RC/reaudit/` |
| Certifier preflight (MCP-RC) | `CERTIFIER_PREFLIGHT_PASS_SCOPED` | `MCP_SERVER_X_RC_CERTIFIER_PREFLIGHT.md`, `iiac_evidence/PDC-X/MCP-RC/certifier/` |
| Independent audit **addendum 4** (PDC-P7-00, integrated baseline) | `PASS_SCOPED` (manifest value `CONDITIONAL`); combined `PASS_SCOPED` | `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_4.md`, `iiac_evidence/PDC-X/P7-00/addendum_4/` |
| Certifier preflight **addendum 3** | `CERTIFIER_PREFLIGHT_PASS_SCOPED` | `platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_3.md`, `iiac_evidence/PDC-X/P7-00/certifier/addendum_3/` |
| Evidence integration **round 5** (addendum 4) and **round 6** (preflight addendum 3) | evidence import only, no verdict, no certification | `iiac_evidence/PDC-X/P7-00/EVIDENCE_INTEGRATION_RECORD.md` |
| Activation certifier, **re-pin 2** | manifest `PASS` (scoped, certifier-authored from `PASS_SCOPED` audits); gate `IN_FORCE`, zero ACT-N11 lines at the phase-2 commit | `MCP_SERVER_X_RC_ACTIVATION_CERTIFIER_REPORT.md` (section for re-pin 2), `iiac_evidence/PDC-X/MCP-RC/activation/repin_2/` |
| Signed activation decision | status AUTHORIZED, signed under `DELEGATION-MCP-RC-2` | `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` |
| Implementer qualification package | 481 passed, 1 xfailed at `b98dc7719` | `MCP_SERVER_X_RC_QUALIFICATION_PACKAGE.md` |

## 3. Terminal verification on `development` (this session)

Environment: Windows, `D:/Projekty/intergrax/.venv/Scripts/python.exe`, `PYTHONPATH` = the checkout, one pytest process.

| Step | Check | Result |
| --- | --- | --- |
| 1 | Branch `development`; `git fetch` done; `git merge-base --is-ancestor 61531828f HEAD` | PASS; `git rev-parse HEAD` = `61531828f5728acc8f2ed357b4b835d7f5dc4ebc` (equals `origin/development`) |
| 2a | `verify_iiac_policy_activation --require-in-force` | PASS: `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from 88fd679e63188303e0590e4033abc81a3c30805b`, exit 0, no ACT-N11 line |
| 2b | 23 MCP-RC product paths | PASS: `git diff --name-only 83d939095 HEAD -- intergrax agents applications` lists exactly 23 paths, each blob-identical to `b0e19bf61` |
| 2c | Signed decision hash | PASS: SHA-256 `0e0aed5816ff76c631d8aefb897e83399d26aa6d9692f3b0c2fe6b1fd8d25147`, file unmodified |
| 2d | FRZ counter | PASS: `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` still 225 rows and unchanged since `83d939095`; no FRZ status moved |
| 2e | P8 history | PASS: `83d939095` and `ae467c8c` are ancestors of HEAD, no rewrite visible (ancestry; the local reflog of `development` shows fast-forward merges only), the `MCP_SERVER_X_P8_*` files are unchanged since `83d939095` |
| 3 | MCP RC contract (`MCP_SERVER_X_RC_SELECTION`, 39 files) | PASS: **481 passed, 1 xfailed, 103 warnings, 398.41 s**, exit 0, equal to the expected result. The single xfail is the strict `[product]` case (section 4). The local log is `.tmp/session/mcp-rc-closure/rc_contract_at_61531828f.log` (not committed; SHA-256 `180807493a0fd86a0b043fbd8191eb8bb8dd859157d7c7fbf15c9302594483aa`) |
| 4 | PR #554 CI on head `b4d2a3c17` (`gh pr view 554`) | PASS: `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate` SUCCESS, plus `governance-policy-pin` SUCCESS; the other contexts were SKIPPED; PR state MERGED at 2026-11-15T05:31:21Z with merge commit `61531828f` |

Not done here: no mutation run, no full `tests/unit`, no re-audit of any product path, no live-provider or network test.

## 4. Residual open items and owners

| Item | Status | Owner |
| --- | --- | --- |
| F-56 (MEDIUM): `test_b6_03_scoped_disposition_valid_at_head` red at the audited tree (frozen PRE-ENT list names the deleted `ollama/manifest.py`) | open | PDC-X |
| F-35 (17 red COMPAT-X gates per the activation certifier report) | open | PDC-X |
| F-51 / F-52 (openai embedding bundle soft spots), F-53 (open, unchanged in the audit and certifier records) | accepted as declared in the audit and certifier records, not resolved | as declared in those records |
| RA-02, RA-03, RA-04 (optional hardening from the delta re-audit) | accepted as declared, not implemented | as declared |
| F-57 (INFO): `.github/workflows/unit-tests.yml` EBH-7 pins and `pdc_x_wbs.py` changed in the merged range | accepted by the auditor and the certifier | none needed |
| `xfail` `test_scaffold_defaults_mcp_to_opt_in[product]` (AF-05 / AF-16, ticket `T-IGX-039-A`) | owner named on 2026-11-15: the operator (`jakbuczarnecki`) owns `T-IGX-039-A`; the xfail stays until the real product-factory composition is restored (no plan yet) | **operator** (named 2026-11-15) |
| Domain audits IA-D1..IA-D8 (27 routed findings) | not done; **explicitly deferred** by the operator on 2026-11-15, outside the MCP-RC scope; the stage closes without them | **operator** (deferral decision 2026-11-15) |
| Unreproduced leads F-18, F-24..F-28; full `tests/unit`, network and live-provider paths | not covered | not assigned in this dossier |
| Drift: any product commit on `development` after `83d939095` voids the pin; the freeze is a declaration | standing condition | operator |
| The signed decision text is stale (158 paths, tree `324d9dba3`); the operator confirmed the signature stands (`OD-MCP-RC-05`), the text was not edited | recorded | operator |

## 5. Explicit statements

- **No FRZ promotion.** The FRZ counter stays 225 and no FRZ status was changed by this campaign step or by this dossier.
- **No production-readiness claim.** `PASS_SCOPED` and the scoped certifier `PASS` apply to the stated scope only; the C10/C15 freeze, R7-Q and PROD-Q are untouched. This dossier does not close MCP-SERVER-X and does not state that the platform is production ready.
- **Independence of roles.** The records name separate sessions for implementer, auditors, certifier, signer, activation certifier and evidence integrator, and this terminal verification was done by a further session that holds none of those roles. The attestation is `process_enforced`: one git identity signs all commits (as the certifier records state), so separation rests on session separation and the role records, not on distinct signing identities. This dossier does not verify those records beyond what the cited files say.
- **No alteration of published history.** P8 files and the published history are unchanged; this dossier is additive.
- **No edit** of product paths, of the IIAC activation record or of the signed decision.

## 8. Operator closing decisions (2026-11-15)

Recorded as an English rendering of the operator's chat answers; this addendum adds no verdict, audit or certification.

1. The named owner of the strict `xfail` `test_scaffold_defaults_mcp_to_opt_in[product]` (`T-IGX-039-A`) is the operator (`jakbuczarnecki`).
2. The domain audits IA-D1..IA-D8 are explicitly deferred and are not part of the MCP-RC closure. Nothing in this dossier claims that they were performed.

With these two decisions no operator-owned MCP-RC item remains open. F-56 and F-35 stay with PDC-X. No FRZ status changed and no production-readiness claim is made.
