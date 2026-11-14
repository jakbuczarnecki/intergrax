# MCP-RC activation certifier report (PDC-P7-00 window, integrated baseline)

Ticket: `governance/tickets/T-IGX-038-F.md`. Role: activation certifier (Session B pattern), new dedicated session `local_8974948e-d8d3-45e3-96ca-fb61dfd5f249` (agent id `certification-governance-agent`), started 2026-11-14 on branch `mcp-rc-act-cert` cut from `origin/mcp-rc` @ `afdf27e622369c3c635aeb1301c9d90d962d8371`. This session is not the implementer, an auditor, the preflight certifier (`MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13`, `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`), an evidence-integration session or the signing delegate (`129c7788-d201-43f3-b695-ef48cec3a73f`). `process_enforced`: one git identity signs all commits; the separation is an attestation.

## Verdict

**Mechanical result: the activation gate passes.** `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` at the phase-2 commit prints `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from 92d7308213bd46e76ce0b545cd94b36bc65d8e0e` and exits 0.

**Certifier verdict on the binding manifest: `PASS`, scoped.** It is the certifier's own `PASS` authored from audits whose dossier verdicts are `PASS_SCOPED` (manifests `CONDITIONAL`), as the operator recorded in `OD-MCP-RC-04` section 3 and as the signed decision states. It is not an auditor verdict. The limits in the next sections are part of the verdict.

**One finding that needs the operator before any pull request (not a gate failure on this branch):** `origin/development` is no longer at `a156cbbd5`. Commit `f209febbe` (2026-11-14 08:18:23 +0100, `fix(integrations): resolve the single ollama catalog row for the embedding category (T-IGX-036-X)`) changes the product path `intergrax/integrations/providers/embedding_provider/ollama/register.py`, which is one of the 135 non-MCP paths audited by the PDC-P7-00 audits (byte-different from the audited tree, +28/-6). The decision lists "any product commit on `development` after `a156cbbd5`" as drift (open condition 8). On this branch the gate does not see it (the branch does not contain `f209febbe`; product diff to the audited tree is empty). In a pull request to `development` the CI merge ref contains it, so ACT-N11 will report that path against `audit_execution_sha` `29d66ae3f` until an audit addendum covers the file and the pin is re-bound. It looks like the F-42 repair, but I did not verify that; I did not run or audit that commit.

## SHAs

| Item | SHA |
| --- | --- |
| Starting tip (`origin/mcp-rc`, signed decision) | `afdf27e622369c3c635aeb1301c9d90d962d8371` |
| Binding manifest + handoff + pre-checks commit | `2899735e56556b3d022da9fe427042ac66a08922` |
| Phase 1 commit (only the record) | `92d7308213bd46e76ce0b545cd94b36bc65d8e0e` |
| Phase 2 commit (two lines) | `9949fe2c3dd64064f3b2002565b31c3c80b8cad6` |
| This report | the commit that carries this file (reported in the session message; a file cannot hold its own commit SHA) |
| `governance_implementation_sha` | `2ae1014116d353bc7742985e62f2bccc2bfac798` (unchanged) |
| `corrective_implementation_sha` = manifest `audited_implementation_sha`, `test_execution_sha` | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` |
| `audit_execution_sha`, `evidence_publication_sha` (manifest and record) | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` |
| `certification_publication_sha` | `7801dcb1293e31ee0c483184efd59c01a2dc606b` |
| Previous pin | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (P6, phase-2 `19c617365`, preserved in history) |
| `effective_from_sha` = `policy_activation_sha` | `92d7308213bd46e76ce0b545cd94b36bc65d8e0e` |

## Digests (SHA-256 of working-tree bytes)

| Artifact | SHA-256 |
| --- | --- |
| Binding manifest `iiac_evidence/PDC-X/MCP-RC/activation/iiac_audit_session_manifest.IIAC-ACT-N11-MCP-RC-ACTIVATION.json` | `ba9e8f5a2d541412351d36915a941141b87bdb84098fc7b9de926d86c7185cc7` |
| Handoff `iiac_evidence/PDC-X/MCP-RC/activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_MCP_RC_ACTIVATION.md` (CRLF) | `955a589d13b86d2cf5c75412dfd8c795d4ba628c9e261378322ef4ff9b39fa62` |
| Audit report (`IIAC-GOV-1` dossier, the digest the decision cites) | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` |
| Audit scope (`ENT-AUDIT-X/act_n11_recovery/audit_scope.json`) | `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f` |
| Test matrix | `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b` |
| Findings (`HIST_P5_REACTIVATION_FINDINGS.json`) | `df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831` |
| Signed decision `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (not bound by a digest field; gate checks its lines) | `0e0aed5816ff76c631d8aefb897e83399d26aa6d9692f3b0c2fe6b1fd8d25147` |

## Choices and why

- **Bound artifacts (runbook Q3).** The record keeps the `IIAC-GOV-1` report, scope, matrix and findings set (P6 pattern). All five bound paths sit under `iiac_evidence/` (ACT-N14), the findings register has no open BLOCKER/CRITICAL (ACT-N15, 4 entries, checked with the gate's own `_findings_errors`), and the report bytes equal the bytes at `evidence_publication_sha` (ACT-N11 report check). The decision already cites `c405c65f...`, so no decision revision was needed. The audits of the 158 paths are the MCP-RC and PDC-P7-00 dossiers, which cannot be bound (they sit outside `iiac_evidence/`); the manifest `limitations` say that the bound set is the gate's anchor and not the audit of the 158 paths.
- **Single `auditor_session_id` (Q4).** `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-2`: it audited exactly the tree named as `audited_implementation_sha` (`324d9dba3`), its scope covers all 158 paths, and it published the evidence at `29d66ae3f`. The other four auditor ids and all implementer, certifier, integration and signing designations are listed in the handoff table. `implementation_session_id` is the addendum 2 designation `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`. These ids are labels, not verified application ids (Q7); only my own `local_...` id and the signing delegate's scratchpad id are first-hand.
- **`audit_execution_sha` = `evidence_publication_sha` = `29d66ae3f`.** P6 pattern and runbook candidate. The audit ran at the product-identical `324d9dba3`; the later docs-only commit gives the gate a pin that also covers the material governance paths from the publication point. No `MATERIAL_GOVERNANCE_PATHS` file changed after the previous pin or after `29d66ae3f` (checked).
- **`certification_publication_sha` (Q5).** `7801dcb12`, the preflight addendum commit, as the runbook candidate; differs from the corrective SHA (gate rule).
- **Handoff placement and line endings (Q9).** The handoff is under `iiac_evidence/**`, which `.gitattributes` marks `-text`, so the CRLF bytes whose digest the record binds are the same in every checkout mode; the W4-F01 tamper test in `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` needs CRLF (its `lf-collapsed` mutation is otherwise a no-op). Verified: `file` reports CRLF, blob digest equals working-tree digest. I did not run that test (one-process rule; it is in the governance suite).
- **`notes`, `operator_decision_class`** are free text; no ACT-N18 pattern appears in the record or in the decision (gate `closure_conflicts` empty).

## What I verified myself

All with logs in `iiac_evidence/PDC-X/MCP-RC/activation/logs/` and the scripts in `.../activation/scripts/`:

1. `certifier_activation_01_precheck.log` (script `precheck.py`, gate functions imported, checkout at `afdf27e62`): ancestry chain `2ae101411 -> 324d9dba3 -> 29d66ae3f -> 7801dcb12 -> HEAD` and `2d3eceb6c -> 324d9dba3`; the gate's own `product_paths_changed_between` gives **158** paths from the pin to the audited tree, to `29d66ae3f` and to HEAD (identical sets); **23** of them equal `product_paths_delta_vs_base` of the MCP-RC `audit_scope.json` and are byte-identical to `b0e19bf61` at HEAD; 135 non-MCP; product diff `324d9dba3..HEAD`, `324d9dba3..29d66ae3f` and `29d66ae3f..HEAD` empty; the only non-`docs/` differences between the audited tree and HEAD are two ticket files, `scripts/qualification/pdc_x_inventory.py`, `pdc_x_wbs.py` and `uv.lock` (no product path, no `MATERIAL_GOVERNANCE_PATHS` path).
2. Signed line: exactly one line, matches `_SIGNATURE` fullmatch (`2026-11-14T08:21:13+01:00`), no CR, gate `_operator_errors` empty with the record's decision id and the report digest; ACT-N18 empty over record plus decision text; none of the sixteen underscores, `NOT SIGNED`, `Decision status: PENDING`; file name rule passes; no `Approved:` line anywhere.
3. Four bound-artifact digests recomputed from working-tree bytes equal the manifest and decision values, and the bytes equal those at `29d66ae3f`.
4. `certifier_activation_02_rc_contract.log`: MCP-RC contract, selection list from Python (`MCP_SERVER_X_RC_SELECTION`, 39 files), one pytest process, `PYTHONPATH` = this checkout (imports resolved to it), at HEAD `afdf27e62`: **481 passed, 1 xfailed (strict), 0 failed, exit 0, 417.04 s**. The xfail is the TRANSFERRED `tests/unit/applications/test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]`.
5. `certifier_activation_03_binding_manifest_validators.log`: `verify_iiac_audit_manifest <manifest> --repo . --require-handoff-template` PASS (exit 0); `verify_iiac_role_separation <manifest> --strict-product-diff --repo .` PASS (exit 0). The log was taken before the manifest commit; blob digests after the commit equal the logged working-tree digests.
6. `certifier_activation_04_phase1_validator.log`: at phase 1 the gate fails with **exactly the four errors** of the P6 log (both `effective_from_sha` / `policy_activation_sha` missing and not 40-hex), nothing else, exit 1.
7. `certifier_activation_05_phase2_validator.log`: at phase 2 `IN_FORCE`, exit 0 (this includes the gate's own companion run of both validators, the ancestry chain, the phase-1 blob check against the ten `BINDING_KEYS`, the digest checks, the operator-file rules and the ACT-N11 product and material-path scan from `29d66ae3f` to HEAD).
8. Phase 1 changed only `IIAC_POLICY_ACTIVATION_RECORD.json` (15 lines); phase 2 changed only its two SHA lines.

## What I took from the audits and preflights (not re-done)

The substance of the audits: the dossier verdicts (`PARTIAL` superseded, four `PASS_SCOPED`), the findings F-04..F-44, the depth of review of the 135 non-MCP paths (diff scan, sampling, probes, mutants), the build-up 153 / 156 / 158, the 40 pin-red regression failures, the CI-context reading of the preflights, and the identity of the sessions. I did not re-audit any product path, run the mutation harness, or run the CI contexts.

## Scope limits and open conditions (none resolved or accepted by this report)

1. **Drift on `development` (new, see Verdict):** `f209febbe` changes `intergrax/integrations/providers/embedding_provider/ollama/register.py` after `a156cbbd5`. Needs an audit addendum for that file and a re-pin before a pull request can pass ACT-N11 on the merge ref; if an addendum 3 is made for F-41 as well, the evidence SHAs and digests above move.
2. **F-41 (MEDIUM):** the ENT-AUDIT impact-map rule for `applications/*/mcp/server.py` is not on `development` (`git diff 324d9dba3 origin/development -- intergrax/platform/enterprise_audit` is empty at my fetch). Two tests stay red on the integrated tree until it lands; when it lands, the two impact-map paths inside the audit change and addendum 2 is void for them.
3. **F-42:** probably what `f209febbe` repairs; unconfirmed, not run. **F-35, F-09:** COMPAT-X gates and the pytest exit hang, owner repairs on `development`, status not re-checked. **F-07, F-43, F-44, F-38, F-04/F-37, F-05, F-10:** as the dossiers and the decision state.
4. **CI contexts not verified.** `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate` were not run by me; no claim is made that any is green. `governance-fast` needs this rebind in the PR head to clear the ACT-N11 tests.
5. **Domain audits IA-D1..IA-D8 not done** (27 routed findings); unreproduced leads F-18, F-24..F-28; full `tests/unit`, network and live-provider paths, G-DOC-01..12 and P8 not covered.
6. **MCP-RC carry-overs:** owner of the strict xfail (AF-05, AF-16, `T-IGX-039-A`) not named; operator acceptance of RA-02, RA-03, RA-04 and AF-05 not recorded.
7. **Identity:** implementer and preflight-certifier ids are labels; independence is `process_enforced`, one git identity.
8. **Freeze:** the freeze on product paths is a declaration; item 1 shows it was not held.

## Not done, not claimed

This is **not stage closure**: MCP-SERVER-X is not closed. It is **not FRZ promotion**: the counter stays 225 and no FRZ file was touched. It is **not production readiness**. No pull request was opened, nothing was merged, no ruleset changed, no `Approved:` line was written, no product, test, ticket, gate-script, `MATERIAL_GOVERNANCE_PATHS`, P8 or auditor file was edited. The merge is a separate operator act (runbook section (d)); the terminal verification is a separate session (section (e)).
