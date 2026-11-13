# IIAC operator decision — MCP-RC ACT-N11 product re-bind (MCP-RC corrective campaign)

Decision ID: IIAC-OD-MCP-RC-ACT-N11-2026-11-13
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation (MCP-RC product delta, and the product delta of the merged `development` tree, both vs `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`; see "Widened scope")
Decision status: DRAFT
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Product independent audit dossier digest (MCP-RC first audit, dossier verdict PARTIAL): 30e002c28af0c51070b5236dd674fb0465d439f976bb550c39927e23703eccf1

Product independent audit dossier digest (MCP-RC delta re-audit, dossier verdict PASS_SCOPED): fb4ba5a70ad20578513fec793ef4e1421b2ed7ba06d599ce3d89ad3e23aa4f34

________________

Drafting note: the date in the Decision ID is the drafting date (2026-11-13), not a signing timestamp. This record is an unsigned draft written by the activation-package drafting session; it carries no `Approved IIAC policy activation` line and no agent composes one. The line above is intentionally left blank. It is written only by an operator-designated signing session under `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-2.md`, and only in the plain form that the IIAC activation verifiers accept.

## Scope

23 product paths changed under `intergrax/`, `agents/` and `applications/` since the current activation pin `audit_execution_sha` `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (`IIAC_POLICY_ACTIVATION_RECORD.json`, P6 activation). They are the ACT-N11 trigger, derived with the gate's own rule (`git diff --name-only 2d3eceb6c..c3f962636`, prefixes `intergrax/`, `agents/`, `applications/`). The list equals `product_paths_delta_vs_base` of `iiac_evidence/PDC-X/MCP-RC/audit_scope.json` and `product_paths_combined_change_vs_base` of `iiac_evidence/PDC-X/MCP-RC/reaudit/audit_scope.json` (23 = 23, set-equal). The campaign base `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` has no product diff against the pin (`git diff --name-only 6a6279a49..2d3eceb6c` is empty for the three prefixes), so the delta is the same against either.

| # | Path |
| --- | --- |
| 1 | `applications/dispute_sim_application/mcp/server.py` |
| 2 | `applications/governed_contractor_application/mcp/server.py` |
| 3 | `applications/intergrax_assistant_application/mcp/server.py` |
| 4 | `applications/lab_application/mcp/server.py` |
| 5 | `applications/legal_application/mcp/server.py` |
| 6 | `applications/local_workspace_application/mcp/server.py` |
| 7 | `applications/poc_template_application/mcp/server.py` |
| 8 | `applications/research_application/mcp/server.py` |
| 9 | `intergrax/applications/_shared/fastapi_mcp.py` |
| 10 | `intergrax/applications/_shared/mcp_application_server.py` |
| 11 | `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py` |
| 12 | `intergrax/applications/_shared/mcp_catalog_tools.py` |
| 13 | `intergrax/applications/_shared/mcp_developer_api.py` |
| 14 | `intergrax/applications/_shared/mcp_engine_concurrency.py` |
| 15 | `intergrax/applications/_shared/mcp_harness_principal.py` |
| 16 | `intergrax/applications/_shared/mcp_host_execution_transport_cancel.py` |
| 17 | `intergrax/applications/_shared/mcp_identity_mode.py` |
| 18 | `intergrax/applications/_shared/mcp_invocation_id.py` |
| 19 | `intergrax/applications/_shared/mcp_nexus_server.py` |
| 20 | `intergrax/applications/_shared/mcp_server_engine.py` |
| 21 | `intergrax/applications/_shared/mcp_wire_errors.py` |
| 22 | `intergrax/contracts/mcp_server_definition.py` |
| 23 | `intergrax/tools/mcp_result_bounds.py` |

Last product change on the audited line: `b0e19bf61b346e6941b2e190b703c6922e708b34`. Every later commit on `mcp-rc` up to the drafting head `c3f962636b104ceccce07d4252f7e60b16bf53f4` changes docs, tests, harness scripts or governance tickets only (certifier preflight check 2; `EVIDENCE_INTEGRATION_RECORD.md`).

## Widened scope (operator answer of 2026-11-13)

Fact, verified at drafting: `origin/development` @ `30c267dbe5a4901d526403d16e0a19034beeb7c0` differs from the same pin in 125 other product paths (none of them among the 23, none with `mcp` in its name; by top-level directory: `intergrax/runtime` 44, `intergrax/llm_adapters` 20, `intergrax/agent_distribution` 10, `intergrax/hosting` 9, `intergrax/applications` 9, `intergrax/rag` 5, `intergrax/core` 5, `intergrax/queueing` 4, the rest in smaller directories). The count moves with every merge to `development`.

Operator instructions, relayed in the chat with the drafting session on 2026-11-13 (original, Polish, as typed): the order is `Rebind po merge'u`, and to the question whether to widen the scope of this unsigned decision before the delegate signs: `Tak, rozszerz`. English renderings by the drafting session (not operator-written lines): "Rebind after the merge"; "Yes, widen it". Because the rebind is made after `development` is merged into `mcp-rc`, ACT-N11 will measure the merged tree against the pin, so the scope of this decision is widened as follows:

1. **MCP-RC paths:** the 23 paths in the table above. Independently audited (`PASS_SCOPED`) and preflighted; evidence as cited in this record.
2. **Merged `development` paths:** every other product path under `intergrax/`, `agents/` and `applications/` that differs from the pin `2d3eceb6c` at the integrated baseline SHA bound by the rebind. The MCP-RC audits do **not** cover them, and this record does not claim any audit of them.

Before this record is signed, a revision of this draft adds the facts that make scope item 2 sound, each taken from published evidence and none invented here: the integrated baseline SHA; the exact path count and a pointer to the audit-scope file that lists the paths; the independent audit dossier (verdict and SHA-256) and the certifier preflight that cover them; their auditor and certifier session ids. Until that revision exists, scope item 2 is **not covered** and the draft is not signable. The delegation record must then be updated to the new digest of the revised text, because the delegation covers only the digest it names.

This record still does not say how the PDC-P7-00 domain batches are audited; it only cites the evidence once it exists.

## Independent audit and preflight

- First independent audit: dossier verdict `PARTIAL` at `93a089d3cca941592f1ad4f65435333c86da82e5` (manifest field `verdict` carries `CONDITIONAL`, the closest schema value; three product findings AF-01..AF-03 and two evidence items AF-04, AF-05 open). Auditor session `MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2`. Dossier `docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md`.
- Delta re-audit: dossier verdict `PASS_SCOPED` for the delta `93a089d3c..b0e19bf61` and for the combined change `6a6279a49..b0e19bf61` (manifest field `verdict` carries `CONDITIONAL`; 0 HIGH, 0 MEDIUM, 4 LOW, 3 INFO new findings, no product change required; AF-01, AF-02, AF-03, AF-04, AF-10 closed). Auditor session `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48`. Dossier `docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md`.
- Certifier preflight: `CERTIFIER_PREFLIGHT_PASS_SCOPED` (not an activation certification, not a rebind) at record head `1b5f26f249beaafe007ffff2f1e396ae393a07dc`; session designation `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` (a designation recorded in the manifests, not a verified application session id). Result `iiac_evidence/PDC-X/MCP-RC/certifier/certifier_preflight_result.json`, report `mcp_server_x/MCP_SERVER_X_RC_CERTIFIER_PREFLIGHT.md`.
- Implementer session: `T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (label as recorded in the manifests; no UUID is recorded and none is invented).
- Evidence publication SHA: `78d4c496071141ca4092b926b838fb80016f1b91` (81 files imported byte-for-byte; six pin-only manifest edits followed in `c3f962636b104ceccce07d4252f7e60b16bf53f4`, see `iiac_evidence/PDC-X/MCP-RC/EVIDENCE_INTEGRATION_RECORD.md`).

Current gate state (certifier log `iiac_evidence/PDC-X/MCP-RC/certifier/logs/verify_iiac_policy_activation_at_1b5f26f24.log`; not re-run by the drafting session): `verify_iiac_policy_activation --require-in-force` exits 1 with 23 ACT-N11 product-path errors. That is the intended trigger until the two-phase rebind.

## Digests (SHA-256 of committed file bytes, `git -c core.autocrlf=false show <commit>:<path>`)

Values below were recomputed at the drafting head `c3f962636` and, for the audit artifacts, equal the digests recorded in the auditor manifests (and also hold at `78d4c496`).

| Artifact (under `docs/project/maintainers/qualification/`) | SHA-256 |
| --- | --- |
| `mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md` | `30e002c28af0c51070b5236dd674fb0465d439f976bb550c39927e23703eccf1` |
| `iiac_evidence/PDC-X/MCP-RC/iiac_audit_mcp_rc_findings.json` | `a3a197293128d8c29167c7f9220e01a54de53e448074b74b114da6db6af42f1c` |
| `iiac_evidence/PDC-X/MCP-RC/audit_scope.json` | `d9badd094325003529e63ca2c8511a453edf1e09cc6053a9ab8385d3f20770a1` |
| `iiac_evidence/PDC-X/MCP-RC/test_matrix.txt` | `5f0ee3302bfd253c25f5da466ea1f7c445ece4aea383352e0d5193511f8f544e` |
| `iiac_evidence/PDC-X/MCP-RC/iiac_audit_session_manifest.PDC-X.mcp_rc.json` (after pin-only edits) | `a9ae928e810e5935cbe1ae32cc1a9126a980204739f6c24f51e8d7dcd278aa0b` |
| `mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md` | `fb4ba5a70ad20578513fec793ef4e1421b2ed7ba06d599ce3d89ad3e23aa4f34` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_mcp_rc_reaudit_findings.json` | `2c24f7f755903f848c2f2b1c1afcd250e284b4d322dc1d23bffd9f96bd385a12` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/audit_scope.json` | `d1ca3598d982264e199061e95f82e5c058d01a1ebf279a6a84e6496094576a3c` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/test_matrix.txt` | `4e81fcbdca6a9d32f4239affa17e76486ac774823a0aa03380c94d0de2bb03f5` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json` (after pin-only edits) | `5095559de8d6b2d1c10c476f3ba6c68e0f98a105630d0951a0ad2e268a071c8f` |
| `mcp_server_x/MCP_SERVER_X_RC_CERTIFIER_PREFLIGHT.md` | `c1f6027c55a79e8d1cfb4eded7c4c543996d5212998ea3c70d5cf9248325ef0b` |
| `iiac_evidence/PDC-X/MCP-RC/certifier/certifier_preflight_result.json` | `f4c60c3f961c11e7bd684de38d0b91a090bb5f21417f4e508318616c0c90d87a` |

The `Audit report digest` at the top of this record is the digest of the `IIAC-GOV-1` governance dossier the activation record already references (`iiac_evidence/IIAC-GOV-1/certification/IIAC_ACT01_INDEPENDENT_REAUDIT_DOSSIER.md`, `session_b_governance_audit_report_digest`). It is taken over unchanged from the P6 decision because the activation gate compares that line with the record field. Whether the MCP-RC rebind keeps that governance dossier set or binds the MCP-RC dossier is for the activation certifier and the orchestrator to settle; if the record changes its report, this line must change with it before signing. The pre-edit manifest digests at `78d4c496` were `c76fe945d5967e67952b09f76d282d27f075ea018179bdccfd051676b0d8b9e9` (first audit) and `bb9404793f341f57d8ea2317fe7e41fbe9a8e54fca83828fa8d33dfc96f10ea0` (re-audit); they differ from the table only by the six pin-only edits.

## Session provenance (operator confirms by signing)

The P6-style separation applies: the MCP-RC implementer (`T-IGX-038-A-F-MCP-RC-IMPLEMENTER`), the first G6 auditor, the delta re-audit auditor, the preflight certifier, the evidence-integration session, the signing delegate and the activation certifier (a new dedicated session, id recorded in the activation binding manifest) are separate sessions; none is the implementer. This is a `process_enforced` attestation, not cryptographic proof.

## Operator-owned open items (not resolved by this record)

1. **`[product]` xfail owner.** The strict xfail `tests/unit/applications/test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]` (AF-05, AF-16) is transferred to `governance/tickets/T-IGX-039-A.md` (draft, owner role "strict-product scaffold owner", no named person). The re-audit's open condition is "named owner or ticket for AF-05 and AF-16"; the operator names the owner.
2. **RA-02, RA-03, RA-04.** The re-audit's LOW findings are recorded by the implementer as "open, accepted as declared" (`MCP_SERVER_X_RC_CORRECTIONS_RECORD.md`); no operator acceptance is recorded. RA-05..RA-07 (INFO) likewise.
3. **AF-05** (real factory composition unverified for MCP opt-in): the auditor's judgement is `ACCEPT_SCOPED`; operator acceptance is not recorded.
4. **AF-06..AF-09 and AF-11..AF-15** (auditor judgement `ACCEPT`) and **AF-16 / G-DOC items** (auditor judgement `ACCEPT_SCOPED`) are carried as declared limits; no operator disposition is recorded.
5. **Coverage of the merged `development` paths** (125 at drafting, see "Widened scope"): the independent audit and preflight evidence, the integrated baseline SHA and the path list are not yet available to this record.

Signing the activation line does not record the operator's acceptance of any item above. They remain open until the operator answers them separately.

## Non-claims

This decision authorizes only the IIAC activation named in its scope: the 23 audited MCP-RC paths and, once the revision described under "Widened scope" cites its evidence, the merged `development` paths that evidence covers. It does not bypass governance checks, does not close the MCP-SERVER-X stage, does not promote any FRZ status (counter stays 225), does not claim production readiness, does not authorize a merge, a pull request, a ticket approval or P8 changes, and does not edit `IIAC_POLICY_ACTIVATION_RECORD.json`. The two-phase rebind of that record is a separate step by a separate activation-certifier session inside the PDC-P7-00 window.

## Provenance

| Item | Value |
| --- | --- |
| Ticket | `governance/tickets/T-IGX-038-F.md` |
| Prior `audit_execution_sha` (current pin) | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| Campaign base | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` |
| First audit: audited / test execution SHA | `93a089d3cca941592f1ad4f65435333c86da82e5` |
| First audit: execution base | `f3078b293ecee402583b3d6ad9ef5aaeff0b2d1d` |
| Re-audit: audited / test execution SHA (last product change) | `b0e19bf61b346e6941b2e190b703c6922e708b34` |
| Re-audit: execution base | `f1dde5a3e19924b82e902be18f11a21b64b822ba` |
| Tests and harness after the re-audit (RA-01 closure) | `b98dc77196cee10a9f06323bd47c4b69d0f1baa6` |
| Certifier preflight record head | `1b5f26f249beaafe007ffff2f1e396ae393a07dc` |
| Evidence publication | `78d4c496071141ca4092b926b838fb80016f1b91` |
| Drafting head (`mcp-rc`) | `c3f962636b104ceccce07d4252f7e60b16bf53f4` |
| Operator commissions | `mcp_server_x/operator_decisions/OD-MCP-RC-01.md`, `OD-MCP-RC-02.md`, `DELEGATION-MCP-RC-1.md`, `DELEGATION-MCP-RC-2.md` |
| Runbook | `mcp_server_x/MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md` |
| Precedent | `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` |
