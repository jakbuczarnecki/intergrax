# MCP-RC activation runbook (signing, two-phase rebind, merge, verification)

Role of this file: ordered hand-over for the sessions that come after the audit and the certifier preflight. It is written by the activation-package drafting session, which is not the implementer, an auditor or a certifier. It carries no verdict, no signature, no rebind and no `Approved` line. It does not close MCP-SERVER-X, promote any FRZ status (counter stays 225) or claim production readiness.

Ticket: `governance/tickets/T-IGX-038-F.md`. Precedent: MCP P6 (`MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`, activation commits `a65b45987` binding manifest, `92ad35557` phase 1, `19c617365` phase 2, PR merge `66dcd1bfe`). Gate sources read for this runbook: `scripts/governance/iiac_activation_gate.py`, `verify_iiac_policy_activation.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`.

Legend: **[gate]** read from the gate code; **[P6]** copied from the P6 commits and logs; **[OPEN]** could not be derived, needs the operator or the orchestrator (collected in the last section).

## Order decided by the operator (2026-11-13)

Question put to the operator in chat (question 1, order versus the 125 non-MCP product paths on `development`): rebind before or after merging `development` into `mcp-rc`. Answer (original, Polish, quoted as typed in the chat to the drafting session): `Rebind po merge'u`. English rendering by the drafting session (not an operator-written line): "Rebind after the merge". Recorded as an instruction, not as a signature.

Execution order is therefore **(a) sign, (c) merge `development` into `mcp-rc`, (b) two-phase rebind, (d) PR, (e) merge, (f) terminal verification**. The letters keep the names used elsewhere (the delegation and the decision refer to them); only the order changed. Consequences, not decided here and listed under Q1/Q2: the rebind pins must sit at or after the merged (integrated) tree, so the candidate pins `b0e19bf61`/`78d4c496` in (b) no longer fit as they are; an independent audit and preflight must cover the 125 non-MCP paths that the merge brings under ACT-N11 (the MCP-RC audit covers only the 23); and the scope text of the unsigned decision (23 paths vs the pin `2d3eceb6c`) may need to be widened or replaced before it is signed. Signing (a) is not affected by the order, but a signature written on the current text binds that text's digest (`DELEGATION-MCP-RC-2.md`).

## State at drafting

| Item | Value |
| --- | --- |
| `mcp-rc` head | `c3f962636b104ceccce07d4252f7e60b16bf53f4` (last product change `b0e19bf61b346e6941b2e190b703c6922e708b34`) |
| Evidence publication | `78d4c496071141ca4092b926b838fb80016f1b91` |
| Current pin (`audit_execution_sha`) | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`, record status `IN_FORCE`, `effective_from_sha` `92ad3555775649742299826e3680ec8ce703cc8a` |
| ACT-N11 now | 23 product paths differ from the pin; `verify_iiac_policy_activation --require-in-force` exits 1 by design |
| Unsigned decision | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (`Decision status: DRAFT`) |
| Delegation record | `mcp_server_x/operator_decisions/DELEGATION-MCP-RC-2.md` |
| `origin/development` (fetched at drafting) | `30c267dbe5a4901d526403d16e0a19034beeb7c0`: 99 commits and 125 product paths ahead of the campaign base `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb`; none of the 125 is one of the 23 MCP-RC paths |

Rules for every session below: one process at a time on the shared Windows workstation (`uv run` once per verification step); logs under `.tmp/session/<task>/` and, where this runbook says so, published evidence under `iiac_evidence/PDC-X/MCP-RC/`; commits touching `scripts/`, `intergrax/`, `agents/`, `applications/` carry `Ref: governance/tickets/T-IGX-038-F.md`; no product-path edit anywhere (any product edit voids the audit and the pin, ACT-N11); IMPLEMENTER != AUDITOR != CERTIFIER, and the signing delegate and the activation certifier are neither.

## (a) Designated signing session: the activation line

Who: a new, narrow session under `DELEGATION-MCP-RC-2` (single use, 24 h expiry, scope only this line). Not the activation certifier.

1. Start from the commit that carries the three files of this package. Re-verify, read-only, the four checks in `DELEGATION-MCP-RC-2.md` ("What the designated session verifies before it signs"): delta re-audit `PASS_SCOPED`, preflight `CERTIFIER_PREFLIGHT_PASS_SCOPED`, unsigned-draft digest `9edf21ed4e22a4a4a66d722a4c5399b0072c2a6705d55ce2fcdbbfcb754818dd`, no activation-record change.
2. Ask the operator for the direct confirmation (P6 practice) and record it verbatim with a labelled English rendering in the delegation record.
3. In the decision file only: replace the standalone `________________` line by `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp from its own clock>`; `Decision status: DRAFT` to `Decision status: AUTHORIZED`; the `Drafting note` paragraph to a `Provenance note`. In the delegation record only: fill the `PLACEHOLDER` fields and tick nothing for the operator. **[gate]** the plain line must match `^Approved IIAC policy activation IIAC-2026-11-05\.1: operator @ <ISO-8601>$`, must be one whole line of the file, and the file must not contain `________________`, `NOT SIGNED` or `Decision status: PENDING`; the file name must not contain `PACKET` or `DRAFT` and must start with `IIAC_OD_ACTIVATION_`.
4. Commit on its own branch (named by the orchestrator) with the `Ref` trailer. No rebind, no merge, no edit of `IIAC_POLICY_ACTIVATION_RECORD.json`. Report: commit SHA, the written line, which operator-owned open items are still open.
5. Stop. The line stays in the file; the file is the `operator_decision_path` of the later rebind.

## (b) Activation certifier: phase 1 and phase 2 (inside the PDC-P7-00 window)

Who: a new dedicated session ("activation certifier"), not the signing delegate, not the preflight certifier, not an auditor, not the implementer. Window: PDC-P7-00 **[OPEN: who declares it open and where it is recorded]**.

**(b) runs after (c)** (operator order above). **Do not start (b) until the open questions Q1 and Q2 below are answered**: the pins must sit at or after the merged tree, and the audit and preflight evidence must cover it. The pins in the tables are the P6-analogous candidates for the un-merged `mcp-rc` line; with the merge first they no longer pass `--strict-product-diff` or ACT-N11 as they are (Q1), so treat them as placeholders for the integrated-baseline values.

### b1. Pre-rebind commit (P6: `a65b45987`, one commit, three kinds of file)

Place under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC/activation/` **[OPEN: location and file names are proposals; the gate requires only the prefix `iiac_evidence/` for the manifest and the prefix `IIAC_SESSION_HANDOFF_RECORD_` plus `.md` for the handoff]**.

**Binding manifest** (Session B style; P6 example `iiac_evidence/MCP-SERVER-X/act_n11_mcp_p6/iiac_audit_session_manifest.IIAC-ACT-N11-MCP-P6-ACTIVATION.json`). **[gate]** it must satisfy `validate_manifest_dict` and the JSON schema, and the record must agree with it field by field:

| Field | Value or source |
| --- | --- |
| `stage_id` | `IIAC-GOV-1` **[gate]** (the gate rejects anything else; the MCP-RC identity goes into `limitations`) |
| `certification_profile`, `policy_version` | `iiac-v1`, `IIAC-2026-11-05.1` |
| `implementation_session_id`, `implementation_agent_id` | `T-IGX-038-A-F-MCP-RC-IMPLEMENTER`, `implementer` |
| `auditor_session_id`, `auditor_agent_id` | one id, `independent-quality-auditor`; candidate the delta re-audit auditor `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48` **[OPEN Q4: the schema holds one auditor id, MCP-RC has two]** |
| `certifier_session_id`, `certifier_agent_id` | the activation certifier's own verifiable id, `certification-governance-agent`; must differ from implementer and auditor **[gate]** and from the preflight designation by process |
| `audited_implementation_sha` | must equal the record's `corrective_implementation_sha` **[gate]**; candidate `b0e19bf61b346e6941b2e190b703c6922e708b34` |
| `test_execution_sha` | candidate `b0e19bf61b346e6941b2e190b703c6922e708b34` (the re-audit's test execution SHA) |
| `audit_execution_sha`, `evidence_publication_sha` | must equal the record's **[gate]**; P6 used the evidence publication SHA for both; candidate `78d4c496071141ca4092b926b838fb80016f1b91` |
| `certification_publication_sha` | must equal the record's and differ from `audited_implementation_sha` **[gate]**; candidate `c3f962636b104ceccce07d4252f7e60b16bf53f4` (the commit with the pin-only edits; P6 used the certifier's pin-only commit) **[OPEN Q5]** |
| `audit_report_digest`, `audit_scope_manifest_digest`, `test_matrix_digest`, `findings_digest` | SHA-256 of the working-tree bytes of the files at the record's `audit_report_path`, `audit_scope_path`, `test_matrix_path`, `findings_path` **[gate]**; see Q3 for which files |
| `verdict` | `PASS` **[gate]** (the product manifests carry `CONDITIONAL`; they are not evaluated by the activation gate) |
| `limitations` | MCP-RC scope (23 paths), the two product manifests and their dossier verdicts, the preserved open items, "does not close MCP-SERVER-X, promote FRZ or authorize P8" **[P6]** |
| `assurance_level`, `handoff_record_path` | `process_enforced`, path of the handoff record below **[P6]** |
| extra fields | `activation_audit_report_path`, `product_session_manifest_path` **[P6]**; the schema allows additional properties |

**Handoff record**, file name starting `IIAC_SESSION_HANDOFF_RECORD_` and ending `.md`, under `docs/project/maintainers/qualification/` **[gate]**. Its text must contain `IIAC-2026-11-05.1`, the manifest's `implementation_session_id` and its `auditor_session_id` **[gate]**; P6 shape: from/to, verdict, session ids, evidence SHA, binding, limits. Mention both MCP-RC auditor ids in the text. **[P6, from the certifier log, not from gate code]** write it with CRLF line endings: an LF-only handoff made the W4-F01 tamper test a no-op in P6.

**Validators and logs** (P6: `certifier_activation_binding_manifest_validators.log`, command, HEAD, UTC and exit code in each log):

```text
python -m scripts.governance.verify_iiac_audit_manifest <manifest> --repo . --require-handoff-template
python -m scripts.governance.verify_iiac_role_separation <manifest> --strict-product-diff --repo .
```

Recompute and log the deltas and digests (P6: `certifier_activation_recomputed_deltas_digests.log`): the 23 paths versus the pin, each digest used, product-diff between `audited_implementation_sha` and `audit_execution_sha`.

### b2. Phase 1 commit (P6: `92ad35557`, only the record changes)

Edit `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json`. Fields that change from the current record:

| Field | New value |
| --- | --- |
| `effective_from_sha`, `policy_activation_sha` | empty strings **[gate]** (phase-1 blob must leave `effective_from_sha` empty) |
| `corrective_implementation_sha` | candidate `b0e19bf61b346e6941b2e190b703c6922e708b34` (= manifest `audited_implementation_sha`) |
| `audit_execution_sha` | candidate `78d4c496071141ca4092b926b838fb80016f1b91` (**Q1, Q2**) |
| `evidence_publication_sha` | candidate `78d4c496071141ca4092b926b838fb80016f1b91` |
| `certification_publication_sha` | candidate `c3f962636b104ceccce07d4252f7e60b16bf53f4` (**Q5**) |
| `session_b_verdict` | `PASS` |
| `session_b_manifest_path`, `session_b_manifest_digest` | the binding manifest from b1 and the SHA-256 of its working-tree bytes (the gate reads bytes as stored; `iiac_evidence/**` is `-text`) |
| `handoff_record_path`, `handoff_digest` | the handoff from b1 and the SHA-256 of its working-tree bytes |
| `operator_decision_path`, `operator_decision_id` | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md`, `IIAC-OD-MCP-RC-ACT-N11-2026-11-13` |
| `operator_activation_line` | the exact signed line from (a), copied byte for byte **[gate]** |
| `operator_decision_class`, `notes` | free text; P6 `OD-MCP-ACT-N11-P6` and a phase-1 note; proposal `OD-MCP-RC-ACT-N11` **[OPEN Q8]** |

Unchanged unless Q3 says otherwise: `policy_version`, `status` `IN_FORCE`, `governance_implementation_sha` `2ae1014116d353bc7742985e62f2bccc2bfac798`, `session_b_governance_audit_report_digest` `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e`, `audit_report_path`, `audit_scope_path`, `test_matrix_path`, `findings_path`. **[gate]** The record must not contain `EVOL-X`, `ENT-AUDIT-X` or `ARCH-FREEZE` within 80 characters of `CLOSED` (ACT-N18; the same scan covers the decision text). **[gate]** ancestry chain: `governance_implementation_sha` ancestor of `corrective_implementation_sha` ancestor of `audit_execution_sha` ancestor of `evidence_publication_sha` ancestor of `effective_from_sha`; `certification_publication_sha` ancestor of `effective_from_sha`.

Expected result at the phase-1 commit **[P6 log]**: `verify_iiac_policy_activation --require-in-force` FAIL with exactly four errors (both `effective_from_sha` and `policy_activation_sha` missing and not 40-character SHAs). Anything else (ACT-N11 paths, digest mismatches) is a defect; stop and report.

### b3. Phase 2 commit (P6: `19c617365`, two lines change)

Set `effective_from_sha` and `policy_activation_sha` to the phase-1 commit SHA, nothing else **[gate]** (they must be equal, must not equal `HEAD`, and the phase-1 blob must carry the same ten binding fields). Expected **[P6 log]**: exit 0, `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from <phase-1 SHA>`. Log the validator output as in P6 (`certifier_activation_phase1_phase2_validator.log`).

The certifier writes its report under the same evidence folder. It writes no audit verdict about the product, and it makes no pin-only edit to auditor artifacts unless it records exactly which fields changed (P6 and `EVIDENCE_INTEGRATION_RECORD.md` practice).

## (c) Merge `development` into `mcp-rc`, then prove no MCP path moved

Runs after (a) and **before (b)** (operator order, 2026-11-13). Who performs the merge is not stated in the commission **[OPEN]**; it is neither the signing delegate nor the activation certifier's rebind step, and it edits no product path by hand.

1. `git fetch origin development`; on `mcp-rc`, merge `origin/development` (99 commits at drafting). Resolve conflicts only outside the 23 paths; a conflict inside any of them stops the run (changing an audited path voids the audit for that path, `LIM-delta`).
2. Prove, with the exact commands, that nothing audited moved: `git diff --stat b0e19bf61b346e6941b2e190b703c6922e708b34 HEAD -- <the 23 paths>` is empty; `git diff --name-only <pre-merge mcp-rc head> HEAD -- docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC` is empty; no `MATERIAL_GOVERNANCE_PATHS` path changed (**[gate]** the six listed in `iiac_activation_gate.py`). Log under `.tmp/session/mcp-rc-activation/`.
3. Run `verify_iiac_policy_activation --require-in-force` once on the merge result. **Expected before (b): FAIL** with ACT-N11 lines for the 23 MCP-RC paths plus the `development` product paths (125 at drafting) against the old pin `2d3eceb6c`, and no other error class. That result is the input of (b); record it. Do not push the merge result as if it were green, and do not open the PR (d) before (b) is committed.
4. Merge commits keep the `Ref: governance/tickets/T-IGX-038-F.md` trailer. Before pushing a new branch set its upstream (`git branch --set-upstream-to=origin/development`) and push with an explicit refspec; the pre-push ticket-reference gate otherwise scans all history.

## (d) Pull request to `development`

1. Open the PR from `mcp-rc` (or the integration branch) after (b) and (c). Do not enable auto-merge. Pin the head: `gh pr view <n> --json headRefOid` equals `git rev-parse HEAD` of the pushed branch; compare again after every push.
2. Wait until every required context has completed `success` on that exact head: `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate` (these five job names exist in `.github/workflows/unit-tests.yml` and `enterprise-audit.yml`; `governance-policy-pin` also exists there and was green in P6). `governance-fast` is red while ACT-N11 fires, so the PR head must already contain the committed rebind. A run still `in_progress` is not success; P6 PRs #459, #464 and #468 were merged with checks pending, and the structural cause is `OPEN-BP-01` (no required checks on the `development` ruleset). Record the run ids.
3. The PR body states: scope (23 paths), the three evidence SHAs, the open operator items, and the non-claims. No ruleset change.

## (e) Merge

Merge only together with the rebind: the PR head contains the signed decision, the binding manifest, the handoff, and both rebind commits, and the five contexts are green on the pinned head. The merge is a separate act that needs the operator's instruction at that moment; this runbook does not authorize it. Do not merge if `headRefOid` differs from the SHA the checks ran on.

## (f) Terminal verification (separate session)

A new session, not the implementer, an auditor, the signing delegate or the activation certifier, writes the verification report (P6 analogue: `MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md`; MCP-RC file name **[OPEN]**). Minimum: the merged `development` SHA; `verify_iiac_policy_activation --require-in-force` exit 0 on it; the record fields against this runbook; the five contexts and run ids on the PR head and, where a push run exists, on the merge; `git diff` of the 23 paths against `b0e19bf61` is empty; no FRZ change (counter 225), no P8 record edit, no ruleset change; the open operator items listed with their then-current state; the P6 process observations (merge while checks pending, `paths-ignore` push runs) checked again. Ticket lifecycle (`T-IGX-038-F`, and `T-IGX-039-A` naming) is the operator's or a separate commissioned step.

## Open gate questions and requirements this drafting session could not derive

| # | Question or requirement | Why it is open |
| --- | --- | --- |
| Q1 | **Order and pin versus the other 125 product paths on `development`.** ACT-N11 compares `git diff --name-only <audit_execution_sha> HEAD` over `intergrax/`, `agents/`, `applications/` [gate]; `verify_iiac_role_separation --strict-product-diff` requires no product diff between `audited_implementation_sha` and `audit_execution_sha` [gate]. A pin on the `mcp-rc` line (`78d4c496`) is green only on a tree whose product files equal that line's. After merging `development` the diff is the 125 non-MCP paths (runtime, llm_adapters, agent_distribution, hosting, rag, core, queueing, ...). `development` already differs from the current pin in the same 125 paths, so the gate would already report them against that pin today. A green gate on the merged result needs pins at or after an integrated tree that an independent audit and preflight cover (the `INTEGRATED_BASELINE_SHA` idea in `OD-PDC-IIAC-SCOPE` and `OD-MCP-P8-RESIDUAL-01`: one delta audit and one rebind for MCP-RC together with the PDC-X batches). The MCP-RC audit covers only the 23 paths. | **Order decided: rebind after the merge** (operator, 2026-11-13). Still open: which independent audit and preflight cover the 125 non-MCP paths at the integrated tree, which SHAs become `audit_execution_sha` and `corrective_implementation_sha`, and whether the unsigned decision's scope (23 paths vs `2d3eceb6c`) is widened or replaced by a combined PDC-P7-00 decision before it is signed. Operator or orchestrator. |
| Q2 | Which `audit_execution_sha` and `corrective_implementation_sha` the rebind binds. With the merge first, `b0e19bf61`/`78d4c496` fail ACT-N11 for the `development` paths and `--strict-product-diff` fails unless the audited SHA also sits at the integrated tree; so an integrated-baseline SHA is needed and an audit that covers it. | Follows Q1. |
| Q3 | Whether the record keeps the `IIAC-GOV-1` governance dossier set (P6 pattern, `Audit report digest` line `c405c...` in the OD) or binds MCP-RC artifacts. If MCP-RC files are bound: the re-audit findings file has no `findings` array, so `ACT-N15 findings register has no findings array` would fire; the first-audit findings file has one and no BLOCKER/CRITICAL entry. The OD's `Audit report digest` line must equal the record's digest either way. | Gate shape vs P6 convention; certifier plus orchestrator. |
| Q4 | Which single `auditor_session_id` the binding manifest carries (first audit or re-audit). The gate requires the manifest's `implementation_session_id` and `auditor_session_id` to appear in the handoff text; the handoff can name both auditors in prose. | Schema holds one id. |
| Q5 | `certification_publication_sha`: P6 used the certifier's pin-only commit; in MCP-RC the preflight certifier made no pin-only edit (`pin_only_edits: []`), the integration session made them in `c3f962636`. | Convention, not gate code. |
| Q6 | What "the window PDC-P7-00 is open" means and who records it; the DAG row names `OD-PDC-IIAC-SCOPE` plus an operator activation line as the precondition of PDC-P7-00. Whether the MCP-RC line signed in (a) is that line or a separate one. | No PDC-P7-00 opening record exists in the repository. |
| Q7 | Identifier format and provenance for the activation certifier and the signing delegate (P6 used the app session id; the MCP-RC manifests use labels such as `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13`, which is a designation, not a verified id). | Process. |
| Q8 | `operator_decision_class` value and the `notes` wording of the record. | Free text in the record. |
| Q9 | Handoff record CRLF requirement, file location and names of the activation evidence folder. | From P6 logs and convention, not from the gate code. |
| Q10 | Owner of the `[product]` xfail (`T-IGX-039-A`, "no named person yet"), operator acceptance of RA-02..RA-04 and AF-05; whether they must be answered before (a), before (b) or only before terminal verification. The re-audit lists "named owner or ticket for AF-05 and AF-16" as an open condition; the delegation does not make it a precondition of the signature. | Operator. |
| Q11 | Minor evidence note, not a gate item: both MCP-RC auditor manifests say the schema has no `PASS_SCOPED` value (`LIM-verdict-mapping`), but `iiac_audit_session_manifest.v1.schema.json` lists `PASS_SCOPED` in its `verdict` enum and the P6 product manifest used it. Auditor-authored files are not edited. | Orchestrator may ask the auditors or leave it. |
