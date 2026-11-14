# MCP-RC activation runbook (signing, two-phase rebind, PR, merge, verification) — integrated baseline

Role of this file: ordered hand-over for the sessions that come after the audits and the certifier preflights. Written by the activation-package drafting session, which is not the implementer, an auditor or a certifier. It carries no verdict, no signature, no rebind and no `Approved` line. It does not close MCP-SERVER-X, promote any FRZ status (counter stays 225) or claim production readiness. Revised 2026-11-14 for the integrated baseline; the first version (2026-11-13) assumed an un-merged `mcp-rc` line and is superseded.

Ticket: `governance/tickets/T-IGX-038-F.md`. Precedent: MCP P6 (`MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`: binding manifest commit `a65b45987`, phase 1 `92ad35557`, phase 2 `19c617365`, PR merge `66dcd1bfe`). Gate sources: `scripts/governance/iiac_activation_gate.py`, `verify_iiac_policy_activation.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`; corrections taken from `platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT.md` section 5 and `PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_1.md` sections 5 and 6.

Legend: **[gate]** read from the gate code (confirmed by the certifier preflights); **[P6]** copied from the P6 commits and logs, not shown by the gate code; **[OPEN]** not derivable, needs the operator, the orchestrator or an owner (collected at the end).

## Operator decisions that fix the order and scope

`OD-MCP-RC-03` (`Tak, czekaj na okno`) and `OD-MCP-RC-04` (window open; `Przepnij i zamroź (Recommended)`; `Tak, wzór P6 (Recommended)`; `Tak, odnawiam (Recommended)`; `PDC-X dodaje regułę na development (Recommended)`; renderings in those files are by the orchestrator, not operator-written), plus the chat answers `Rebind po merge'u`, `Tak, rozszerz` and `Orkiestrator zleca` to the drafting session (English renderings by the drafting session, not operator-written lines: "Rebind after the merge", "Yes, widen it", "The orchestrator commissions it"). Result: the merge of `development` into `mcp-rc` is done, the audits cover the integrated tree, the rebind happens once at the integrated baseline, and the activation line is signed only on the rewritten decision.

## State at drafting (2026-11-14, from git)

| Item | Value |
| --- | --- |
| Current pin (`audit_execution_sha`) | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`; record status `IN_FORCE`, `effective_from_sha` `92ad3555775649742299826e3680ec8ce703cc8a`; record unchanged since the campaign base `6a6279a49` |
| Audited product tree | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` (merge of `development` `a156cbbd5`, PR #525, the PDC-X `INTEGRATED_BASELINE_SHA`); product diff to the integrated head is empty |
| ACT-N11 union against the pin | 158 = 23 MCP-RC + 135 PDC-X (153 first audit, 156 addendum 1, 158 addendum 2) |
| `origin/mcp-rc` | `7801dcb1293e31ee0c483184efd59c01a2dc606b` (last MCP-RC product change `b0e19bf61b346e6941b2e190b703c6922e708b34`) |
| Evidence publications (pin-only edit commits) | MCP-RC `78d4c496071141ca4092b926b838fb80016f1b91` (`c3f962636`); PDC-P7-00 first audit and addendum 1 `6b195980d20067449ade0d2c2fd2280781c5a25d` (`ae5623efb`); addendum 2 `29d66ae3fcc7a06a919d7e07541740841e8abfe8` (`fde556183`) |
| Preflights | MCP-RC (`1b5f26f24`), PDC-P7-00 (`5c25d4d8f`) and addendum 1 (`7801dcb12`): all `CERTIFIER_PREFLIGHT_PASS_SCOPED` |
| Gate now | `verify_iiac_policy_activation --require-in-force` exits 1, only ACT-N11 product-path errors (158) |
| `origin/development` | no product-path change after `a156cbbd5` (checked at drafting); the F-41 impact-map rule is **not** on it yet |
| Unsigned decision | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (`Decision status: DRAFT`), digest in `mcp_server_x/operator_decisions/DELEGATION-MCP-RC-2.md` |

Rules for every session: one process at a time on the shared Windows workstation (`uv run` once per verification step); logs under `.tmp/session/<task>/`, published evidence under `iiac_evidence/`; commits carry `Ref: governance/tickets/T-IGX-038-F.md` and no AI attribution lines (repo rule); no product-path edit by hand anywhere (a product edit voids the audit and the pin, ACT-N11); IMPLEMENTER != AUDITOR != CERTIFIER, and the signing delegate and the activation certifier are neither.

## Step 0. Before anything is signed: the F-41 cycle and the red tests

1. **F-41 (impact map).** PDC-X / ENT-AUDIT owner adds the rule for the `applications/*` roots on `development` (one-time exception to the freeze, `OD-MCP-RC-04` section 5; no test narrowed). Then `mcp-rc` merges `development` again. Prove with git: the 23 MCP-RC paths still byte-identical to `b0e19bf61`; the product diff against `324d9dba3` is exactly the files the rule changes and nothing else; no `MATERIAL_GOVERNANCE_PATHS` path changed (six files in `iiac_activation_gate.py`).
2. The independent auditor adds addendum 3 for the changed map files; the certifier adds a preflight addendum; an evidence-integration session publishes both rounds together (`OD-MCP-RC-04` section 5; evidence integration of addendum 2 waits for this).
3. The drafting session (or the orchestrator) revises the decision with the new tree, SHAs and digests, and records the new digest in an addendum to `DELEGATION-MCP-RC-2.md`. **If the rule does not land before the rebind** ([OPEN]), the decision as drafted stands, with F-41 listed as an open condition that keeps two tests red on the integrated tree.
4. Test-only repairs by their owners on `development` (`OD-MCP-RC-04` section 2: F-35, F-09; F-07 already passes per the certifier). F-42 (`embedding_provider`, inside `ent-audit-fast` through `VBN-REACHABILITY`) has no owner [OPEN]. Each repair that touches no product path needs no addendum; any product commit on `development` does (the freeze is a declaration, not a control).

## (a) Designated signing session: the activation line

New, narrow session under `DELEGATION-MCP-RC-2` as renewed by Addendum 2 (activation line only, 24 hours from the start of this session, no rebind, no merge). Not the activation certifier.

1. Record the session's own start time and the expiry in the delegation record. Start from the commit that carries the final decision text and the delegation addendum with its digest. Run the read-only checks listed in `DELEGATION-MCP-RC-2.md` (audits and preflights present; digest equals the table; draft not stale per check 5; no activation line yet; activation record unchanged). Any failure: write nothing and report.
2. Ask the operator for the direct confirmation (P6 practice); record it verbatim with a labelled English rendering.
3. In the decision file only: replace the standalone placeholder line (sixteen underscores, alone on its line) by `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp from its own clock>`; `Decision status: DRAFT` to `Decision status: AUTHORIZED`; the `Drafting note` paragraph by a `Provenance note` (P6 wording). **[gate]** the line matches `^Approved IIAC policy activation IIAC-2026-11-05\.1: operator @ <ISO-8601>$` and is a whole line of the file; the file must contain exactly the lines `Decision status: AUTHORIZED`, `Decision ID: <record operator_decision_id>`, `Policy version: IIAC-2026-11-05.1`, `Decision scope: IIAC policy activation...` and `Audit report digest: <record digest>`; it must not contain the sixteen-underscore string, the words "NOT SIGNED" in capitals or `Decision status: PENDING` (ACT-N06/N07); the file name must start with `IIAC_OD_ACTIVATION_` and contain neither `PACKET` nor `DRAFT` in any case (ACT-N06); no `EVOL-X`, `ENT-AUDIT-X` or `ARCH-FREEZE` within 80 characters before `CLOSED` (ACT-N18, also scanned in the record). The unsigned draft already satisfies the non-signature lines (checked at drafting).
4. Commit on its own branch (named by the orchestrator) with the `Ref` trailer; no rebind, no merge, no edit of `IIAC_POLICY_ACTIVATION_RECORD.json`. Report: commit SHA, the written line, the start time, and which open conditions are still open. Stop.

## (b) Activation certifier: binding manifest, handoff record, phase 1, phase 2

A new dedicated session (not the signing delegate, the preflight certifier, an auditor, the implementer or an evidence-integration session). Precondition: (a) is committed and its branch is in the history the certifier works on; the window is open (`OD-MCP-RC-04` section 1). Everything below is the certifier's work; this runbook and the drafting session produce none of it.

### b1. Pre-rebind commit (P6 `a65b45987`: manifest, handoff, validator log)

Evidence folder proposal `iiac_evidence/PDC-X/P7-00/activation/` **[OPEN: names and place are proposals]**.

**Binding manifest** (stage `IIAC-GOV-1` **[gate]**; the MCP-RC and PDC-X identity goes into `limitations`). `verdict` = `PASS` **[gate: ACT-N05 rejects `CONDITIONAL` and `PASS_SCOPED`]**, authored by the certifier from `PASS_SCOPED` audits as the operator recorded (`OD-MCP-RC-04` section 3); the auditors' manifests are not rewritten.

| Field | Value or source |
| --- | --- |
| `stage_id`, `certification_profile`, `policy_version` | `IIAC-GOV-1`, `iiac-v1`, `IIAC-2026-11-05.1` |
| `implementation_session_id`, `implementation_agent_id` | candidate `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (addendum 2), `implementer` **[OPEN Q4]**; the id must appear verbatim in the handoff text **[gate]** |
| `auditor_session_id`, `auditor_agent_id` | one id, `independent-quality-auditor`; candidate `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-2` **[OPEN Q4]**; also verbatim in the handoff **[gate]**; name all other auditor ids in the handoff prose |
| `certifier_session_id`, `certifier_agent_id` | the activation certifier's own verifiable id, `certification-governance-agent`; differs from implementer and auditor **[gate]** |
| `audited_implementation_sha` | equals the record's `corrective_implementation_sha` **[gate]**; candidate `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` (product tree identical to the head) |
| `test_execution_sha` | candidate `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` |
| `audit_execution_sha`, `evidence_publication_sha` | equal the record's **[gate]**; P6 used the evidence publication SHA for both; candidate `29d66ae3fcc7a06a919d7e07541740841e8abfe8`; moves if addendum 3 is made |
| `certification_publication_sha` | equals the record's, differs from `audited_implementation_sha` **[gate]**; candidate `7801dcb1293e31ee0c483184efd59c01a2dc606b` (the preflight addendum commit; P6 used the certifier's publication commit); moves with addendum 3 **[OPEN Q5]** |
| `audit_report_digest`, `audit_scope_manifest_digest`, `test_matrix_digest`, `findings_digest` | SHA-256 of the working-tree bytes of the files at the record's four artifact paths **[gate]**. With the P6 pattern (record keeps the `IIAC-GOV-1` set): `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e`, `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f`, `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b`, `df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831` (recomputed at drafting) |
| `verdict` | `PASS` **[gate]** |
| `limitations` | scope 158 = 23 + 135; the five audit dossiers' `PASS_SCOPED` and their manifests' `CONDITIONAL`; the red-test caveat (F-35, F-09, F-42, F-07 status, F-41); the five CI contexts not verified; IA-D1..IA-D8 not done and 27 routed findings; AF-05/AF-16 and RA-02..RA-04 open; drift/freeze caveat; "does not close MCP-SERVER-X, promote FRZ or claim production readiness" **[P6]** (text of the decision, section "Open conditions") |
| `assurance_level`, `handoff_record_path`, extras | `process_enforced`; handoff path; optional `activation_audit_report_path`, `product_session_manifest_path` (list the MCP-RC and PDC-P7-00 manifests; the schema allows extra properties) **[P6]** |

**Evidence paths [gate: ACT-N14, correction of the first runbook].** All **five** of `audit_report_path`, `session_b_manifest_path`, `audit_scope_path`, `test_matrix_path`, `findings_path` must sit under `docs/project/maintainers/qualification/iiac_evidence/`. The MCP-RC dossiers (`mcp_server_x/`) and the PDC-P7-00 dossiers (`platform_domain_completeness/`) therefore cannot be `audit_report_path`; the P6 pattern keeps the `IIAC-GOV-1` report set already in the record, whose digest `c405c65f...` the decision cites. The report bytes at the working tree must equal the bytes at `evidence_publication_sha` [gate: ACT-N11 report check]. If the certifier binds other artifacts it must place certifier-authored copies under `iiac_evidence/`, cite them, and the decision's `Audit report digest` line and digest table are revised before signing.

**Findings register [gate: ACT-N15].** The bound `findings_path` is a JSON object with a `findings` array and no `BLOCKER`/`CRITICAL` entry that is not `RESOLVED`/`CLOSED`. Kept `IIAC-GOV-1` set: passes (P6). Of the audits' files: the MCP-RC first audit and the three PDC-P7-00 files pass; the MCP-RC re-audit file has no `findings` array and fails.

**Handoff record [gate]** named `IIAC_SESSION_HANDOFF_RECORD_*.md` (not the template) under `docs/project/maintainers/qualification/`; text contains `IIAC-2026-11-05.1`, the manifest's `implementation_session_id` and `auditor_session_id`; the record stores its SHA-256 (working-tree bytes). **[P6, from a certifier log, not from gate code]** the P6 handoff was written with CRLF line endings (an LF-only file made the W4-F01 tamper test a no-op); the preflight did not verify this.

**Validators and logs** (command, HEAD, UTC, exit code in each log; P6 names `certifier_activation_*.log`):

```text
python -m scripts.governance.verify_iiac_audit_manifest <manifest> --repo . --require-handoff-template
python -m scripts.governance.verify_iiac_role_separation <manifest> --strict-product-diff --repo .
```

Also log the recomputed deltas and digests: the 158 paths against the pin, every digest used, the product diff between `audited_implementation_sha` and `audit_execution_sha` (must be empty).

### b2. Phase 1 commit (P6 `92ad35557`: only the record changes)

Edit `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json` (the drafting session and the signing delegate never do):

| Field | New value |
| --- | --- |
| `effective_from_sha`, `policy_activation_sha` | empty strings **[gate]** |
| `corrective_implementation_sha` | candidate `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` (= manifest `audited_implementation_sha`) |
| `audit_execution_sha`, `evidence_publication_sha` | candidate `29d66ae3fcc7a06a919d7e07541740841e8abfe8` |
| `certification_publication_sha` | candidate `7801dcb1293e31ee0c483184efd59c01a2dc606b` |
| `session_b_verdict` | `PASS` |
| `session_b_manifest_path`, `session_b_manifest_digest` | the binding manifest and the SHA-256 of its working-tree bytes (`iiac_evidence/**` is `-text`) |
| `handoff_record_path`, `handoff_digest` | the handoff and the SHA-256 of its working-tree bytes |
| `operator_decision_path`, `operator_decision_id` | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md`, `IIAC-OD-MCP-RC-ACT-N11-2026-11-13` |
| `operator_activation_line` | the signed line from (a), copied byte for byte **[gate]** |
| `operator_decision_class`, `notes` | free text; proposal `OD-MCP-RC-ACT-N11` and a phase-1 note |

Unchanged with the P6 pattern: `status` `IN_FORCE`, `governance_implementation_sha` `2ae1014116d353bc7742985e62f2bccc2bfac798`, `session_b_governance_audit_report_digest` `c405c65f...`, `audit_report_path`, `audit_scope_path`, `test_matrix_path`, `findings_path`. **[gate]** ancestry: `governance_implementation_sha` -> `corrective_implementation_sha` -> `audit_execution_sha` -> `evidence_publication_sha` -> `effective_from_sha`, and `certification_publication_sha` -> `effective_from_sha`; `certification_publication_sha` != `corrective_implementation_sha`. Checked at drafting with git for the candidates (`2ae101411` -> `324d9dba3` -> `29d66ae3f`; `7801dcb12` and `29d66ae3f` ancestors of the head; product diff `324d9dba3..29d66ae3f` empty; product diff `324d9dba3..HEAD` empty). No `MATERIAL_GOVERNANCE_PATHS` path may change after `audit_execution_sha`.

Expected at the phase-1 commit **[P6 log]**: `verify_iiac_policy_activation --require-in-force` FAIL with exactly four errors (both fields missing and not 40-hex). ACT-N11 lines or digest errors at this point are defects: stop and report.

### b3. Phase 2 commit (P6 `19c617365`: two lines)

Set `effective_from_sha` and `policy_activation_sha` to the phase-1 commit SHA, nothing else **[gate]**: equal to each other, not equal to `HEAD`, the phase-1 blob carries the same ten binding fields (`BINDING_KEYS`), the working-tree record equals `HEAD`. Expected: exit 0, `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from <phase-1 SHA>`. Log it as in P6 (`certifier_activation_phase1_phase2_validator.log`). The certifier writes its own short report in the evidence folder and states in it the scope limits and the open items of the decision.

## (c) Pull request to `development`

Who opens it and from which branch is [OPEN Q6]: the PR head must contain the signed decision, the binding manifest, the handoff and both rebind commits, and descend from `mcp-rc` (`OD-MCP-RC-01` D-1b). No auto-merge. Pin the head: `gh pr view <n> --json headRefOid` equals the pushed SHA, compared again after every push (a PR was once merged without its last fix commit). Before pushing a new branch set its upstream (`git branch --set-upstream-to=origin/development`) and push with an explicit refspec; the pre-push ticket-reference gate otherwise scans all history.

Wait until the five contexts are `completed/success` **on that exact head**: `ci-smoke`, `governance-fast`, `evol-x-b4-preflight` (`unit-tests.yml`), `ent-audit-fast`, `ent-audit-pr-gate` (`enterprise-audit.yml`); `governance-policy-pin` also exists and was green in P6. `governance-fast` is red while ACT-N11 fires, so the rebind must already be in the PR head. **Known red inside these contexts (from the preflights, not re-run here):** F-42 (two `embedding_provider` tests inside `ent-audit-fast` via `VBN-REACHABILITY`), the 15 fresh-clone fixtures of `governance-fast` on the Windows workstation, and the ACT-N11 gate tests until the rebind; F-35 and F-09 are outside the selections by reading, F-07 passes at the head. The CI ACT-N11 evaluation runs on the merge ref, which contains the current `development`: any product commit there after `a156cbbd5` adds paths and voids the pin until another addendum and re-pin. A run `in_progress` is not success (P6 PRs #459, #464, #468 were merged with checks pending; structural cause `OPEN-BP-01`). Record the run ids.

## (d) Merge

Only together with the rebind, with the five contexts green on the pinned head and `headRefOid` unchanged. The merge is a separate act that needs the operator's instruction at that moment; this runbook does not authorize it. If a context is red for a reason on the known list above, stop and report; do not merge on the argument that it was red before.

## (e) Terminal verification (separate session)

New session, not the implementer, an auditor, the signing delegate or the activation certifier (P6 analogue `MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md`; MCP-RC file name [OPEN]). Minimum: merged `development` SHA; `verify_iiac_policy_activation --require-in-force` exit 0 on it; record fields against section (b); the five contexts and run ids on the PR head and, where a push run exists, on the merge; empty `git diff` of the 23 MCP-RC paths against `b0e19bf61`; product diff of the merged tree against the pinned tree; no FRZ change (counter 225), no P8 edit, no ruleset change; the open conditions of the decision with their then-current state; the P6 process observations (merge while checks pending, `paths-ignore` push runs). Ticket lifecycle (`T-IGX-038-F`, naming the owner in `T-IGX-039-A`) is the operator's or a commissioned step.

## Open questions and requirements still not derivable

| # | Open item | Note |
| --- | --- | --- |
| Q1 | Order versus the 125 (now 135) non-MCP paths | **Answered**: rebind after the merge, scope widened, orchestrator commissions the integrated audit (operator chat answers); audits and preflights now exist. |
| Q2 | Pins | Candidates in (b) are derived from git and checked for ancestry and empty product diff; they move if addendum 3 is made. |
| Q3 | Which record artifacts | Preflight confirms the P6 pattern works (keep the `IIAC-GOV-1` set, all five under `iiac_evidence/`); binding other artifacts needs certifier-authored copies and a revised decision. Certifier's choice. |
| Q4 | Single `implementation_session_id` and `auditor_session_id` in the binding manifest | Schema holds one each; MCP-RC and PDC-X have several. Certifier chooses and lists the rest in the handoff text. |
| Q5 | `certification_publication_sha` | P6 used the certifier's publication commit; here the preflight addendum commit is the candidate. |
| Q6 | Which branch carries the package, the signature and the rebind into the PR, and who opens the PR | Not stated by any record. |
| Q7 | Identifier format and provenance of the signing delegate and the activation certifier | Labels such as `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` are designations, not verified application ids. |
| Q8 | `operator_decision_class` and `notes` wording | Free text. |
| Q9 | Handoff CRLF requirement; names and place of the activation evidence folder | From P6 logs and convention, not from gate code. |
| Q10 | Owners and acceptances | Owner of the `[product]` xfail (`T-IGX-039-A`), operator acceptance of RA-02..RA-04 and AF-05 (listed in the decision; not preconditions of the signature), D-10, P8 documentation owner (G-DOC-01..12), host composition (F-04, F-05, F-37). |
| Q11 | F-42 owner | `embedding_provider` / `ollama` catalog registration inside `ent-audit-fast`; probably the known B-REG-P item; unowned. |
| Q12 | F-41 timing | The rule is not on `development`; whether it lands before the rebind decides whether addendum 3 and a decision revision are needed. |
| Q13 | Five CI contexts | Not verified green anywhere; known reds listed in (c). A merge needs green on the pinned head. |
| Q14 | Window and freeze | The window is declared open by the operator; the freeze on product paths is a declaration, not enforced. |
| Q15 | Evidence note, not a gate item | The MCP-RC auditor manifests say the schema lacks `PASS_SCOPED`; `iiac_audit_session_manifest.v1.schema.json` lists it. Auditor files are not edited. |
