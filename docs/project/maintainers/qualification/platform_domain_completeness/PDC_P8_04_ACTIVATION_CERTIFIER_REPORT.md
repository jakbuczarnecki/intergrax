# PDC-X P8-04 activation certifier report (ACT-N11 rebind at the global audit SHA)

Ticket reference: `governance/tickets/T-IGX-040-C.md`. Role: CERT-PDC certifier, new dedicated session `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791` (agent id `certification-governance-agent`), started 2026-11-16 by the orchestrating session `local_80099b72-d308-4d1a-a3cb-99adbd0abe51` on branch `p8-04-rebind` cut from `origin/development` `859d64c084c527d122d7d5a8ab86f77fb9489c87`. This session is not an implementer, IA-D1..IA-D8, IA-GLOBAL (`PDC-P8-IA-GLOBAL-INDEPENDENT-AUDITOR-6573093f-9178-4c2c-93a3-64b0a5b6c657`), a signing delegate or the activation delegate. `process_enforced`: one git identity signs all commits; the separation is an attestation.

## Verdict

**Certifier verdict on the binding manifest: `PASS`, scoped.** It is the certifier's own `PASS`, authored under `operator_decisions/DELEGATION-PDC-P8-ACT.md` (b) from the IA-GLOBAL final verdict **PASS_SCOPED** at `969cbd964` (`PDC_INDEPENDENT_AUDIT.md` section 15; the gate accepts only `PASS`, ACT-N05). It is not an auditor verdict. The limitations in the manifest (19 entries) are part of the verdict: the four Operator-accepted exceptions (G-05 residual, E1 to E11, G-10 `trace_bridge.v1`, G-28), the exceptions of `DELEGATION-PDC-CLOSE.md` decision 1, the failed EBH-7 L3 replay (run 43738431866, not a B6 PASS, not re-run), the recomputed product-path set, and the sentence that this is not a verdict on B6, not a certification of the whole PDC-X stage and not an FRZ or roadmap action.

**Mechanical result (certifier-side):** both validators pass on the binding manifest and a read-only dry run of its gate-side conditions passes (below). The activation gate itself was **not** run against a record, because writing the record is the activation delegate's step.

## SHAs

| Item | SHA |
| --- | --- |
| Starting tip (`origin/development`, PRs #597, #598, #599 merged) | `859d64c084c527d122d7d5a8ab86f77fb9489c87` |
| `governance_implementation_sha` (unchanged) | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| `G` = `corrective_implementation_sha` = manifest `audited_implementation_sha` = `test_execution_sha` | `969cbd964c679d563b02e55f065b452d46fdf90b` |
| `E` = `audit_execution_sha` = `evidence_publication_sha` | `a73d15ab341b30e9748aad03894f6faa75ad4e09` (merge of PR #597) |
| `C` = `certification_publication_sha` | `fd2ad28efde6216be38194b020188d116264adb6` (matrix, preflight, pre-check scripts and logs 01 to 02) |
| `B` (binding manifest, handoff, logs 03 to 05, this report) | the commit that carries this file (reported in the PR; a file cannot hold its own SHA) |
| `effective_from_sha` = `policy_activation_sha` | not set; phase 1 and phase 2 are the activation delegate's |

## Digests (SHA-256 of working-tree bytes)

| Artifact | SHA-256 |
| --- | --- |
| Binding manifest `iiac_evidence/PDC-X/P8/activation/iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json` | `77fa2babd02df6fbbd2e0d5b34413a769e069ac293e485daf49ec9a779dc476f` |
| Handoff `iiac_evidence/PDC-X/P8/activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_P8_04_ACTIVATION.md` (CRLF, `-text` path) | `a36b2ac35eaf0782f4d40ef9b76a4dd745b0c4e70d839e2df2a7fa819b1fd121` |
| Bound `IIAC-GOV-1` report, scope, matrix, findings (unchanged, kept) | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e`, `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f`, `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b`, `df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831` |

The digests of the manifest and handoff stay valid only while those files are not edited after this commit.

## Choices and why

- **`E` = merge of PR #597, not `G`, not the tip.** The gate chain is `governance -> G -> E -> phase 1` and `C -> phase 1`; `E` must be a descendant of `G` with an empty product diff and must hold the five bound artifacts at their final bytes. `#597` is the first commit where the global audit (report, manifest, scope, findings, test matrix, Operator exception record) is final. The later PRs #598 and #599 change no bound artifact, product path or material governance path (checked), so either would pass; the earlier choice keeps `E` tied to the evidence it publishes. The global manifest digests equal the bytes at `E` (logged).
- **`C` is a commit of this branch** (matrix plus preflight, docs only): descendant of `E`, different from `G` (gate rule), ancestor of the phase-1 commit that will follow on the same branch.
- **Bound set kept** (`IIAC-GOV-1`, runbook 5.1): the gate needs all five bound paths under `iiac_evidence/` (ACT-N14); `PDC_INDEPENDENT_AUDIT.md` sits outside it. Kept bytes mean the digest `c405c65f...` that the next decision file must cite is unchanged; the activation record's bound paths need no edit.
- **Product-path set recomputed** with the gate's `product_paths_changed_between`: **74** paths `a1b3053b4..G` (all `intergrax/`; none under `agents/` or `applications/`), **192** paths from the P6 pin to `G`. Not the 62 of the runbook draft, which predated the T-IGX-041-M, N, O and P changes.
- **Single `auditor_session_id`** = IA-GLOBAL; the eight domain auditors and all implementer lanes are listed in the handoff. Implementer ids are labels, not verified application ids.
- **Handoff in CRLF** under a `-text` path (precedent); verified by `file` and `git check-attr`.

## What I verified myself (logs in `iiac_evidence/PDC-X/P8/activation/logs/`, scripts in `.../scripts/`)

1. `p8_04_01_precheck_preC.log` and `p8_04_03_precheck_withC.log` (script `precheck_p8_04.py`, gate functions imported from this checkout): ancestry; the product diffs `G..E`, `G..C`, `E..HEAD`, `G..HEAD` empty; no `MATERIAL_GOVERNANCE_PATHS` change after `E` or `G` (the protocol last changed in `7b1c0a4b7`, an ancestor of `E`); the four bound artifacts equal the record digests and the bytes at `E`; the findings register passes ACT-N15 (4 entries); the global audit manifest digests equal the bytes at `E`; no non-docs path between `G` and HEAD. Exit 0 each.
2. `p8_04_02_auditor_manifests_d1_d8_global.log`: `verify_iiac_audit_manifest --require-handoff-template` and `verify_iiac_role_separation --strict-product-diff` on D1..D8 and the global manifest: 18 calls, all `PASS`, exit 0.
3. `p8_04_04_binding_manifest_validators.log`: the same two validators on the binding manifest: `PASS`, `PASS`, exit 0.
4. `p8_04_05_binding_manifest_dryrun.log` (script `check_binding_manifest.py`): the gate's schema validator, `PASS` verdict, SHA equalities, ancestry, ACT-N10 and ACT-N17 conditions, digests against the record's bound files, handoff ids verbatim and CRLF: all OK. This is a dry run of manifest-side conditions only.
5. Environment: the main checkout's `.venv` Python (3.12.11, has `jsonschema`) with `PYTHONPATH` set to the worktree and the worktree as cwd; each log shows the interpreter. One process at a time; no pytest was run.

## What I took from the audits (not re-done)

The IA-GLOBAL verdict and every row of its sections 7, 13, 14 and 15, the domain verdicts of the eight P7 reports (verdict lines read, not the rows), the plan of T-IGX-041-M (E1 to E11), the Operator records, the merge state of PRs #593 to #599 as `gh` reported. I did not re-audit any product path, run any product test, or run any CI context. The auditor's net count of repaired rows (63 fixed, 13 partial) differs by one from my recount of its tables (64 and 12); I did not resolve it and it changes no disposition.

## Matters for the activation delegate and the orchestrator

1. **Nothing in my artifacts blocks the activation** that I found. The delegate must still run its own confirmations (`DELEGATION-PDC-P8-ACT.md` Limits): no product delta between `G` and the activation HEAD, the validators, the published global audit, protocol and Amendment 01 permit it.
2. **The signed MCP-RC activation line cannot be reused silently.** `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` belongs to the previous binding and the Operator's statement that "the signature stands" (`OD-MCP-RC-05`) covers re-pin 2 only. For P8-04 the route of runbook 6.1 applies: a new `IIAC_OD_ACTIVATION_PDC_X_P8_<date>.md`, decision id, `Audit report digest: c405c65f...7a4e`, the plain-form line written by the delegate from its own clock (named by `DELEGATION-PDC-P8-ACT.md` (c)), no `NOT SIGNED`, no sixteen underscores, no `Decision status: PENDING`, no `EVOL-X`, `ENT-AUDIT-X` or `ARCH-FREEZE` within 80 characters before `CLOSED`.
3. **Phase 1 values** (runbook 6.2): `corrective_implementation_sha` `G`; `audit_execution_sha` and `evidence_publication_sha` `E`; `certification_publication_sha` `C`; `session_b_manifest_path` and `session_b_manifest_digest` from the table above; `handoff_record_path` and `handoff_digest` likewise; `session_b_verdict` `PASS`; the four bound artifact paths and `session_b_governance_audit_report_digest` unchanged. Phase 1 must contain an `IN_FORCE` blob with empty `effective_from_sha` and the same ten binding keys.
4. **Freeze and ordering.** The product merge hold must stay until the P8-04 PR is merged; merge with a **merge commit** (a squash destroys the phase-1 SHA). Before the first push of any further commit set the upstream to `origin/development` and push with an explicit refspec. Every file of P8-04 lives under `docs/project/maintainers/qualification/`; do not edit `governance/tickets/**` (EBH-7 identity gate).
5. **Known red classes on the PR:** the three IIAC activation files until phase 2 is in the head, the EBH-7 identity gate (the L3 replay failed; no B6 claim). I did not verify any CI context green.
6. **Matrix housekeeping not done here:** `PDC_INDEX.md` still lists the matrix as "Not started" and the SHA ladder rows for `GLOBAL_AUDITED_SHA`, `AUDIT_EXECUTION_SHA`, `EVIDENCE_PUBLICATION_SHA` and `CERTIFICATION_PUBLICATION_SHA` are `—`; both are PDC-P9-02 tasks and were left untouched.

## Not done, not claimed

Not stage closure, not FRZ promotion, not a verdict on B6, not production readiness; C10/C15 untouched. No activation decision file, no `Approved` or activation line, no record phase, no merge. No product, test, script (outside the new evidence scripts), ticket, `.github`, gate, verifier, protocol, record, delegation or auditor-file edit. Draft pull request only.
