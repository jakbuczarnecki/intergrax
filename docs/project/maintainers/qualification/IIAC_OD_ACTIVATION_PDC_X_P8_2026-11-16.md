# IIAC operator decision — PDC-X P8-04 ACT-N11 rebind at the global audit SHA

Decision ID: IIAC-OD-PDC-X-P8-ACT-N11-2026-11-16
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation (ACT-N11 rebind of the PDC-X product tree at the global audit SHA `969cbd964c679d563b02e55f065b452d46fdf90b`, 74 product paths versus `a1b3053b49c28124a0abde679d733f21a597797f`)
Decision status: AUTHORIZED
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-16T09:06:54+01:00

Delegation: `docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P8-ACT.md` at `3529217dde07f41389d539d6603b4f2253e7b8f2`, part (c).

## Provenance of the line (disclosure)

The line above was written by an operator-designated session, the PDC-X P8-04 activation delegate (IIAC Amendment 01, R-1..R-6, safeguard S-2), not by the Operator personally. The commission is the Operator's recorded answer "Yes, I approve (a)-(c)" (original wording in Polish: "Tak, zatwierdzam (a)-(c)"), transcribed in `DELEGATION-PDC-P8-ACT.md`; part (c) names the plain form `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` written by the activation delegate. The authority chain is `DELEGATION-PDC-STANDING.md` answer 6 (`9a9e28963d6e2c2f6392ee0de40b92043ca5ffd8`), `DELEGATION-PDC-CLOSE.md` decision 3 (`7857632dd064e9fd6583d975d781cca516c5bca5`) and Amendment 01 (S-3 struck, S-2 kept: the plain form is written because the commission names it). The line is in the plain form because the IIAC activation verifiers accept only that form; it is not claimed to be operator-written. The timestamp is the activation delegate's own system clock at the moment of writing. The delegate is a new session dedicated solely to this activation step; it is not a continuation of another session and not an implementation session, and it is not an implementer, auditor (IA-D1 to IA-D8, IA-GLOBAL) or certifier (CERT-PDC) of this campaign. The assurance level is `process_enforced` (one git identity signs all commits; the separation is an attestation).

The date in the Decision ID is the date on which the activation delegate wrote this decision.

## What this decision binds

It authorizes, for the activation gate only, the binding of policy `IIAC-2026-11-05.1` to the following SHAs (values of `IIAC_POLICY_ACTIVATION_RECORD.json`, phase 1 and phase 2 of the two-phase publication):

| Field | SHA |
| --- | --- |
| `governance_implementation_sha` (unchanged) | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| `corrective_implementation_sha` (audited tree, test execution) | `969cbd964c679d563b02e55f065b452d46fdf90b` |
| `audit_execution_sha` and `evidence_publication_sha` (merge of PR #597) | `a73d15ab341b30e9748aad03894f6faa75ad4e09` |
| `certification_publication_sha` | `fd2ad28efde6216be38194b020188d116264adb6` |
| binding manifest (verdict `PASS`, SHA-256) | `77fa2babd02df6fbbd2e0d5b34413a769e069ac293e485daf49ec9a779dc476f` |
| session handoff record (SHA-256) | `a36b2ac35eaf0782f4d40ef9b76a4dd745b0c4e70d839e2df2a7fa819b1fd121` |

The bound `IIAC-GOV-1` report set is unchanged; the audit report digest cited above is the digest of that set's report, as the certifier report cites it.

## What a signature on this record means (bound verdict)

The activation gate accepts only the verdict `PASS` in the binding manifest (ACT-N05). The global audit carries the final verdict **PASS_SCOPED** at `969cbd964` (`PDC_INDEPENDENT_AUDIT.md` section 15.2). By the Operator's recorded answer `DELEGATION-PDC-P8-ACT.md` (b), the certifier session authored the binding manifest with `PASS` from that `PASS_SCOPED` audit, with every scope limit and declared exception in the manifest `limitations` (19 entries). The `PASS` is the certifier's own, not the auditor's. This decision does not change the meaning of the gate.

## Scope limits and declared exceptions (carried from the binding manifest)

- ACT-N11 binding only. It is not a verdict on B6, not a certification of the whole PDC-X stage, not an FRZ promotion, not a roadmap action and not a production-readiness claim.
- Operator-accepted declared exceptions (`DELEGATION-PDC-EXCEPTIONS.md`, `b0b43e5d68fbe349f2baf308711812d59987cce5`): the G-05 residual (77 red incident scenario tests), E1 to E11 (the product MEDIUM findings excluded by T-IGX-041-M), the G-10 residual (`trace_bridge.v1` mismatch) and G-28 (Cypher/Neo4j graph store). The exceptions of `DELEGATION-PDC-CLOSE.md` decision 1 stay as declared. LOW and INFO findings go to the debt register of the P9 package.
- The EBH-7 L3 replay at pin `d39a520dd` (run 43738431866) failed and is recorded as not a B6 PASS; it is not re-run and there is no new pin. The EBH-7 identity gate stays a known red class.
- The audit evidence at `969cbd964` is the targeted tests of the audit section 14.4 only; ECAP FULL, the contract suites and the combined pytest last ran at `d39a520dd`.
- Any product-path commit after `969cbd964` voids this binding until a further audit addendum and re-pin. Docker, new vendor integrations and production configuration (C10/C15) are untouched.
- No stage closure and no FRZ promotion are made or implied by this decision. The Operator's own signature is still required for the closure lines (PDC-P9-03, PDC-P9-04). The merge of the pull request that carries the activation record is done by the orchestrating session with a merge commit (`DELEGATION-PDC-P8-ACT.md` (a)); the activation delegate does not merge.

## Confirmations made by the activation delegate before writing the line

Logs (command, HEAD, UTC start and end, exit code) are under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/P8/activation/activation_delegate/logs/`, scripts under `.../activation_delegate/scripts/`:

1. No product-path delta between the audited SHA `969cbd964` and the activation HEAD, by the gate's own helper `product_paths_changed_between` (`G..E`, `G..C`, `E..HEAD`, `G..HEAD` all empty); none of the six `MATERIAL_GOVERNANCE_PATHS` changed after `E` or after `G`.
2. `verify_iiac_role_separation --strict-product-diff` and `verify_iiac_audit_manifest --require-handoff-template` pass on the binding manifest; the certifier's dry run of the manifest-side gate conditions passes at the same HEAD. `verify_iiac_policy_activation --require-in-force` on the record as it stood before this rebind fails on ACT-N11 for the previous binding, which is the staleness this rebind removes.
3. The global audit is published (`PDC_INDEPENDENT_AUDIT.md` section 15.2: PASS_SCOPED, present on `origin/development`) and the binding manifest verdict is `PASS`.
4. The IIAC protocol (Policy activation, item 3: the plain operator-form line) and Amendment 01 (S-3 struck; S-2: the plain form where the commission names it; R-6) permit a delegated activation line, and `DELEGATION-PDC-P8-ACT.md` (c) names the plain form.
