# MCP-SERVER-X P8 — IIAC certification record (MCP-P8-05)

**Ticket:** [`T-IGX-035-I`](../../../../../governance/tickets/T-IGX-035-I.md) · **Policy:** `IIAC-2026-11-05.1` (profile `iiac-v1`, `IN_FORCE`) · **Role:** `certification-governance-agent` · **Assurance level:** `process_enforced`

## Verdict

**`CERTIFIED_PASS_SCOPED`** — the process and the records of the independent P8 audit validate. This is internal-agent certification of the auditor's `PASS_SCOPED`; it is not stronger than that verdict, and it is not a re-audit of the product. The certifier did not run the P7 replay or any `pytest` (see "What the certifier did not verify").

What the certification covers: the audit manifest and its artifacts are internally consistent and match git; the three sessions (implementer, auditor, certifier) are distinct; the audit window has no product-path delta; every finding has a recorded disposition; no mandatory blocker is open; the IIAC activation gate is in force.

What it does **not** do: close MCP-SERVER-X (the closure decision `OD-MCP-P8-CLOSE-01` is conditional on C1–C10, including an independent `P8_TERMINAL_VERIFIED` report that does not exist yet), close or touch ARCH-FREEZE, promote any FRZ status, claim production readiness, or touch PROD-Q Path B.

## Sessions

| Role | Id | Agent |
| --- | --- | --- |
| Implementer (P7 lanes L1, L2, L3, FI) | `T-IGX-035-H-MCP-P7-IMPLEMENTERS` (label) | `implementer` |
| Independent auditor | `e96831ad-2c06-4753-b8e9-e6aca67c2f5b` (taken by the auditor from its scratchpad directory name; host session `local_b3377a14-67f6-4a51-9a16-289a2162a1f7`) | `independent-quality-auditor` |
| Certifier | `local_f418d991-1c34-42a4-bc9d-d96734951f42` | `certification-governance-agent` |

The three ids differ (checked by both verifiers). The certifier authored no P2–P7 file, test or record, and no auditor artifact.

## SHA table

| Item | SHA |
| --- | --- |
| `audited_implementation_sha` = `IMPLEMENTATION_SHA` = `audit_execution_sha` | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` (merge of PR #474; PR head `94b6f6d6b8ae9882568c778fd7180b8b958fc175`) |
| `test_execution_sha` (P7 package) | `abbe1525450da77da6801b080619a600c20cd497` |
| P6 terminal head (scope baseline) | `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` |
| IIAC activation (`IN_FORCE` from) | `92ad3555775649742299826e3680ec8ce703cc8a` |
| Audit evidence commit | `69a2333cc67071656f1e573a35339ade9a8b5b7b` |
| P8 corrections (docs only) / auditor re-review SHA | `c50547012c353463e86525411a5a065f43e5bab8` |
| Auditor re-review commit (digests in the manifest are of this state) | `0c69c20652d5a78229a672ba0ae743aa8fa9a9c3` |
| Operator disposition of F-P8-01 recorded (PR #477 head) | `32a50991d1cef4eaf68b727f953d138890f1b461` |
| `evidence_publication_sha` = merge of PR #477 on `origin/development` (certified head) | `a618a03a8b245ceefef6ab5a922213aaef46dcd1` |
| `certification_publication_sha` | **`PENDING_PUBLICATION_MERGE`** — the manifest holds 40 zeros plus `certification_publication_sha_status`; the orchestrator pins the real SHA in a later publication commit |

## Validation performed (logs in [`iiac_evidence/MCP-SERVER-X/p8_certification/`](../iiac_evidence/MCP-SERVER-X/p8_certification/EVIDENCE_MANIFEST.txt); each log has command, git head, UTC date)

| Check | Result | Log |
| --- | --- | --- |
| `origin/development` first-parent head is the #477 merge, which contains the auditor re-review `0c69c2065`; auditor evidence folder and dossier are byte-unchanged since `0c69c2065` | OK | `p8_cert_independence_digests_dispositions.log` |
| `product_paths_changed_between(6e53fb36f…, HEAD)` and `(eadb1c211…, HEAD)` | `[]` both; no file outside `docs/project/maintainers/qualification/` changed since `eadb1c211` | same |
| `verify_iiac_policy_activation --require-in-force` | exit 0, `IN_FORCE @ IIAC-2026-11-05.1` | `p8_cert_verify_activation_require_in_force.log` |
| `verify_iiac_audit_manifest.py <auditor manifest> --repo . --require-handoff-template` | PASS, exit 0 | `p8_cert_verify_auditor_manifest.log` |
| `verify_iiac_role_separation.py <auditor manifest> --strict-product-diff --repo .` | PASS, exit 0 | `p8_cert_verify_auditor_role_separation.log` |
| Same two verifiers on the certification manifest | PASS, exit 0 both | `p8_cert_verify_certified_manifest.log`, `p8_cert_verify_certified_role_separation.log` |
| SHA-256 recomputed for dossier, scope, matrix, findings | all four match the manifest (`8c1015d1…`, `e6b0733f…`, `79bf1fbf…`, `16dfa529…`) | `p8_cert_independence_digests_dispositions.log` |
| Findings: dispositions, blockers | 9 findings, each with a `reaudit_disposition`; `blocking_findings` empty; 0 `STILL_OPEN` | same |
| PR #477 required checks | `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate` all `success` on PR head `32a50991d`, completed 11:54Z–12:10Z, before the merge at 12:14:48Z | `p8_cert_pr477_required_checks_before_merge.log` |
| FRZ traceability validator | OK: `FRZ=225 families=26 PASS=108 OPEN=109`, dangling_open_frz=0; checklist and acceptance matrix byte-identical to `6e53fb36f` | `p8_cert_frz_traceability.log`, `p8_cert_independence_digests_dispositions.log` |

## Findings and dispositions (as recorded; none weakened)

| Finding | Severity | Disposition carried |
| --- | --- | --- |
| F-P8-01 | HIGH, governance-process | RECORDED, NOT WAIVED (P7-PROC-03). Operator chat disposition, option A "accept with the finding": no retroactive approval of the PR #474 merge; prospective rule: merge only after every required check finished green; OPEN-BP-01 is the structural cause. The auditor's register still reads `RECORDED_PENDING_OPERATOR_DISPOSITION` (auditor file, not edited); the disposition lives in the corrections record |
| F-P8-02 | MEDIUM | `ACCEPTED_SCOPED_LIMITATION` → R-P8-01 |
| F-P8-03 | MEDIUM | `ACCEPTED_SCOPED_LIMITATION` → R-P7-02 (open, not counted as passing; test debt untracked outside the corrections record) |
| F-P8-04, F-P8-05, F-P8-06, F-P8-09 | MEDIUM/MEDIUM/LOW/LOW | `RESOLVED` by documentation corrections |
| F-P8-07, F-P8-08 | LOW | `ACCEPTED_SCOPED_LIMITATION` |

## Limitations carried (stated here, in the text; the manifest repeats them)

1. **Scope of the audited claims:** tools-only streamable HTTP on loopback with the reference server; ACC-09 `run_agent` with a substituted delegate or mocked host; ACC-13/15 with a cooperative host double (drain bound patched to 0.25 s); ACC-14 client-side deadlines only; ACC-18 as an observed subset.
2. **R-P7-01** (`REQ-MCP-OBS-01`): no dedicated MCP trace id; ACC-17 scoped. **R-P7-03..06** open as the package states (result bound not applied to custom tools; no server-side deadline; three declared codes never raised; `harness_auth=None` passed by the engine).
3. **R-P7-02:** three MCP files fail on this machine (stable core of 11, observed 11–13 with load-dependent TL11 timeouts). They are excluded from the regression selection and **not counted as passing**; one of them guards the MCP opt-in default; PR/push CI is smoke-only, so green PR CI says nothing about them. The test debt is tracked only in the corrections record, not in a ticket.
4. **R-P8-01:** a host without harness auth trusts the client-asserted tenant/user; ACC-10/11/12 hold only for hosts with a configured credential. Fail-closed composition is a product change and an operator decision (ACT-N11).
5. **F-P8-01 (HIGH)** and **OPEN-BP-01** (no required status checks on the `development` ruleset) stay on the record; neither is waived.
6. **F-P8-07 / F-P8-08:** cooperative host double; no cancel-isolation, slot-release-after-handler-exception or non-cooperative-host tests; weak or self-referential assertions in ACC-14 and the consistency checks. Items F-P8-07(f) and F-P8-08(c)–(f) are recorded only in the audit register and dossier, not in the package cells.
7. **`LIM-RESEARCH-STRICT-FACTORY-MEMORY`** (PROD-Q, OPEN, non-blocking), **F-MCP-P5-GOV-001/002**, **F-MCP-P5-MEM-001**, **OD-MCP-P5-GOV-01**, **F-MCP-P6-GOV-001/002/003**, **P7-PROC-01/02/03**: carried unchanged, not waived.
8. The audit executed no mutation against product code; its mutation checks are reasoning (several mutants survive).

## What the certifier did not verify

- **No test replay.** The test results (164 new, 14 consistency, 141 regression passed at `eadb1c211`, plus 55 at the re-review) rest on the auditor's logs; the certifier ran no `pytest`. Running broad pytest is outside this role and the C10/C15 restrictions.
- **Auditor session identity.** `e96831ad-…` is not a host-issued id; the certifier could not map it mechanically to `local_b3377a14-…` (the host session was created 11:19Z and last active 11:49Z, which fits the manifest log time 11:39Z). The auditor's handoff record has two unchecked operator attestation boxes (new chat for the auditor; certifier id) and a blank operator signature line; the certifier does not attest on the operator's behalf.
- **The operator's F-P8-01 decision** is a chat transcription written by the orchestrator (the corrections record says so itself); there is no signature and the certifier cannot check the chat. The disposition is therefore `process_enforced`, not cryptographic.
- **The corrections-record addition `32a50991d`** came after the auditor's re-review and was not reviewed by the auditor.
- Items the auditor itself did not verify: dynamic FastMCP progress behaviour, the post-merge `development` push CI run, Windows-only behaviour on other machines, lane-report content beyond the cross-checked numbers.
- `certification_publication_sha` does not exist yet by design.

## Notes on the manifest

- `evidence_publication_sha` is set to the #477 merge (`a618a03a8…`), which published the auditor's evidence; the auditor left it as 40 zeros and asked the certifier to replace it. All other SHAs, the four digests, `verdict` (`PASS_SCOPED`) and `assurance_level` equal the auditor's manifest. Extra fields (`certification_verdict`, `certified_head_sha`, status and path fields) are allowed by the schema.
- The mission asked for the literal text `PENDING_PUBLICATION_MERGE` in `certification_publication_sha`. The schema and both verifiers require 40-hex, and the literal text fails both (`p8_cert_literal_pending_text_fails_verifiers.log`). The manifest therefore keeps the auditor's 40-zero placeholder convention and states `PENDING_PUBLICATION_MERGE` in `certification_publication_sha_status`. Whoever pins the real SHA must replace the zeros and re-run both verifiers.

## Non-claims

This record does not close MCP-SERVER-X, does not close ARCH-FREEZE, does not promote or change any FRZ status (counter 225), does not claim production readiness, does not execute FRZ Path B (`FRZ-MCP-*`), and does not touch PROD-Q Path B. It writes no `CLOSED` claim. Closure remains with `OD-MCP-P8-CLOSE-01` (conditions C1–C10) after publication and an independent terminal verification.
