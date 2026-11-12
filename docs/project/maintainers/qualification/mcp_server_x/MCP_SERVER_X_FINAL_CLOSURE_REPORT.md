# MCP-SERVER-X — final closure report (MCP-P8-09)

**Ticket:** [`T-IGX-035-I`](../../../../../governance/tickets/T-IGX-035-I.md) · **Policy:** `IIAC-2026-11-05.1` (profile `iiac-v1`) · **Author role:** `certification-governance-agent` (session `local_f418d991-1c34-42a4-bc9d-d96734951f42`) · **Assurance level:** `process_enforced`

## Status of this report

This report states the scoped result of the MCP-SERVER-X P7/P8 campaign and cites SHAs and published records, not chat summaries. It changes no verdict: the audit (`PASS_SCOPED`), the certification (`CERTIFIED_PASS_SCOPED`) and the terminal verification (`P8_TERMINAL_VERIFIED`) stand exactly as their own records state them.

The report itself does not set any roadmap status. The roadmap text that the conditional decision `OD-MCP-P8-CLOSE-01` would authorize is, quoted from that decision only: "`CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`" with "the scoped limitations stated in the text". That decision is "void unless C1–C10 hold" and "takes effect only after an independent report with the verdict `P8_TERMINAL_VERIFIED`". Whether C5 and C8 hold, given the deviations listed below, is for the operator and the roadmap-sync step to decide; this report neither decides it nor waives anything.

Items that exist only after this report is merged are written as literal text: `FORMAL_CLOSURE_SHA` = `pinned after merge`; merge of the closure PR = `pinned after merge`; `FINAL_REMOTE_HEAD` = `pinned after merge`.

## SHA chain

| Role | SHA | Source |
| --- | --- | --- |
| Campaign `BASELINE_SHA` | `c228799785db8f35826f47ba0790d6a21d97fbe3` | [`MCP_SERVER_X_SHA_LADDER.md`](./MCP_SERVER_X_SHA_LADDER.md) |
| P6 terminal head (merge of PR #468; scope baseline of P7/P8) | `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` | SHA ladder `P6_TERMINAL_CLOSURE_SHA` |
| IIAC activation (`IN_FORCE` from, ACT-N11 P6 re-bind phase 1) | `92ad3555775649742299826e3680ec8ce703cc8a` | SHA ladder `P6_ACTIVATION_SHA`; activation gate exit 0 in the audit, certification and terminal logs |
| P7 publication = `IMPLEMENTATION_SHA` = `audited_implementation_sha` (merge of PR #474; PR head `94b6f6d6b8ae9882568c778fd7180b8b958fc175`) | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` | [`MCP_SERVER_X_P8_INDEPENDENT_AUDIT_DOSSIER.md`](./MCP_SERVER_X_P8_INDEPENDENT_AUDIT_DOSSIER.md) PIN block |
| `TEST_EXECUTION_SHA` (P7 exact-SHA qualification) | `abbe1525450da77da6801b080619a600c20cd497` | [`MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md`](./MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md) |
| Audit evidence commit / corrections (documentation only) / auditor re-review | `69a2333cc67071656f1e573a35339ade9a8b5b7b` / `c50547012c353463e86525411a5a065f43e5bab8` / `0c69c20652d5a78229a672ba0ae743aa8fa9a9c3` | git history; dossier "Re-review" section |
| Audit publication = `evidence_publication_sha` (merge of PR #477) | `a618a03a8b245ceefef6ab5a922213aaef46dcd1` | certified manifest; terminal report C5 |
| Certification publication (merge of PR #479) | `919e34b8f9fbccfc62a16205d8647d6e6070ce45` | terminal report "Pins re-derived". The field `certification_publication_sha` of the certified manifest is still the documented 40-zero placeholder with `certification_publication_sha_status` = `PENDING_PUBLICATION_MERGE`; no file pins `919e34b8f` as that field yet |
| Terminal verification report (commit on branch `mcp-p8-closure`) | `051178472067f06666a0a0d06bfe70a5aab2f524` | [`MCP_SERVER_X_P8_TERMINAL_VERIFICATION_REPORT.md`](./MCP_SERVER_X_P8_TERMINAL_VERIFICATION_REPORT.md) |
| P8-PROC-01 recorded (commit on branch `mcp-p8-closure`; tip of the branch when this report was written) | `4366bce314cb2e147559d1508aedabe9b81fc3a8` | [`MCP_SERVER_X_P8_CORRECTIONS_RECORD.md`](./MCP_SERVER_X_P8_CORRECTIONS_RECORD.md) |
| `FORMAL_CLOSURE_SHA` | `8864cee842a2bc745d26f2a3501dc500e5f3e7df` | pinned by the orchestrator after the closure merge |
| Merge of the closure PR (#480) | `c11184164af3072d0a48102a5e8d62adfec53695` | pinned by the orchestrator after the closure merge |
| `FINAL_REMOTE_HEAD` | `c11184164af3072d0a48102a5e8d62adfec53695` | head at the closure merge; later pin-only merges are listed in the SHA ladder |

## Verdicts

| Stage | Verdict | Record |
| --- | --- | --- |
| P7 qualification | MCP-ACC-01..20 evidenced on one `TEST_EXECUTION_SHA` (164 new, 14 consistency, 141 regression passed in the auditor's replay; rows 09, 14, 17, 18 scoped) | P7 package; `p8_audit_replay_*.log` |
| P8 independent audit | `PASS_SCOPED`; no mandatory blocker; findings F-P8-01..09 each with a disposition, 0 `STILL_OPEN` | Audit dossier; `iiac_audit_findings_p8.json` |
| IIAC certification | `CERTIFIED_PASS_SCOPED` (internal-agent certification, `process_enforced`; not stronger than the audit) | [`MCP_SERVER_X_P8_IIAC_CERTIFICATION_RECORD.md`](./MCP_SERVER_X_P8_IIAC_CERTIFICATION_RECORD.md); certified manifest under `iiac_evidence/MCP-SERVER-X/p8_certification/` |
| FRZ (MCP-P8-06) | Validator green (`FRZ=225 PASS=108 OPEN=109`); status diff empty; no row with sufficient audit evidence for a note | [`MCP_SERVER_X_P8_FRZ_EVIDENCE_NOTES.md`](./MCP_SERVER_X_P8_FRZ_EVIDENCE_NOTES.md) |
| Independent terminal verification | `P8_TERMINAL_VERIFIED` at `919e34b8f9fbccfc62a16205d8647d6e6070ce45`, with notes N-1..N-7 and deviations D-1..D-8 | Terminal verification report |
| ARCH-FREEZE handoff (MCP-P8-10) | Published on `origin/development` with PR #479; status "HANDOFF PREPARED — PENDING CLOSURE" | [`MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md`](./MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md) |

## The conditional decision `OD-MCP-P8-CLOSE-01` and its conditions

Decision file: [`operator_decisions/OD-MCP-P8-CLOSE-01.md`](./operator_decisions/OD-MCP-P8-CLOSE-01.md). Status per the decision: "APPROVED (conditional: void unless C1–C10 hold)". It was signed by an operator-designated signing session under the delegation record, not written by the operator (stated in the decision). Statuses below are the terminal verifier's, copied from its report; the certifier did not re-execute them at the closure-branch head and read only the records that changed after the verified head (the verification report and the corrections record).

| Cond. | Subject | Verifier status | Note carried |
| --- | --- | --- | --- |
| C1 | MCP-ACC-01..20 on one `TEST_EXECUTION_SHA` | PASS | Static counts equal the P7 table; replay at the verified head: 164 passed and 14 passed. The 141-test regression selection and the R-P7-02 files were not replayed by the verifier |
| C2 | Audit verdict, no blocker, every finding disposed | PASS (N-5) | The auditor's register still reads `RECORDED_PENDING_OPERATOR_DISPOSITION` for F-P8-01; the disposition (option A) is in the corrections record |
| C3 | Manifests and role separation | PASS (N-6) | All four verifier invocations exit 0. `implementation_session_id` is a group label, so the implementer side is not mechanically checkable |
| C4 | FRZ counter 225, no promotion | PASS | Checklist and acceptance matrix byte-identical to `6e53fb36f` |
| C5 | Evidence on `origin/development`; required CI of the publication merges finished green | PASS (finished green) **with deviations D-1, D-2** | All five required checks of PR #474, #477, #479 ended `success`, attempt 1, none still running at 12:38:52Z. #474 and #479 were merged before some of them finished (see governance record below) |
| C6 | Activation in force; no product path after the audit pin | PASS (N-3) | `IN_FORCE @ IIAC-2026-11-05.1`; `product_paths_changed_between` = `[]`; non-product PDC-X-lineage changes exist outside `intergrax/`, `agents/`, `applications/` |
| C7 | PROD-Q Path B untouched; limitations recorded | PASS | PROD-Q diff against `6e53fb36f` empty |
| C8 | Findings and deviations stay on record, not waived | PASS with N-1 | N-1 (PR #479 deviation not yet on a published record) is addressed by P8-PROC-01 in the corrections record, commit `4366bce31`, on the closure branch; the verifier left the strict reading of C8 to the operator |
| C9 | `MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md` published | PASS | On `origin/development` via PR #479 |
| C10 | Independent terminal verification report with `P8_TERMINAL_VERIFIED` | Verdict `P8_TERMINAL_VERIFIED`. The report is not yet on `origin/development`: it is published on branch `mcp-p8-closure` (commit `051178472`) and reaches `origin/development` with the closure PR (`pinned after merge`) | Verifier N-2 |

## Limitations of the scoped closure (in force; none waived)

1. **Audited scope.** Tools-only streamable HTTP on loopback with the reference server; ACC-09 `run_agent` with a substituted delegate or mocked host; ACC-13/15 with a cooperative host double (drain bound patched to 0.25 s, production 30 s); ACC-14 client-side deadlines only; ACC-18 as an observed subset.
2. **R-P7-01 / `REQ-MCP-OBS-01`.** No dedicated MCP invocation trace id; ACC-17 pins `task_id`, `run_id`, `execution_id` only (operator decision DR-1; no product change).
3. **R-P7-02.** Three MCP test files fail on the test machine (stable core of 11, observed 11–13 with load-dependent TL11 timeouts). Excluded from the regression selection, **not counted as passing**; one guards the MCP opt-in default; PR/push CI is smoke-only; the test debt is tracked only in the corrections record, not in a ticket. The verifier did not replay them (N-4).
4. **R-P7-03..06.** Result bound not applied to custom tools; no server-side deadline and `run_agent` has no timeout; three declared codes (`MCP_PRINCIPAL_REQUIRED`, `MCP_TENANT_ASSERTION_CONFLICT`, `MCP_RESULT_BOUND_EXCEEDED`) never raised; the engine passes `harness_auth=None`.
5. **R-P8-01.** A host mounted without harness auth trusts the client-asserted tenant/user (fail-open identity); ACC-10/11/12 hold only for hosts with a configured credential. Making composition fail closed is a product change and an operator decision (ACT-N11).
6. **Not covered by the engine.** Resources and Prompts (advertised by FastMCP, always empty, `UNSUPPORTED_BY_DESIGN`); stdio and SSE transports (not published; stdio absence is not enforced by the product); progress notifications (not claimed, not evidenced).
7. **OPEN-BP-01.** The `development` ruleset requires no status checks (confirmed by the verifier with `gh api`); structural cause of F-P8-01 and P8-PROC-01.
8. **`LIM-RESEARCH-STRICT-FACTORY-MEMORY`.** PROD-Q, OPEN, non-blocking, not re-examined.
9. **F-P8-07 / F-P8-08.** Cooperative host double; no cancel-isolation, slot-release-after-handler-exception or non-cooperative-host drain tests; weak or self-referential assertions in ACC-14 and the consistency checks. F-P8-07(f) and F-P8-08(c)–(f) are recorded only in the audit register and dossier.
10. **No mutation was executed** against product code in the audit; its mutation checks are reasoning, and several mutants survive.
11. **Pin validity.** The IIAC activation is bound to the P6 audit pin; any later change under `intergrax/`, `agents/` or `applications/` (including the held PDC-X pull requests #475 and #476) needs its own independent delta audit and a two-phase rebind before merge.
12. **Identity and signature limits.** The auditor id `e96831ad-…` is not host-issued and was not mechanically mapped to `local_b3377a14-…`; the auditor's handoff record has unchecked operator attestation boxes and a blank signature line; the F-P8-01 disposition is a chat transcription by the orchestrator, not a signature; `OD-MCP-P8-CLOSE-01` was signed by a designated signing session. All at `process_enforced`.

## Governance findings and deviations (recorded, not weakened, not waived)

| Id | Where | Statement | Disposition on record |
| --- | --- | --- | --- |
| F-MCP-P5-GOV-001, F-MCP-P5-GOV-002, F-MCP-P5-MEM-001; OD-MCP-P5-GOV-01 | P5 records; P7 package "carried unchanged" list (F-P8-09) | P5 governance and memory findings; the prospective merge gate | Carried unchanged |
| F-MCP-P6-GOV-001, -002, -003 | P6 operator closure record; P6 terminal dossier; SHA ladder | PR #459 and PR #460 merged red or before the evidence; stale ticket (F-003: #460 merged with failing `governance-fast`) | Carried unchanged |
| P7-PROC-01 | P7 package | One lane-L3 commit made with hooks disabled (`core.hooksPath`); checks run by hand and again through hooks and CI | Recorded |
| P7-PROC-02 | P7 package | PR #470 merged while `ent-audit-fast` was in progress | Recorded |
| F-P8-01 = P7-PROC-03 (HIGH) | P7 package; corrections record; certification record | PR #474 merged at 2026-11-12T10:42:10Z before `governance-fast` (10:48:30Z), `evol-x-b4-preflight` (10:44:50Z) and `ent-audit-fast` (10:58:08Z) finished; all later `success`, attempt 1 | **Recorded, not waived, no retroactive approval.** Operator chat disposition, option A, original label `A: przyjąć z findingiem (Recommended)`, English rendering by the orchestrator "A: accept with the finding (recommended)": the finding stays HIGH; prospective rule "merge only after every required check finished green"; OPEN-BP-01 remains the structural cause |
| P8-PROC-01 | Corrections record (section added at `4366bce31`) | PR #479 merged at 12:32:50Z before `governance-fast` (12:33:24Z) and `ent-audit-fast` (12:37:54Z) finished; both `success`, attempt 1; after the option A rule | **Recorded, not waived, not retroactively approved.** The #479 merge must not be described as respecting the prospective rule |
| OPEN-BP-01 | Gap register; P6 terminal dossier; certification record | No required status checks on the `development` ruleset | Carried; structural |
| Verifier deviations D-7, D-8 | Terminal verification report | Signing by a designated session and a chat transcription (D-7); zero placeholder with a status field outside the schema for `certification_publication_sha` (D-8) | Disclosed in the verification report; stated here as limitation 12 and in the certification record |

## Non-claims

- This report does not close ARCH-FREEZE and does not start it. The ARCH-FREEZE dependency on MCP-SERVER-X is satisfied only by the roadmap status published after the closure PR, never by this report.
- No FRZ status is promoted or changed; the counter stays 225; no new FRZ family; Path B (`FRZ-MCP-*`) is not executed.
- No production-readiness claim. PROD-Q Path B is untouched; `LIM-RESEARCH-STRICT-FACTORY-MEMORY` stays OPEN.
- No unscoped PASS: every verdict above is the scoped one, and the limitations above are part of the result, not a footnote.
- This report does not edit the roadmap, the SHA ladder, the freeze checklist, any ticket or any auditor, certifier or verifier artifact.

## Orchestrator addendum after the closure merge (not part of the certifier's report)

- **P8-PROC-02:** the closure PR #480 was merged at 2026-11-12T12:47:37Z, about 11 seconds after it was opened and before four of its five required checks had finished. They finished `success` afterwards (`governance-fast` 12:53:27Z, `evol-x-b4-preflight` 12:53:06Z, `ci-smoke` 12:48:33Z, `ent-audit-fast` 12:59:31Z; `ent-audit-pr-gate` 12:47:31Z before the merge). Recorded, not waived, no retroactive approval; same kind as F-P8-01 and P8-PROC-01; OPEN-BP-01 is the structural cause. Log: `iiac_evidence/MCP-SERVER-X/p8_closure/pr480_ci_after_merge.log`.
- The roadmap status of MCP-SERVER-X therefore became effective on `development` (`c11184164af3072d0a48102a5e8d62adfec53695`) before its CI evidence existed; the evidence confirms it afterwards. No push run exists for the docs-only merge commits (`paths-ignore`).
