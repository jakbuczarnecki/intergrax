# MCP-SERVER-X P8 — corrections record (MCP-P8-03)

**Ticket:** `governance/tickets/T-IGX-035-I.md` · **Audit:** [`MCP_SERVER_X_P8_INDEPENDENT_AUDIT_DOSSIER.md`](./MCP_SERVER_X_P8_INDEPENDENT_AUDIT_DOSSIER.md) (verdict `PASS_SCOPED`) @ `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` · **Findings register:** `iiac_evidence/MCP-SERVER-X/p8_audit/iiac_audit_findings_p8.json`

Corrector: the orchestrator session acting as P8 implementer (documentation only; it is not the auditor and not the certifier). Product code, tests and the activation record are not changed, so the P7 `TEST_EXECUTION_SHA` `abbe1525450da77da6801b080619a600c20cd497` and its results stay valid. Dispositions follow the standing decision rules DR-2 and DR-6 (`OD-MCP-P7-P8-AUTH-01`): no waivers, no retroactive approvals.

| Finding | Severity | Class | Statement | Disposition | Corrected in |
| --- | --- | --- | --- | --- | --- |
| F-P8-01 | HIGH | governance-process | PR #474 merged before its `governance-fast`, `evol-x-b4-preflight` and `ent-audit-fast` runs finished (all later `success`, attempt 1) | RECORDED, NOT WAIVED. No retroactive approval. Recorded as P7-PROC-03 in the P7 package. Prospective discipline: merges only after every required check finished green (standing authorization SA-3); OPEN-BP-01 remains the structural cause | P7 package (P7-PROC-03) |
| F-P8-02 | MEDIUM | security residual unrecorded | Host without harness auth trusts the client-asserted tenant/user | ACCEPTED AS SCOPED LIMITATION R-P8-01. Fail-closed composition is a product change and an operator decision (ACT-N11), outside the findings-only loop | P7 package (R-P8-01, ACC-10/11/12 cells), gap register |
| F-P8-03 | MEDIUM | baseline-red tests | R-P7-02 wording `same set` not reproduced (13/12 vs 11/11) | CORRECTED WORDING; stays an open limitation, not a blocker; tracked as test debt owned by the orchestrator/operator (explicit LLM provider and revision-bound authority fixtures, or prove green in the full CI gate) | P7 package (R-P7-02), gap register |
| F-P8-04 | MEDIUM | documentation overclaim | Matrix claimed `Progress / cancel via transport: Supported`; progress is not evidenced | CORRECTED: row split into Cancel (supported) and Progress (not claimed, not evidenced) | Contract and protocol matrix |
| F-P8-05 | MEDIUM | documentation incomplete | Cross-reference to a wire-error list that did not exist; `When` columns of two codes inaccurate | CORRECTED: `Wire errors observed at P7` table added; `When` columns fixed | Contract and protocol matrix |
| F-P8-06 | LOW | documentation overclaim | Absence of stdio claimed from HTTP 404 probes | CORRECTED WORDING: SSE absence shown for the served ASGI app; stdio absence is not enforced by the product | P7 package (ACC-05) |
| F-P8-07 | LOW | evidence scope | Scope facts of the test docstrings missing from the limitation cells; several mutation-style checks would survive | ACCEPTED AS SCOPED LIMITATIONS; scope statements added to ACC-10, 11, 13, 15; optional hardening tests (non-cooperative-host drain, cancel isolation, slot release after handler exception, wrong_tenant at L3) are a separate hardening item, not part of P7/P8 closure | P7 package (ACC cells) |
| F-P8-08 | LOW | test quality | Weak or self-referential assertions in ACC-14 and the consistency checks | ACCEPTED AS KNOWN LIMITS; stated in the ACC-14 cell; tests not changed, so the `TEST_EXECUTION_SHA` results stay valid. Tightening (a), (d), (e) when the files are next touched | P7 package (ACC-14) |
| F-P8-09 | LOW | governance record completeness | P5 findings and OD-MCP-P5-GOV-01 missing from the carried-unchanged list | CORRECTED: added to the list | P7 package |

## Notes

- No finding is closed by weakening its statement. Findings that stay open are limitations of the closure (R-P7-01..06, R-P8-01, R-P7-02, OPEN-BP-01, `LIM-RESEARCH-STRICT-FACTORY-MEMORY`).
- Round 1 of at most 3 (DR-2). The independent auditor reviews these dispositions before certification; the certifier is a separate session.
- PDC-X pull requests #475 and #476 stay unmerged until P8 closes (their product paths would invalidate the audit pin).
