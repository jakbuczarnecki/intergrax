# T-IGX-035-I — MCP-SERVER-X P8: independent audit, IIAC certification, stage closure

## Scope

Batch **MCP-P8** only (MCP-P8-01..10): candidate `IMPLEMENTATION_SHA` freeze, independent technical audit with replay, findings-only corrections and requalification, IIAC certification (manifest `iiac-v1`, policy `IIAC-2026-11-05.1`), FRZ evidence notes at existing IDs only (counter 225), B6 evidence publication, roadmap synchronization, final closure report, ARCH-FREEZE handoff.

**Parent:** [`T-IGX-035-H.md`](T-IGX-035-H.md) — P7 exact-SHA qualification package (`TEST_EXECUTION_SHA`, MCP-ACC-01..20 matrix).

**Baseline SHA:** the P7 publication merge on `origin/development` (recorded in the SHA ladder when P7 closes).

**Tasks:** MCP-P8-01 .. MCP-P8-10 (canonical text in the plan below). Corrections `MCP-P8-Cxx` exist only for audit findings.

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P8-01..10.

## Approved

Approved: operator-delegate @ 2026-11-12T09:07:52+01:00
Delegation: docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_P7_P8_AUTH_2026-11-12.md@bddcbb293f53812bebb0134b2278d8f038a86825

## Test contract

Bounded, one `pytest` process at a time. Only the replay named in the audit manifest and the correction scope plus the regression record:

```text
.venv/Scripts/python.exe -m pytest <replay files named in the audit manifest> -q
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
.venv/Scripts/python.exe scripts/governance/verify_iiac_audit_manifest.py <manifest> --repo . --require-handoff-template
.venv/Scripts/python.exe scripts/governance/verify_iiac_role_separation.py <manifest> --strict-product-diff --repo .
.venv/Scripts/python.exe -m scripts.governance.igx_gov ticket
```

## P8 acceptance criteria (pre-implementation)

| # | Criterion | Verification |
| --- | --- | --- |
| AC-P8-01 | Audit by a session that is neither implementer (P2–P7), tester nor certifier; verdict PASS / PASS_SCOPED / FAIL on an IIAC manifest. | Manifest + role separation |
| AC-P8-02 | Every finding has a disposition; no silent closure; mandatory blocker = 0 at closure. | Findings register |
| AC-P8-03 | Correction loop is limited to findings; at most three rounds before escalation to the operator. | Round log |
| AC-P8-04 | IIAC certification by a third session; no self-certification; limitations stated in the text, not in a false global header. | `verify_iiac_*` |
| AC-P8-05 | FRZ: evidence notes at existing IDs only; counter 225 unchanged; no PASS promotion; Path B (FRZ-MCP-*) not executed. | FRZ validator |
| AC-P8-06 | `EVIDENCE_PUBLICATION_SHA` on `origin/development` with green required CI; roadmap status only after publication. | `gh` + git |
| AC-P8-07 | `MCP_SERVER_X_FINAL_CLOSURE_REPORT.md` cites SHAs (not chat summaries) and states scoped limits (REQ-MCP-OBS-01, LIM-RESEARCH-STRICT-FACTORY-MEMORY, OPEN-BP-01). | Report |
| AC-P8-08 | `MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md` published (what freeze may accept and what not); ARCH-FREEZE is NOT closed. | Document |
| AC-P8-09 | Independent final verification of the closed state by another session. | Verification report |

## Out of scope

ARCH-FREEZE closure, FRZ PASS promotion, a new FRZ family, PROD-Q Path B readiness, stdio/SSE, Resources/Prompts, Docker, new vendors, production configuration, broad diagnostic pytest.

## Lifecycle state

**closed** — scoped P8 acceptance; MCP-SERVER-X stage closed (scoped) per `OD-MCP-P8-CLOSE-01` and `P8_TERMINAL_VERIFIED`; no FRZ promotion; not ARCH-FREEZE; residuals R-P7-01..06, R-P8-01, OPEN-BP-01 and all recorded findings/deviations remain on record, not waived.

## Closure note

Stage-level closure of MCP-SERVER-X (scoped) is recorded by the roadmap status and the final closure report under the conditional decision `OD-MCP-P8-CLOSE-01` (terminal verification `P8_TERMINAL_VERIFIED`). The lifecycle state of this P8 ticket was first left unchanged because `OD-MCP-P8-CLOSE-01` contains no operator note for it; it is now `closed` on the following operator note. Operator note (chat, 2026-11-12, Polish original: "Wykonaj po kolei wszystkie pięć kroków aż do zamknięcia."; English rendering: "Carry out all five steps in order through to closure."), given in answer to the orchestrator's open-items list whose item 3 named the closure of tickets T-IGX-035-H and T-IGX-035-I. It is recorded here as the operator note required by the ticket README for `closed`; it is not a signed `Authorized:` decision line and carries no agent-composed timestamp.
