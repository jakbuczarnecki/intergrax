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

PENDING — the approval line is written only by the operator or by a session the operator commissioned (IIAC Amendment 01); an implementer never writes it.

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

**plan_pending**
