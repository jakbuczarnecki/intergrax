# EBH-7 — Operator decision register

| OD ID | Subject | Status | Notes |
| --- | --- | --- | --- |
| OD-EBH7-01 | Program entry @ `81679e672` | **APPROVED** | Canonical roadmap + qualification; stage gates + arch controls; see entry readiness record |
| W0-RC04-D | TEN-04 route — integrate-test-first vs ADR | **integrate-test-first** | ADR only if integrated test fails after minimal canonical remediation |
| W1-PROD-DELTA | Invoker tenant guard @ `5df074fcb` | **RETAIN — pending architectural reconciliation** | Interim fail-closed; historical auth deviation acknowledged — **not** retroactive prod approval; no revert mandate; no duplicate tenant/gov authority |
| W1-PROD-ADR | Targeted architecture review / minimal remediation | **AUTHORIZED** | Operator @ OD-EBH7-01; agent executes only canonical minimal path after review |
| RPL-REL-SQLITE-DUCKDB | Sanctioned registry sqlite + duckdb qualification | **AUTHORIZED** | Contract compatibility + behavioral evidence required; no fake dual-instance sqlite |
| OD-EBH7-08 | Program closure — B6 + evidence dossier | **OPEN** | Independent B6 — not granted by OD-EBH7-01 |
| B6 | Protocol 03 independent audit | **OPEN** | Separate gate from OD-EBH7-01 |
| OD-EBH7-SCOPE-RESIDUAL | Scoped CLOSED with documented non-PROHIBITED residuals | **OPEN** | Required if normative RPL gaps remain at B6 |

Decisions are operator-owned. OD-EBH7-01 does **not** substitute B6 or OD-EBH7-08.
