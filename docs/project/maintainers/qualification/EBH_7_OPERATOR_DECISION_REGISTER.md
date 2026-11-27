# EBH-7 — Operator decision register

> **Currency note (EQO-W4 reconciliation, as of 2026-11-26 @ `646e63953`, `EDER-F-0062`):** The rows below are preserved verbatim as historical decision records. EBH-7 parent closure was recorded via `OD-EBH7-08` (2026-11-01). `OD-EBH7-SCOPE-RESIDUAL` (`OPEN`) and `W1-PROD-DELTA` (`RETAIN — pending architectural reconciliation`) remain historical records and are not waived or closed by this note. Post-closure EBH-7 re-pin status is tracked in [`platform_domain_completeness/PDC_DOMAIN_CERTIFICATION_MATRIX.md`](platform_domain_completeness/PDC_DOMAIN_CERTIFICATION_MATRIX.md) and [`platform_domain_completeness/operator_decisions/OD-PDC-X-CLOSE.md`](platform_domain_completeness/operator_decisions/OD-PDC-X-CLOSE.md) (Decision 7). This note adds no new waiver, PASS, or closure claim.

| OD ID | Subject | Status | Notes |
| --- | --- | --- | --- |
| OD-EBH7-01 | Program entry @ `81679e672` | **APPROVED** | Canonical roadmap + qualification; stage gates + arch controls; see entry readiness record |
| W0-RC04-D | TEN-04 route — integrate-test-first vs ADR | **integrate-test-first** | ADR only if integrated test fails after minimal canonical remediation |
| W1-PROD-DELTA | Invoker tenant guard @ `5df074fcb` | **RETAIN — pending architectural reconciliation** | Interim fail-closed; historical auth deviation acknowledged — **not** retroactive prod approval; no revert mandate; no duplicate tenant/gov authority |
| W1-PROD-ADR | Targeted architecture review / minimal remediation | **AUTHORIZED** | Operator @ OD-EBH7-01; agent executes only canonical minimal path after review |
| RPL-REL-SQLITE-DUCKDB | Sanctioned registry sqlite + duckdb qualification | **AUTHORIZED** | Contract compatibility + behavioral evidence required; no fake dual-instance sqlite |
| OD-EBH7-08 | Program closure — B6 + evidence dossier | **APPROVED** | `OPERATOR APPROVED — OD-EBH7-08` (2026-11-01); [`EBH_7_OD_EBH7_08_CLOSURE_SIGNATURE_RECORD.md`](EBH_7_OD_EBH7_08_CLOSURE_SIGNATURE_RECORD.md) |
| B6 | Protocol 03 independent audit | **PASS (SCOPED)** | [`EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) @ `dc18e36d9…` |
| OD-EBH7-SCOPE-RESIDUAL | Scoped CLOSED with documented non-PROHIBITED residuals | **OPEN** | Required if normative RPL gaps remain at B6 |

Decisions are operator-owned. OD-EBH7-01 does **not** substitute B6 or OD-EBH7-08.
