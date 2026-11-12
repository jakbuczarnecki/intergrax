# MCP-SERVER-X — gap register (REQ-MCP-*)

Each row: requirement → owner → gap → risk → task → test level → FRZ/ECAP touch (incremental evidence only; **no new FRZ IDs** on default path).

**P1 links:** architecture [`MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md`](./MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md); contract/protocol [`MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`](./MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md); ADR-MCP-01..07 map gaps to decisions — closure remains P2–P7.

## Protocol / transport

| REQ-ID | Owner | Gap | Risk | TASK | TEST | FRZ/ECAP |
| --- | --- | --- | --- | --- | --- | --- |
| REQ-MCP-P-01 | Transport adapter | No platform contract pin for protocol revision FastMCP announces | False “full MCP” claim | MCP-P3-02 | L3 streamable HTTP only | FRZ-BND-06 (incremental) |
| REQ-MCP-P-02 | SDK | Negotiation deferred to FastMCP single transport | Low | — | DEFERRED | — |
| REQ-MCP-P-03 | Design | stdio/SSE not attached to host | By design v1 | — | OD-MCP-03 UNSUPPORTED | — |
| REQ-MCP-P-04 | Execution + cancel | Reference cancel/disconnect/unknown on P5 batch | Medium — **scoped qualified P5** | MCP-P7-03 | L2/L3 | EXE/REL incremental |
| REQ-MCP-P-05 | Definition contract | No shared error code catalog on definition contract | Medium | MCP-P2-05 | L1 | CMP incremental |

## Server definition & exposure

| REQ-ID | Owner | Gap | Risk | TASK | TEST | FRZ/ECAP |
| --- | --- | --- | --- | --- | --- | --- |
| REQ-MCP-S-01 | Platform contracts | No typed server identity | High | MCP-P2-01 | L1 | TYP/CMP incremental |
| REQ-MCP-S-02 | Composition | No enable/disable exposure policy | High | MCP-P2-03 | L1/L2 | CTR incremental |
| REQ-MCP-S-03 | Host mount | One mount per app; multi-definition ASGI compose missing | Medium | MCP-P2-04 | L1 | BND incremental |

## Capability classification (mandatory)

| Capability | Classification | Notes |
| --- | --- | --- |
| Tools | **MANDATORY** | Via `ToolContract` + host tools |
| Resources | **UNSUPPORTED_BY_DESIGN** v1 | OD-MCP-02 |
| Prompts | **UNSUPPORTED_BY_DESIGN** v1 | Skill-X owns behavior influence |
| streamable HTTP | **MANDATORY** | FastMCP 3.x |
| Other transports | **UNSUPPORTED_BY_DESIGN** v1 | OD-MCP-03 |
| sampling/roots extensions | **DEFERRED_WITH_JUSTIFICATION** | — |

## Contracts, security, execution

| REQ-ID | Owner | Gap | Risk | TASK | TEST | FRZ/ECAP |
| --- | --- | --- | --- | --- | --- | --- |
| REQ-MCP-C-01 | COMPAT exporter | Schema export exists; no surface version/fingerprint | High | MCP-P4-03 | L1 | FRZ-CMP-* incremental |
| REQ-MCP-SEC-01 | Harness auth | Dev bypass documented; prod fail-closed qualified P5 | High — **P5 qual @ c59892829** | maintain | L2/L3 neg | FRZ-SEC/TEN incremental |
| REQ-MCP-SEC-02 | Principal resolver | Client `tenant_id` is assertion only | — | maintain | L2 | TEN incremental |
| REQ-MCP-X-01 | Host execution | `run_agent` path exists | — | requal if signature changes | regresja | EXE |
| REQ-MCP-X-02 | Catalog bridge | Child execution per ADR TL11 exists | — | EVOL-gated requal | L2 | EXE |
| REQ-MCP-X-03 | TOOL-LIFE-X | Retry/deadline owner unchanged | — | no engine retry | — | — |

## Operations, observability, DX

| REQ-ID | Owner | Gap | Risk | TASK | TEST | FRZ/ECAP |
| --- | --- | --- | --- | --- | --- | --- |
| REQ-MCP-O-01 | Lifespan | No separate MCP readiness probe | Low | DEFERRED unless host pattern exists | — | — |
| REQ-MCP-O-02 | Engine bounds | Result bound partial; concurrent session cap missing | Medium | MCP-P3-06 | L2 | REL incremental |
| REQ-MCP-OBS-01 | Trace | No single MCP invocation id on execution event | Medium | MCP-P5-07 | L2 | TRC incremental |
| REQ-MCP-DX-01 | Public API | Consolidated `mcp_developer_api` + guide @ P6 | High — **closed in P6 (scoped)**: `T-IGX-035-G` closed; P6 batch @ `5ff8606b0`, G6 `ACCEPT_SCOPED` @ `9b8c19d75`, IIIC `PASS_SCOPED`; requalification at the P7 exact SHA required | MCP-P6-01 | L1 import | DOC incremental |

## P6 residuals (unchanged by P6 closure)

REQ-MCP-DX-01 is the only row changed by P6 closure (scoped). All other rows above keep their status and risk. Open cross-stage items: `LIM-RESEARCH-STRICT-FACTORY-MEMORY` (PROD-Q, OPEN, non-blocking, no fabricated vendor qualification); OPEN-BP-01 (no required status checks on the `development` ruleset); REQ-MCP-OBS-01 dedicated trace id gap (P7). Findings F-MCP-P6-GOV-001/002/003 are preserved in the [P6 operator closure record](./MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md).

## Qualification

Historical TOOL-LIFE-X / host MCP tests are **not** MCP-SERVER-X stage closure. Matrix **MCP-ACC-01..20** is mandatory (@ [`MCP_SERVER_X_QUALIFICATION_MATRIX.md`](./MCP_SERVER_X_QUALIFICATION_MATRIX.md)).

## P7 residuals and findings

Source: [P7 qualification package](./MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md) @ `abbe1525450da77da6801b080619a600c20cd497`. R-P7-01 = REQ-MCP-OBS-01 (dedicated trace id, operator decision DR-1, no product change in P7); R-P7-02 = 11 pre-existing failures in `test_mcp_surface_opt_in.py`, `test_mcp_surface_opt_in_runtime.py` and `test_mcp_catalog_e2e_hung_server_qualification.py` (core set of 11 at `c59892829`, `d5c1383fc` and the P7 head, observed 11-13 because TL11 subprocess startup timeouts vary; excluded, not counted as passing; marked `gate`/`no_ci`, so green PR CI says nothing about them); R-P7-03 = no result bound on custom tools; R-P7-04 = no server-side deadline / client deadline does not stop server work; R-P7-05 = three error-code enum members declared but not raised; R-P7-06 = `harness_auth=None` in composition; R-P8-01 = a host mounted without harness auth trusts the client-asserted tenant/user (fail-open identity, existing tested behaviour; product decision for the operator, outside the findings-only loop). Documentation findings D-P7-01 and D-P7-02 are fixed in the contract and protocol matrix.
