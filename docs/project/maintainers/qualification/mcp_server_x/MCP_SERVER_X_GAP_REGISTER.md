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
| REQ-MCP-DX-01 | Public API | No “define server and publish” developer API | High | MCP-P6-01 | L1 import | DOC incremental |

## Qualification

Historical TOOL-LIFE-X / host MCP tests are **not** MCP-SERVER-X stage closure. Matrix **MCP-ACC-01..20** is mandatory (@ [`MCP_SERVER_X_QUALIFICATION_MATRIX.md`](./MCP_SERVER_X_QUALIFICATION_MATRIX.md)).
