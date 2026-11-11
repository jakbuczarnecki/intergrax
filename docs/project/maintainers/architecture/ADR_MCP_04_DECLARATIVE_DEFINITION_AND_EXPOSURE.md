# ADR-MCP-04 — Declarative server definition and exposure policy

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-04 |
| **Status** | **Proposed** (pending OD-MCP-04) |
| **Related OD** | OD-MCP-04 |

## Context

Catalog mount today is all-or-nothing: enabled registry ⇒ all catalog tools on the server. There is no allowlist or disable-without-registry-mutation.

## Decision (proposed)

1. **Declarative typed definition** (contract v1) is the SSOT for what may appear on a server.
2. **Programmatic builder** is a thin overlay producing the same typed object — not a second semantic path.
3. **Exposure changes** at runtime require CONFIG-X-driven **recomposition** (restart / remount), not mutation of `ToolRegistry`.
4. Disabled or retired bindings are **absent from discovery**; invoke by retired id returns contract error (P4).

## Alternatives

| Option | Outcome |
| --- | --- |
| A. Shadow tool map inside MCP engine | **Rejected** — duplicate registry |
| B. Dynamic hot-reload without config | **Rejected** — second control plane |
| C. Definition + allowlist + config recompose | **Recommended** |

## Consequences

- MCP-P2-03 implements allowlist; MCP-P2-06 AST/import test guards against shadow registry.
- Multi-definition on one ASGI app is composition of multiple definitions, not multiple processes (MCP-P2-04).
