# T-IGX-039-A — Handoff: strict-product scaffold MCP opt-in xfail (transferred from MCP-RC)

## Scope

Handoff record for the single strict `xfail` that the MCP-RC campaign transferred out of MCP ownership: `tests/unit/applications/test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]`. The product scaffold fails the STRICT agent-contract authority (a non-MCP concern), so the MCP opt-in default cannot be proven against the real product factory composition. Findings: AF-05 and AF-16 of the MCP-RC independent audit; `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md` ("Audit follow-up").

Owner: **strict-product scaffold owner** (role, no named person yet; the operator names the person or team). This ticket creates no product change and no MCP change: it records who restores real product-factory composition and removes the xfail.

## Plan digest

Plan digest: (pending - no plan until the owner is named)

## Approved

(pending - handoff ticket; not approved by the MCP-RC delegation, which covers T-IGX-038-A..F only)

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_surface_opt_in.py -q
```

Done when the `[product]` case passes without `xfail` against the real product factory composition, with the MCP default still opt-in.

## Out of scope

- MCP engine, identity or wire changes; ACT-N11 / IIAC logic; FRZ changes; weakening the STRICT agent-contract authority to make the case pass.

## Lifecycle state

draft
