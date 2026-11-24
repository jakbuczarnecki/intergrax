# T-AF2-RW-TENANT — GAP-10 / GAP-11 disposition (Package C)

**Train compared:** `origin/arch-freeze/rw-train` @ `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`

## Ticket cited

From `governance/tickets/T-IGX-051-A.md` (Scope):

> Requests that A1 v18 does not satisfy and that are **not part of the write set until RW-PLAN is amended** and the digest recomputed: **GAP-10** (`tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py`, the carrier pin; proposed as a firm path of this ticket, Operator-visible), **GAP-11** (the application MCP facades that would pass `fails_closed`) …

From **Out of scope**:

> The requests **GAP-10 to GAP-12** of plan section 4 until an amendment adds them … the application MCP facades `applications/*/mcp/server.py`.

## Paths checked on train @ `e9bebe0ac`

| Request | Path | On train |
| --- | --- | --- |
| GAP-10 carrier pin | `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py` | present (pre–Parts G/H pin not re-run in this wave) |
| GAP-11 facades | `applications/*/mcp/server.py` | not required by ticket without amendment |

Firm Parts A–H slices from `T-IGX-051-A` landed via **PR #811** (`876af7e5d`…`e9bebe0ac`).

## Disposition

**GAP-10 and GAP-11 are outside the approved write set.** Implementing them here would widen `T-IGX-051-A` beyond the ticket text. Closure of GAP-10 requires RW-PLAN amendment, digest refresh, and Operator-visible Part I (Q-TEN-13), not a Package C train hotfix.