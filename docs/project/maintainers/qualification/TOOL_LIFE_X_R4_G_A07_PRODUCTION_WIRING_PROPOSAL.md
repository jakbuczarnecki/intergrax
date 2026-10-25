# G-A07 — Production host integration proposal (no activation)

| Field | Value |
| --- | --- |
| **Status** | **PROPOSAL ONLY** — operator approved preparation, not bootstrap change |
| **Evidence gap** | R10-F04; no `wire_tool_life_*` in [`harness_host_runtime.py`](../../applications/_shared/harness_host_runtime.py) |

## Smallest safe wiring (when separately authorized)

| Layer | Proposal |
| --- | --- |
| **Wiring availability** | Call `wire_tool_life_durable_single_host_execution_recovery` from sanctioned composition factory used by harness host runtime, gated on `PersistenceTopology.DURABLE_SINGLE_HOST` env contract |
| **Configuration enabled** | Explicit env flag (e.g. `INTERGRAX_TOOL_LIFE_DURABLE_SINGLE_HOST=1`) — fail-closed if unset |
| **Effective runtime activation** | Recovery hooks active only when flag + topology + issuer binding configured; **certified recovery default-off** |
| **Operator authorization** | Separate **OD-R4-G-A07-PROD-ACTIVATE** before any production default-on |
| **External provider effects** | None from wiring alone; Gate A L3 still required per sink |

## Closure impact

G-A07 **BLOCKED** for live prod until wiring lands and is qualified. TOOL-LIFE-X parent may still reconcile qual-scoped Gate A per Path B **only if** independent audit accepts factory/qual topology — **not** via deferral without evidence.
