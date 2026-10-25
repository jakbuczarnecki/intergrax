# TOOL-LIFE-X R3.1 — F4-01 closed-world Tool class / deadline audit

| Field | Value |
| --- | --- |
| **Task** | `F4-01` |
| **START_HEAD** | `afb10fcf4ff74767435e0a242cf33490467db352` |
| **EC7** | **not PASS** — operator decision only |
| **Recommendation** | **`EC7 DEADLINE_ONLY — ELIGIBLE FOR OPERATOR DECISION`** (scoped; see gaps) |

## Canonical enforcement owner

Per-invocation timeout, cooperative cancellation, retry/abandon, and queue-wait admission: **`RuntimeToolInvoker`** (`intergrax/runtime/nexus/tools/invoker.py`) composed via **`build_production_runtime_tool_invoker`**. Execution parent deadlines: **`ExecutionRuntime`** / `ExecutionDeadlineAuthority` (Batch A evidence).

## Closed-world matrix (production Tool surfaces)

| Tool class / path | Effective timeout | Parent deadline | Queue wait | Cancellation | STALLED / ABANDONED | Evidence | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Registered provider tools** (`ToolContract.timeout_ms` → `ToolExecutionProfile`) | Per-contract `timeout_ms` (default 30_000 ms) | Narrowed by active execution deadline projection when bound | `DependencyAttemptExecutionBoundary` permit / queue wait | `CancellationCoordinator` + invoker loop | `must_abandon_uncertain_outcome` / `must_abandon_after_retry_exhaustion` in invoker | P0 pre-audit §3; Batch A; `invoker.py` | Provider-specific hung-I/O without heartbeat remains **DEADLINE_ONLY** (no universal liveness proof) |
| **Catalog via `ExecutionBoundCatalogToolInvoker`** | Same as RuntimeToolInvoker path for underlying tool | Same | Same | Same | Same | `execution_bound_catalog_tool_composition.py` | No separate catalog timeout plane |
| **MCP catalog child (R3.1 F1)** | Inherited via `ChildExecutionRunner` `narrow_child_deadline_at_utc` | Parent projection required | N/A (in-process child only) | Pre/post invoke cancel checks in bridge | Child abandon follows parent/tool invoker if delegate runs | F1 implementation | HTTP ingress **not enabled** |
| **ServiceToolHandler providers** (`intergrax/tools/providers/**`) | Contract-level `timeout_ms` only | Via invoker when called through ToolRuntime | Boundary admission when wired | Via invoker | Uncertain → abandon policy in invoker | P0 inventory | Individual handlers may not bound external SDK calls beyond contract timeout |
| **HTTP / websearch / browser fetch** | Contract + param timeouts (e.g. `timeout_seconds`) | Invoker envelope | Transport boundary | Cooperative | Physical hang → uncertain/abandon | Batch D resource lifecycle qual | **Physical remote fence** not proven (Gate A OPEN) |
| **Relational / database tools** | Contract `timeout_ms` + driver/session limits | Invoker + execution scope | Pool/connection admission | Cooperative | Stalled RUNNING → certified owner-loss path (TL8 blocked for positive recovery) | Batch D; L3 PG qual | Multi-host durable stall **BLOCKED** |
| **MCP provider tools** | Contract `timeout_ms` | Invoker | Transport | Cooperative | Same abandon rules | TL11 MCP qual lineage | Catalog HTTP invoke disabled R3.1 |
| **Sandbox / eval / LTM / storage adapters** | Contract defaults | Invoker | Provider-specific | Cooperative | Invoker abandon | P0 closed-world list | No per-class L3 physical proof in this audit |

## Factual conclusion

- **DEADLINE_ONLY** semantics are **implemented at the canonical invoker + execution deadline layer** for tools admitted through `RuntimeToolInvoker`.
- **Universal certification** that every production `ToolContract` registration has been physically probed for stall recovery is **not claimed**.
- **EC7 PASS** is **not** recorded; operator may consider **OD-EC7** for DEADLINE_ONLY acceptance with documented gaps (remote physical fence, TL8 positive recovery, per-sink Gate A).
