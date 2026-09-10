# Enterprise Execution Scale & Resilience — P0 Inventory & Qualification

**Task:** Enterprise Scale & Resilience/P0 — Execution Plane Capacity, Concurrency & Failure-Domain Inventory  
**Production code changed:** NO (docs + static arch test only)

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `b5cdef98200667b3b559673e91b7bbdf8ca013b9` |
| START_ORIGIN | `b5cdef98200667b3b559673e91b7bbdf8ca013b9` |
| Branch | `development` |

Architecture companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).

## Concurrency inventory (production-relevant)

| component | mechanism | owner | resource protected | scope | tenant | execution | local/distributed | bounded? | timeout? | cancel-safe? | backpressure | failure mode |
|-----------|-----------|-------|-------------------|-------|--------|-------------|-------------------|----------|----------|--------------|--------------|--------------|
| GraphExecutor | `asyncio.Semaphore` | Nexus | parallel nodes / inflight nodes | per executor instance | no | per graph batch | process-local | if caps configured | no | cooperative checks between batches | emit GRAPH_BACKPRESSURE when inflight locked | unbounded gather if caps None |
| GraphExecutor orchestration | `gather` + semaphore | Nexus | fan-out slots in batch | same | no | per topology submission | process-local | min(platform, policy) or unbounded | no | batch-level cancel skip | wait on semaphore | slot failures isolated in outcomes |
| ChildExecutionRunner | async boundary | Execution | child budget ledger | parent execution tree | via lineage scope | per child | process-local | budget policy | delegate-dependent | via boundary | none explicit | fail on authority/budget |
| ConcurrentExecutionWork | `asyncio.gather` | Execution | none (fan-in to port) | caller tuple size | no | per call | process-local | **no platform cap** | port-dependent | no | none | one failure fails gather (strict mode) |
| ActiveTaskRegistry | `asyncio.Lock` | Runtime task | registry maps | process | no | per active task | process-local | map size = active tasks | no | N/A | none | conflict exceptions |
| DeclarativeToolInvoker | `ThreadPoolExecutor()` | Nexus tools | tool invoke isolation | shared pool per invoker | no | per tool call | process-local | stdlib default max_workers (bounded); no platform capacity contract | invoke timeout paths | partial | implicit pending-work queue | worker starvation; shared-pool noisy neighbor |
| Tool loop (read-only) | `ThreadPoolExecutor` + `threading.Lock` | Nexus | parallel read-only tools | per loop | no | per planned batch | process-local | `max_parallel_read_only` | call timeout | partial | lock on mutating path | mutating serialized |
| LongRunningScheduler | poll loop + `claim_due` | Long-running | scheduled resume rows | scheduler owner | row tenant_id | per claimed resume | **cross-process** via DB lease | claim `limit` | lease expiry | resume admission checks | poll interval | uncertain / lease_busy |
| SQLiteTaskCheckpointStore | SQLite transactions | Long-running | checkpoint rows | DB file | tenant_id column | per task stream | cross-process | connection per op | lock wait | N/A | SQLITE_BUSY | stale write / CAS reject |
| AttemptLifecycleService | CAS | Execution R1 | attempt generation | store | tenant_id | per run | durable store dependent | max_attempts policy | no | cancel in eligibility | none | fail closed |
| IntegrationCircuitBreaker | in-memory state | Integrations | single slug backend | process | no | per call | process-local | failure_threshold | recovery_timeout | N/A | fail fast open | IntegrationDependencyError |
| Slack backend (integration) | Semaphore + Lock | Integrations | handler concurrency | process | n/a | n/a | process-local | `max_in_flight_handlers` | n/a | gather on shutdown | wait | handler backlog |
| CapacityScheduler | asyncio task loop | ECP | infra scaling ticks | deployment | optional tenant scope | N/A | process-local | interval | n/a | N/A | does not block Nexus | governance block |
| Event bus | `create_task` | Runtime events | subscriber delivery | process | no | per event | process-local | unbounded tasks | no | no | none | subscriber exceptions |

## Required inventory table

| Component | Parallelism | Bound | Isolation scope | Cross-process? | Backpressure | Risk | Evidence |
|-----------|-------------|-------|-----------------|----------------|--------------|------|----------|
| Bounded fan-out | per-request concurrency | 64 / 256 items | fan-out request | no | validation reject | low if validated | `bounded_multi_agent_fanout.py` |
| Fan-out adapter | topology → Nexus | inherits bounds | orchestration submission | no | via GraphExecutor | medium without host caps | `fan_out_orchestration_adapter.py` |
| GraphExecutor (legacy graph) | batch parallel | optional 256 cap | executor instance | no | GRAPH_BACKPRESSURE | high if caps unset | `graph_executor.py` |
| Orchestration topology | slot batch | min(host, policy) | submission | no | semaphore wait | unbounded if both None | `orchestration_topology.py` |
| Child execution | sequential per spawn | budget ledger | parent tree | no | none | depth/budget exhaustion | `child.py` |
| Concurrent council work | worker pool | **policy max_concurrency ≤64** | caller | no | bounded per call | `concurrent_execution_work.py` |
| R1 retry | attempt generations | max_attempts | run | durable CAS | none | retry storm | `retry/policy.py`, `retry/service.py` |
| R2 checkpoint | writes per task | revision CAS | tenant+task stream | yes (SQLite) | lock contention | hotspot | `long_running/store.py` |
| R3 partial recovery | per failed slot | fan-out bounds | topology | no | same as fan-out | recovery storm | `fan_out_partial_recovery.py` |
| Tool invoker pool | threads | CPython default | all tools on invoker | no | implicit | noisy neighbor tools | `nexus/tools/invoker.py` |
| Circuit breaker | serial when open | threshold | per slug / process | no | fail fast | shared slug across tenants | `integrations/_shared/circuit_breaker.py` |

## Fan-out qualification (NPSC-5B aligned)

| Dimension | Value |
|-----------|--------|
| Platform max items | 256 (`MAX_FAN_OUT_ITEMS`) |
| Platform max concurrency | 64 (`MAX_FAN_OUT_CONCURRENCY`) |
| Per-request max | `FanOutRequest.max_concurrency` ≤ platform |
| Worker capacity | Nexus child slots; limited by executor caps when configured |
| Queue behavior | No dedicated fan-out queue — asyncio tasks |
| Result accumulation | `FanOutResult.items` tuple — O(n) memory in request order |
| Uncontrolled create_task | Fan-out path uses gather on batch, not unbounded fan-out width beyond validation |

## Retry storm (R1)

| Control | Present? | Evidence |
|---------|----------|----------|
| Backoff | yes | `compute_backoff_delay` |
| Jitter | yes (JITTERED kind) | `backoff.py` |
| Global deadline on eligibility | contract yes; Nexus graph retry passes when budget bound | `active_execution_budget.py`, `graph_runner.py` |
| Retry budget | max_attempts | `evaluate_execution_retry_eligibility` |
| Per-provider isolation | no global retry semaphore | gap |
| Aligned retry risk | **yes** under outage + shared provider + zero backoff presets | `execution_mode_defaults.py` |

## Checkpoint pressure (R2)

| Topic | Architectural | Adapter-specific |
|-------|---------------|------------------|
| Stream CAS / revision | yes — `StaleCheckpointWriteError` | — |
| Same-stream contention | hot task_id + tenant_id | SQLite single-writer |
| Scheduler claim exclusivity | lease + fence | `store.claim_due` |
| Provider-neutral contract | `TaskCheckpointPersistence` | SQLite is one adapter |

## Partial recovery storm (R3)

| Control | Behavior |
|---------|----------|
| Same-slot exclusivity | revision + disposition contracts |
| Different-slot parallelism | orchestration concurrency rules |
| Bounded recovery width | fan-out validation on recovery request |
| Nexus admission | via orchestration submission port |
| Retry interaction | slot failure kinds mapped in `fan_out_partial_recovery.py` |

## Bulkhead / isolation gaps

| Domain | Exists? | Local/Global | Gap |
|--------|---------|--------------|-----|
| Integration provider | partial (CB) | per-process per slug | no cross-worker CB; no retry bulkhead |
| Tenant | data keys only | store partition | no execution fairness |
| Tool | scope policy | shared thread pool | one slow tool consumes pool |
| Agent | router/registry | per call | no agent-level concurrency cap |
| Execution class | host profile caps | deployment | optional unset → unbounded |
| External dependency | CB on some paths | in-process | LLM/provider pools not unified |

## Circuit breakers (inventory)

| Location | Scope | Threshold | Half-open | vs R1 |
|----------|-------|-----------|-----------|-------|
| `IntegrationCircuitBreaker` | integration slug | consecutive failures | after `recovery_timeout_seconds` | independent |
| `RetrieverVectorCircuitBreaker` | vector retrieve | config | same pattern | independent |
| `step_kernel` session reliability | step/session | config | open diagnostics | independent |

No execution-plane-wide circuit breaker.

## Timeouts / deadlines

| Kind | Mechanism | Propagation gap |
|------|-----------|-----------------|
| Tool | invoker timeout / sleep retry | partial cancel |
| Provider | integration timeouts | per adapter |
| Child | delegate / boundary | parent deadline not auto-reset in child mint |
| Topology | batch cooperative cancel | in-flight slot may complete |
| Global deadline | R1 eligibility field | **W1:** wired in GraphRunner when active execution budget carries deadline |
| HITL | `HumanTimeoutCoordinator` | long-running scheduler |

## Cancellation / orphan-work risks

| Scenario | Risk class | Evidence |
|----------|------------|----------|
| Parent cancelled, pending graph nodes | mitigated (skipped) | `CancellationCoordinator.mark_pending_graph_nodes_cancelled` |
| In-flight child / tool thread | **orphan-work risk** | cooperative cancel only |
| Fan-out batch mid-gather | slots may complete current await | orchestration parallel batch |
| Retry backoff sleep | Nexus `policy_enforcer` sleep | may delay cancel observation |

## Backpressure gaps

1. No bounded admission on `ExecutionRuntime` root execute (W1 ADR pending).  
2. ~~`execute_concurrent_execution_work` — no width cap.~~ **W1:** explicit `ConcurrentExecutionWorkPolicy`.  
3. GraphExecutor — unbounded when caps unset.  
4. Event bus — fire-and-forget tasks.  
5. Evidence/checkpoint writes — no explicit write shed.

## Memory pressure risks

- Full fan-out result tuples and topology recovery snapshots scale O(slots).  
- `prior_outputs` dict on graph execution grows with node outputs.  
- In-memory lineage / active registries per process.  
- Unbounded event subscriber tasks.

## Failure domain matrix

| Scenario | Trigger | Blast radius | Mitigation | Gap | Severity |
|----------|---------|--------------|------------|-----|----------|
| One tool/provider slow | blocking invoke | tasks sharing invoker pool | tool timeout | pool shared | P1 |
| One tenant overloads | many concurrent runs | same worker process | tenant store keys only | no fair scheduling | P1 |
| Evidence DB slow | write latency | runs using same store | retries/timeouts vary | no shed | P2 |
| Checkpoint DB locked | SQLITE_BUSY | resumes/checkpoints | CAS retry fail | hotspot file | P1 |
| Nexus worker crash | process death | in-flight runs on worker | durable checkpoint + lease uncertainty | running work lost until resume | P1 |
| Child execution crash | unhandled in child | parent slot outcome | partial recovery | orphan if outside checkpoint | P1 |
| Provider outage | errors + retries | all tenants on provider | CB + max_attempts | retry storm | P0 |

## Noisy neighbor

One tenant can exhaust: worker CPU/memory (no tenant cap), shared tool thread pool, fan-out width up to platform max (256), checkpoint write throughput on shared DB, provider rate limits shared across tenants, retry traffic to same integration slug (shared CB state per process).

## Fairness / admission

| Mechanism | Behavior |
|-----------|----------|
| Execution admission | hooks at boundary; no global queue |
| Fan-out admission | `validate_fan_out_request` reject |
| Child admission | authority + budget + lineage hooks |
| Provider/tool | scope policy; no global fair queue |

Overload: implicit queue (thread pool, asyncio wait on semaphore) or unbounded gather — **gap** where caps unset.

## Global contention

- `ActiveTaskRegistry._LOCK` — all register/unregister serialised.  
- SQLite checkpoint file — cross-tenant writer serialization.  
- Per-slug circuit breaker lock implicit in single-threaded call path (not async-safe across threads without review).  
- `runtime_context.llm_usage_lock` — per runtime context.

## Distributed semantics summary

| Mechanism | Semantics |
|-----------|-----------|
| GraphExecutor semaphore | process-local |
| Scheduler claim | cross-process (durable lease) |
| Checkpoint CAS | cross-process |
| Application mutex (idempotency claim) | cross-process when SQLite |
| In-memory CB | process-local |

## Idempotency / rate limits / budgets

| Budget | Exists |
|--------|--------|
| Fan-out width | 256 / concurrency 64 |
| max_attempts | yes (R1) |
| max_delegation_depth | optional on GraphExecutor |
| Run budget / child budget | ledger |
| Per-tenant execution count cap | **no** |
| Global rate limit | **no** (except integration-specific) |
| Provider retry concurrency | **no** |

## Observability under load risks

High event volume via `event_bus.create_task`; checkpoint write amplification on frequent graph checkpoints; evidence persistence coupled to execution frequency — no P0 change.

## Capacity model (qualitative)

Let:

- **E** = concurrent root executions (per worker, unbounded by platform)  
- **C** = average active child executions per root  
- **F** = fan-out width (≤ 256)  
- **R** = retry amplification (≤ max_attempts)  
- **P** = effective provider/tool parallelism (thread pools, HTTP client pools)

Effective concurrent work in one process scales roughly as:

`E × (C + F_slot_concurrency) × R_tool_retries`

Fan-out adds independent slot expansion bounded by `min(64, F)` concurrent slots per topology batch when caps configured; orchestration + graph paths can stack.

Do not infer numeric SLOs from this model without measurement.

## Scale qualification classes

| Class | Description |
|-------|-------------|
| **S1** | Normal enterprise load; host caps configured; single-region workers |
| **S2** | High parallel fan-out (256 slots, concurrency 64) |
| **S3** | Dependency outage with circuit breaker + retries |
| **S4** | Retry storm (aligned backoff, shared provider) |
| **S5** | Noisy tenant (dominates worker + checkpoint DB) |
| **S6** | Process crash under load (lease uncertainty + resume storm) |

## Risk table

| Risk | Severity | Trigger | Blast radius | Mitigation | Gap | Future wave |
|------|----------|---------|--------------|------------|-----|-------------|
| Unbounded graph/orchestration parallelism | P0 | caps unset in non-strict | whole worker | host profile wiring | strict mode fail-closed (W0) | W1 admission |
| Retry storm on provider outage | P0 | many runs fail together | provider + all tenants | backoff, CB, max_attempts | no jitter on all presets; no retry bulkhead | W2 provider retry budget |
| SQLite checkpoint hotspot | P1 | many concurrent writes | all tenants on DB file | CAS, indexes | single-file SQLite | W3 store sharding |
| Shared tool thread pool | P1 | slow tool | all agents on host | timeouts | no per-tool bulkhead | W2 tool pools |
| Cancel orphan work | P1 | cancel during tool/child | wasted spend | cooperative cancel | no hard preemption | W4 cancel propagation audit |
| Deadline not on Nexus retry request | P1 | long graph retry | parent SLA | R1 contract | graph_runner omits field | W1 deadline wiring |
| ConcurrentExecutionWork unbounded | P1 | large council tuple | memory/tasks | none | no cap | W1 width limit at port |
| Process-local semaphore illusion | P1 | multi-worker deploy | N× local slots vs one cap | ops scaling + W0 docs | semantic misunderstanding | W1 distributed admission |
| Recovery storm post outage | P2 | many resumes due | Nexus + DB | claim limit | no global throttle | W3 scheduler budgets |
| Event bus task spam | P2 | high emit rate | CPU | none | unbounded create_task | W5 observability shed |

## Proposed scale waves

| Wave | Focus |
|------|--------|
| **W0** | Mandatory host caps guardrails + process-local semantics (`host_execution_capacity_policy`; strict Nexus composition) |
| **W1** | Execution admission + deadline propagation on all retry paths |
| **W2** | Provider/tool bulkheads + distributed rate limits |
| **W3** | Checkpoint/evidence store scaling + recovery throttles |
| **W4** | Cancellation hardening (child/tool/provider) |
| **W5** | Load-aware observability sampling / backpressure on event pipeline |

## Static architectural test

`tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py` — guards documented bounds and surfaces documented gaps (no load benchmark).

## Regression qualification (commands)

```text
uv run pytest tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py -q
uv run pytest tests/unit/contracts/test_execution_lineage_contracts.py tests/unit/runtime/execution/lineage/ -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py -q
uv run ruff check docs/project/maintainers/architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py
uv run pyright tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py
```

## LEGACY REMOVAL CANDIDATE

None identified with zero production callers in P0 scope (legacy supervisor fan-out paths already gated out of NPSC-5B production surface).
