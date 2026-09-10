# Enterprise Execution Scale & Resilience — Architecture (P0 baseline)

**Status:** P0 inventory baseline (evidence-backed; no production changes).  
**Baseline:** `origin/development` at audit start.  
**Scope:** Execution plane capacity, concurrency ownership, failure domains, process-local vs distributed semantics.

## Canonical execution topology

| Layer | Owner | Scale note |
|-------|--------|------------|
| Root lifecycle | `ExecutionRuntime` (`intergrax/runtime/execution/runtime.py`) | Per-request context binding; no global execution admission queue |
| Strategy / graph | `NexusLoop` → `GraphExecutor` | In-process asyncio parallelism; optional caps on executor |
| Child work | `ChildExecutionRunner` + `ExecutionBoundary` | One child per spawn; budget via shared ledger |
| Fan-out / fan-in | `bounded_multi_agent_fanout` + `CanonicalFanOutOrchestrationAdapter` | Hard platform bounds on item count and concurrency |
| Long-running / resume | `LongRunningCoordinator`, `SQLiteTaskCheckpointStore`, `LongRunningScheduler` | Durable store + lease claims; scheduler poll is process-local |
| Attempt retry (R1) | `ExecutionAttemptRetryService` + policy | Bounded attempts; backoff/jitter in `compute_backoff_delay` |
| Partial recovery (R3) | `FanOutPartialRecoveryService` | Slot-level; reuses fan-out bounds and Nexus orchestration path |

Nexus remains the canonical topology scheduler. P0 does not propose alternate orchestration layers.

## Concurrency ownership model

### Process-local (default)

Most execution-plane limits are **per host process**:

- `asyncio.Semaphore` for `max_parallel_nodes` / `max_inflight_nodes` on `GraphExecutor` (instance-scoped `_inflight_semaphore`).
- `ActiveTaskRegistry` (`_LOCK`, in-memory maps) — mid-run cancel lookup; not cross-worker.
- `IntegrationCircuitBreaker` / registry — in-process state per integration slug.
- `DeclarativeToolInvoker._execution_pool` — `ThreadPoolExecutor()` with no explicit `max_workers` (stdlib default **bounded** worker concurrency); Integrax does not define a platform-owned capacity/admission contract; overload can accumulate pending work in the executor's internal queue; one **shared** pool per invoker across tool calls (noisy-neighbor risk between tools).
- `ConcurrentExecutionWork` — `asyncio.gather` over caller-supplied tuple length (no platform cap in module).

### Cross-process / durable

- **Scheduler claims:** `SQLiteTaskCheckpointStore.claim_due` / `complete_claim` — lease + fence for scheduled resume (not a global execution semaphore).
- **Checkpoint revision CAS:** `TaskCheckpointPersistence` logical revision — stale writer protection (R2-H2).
- **Attempt lifecycle CAS:** `AttemptLifecycleService` + store `compare_and_swap`.
- **Idempotency ledger claims:** `sqlite_idempotency_store.claim` — per-key lease.

These mechanisms coordinate **durability and resume**, not fair sharing of CPU across tenants.

## Bounded vs unbounded parallelism

### Explicit platform bounds (fan-out)

`MAX_FAN_OUT_ITEMS = 256`, `MAX_FAN_OUT_CONCURRENCY = 64` in `bounded_multi_agent_fanout.py`; enforced by `validate_fan_out_request`.

Orchestration concurrency resolves as `min(platform_limit, submission_limit)` when either is set (`resolve_effective_orchestration_concurrency`). When **both** GraphExecutor caps and policy limits are `None`, parallel batches run **unbounded** within the batch (`asyncio.gather` on full batch).

Host profile contract allows `max_parallel_nodes` / `max_inflight_nodes` up to 256 (`host_profile_slices.py`). Application wiring often sets defaults (e.g. `resolve_max_inflight_nodes` → 8 in `scaling_wiring.py`) — **deployment configuration**, not a hard runtime invariant when unset.

### Child execution scale

Each child: new `ExecutionId`, ledger grant, boundary invoke. No global counter on child count; practical limits = parent budget, graph depth (`max_delegation_depth` on executor), and host resources. At 1000+ children, memory grows with lineage/checkpoint snapshots and in-flight asyncio tasks if orchestration runs wide batches without caps.

## Backpressure

| Boundary | Behavior | Evidence |
|----------|----------|----------|
| Graph parallel batch | Optional semaphore wait; `GRAPH_BACKPRESSURE` event when inflight semaphore locked | `GraphExecutor._execute_parallel_batch`, `_emit_backpressure` |
| Fan-out submission | Reject at validation (invalid request) | `validate_fan_out_request` |
| Root execution admission | No central bounded queue in ExecutionRuntime | `runtime.py` lifecycle only |
| Tool invoker | Bounded default workers; implicit pending-work queue (no admission shed); blocking wait on shared pool | `invoker.py` `_execution_pool` |
| Event bus | `create_task` on publish | `event_bus.py` |

Enterprise gap: overload without configured caps tends toward **unbounded task creation** and implicit OS/thread-pool queues rather than reject/shed at execution admission.

## Failure domains (architectural)

| Domain | Isolation | Notes |
|--------|-----------|-------|
| Integration provider | Per-slug in-process circuit breaker | Opens after threshold; not coordinated across workers |
| Tenant | Data partition keys in stores; lineage admission scope | No global per-tenant execution semaphore |
| Tool | Scope policy on invoker | Shared thread pool across tools |
| Nexus graph | Per-task graph state | Global inflight cap optional per executor instance |
| Checkpoint DB | SQLite file lock / connection per operation | Hot tenant/task streams contend on same DB file |
| Long-running scheduler | Single poll loop per scheduler instance | `claim_due` limit parameter bounds batch claim size |

## Retry vs circuit breaker

- **R1 retry:** exponential/fixed/jittered backoff; `max_attempts`; eligibility includes `global_deadline_monotonic` when populated on `ExecutionRetryEligibilityRequest`.
- **Circuit breaker:** integration/RAG vector paths; fails fast when open — orthogonal to attempt retry budget.

Retry storm risk: many failures with aligned backoff and **no per-provider retry concurrency cap** can amplify load on a shared provider. Jitter exists in backoff config but default resilience policies may use zero backoff in some Nexus presets (`execution_mode_defaults.py`).

## Cancellation

Cooperative: `CancellationCoordinator` metadata flag; graph marks pending nodes skipped. Propagation depends on call sites checking metadata between batches/nodes. Thread-pool tool work and in-flight provider HTTP may continue until completion unless the delegate respects cancellation (orphan-work risk under cancel).

## Partial recovery (R3)

Same-slot recovery is idempotent via checkpoint revision and slot disposition contracts. Different slots may recover in parallel subject to the same orchestration concurrency rules as initial fan-out. Recovery storm: many tenants resuming after outage can stress checkpoint store and Nexus concurrently — bounded by scheduler claim `limit`, not by global execution throttle.

## Process-local assumptions (explicit)

Do not treat `asyncio.Lock` / `Semaphore` on GraphExecutor as protecting resources across Celery workers or K8s pods. Scheduler lease claims are the cross-worker primitive for **resume scheduling**, not for limiting simultaneous graph execution.

## Target problems for follow-on waves (not P0)

1. **Execution admission** — bounded global/regional concurrency with explicit reject/shed policy.  
2. **Deadline propagation** — wire `global_deadline_monotonic` on all retry eligibility paths (e.g. Nexus `GraphRunner._transition_attempt_for_retry`).  
3. **Provider bulkhead** — per-dependency concurrency and retry budgets (without bypassing canonical ports).  
4. **Distributed rate limiting** — tenant/provider fairness across workers.  
5. **Checkpoint store scaling** — reduce SQLite hotspot or shard by tenant for write-heavy fleets.
