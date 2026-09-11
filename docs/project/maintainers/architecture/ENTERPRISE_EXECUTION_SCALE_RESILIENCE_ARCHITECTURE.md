# Enterprise Execution Scale & Resilience — Architecture (P0 baseline)

**Status:** P0 inventory baseline; **W0** strict host capacity guardrails; **W1 FINAL (qualified)** — process-local root admission (W1-A), explicit concurrent work policy (W1-B), absolute global deadline into R1 retry (W1-C). Qualification: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W1_ADMISSION_DEADLINE.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W1_ADMISSION_DEADLINE.md). **W2 FINAL (qualified, process-local)** — dependency admission bulkheads (B1–B3) + retry budget / provider rate limit / LLM circuit composition (C); final matrix: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_FINAL_QUALIFICATION.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_FINAL_QUALIFICATION.md). **W2-A** inventory: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_DEPENDENCY_ISOLATION_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_DEPENDENCY_ISOLATION_INVENTORY.md); **W2-ADR (Accepted)** — [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md); **W2-B1** `LocalDependencyConcurrencyAdmission`; **W2-B2** `DependencyAttemptExecutionBoundary` on `RuntimeToolInvoker`; **W2-B3** provider boundary on `LLMAdapter` seams; **W2-C** `execute_with_resilience` order: tenant quota (adapter) → retry budget → rate limit → circuit breaker → physical attempt (admission inside `_run_physical_provider_attempt`).
**Baseline:** `origin/development` at audit start.  
**Scope:** Execution plane capacity, concurrency ownership, failure domains, process-local vs distributed semantics.

## Canonical execution topology

| Layer | Owner | Scale note |
|-------|--------|------------|
| Root lifecycle | `ExecutionRuntime` (`intergrax/runtime/execution/runtime.py`) | Per-request context binding; optional `ExecutionCapacityAdmissionPort` (W1-A) for bounded process-local root slots |
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
- `ConcurrentExecutionWork` — execution-owned bounded worker pool via required `ConcurrentExecutionWorkPolicy` (no platform default; independent from fan-out 64; see W1-B qualification).

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

Host profile contract allows `max_parallel_nodes` / `max_inflight_nodes` up to 256 (`host_profile_slices.py` / environment `OrchestrationProfile`). Application wiring often sets deployment defaults (e.g. product template 8/8, scaling ceiling patcher fallback 8) — **W0:** `ExecutionMode.STRICT` requires both caps explicitly set before Nexus composition (`host_execution_capacity_policy.py`); `None` is not a safe production default.

### Child execution scale

Each child: new `ExecutionId`, ledger grant, boundary invoke. No global counter on child count; practical limits = parent budget, graph depth (`max_delegation_depth` on executor), and host resources. At 1000+ children, memory grows with lineage/checkpoint snapshots and in-flight asyncio tasks if orchestration runs wide batches without caps.

## Backpressure

| Boundary | Behavior | Evidence |
|----------|----------|----------|
| Graph parallel batch | Optional semaphore wait; `GRAPH_BACKPRESSURE` event when inflight semaphore locked | `GraphExecutor._execute_parallel_batch`, `_emit_backpressure` |
| Fan-out submission | Reject at validation (invalid request) | `validate_fan_out_request` |
| Root execution admission | Optional typed port on `ExecutionRuntime` (`ExecutionCapacityAdmissionPort`); default `None` preserves legacy callers | `execution_capacity_admission.py` + `local_execution_capacity_admission.py` |
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

Do not treat `asyncio.Lock` / `Semaphore` on GraphExecutor as protecting resources across Celery workers or K8s pods. Each worker process holds its **own** local caps; multiplying worker count multiplies local capacity unless a future distributed admission wave says otherwise (W0 qualification: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md)). Scheduler lease claims are the cross-worker primitive for **resume scheduling**, not for limiting simultaneous graph execution.

## W2-A ownership evidence (inventory)

- **Integration circuit breaker:** `IntegrationCircuitBreaker` + slug registry — **owner:** integrations `_shared`; **production wiring:** Tier-3 health/bootstrap (`health_check_all`) and config from `wire_application_reliability`; **not** on Nexus `RuntimeToolInvoker` path. RAG retrieve uses wrapper on real calls.
- **Tool execution:** `RuntimeToolInvoker` — **owner:** Nexus tools; single shared `ThreadPoolExecutor()` per invoker; timeout + contract-level retry; **no** per-`tool_id` concurrency port.
- **Provider calls:** `LLMAdapter._execute` → `execute_with_resilience` — **owner:** llm_adapters; per-physical-attempt retry budget + `ProviderRateLimitPort` (process-local default) + optional RPM/CB/retry via `LLMCallConfig`; in-flight concurrency via W2-B3 admission boundary (orthogonal to W2-C throughput/retry caps).
- **Tenant:** `tenant_id` on runtime request, idempotency, capacity request metadata — **no** per-tenant root slot partitioning (`LocalExecutionCapacityAdmission` ignores tenant).
- **W2-ADR (Accepted):** `DependencyConcurrencyAdmissionPort` at external boundaries — acquire → one external attempt → release; typed `DependencyConcurrencyIdentity` (`TOOL`, `LLM_PROVIDER`, `INTEGRATION`, `RETRIEVER`); orthogonal to W1 root admission, W1-B concurrent work, circuit breakers, rate limits, retry, tenant fairness. Tool seam: before shared executor enqueue; provider seam: inside physical attempt after W2-C gates. **W2 Final qualified** — see final qualification doc; no new managers/schedulers.

## W2 Final — composition (qualified)

| Layer | LLM provider call | Tool external effect |
|-------|-------------------|----------------------|
| Tenant quota | `check_llm_tenant_quota` | N/A (W2 scope) |
| Retry budget / rate / CB | `execute_with_resilience` | N/A |
| Concurrency admission | `DependencyAttemptExecutionBoundary` | Same boundary pattern on invoker |
| Retry policy | `LLMCallConfig` + `retry.py` | `ToolContract.retry_policy` |

Deferred: tenant fairness partitioning, distributed admission, integration slug breakers remain separate from LLM in-process CB.

## Target problems for follow-on waves (not P0)

1. **Execution admission** — W1-A: bounded process-local root slots via injectable port; distributed/global cap deferred.
2. ~~**Deadline propagation**~~ — **W1:** `global_deadline_monotonic` wired from active execution budget into `GraphRunner` retry eligibility when root wall-time budget is set.  
3. ~~**Provider/tool bulkheads (process-local)**~~ — **W2 Final:** `DependencyConcurrencyAdmissionPort` wired for tools and LLM providers (qualified).
4. **Distributed rate limiting** — tenant/provider fairness across workers (optional Redis LLM limiter exists; generic platform limiter absent).  
5. **Checkpoint store scaling** — reduce SQLite hotspot or shard by tenant for write-heavy fleets.
