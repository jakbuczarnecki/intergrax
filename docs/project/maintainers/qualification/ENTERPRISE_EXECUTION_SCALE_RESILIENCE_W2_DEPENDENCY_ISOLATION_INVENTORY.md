# Enterprise Execution Scale & Resilience — W2-A Dependency Isolation Inventory

**Task:** Enterprise Scale & Resilience/W2-A — Dependency Isolation Inventory & Canonical Ownership Qualification  
**Status:** INVENTORY COMPLETE (W2-A) · **ARCHITECTURAL DECISION RESOLVED BY** [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](../architecture/ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) (W2-ADR Accepted) · **W2-B OPEN** (implementation + wiring)
**Production runtime changed:** NO (W2-A docs only; W2-ADR adds contract module under `intergrax/contracts/` without invocation wiring)

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `7c8193c8aa67a0eeeb4557c6f5a354f43a630224` |
| START_ORIGIN | `7c8193c8aa67a0eeeb4557c6f5a354f43a630224` |
| Branch | `development` |

Companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) · P0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md).

## Required findings (summary)

| Concern | Verdict |
|---------|---------|
| TOOL BULKHEAD | **ABSENT** |
| PROVIDER BULKHEAD | **PARTIAL** (optional LLM circuit + rate only when `LLMCallConfig` enabled; no concurrency cap) |
| TENANT FAIRNESS | **ABSENT** (identity present; execution capacity not partitioned) |
| GENERIC RATE LIMITER | **ABSENT** (domain-specific: LLM `calls_per_minute`, Slack adapter, optional Redis distributed LLM) |
| INTEGRATION CIRCUIT BREAKER | **PARTIAL** (exists; production wiring mostly health/bootstrap + RAG retrieve) |
| RETRY-STORM PROTECTION | **PARTIAL** (R1 bounds + backoff/jitter; no per-dependency retry concurrency; layer multiplication risk) |

## Circuit breaker inventory

### `IntegrationCircuitBreaker` (`intergrax/integrations/_shared/circuit_breaker.py`)

| Attribute | Value |
|-----------|--------|
| **Owner** | Integrations (`_shared`) |
| **Scope** | One breaker instance per integration **catalog slug** (string name) |
| **Key** | `slug` via `get_breaker_for_slug(slug)` |
| **State lifetime** | Process lifetime; registry holds singleton per slug |
| **Process-local / durable** | **PROCESS-LOCAL** |
| **Failure threshold** | `IntegrationCircuitBreakerConfig.failure_threshold` (default 5 consecutive failures) |
| **Recovery** | OPEN → after `recovery_timeout_seconds` (default 30s) next `call` transitions to CLOSED (no half-open probe semantics beyond reset-on-timeout) |
| **Production wiring** | Tier-3: `wire_application_reliability` → `IntegrationCircuitBreakerConfig` → `probe_integration_profile_health` / `health_check_all` |
| **Call path** | `health.py`: `_resolve_with_breaker` wraps **factory resolve** before `health()` probe — not general integration invocation |

**Integration circuit breaker on real invocation path:** **NO** for catalog integration runtime calls. Grep shows `get_breaker_for_slug` / `breaker.call` only in `health.py`, `circuit_breaker_registry.py`, tests, and RAG vector wrapper.

### `RetrieverVectorCircuitBreaker`

| Attribute | Value |
|-----------|--------|
| **Owner** | RAG retrievers (`retriever_engine.py`) |
| **Scope** | Vector retrieve operations (default name `rag.vector_store`) |
| **Real invocation path** | **YES** — `retriever_engine` wraps `retriever.retrieve(query)` |
| **Reuse** | Delegates to `IntegrationCircuitBreaker` (shared semantics, separate instance from slug registry) |

### LLM provider circuit (`llm_adapters/_shared/resilience.py`)

| Attribute | Value |
|-----------|--------|
| **Owner** | LLM adapters |
| **Scope** | Per `provider` string (`_state_key`) |
| **Enabled when** | `LLMCallConfig.circuit_breaker_threshold > 0` (default **0 = disabled**) |
| **Real invocation path** | **YES** — `LLMAdapter._execute` → `execute_with_resilience` |
| **Not** | Same type/registry as `IntegrationCircuitBreaker` (parallel implementation) |

### `HarnessKernel` session reliability (`step_kernel.py`)

| Attribute | Value |
|-----------|--------|
| **Owner** | Agent harness kernel |
| **Behavior** | If `kernel_ctx.reliability.circuit_open`, step fails without external call |
| **Scope** | ACP harness session — **not** Nexus tool/integration plane |

### Circuit breaker registry

| Topic | Finding |
|-------|---------|
| Keying | `slug` string, lowercased at catalog probe boundary |
| Thread safety | `threading.Lock` on registry get/create |
| Distributed | **No** — each process has independent OPEN/CLOSED state |
| Duplication | Three families: integration slug, RAG vector wrapper, LLM in-process state |

**Circuit breaker ≠ bulkhead:** breakers fail fast when unhealthy; they do **not** cap concurrent in-flight calls.

## Tool execution path (canonical)

```text
Agent / graph node / ACP declarative action
  → DeclarativeToolExecutor (idempotency ledger, tenant_id on claims)
  → CatalogDeclarativeToolInvoker.invoke
  → invoke_catalog_tool_request / RuntimeToolInvoker.invoke
  → policy + idempotency pre-effect (optional)
  → _execute_external_effect
  → _execute_once: ThreadPoolExecutor.submit(ToolExecutor.execute)
  → future.result(timeout=contract.timeout_ms)
```

| Concern | Where | Owner |
|---------|-------|--------|
| External effect boundary | `ToolExecutor.execute` inside pool thread (`_ExternalEffectBoundary`) | Nexus `RuntimeToolInvoker` |
| Timeout | `contract.timeout_ms` on `future.result` | Tool contract + invoker |
| Retry | Loop in `_execute_external_effect`; `contract.retry_policy.max_attempts`, `backoff_ms` | Nexus invoker (not R1) |
| Idempotency | `IdempotencyPreEffectCoordinator` before boundary | Runtime tools |
| Circuit breaker | **None** on this path | — |
| Thread pool | **One** `ThreadPoolExecutor()` per `RuntimeToolInvoker` instance (default `max_workers` = CPython bounded default) | Nexus |

**Tool loop (read-only parallel):** separate short-lived `ThreadPoolExecutor(max_workers=workers)` per batch; `max_parallel_read_only` from runtime config — limits parallel read-only tools **within one loop**, not global tool isolation.

### Shared tool pool (P0 risk confirmation)

| Field | Value |
|-------|--------|
| Owner | `RuntimeToolInvoker` |
| max workers | Implicit stdlib default (bounded active threads) |
| Queue | Executor internal queue — **unbounded pending work** |
| Per-tool isolation | **No** |
| Per-provider isolation | **No** |
| Per-tenant isolation | **No** |

**Failure domain — tool:** If Tool A submits 1000 slow calls, Tool B on the **same** `RuntimeToolInvoker` queues behind the shared pool → **W2 TOOL BULKHEAD GAP**.

**Backpressure (tool):** `WAIT` on `future.result(timeout)` per call; pool queue growth = **UNBOUNDED QUEUE** under overload; no REJECT at admission.

## Provider (LLM) invocation path

| Attribute | Value |
|-----------|--------|
| **Canonical seam** | `LLMAdapter._execute` → `execute_with_resilience` → provider SDK callable |
| **Identity** | `(provider slug, model)` via `_adapter_identity`; rate/circuit keyed by `provider` string |
| **External boundary** | Inside `_execute` wrapper, before/around SDK HTTP |
| **Retry ownership** | `llm_adapters/_shared/retry.call_with_retry` when `max_retries > 0` (default `max_retries=0` → single attempt unless configured) |
| **Timeout ownership** | `LLMCallConfig.timeout_sec` per adapter |
| **Tenant** | `get_llm_tenant_id()` for distributed rate limit key; `check_llm_tenant_quota` (token budget env, default off) |
| **Concurrency cap** | **None** at adapter layer |
| **Rate limit** | Optional `calls_per_minute` (process-local deque window) + optional `use_distributed_rate_limit` + Redis limiter |
| **Token rate** | No requests/tokens-per-minute platform contract beyond optional tenant token quota env |

### Provider outage scenario (15 min, 1000 executions retrying)

| Question | Answer |
|----------|--------|
| Concurrent active provider calls | Bounded only by **host/async concurrency** (graph, fan-out, thread pools) — **no provider-specific in-flight cap** |
| Provider-specific cap | **No** (unless operator sets `calls_per_minute` / distributed limiter) |
| Retry jitter/backoff | R1: `compute_backoff_delay` + jitter kinds; Nexus presets may use `backoff_seconds=0.0` (`execution_mode_defaults.py`). LLM: exponential on `retry_backoff_sec` when `max_retries>0` |
| Circuit breaker | LLM CB **off by default**; integration slug CB **not on LLM path** |
| Retry generations | R1: `ResiliencePolicy.max_attempts` default **3** (mode presets: 1 / 5 / 3). Tool path separate. |
| One provider can consume host capacity | **Yes** — **W2 PROVIDER BULKHEAD GAP** for concurrency; partial rate limit if configured |

**Retry multiplication (documented values only):**

| Layer | Max attempts formula | Default when unset |
|-------|---------------------|-------------------|
| R1 execution retry | `ResiliencePolicy.max_attempts` | 3 (policy model); STRICT mode preset **1**, BALANCED **5** |
| Tool invoker | `contract.retry_policy.max_attempts` (side-effect NOT_RETRY_SAFE → **1**) | per tool contract |
| LLM SDK wrapper | `max(1, max_retries + 1)` | **1** call (`max_retries=0`) |
| Nexus `RetryEngine` | `RetryPolicy.max_retries` + alternate agent | graph/agent path, separate from R1 attempt CAS |

Independent layers can multiply (e.g. R1 `max_attempts=5` × tool `max_attempts` × LLM `max_retries+1` when all enabled).

**HTTP 429 / Retry-After:** LLM retry treats status in `retry_on_status` (includes **429**) but does **not** parse `Retry-After` header. R1 `ExecutionAttemptRetryService.compute_backoff` accepts `retry_after_seconds` when caller supplies it — **no automatic propagation from provider 429** on canonical LLM path. Slack knowledge reader extracts `Retry-After` into typed errors (integration-local).

## Tenant identity & fairness

| Location | `tenant_id` |
|----------|-------------|
| `RuntimeRequest` / `RuntimeConfig` | yes |
| `CatalogDeclarativeRunBinding` | yes |
| Idempotency store claims | yes |
| `ExecutionCapacityAdmissionRequest` | optional field — **not used** by `LocalExecutionCapacityAdmission` for partitioning |
| `step_kernel.StepKernelContext` | yes |

**Noisy neighbor:** Tenant A with 1000 root executions can consume all configured **root** permits; Tenant B starved → **W2 TENANT FAIRNESS GAP** (expected). `ExecutionCapacityPolicy` is **not** tenant fairness.

## Rate limit inventory

| Mechanism | Owner | Type | Scope |
|-----------|-------|------|-------|
| `LLMCallConfig.calls_per_minute` | llm_adapters | fixed window deque | per provider (process-local) |
| Distributed LLM limiter | optional Redis | token bucket-like | per tenant + `llm:{provider}` |
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | llm governance | cumulative token quota | per tenant |
| Slack backend | integrations | `asyncio.Semaphore(max_in_flight_handlers)` | inbound handlers |
| Websearch utils | domain-specific | unit tests reference rate limit | not generic platform |

**GENERIC RATE LIMITER: ABSENT** (no shared `RateLimitPort` for arbitrary dependencies).

## Retry ownership matrix

| Retry | Owner | Max attempts | Backoff | Jitter | Deadline | Scope |
|-------|-------|--------------|---------|--------|----------|-------|
| R1 execution attempt | `ExecutionAttemptRetryService` + policy | `ResiliencePolicy.max_attempts` | `compute_backoff_delay` | via `BackoffPolicyConfig` | `global_deadline_monotonic` on eligibility (W1-C) | run/attempt |
| Tool invocation | `RuntimeToolInvoker` | tool `retry_policy` | `backoff_ms` fixed sleep | no | tool timeout only | per tool call |
| LLM adapter | `call_with_retry` | `max_retries+1` | `retry_backoff_sec * 2**attempt` | no | `timeout_sec` per call | per adapter call |
| Nexus RetryEngine | Nexus retry | `RetryPolicy` | resilience policy resolver | partial | graph context | alternate agent |

**Breaker vs retry order (integration health):** `breaker.call` → resolve factory → health probe; on failure `_record_failure`. No retry inside breaker.

**Breaker vs retry (LLM):** circuit check → rate limit → call (with optional retry wrapper) → success/failure updates circuit.

Open integration slug breaker **does not** block runtime tool calls that use resolved clients obtained outside `health_check_all`.

## Failure-domain matrix

| RESOURCE | CANONICAL KEY | CURRENT LIMIT | CURRENT OWNER | ISOLATED? | BACKPRESSURE | CIRCUIT BREAKER | RATE LIMIT | RETRY | PROCESS/DISTRIBUTED | W2 GAP |
|----------|---------------|---------------|---------------|-----------|--------------|-----------------|------------|-------|---------------------|--------|
| root execution | run/attempt ids | `ExecutionCapacityPolicy.max_concurrent_root_executions` (optional port) | `ExecutionRuntime` + `LocalExecutionCapacityAdmission` | per-process global | REJECT / WAIT_WITH_TIMEOUT | no | no | R1 | PROCESS-LOCAL | tenant fairness |
| tenant | `tenant_id` | LLM token env quota only (opt-in) | llm governance | no execution share | REJECT quota | no | partial tokens | no | PROCESS-LOCAL | **TENANT FAIRNESS** |
| provider | provider slug | optional RPM / distributed | `llm_adapters._shared.resilience` | per-provider state only | REJECT rate | optional threshold | optional RPM | adapter + R1 | PROCESS-LOCAL (+ opt Redis) | **concurrency bulkhead** |
| integration | catalog `slug` | CB fail-fast when open | integrations `_shared` | slug per process | fail fast | health/resolve only | no | via tools/R1 | PROCESS-LOCAL | **real invocation CB** |
| tool | `tool_id` | pool workers + timeout | `RuntimeToolInvoker` | **shared pool** | UNBOUNDED QUEUE + WAIT timeout | no | no | invoker policy | PROCESS-LOCAL | **TOOL BULKHEAD** |
| retriever/backend | vector store name | CB on retrieve | RAG engine | per-engine instance | fail fast | yes (retrieve) | no | engine-level | PROCESS-LOCAL | backend concurrency |

## Ownership matrix

| CONCERN | CORRECT OWNER (target) | CURRENT OWNER | MATCH? |
|---------|------------------------|---------------|--------|
| root admission | `ExecutionCapacityAdmissionPort` | same (W1-A) | yes |
| tenant fairness | future tenant execution policy (not W1 port) | none | no |
| provider bulkhead | dependency concurrency at `LLMAdapter._execute` boundary | none (optional rate/CB config only) | no |
| tool bulkhead | dependency concurrency at `RuntimeToolInvoker` external boundary | shared `ThreadPoolExecutor` | no |
| integration breaker | integrations at external integration call seam | health bootstrap + RAG only | partial |
| provider rate limit | llm_adapters call config / distributed limiter | same | partial |
| retry policy | R1 for attempts; adapter for SDK; invoker for tools | split as documented | partial (no storm cap) |

## Configuration ownership

| Settings | Source |
|----------|--------|
| Circuit breaker threshold (integration) | `ApplicationEnvironmentProfile.reliability_profile` → `wire_application_reliability` → `ReliabilityWiringOptions.circuit_breaker_failure_threshold` |
| Resilience / retry (execution) | `reliability_profile.resilience_policy`, task metadata `resilience_policy.v1` |
| Root capacity | host wiring of `ExecutionCapacityPolicy` (W1) |
| LLM retries/rate/CB | adapter constructor defaults / `LLMCallConfig` |
| Tool timeout/retry | per `ToolContract` in registry |

**Config vs runtime:** breaker failure counts, rate-limit deques, circuit `open_until`, pool active threads = **runtime state** — not profile fields.

## Multi-process note (S/R4)

All inventory bulkheads/rate limits above are **PROCESS-LOCAL** unless Redis distributed LLM limiter is wired. Cluster capacity ≈ `N processes × local limit`.

## Observability gaps

No unified metrics/events for: dependency capacity reject, bulkhead wait timeout, rate-limit reject (except LLM exceptions), integration CB open on invocation (health only). Tool traces include retry steps; R1 has attempt transitions.

## P0 findings mapping (W2-A status)

| P0 finding | W2-A |
|------------|------|
| Shared tool thread pool | **Confirmed** — still one pool per invoker; inventory only |
| Recovery / retry storm | **Partial** — R1 bounded; no per-provider retry concurrency |
| Process-local isolation | **Documented** — CB/rate do not cross workers |
| Provider outage | **Gap** — no concurrency bulkhead; CB off by default on LLM |
| Tenant noisy neighbor | **Gap** — root capacity not tenant-scoped |

## Target wiring (reuse, no new framework)

Minimal future W2 implementation should **reuse typed seams**:

```text
RuntimeToolInvoker._execute_once (external boundary)
  → [future DependencyConcurrencyPort keyed by tool_id]
  → ToolExecutor.execute

LLMAdapter._execute
  → existing execute_with_resilience (extend with concurrency permit, not new manager)

Integration runtime calls (when identified per slug)
  → reuse IntegrationCircuitBreaker registry at resolve/invoke boundary (not only health)
```

Do **not** add `GlobalBulkheadManager` / `UniversalRateLimiter`.

## ARCHITECTURAL DECISION — RESOLVED BY ADR

**Decision record:** [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](../architecture/ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) (**Accepted**).

**Contract:** `intergrax/contracts/dependency_concurrency_admission.py` — `DependencyConcurrencyAdmissionPort.acquire` → `DependencyConcurrencyPermit.release` for exactly one in-flight external dependency attempt; overload `REJECT` / `WAIT_WITH_TIMEOUT`; typed `DependencyConcurrencyIdentity` (not arbitrary string keys); `tenant_id` metadata only.

**Distinct from:** `ExecutionCapacityAdmissionPort`, circuit breakers, rate limits, `ConcurrentExecutionWorkPolicy`, retry loops, tenant fairness.

**W2-B (OPEN):** process-local implementation (preferred `intergrax/runtime/resilience/`), tool boundary before shared executor enqueue, provider boundary before network/SDK, integration/retriever wiring when canonical seams are owned — no `GlobalBulkheadManager`.

## Static test

`tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_dependency_isolation_inventory.py`

## Regression commands (W2-A)

```text
uv run pytest tests/unit/integrations/test_integration_circuit_breaker.py -q
uv run pytest tests/unit/applications/test_reliability_wiring_provider_boundary.py -q
uv run pytest tests/unit/runtime/execution/test_enterprise_scale_resilience_w1_a_root_capacity_admission.py -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_dependency_isolation_inventory.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py -q
```
