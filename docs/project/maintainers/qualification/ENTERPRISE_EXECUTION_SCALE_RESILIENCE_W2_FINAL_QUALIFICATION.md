# W2 Final — Dependency resilience qualification

| Field | Value |
|-------|-------|
| **Status** | **PASS** (process-local; LLM provider path + tool/provider admission) |
| **START_HEAD** | `77358b80039f9301a15c127e4d1fecb4d29e4aba` |
| **Branch** | `development` |
| **Production code changed** | NO (qualification: tests + docs; regression fix in `test_distributed_rate_limit.py` only) |

Companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).

## Mechanism ownership matrix

| Mechanism | Owner | Contract | Implementation | Qualification tests |
|-----------|--------|----------|----------------|---------------------|
| Dependency admission | Runtime / resilience | `dependency_concurrency_admission.py` | `local_dependency_concurrency_admission.py`, `dependency_attempt_execution_boundary.py` | `test_local_dependency_concurrency_admission.py`, `test_dependency_attempt_execution_boundary.py` |
| Tool bulkhead | Nexus `RuntimeToolInvoker` | W2-ADR + boundary | Optional `DependencyAttemptExecutionBoundary` on invoker | `test_runtime_tool_invoker_dependency_admission.py` |
| Provider bulkhead | LLM adapters | W2-ADR + boundary | `provider_dependency_boundary.py` + adapter seams | `test_llm_provider_dependency_admission.py` |
| Rate limit | LLM orchestration | `provider_rate_limit.py` | `local_provider_rate_limit.py`, `execute_with_resilience` | `test_enterprise_scale_resilience_w2_c_retry_containment.py`, `test_resilience.py`, `test_distributed_rate_limit.py` |
| Retry budget | Runtime + LLM ports | `retry_budget.py` | `local_provider_retry_budget.py` | W2-C suite |
| Retry policy | LLM adapters | `LLMCallConfig` + `retry.py` | `is_retriable_provider_error`, `compute_provider_retry_delay` | `test_call_config_retry.py` |
| Circuit breaker (LLM) | LLM adapters | `LLMCallConfig` thresholds | `resilience.py` in-process state | W2-C + `test_resilience.py` |

**Out of W2 Final scope (deferred):** tenant fairness, distributed bulkhead / multi-worker admission, checkpoint scaling (W3), cancellation hardening (W4), observability backpressure (W5).

## Responsibility diagram (LLM physical attempt)

```text
check_llm_tenant_quota (governance)
  → execute_with_resilience retry loop
      → RetryBudgetPort.begin_physical_attempt
      → distributed rate limit (optional Redis)
      → ProviderRateLimitPort.acquire_for_physical_attempt
      → circuit breaker check (_check_circuit)
      → physical_attempt()
          → DependencyAttemptExecutionBoundary.acquire (W2-B3)
          → provider SDK / network
      → complete_physical_attempt (budget)
```

Tool path: **admission before pool submit** (`RuntimeToolInvoker`); tool retry is contract `retry_policy` only — **no** W2-C budget/rate/CB stack on tools.

## Confirmed invariants

1. Admission failures (`DependencyConcurrencyExceededError`, timeout, boundary closed) are **non-retriable** and **do not** increment LLM circuit breaker (`is_non_retriable_dependency_admission_failure`).
2. Local/distributed **rate limit reject** does not poison circuit breaker and is not classified as provider retriable failure.
3. **Retry budget** and **rate limit** apply **per physical attempt** inside the retry loop.
4. **Provider isolation:** separate `RetryBudgetIdentity` / rate-limit / circuit state per provider slug; tool policies keyed by `DependencyConcurrencyIdentity`.
5. **Permit lifecycle:** acquire → physical work → release on success, exception, timeout, cancellation, stream close (boundary + stream admission tests).

## Qualification scenario matrix (1–8)

| # | Scenario | Evidence |
|---|----------|----------|
| 1 | Provider outage + retry storm | `test_retry_storm_caps_provider_calls` (W2-C) |
| 2 | Provider isolation | `test_provider_isolation_openai_budget_exhausted_claude_works`, `test_failover_separate_provider_permits` |
| 3 | Bulkhead isolation | `test_tool_b_runs_when_tool_a_saturated`, `test_identity_isolation_across_tool_and_provider` |
| 4 | Retry budget exhaustion | `test_retry_budget_exhaustion_blocks_third_attempt` |
| 5 | Rate limit local reject | `test_local_rate_limit_does_not_poison_circuit` |
| 6 | Provider HTTP 429 | `test_provider_http_429_records_circuit_failure`, `test_retry_after_capped` |
| 7 | Admission failure | `test_admission_exceeded_does_not_call_sdk_or_poison_circuit` |
| 8 | Permit lifecycle | `test_dependency_attempt_execution_boundary.py` (+ stream tests in B3) |

Gate closure: `tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_final_qualification.py`.

## Orthogonality (ETAP 4)

| Module | Must not know |
|--------|----------------|
| `LocalProviderRetryBudget` | circuit breaker, rate limiter, provider SDK |
| `LocalProviderRateLimit` | retry budget port semantics, admission, circuit |
| `LocalDependencyConcurrencyAdmission` | retry, rate limit, circuit |
| `DependencyAttemptExecutionBoundary` | retry budget, rate limit, LLM resilience loop |

**Note:** `ProviderRateLimitIdentity` shares `RetryBudgetKind` with retry budget for key shape only (`provider_rate_limit.py`); implementations remain separate.

**Orchestration:** `execute_with_resilience` composes budget + rate + CB; admission stays in `LLMAdapter._run_physical_provider_attempt` — intentional composition layer, not a violation inside local port modules.

## Known deferred risks

| Risk | Disposition |
|------|-------------|
| Tenant fairness on execution / provider capacity | Later wave |
| Distributed dependency admission (S/R4) | Not W2 |
| Integration slug circuit breaker ≠ LLM CB (dual families) | Documented W2-A; unchanged |
| Tool invoker shared `ThreadPoolExecutor` without per-tool RPM | W2 admission caps in-flight only |
| `LocalDependencyConcurrencyAdmission` asyncio loop affinity | Documented W2-B1; boundary bridge for sync callers |

## Regression command

```powershell
uv run pytest `
  tests/unit/llm_adapters/test_llm_provider_dependency_admission.py `
  tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py `
  tests/unit/runtime/resilience/test_local_dependency_concurrency_admission.py `
  tests/unit/llm_adapters/test_resilience.py `
  tests/unit/llm_adapters/test_call_config_retry.py `
  tests/unit/llm_adapters/test_distributed_rate_limit.py `
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_dependency_isolation_inventory.py `
  tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py `
  tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py `
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w2_final_qualification.py
```
