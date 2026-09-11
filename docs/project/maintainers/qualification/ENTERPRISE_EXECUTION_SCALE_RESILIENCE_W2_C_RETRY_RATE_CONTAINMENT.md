# W2-C — Retry budget & provider rate limit (qualification)

**Status:** PASS (process-local)  
**Contracts:** `intergrax/contracts/retry_budget.py`, `intergrax/contracts/provider_rate_limit.py`  
**Implementations:** `LocalProviderRetryBudget`, `LocalProviderRateLimit` under `intergrax/runtime/resilience/`  
**LLM seam:** `execute_with_resilience` (retry loop wraps per-attempt budget + rate limit + CB); admission unchanged in `LLMAdapter._run_physical_provider_attempt`.

## Evidence

| Requirement | Mechanism |
|-------------|-----------|
| Retry budget | `RetryBudgetPort.open_logical_call` / `begin_physical_attempt` |
| Rate limit per attempt | `ProviderRateLimitPort.acquire_for_physical_attempt` inside retry loop |
| Retry storm | RPM window + optional aggregate concurrent cap on `LocalProviderRetryBudget` |
| Retry-After | `extract_retry_after_seconds` + `max_retry_after_sec` cap |
| Local rate ≠ provider failure | `LLMRateLimitError` / `ProviderRateLimitExceededError` excluded from CB `_record_failure` and from retriable classification |
| Provider 429 | Retriable per `retry_on_status`; CB failure recorded |
| Provider isolation | Separate `RetryBudgetIdentity` / rate-limit state per provider slug |
| No new managers | Injectable ports only (`set_llm_provider_retry_budget_port`, `set_llm_provider_rate_limit_port`) |

## Tests

`tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py` plus regression:

- `test_llm_provider_dependency_admission.py`
- `test_resilience.py`
- `test_call_config_retry.py`
- `test_distributed_rate_limit.py`
