# © Artur Czarnecki. All rights reserved.

"""W2-C — retry budget, per-attempt rate limit, retry storm containment."""

from __future__ import annotations

import threading
import time
import pytest

from intergrax.contracts.provider_rate_limit import (
    ProviderRateLimitIdentity,
    ProviderRateLimitPolicy,
    ProviderRateLimitPort,
)
from intergrax.contracts.retry_budget import (
    RetryBudgetExhaustedError,
    RetryBudgetIdentity,
    RetryBudgetKind,
)
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.provider_rate_limit_port import (
    get_default_local_provider_rate_limit,
    set_llm_provider_rate_limit_port,
)
from intergrax.llm_adapters._shared.provider_retry_budget import (
    set_llm_provider_retry_budget_port,
)
from intergrax.llm_adapters._shared.resilience import (
    LLMCircuitOpenError,
    LLMRateLimitError,
    execute_with_resilience,
    reset_provider_resilience,
)
from intergrax.llm_adapters._shared.retry import (
    compute_provider_retry_delay,
    extract_retry_after_seconds,
)
from intergrax.runtime.resilience.local_provider_retry_budget import LocalProviderRetryBudget

pytestmark = pytest.mark.unit


def _identity(slug: str) -> RetryBudgetIdentity:
    return RetryBudgetIdentity(kind=RetryBudgetKind.LLM_PROVIDER, value=slug)


def test_retry_budget_exhaustion_blocks_third_attempt() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(max_retries=1, retry_backoff_sec=0.0)
    calls = {"n": 0}

    class Transient(RuntimeError):
        status_code = 503

    def _fn() -> str:
        calls["n"] += 1
        raise Transient("down")

    with pytest.raises(Transient):
        execute_with_resilience(_fn, provider="openai", config=cfg, retry_fn=lambda f: f())
    assert calls["n"] == 2
    reset_provider_resilience("openai")


def test_provider_isolation_openai_budget_exhausted_claude_works() -> None:
    reset_provider_resilience()
    port = LocalProviderRetryBudget({_identity("openai"): 1})
    set_llm_provider_retry_budget_port(port)
    cfg = LLMCallConfig(max_retries=0, retry_backoff_sec=0.0)
    gate = threading.Event()

    def _openai() -> str:
        gate.wait(timeout=5)
        return "openai"

    def _claude() -> str:
        return "claude-ok"

    worker = threading.Thread(
        target=lambda: execute_with_resilience(
            _openai,
            provider="openai",
            config=cfg,
            retry_fn=lambda f: f(),
        )
    )
    worker.start()
    time.sleep(0.05)
    with pytest.raises(RetryBudgetExhaustedError):
        execute_with_resilience(
            lambda: "blocked",
            provider="openai",
            config=cfg,
            retry_fn=lambda f: f(),
        )
    assert (
        execute_with_resilience(_claude, provider="claude", config=cfg, retry_fn=lambda f: f())
        == "claude-ok"
    )
    gate.set()
    worker.join(timeout=5)
    set_llm_provider_retry_budget_port(None)
    reset_provider_resilience()


class _CountingRateLimitPort(ProviderRateLimitPort):
    def __init__(self, inner: ProviderRateLimitPort) -> None:
        self.inner = inner
        self.acquire_calls = 0

    def acquire_for_physical_attempt(
        self,
        identity: ProviderRateLimitIdentity,
        policy: ProviderRateLimitPolicy,
    ) -> None:
        self.acquire_calls += 1
        self.inner.acquire_for_physical_attempt(identity, policy)


def test_each_retry_acquires_rate_limit() -> None:
    reset_provider_resilience("openai")
    inner = get_default_local_provider_rate_limit()
    counter = _CountingRateLimitPort(inner)
    set_llm_provider_rate_limit_port(counter)
    cfg = LLMCallConfig(
        calls_per_minute=100,
        max_retries=2,
        retry_backoff_sec=0.0,
    )
    attempts = {"n": 0}

    class Transient(RuntimeError):
        status_code = 503

    def _fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Transient("x")
        return "ok"

    assert execute_with_resilience(_fn, provider="openai", config=cfg, retry_fn=lambda f: f()) == "ok"
    assert attempts["n"] == 3
    assert counter.acquire_calls == 3
    set_llm_provider_rate_limit_port(None)
    reset_provider_resilience("openai")


def test_retry_storm_caps_provider_calls() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(calls_per_minute=12, max_retries=20, retry_backoff_sec=0.0)
    provider_calls = {"n": 0}
    barrier = threading.Barrier(101)
    lock = threading.Lock()

    class Transient(RuntimeError):
        status_code = 503

    def _fn() -> str:
        with lock:
            provider_calls["n"] += 1
        raise Transient("outage")

    def _worker() -> None:
        barrier.wait()
        try:
            execute_with_resilience(_fn, provider="openai", config=cfg, retry_fn=lambda f: f())
        except Transient:
            pass

    threads = [threading.Thread(target=_worker) for _ in range(100)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30)
    assert provider_calls["n"] <= 12
    reset_provider_resilience("openai")


def test_retry_after_capped() -> None:
    cfg = LLMCallConfig(max_retry_after_sec=5.0)

    class _Response:
        headers = {"Retry-After": "600"}

    class _Exc(RuntimeError):
        response = _Response()

    assert extract_retry_after_seconds(_Exc()) == 600.0
    delay = compute_provider_retry_delay(attempt_index=0, config=cfg, exc=_Exc())
    assert delay == 5.0


def test_local_rate_limit_does_not_poison_circuit() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(calls_per_minute=2, circuit_breaker_threshold=2)
    execute_with_resilience(lambda: "ok", provider="openai", config=cfg, retry_fn=lambda f: f())
    execute_with_resilience(lambda: "ok2", provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(LLMRateLimitError):
        execute_with_resilience(lambda: "x", provider="openai", config=cfg, retry_fn=lambda f: f())
    get_default_local_provider_rate_limit().reset(
        ProviderRateLimitIdentity(kind=RetryBudgetKind.LLM_PROVIDER, value="openai")
    )
    cfg_failures = LLMCallConfig(circuit_breaker_threshold=2, max_retries=0)

    def _boom() -> str:
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        execute_with_resilience(
            _boom,
            provider="openai",
            config=cfg_failures,
            retry_fn=lambda f: f(),
        )
    with pytest.raises(RuntimeError):
        execute_with_resilience(
            _boom,
            provider="openai",
            config=cfg_failures,
            retry_fn=lambda f: f(),
        )
    with pytest.raises(LLMCircuitOpenError):
        execute_with_resilience(
            _boom,
            provider="openai",
            config=cfg_failures,
            retry_fn=lambda f: f(),
        )
    reset_provider_resilience("openai")


def test_provider_http_429_records_circuit_failure() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(circuit_breaker_threshold=1, max_retries=0)

    class RateLimited(RuntimeError):
        status_code = 429

    with pytest.raises(RateLimited):
        execute_with_resilience(
            lambda: (_ for _ in ()).throw(RateLimited("throttled")),
            provider="openai",
            config=cfg,
            retry_fn=lambda f: f(),
        )
    with pytest.raises(LLMCircuitOpenError):
        execute_with_resilience(
            lambda: "never",
            provider="openai",
            config=cfg,
            retry_fn=lambda f: f(),
        )
    reset_provider_resilience("openai")
