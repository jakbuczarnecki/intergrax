# © Artur Czarnecki. All rights reserved.

"""W2-B3 — process-local LLM provider dependency admission."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Callable

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.provider_dependency_boundary import (
    get_llm_provider_dependency_boundary,
    set_llm_provider_dependency_boundary,
)
from intergrax.llm_adapters._shared.resilience import (
    LLMCircuitOpenError,
    execute_with_resilience,
    reset_provider_resilience,
)
from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

pytestmark = pytest.mark.unit


def _provider_policy(capacity: int) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _llm_identity(slug: str) -> DependencyConcurrencyIdentity:
    return DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.LLM_PROVIDER,
        value=slug,
    )


@pytest.fixture(autouse=True)
def _clear_provider_boundary() -> Iterator[None]:
    set_llm_provider_dependency_boundary(None)
    yield
    set_llm_provider_dependency_boundary(None)


@pytest.fixture
def admission_boundary() -> Iterator[DependencyAttemptExecutionBoundary]:
    admission = LocalDependencyConcurrencyAdmission(
        {
            _llm_identity("openai"): _provider_policy(1),
            _llm_identity("claude"): _provider_policy(1),
        }
    )
    boundary = DependencyAttemptExecutionBoundary(admission)
    set_llm_provider_dependency_boundary(boundary)
    yield boundary
    boundary.close()


class _SdkCountingAdapter(LLMAdapter):
    def __init__(self, slug: str) -> None:
        super().__init__()
        self.provider = slug
        self.model = "test-model"
        self.sdk_calls = 0
        self._sdk_factory: Callable[[], str] = lambda: "ok"
        boundary = get_llm_provider_dependency_boundary()
        if boundary is not None:
            self.bind_provider_dependency_boundary(boundary)

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def set_sdk(self, factory: Callable[[], str]) -> None:
        self._sdk_factory = factory

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ):
        del messages, temperature, max_tokens, run_id

        def _sdk() -> str:
            self.sdk_calls += 1
            return self._sdk_factory()

        text = self._execute(_sdk)
        return build_adapter_response(
            content=text,
            model=self.model,
            provider=self._provider_slug(),
        )


def test_admission_exceeded_does_not_call_sdk_or_poison_circuit(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    reset_provider_resilience("openai")
    adapter = _SdkCountingAdapter("openai")
    adapter.call_config = LLMCallConfig(circuit_breaker_threshold=1)
    gate = threading.Event()

    adapter.set_sdk(lambda: gate.wait(timeout=5) or "held")
    worker = threading.Thread(
        target=lambda: adapter.generate_messages([ChatMessage(role="user", content="a")])
    )
    worker.start()
    time.sleep(0.05)

    blocked = _SdkCountingAdapter("openai")
    blocked.call_config = adapter.call_config
    blocked.set_sdk(lambda: "second")
    with pytest.raises(DependencyConcurrencyExceededError):
        blocked.generate_messages([ChatMessage(role="user", content="b")])
    assert blocked.sdk_calls == 0

    gate.set()
    worker.join(timeout=5)

    reset_provider_resilience("openai")
    cfg = LLMCallConfig(circuit_breaker_threshold=2)

    def _boom() -> str:
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        execute_with_resilience(_boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    execute_with_resilience(lambda: "ok", provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(RuntimeError):
        execute_with_resilience(_boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(RuntimeError):
        execute_with_resilience(_boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(LLMCircuitOpenError):
        execute_with_resilience(_boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    reset_provider_resilience("openai")


def test_admission_timeout_not_retriable() -> None:
    cfg = LLMCallConfig()
    err = DependencyConcurrencyAdmissionTimeoutError("waited")
    assert is_retriable_provider_error(err, cfg) is False


def test_retry_reacquires_per_physical_attempt(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    adapter = _SdkCountingAdapter("openai")
    adapter.call_config = LLMCallConfig(max_retries=2, retry_backoff_sec=0.0)
    attempts = {"n": 0}

    class Transient(RuntimeError):
        status_code = 503

    def _sdk() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Transient("x")
        return "ok"

    adapter.set_sdk(_sdk)
    result = adapter.generate_messages([ChatMessage(role="user", content="x")])
    assert result.content == "ok"
    assert adapter.sdk_calls == 3


def test_failover_separate_provider_permits(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    openai = _SdkCountingAdapter("openai")
    claude = _SdkCountingAdapter("claude")

    class Transient(RuntimeError):
        status_code = 503

    openai.set_sdk(lambda: (_ for _ in ()).throw(Transient("openai")))
    claude.set_sdk(lambda: "claude-ok")
    failover = FailoverLLMAdapter([openai, claude], profile_ids=("p0", "p1"))
    failover.call_config = LLMCallConfig(max_retries=0)
    response = failover.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "claude-ok"
    assert openai.sdk_calls == 1
    assert claude.sdk_calls == 1


def test_same_provider_shared_capacity(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    a = _SdkCountingAdapter("openai")
    b = _SdkCountingAdapter("openai")
    gate = threading.Event()
    a.set_sdk(lambda: gate.wait(timeout=5) or "a")
    b.set_sdk(lambda: "b")
    worker = threading.Thread(
        target=lambda: a.generate_messages([ChatMessage(role="user", content="1")])
    )
    worker.start()
    time.sleep(0.05)
    with pytest.raises(DependencyConcurrencyExceededError):
        b.generate_messages([ChatMessage(role="user", content="2")])
    gate.set()
    worker.join(timeout=5)


def test_different_provider_not_blocked(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    openai = _SdkCountingAdapter("openai")
    claude = _SdkCountingAdapter("claude")
    gate = threading.Event()
    openai.set_sdk(lambda: gate.wait(timeout=5) or "openai")
    claude.set_sdk(lambda: "free")
    worker = threading.Thread(
        target=lambda: openai.generate_messages([ChatMessage(role="user", content="x")])
    )
    worker.start()
    time.sleep(0.05)
    out = claude.generate_messages([ChatMessage(role="user", content="y")])
    assert out.content == "free"
    gate.set()
    worker.join(timeout=5)


def test_stream_holds_permit_until_exhausted(
    admission_boundary: DependencyAttemptExecutionBoundary,
) -> None:
    del admission_boundary
    holder = _SdkCountingAdapter("openai")
    gate = threading.Event()

    class _TwoStepStream:
        def __init__(self) -> None:
            self._step = 0

        def __iter__(self) -> _TwoStepStream:
            return self

        def __next__(self) -> str:
            if self._step == 0:
                self._step = 1
                return "a"
            if self._step == 1:
                gate.wait(timeout=5)
                self._step = 2
                return "b"
            raise StopIteration

    def _factory() -> Iterable[str]:
        holder.sdk_calls += 1
        return _TwoStepStream()

    stream = holder._execute_streaming(_factory)
    it = iter(stream)
    assert next(it) == "a"
    blocked = _SdkCountingAdapter("openai")
    blocked.set_sdk(lambda: "x")
    with pytest.raises(DependencyConcurrencyExceededError):
        blocked.generate_messages([ChatMessage(role="user", content="z")])
    gate.set()
    assert next(it) == "b"
    with pytest.raises(StopIteration):
        next(it)
