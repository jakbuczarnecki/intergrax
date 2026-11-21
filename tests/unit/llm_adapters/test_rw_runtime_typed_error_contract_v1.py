# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-03 (NT-ROW-105, EX-D3-03): typed provider-error contract v1 at the adapter boundary.

Before the change no typed provider-error contract existed: SDK exceptions left the adapter as whatever the SDK
raised, and retry, failover and the model-routing qualifier classified them by class-name words, message
substrings or a direct SDK import. Fake SDK error shapes only (no network, no credential).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import (
    LLMCircuitOpenError,
    LLMRateLimitError,
    reset_provider_resilience,
)
from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PROVIDER = "rw-runtime-error-probe"
_SECRET = "sdk-detail-that-must-not-leak"


def _contract() -> ModuleType:
    spec = importlib.util.find_spec("intergrax.llm_adapters.contracts.provider_error")
    assert spec is not None, "typed provider-error contract v1 (contracts/provider_error.py) is missing"
    return importlib.import_module("intergrax.llm_adapters.contracts.provider_error")


class _ProbeAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.provider = _PROVIDER
        self.model = "probe-model"
        self.call_config = LLMCallConfig()

    @property
    def context_window_tokens(self) -> int:
        return 8_192

    def generate_messages(self, messages: Sequence[ChatMessage], **_: object) -> LLMAdapterResponse:
        return LLMAdapterResponse(content="")


@pytest.fixture(autouse=True)
def _clean_breaker() -> Iterator[None]:
    reset_provider_resilience(_PROVIDER)
    yield
    reset_provider_resilience(_PROVIDER)


# ---------------------------------------------------------------------------------------------
# SDK error shapes (class names and attributes as the SDKs expose them; no SDK import)
# ---------------------------------------------------------------------------------------------


def _status_error(name: str, status: int, *, headers: dict[str, str] | None = None) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls(f"{name}: {_SECRET}")
    exc.status_code = status  # type: ignore[attr-defined]
    exc.response = SimpleNamespace(status_code=status, headers=headers or {})  # type: ignore[attr-defined]
    return exc


def _named_error(name: str) -> Exception:
    return type(name, (Exception,), {})(f"{name}: {_SECRET}")


#: fixture id -> (raw error factory, expected category value, expected status, expected retry-after)
_FIXTURES: dict[str, tuple[Any, str, int | None, float | None]] = {
    "openai-timeout": (lambda: _named_error("APITimeoutError"), "timeout", None, None),
    "openai-connection": (lambda: _named_error("APIConnectionError"), "connection", None, None),
    "openai-rate-limit": (
        lambda: _status_error("RateLimitError", 429, headers={"retry-after": "3"}),
        "rate_limit",
        429,
        3.0,
    ),
    "openai-authentication": (lambda: _status_error("AuthenticationError", 401), "authentication", 401, None),
    "openai-permission": (lambda: _status_error("PermissionDeniedError", 403), "permission", 403, None),
    "openai-bad-request": (lambda: _status_error("BadRequestError", 400), "invalid_request", 400, None),
    "openai-unprocessable": (lambda: _status_error("UnprocessableEntityError", 422), "invalid_request", 422, None),
    "openai-not-found": (lambda: _status_error("NotFoundError", 404), "model_not_found", 404, None),
    "generic-payment-required": (lambda: _status_error("APIStatusError", 402), "quota_exhausted", 402, None),
    "openai-server": (lambda: _status_error("InternalServerError", 500), "server_error", 500, None),
    "generic-unavailable": (lambda: _status_error("APIStatusError", 503), "overloaded", 503, None),
    "anthropic-overloaded": (lambda: _status_error("OverloadedError", 529), "overloaded", 529, None),
    "ollama-response-error-404": (lambda: _status_error("ResponseError", 404), "model_not_found", 404, None),
    "ollama-response-error-500": (lambda: _status_error("ResponseError", 500), "server_error", 500, None),
    "httpx-read-timeout": (lambda: _named_error("ReadTimeout"), "timeout", None, None),
    "builtin-connection-refused": (lambda: ConnectionRefusedError(_SECRET), "connection", None, None),
    "builtin-timeout": (lambda: TimeoutError(_SECRET), "timeout", None, None),
}


def test_the_contract_module_exists_and_is_version_1() -> None:
    contract = _contract()
    assert contract.LLM_PROVIDER_ERROR_CONTRACT_VERSION == 1
    assert {member.value for member in contract.LLMProviderErrorCategory} == {
        "timeout",
        "rate_limit",
        "authentication",
        "permission",
        "invalid_request",
        "model_not_found",
        "quota_exhausted",
        "overloaded",
        "server_error",
        "connection",
        "unknown",
    }
    assert issubclass(contract.LLMProviderError, RuntimeError)
    for category in contract.LLMProviderErrorCategory:
        cls = contract.provider_error_class_for(category)
        assert issubclass(cls, contract.LLMProviderError)
        assert cls.category is category


@pytest.mark.parametrize("fixture_id", sorted(_FIXTURES))
def test_an_sdk_error_maps_to_its_typed_class_at_the_adapter_boundary(fixture_id: str) -> None:
    contract = _contract()
    make, category, status, retry_after = _FIXTURES[fixture_id]
    raw = make()
    adapter = _ProbeAdapter()

    def _sdk() -> str:
        raise raw

    with pytest.raises(contract.LLMProviderError) as info:
        adapter._execute(_sdk)
    typed = info.value
    assert typed.category is contract.LLMProviderErrorCategory(category)
    assert type(typed) is contract.provider_error_class_for(typed.category)
    assert typed.provider == _PROVIDER
    assert typed.model == "probe-model"
    assert typed.status_code == status
    assert typed.retry_after_seconds == retry_after
    assert typed.__cause__ is raw
    assert _SECRET not in str(typed), "the SDK message text stays on __cause__ (Q-RUN-03)"


@pytest.mark.parametrize("fixture_id", sorted(_FIXTURES))
def test_retry_and_failover_classification_is_unchanged_by_the_mapping(fixture_id: str) -> None:
    contract = _contract()
    raw = _FIXTURES[fixture_id][0]()
    adapter = _ProbeAdapter()

    def _sdk() -> str:
        raise raw

    with pytest.raises(contract.LLMProviderError) as info:
        adapter._execute(_sdk)
    config = LLMCallConfig()
    assert is_retriable_provider_error(info.value, config) == is_retriable_provider_error(raw, config)
    assert info.value.retriable == is_retriable_provider_error(raw, config)


def _pass_through_errors() -> list[BaseException]:
    from intergrax.contracts.execution_deadline.admission import ExecutionProtectedWorkAdmissionResult
    from intergrax.contracts.retry_budget import RetryBudgetExhaustedError
    from intergrax.llm_adapters.governance.quota import LLMQuotaExceeded
    from intergrax.runtime.cancellation.coordinator import CooperativeCancellationAbort
    from intergrax.runtime.execution.deadline_provider_guard import ExecutionProtectedWorkDeniedError

    class _HostInterrupt(BaseException):
        pass

    return [
        LLMRateLimitError("rate"),
        LLMCircuitOpenError("open"),
        RetryBudgetExhaustedError("budget"),
        LLMQuotaExceeded("quota"),
        ExecutionProtectedWorkDeniedError(ExecutionProtectedWorkAdmissionResult.EXPIRED),
        LLMCapabilityMismatchError(LLMFeature.TOOLS),
        CooperativeCancellationAbort(),
        _HostInterrupt(),
        ValueError("a local error that carries no provider signal"),
    ]


@pytest.mark.parametrize("index", range(len(_pass_through_errors())))
def test_pre_existing_typed_errors_and_signal_free_local_errors_pass_through_unchanged(index: int) -> None:
    _contract()
    error = _pass_through_errors()[index]
    adapter = _ProbeAdapter()

    def _sdk() -> str:
        raise error

    with pytest.raises(BaseException) as info:
        adapter._execute(_sdk)
    assert info.value is error


def test_an_sdk_error_without_a_signal_is_typed_unknown() -> None:
    contract = _contract()
    import openai

    raw = openai.OpenAIError(_SECRET)
    adapter = _ProbeAdapter()

    def _sdk() -> str:
        raise raw

    with pytest.raises(contract.LLMProviderUnknownError) as info:
        adapter._execute(_sdk)
    assert info.value.__cause__ is raw


def test_a_typed_error_is_not_wrapped_twice() -> None:
    contract = _contract()
    original = contract.LLMProviderServerError(provider="p", model="m", status_code=500)
    adapter = _ProbeAdapter()

    def _sdk() -> str:
        raise original

    with pytest.raises(contract.LLMProviderServerError) as info:
        adapter._execute(_sdk)
    assert info.value is original


def test_a_stream_open_error_and_a_mid_stream_error_are_both_typed() -> None:
    contract = _contract()
    adapter = _ProbeAdapter()

    def failing_open() -> Iterable[str]:
        raise _status_error("InternalServerError", 500)

    with pytest.raises(contract.LLMProviderServerError):
        list(adapter._execute_streaming(failing_open))

    raw_mid = _status_error("InternalServerError", 500)

    def mid_stream() -> Iterable[str]:
        def events() -> Iterator[str]:
            yield "a"
            raise raw_mid

        return events()

    stream = iter(adapter._execute_streaming(mid_stream))
    assert next(stream) == "a"
    with pytest.raises(contract.LLMProviderServerError) as info:
        next(stream)
    assert info.value.__cause__ is raw_mid


def test_a_shipped_adapter_maps_its_sdk_error_and_labels_the_failure_with_the_typed_class() -> None:
    contract = _contract()
    from unittest.mock import MagicMock

    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.side_effect = _status_error("NotFoundError", 404)
    adapter = OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    ended: list[Any] = []
    original_end = adapter.usage.end_call

    def _spy(call: Any, **kwargs: Any) -> None:
        ended.append(kwargs.get("error_type"))
        original_end(call, **kwargs)

    adapter.usage.end_call = _spy  # type: ignore[method-assign]
    with pytest.raises(contract.LLMProviderModelNotFoundError) as info:
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert info.value.provider == "openai"
    assert info.value.model == "gpt-4o-mini"
    assert ended == ["LLMProviderModelNotFoundError"]


@pytest.mark.parametrize(
    ("code", "status", "category"),
    [
        ("ThrottlingException", 429, "rate_limit"),
        ("ResourceNotFoundException", 404, "model_not_found"),
        ("AccessDeniedException", 403, "permission"),
        ("ValidationException", 400, "invalid_request"),
        ("ServiceQuotaExceededException", 400, "quota_exhausted"),
        ("ModelTimeoutException", 408, "timeout"),
        ("ServiceUnavailableException", 503, "overloaded"),
        ("InternalServerException", 500, "server_error"),
    ],
)
def test_bedrock_client_error_codes_map_through_the_provider_hook(code: str, status: int, category: str) -> None:
    contract = _contract()
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    raw = ClientError(
        {"Error": {"Code": code, "Message": _SECRET}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Converse",
    )
    client = SimpleNamespace(converse=MagicMock(side_effect=raw), converse_stream=MagicMock())
    adapter = BedrockChatAdapter(
        client=client, model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1", use_converse=True
    )
    reset_provider_resilience(adapter._provider_slug())
    with pytest.raises(contract.LLMProviderError) as info:
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert info.value.category is contract.LLMProviderErrorCategory(category)
    assert info.value.status_code == status
    assert info.value.__cause__ is raw
    assert _SECRET not in str(info.value)


def test_the_model_routing_qualifier_classifies_typed_errors_without_the_sdk() -> None:
    contract = _contract()
    from model_routing_qualifier.qualification_types import (
        InvocationFailureKind,
        classify_invocation_failure,
    )

    not_found = contract.LLMProviderModelNotFoundError(provider="ollama", model="m", status_code=404)
    timeout = contract.LLMProviderTimeoutError(provider="ollama", model="m", status_code=None)
    connection = contract.LLMProviderConnectionError(provider="ollama", model="m", status_code=None)
    assert classify_invocation_failure(not_found) is InvocationFailureKind.PROVIDER_MODEL_NOT_FOUND
    assert classify_invocation_failure(timeout) is InvocationFailureKind.PROVIDER_TRANSPORT_FAILURE
    assert classify_invocation_failure(connection) is InvocationFailureKind.PROVIDER_TRANSPORT_FAILURE

    source = (_REPO / "agents" / "model_routing_qualifier" / "qualification_types.py").read_text(encoding="utf-8")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {(node.module or "").split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    assert "ollama" not in imported


def test_the_contract_module_uses_concrete_types_only() -> None:
    """OD-AF-11 ratchet: no ``Any``, no ``object`` annotation, no reflection dispatch in the new contract."""
    _contract()
    path = _REPO / "intergrax" / "llm_adapters" / "contracts" / "provider_error.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "Any" not in names and "Any" not in attributes
    assert "object" not in names
    assert not {"getattr", "setattr", "hasattr", "eval", "exec", "__import__"} & names


class _NoTimeoutForwardingProbeAdapter(_ProbeAdapter):
    def supports_call_timeout(self) -> bool:
        return False


def test_the_fail_closed_timeout_refusal_is_not_retriable_and_stops_failover() -> None:
    """Review finding L-2 of PR #670: the refusal is deterministic, so retry and failover must not see it as a
    transient error (they read the transient words of a class name when there is no status code)."""
    from intergrax.llm_adapters.contracts.failover_policy import FailoverDecision, FailoverProgressionContext
    from intergrax.llm_adapters.registry.failover_policy import PlatformDefaultFailoverPolicy

    adapter = _NoTimeoutForwardingProbeAdapter()
    adapter.call_config = LLMCallConfig(timeout_sec=5.0, max_retries=2, retry_backoff_sec=0.0)
    calls: list[int] = []
    with pytest.raises(LLMCapabilityMismatchError) as info:
        adapter._execute(lambda: calls.append(1))
    refusal = info.value
    assert calls == [], "the refusal is raised before the provider is touched"
    assert is_retriable_provider_error(refusal, LLMCallConfig()) is False
    assert _contract().category_from_class_name(type(refusal).__name__) is None
    context = FailoverProgressionContext(
        attempt_index=0, is_last_candidate=False, error=refusal, failover_retry_config=LLMCallConfig()
    )
    assert PlatformDefaultFailoverPolicy().decide_after_failure(context) is FailoverDecision.STOP


# ---------------------------------------------------------------------------------------------
# Builtin compatibility of the timeout and connection categories (orchestrator decision on the PDC-D3
# f08 red: ``except ConnectionError`` / ``except TimeoutError`` consumers keep working)
# ---------------------------------------------------------------------------------------------

#: fixture id -> (typed class name, builtin base the typed error must also be)
_BUILTIN_COMPATIBLE: dict[str, tuple[str, type[BaseException]]] = {
    "openai-timeout": ("LLMProviderTimeoutError", TimeoutError),
    "httpx-read-timeout": ("LLMProviderTimeoutError", TimeoutError),
    "builtin-timeout": ("LLMProviderTimeoutError", TimeoutError),
    "openai-connection": ("LLMProviderConnectionError", ConnectionError),
    "builtin-connection-refused": ("LLMProviderConnectionError", ConnectionError),
}


@pytest.mark.parametrize("fixture_id", sorted(_BUILTIN_COMPATIBLE))
def test_a_typed_timeout_or_connection_error_is_caught_by_the_builtin_except(fixture_id: str) -> None:
    contract = _contract()
    make, category, status, retry_after = _FIXTURES[fixture_id]
    class_name, builtin = _BUILTIN_COMPATIBLE[fixture_id]
    raw = make()
    adapter = _ProbeAdapter()

    def _raise() -> None:
        raise raw

    with pytest.raises(builtin) as info:
        adapter._execute(_raise)
    typed = info.value
    assert type(typed).__name__ == class_name
    assert isinstance(typed, contract.LLMProviderError) and isinstance(typed, RuntimeError)
    assert typed.category.value == category
    assert (typed.provider, typed.model) == (_PROVIDER, "probe-model")
    assert typed.status_code == status and typed.retry_after_seconds == retry_after
    assert typed.__cause__ is raw
    assert _SECRET not in str(typed)
    assert str(typed).startswith("LLM provider error category=")


def test_the_builtin_bases_keep_the_typed_layout_and_the_retry_classification() -> None:
    contract = _contract()
    config = LLMCallConfig()
    for cls, builtin in (
        (contract.LLMProviderTimeoutError, TimeoutError),
        (contract.LLMProviderConnectionError, ConnectionError),
    ):
        assert issubclass(cls, builtin) and issubclass(cls, RuntimeError)
        assert cls.__mro__.index(contract.LLMProviderError) < cls.__mro__.index(builtin)
        error = cls(provider="p", model="m", status_code=None, retry_after_seconds=1.5, retriable=True)
        assert (error.provider, error.model, error.status_code, error.retry_after_seconds) == ("p", "m", None, 1.5)
        assert error.retriable is True
        assert error.errno is None and error.strerror is None
        assert error.args == (str(error),)
        # Same verdict as before the builtin base: transient by class-name word, status-driven when set.
        assert is_retriable_provider_error(error, config) is True
        assert is_retriable_provider_error(cls(provider="p", model="m", status_code=503), config) is True
        assert is_retriable_provider_error(cls(provider="p", model="m", status_code=400), config) is True
    for other in contract.LLMProviderErrorCategory:
        cls = contract.provider_error_class_for(other)
        if other not in (contract.LLMProviderErrorCategory.TIMEOUT, contract.LLMProviderErrorCategory.CONNECTION):
            assert not issubclass(cls, OSError), f"{cls.__name__} must not gain a builtin OSError base"
    assert not issubclass(contract.LLMCallBudgetNotForwardableError, OSError)
    assert not issubclass(contract.LLMCallBudgetNotForwardableError, TimeoutError)
