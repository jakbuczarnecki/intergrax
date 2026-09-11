# © Artur Czarnecki. All rights reserved.

"""W2-ADR — dependency concurrency admission contract validation."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionError,
    DependencyConcurrencyAdmissionPort,
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPermit,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyMissingError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "intergrax" / "contracts" / "dependency_concurrency_admission.py"


def test_identity_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        DependencyConcurrencyIdentity(kind=DependencyConcurrencyKind.TOOL, value="")


def test_identity_rejects_untrimmed_value() -> None:
    with pytest.raises(ValueError, match="stripped"):
        DependencyConcurrencyIdentity(kind=DependencyConcurrencyKind.TOOL, value=" foo")


def test_policy_rejects_max_below_one() -> None:
    with pytest.raises(ValidationError):
        DependencyConcurrencyPolicy(
            max_concurrent_calls=0,
            overload_mode=DependencyConcurrencyOverloadMode.REJECT,
            wait_timeout_seconds=None,
        )


def test_wait_with_timeout_requires_timeout() -> None:
    with pytest.raises(ValidationError, match="wait_timeout_seconds"):
        DependencyConcurrencyPolicy(
            max_concurrent_calls=1,
            overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
            wait_timeout_seconds=None,
        )


def test_reject_rejects_timeout_field() -> None:
    with pytest.raises(ValidationError, match="wait_timeout_seconds"):
        DependencyConcurrencyPolicy(
            max_concurrent_calls=1,
            overload_mode=DependencyConcurrencyOverloadMode.REJECT,
            wait_timeout_seconds=1.0,
        )


def test_protocols_runtime_checkable() -> None:
    class _Permit:
        async def release(self) -> None:
            return None

    class _Port:
        async def acquire(
            self,
            request: DependencyConcurrencyAdmissionRequest,
        ) -> DependencyConcurrencyPermit:
            return _Permit()

    assert isinstance(_Permit(), DependencyConcurrencyPermit)
    assert isinstance(_Port(), DependencyConcurrencyAdmissionPort)


def test_request_immutable() -> None:
    identity = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value="search",
    )
    request = DependencyConcurrencyAdmissionRequest(dependency=identity, tenant_id="t1")
    with pytest.raises(FrozenInstanceError):
        setattr(request, "tenant_id", "t2")


def test_policy_immutable_and_extra_forbidden() -> None:
    policy = DependencyConcurrencyPolicy(
        max_concurrent_calls=2,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )
    with pytest.raises(ValidationError):
        DependencyConcurrencyPolicy.model_validate(
            {
                "max_concurrent_calls": 2,
                "overload_mode": "REJECT",
                "wait_timeout_seconds": None,
                "surprise": 1,
            }
        )
    with pytest.raises(ValidationError):
        policy.max_concurrent_calls = 3


def test_tool_and_llm_provider_same_value_differ_by_kind() -> None:
    tool = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value="openai",
    )
    provider = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.LLM_PROVIDER,
        value="openai",
    )
    assert tool != provider
    assert hash(tool) != hash(provider)


def test_identity_hashable_as_dict_key() -> None:
    a = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.LLM_PROVIDER,
        value="anthropic",
    )
    b = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.LLM_PROVIDER,
        value="anthropic",
    )
    c = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.LLM_PROVIDER,
        value="openai",
    )
    mapping = {a: "first"}
    mapping[b] = "second"
    assert mapping[a] == "second"
    assert c not in mapping


def test_typed_errors_distinct() -> None:
    assert issubclass(DependencyConcurrencyExceededError, DependencyConcurrencyAdmissionError)
    assert issubclass(DependencyConcurrencyAdmissionTimeoutError, DependencyConcurrencyAdmissionError)
    assert issubclass(DependencyConcurrencyPolicyMissingError, DependencyConcurrencyAdmissionError)
    assert not issubclass(
        DependencyConcurrencyExceededError,
        DependencyConcurrencyAdmissionTimeoutError,
    )


def test_w2_contract_does_not_import_w1_capacity_modules() -> None:
    tree = ast.parse(_CONTRACT_PATH.read_text(encoding="utf-8"), filename=str(_CONTRACT_PATH))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = (
        "intergrax.contracts.execution_capacity_admission",
        "intergrax.contracts.concurrent_execution_work",
    )
    for name in imported:
        for bad in forbidden:
            assert name != bad and not name.startswith(f"{bad}."), (
                f"forbidden import {name} in dependency concurrency contract"
            )


def test_module_docstring_states_non_goals() -> None:
    doc = inspect.getdoc(
        __import__(
            "intergrax.contracts.dependency_concurrency_admission",
            fromlist=["dependency_concurrency_admission"],
        )
    )
    assert doc is not None
    assert "rate limiting" in doc.lower()
    assert "root execution admission" in doc.lower()
