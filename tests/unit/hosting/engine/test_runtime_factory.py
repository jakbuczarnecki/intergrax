# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect

import pytest

from intergrax.hosting import HostedApplicationContext
from intergrax.hosting.engine.runtime import invoke_application_factory
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationRuntimeError
from tests.unit.hosting.engine._fakes import (
    FakeRuntime,
    async_runtime_factory,
    async_runtime_factory_with_context,
)
from tests.unit.hosting._helpers import build_sample_context

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_sync_zero_argument_factory() -> None:
    calls = {"count": 0}

    def factory() -> FakeRuntime:
        calls["count"] += 1
        return FakeRuntime()

    runtime = await invoke_application_factory(factory, build_sample_context())
    assert isinstance(runtime, FakeRuntime)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_sync_context_factory() -> None:
    seen: dict[str, HostedApplicationContext | None] = {"context": None}

    def factory(context: HostedApplicationContext) -> FakeRuntime:
        seen["context"] = context
        return FakeRuntime()

    await invoke_application_factory(factory, build_sample_context())
    assert seen["context"] is not None


@pytest.mark.asyncio
async def test_async_factories() -> None:
    await invoke_application_factory(async_runtime_factory, build_sample_context())
    await invoke_application_factory(async_runtime_factory_with_context, build_sample_context())


def test_invalid_signature_rejected() -> None:
    def factory(a: str, b: str) -> FakeRuntime:
        return FakeRuntime()

    with pytest.raises(HostedApplicationConfigurationError):
        inspect.signature(factory)
        __import__("asyncio").run(invoke_application_factory(factory, build_sample_context()))


@pytest.mark.asyncio
async def test_incompatible_result_rejected() -> None:
    def factory() -> object:
        return object()

    with pytest.raises(HostedApplicationConfigurationError):
        await invoke_application_factory(factory, build_sample_context())


@pytest.mark.asyncio
async def test_factory_exception_preserved() -> None:
    def factory() -> FakeRuntime:
        raise ValueError("boom")

    with pytest.raises(HostedApplicationRuntimeError) as exc_info:
        await invoke_application_factory(factory, build_sample_context())
    assert isinstance(exc_info.value.__cause__, ValueError)
