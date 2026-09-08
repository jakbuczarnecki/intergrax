# © Artur Czarnecki. All rights reserved.

"""Message bus provider selection vs materialization in host integration wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.integration_tool_wiring import wire_integration_tool_context
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationStatus,
)
from intergrax.integrations.registry.catalog import clear_catalog, register_integration
from intergrax.integrations.registry.catalog_manifests import KAFKA
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.host.message_bus_wiring import (
    local_workspace_message_bus_enabled,
    materialize_local_workspace_message_bus_profile,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_integration_profile,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def _register_fake(
    slug: str,
    *,
    categories: tuple[IntegrationCategory, ...],
    factory: Any,
) -> None:
    register_integration(
        IntegrationEntry(
            slug=slug,
            categories=categories,
            factory=factory,
            status=IntegrationStatus.BETA,
        )
    )


def test_wire_integration_tool_context_does_not_materialize_unselected_message_bus() -> None:
    kafka_factory = MagicMock(return_value=object())
    _register_fake(
        "kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=kafka_factory,
    )

    profile = IntegrationProfile()
    ctx = wire_integration_tool_context(ToolWiringContext(), profile)

    assert ctx.message_bus is None
    kafka_factory.assert_not_called()


def test_wire_integration_tool_context_does_not_materialize_kafka_binding_only() -> None:
    kafka_factory = MagicMock(return_value=object())
    redis_factory = MagicMock(return_value=object())
    _register_fake(
        "kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=kafka_factory,
    )
    _register_fake(
        "redis",
        categories=(IntegrationCategory.KEY_VALUE_CACHE,),
        factory=redis_factory,
    )
    kv_store = MagicMock(spec=DistributedKVStore)

    profile = IntegrationProfile.model_validate(
        {
            "message_bus": "kafka",
            "key_value_cache": "redis",
        }
    )
    ctx = wire_integration_tool_context(
        ToolWiringContext(key_value_cache=kv_store),
        profile,
    )

    assert ctx.message_bus is None
    kafka_factory.assert_not_called()


def test_wire_integration_tool_context_keeps_prebuilt_message_bus_instance() -> None:
    bus = object()
    profile = IntegrationProfile(message_bus=bus)
    ctx = wire_integration_tool_context(ToolWiringContext(), profile)
    assert ctx.message_bus is bus


def test_explicit_non_kafka_selection_materializes_only_selected_provider() -> None:
    kafka_factory = MagicMock(return_value=object())
    celery_factory = MagicMock(return_value=object())
    _register_fake(
        "kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=kafka_factory,
    )
    _register_fake(
        "celery",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=celery_factory,
    )

    profile = IntegrationProfile.model_validate({"message_bus": "celery"})
    resolved = resolve(IntegrationCategory.MESSAGE_BUS, profile=profile)
    ctx = wire_integration_tool_context(ToolWiringContext(), profile)

    assert resolved is not None
    celery_factory.assert_called_once()
    kafka_factory.assert_not_called()
    assert ctx.message_bus is None


def test_explicit_kafka_selection_materializes_via_profile_resolve() -> None:
    kafka_factory = MagicMock(return_value={"provider": "kafka"})
    _register_fake(
        "kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=kafka_factory,
    )
    kv_store = MagicMock(spec=DistributedKVStore)
    profile = IntegrationProfile.model_validate({"message_bus": "kafka"})

    resolved = resolve(
        IntegrationCategory.MESSAGE_BUS,
        profile=profile,
        config={"kv_store": kv_store},
    )

    assert resolved == {"provider": "kafka"}
    kafka_factory.assert_called_once()


def test_lkw_message_bus_disabled_does_not_select_kafka_or_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", raising=False)
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", raising=False)
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_REDIS", raising=False)

    assert local_workspace_message_bus_enabled() is False
    profile = materialize_local_workspace_message_bus_profile(
        build_local_workspace_integration_profile(),
    )

    assert profile.message_bus is None
    assert profile.key_value_cache is None


def test_lkw_message_bus_enabled_selects_kafka_and_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", raising=False)

    profile = materialize_local_workspace_message_bus_profile(
        build_local_workspace_integration_profile(),
    )

    assert profile.key_value_cache is not None
    assert profile.key_value_cache.manifest is not None
    assert profile.key_value_cache.manifest.slug == "redis"
    assert profile.message_bus is not None
    assert profile.message_bus.manifest is not None
    assert profile.message_bus.manifest.slug == KAFKA.slug


def test_wire_integration_tool_context_never_imports_kafka_factory() -> None:
    kafka_factory = MagicMock(return_value=object())
    _register_fake(
        "kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        factory=kafka_factory,
    )
    _register_fake(
        "redis",
        categories=(IntegrationCategory.KEY_VALUE_CACHE,),
        factory=MagicMock(return_value=object()),
    )
    profile = IntegrationProfile.model_validate(
        {
            "message_bus": "kafka",
            "key_value_cache": "redis",
        }
    )

    with patch(
        "intergrax.integrations.providers.message_bus.kafka.bundle.create_kafka_message_bus",
        side_effect=AssertionError("kafka factory must not be imported during wiring"),
    ):
        ctx = wire_integration_tool_context(ToolWiringContext(), profile)

    assert ctx.message_bus is None
