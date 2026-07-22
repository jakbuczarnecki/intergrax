# © Artur Czarnecki. All rights reserved.

"""lab_json / slash_command are runtime adapters — not provider taxonomy entries."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.integrations.registry import list_slugs, register_default_integrations
from intergrax.runtime.interactions.adapters.lab_json_adapter import LabJsonInteractionAdapter
from intergrax.runtime.interactions.adapters.slash_command_adapter import SlashCommandInteractionAdapter
from intergrax.runtime.interactions.factory import create_interaction_adapter, resolve_interaction_settings
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.contracts import PlatformIntegrationKind

pytestmark = pytest.mark.unit


def test_interaction_surface_absent_from_taxonomy() -> None:
    assert "interaction_surface" not in PROVIDER_CATEGORY_CONTRACT_REGISTRY
    assert "interaction_surface" not in {v for v in SLUG_CATEGORY.values()}
    assert not hasattr(PlatformIntegrationKind, "INTERACTION_SURFACE")
    assert "lab_json" not in SLUG_CATEGORY
    assert "slash_command" not in SLUG_CATEGORY


def test_mailgun_and_ollama_slug_mapping() -> None:
    assert SLUG_CATEGORY["mailgun"] == "notification_channel"
    assert SLUG_CATEGORY["ollama"] == "model_serving_runtime"


def test_lab_json_runtime_adapter_roundtrip() -> None:
    adapter = LabJsonInteractionAdapter()
    assert adapter.can_handle({"message": "hello", "capability": "echo.basic"})
    task = adapter.to_task(
        {"message": "hello", "capability": "echo.basic"},
        tenant_id="t1",
        user_id="u1",
    )
    assert task.message == "hello"
    assert task.context.capability == "echo.basic"


def test_slash_command_runtime_adapter_handles_payload() -> None:
    adapter = SlashCommandInteractionAdapter()
    assert adapter.can_handle({"command": "/echo", "text": "hi", "user_id": "u1"})
    inbound = adapter.to_inbound(
        {"command": "/echo", "text": "hi", "user_id": "u1"},
        tenant_id="t1",
        user_id="u1",
    )
    assert inbound.channel == "slash_command"


def test_runtime_factory_lab_json_surface() -> None:
    adapter = create_interaction_adapter(resolve_interaction_settings(surface="lab_json"))
    assert isinstance(adapter, LabJsonInteractionAdapter)


def test_lab_json_and_slash_command_absent_from_provider_registry() -> None:
    register_default_integrations(override=True)
    slugs = set(list_slugs())
    assert "lab_json" not in slugs
    assert "slash_command" not in slugs


def test_mailgun_notification_channel_disabled_construction() -> None:
    from intergrax.integrations.providers.notification_channel.mailgun.bundle import (
        create_mailgun_notification_channel_integration,
    )
    from intergrax.integrations.providers.notification_channel.mailgun.integration import (
        MailgunNotificationChannelIntegration,
    )
    from intergrax.runtime.integrations.categories.messaging import (
        NotificationChannelIntegrationContract,
    )

    integration = create_mailgun_notification_channel_integration(enabled=False)
    assert isinstance(integration, MailgunNotificationChannelIntegration)
    assert isinstance(integration, NotificationChannelIntegrationContract)
    assert integration.config.enabled is False
    assert integration.provider_id == "mailgun"
    assert integration.integration_kind == "notification_channel"


def test_ollama_model_serving_runtime_disabled_construction() -> None:
    from intergrax.integrations.providers.model_serving_runtime.ollama.bundle import (
        create_ollama_model_serving_runtime_integration,
    )
    from intergrax.integrations.providers.model_serving_runtime.ollama.integration import (
        OllamaModelServingRuntimeIntegration,
    )
    from intergrax.runtime.integrations.categories.ai import ModelServingRuntimeIntegrationContract

    integration = create_ollama_model_serving_runtime_integration(enabled=False)
    assert isinstance(integration, OllamaModelServingRuntimeIntegration)
    assert isinstance(integration, ModelServingRuntimeIntegrationContract)
    assert integration.config.enabled is False
    assert integration.provider_id == "ollama"
    assert integration.integration_kind == "model_serving_runtime"


def test_mailgun_inbound_adapter_preserved() -> None:
    from intergrax.integrations._shared.p5.factories import create_mailgun_inbound_adapter

    adapter = create_mailgun_inbound_adapter()
    payload = {"sender": "a@b.c", "body-plain": "hello", "subject": "s"}
    assert adapter.can_handle(payload)
    inbound = adapter.to_inbound(payload, tenant_id="t1", user_id="u1")
    assert inbound.channel == "mailgun"
    assert inbound.message == "hello"


def test_ollama_host_client_list_models() -> None:
    from intergrax.integrations.providers.model_serving_runtime.ollama.bundle import (
        create_ollama_model_serving_runtime,
    )

    class _Fake:
        def health(self) -> bool:
            return True

        def list_models(self) -> list[str]:
            return ["llama3"]

    runtime = create_ollama_model_serving_runtime(client=_Fake())
    assert runtime.list_models() == ["llama3"]
    assert runtime.health().healthy is True or runtime.health() is True
