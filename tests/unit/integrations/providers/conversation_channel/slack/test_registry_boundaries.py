# © Artur Czarnecki. All rights reserved.

"""Slack conversation registry metadata and import-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.integrations.registry_v2 import build_contract_registry, build_integration_registration

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _canonical_catalog_bootstrap() -> None:
    from intergrax.integrations.registry.bootstrap import (
        register_default_integrations,
        reset_default_integrations_state,
    )
    from intergrax.integrations.registry.catalog import clear_catalog

    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    from intergrax.integrations.providers.conversation_channel.google_chat.register import (
        register_google_chat_integration,
    )
    from intergrax.integrations.providers.conversation_channel.mattermost.register import (
        register_mattermost_integration,
    )
    from intergrax.integrations.providers.conversation_channel.rocket_chat.register import (
        register_rocket_chat_integration,
    )

    register_mattermost_integration()
    register_rocket_chat_integration()
    register_google_chat_integration()
    yield
    clear_catalog()
    reset_default_integrations_state()


_SLACK_PROVIDER_ROOT = (
    Path(__file__).resolve().parents[6]
    / "intergrax"
    / "integrations"
    / "providers"
    / "conversation_channel"
    / "slack"
)

_FORBIDDEN_IMPORT_MARKERS = (
    "local_workspace_application",
    "WorkspaceAskService",
    "DocumentStore",
    "Qdrant",
)


def test_slack_conversation_runtime_binding_supported() -> None:
    registration = build_integration_registration("slack", category="conversation_channel")
    assert registration.supports_runtime_binding is True
    assert registration.supports_health_check is True
    assert registration.metadata["runtime_binding_supported"] is True
    assert registration.metadata["runtime_implemented"] is True


@pytest.mark.parametrize(
    "slug",
    ("teams", "discord", "telegram", "mattermost", "rocket_chat", "google_chat"),
)
def test_other_conversation_providers_remain_unbound(slug: str) -> None:
    registration = build_integration_registration(slug, category="conversation_channel")
    assert registration.supports_runtime_binding is False
    assert registration.metadata["runtime_implemented"] is False


def test_slack_notification_registration_remains_separate() -> None:
    registry = build_contract_registry(slugs=("slack",))
    notification = registry.get(provider_id="slack", category="notification_channel")
    conversation = registry.get(provider_id="slack", category="conversation_channel")
    assert notification.integration_class is not conversation.integration_class
    assert notification.supports_runtime_binding is True
    assert conversation.supports_runtime_binding is True


def test_slack_provider_has_no_lkw_imports() -> None:
    assert _SLACK_PROVIDER_ROOT.is_dir()
    offenders: list[str] = []
    for path in _SLACK_PROVIDER_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_IMPORT_MARKERS:
            if marker in text:
                offenders.append(f"{path.name}:{marker}")
    assert offenders == []
