from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
    VendorKnowledgeSourcePluginConflict,
    VendorKnowledgeSourcePluginNotFound,
    VendorKnowledgeSourcePluginRegistry,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MsGraphTeamsChatKnowledgeAdapter,
    register_msgraph_teams_chat_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SlackConversationKnowledgeAdapter,
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph.registration import (
    build_msgraph_live_registration_bundles,
    build_msgraph_mail_vendor_knowledge_source_plugin,
    build_msgraph_teams_chat_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    publish_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_live_registration_bundles,
    build_slack_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry


def _identity(
    *,
    provider_id: str = "provider",
    source_kind: str = "source",
    integration_category: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
) -> VendorKnowledgeSourceIdentity:
    return VendorKnowledgeSourceIdentity(
        provider_id=provider_id,
        integration_category=integration_category,
        source_kind=source_kind,
    )


def _mode(
    mode: VendorKnowledgeMode,
    *,
    runtime_ref: str = "runtime:source",
    capability_refs: tuple[str, ...] = (),
    constraints: dict[str, object] | None = None,
) -> VendorKnowledgeModeCapability:
    return VendorKnowledgeModeCapability(
        mode=mode,
        contract_version=f"vendor-knowledge.{mode.value.casefold()}.v1",
        operations=("read",),
        runtime_ref=runtime_ref,
        capability_refs=capability_refs,
        constraints=constraints or {},
    )


def test_identity_and_mode_subsets_are_explicit() -> None:
    slack = build_slack_vendor_knowledge_source_plugin()
    graph = build_msgraph_teams_chat_vendor_knowledge_source_plugin()
    graph_mail = build_msgraph_mail_vendor_knowledge_source_plugin()

    assert slack.identity.key == (
        SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        IntegrationCategory.CONVERSATION_CHANNEL,
        SLACK_CONVERSATION_SOURCE_KIND,
    )
    assert {
        capability.mode for capability in slack.capabilities
    } == {
        VendorKnowledgeMode.INDEXED,
        VendorKnowledgeMode.DURABLE,
        VendorKnowledgeMode.LIVE,
    }
    assert graph.identity.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    assert graph.supports(VendorKnowledgeMode.DURABLE)
    assert graph.supports(VendorKnowledgeMode.LIVE)
    assert graph.supports(VendorKnowledgeMode.INDEXED)
    assert graph_mail.identity.source_kind == MSGRAPH_MAIL_SOURCE_KIND
    assert graph_mail.supports(VendorKnowledgeMode.DURABLE)
    assert graph_mail.supports(VendorKnowledgeMode.LIVE)
    assert graph_mail.supports(VendorKnowledgeMode.INDEXED)


def test_live_capability_refs_are_existing_registrations() -> None:
    slack = build_slack_vendor_knowledge_source_plugin()
    slack_published = publish_live_registration_bundles(build_slack_live_registration_bundles())
    slack_live = slack.capability(VendorKnowledgeMode.LIVE)
    assert slack_live is not None
    assert set(slack_live.capability_refs) == {
        key[2] for key in slack_published.handlers
    }

    graph = build_msgraph_teams_chat_vendor_knowledge_source_plugin()
    graph_published = publish_live_registration_bundles(build_msgraph_live_registration_bundles())
    graph_live = graph.capability(VendorKnowledgeMode.LIVE)
    assert graph_live is not None
    teams_chat_refs = {
        key for key, descriptor in graph_published.descriptors.items()
        if descriptor.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    }
    assert set(graph_live.capability_refs) == {key[2] for key in teams_chat_refs}


def test_existing_adapter_registry_remains_the_durable_execution_registry() -> None:
    registry = KnowledgeAdapterRegistry()
    slack = register_slack_conversation_knowledge_adapter(registry)
    graph = register_msgraph_teams_chat_knowledge_adapter(registry)

    assert isinstance(slack, SlackConversationKnowledgeAdapter)
    assert isinstance(graph, MsGraphTeamsChatKnowledgeAdapter)
    assert (
        SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        IntegrationCategory.CONVERSATION_CHANNEL,
        SLACK_CONVERSATION_SOURCE_KIND,
    ) in registry.registered_keys()
    assert (
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    ) in registry.registered_keys()


def test_registry_discovery_is_deterministic_and_unknown_is_distinct() -> None:
    slack = build_slack_vendor_knowledge_source_plugin()
    graph = build_msgraph_teams_chat_vendor_knowledge_source_plugin()
    first = VendorKnowledgeSourcePluginRegistry()
    second = VendorKnowledgeSourcePluginRegistry()
    first.register(graph)
    first.register(slack)
    second.register(slack)
    second.register(graph)

    assert first.list_plugins() == second.list_plugins()
    assert first.list_source_kinds() == tuple(
        plugin.identity for plugin in first.list_plugins()
    )
    assert first.get(
        provider_id=slack.identity.provider_id,
        integration_category=slack.identity.integration_category,
        source_kind=slack.identity.source_kind,
    ) == slack
    assert first.get(
        provider_id="missing",
        integration_category=IntegrationCategory.ISSUE_TRACKER,
        source_kind="missing",
    ) is None
    with pytest.raises(VendorKnowledgeSourcePluginNotFound):
        first.supports(
            _identity(provider_id="missing", source_kind="missing"),
            VendorKnowledgeMode.DURABLE,
        )
    assert first.supports(graph.identity, VendorKnowledgeMode.INDEXED)


def test_duplicate_registration_is_idempotent_and_conflict_fails_closed() -> None:
    plugin = build_slack_vendor_knowledge_source_plugin()
    registry = VendorKnowledgeSourcePluginRegistry()
    registry.register(plugin)
    registry.register(build_slack_vendor_knowledge_source_plugin())
    assert registry.list_plugins() == (plugin,)

    conflicting = replace(plugin, metadata={"revision": "different"})
    with pytest.raises(VendorKnowledgeSourcePluginConflict):
        registry.register(conflicting)


def test_metadata_and_constraints_are_defensively_frozen() -> None:
    metadata = {"nested": {"items": ["original"]}}
    plugin = VendorKnowledgeSourcePlugin(
        identity=_identity(),
        capabilities=(_mode(VendorKnowledgeMode.DURABLE, constraints=metadata),),
        metadata=metadata,
    )
    metadata["nested"]["items"].append("leaked")

    assert plugin.metadata["nested"]["items"] == ("original",)
    assert plugin.capability(VendorKnowledgeMode.DURABLE) is not None
    constraints = plugin.capability(VendorKnowledgeMode.DURABLE).constraints  # type: ignore[union-attr]
    assert constraints["nested"]["items"] == ("original",)
    with pytest.raises(TypeError):
        plugin.metadata["new"] = "not allowed"  # type: ignore[index]
    with pytest.raises(TypeError):
        constraints["new"] = "not allowed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("identity_kwargs", "expected"),
    [
        ({"provider_id": "", "source_kind": "source"}, "provider_id_invalid"),
        ({"provider_id": "provider", "source_kind": ""}, "source_kind_invalid"),
    ],
)
def test_missing_source_identity_is_rejected(
    identity_kwargs: dict[str, str],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _identity(**identity_kwargs)


def test_invalid_mode_declarations_fail_before_registration() -> None:
    with pytest.raises(ValueError, match="mode_contract_version"):
        VendorKnowledgeModeCapability(
            mode=VendorKnowledgeMode.DURABLE,
            contract_version="",
            operations=("read",),
            runtime_ref="runtime:source",
        )
    with pytest.raises(ValueError, match="live_capability_reference_required"):
        _mode(VendorKnowledgeMode.LIVE)
    with pytest.raises(ValueError, match="source_mismatch"):
        VendorKnowledgeSourcePlugin(
            identity=_identity(provider_id="provider", source_kind="source"),
            capabilities=(
                _mode(
                    VendorKnowledgeMode.LIVE,
                    capability_refs=("vendor.other.other_source.list",),
                ),
            ),
        )


def test_duplicate_capability_identity_is_rejected() -> None:
    capability_ref = "vendor.provider.source.list"
    with pytest.raises(ValueError, match="duplicate_capability_identity"):
        VendorKnowledgeSourcePlugin(
            identity=_identity(),
            capabilities=(
                _mode(
                    VendorKnowledgeMode.DURABLE,
                    capability_refs=(capability_ref,),
                ),
                _mode(
                    VendorKnowledgeMode.LIVE,
                    capability_refs=(capability_ref,),
                ),
            ),
        )
