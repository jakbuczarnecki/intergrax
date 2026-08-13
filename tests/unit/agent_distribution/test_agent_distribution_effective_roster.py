# © Artur Czarnecki. All rights reserved.

"""AP-6 effective roster merge builder tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution._immutable_json import freeze_distribution_json_object
from intergrax.agent_distribution.binding import (
    AgentBindingFactoryReference,
    AgentBindingPolicyOverrides,
    ApplicationAgentBinding,
)
from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.errors import EffectiveRosterConflict
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.roster import ManifestDefaultAgentDeclaration
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_APP_ID = "app-local-workspace"
_ENV_ID = "env-prod"
_RELEASE_ID = "rel-2026-08-13"
_PACKAGE_A = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
)
_PACKAGE_B = _PACKAGE_A.model_copy(
    update={"package_version": "2.0.0", "package_digest": _DIGEST_B}
)


def _trust_record(digest: str = _DIGEST_A) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _builder() -> tuple[EffectiveRosterBuilder, AgentDistributionStoreState]:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    return EffectiveRosterBuilder(installation_store), state


def _persist_active_installation(
    state: AgentDistributionStoreState,
    *,
    installation_id: str,
    slot_id: str,
    package_identity: AgentPackageIdentity = _PACKAGE_A,
) -> None:
    state.installations[installation_id] = AgentInstallationRecord(
        installation_id=installation_id,
        installation_slot_id=slot_id,
        environment_id=_ENV_ID,
        package_identity=package_identity,
        installation_state=InstallationState.INSTALLED_ACTIVE,
        active_for_slot=True,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=_trust_record(package_identity.package_digest),
    )
    state.active_installation_by_slot[slot_id] = installation_id


def _manifest_search(
    *, enabled: bool = True, digest: str = _DIGEST_A
) -> ManifestDefaultAgentDeclaration:
    return ManifestDefaultAgentDeclaration(
        logical_agent_id="search",
        manifest_origin_ref="manifest:agents/search",
        installation_slot_id="slot-search-prod",
        distribution_package_id="intergrax-local-search-agent",
        package_digest=digest,
        enabled=enabled,
        config={"timeout_seconds": 30, "nested": {"mode": "fast"}},
        secret_refs=("secret://search-api-key",),
        policy_overrides=AgentBindingPolicyOverrides(tool_allowlist=("search",)),
        factory_reference=AgentBindingFactoryReference(builder_key="search-default"),
        builtin_package_ref="builtin:intergrax-local-search-agent",
    )


def _binding_search(
    *, enabled: bool = True, revision: int = 1
) -> ApplicationAgentBinding:
    return ApplicationAgentBinding(
        application_binding_id="bind-search-prod",
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        active_installation_id="inst-v1",
        enablement=enabled,
        config={"timeout_seconds": 45, "nested": {"retries": 2}},
        secret_refs=("secret://search-api-key", "secret://search-index"),
        policy_overrides=AgentBindingPolicyOverrides(tool_denylist=("delete",)),
        factory_reference=AgentBindingFactoryReference(
            factory_path="agents.search:build"
        ),
        manifest_origin_ref="manifest:agents/search",
        binding_revision=revision,
    )


def test_manifest_only_builtin_produces_deterministic_entry() -> None:
    builder, _ = _builder()
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(),
    )
    assert len(roster.entries) == 1
    entry = roster.entries[0]
    assert entry.logical_agent_id == "search"
    assert entry.package_digest == _DIGEST_A
    assert entry.effective_enablement is True
    assert entry.application_binding_id is None
    assert (
        roster.effective_roster_revision_id
        == roster.with_revision_id().effective_roster_revision_id
    )


def test_durable_binding_overlays_manifest_defaults() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(_binding_search(),),
    )
    entry = roster.entries[0]
    assert entry.effective_enablement is True
    assert entry.application_binding_id == "bind-search-prod"
    assert entry.active_installation_id == "inst-v1"
    assert entry.package_digest == _DIGEST_A
    assert entry.secret_refs == ("secret://search-api-key", "secret://search-index")
    assert entry.policy_overrides == AgentBindingPolicyOverrides(
        tool_denylist=("delete",)
    )
    assert entry.factory_reference == AgentBindingFactoryReference(
        factory_path="agents.search:build"
    )
    merged = entry.merged_config
    assert merged["timeout_seconds"] == 45
    assert merged["nested"] == freeze_distribution_json_object(
        {"mode": "fast", "retries": 2}
    )


def test_enabled_binding_resolves_exact_active_installation_digest() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state,
        installation_id="inst-v2",
        slot_id="slot-search-prod",
        package_identity=_PACKAGE_B,
    )
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(digest=_DIGEST_A),),
        durable_bindings=(_binding_search(),),
    )
    entry = roster.entries[0]
    assert entry.active_installation_id == "inst-v2"
    assert entry.package_digest == _DIGEST_B


def test_disabled_binding_produces_effective_enablement_false() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(enabled=True),),
        durable_bindings=(_binding_search(enabled=False),),
    )
    assert roster.entries[0].effective_enablement is False


def test_installation_upgrade_preserves_binding_identity_and_config() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    binding = _binding_search()
    roster_a = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(binding,),
    )
    _persist_active_installation(
        state,
        installation_id="inst-v2",
        slot_id="slot-search-prod",
        package_identity=_PACKAGE_B,
    )
    state.installations["inst-v1"] = state.installations["inst-v1"].model_copy(
        update={
            "installation_state": InstallationState.INSTALLED_PREVIOUS,
            "active_for_slot": False,
        }
    )
    state.active_installation_by_slot["slot-search-prod"] = "inst-v2"
    roster_b = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(binding,),
    )
    entry_a = roster_a.entries[0]
    entry_b = roster_b.entries[0]
    assert (
        entry_a.application_binding_id
        == entry_b.application_binding_id
        == "bind-search-prod"
    )
    assert entry_a.logical_agent_id == entry_b.logical_agent_id == "search"
    assert entry_a.merged_config == entry_b.merged_config
    assert entry_a.secret_refs == entry_b.secret_refs
    assert entry_a.package_digest == _DIGEST_A
    assert entry_b.package_digest == _DIGEST_B


def test_tombstoned_binding_excludes_manifest_default_from_effective_roster() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    tombstone = _binding_search().model_copy(
        update={"tombstone": True, "enablement": False}
    )
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(tombstone,),
    )
    assert roster.entries == ()


def test_missing_active_installation_for_enabled_binding_fails_closed() -> None:
    builder, _ = _builder()
    with pytest.raises(EffectiveRosterConflict):
        builder.build(
            application_id=_APP_ID,
            application_environment_id=_ENV_ID,
            manifest_release_id=_RELEASE_ID,
            manifest_defaults=(),
            durable_bindings=(_binding_search(enabled=True),),
        )


def test_duplicate_logical_agent_identity_fails_closed() -> None:
    builder, _ = _builder()
    duplicate = _binding_search().model_copy(
        update={"application_binding_id": "bind-search-copy"}
    )
    with pytest.raises(EffectiveRosterConflict):
        builder.build(
            application_id=_APP_ID,
            application_environment_id=_ENV_ID,
            manifest_release_id=_RELEASE_ID,
            manifest_defaults=(),
            durable_bindings=(_binding_search(), duplicate),
        )


def test_equivalent_inputs_with_different_ordering_share_revision_id() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    manifest = _manifest_search()
    binding_a = _binding_search(revision=1)
    binding_b = ApplicationAgentBinding(
        application_binding_id="bind-audit-prod",
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        logical_agent_id="audit",
        installation_slot_id="slot-audit-prod",
        enablement=False,
        config={},
        manifest_origin_ref="manifest:agents/audit",
        binding_revision=3,
        builtin_package_ref="builtin:audit",
    )
    audit_manifest = ManifestDefaultAgentDeclaration(
        logical_agent_id="audit",
        installation_slot_id="slot-audit-prod",
        distribution_package_id="intergrax-local-audit-agent",
        package_digest=_DIGEST_B,
        enabled=False,
        builtin_package_ref="builtin:audit",
    )
    roster_a = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(manifest, audit_manifest),
        durable_bindings=(binding_a, binding_b),
    )
    roster_b = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(audit_manifest, manifest),
        durable_bindings=(binding_b, binding_a),
    )
    assert (
        roster_a.effective_roster_revision_id == roster_b.effective_roster_revision_id
    )


def test_deep_immutable_merged_config_preserved() -> None:
    builder, state = _builder()
    _persist_active_installation(
        state, installation_id="inst-v1", slot_id="slot-search-prod"
    )
    roster = builder.build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(_manifest_search(),),
        durable_bindings=(_binding_search(),),
    )
    merged = roster.entries[0].merged_config
    with pytest.raises(TypeError):
        merged["timeout_seconds"] = 99  # type: ignore[index]
