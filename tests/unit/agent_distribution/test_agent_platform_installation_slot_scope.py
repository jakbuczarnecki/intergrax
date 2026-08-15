# © Artur Czarnecki. All rights reserved.

"""Environment-scoped installation slot coexistence tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.roster import ManifestDefaultAgentDeclaration
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app-a"
_ENV_DEV = "env-dev"
_ENV_PROD = "env-prod"
_SLOT = "slot-researcher"
_DIGEST_DEV = "sha256:" + ("a" * 64)
_DIGEST_PROD = "sha256:" + ("b" * 64)


def _trust(digest: str) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:slot-scope:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _manifest() -> ManifestDefaultAgentDeclaration:
    return ManifestDefaultAgentDeclaration(
        logical_agent_id="researcher",
        installation_slot_id=_SLOT,
        distribution_package_id="intergrax-local-search-agent",
        package_digest=_DIGEST_DEV,
        enabled=True,
        builtin_package_ref="builtin:intergrax-local-search-agent",
    )


def _install_active(
    installation_service: InstallationService,
    *,
    installation_id: str,
    environment_id: str,
    digest: str,
) -> None:
    installation_service.create_candidate_installation(
        installation_id=installation_id,
        installation_slot_id=_SLOT,
        environment_id=environment_id,
        package_identity=AgentPackageIdentity(
            distribution_package_id="intergrax-local-search-agent",
            package_version="1.0.0",
            package_digest=digest,
        ),
    )
    installation_service.mark_verified(
        installation_id,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=_trust(digest),
    )
    installation_service.promote_verified_to_active(installation_id)


def test_same_slot_name_coexists_across_environments_with_distinct_digests() -> None:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    installation_service = InstallationService(installation_store)
    roster_builder = EffectiveRosterBuilder(installation_store)

    _install_active(
        installation_service,
        installation_id="inst-prod",
        environment_id=_ENV_PROD,
        digest=_DIGEST_PROD,
    )
    _install_active(
        installation_service,
        installation_id="inst-dev",
        environment_id=_ENV_DEV,
        digest=_DIGEST_DEV,
    )

    prod_roster = roster_builder.build(
        application_id=_APP,
        application_environment_id=_ENV_PROD,
        manifest_release_id="rel-1",
        manifest_defaults=(_manifest(),),
        durable_bindings=(),
    )
    dev_roster = roster_builder.build(
        application_id=_APP,
        application_environment_id=_ENV_DEV,
        manifest_release_id="rel-1",
        manifest_defaults=(_manifest(),),
        durable_bindings=(),
    )

    assert prod_roster.entries[0].package_digest == _DIGEST_PROD
    assert dev_roster.entries[0].package_digest == _DIGEST_DEV


def test_dev_installation_promotion_does_not_change_prod_active_digest() -> None:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    installation_service = InstallationService(installation_store)
    roster_builder = EffectiveRosterBuilder(installation_store)

    _install_active(
        installation_service,
        installation_id="inst-prod-v1",
        environment_id=_ENV_PROD,
        digest=_DIGEST_PROD,
    )
    _install_active(
        installation_service,
        installation_id="inst-dev-v1",
        environment_id=_ENV_DEV,
        digest=_DIGEST_DEV,
    )

    installation_service.create_candidate_installation(
        installation_id="inst-dev-v2",
        installation_slot_id=_SLOT,
        environment_id=_ENV_DEV,
        package_identity=AgentPackageIdentity(
            distribution_package_id="intergrax-local-search-agent",
            package_version="2.0.0",
            package_digest="sha256:" + ("c" * 64),
        ),
    )
    installation_service.mark_verified(
        "inst-dev-v2",
        artifact_store_ref="store://artifacts/inst-dev-v2",
        trust_record=_trust("sha256:" + ("c" * 64)),
    )
    installation_service.promote_verified_to_active(
        "inst-dev-v2",
        expected_active_installation_id="inst-dev-v1",
    )

    prod_roster = roster_builder.build(
        application_id=_APP,
        application_environment_id=_ENV_PROD,
        manifest_release_id="rel-1",
        manifest_defaults=(_manifest(),),
        durable_bindings=(),
    )
    assert prod_roster.entries[0].package_digest == _DIGEST_PROD

    dev_active = installation_service.resolve_active_for_slot(_ENV_DEV, _SLOT)
    assert dev_active is not None
    assert dev_active.installation_id == "inst-dev-v2"

    prod_active = installation_service.resolve_active_for_slot(_ENV_PROD, _SLOT)
    assert prod_active is not None
    assert prod_active.installation_id == "inst-prod-v1"
