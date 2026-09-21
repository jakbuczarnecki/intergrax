# © Artur Czarnecki. All rights reserved.

"""ME-15 — Marketplace → Skill domain → composition/binding reference E2E."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    CapabilityGovernanceDecision,
    RankedCapabilityCandidate,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernanceReasonCode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryDisposition,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.marketplace.handoff_traceability.errors import MarketplaceHandoffSelectionError
from intergrax.skills.catalog import SkillCatalogProviderRegistry
from intergrax.skills.dynamic_acquisition import (
    DynamicSkillAcquisitionRequest,
    DynamicSkillAcquisitionService,
)
from intergrax.skills.errors import DynamicSkillAcquisitionResolutionError
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
from intergrax.skills.registry.profile import is_skill_enabled
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_PACKAGE_REFERENCE_V1,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    ME15_VERSION_V2,
)
from testing_support.marketplace_skill_composition import (
    MarketplaceSkillCompositionProofStack,
    _marketplace_listing_record,
    me15_default_listing_v1,
)
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import (
    ME15_CATALOG_SOURCE_ID,
    Me15SkillCatalogProvider,
    _CustomMe15SkillCatalogProvider,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _global_discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


class _DenyAllGovernanceEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "me15.deny_all"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.BLOCKED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.BLOCKED,
                reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
            ),
        )


def test_me15_marketplace_skill_composition_happy_path() -> None:
    stack = MarketplaceSkillCompositionProofStack.build()
    evidence = stack.run_marketplace_skill_e2e()
    envelope = stack.handoff_consumer.last_envelope
    assert envelope is not None
    assert evidence.selected_release == envelope.selected_release
    assert ME15_SKILL_LOGICAL_ID in evidence.profile_skill_ids
    assert stack.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_me15_selected_skill_release_is_preserved_into_binding() -> None:
    stack = MarketplaceSkillCompositionProofStack.build(
        listing_records=(me15_default_listing_v1(),),
    )
    evidence = stack.run_marketplace_skill_e2e(
        discovery_correlation_id="discovery-corr-exact",
        selection_id="selection-exact-v1",
        handoff_id="handoff-exact-v1",
    )
    assert evidence.selected_release.version_label == ME15_VERSION_V1
    assert evidence.selected_release.content_digest == ME15_DIGEST_V1
    assert evidence.bound_version_label == ME15_VERSION_V1
    assert evidence.bound_content_digest == ME15_DIGEST_V1
    manifest = stack.lifecycle.registry.get(ME15_SKILL_LOGICAL_ID).manifest
    assert manifest.version == ME15_VERSION_V1


def test_me15_selected_v1_is_bound_when_v1_and_v2_are_available() -> None:
    stack = MarketplaceSkillCompositionProofStack.build(
        listing_records=(me15_default_listing_v1(),),
    )
    evidence = stack.run_marketplace_skill_e2e(handoff_id="handoff-v1-exact")
    assert evidence.selected_release.version_label == ME15_VERSION_V1
    assert evidence.bound_version_label == ME15_VERSION_V1


def test_me15_tenant_private_foreign_tenant_cannot_handoff() -> None:
    stack = MarketplaceSkillCompositionProofStack.build(
        listing_records=(
            _marketplace_listing_record(
                version_label=ME15_VERSION_V1,
                content_digest=ME15_DIGEST_V1,
                tenant_id="tenant-a-me15",
            ),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_global_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b-me15"),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
            selector_id="operator.me15.foreign",
            discovery_correlation_id="discovery-corr-foreign",
            selection_id="selection-foreign",
            handoff_id="handoff-foreign",
        )


def test_me15_org_private_foreign_org_cannot_handoff() -> None:
    stack = MarketplaceSkillCompositionProofStack.build(
        listing_records=(
            _marketplace_listing_record(
                version_label=ME15_VERSION_V1,
                content_digest=ME15_DIGEST_V1,
                organization_id="org-a-me15",
            ),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_global_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(organization_id="org-b-me15"),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
            selector_id="operator.me15.org-foreign",
            discovery_correlation_id="discovery-corr-org",
            selection_id="selection-org",
            handoff_id="handoff-org",
        )


def test_me15_governance_denied_blocks_skill_handoff() -> None:
    stack = MarketplaceSkillCompositionProofStack.build()
    from intergrax.marketplace.handoff_traceability import (
        MarketplaceDiscoveryHandoffOrchestrator,
    )

    deny_orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=stack.catalog_service,
        discovery_service=stack.orchestrator.discovery_service,
        governance_evaluators=(_DenyAllGovernanceEvaluator(),),
        governance_context=stack.orchestrator.governance_context,
        delivery_service=stack.orchestrator.delivery_service,
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        deny_orchestrator.execute_explicit_selection_handoff(
            discovery_query=_global_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
            selector_id="operator.me15.gov",
            discovery_correlation_id="discovery-corr-gov",
            selection_id="selection-gov",
            handoff_id="handoff-gov-deny",
        )


def test_me15_skill_source_mismatch_fails_closed() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-profile-me15")
    provider = Me15SkillCatalogProvider()
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        binding=lifecycle,
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    bad_identity = SkillDiscoveryCandidateIdentity(
        catalog_source_id="wrong.source",
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=ME15_PACKAGE_REFERENCE_V1,
            package_version=ME15_VERSION_V1,
            package_digest=ME15_DIGEST_V1,
        ),
    )
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=ME15_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-source-mismatch",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad_identity,
            ),
        )


def test_me15_skill_version_mismatch_fails_closed() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-profile-me15")
    provider = Me15SkillCatalogProvider()
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=lifecycle,
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=ME15_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )
    bad = SkillDiscoveryCandidateIdentity(
        catalog_source_id=ME15_CATALOG_SOURCE_ID,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=ME15_PACKAGE_REFERENCE_V1,
            package_version=ME15_VERSION_V2,
            package_digest=ME15_DIGEST_V1,
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-version-mismatch",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me15_skill_digest_or_provenance_mismatch_fails_closed() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-profile-me15")
    provider = Me15SkillCatalogProvider()
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=lifecycle,
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=ME15_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )
    bad = SkillDiscoveryCandidateIdentity(
        catalog_source_id=ME15_CATALOG_SOURCE_ID,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=ME15_PACKAGE_REFERENCE_V1,
            package_version=ME15_VERSION_V1,
            package_digest=ME15_DIGEST_V2,
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-digest-mismatch",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me15_duplicate_handoff_does_not_repeat_skill_binding() -> None:
    stack = MarketplaceSkillCompositionProofStack.build()
    kwargs = {
        "discovery_correlation_id": "discovery-corr-dup",
        "selection_id": "selection-dup",
        "handoff_id": "handoff-dup",
    }
    stack.run_marketplace_skill_e2e(**kwargs)
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]
    envelope = stack.handoff_consumer.last_envelope
    assert envelope is not None
    result = stack.delivery_service.deliver(envelope)
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]


def test_me15_skill_absent_before_lifecycle_and_bound_after_handoff() -> None:
    stack = MarketplaceSkillCompositionProofStack.build()
    assert not stack.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    stack.run_marketplace_skill_e2e(handoff_id="handoff-before-after")
    assert stack.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert is_skill_enabled(stack.lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)


def test_me15_custom_skill_provider_plugs_in_without_core_changes() -> None:
    custom = _CustomMe15SkillCatalogProvider()
    assert custom.catalog_source_id == "custom.me15.provider"
    assert custom.list_entries()


def test_me15_marketplace_core_has_no_skill_domain_implementation_imports() -> None:
    root = Path(__file__).resolve().parents[3] / "intergrax" / "marketplace"
    forbidden = (
        "intergrax.skills.registry.runtime",
        "intergrax.skills.resolver",
        "intergrax.runtime.nexus",
    )
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        raise AssertionError(f"{path}: forbidden import {node.module}")


def test_me15_semantic_gate_no_skill_executor_symbols_in_me15_production_modules() -> None:
    root = Path(__file__).resolve().parents[3] / "intergrax" / "skills"
    patterns = ("SkillExecutor", "SkillInvoker", "skill.execute(")
    for path in root.rglob("*.py"):
        if path.name == "__pycache__":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern not in text, f"{path} contains forbidden {pattern!r}"


def test_me15_composition_has_no_nexus_imports() -> None:
    module = importlib.import_module("testing_support.marketplace_skill_composition")
    root = Path(module.__file__).resolve()
    forbidden = ("intergrax.runtime.nexus", "intergrax.nexus")
    tree = ast.parse(root.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden:
                    if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                        raise AssertionError(f"forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden:
                if node.module == prefix or node.module.startswith(f"{prefix}."):
                    raise AssertionError(f"forbidden import {node.module}")


def test_me15_bound_skill_composition_uses_canonical_execution_binding() -> None:
    from intergrax.contracts.execution_identity import ExecutionId
    from intergrax.skills.execution_binding import (
        InMemorySkillExecutionPinningStore,
        bind_resolved_skill_pack,
    )

    stack = MarketplaceSkillCompositionProofStack.build()
    stack.run_marketplace_skill_e2e(handoff_id="handoff-exec-binding")
    store = InMemorySkillExecutionPinningStore()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-me15",
        execution_id=ExecutionId("exec-me15-proof"),
        skill_profile=stack.lifecycle.skill_profile,
        skill_registry=stack.lifecycle.registry,
        pinning_store=store,
    )
    assert binding.resolved_pack.snapshot_digest
    assert ME15_SKILL_LOGICAL_ID in binding.resolved_pack.skill_ids
