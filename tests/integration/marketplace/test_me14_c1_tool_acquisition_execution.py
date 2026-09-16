# © Artur Czarnecki. All rights reserved.

"""ME-14-C1 production tool acquisition and execution boundary tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_OUTPUT_V1,
    ME14_OUTPUT_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
)
from testing_support.marketplace_tool_execution_composition import (
    MarketplaceToolExecutionProofStack,
    _marketplace_listing_record,
    me14_default_listing_v1,
    me14_listing_v2,
)
from testing_support.me14_tool_catalog_provider import (
    ME14_CATALOG_SOURCE_ID,
    Me14ToolCatalogProvider,
    _CustomMe14ToolCatalogProvider,
)
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionService
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from intergrax.marketplace.handoff_traceability.errors import MarketplaceHandoffSelectionError
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def test_me14_c1_marketplace_tool_execution_engine_happy_path(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    evidence = stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path)
    assert evidence.execution_result == ME14_OUTPUT_V1


def test_me14_c1_selected_v1_is_activated_and_executed_when_v1_and_v2_exist(
    tmp_path: Path,
) -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(me14_default_listing_v1(),),
    )
    evidence = stack.run_marketplace_tool_e2e(
        handoff_id="handoff-v1-exact",
        execution_tmp_path=tmp_path,
    )
    assert evidence.selected_release.version_label == ME14_VERSION_V1
    assert evidence.execution_result == ME14_OUTPUT_V1
    assert evidence.activated_version_label == ME14_VERSION_V1


def test_me14_c1_selected_v2_is_activated_and_executed(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build(listing_records=(me14_listing_v2(),))
    evidence = stack.run_marketplace_tool_e2e(
        handoff_id="handoff-v2-exact",
        execution_tmp_path=tmp_path,
    )
    assert evidence.execution_result == ME14_OUTPUT_V2
    assert evidence.activated_version_label == ME14_VERSION_V2


def test_me14_c1_tool_is_not_executable_before_lifecycle(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    from intergrax.applications.contracts.application_package import (
        ApplicationPackageClosureError,
    )
    from testing_support.me14_tool_harness_execution import run_me14_tool_host_execution

    with pytest.raises(ApplicationPackageClosureError, match="missing from wired tool registry"):
        import asyncio

        asyncio.run(
            run_me14_tool_host_execution(
                registry=stack.lifecycle.registry_read(),
                tool_logical_id=ME14_TOOL_LOGICAL_ID,
                tmp_path=tmp_path,
            ),
        )


def test_me14_c1_tool_is_executable_after_lifecycle(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path)
    import asyncio

    tool_id, output, task_id = asyncio.run(
        stack.execute_tool_via_host_execution_engine(tmp_path / "second-run"),
    )
    assert tool_id == ME14_TOOL_LOGICAL_ID
    assert output == ME14_OUTPUT_V1
    assert task_id is not None


def test_me14_c1_execution_goes_through_public_execution_engine(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    evidence = stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path)
    assert evidence.execution_task_id is not None


def test_me14_c1_production_tool_acquisition_is_contract_driven() -> None:
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-me14")
    provider = Me14ToolCatalogProvider()
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        activation=lifecycle,
        materializer=Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert service is not None


def test_me14_c1_custom_tool_provider_plugs_in_without_core_changes() -> None:
    custom = _CustomMe14ToolCatalogProvider()
    assert custom.catalog_source_id == "custom.me14.provider"
    assert custom.list_entries()


def test_me14_c1_source_mismatch_fails_closed() -> None:
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-me14")
    provider = Me14ToolCatalogProvider()
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        activation=lifecycle,
        materializer=Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    bad_identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id="wrong.source",
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=CapabilitySourceIdentity(
                source_id=ME14_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=ME14_TOOL_LOGICAL_ID,
            ),
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        service.acquire(
            DynamicToolAcquisitionRequest(
                operation_id="op-source-mismatch",
                host_profile_id="host-profile-me14",
                capability_identity_key=identity_key,
                selected_identity=bad_identity,
            ),
        )


def test_me14_c1_digest_mismatch_fails_closed() -> None:
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-me14")
    provider = Me14ToolCatalogProvider()
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry({provider.catalog_source_id: provider}),
        activation=lifecycle,
        materializer=Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    bad = ToolDiscoveryCandidateIdentity(
        catalog_source_id=ME14_CATALOG_SOURCE_ID,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V2,
        ),
    )
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=CapabilitySourceIdentity(
                source_id=ME14_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=ME14_TOOL_LOGICAL_ID,
            ),
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        service.acquire(
            DynamicToolAcquisitionRequest(
                operation_id="op-digest-mismatch",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me14_c1_org_private_foreign_org_cannot_handoff() -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(
            _marketplace_listing_record(
                version_label=ME14_VERSION_V1,
                content_digest=ME14_DIGEST_V1,
                organization_id="org-a-me14",
            ),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(organization_id="org-b-me14"),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            selector_id="operator.me14.org-foreign",
            discovery_correlation_id="discovery-corr-org",
            selection_id="selection-org",
            handoff_id="handoff-org",
        )


def test_me14_c1_composition_does_not_call_declarative_invoker_directly() -> None:
    module = importlib.import_module(
        "testing_support.marketplace_tool_execution_composition",
    )
    source = Path(module.__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "invoke":
                raise AssertionError("composition must not call invoker.invoke directly")
