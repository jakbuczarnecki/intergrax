# © Artur Czarnecki. All rights reserved.

"""ME-14 — Marketplace → Tool domain → catalog execution reference E2E."""

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
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernanceReasonCode,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryDisposition,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.marketplace.handoff_traceability.errors import MarketplaceHandoffSelectionError
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_OUTPUT_V1,
    ME14_OUTPUT_V2,
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

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _global_discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


class _DenyAllGovernanceEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "me14.deny_all"

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


def test_me14_marketplace_tool_execution_happy_path(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    evidence = stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path)
    envelope_release = stack.handoff_consumer.last_envelope
    assert envelope_release is not None
    assert evidence.selected_release == envelope_release.selected_release
    assert evidence.execution_result == ME14_OUTPUT_V1
    assert evidence.execution_tool_id == ME14_TOOL_LOGICAL_ID
    assert stack.lifecycle.registry_read().has(ME14_TOOL_LOGICAL_ID)


def test_me14_exact_tool_release_is_preserved_to_execution(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(me14_default_listing_v1(),),
    )
    evidence = stack.run_marketplace_tool_e2e(
        discovery_correlation_id="discovery-corr-exact",
        selection_id="selection-exact-v1",
        handoff_id="handoff-exact-v1",
        execution_tmp_path=tmp_path,
    )
    assert evidence.selected_release.version_label == ME14_VERSION_V1
    assert evidence.selected_release.content_digest == ME14_DIGEST_V1
    assert evidence.execution_result == ME14_OUTPUT_V1
    assert evidence.activated_version_label == ME14_VERSION_V1


def test_me14_exact_tool_release_v2_when_listing_points_v2(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(me14_listing_v2(),),
    )
    evidence = stack.run_marketplace_tool_e2e(
        handoff_id="handoff-exact-v2",
        execution_tmp_path=tmp_path,
    )
    assert evidence.selected_release.version_label == ME14_VERSION_V2
    assert evidence.execution_result == ME14_OUTPUT_V2


def test_me14_tenant_private_foreign_tenant_cannot_handoff() -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(
            _marketplace_listing_record(
                version_label=ME14_VERSION_V1,
                content_digest=ME14_DIGEST_V1,
                tenant_id="tenant-a-me14",
            ),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_global_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b-me14"),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            selector_id="operator.me14.foreign",
            discovery_correlation_id="discovery-corr-foreign",
            selection_id="selection-foreign",
            handoff_id="handoff-foreign",
        )


def test_me14_governance_denied_blocks_handoff() -> None:
    stack = MarketplaceToolExecutionProofStack.build()
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
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            selector_id="operator.me14.gov",
            discovery_correlation_id="discovery-corr-gov",
            selection_id="selection-gov",
            handoff_id="handoff-gov-deny",
        )


def test_me14_trust_denied_blocks_activation(tmp_path: Path) -> None:
    pytest.skip("canonical Tool trust authority not available — reference-only trust_allowed fixture")


def test_me14_duplicate_handoff_id_does_not_repeat_tool_lifecycle(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    kwargs = {
        "discovery_correlation_id": "discovery-corr-dup",
        "selection_id": "selection-dup",
        "handoff_id": "handoff-dup",
        "execution_tmp_path": tmp_path,
    }
    stack.run_marketplace_tool_e2e(**kwargs)
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]
    envelope = stack.handoff_consumer.last_envelope
    assert envelope is not None
    result = stack.delivery_service.deliver(envelope)
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]


def test_me14_execution_uses_tool_from_domain_registry(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    assert not stack.lifecycle.registry_read().has(ME14_TOOL_LOGICAL_ID)
    stack.run_marketplace_tool_e2e(
        handoff_id="handoff-registry-read",
        execution_tmp_path=tmp_path,
    )
    registry = stack.lifecycle.registry_read()
    assert registry.has(ME14_TOOL_LOGICAL_ID)
    registered = registry.get(ME14_TOOL_LOGICAL_ID)
    assert registered.contract.tool_id == ME14_TOOL_LOGICAL_ID


def test_me14_composition_has_no_nexus_imports() -> None:
    module = importlib.import_module(
        "testing_support.marketplace_tool_execution_composition",
    )
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


def test_me14_marketplace_core_has_no_tool_runtime_impl_imports() -> None:
    root = Path(__file__).resolve().parents[3] / "intergrax" / "marketplace"
    forbidden = (
        "intergrax.tools.registry.runtime",
        "intergrax.runtime.nexus.tools.invoker",
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
