# © Artur Czarnecki. All rights reserved.

"""ME-13 — Marketplace → Agent Distribution → Execution reference E2E."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.trust import (
    AgentPackageTrustDecision,
    AgentPackageTrustPolicy,
)
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
from testing_support.canonical_agent_lifecycle_composition import (
    catalog_package_resolution_for_config,
)
from testing_support.canonical_lifecycle_ping_agent import (
    CANONICAL_PING_CONTRACT_ID,
    CANONICAL_PING_OUTPUT,
)
from testing_support.marketplace_agent_distribution_execution_composition import (
    MarketplaceAgentDistributionExecutionProofStack,
    _marketplace_listing_record,
    me13_lifecycle_proof_config,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter
    from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

    install_diagnostic_cursor_secret(monkeypatch)

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def _global_discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


class _DenyAllGovernanceEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "me13.deny_all"

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


def test_me13_marketplace_agent_distribution_execution_happy_path(
    tmp_path: Path,
) -> None:
    stack = MarketplaceAgentDistributionExecutionProofStack.build(tmp_path)
    evidence = stack.run_marketplace_agent_e2e()
    config = stack.lifecycle_stack.config
    assert evidence.execution_answer == CANONICAL_PING_OUTPUT
    assert evidence.execution_agent_id == CANONICAL_PING_CONTRACT_ID
    assert evidence.selected_release.version_label == config.package_version
    assert evidence.selected_release.content_digest == config.package_digest
    assert evidence.selected_release.package_reference == config.distribution_package_id
    roster = stack.lifecycle_stack.admin.inspect_effective_roster(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    entry = next(
        item for item in roster.entries if item.logical_agent_id == config.logical_agent_id
    )
    assert entry.distribution_package_id == config.distribution_package_id
    assert entry.package_digest == config.package_digest


def test_me13_exact_release_survives_catalog_version_drift(tmp_path: Path) -> None:
    config = me13_lifecycle_proof_config()
    stack = MarketplaceAgentDistributionExecutionProofStack.build(
        tmp_path,
        lifecycle_config=config,
        listing_records=(_marketplace_listing_record(config, version_label="1.0.0"),),
    )
    stack.run_marketplace_agent_e2e(
        discovery_correlation_id="discovery-corr-drift",
        selection_id="selection-drift",
        handoff_id="handoff-drift-v1",
    )
    catalog_entry = stack.lifecycle_stack.discover_catalog_entry()
    v2_digest = "sha256:" + ("c" * 64)
    stack.lifecycle_stack.catalog_provider.register_resolution(
        catalog_package_resolution_for_config(
            config=config,
            entry=catalog_entry,
            package_version="2.0.0",
            package_digest=v2_digest,
        ),
    )
    roster = stack.lifecycle_stack.admin.inspect_effective_roster(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    entry = next(
        item for item in roster.entries if item.logical_agent_id == config.logical_agent_id
    )
    assert entry.distribution_package_id == config.distribution_package_id
    assert entry.package_digest == config.package_digest


def test_me13_tenant_private_foreign_tenant_cannot_handoff(tmp_path: Path) -> None:
    config = me13_lifecycle_proof_config()
    stack = MarketplaceAgentDistributionExecutionProofStack.build(
        tmp_path,
        lifecycle_config=config,
        listing_records=(
            _marketplace_listing_record(config, tenant_id="tenant-a-me13"),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_global_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b-me13"),
            selected_identity_key=stack.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
            selector_id="operator.me13.foreign",
            discovery_correlation_id="discovery-corr-foreign",
            selection_id="selection-foreign",
            handoff_id="handoff-foreign",
        )


def test_me13_governance_denied_blocks_handoff(tmp_path: Path) -> None:
    stack = MarketplaceAgentDistributionExecutionProofStack.build(tmp_path)
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
            consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
            selector_id="operator.me13.gov",
            discovery_correlation_id="discovery-corr-gov",
            selection_id="selection-gov",
            handoff_id="handoff-gov-deny",
        )


def test_me13_duplicate_handoff_id_skips_second_lifecycle(tmp_path: Path) -> None:
    stack = MarketplaceAgentDistributionExecutionProofStack.build(tmp_path)
    kwargs = {
        "discovery_correlation_id": "discovery-corr-dup",
        "selection_id": "selection-dup",
        "handoff_id": "handoff-dup",
    }
    stack.run_marketplace_agent_e2e(**kwargs)
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]
    envelope = stack.handoff_consumer.last_envelope
    assert envelope is not None
    result = stack.delivery_service.deliver(envelope)
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert stack.handoff_consumer._delivered_handoffs == ["handoff-dup"]


def test_me13_trust_denial_blocks_activation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = me13_lifecycle_proof_config()
    original_init = AgentPlatformAdminService.__init__

    def _init_with_deny_trust(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["package_trust_policy_source"] = lambda: AgentPackageTrustPolicy(
            default_decision=AgentPackageTrustDecision.DENY,
        )
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(AgentPlatformAdminService, "__init__", _init_with_deny_trust)
    stack = MarketplaceAgentDistributionExecutionProofStack.build(
        tmp_path,
        lifecycle_config=config,
    )
    with pytest.raises(CapabilityHandoffConsumerError):
        stack.run_marketplace_agent_e2e(handoff_id="handoff-trust-deny")


def test_me13_composition_has_no_nexus_imports() -> None:
    module = importlib.import_module(
        "testing_support.marketplace_agent_distribution_execution_composition",
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
