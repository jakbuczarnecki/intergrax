# © Artur Czarnecki. All rights reserved.

"""ME-16 — Marketplace → Agent + Tool + Skill → mixed execution E2E."""

from __future__ import annotations

import ast
import asyncio
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
    CapabilityKind,
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
from intergrax.skills.errors import DynamicSkillAcquisitionResolutionError
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_OUTPUT_V1,
    ME14_OUTPUT_V2,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
)
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    ME15_VERSION_V2,
    instruction_marker_for_release,
)
from testing_support.canonical_me16_mixed_agent import (
    ME16_MARKETPLACE_LOGICAL_ID,
    ME16_MIXED_CONTRACT_ID,
    ME16_MIXED_TENANT,
    expected_mixed_output_for_releases,
)
from testing_support.marketplace_mixed_capability_execution_composition import (
    MarketplaceMixedCapabilityProofStack,
    MixedCapabilityCompositionNotReadyError,
    me16_agent_listing_v1,
)
from testing_support.marketplace_skill_composition import (
    _marketplace_listing_record as skill_listing_record,
    me15_default_listing_v1,
    me15_listing_v2,
)
from testing_support.marketplace_tool_execution_composition import (
    _marketplace_listing_record as tool_listing_record,
    me14_default_listing_v1,
    me14_listing_v2,
)
from testing_support.me15_skill_catalog_provider import (
    Me15SkillCatalogProvider,
    _CustomMe15SkillCatalogProvider,
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
        return "me16.deny_all"

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


def test_me16_agent_tool_skill_mixed_capability_happy_path(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    evidence = stack.run_marketplace_mixed_e2e(execution_tmp_path=tmp_path / "exec")
    assert evidence.execution_agent_id == ME16_MIXED_CONTRACT_ID
    assert "|" in evidence.execution_answer
    assert evidence.agent_envelope.handoff_id == evidence.agent_handoff_id
    assert evidence.tool_envelope.handoff_id == evidence.tool_handoff_id
    assert evidence.skill_envelope.handoff_id == evidence.skill_handoff_id


def test_me16_exact_agent_tool_skill_releases_reach_runtime_composition(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    evidence = stack.run_marketplace_mixed_e2e(
        execution_tmp_path=tmp_path / "exec",
        agent_handoff_id="handoff-exact-agent",
        tool_handoff_id="handoff-exact-tool",
        skill_handoff_id="handoff-exact-skill",
    )
    config = stack.lifecycle_config
    assert evidence.agent_selected_release.version_label == config.package_version
    assert evidence.tool_selected_release.version_label == ME14_VERSION_V1
    assert evidence.skill_selected_release.version_label == ME15_VERSION_V1
    assert evidence.tool_activation_version == ME14_VERSION_V1
    assert evidence.skill_bound_version == ME15_VERSION_V1


def test_me16_execution_is_blocked_until_all_required_capabilities_are_ready(
    tmp_path: Path,
) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    assert not stack.readiness().execution_allowed
    with pytest.raises(MixedCapabilityCompositionNotReadyError):
        asyncio.run(stack.try_execute_when_not_ready(tmp_path / "exec-fail"))


def _deny_orchestrator(stack: MarketplaceMixedCapabilityProofStack) -> object:
    from intergrax.marketplace.handoff_traceability import MarketplaceDiscoveryHandoffOrchestrator

    return MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=stack.catalog_service,
        discovery_service=stack.orchestrator.discovery_service,
        governance_evaluators=(_DenyAllGovernanceEvaluator(),),
        governance_context=stack.orchestrator.governance_context,
        delivery_service=stack.orchestrator.delivery_service,
    )


def test_me16_final_output_depends_on_selected_tool(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        tool_listing_records=(me14_listing_v2(),),
    )
    stack.handoff_agent(
        discovery_correlation_id="c-tool",
        selection_id="s-agent",
        handoff_id="h-agent-tool-v2",
    )
    stack.handoff_tool(
        discovery_correlation_id="c-tool",
        selection_id="s-tool",
        handoff_id="h-tool-v2",
    )
    stack.handoff_skill(
        discovery_correlation_id="c-tool",
        selection_id="s-skill",
        handoff_id="h-skill-tool-v2",
    )
    evidence_tool = stack.tool_lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID)
    assert evidence_tool is not None
    assert evidence_tool.version_label == ME14_VERSION_V2
    _, answer, _ = asyncio.run(
        __import__(
            "testing_support.me16_mixed_harness_execution",
            fromlist=["execute_me16_mixed_via_host_execution_engine"],
        ).execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "exec-tool-v2",
        ),
    )
    assert ME14_OUTPUT_V2 in answer


def test_me16_final_output_depends_on_bound_skill(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        skill_listing_records=(me15_listing_v2(),),
    )
    stack.run_all_handoffs(
        discovery_correlation_id="c-skill-v2",
        skill_handoff_id="h-skill-v2-only",
        tool_handoff_id="h-tool-skill-v2",
        agent_handoff_id="h-agent-skill-v2",
    )
    _, answer, _ = asyncio.run(
        __import__(
            "testing_support.me16_mixed_harness_execution",
            fromlist=["execute_me16_mixed_via_host_execution_engine"],
        ).execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "exec-skill-v2",
        ),
    )
    marker = instruction_marker_for_release(stack.mixed_consumer.skill.last_envelope.selected_release)
    assert marker in answer


def test_me16_execution_uses_selected_agent(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    evidence = stack.run_marketplace_mixed_e2e(execution_tmp_path=tmp_path / "exec-agent")
    assert evidence.execution_agent_id == ME16_MIXED_CONTRACT_ID


def test_me16_governance_denied_blocks_agent_handoff(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    object.__setattr__(stack, "orchestrator", _deny_orchestrator(stack))
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.handoff_agent(
            discovery_correlation_id="gov-agent",
            selection_id="sel-gov-agent",
            handoff_id="h-gov-agent",
        )


def test_me16_governance_denied_blocks_tool_handoff(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    object.__setattr__(stack, "orchestrator", _deny_orchestrator(stack))
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.handoff_tool(
            discovery_correlation_id="gov-tool",
            selection_id="sel-gov-tool",
            handoff_id="h-gov-tool",
        )


def test_me16_governance_denied_blocks_skill_handoff(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    object.__setattr__(stack, "orchestrator", _deny_orchestrator(stack))
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.handoff_skill(
            discovery_correlation_id="gov-skill",
            selection_id="sel-gov-skill",
            handoff_id="h-gov-skill",
        )


def test_me16_tenant_mixed_isolation(tmp_path: Path) -> None:
    config = MarketplaceMixedCapabilityProofStack.build(tmp_path).lifecycle_config
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        agent_listing_records=(me16_agent_listing_v1(config, tenant_id="tenant-a-me16"),),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.handoff_agent(
            discovery_correlation_id="tenant-mix",
            selection_id="sel-tenant",
            handoff_id="h-tenant",
            tenant_id="tenant-b-me16",
        )


def test_me16_custom_tool_provider_pluginability(tmp_path: Path) -> None:
    from testing_support.me14_tool_catalog_provider import _CustomMe14ToolCatalogProvider

    custom = _CustomMe14ToolCatalogProvider()
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        tool_catalog_provider=custom,
    )
    assert custom.catalog_source_id in stack.tool_acquisition._catalog_registry.registered_source_ids


def test_me16_composition_has_no_nexus_imports() -> None:
    module = importlib.import_module(
        "testing_support.marketplace_mixed_capability_execution_composition",
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


def test_me16_no_universal_mixed_engine_classes() -> None:
    module = importlib.import_module(
        "testing_support.marketplace_mixed_capability_execution_composition",
    )
    forbidden_names = {
        "MixedCapabilityLifecycleService",
        "UniversalCapabilityRuntime",
        "CapabilityGodRegistry",
        "UniversalCapabilityExecutor",
        "MixedMarketplaceEngine",
        "MixedCapabilityEngine",
        "UniversalAcquisitionEngine",
    }
    for name in dir(module):
        if name in forbidden_names:
            raise AssertionError(f"forbidden mixed engine class {name}")

