# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.governance.legal_agent_governance_wiring import (
    with_dual_legal_governance,
)
from intergrax.agents_packages.legal_agent.governance.legal_execution_policy_sources import (
    ChainedLegalExecutionPolicy,
    RequestMetadataLegalExecutionPolicy,
    TenantRegistryLegalExecutionPolicy,
)
from intergrax.agents_packages.legal_agent.governance.legal_platform_policy_governance import (
    LegalNexusLayerCaps,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

from testing_support.builder import (
    DummyExecutionGuard,
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_state_for_tests,
)

pytestmark = pytest.mark.unit


def _cfg() -> LegalAgentConfig:
    return LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )


def _state(*, tenant_id: str | None, metadata: dict | None = None) -> RuntimeState:
    st = build_runtime_state_for_tests(run_id="policy-src")
    st.request = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        tenant_id=tenant_id,
        metadata=dict(metadata or {}),
    )
    return st


def test_tenant_registry_resolves_known_and_default() -> None:
    pol = TenantRegistryLegalExecutionPolicy(
        by_tenant={
            "t-a": LegalNexusLayerCaps(allow_rag=False, allow_websearch=True, allow_tools=True),
        },
        default_caps=LegalNexusLayerCaps(),
    )
    cfg = _cfg()
    c1 = pol.resolve_nexus_layer_caps(state=_state(tenant_id="t-a"), legal_config=cfg)
    assert c1.allow_rag is False
    c2 = pol.resolve_nexus_layer_caps(state=_state(tenant_id="unknown"), legal_config=cfg)
    assert c2.allow_rag is True
    c3 = pol.resolve_nexus_layer_caps(state=_state(tenant_id=None), legal_config=cfg)
    assert c3.allow_rag is True


def test_tenant_registry_strict_unknown_raises() -> None:
    pol = TenantRegistryLegalExecutionPolicy(
        by_tenant={"t-a": LegalNexusLayerCaps()},
        strict_tenant=True,
    )
    with pytest.raises(KeyError):
        pol.resolve_nexus_layer_caps(state=_state(tenant_id="missing"), legal_config=_cfg())


def test_request_metadata_partial_keys() -> None:
    pol = RequestMetadataLegalExecutionPolicy()
    cfg = _cfg()
    st = _state(
        tenant_id="t",
        metadata={"legal_nexus_layer_caps": {"allow_rag": False}},
    )
    caps = pol.resolve_nexus_layer_caps(state=st, legal_config=cfg)
    assert caps.allow_rag is False
    assert caps.allow_websearch is True
    assert caps.allow_tools is True


def test_chained_and_caps() -> None:
    tenant = TenantRegistryLegalExecutionPolicy(
        by_tenant={"t": LegalNexusLayerCaps(allow_rag=True, allow_websearch=True, allow_tools=True)},
    )
    meta = RequestMetadataLegalExecutionPolicy()
    chain = ChainedLegalExecutionPolicy(policies=(tenant, meta))
    cfg = _cfg()
    st = _state(
        tenant_id="t",
        metadata={"legal_nexus_layer_caps": {"allow_rag": False}},
    )
    caps = chain.resolve_nexus_layer_caps(state=st, legal_config=cfg)
    assert caps.allow_rag is False


def test_chained_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChainedLegalExecutionPolicy(policies=())


def test_with_dual_legal_governance_same_instance() -> None:
    guard = DummyExecutionGuard()
    policy = TenantRegistryLegalExecutionPolicy(by_tenant={})
    base = _cfg()
    out = with_dual_legal_governance(base, guard=guard, policy=policy)
    assert out.governance_service is out.legal_tool_plan_governance
    assert out.governance_service is not None
