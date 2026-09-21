# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R3 — factory results align with Tier2Agent (no concrete Agent gate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.wiring import (
    build_agent_from_binding,
    build_manifest_development_registry,
    invoke_canonical_agent_factory,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.tier2_agent import Tier2Agent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WIRING = _REPO_ROOT / "intergrax/applications/_shared/wiring.py"


class ExternalTier2Agent:
    """Structural Tier-2 agent — not a subclass of framework ``Agent``."""

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only agent")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="external_structural",
            name="External",
            description="R3 structural proof agent",
            capabilities=["external.run"],
        )


class ExternalFactory:
    def __call__(
        self,
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Tier2Agent:
        _ = ctx, binding
        return ExternalTier2Agent()


def test_canonical_path_has_no_isinstance_agent_on_factory_result() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    validate_block = source.split("def _validate_factory_result", 1)[1].split(
        "\ndef invoke_canonical_agent_factory", 1
    )[0]
    assert "isinstance(result, Agent)" not in validate_block


def test_external_structural_agent_via_canonical_factory() -> None:
    external: CanonicalAgentFactory = ExternalFactory()
    binding = AgentBinding(contract_id="external_structural", factory=external)
    manifest = ApplicationManifest.lab(
        app_id="r3_ext",
        name="R3",
        agents=[binding],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    agent = build_agent_from_binding(binding, ctx)
    assert isinstance(agent, Tier2Agent)
    assert not isinstance(agent, EchoAgent)
    assert type(agent) is ExternalTier2Agent


def test_external_agent_registers_in_development_registry() -> None:
    external: CanonicalAgentFactory = ExternalFactory()
    binding = AgentBinding(contract_id="external_structural", factory=external)
    manifest = ApplicationManifest.lab(
        app_id="r3_reg",
        name="R3",
        agents=[binding],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_manifest_development_registry(manifest, ctx)
    assert registry.has("external_structural")


def test_invalid_factory_result_rejected() -> None:
    def bad_factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> object:
        return object()

    manifest = ApplicationManifest.lab(
        app_id="r3_bad",
        name="R3",
        agents=[AgentBinding(contract_id="bad", factory=bad_factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.agents[0]

    with pytest.raises(AgentImportError, match="Tier2Agent"):
        invoke_canonical_agent_factory(bad_factory, ctx, binding)


def test_framework_echo_agent_still_materializes() -> None:
    def factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        return EchoAgent()

    manifest = ApplicationManifest.lab(
        app_id="r3_echo",
        name="R3",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    agent = build_agent_from_binding(manifest.agents[0], ctx)
    assert isinstance(agent, EchoAgent)
    assert isinstance(agent, Tier2Agent)


def test_binding_agent_type_conformance_still_enforced() -> None:
    external: CanonicalAgentFactory = ExternalFactory()

    manifest = ApplicationManifest.lab(
        app_id="r3_type",
        name="R3",
        agents=[
            AgentBinding.mount(EchoAgent, contract_id="echo", factory=external),
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)

    with pytest.raises(AgentImportError, match="expected instance"):
        build_agent_from_binding(manifest.agents[0], ctx)
