# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R2 — canonical agent factory contract closure gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path
import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.wiring import (
    build_agent_from_binding,
    invoke_canonical_agent_factory,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory, CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FACTORY_CONTRACT = _REPO_ROOT / "intergrax/applications/contracts/factory.py"
_MANIFEST_CONTRACT = _REPO_ROOT / "intergrax/applications/contracts/manifest.py"
_WIRING = _REPO_ROOT / "intergrax/applications/_shared/wiring.py"

_BROAD_FACTORY_RE = re.compile(
    r"AgentFactory\s*=\s*Callable\[\.\.\.",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_agent_factory_is_canonical_alias() -> None:
    assert AgentFactory is CanonicalAgentFactory


def test_factory_contract_has_no_broad_callable_alias() -> None:
    source = _read(_FACTORY_CONTRACT)
    assert not _BROAD_FACTORY_RE.search(source)
    assert "Callable[..., Tier2Agent]" not in source


def test_agent_binding_factory_field_uses_canonical_type() -> None:
    tree = ast.parse(_read(_MANIFEST_CONTRACT))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "factory":
            continue
        assert node.annotation is not None
        ann = ast.unparse(node.annotation)
        assert "ApplicationBuildContext" in ann
        assert "AgentBinding" in ann
        assert "Callable[..., " not in ann
        return
    pytest.fail("AgentBinding.factory annotation not found")


def test_resolve_builder_returns_canonical_factory_type() -> None:
    source = _read(_WIRING)
    match = re.search(
        r"def resolve_builder\([^)]*\)\s*->\s*([^\n:]+)",
        source,
    )
    assert match is not None
    assert "CanonicalAgentFactory" in match.group(1)


def test_build_agent_from_binding_uses_canonical_invoker_for_resolved_factory() -> None:
    source = _read(_WIRING)
    start = source.index("def build_agent_from_binding(")
    end = source.index("\ndef contract_for_binding", start)
    block = source[start:end]
    assert "invoke_canonical_agent_factory" in block
    assert "invoke_agent_factory" not in block
    assert "invoke_legacy_compatible_agent_factory" not in block.split(
        "if binding.factory_path"
    )[0]


def test_canonical_factory_invoked_exactly_once() -> None:
    calls = 0

    def factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        nonlocal calls
        calls += 1
        return EchoAgent()

    manifest = ApplicationManifest.lab(
        app_id="r2",
        name="R2",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.agents[0]
    build_agent_from_binding(binding, ctx)
    assert calls == 1


def test_canonical_factory_typeerror_is_not_signature_probed() -> None:
    calls = 0

    def factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        nonlocal calls
        calls += 1
        raise TypeError("business error")

    manifest = ApplicationManifest.lab(
        app_id="r2_te",
        name="R2",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.agents[0]

    with pytest.raises(TypeError, match="business error"):
        invoke_canonical_agent_factory(factory, ctx, binding)
    assert calls == 1

    with pytest.raises(TypeError, match="business error"):
        build_agent_from_binding(binding, ctx)
    assert calls == 2


def test_serialized_factory_path_branch_uses_legacy_compatibility_invoker() -> None:
    source = _read(_WIRING)
    start = source.index("if binding.factory_path is not None")
    end = source.index(
        "\n    from intergrax.applications._shared.agent_resolution",
        start,
    )
    block = source[start:end]
    assert "invoke_legacy_compatible_agent_factory" in block
    assert "invoke_canonical_agent_factory" not in block


def test_external_structural_canonical_factory_without_inheritance() -> None:
    class ExternalFactory:
        def __call__(
            self,
            ctx: ApplicationBuildContext,
            binding: AgentBinding,
        ) -> EchoAgent:
            _ = ctx, binding
            return EchoAgent()

    external: CanonicalAgentFactory = ExternalFactory()
    manifest = ApplicationManifest.lab(
        app_id="r2_ext",
        name="R2",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=external)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    agent = build_agent_from_binding(manifest.agents[0], ctx)
    assert isinstance(agent, EchoAgent)
