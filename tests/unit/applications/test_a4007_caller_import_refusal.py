# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-I — A4-007 seam b1: the product callers refuse unregistered serialized imports.

Negative tests NT-A7C-01..08 and controls CTL-A7C-01..03, 05, 06 (plan section 8). Every
negative test records ``importlib.import_module`` and asserts (a) the refusal and (b) that the
synthetic module was never imported. Before the change the callers pass no allowlist, so the
import is attempted (and fails as ``ModuleNotFoundError``): the recorded-import assertion is what
turns red. This file imports only modules that exist before the change.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

import pytest
from intergrax.applications._shared import environment_diff_wiring
from intergrax.applications._shared.environment_conformance import EnvironmentSkillToolConsistencyCheck
from intergrax.applications._shared.import_allowlist import ImportAllowlistError, ImportPathNotAllowedError
from intergrax.applications._shared.package_wiring import collect_application_dependencies
from intergrax.applications._shared.roster_agent_contract_authority import (
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.applications._shared.wiring import build_agent_from_binding, invoke_canonical_agent_factory
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

from echo.echo_agent import EchoAgent
from legal.legal_agent import LegalAgent

pytestmark = [pytest.mark.unit]

_SYNTH_PREFIX = "zz_synth"
_SYNTH_AGENT_PATH = "zz_synth_t2.agent.Agent"
_SYNTH_AGENT_MODULE = "zz_synth_t2.agent"
_SYNTH_FACTORY_PATH = "zz_synth_t2.factory.build"


def _record_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``importlib.import_module`` call; real names delegate, synthetic names do not exist."""
    calls: list[str] = []
    real_import = importlib.import_module

    def _recording(name: str, package: str | None = None) -> ModuleType:
        calls.append(name)
        if name.startswith(_SYNTH_PREFIX):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _recording)
    return calls


def _synthetic(calls: list[str]) -> list[str]:
    return [name for name in calls if name.startswith(_SYNTH_PREFIX)]


def _product_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults()
    assert env.application_profile is ApplicationProfile.PRODUCT
    return env


def _lab_strict_env() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.lab_defaults()
    env = base.model_copy(update={"meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})})
    assert env.application_profile is ApplicationProfile.LAB
    assert env.execution_mode is ExecutionMode.STRICT
    return env


def _product_balanced_env() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.product_defaults()
    env = base.model_copy(update={"meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.BALANCED})})
    assert env.application_profile is ApplicationProfile.PRODUCT
    assert env.execution_mode is ExecutionMode.BALANCED
    return env


def _lab_manifest(binding: AgentBinding) -> ApplicationManifest:
    return ApplicationManifest.lab(app_id="a7c_lab", name="A7C lab", agents=[binding])


def _product_manifest(binding: AgentBinding) -> ApplicationManifest:
    manifest = ApplicationManifest.product(
        app_id="a7c_product",
        name="A7C product",
        agents=[binding],
        route_prefix="/v1/a7c",
        env_prefix="A7C_",
    )
    assert manifest.profile is ApplicationProfile.PRODUCT
    return manifest


def _serialized_synth_binding() -> AgentBinding:
    return AgentBinding.deserialize(import_path=_SYNTH_AGENT_PATH, contract_id="zz_synth")


def _build_echo_agent(ctx: ApplicationBuildContext, binding: AgentBinding) -> EchoAgent:
    _ = ctx, binding
    return EchoAgent()


# --- negative tests (b1) -------------------------------------------------------------------


def test_nt_a7c_01_build_agent_product_context_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _serialized_synth_binding()
    ctx = ApplicationBuildContext.for_manifest(_lab_manifest(binding), environment=_product_env())
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        build_agent_from_binding(binding, ctx)
    assert _synthetic(calls) == []


def test_nt_a7c_02_build_agent_strict_only_context_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _serialized_synth_binding()
    ctx = ApplicationBuildContext.for_manifest(_lab_manifest(binding), environment=_lab_strict_env())
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        build_agent_from_binding(binding, ctx)
    assert _synthetic(calls) == []


def test_nt_a7c_03_build_agent_product_context_refuses_unregistered_factory_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = AgentBinding(contract_id="zz_synth_factory", factory_path=_SYNTH_FACTORY_PATH)
    ctx = ApplicationBuildContext.for_manifest(_lab_manifest(binding), environment=_product_env())
    calls = _record_imports(monkeypatch)

    with pytest.raises((AgentImportError, ImportAllowlistError)) as exc_info:
        build_agent_from_binding(binding, ctx)
    assert _synthetic(calls) == []
    assert isinstance(exc_info.value, ImportPathNotAllowedError)


def test_nt_a7c_04_resolve_builder_with_empty_builders_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _serialized_synth_binding()
    ctx = ApplicationBuildContext.for_manifest(_product_manifest(binding))
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        build_agent_from_binding(binding, ctx, builders={})
    assert _synthetic(calls) == []


def test_nt_a7c_05_factory_result_validation_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = AgentBinding(factory=_build_echo_agent, import_path=_SYNTH_AGENT_PATH, contract_id="zz_synth")
    ctx = ApplicationBuildContext.for_manifest(_product_manifest(binding))
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        build_agent_from_binding(binding, ctx)
    assert _synthetic(calls) == []


def test_nt_a7c_06_environment_conformance_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _serialized_synth_binding()
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        EnvironmentSkillToolConsistencyCheck().validate_binding(binding, _product_env())
    assert _synthetic(calls) == []


def test_nt_a7c_07_manifest_lab_compat_authority_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _product_manifest(_serialized_synth_binding())
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        materialize_manifest_contract_authority_lab_compat(manifest)
    assert _synthetic(calls) == []


def test_nt_a7c_08_package_dependencies_product_balanced_refuses_unregistered_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _lab_manifest(_serialized_synth_binding())
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        collect_application_dependencies(manifest, _product_balanced_env())
    assert _synthetic(calls) == []


# --- controls --------------------------------------------------------------------------------


def test_ctl_a7c_01_lab_context_loads_serialized_unregistered_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = AgentBinding.deserialize(import_path="echo.echo_agent.EchoAgent")
    ctx = ApplicationBuildContext.for_manifest(_lab_manifest(binding))
    calls = _record_imports(monkeypatch)

    agent = build_agent_from_binding(binding, ctx)
    assert isinstance(agent, EchoAgent)
    assert "echo.echo_agent" in calls


def test_ctl_a7c_02_product_context_resolves_registered_serialized_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = AgentBinding.deserialize(import_path="legal.legal_agent.LegalAgent", contract_id="legal")
    manifest = _product_manifest(binding)
    calls = _record_imports(monkeypatch)

    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    assert authority.contracts_by_id["legal"].id == "legal"
    assert "legal.legal_agent" in calls

    class _ReachedFactoryError(Exception):
        pass

    def _reached(ctx: ApplicationBuildContext, binding: AgentBinding) -> LegalAgent:
        _ = ctx, binding
        raise _ReachedFactoryError

    ctx = ApplicationBuildContext.for_manifest(manifest, environment=_product_env())
    with pytest.raises(_ReachedFactoryError):
        build_agent_from_binding(binding, ctx, builders={LegalAgent: _reached})


def test_ctl_a7c_03_diff_roster_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    left = [
        AgentBinding.mount(EchoAgent, contract_id="echo"),
        AgentBinding(agent_type=EchoAgent),
        AgentBinding.deserialize(import_path=_SYNTH_AGENT_PATH),
        AgentBinding.reference("ref_only"),
    ]
    right = [AgentBinding.deserialize(import_path="zz_synth_t3.other.OtherAgent", contract_id="other")]
    calls = _record_imports(monkeypatch)

    changes = environment_diff_wiring.diff_roster(left, right)
    assert len(changes) == len(left) + len(right)
    assert calls == []


def test_ctl_a7c_05_operator_added_binding_under_product_context_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = AgentBinding(contract_id="operator_added", factory_path="zz_synth_operator.factory.build_agent")
    assert binding.import_path is None
    assert binding.agent_type is None
    ctx = ApplicationBuildContext.for_manifest(_lab_manifest(binding), environment=_product_env())
    calls = _record_imports(monkeypatch)

    resolved_factory: Callable[[ApplicationBuildContext, AgentBinding], EchoAgent] = _build_echo_agent
    agent = invoke_canonical_agent_factory(resolved_factory, ctx, binding)
    assert isinstance(agent, EchoAgent)
    assert calls == []


def test_ctl_a7c_06_lab_manifest_context_stays_unrestricted_for_unregistered_serialized_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = AgentBinding.deserialize(import_path="echo.echo_agent.EchoAgent", contract_id="echo")
    manifest = _lab_manifest(binding)
    ctx = ApplicationBuildContext.for_manifest(manifest)
    assert ctx.environment is None
    calls = _record_imports(monkeypatch)

    agent = invoke_canonical_agent_factory(_build_echo_agent, ctx, binding)
    assert isinstance(agent, EchoAgent)
    assert "echo.echo_agent" in calls
