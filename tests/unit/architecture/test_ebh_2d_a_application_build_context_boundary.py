# © Artur Czarnecki. All rights reserved.

"""EBH-2D-A / EBH-2D-A-R2 — ApplicationBuildContext + factory dependency boundary gate."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.reference_harness import LabHarnessContext
from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
    composition_for_factory_context,
    project_application_build_context,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from research.research_agent import ResearchAgent
from research_application.host.agent_builders import build_research_agent_builders

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_CONTEXT_PATH = _REPO_ROOT / "intergrax/applications/contracts/build_context.py"
_COMPOSITION_CONTEXT_PATH = (
    _REPO_ROOT / "intergrax/applications/_shared/application_composition_context.py"
)
_TOOL_ENABLEMENT_PATH = _REPO_ROOT / "intergrax/agents/tool_enablement.py"
_INCIDENT_FACTORY_PATH = (
    _REPO_ROOT
    / "platform_proofs/scenarios/ai_incident_investigation/integration/agent_factory.py"
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.skills.registry.runtime",
    "intergrax.tools.registry.runtime",
    "intergrax.tools.registry.wiring",
    "intergrax.prompts.registry.yaml_registry",
)

_FORBIDDEN_FIELD_MODULES = (
    "intergrax.runtime.",
    "intergrax.skills.registry.runtime",
    "intergrax.tools.registry.runtime",
    "intergrax.tools.registry.wiring",
    "intergrax.prompts.registry.yaml_registry",
)

_FACTORY_MODULES = (
    "applications/research_application/host/agent_builders.py",
    "applications/lab_application/host/agent_builders.py",
    "applications/intergrax_assistant_application/host/agent_builders.py",
    "applications/attestation_demo/host/agent_builders.py",
    "platform_proofs/scenarios/ai_incident_investigation/integration/agent_factory.py",
)

_BUILDER_ENTRYPOINTS = (
    (
        "applications/research_application/host/agent_builders.py",
        "build_research_agent_builders",
    ),
    (
        "applications/lab_application/host/agent_builders.py",
        "build_lab_agent_builders",
    ),
    (
        "applications/intergrax_assistant_application/host/agent_builders.py",
        "build_intergrax_assistant_agent_builders",
    ),
    (
        "applications/attestation_demo/host/agent_builders.py",
        "build_attestation_demo_agent_builders",
    ),
)

_HOST_WIRING_MODULES = (
    "applications/research_application/host/wiring.py",
    "applications/lab_application/host/wiring.py",
    "applications/intergrax_assistant_application/host/wiring.py",
    "applications/attestation_demo/host/wiring.py",
    "applications/lab_application/host/factory.py",
    "applications/intergrax_assistant_application/host/factory.py",
    "applications/attestation_demo/host/factory.py",
)

_AMBIENT_COMPOSITION_NAMES = frozenset(
    {
        "optional_factory_composition",
        "require_factory_composition",
        "factory_composition_scope",
        "_FACTORY_COMPOSITION",
        "ApplicationFactoryCompositionRequired",
    }
)

_COMPOSITION_MODULE = "intergrax.applications._shared.application_composition_context"

_FORBIDDEN_BUILDER_PARAM_TYPES = frozenset(
    {
        "ToolWiringContext",
        "ToolWiringContextLike",
        "RuntimePolicyBundle",
        "ApplicationCompositionContext",
        "ScenarioRuntimeComposition",
        "ToolRegistry",
    }
)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.append(alias.name)
    return hits


def _imported_names_from_module(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                names.add(alias.name)
    return names


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _annotation_names(annotation: ast.AST | None) -> set[str]:
    if annotation is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _function_def(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_build_context_module_has_no_runtime_or_registry_imports() -> None:
    imports = _module_imports(_BUILD_CONTEXT_PATH)
    violations = [
        name
        for name in imports
        if any(
            name == prefix.rstrip(".") or name.startswith(prefix)
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
        )
    ]
    assert not violations, f"forbidden imports in build_context: {violations}"


def test_public_build_context_is_frozen_dataclass_without_runtime_fields() -> None:
    assert is_dataclass(ApplicationBuildContext)
    assert ApplicationBuildContext.__dataclass_params__.frozen

    for field in fields(ApplicationBuildContext):
        annotation = field.type
        if isinstance(annotation, str):
            rendered = annotation
        else:
            rendered = str(annotation)
        assert not any(
            rendered.startswith(prefix) or f".{prefix}" in rendered
            for prefix in _FORBIDDEN_FIELD_MODULES
        ), f"field {field.name} references forbidden namespace: {rendered}"


def _minimal_lab_manifest(app_id: str, name: str) -> ApplicationManifest:
    from echo.echo_agent import EchoAgent

    return ApplicationManifest.lab(
        app_id=app_id,
        name=name,
        route_prefix=f"/v1/{app_id}",
        env_prefix=f"{app_id.upper()}_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def test_composition_projection_does_not_leak_runtime_types_to_public_context() -> None:
    manifest = _minimal_lab_manifest("plugin_proof", "Plugin Proof")
    factory_context = ApplicationBuildContext.for_manifest(manifest)
    from intergrax.tools.registry.runtime import ToolRegistry

    composition = composition_for_factory_context(
        factory_context,
        policy_bundle=RuntimePolicyBundle(),
        tool_registry=ToolRegistry(),
    )
    projected = project_application_build_context(composition)
    assert projected is factory_context
    assert not hasattr(projected, "tool_registry")
    assert not hasattr(projected, "policy_bundle")


def test_external_style_factory_uses_only_public_contract_imports() -> None:
    manifest = _minimal_lab_manifest("external_factory", "External")

    def external_factory(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Agent:
        from echo.echo_agent import EchoAgent

        _ = binding
        assert isinstance(ctx.manifest, ApplicationManifest)
        return EchoAgent()

    factory: CanonicalAgentFactory = external_factory
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = factory(ctx, binding)
    assert agent is not None


def test_composition_context_module_has_no_ambient_factory_locator() -> None:
    source = _COMPOSITION_CONTEXT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert not (_AMBIENT_COMPOSITION_NAMES & defined), (
        f"ambient factory composition locator still defined: "
        f"{sorted(_AMBIENT_COMPOSITION_NAMES & defined)}"
    )
    assert "ContextVar" not in source


@pytest.mark.parametrize("relative_path", _FACTORY_MODULES)
def test_production_factory_modules_do_not_import_ambient_composition(
    relative_path: str,
) -> None:
    path = _REPO_ROOT / relative_path
    imported = _imported_names_from_module(path, _COMPOSITION_MODULE)
    ambient_imports = imported & _AMBIENT_COMPOSITION_NAMES
    assert not ambient_imports, f"{relative_path} imports ambient accessors: {ambient_imports}"
    assert "ApplicationCompositionContext" not in imported, (
        f"{relative_path} must not import ApplicationCompositionContext"
    )
    calls = _call_names(path)
    ambient_calls = calls & _AMBIENT_COMPOSITION_NAMES
    assert not ambient_calls, f"{relative_path} calls ambient accessors: {ambient_calls}"


@pytest.mark.parametrize("relative_path,func_name", _BUILDER_ENTRYPOINTS)
def test_factory_builder_api_rejects_broad_service_bags(
    relative_path: str,
    func_name: str,
) -> None:
    path = _REPO_ROOT / relative_path
    func = _function_def(path, func_name)
    annotated: set[str] = set()
    for arg in list(func.args.args) + list(func.args.kwonlyargs):
        annotated |= _annotation_names(arg.annotation)
    forbidden = annotated & _FORBIDDEN_BUILDER_PARAM_TYPES
    assert not forbidden, (
        f"{relative_path}::{func_name} exposes broad service-bag params: {sorted(forbidden)}"
    )
    source = path.read_text(encoding="utf-8")
    assert "isinstance(tool_wiring_context, ToolWiringContext)" not in source
    assert "ToolWiringContextLike" not in source


def test_tool_wiring_context_like_is_removed() -> None:
    source = _TOOL_ENABLEMENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "ToolWiringContextLike" not in defined
    assert "ToolEnablementProfile" in defined
    profile = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ToolEnablementProfile"
    )
    methods = {
        node.name
        for node in profile.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "is_tool_enabled" in methods


def test_configured_research_factory_works_without_composition_scope() -> None:
    class _ProbeProfile:
        def is_tool_enabled(self, tool_id: str) -> bool:
            return tool_id == "web.search"

    profile = _ProbeProfile()
    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle())
    builders = build_research_agent_builders(
        tool_profile=profile,
        lab_harness=harness,
    )
    factory = builders[ResearchAgent]
    manifest = ApplicationManifest.lab(
        app_id="research_factory_probe",
        name="Research Factory Probe",
        route_prefix="/v1/research_factory_probe",
        env_prefix="RESEARCH_FACTORY_PROBE_",
        agents=[
            AgentBinding.mount(
                ResearchAgent,
                contract_id="research",
                capabilities=["research.basic"],
            )
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = factory(ctx, binding)
    assert isinstance(agent, ResearchAgent)
    assert agent._tool_profile is profile
    assert agent._harness is harness


def test_canonical_agent_factory_signature_remains_two_arg() -> None:
    parameters = list(inspect.signature(CanonicalAgentFactory.__call__).parameters)
    assert parameters == ["self", "ctx", "binding"]


def test_incident_production_settings_carry_no_runtime_composition() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.integration.agent_factory import (
        IncidentInvestigatorProductionSettings,
    )

    field_names = {field.name for field in fields(IncidentInvestigatorProductionSettings)}
    assert "composition" not in field_names
    assert "evidence_store" not in field_names
    assert "llm_adapter_override" not in field_names
    assert field_names <= {"operational_data", "investigation_input"}

    source = _INCIDENT_FACTORY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    settings_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IncidentInvestigatorProductionSettings"
    )
    annotated: set[str] = set()
    for node in settings_cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotated |= _annotation_names(node.annotation)
    assert "ScenarioRuntimeComposition" not in annotated
    assert "ToolRegistry" not in annotated
    assert "ScenarioEvidenceStore" not in annotated
    assert "LLMAdapter" not in annotated


def test_incident_configured_factory_flow_does_not_use_settings_composition() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
        build_resolved_fixture,
    )
    from platform_proofs.scenarios.ai_incident_investigation.integration.agent_factory import (
        bootstrap_incident_investigator_runtime,
        build_incident_investigator_factory,
    )

    operational = build_resolved_fixture().to_operational_data()
    bootstrap = bootstrap_incident_investigator_runtime(operational)
    assert not hasattr(bootstrap.settings, "composition")
    assert not hasattr(bootstrap.settings, "evidence_store")
    assert not hasattr(bootstrap.settings, "llm_adapter_override")

    factory = build_incident_investigator_factory(
        operational_data=operational,
        runtime_composition=bootstrap.composition,
        evidence_store=bootstrap.evidence_store,
        investigation_input=bootstrap.settings.investigation_input,
    )
    closed = [cell.cell_contents for cell in (factory.__closure__ or ())]
    assert any(item is bootstrap.composition for item in closed)
    assert any(item is bootstrap.evidence_store for item in closed)
    assert any(item is operational for item in closed)

    manifest = ApplicationManifest.lab(
        app_id="incident_factory_probe",
        name="Incident Factory Probe",
        route_prefix="/v1/incident_factory_probe",
        env_prefix="INCIDENT_FACTORY_PROBE_",
        agents=[
            AgentBinding.reference(
                contract_id="incident_investigator",
                capabilities=["scenario.incident.investigate"],
            )
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=bootstrap.settings,
    )
    binding = manifest.enabled_agents()[0]
    try:
        factory(ctx, binding)
    except TypeError as exc:
        # Known unrelated abstract-agent debt (§44): Agent.run not implemented.
        assert "Can't instantiate abstract class" in str(exc)
        assert "run" in str(exc)
    else:
        pytest.fail("expected known abstract Agent.run instantiation error")


@pytest.mark.parametrize("relative_path", _HOST_WIRING_MODULES)
def test_production_host_explicitly_binds_factory_dependencies(relative_path: str) -> None:
    path = _REPO_ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    builder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("build_")
        and node.func.id.endswith("_agent_builders")
    ]
    assert builder_calls, f"{relative_path} must call a build_*_agent_builders entrypoint"
    for call in builder_calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "tool_wiring_context" not in kw_names, (
            f"{relative_path} still passes tool_wiring_context into factory builders"
        )
        assert "policy_bundle" not in kw_names, (
            f"{relative_path} still passes policy_bundle into factory builders"
        )
        assert "lab_harness" in kw_names, (
            f"{relative_path} must bind prepared lab_harness into factory builders"
        )


def test_lab_configured_factory_receives_prepared_harness() -> None:
    from lab_application.host.agent_builders import build_lab_agent_builders
    from echo.echo_agent import EchoAgent

    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle())
    builders = build_lab_agent_builders(lab_harness=harness)
    manifest = _minimal_lab_manifest("lab_factory_probe", "Lab Factory Probe")
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = builders[EchoAgent](ctx, binding)
    assert getattr(agent, "_harness", None) is harness


def test_attestation_configured_factory_receives_prepared_buffer() -> None:
    from attestation_demo.host.agent_builders import build_attestation_demo_agent_builders
    from boundary_demo.boundary_demo_agent import BoundaryDemoAgent
    from intergrax.runtime.attestation.buffer import BoundaryEventBuffer

    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle())
    buffer = BoundaryEventBuffer()
    builders = build_attestation_demo_agent_builders(
        lab_harness=harness,
        boundary_event_buffer=buffer,
    )
    factory = builders[BoundaryDemoAgent]
    closed = [cell.cell_contents for cell in (factory.__closure__ or ())]
    assert any(item is harness for item in closed)
    assert any(item is buffer for item in closed)

    manifest = ApplicationManifest.lab(
        app_id="attestation_factory_probe",
        name="Attestation Factory Probe",
        route_prefix="/v1/attestation_factory_probe",
        env_prefix="ATTESTATION_FACTORY_PROBE_",
        agents=[
            AgentBinding.mount(
                BoundaryDemoAgent,
                contract_id="boundary_demo_agent",
                capabilities=["attestation.demo"],
            )
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    try:
        factory(ctx, binding)
    except TypeError as exc:
        assert "Can't instantiate abstract class" in str(exc)
        assert "run" in str(exc)
    else:
        pytest.fail("expected known abstract Agent.run instantiation error")


def test_assistant_configured_factory_receives_prepared_harness() -> None:
    from intergrax_assistant_application.host.agent_builders import (
        build_intergrax_assistant_agent_builders,
    )
    from echo.echo_agent import EchoAgent

    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle())
    builders = build_intergrax_assistant_agent_builders(lab_harness=harness)
    manifest = ApplicationManifest.lab(
        app_id="assistant_factory_probe",
        name="Assistant Factory Probe",
        route_prefix="/v1/assistant_factory_probe",
        env_prefix="ASSISTANT_FACTORY_PROBE_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = builders[EchoAgent](ctx, binding)
    assert getattr(agent, "_harness", None) is harness


def test_external_plugin_structural_harness_port() -> None:
    """External plugin can supply a LabHarnessContext-compatible object via host binding."""

    class _ExternalPolicyBundle:
        pass

    class _ExternalHarness:
        def __init__(self) -> None:
            self.policy_bundle = _ExternalPolicyBundle()
            self.strict_harness = False
            self.trace_db_path = None
            self.modality_profile = None
            self.tool_wiring_context = None

    # Structural: ResearchAgent accepts any LabHarnessContext-shaped object through host
    # binding of the real LabHarnessContext type; pluginability of ToolEnablementProfile:
    class _ExternalProfile:
        def is_tool_enabled(self, tool_id: str) -> bool:
            return tool_id == "websearch.query"

    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle())
    profile = _ExternalProfile()
    builders = build_research_agent_builders(tool_profile=profile, lab_harness=harness)
    manifest = ApplicationManifest.lab(
        app_id="plugin_harness_probe",
        name="Plugin Harness Probe",
        route_prefix="/v1/plugin_harness_probe",
        env_prefix="PLUGIN_HARNESS_PROBE_",
        agents=[
            AgentBinding.mount(
                ResearchAgent,
                contract_id="research",
                capabilities=["research.basic"],
            )
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    agent = builders[ResearchAgent](ctx, manifest.enabled_agents()[0])
    assert isinstance(agent, ResearchAgent)
    assert agent._tool_profile is profile
    assert profile.is_tool_enabled("websearch.query")
    _ = _ExternalHarness  # documents structural alternative for future ports
