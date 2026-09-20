# © Artur Czarnecki. All rights reserved.

"""EBH-2D-B — Environment/Profile contract ownership gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.codecraft.profile import CodeCraftProfile as LegacyCodeCraftProfile
from intergrax.contracts.adaptive_loop_kind import AdaptiveLoopKind as CanonicalAdaptiveLoopKind
from intergrax.contracts.codecraft_profile import CodeCraftProfile as CanonicalCodeCraftProfile
from intergrax.contracts.compliance_domain_class import (
    ComplianceDomainClass as CanonicalComplianceDomainClass,
)
from intergrax.contracts.context_optimization_policy import (
    ContextOptimizationPolicy as CanonicalContextOptimizationPolicy,
)
from intergrax.contracts.event_taxonomy import EventCategory as CanonicalEventCategory
from intergrax.contracts.modality_profile import ModalityProfile as CanonicalModalityProfile
from intergrax.contracts.platform_plugin_selection import (
    PlatformPluginSelectionRef as CanonicalPlatformPluginSelectionRef,
)
from intergrax.contracts.policy_enforcement_mode import (
    PolicyEnforcementMode as CanonicalPolicyEnforcementMode,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType as CanonicalRuntimeEventType
from intergrax.contracts.scaling_policy import ScalingPolicy as CanonicalScalingPolicy
from intergrax.contracts.utility_weights import UtilityWeights as CanonicalUtilityWeights
from intergrax.core.plugins.selection_ref import (
    PlatformPluginSelectionRef as LegacyPlatformPluginSelectionRef,
)
from intergrax.integrations.contracts.integration_profile import (
    IntegrationProfile as CanonicalIntegrationProfile,
)
from intergrax.integrations.registry.profile import (
    IntegrationProfile as LegacyIntegrationProfile,
)
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile as CanonicalLLMProfile
from intergrax.llm_adapters.contracts.routing_profile import (
    LLMRoutingProfile as CanonicalLLMRoutingProfile,
)
from intergrax.llm_adapters.registry.profile import LLMProfile as LegacyLLMProfile
from intergrax.llm_adapters.routing.contracts import (
    LLMRoutingProfile as LegacyLLMRoutingProfile,
)
from intergrax.runtime.adaptive.contracts import UtilityWeights as LegacyUtilityWeights
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveLoopKind as LegacyAdaptiveLoopKind,
)
from intergrax.runtime.capacity.contracts import ScalingPolicy as LegacyScalingPolicy
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationPolicy as LegacyContextOptimizationPolicy,
)
from intergrax.runtime.events.event_taxonomy import EventCategory as LegacyEventCategory
from intergrax.runtime.events.runtime_event import RuntimeEventType as LegacyRuntimeEventType
from intergrax.runtime.modality.modality_profile import ModalityProfile as LegacyModalityProfile
from intergrax.runtime.policy.compliance_profiles import (
    ComplianceDomainClass as LegacyComplianceDomainClass,
)
from intergrax.runtime.policy.rules.evaluation import (
    PolicyEnforcementMode as LegacyPolicyEnforcementMode,
)
from intergrax.skills.contracts.skill_profile import SkillProfile as CanonicalSkillProfile
from intergrax.skills.registry.profile import SkillProfile as LegacySkillProfile
from intergrax.tools.contracts.tool_profile import ToolProfile as CanonicalToolProfile
from intergrax.tools.registry.profile import ToolProfile as LegacyToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_MODULES = (
    _REPO_ROOT / "intergrax/applications/contracts/environment_profile/root.py",
    _REPO_ROOT / "intergrax/applications/contracts/environment_profile/bundles.py",
    _REPO_ROOT / "intergrax/applications/contracts/environment_profile/sub_profiles.py",
)

_FORBIDDEN_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.applications._shared.",
)
_FORBIDDEN_SUBSTRINGS = (
    ".registry.",
)
_FORBIDDEN_EXACT = (
    "intergrax.codecraft.profile",
    "intergrax.llm_adapters.routing",
)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.",
        ):
            hits.append(node.module)
    return hits


def test_environment_profile_modules_forbid_runtime_registry_shared_imports() -> None:
    problems: list[str] = []
    for path in _ENV_MODULES:
        for imported in _module_imports(path):
            if any(imported.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
                problems.append(f"{path.name}: {imported}")
            if any(token in f".{imported}." for token in _FORBIDDEN_SUBSTRINGS):
                problems.append(f"{path.name}: {imported}")
            if imported in _FORBIDDEN_EXACT or imported.startswith(
                "intergrax.llm_adapters.routing.",
            ):
                if imported.startswith("intergrax.llm_adapters.contracts."):
                    continue
                problems.append(f"{path.name}: {imported}")
    assert not problems, "\n".join(problems)


def test_canonical_type_identity_no_duplicates() -> None:
    assert CanonicalUtilityWeights is LegacyUtilityWeights
    assert CanonicalAdaptiveLoopKind is LegacyAdaptiveLoopKind
    assert CanonicalScalingPolicy is LegacyScalingPolicy
    assert CanonicalPolicyEnforcementMode is LegacyPolicyEnforcementMode
    assert CanonicalComplianceDomainClass is LegacyComplianceDomainClass
    assert CanonicalModalityProfile is LegacyModalityProfile
    assert CanonicalPlatformPluginSelectionRef is LegacyPlatformPluginSelectionRef
    assert CanonicalContextOptimizationPolicy is LegacyContextOptimizationPolicy
    assert CanonicalEventCategory is LegacyEventCategory
    assert CanonicalRuntimeEventType is LegacyRuntimeEventType
    assert CanonicalToolProfile is LegacyToolProfile
    assert CanonicalSkillProfile is LegacySkillProfile
    assert CanonicalLLMProfile is LegacyLLMProfile
    assert CanonicalLLMRoutingProfile is LegacyLLMRoutingProfile
    assert CanonicalIntegrationProfile is LegacyIntegrationProfile
    assert CanonicalCodeCraftProfile is LegacyCodeCraftProfile


def test_environment_profile_wire_roundtrip_nested_and_flat() -> None:
    nested = ApplicationEnvironmentProfile.lab_defaults(profile_id="wire.nested")
    dumped = nested.model_dump(mode="json")
    restored = ApplicationEnvironmentProfile.model_validate(dumped)
    assert restored.profile_id == "wire.nested"
    assert restored.tool_profile.enabled_bundles == nested.tool_profile.enabled_bundles
    assert restored.llm_profile.model == nested.llm_profile.model

    flat = {
        "profile_id": "wire.flat",
        "spec_version": "1.0.0",
        "tool_profile": {"enabled": ["harness.get_run"], "enabled_bundles": []},
        "skill_profile": {"enabled_bundles": ["harness"]},
    }
    from_flat = ApplicationEnvironmentProfile.model_validate(flat)
    assert from_flat.profile_id == "wire.flat"
    assert "harness.get_run" in from_flat.tool_profile.enabled
    assert from_flat.skill_profile.enabled_bundles == ["harness"]


def test_lab_and_product_defaults_semantics() -> None:
    lab = ApplicationEnvironmentProfile.lab_defaults()
    assert lab.profile_id == "lab.default"
    assert lab.llm_profile is not None
    assert len(lab.tool_profile.enabled_bundles) >= 20
    product = ApplicationEnvironmentProfile.product_defaults()
    assert product.profile_id == "product.default"
    assert product.modality_profile is not None
    assert product.modality_profile.profile_id == "product.plane_c"
    assert product.scaling_profile.policy.enabled is True


def test_enum_value_parity_for_moved_enums() -> None:
    assert {m.value for m in CanonicalAdaptiveLoopKind} == {
        m.value for m in LegacyAdaptiveLoopKind
    }
    assert {m.value for m in CanonicalPolicyEnforcementMode} == {
        m.value for m in LegacyPolicyEnforcementMode
    }
    assert {m.value for m in CanonicalComplianceDomainClass} == {
        m.value for m in LegacyComplianceDomainClass
    }


_COMPAT_PROFILE_MODULES = (
    _REPO_ROOT / "intergrax/integrations/registry/profile.py",
    _REPO_ROOT / "intergrax/llm_adapters/registry/profile.py",
    _REPO_ROOT / "intergrax/tools/registry/profile.py",
    _REPO_ROOT / "intergrax/skills/registry/profile.py",
)

_CANONICAL_PROFILE_CONTRACTS = (
    _REPO_ROOT / "intergrax/integrations/contracts/integration_profile.py",
    _REPO_ROOT / "intergrax/llm_adapters/contracts/llm_profile.py",
    _REPO_ROOT / "intergrax/tools/contracts/tool_profile.py",
    _REPO_ROOT / "intergrax/skills/contracts/skill_profile.py",
)

_RUNTIME_AUGMENTATION_METHODS = frozenset(
    {
        "resolve",
        "lab_stack",
        "legal_stack",
        "research_stack",
        "data_stack",
        "observability_stack",
        "harness_production_stack",
        "create_adapter",
        "create_adapter_with_failover",
        "validate_runtime",
        "create_adapter_from_secrets_store",
    }
)


def _setattr_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "setattr":
            if node.args:
                target = node.args[0]
                hits.append(ast.unparse(target))
        elif isinstance(func, ast.Attribute) and func.attr == "setattr":
            if node.args:
                hits.append(ast.unparse(node.args[0]))
    return hits


def test_legacy_profile_modules_forbid_canonical_setattr() -> None:
    problems: list[str] = []
    for path in _COMPAT_PROFILE_MODULES:
        for target in _setattr_targets(path):
            problems.append(f"{path.name}: setattr({target}, ...)")
    assert not problems, "\n".join(problems)


def test_canonical_profile_contracts_forbid_registry_imports() -> None:
    problems: list[str] = []
    for path in _CANONICAL_PROFILE_CONTRACTS:
        for imported in _module_imports(path):
            if ".registry." in f".{imported}.":
                problems.append(f"{path.name}: {imported}")
            if imported.startswith("intergrax.runtime."):
                problems.append(f"{path.name}: {imported}")
    assert not problems, "\n".join(problems)


def test_import_order_stability_no_runtime_method_augmentation() -> None:
    import subprocess
    import sys

    script = r"""
import importlib

integration = importlib.import_module(
    "intergrax.integrations.contracts.integration_profile",
).IntegrationProfile
llm = importlib.import_module("intergrax.llm_adapters.contracts.llm_profile").LLMProfile
tool = importlib.import_module("intergrax.tools.contracts.tool_profile").ToolProfile
skill = importlib.import_module("intergrax.skills.contracts.skill_profile").SkillProfile

runtime_methods = {
    "resolve",
    "lab_stack",
    "legal_stack",
    "research_stack",
    "data_stack",
    "observability_stack",
    "harness_production_stack",
    "create_adapter",
    "create_adapter_with_failover",
    "validate_runtime",
    "create_adapter_from_secrets_store",
}
before = {
    "IntegrationProfile": frozenset(runtime_methods & set(vars(integration))),
    "LLMProfile": frozenset(runtime_methods & set(vars(llm))),
}
tool_before = tool.is_tool_enabled
skill_before = skill.is_skill_enabled

importlib.import_module("intergrax.integrations.registry.profile")
importlib.import_module("intergrax.llm_adapters.registry.profile")
importlib.import_module("intergrax.tools.registry.profile")
importlib.import_module("intergrax.skills.registry.profile")

after = {
    "IntegrationProfile": frozenset(runtime_methods & set(vars(integration))),
    "LLMProfile": frozenset(runtime_methods & set(vars(llm))),
}
assert before == after
assert tool.is_tool_enabled is tool_before
assert skill.is_skill_enabled is skill_before
assert not hasattr(integration, "resolve")
assert not hasattr(llm, "create_adapter")
print("OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_tool_profile_standalone_exact_id_semantics() -> None:
    profile = CanonicalToolProfile(enabled=["echo.ping"], enabled_bundles=["jira"])
    assert profile.is_tool_enabled("echo.ping") is True
    assert profile.is_tool_enabled("jira.get_issue") is False


def test_skill_profile_standalone_exact_id_semantics() -> None:
    profile = CanonicalSkillProfile(enabled=["skill.a"], enabled_bundles=["harness"])
    assert profile.is_skill_enabled("skill.a") is True
    assert profile.is_skill_enabled("harness.run") is False


def test_catalog_evaluators_remain_registry_owned() -> None:
    from intergrax.skills.registry.profile import is_skill_enabled
    from intergrax.tools.registry.bootstrap import register_default_tools
    from intergrax.tools.registry.profile import is_tool_enabled

    register_default_tools()
    tools = CanonicalToolProfile(enabled_bundles=["harness"])
    assert is_tool_enabled(tools, "harness.get_run") is True
    assert tools.is_tool_enabled("harness.get_run") is False

    skills = CanonicalSkillProfile(enabled=["explicit.skill"])
    assert is_skill_enabled(skills, "explicit.skill") is True
    assert skills.is_skill_enabled("explicit.skill") is True


# --- EBH-2D-B-R2: typed tool enablement boundary ---

_RESEARCH_AGENT_PATH = _REPO_ROOT / "agents/research/research_agent.py"
_TOOL_ENABLEMENT_CONTRACT_PATH = _REPO_ROOT / "intergrax/agents/tool_enablement.py"
_TIER2_TOOL_ENABLEMENT_CONSUMERS = (
    _RESEARCH_AGENT_PATH,
    _REPO_ROOT / "agents/boundary_demo/boundary_demo_agent.py",
)


def _imports_under(path: Path, *, prefix: str) -> list[str]:
    hits: list[str] = []
    for name in _module_imports(path):
        if name == prefix or name.startswith(prefix + "."):
            hits.append(name)
    return hits


def test_research_agent_does_not_import_tools_registry() -> None:
    assert _imports_under(_RESEARCH_AGENT_PATH, prefix="intergrax.tools.registry") == []
    assert _imports_under(_RESEARCH_AGENT_PATH, prefix="intergrax.tools.contracts.tool_profile") == []


def test_tool_enablement_consumers_do_not_import_tools_registry() -> None:
    for path in _TIER2_TOOL_ENABLEMENT_CONSUMERS:
        assert path.is_file(), path
        assert _imports_under(path, prefix="intergrax.tools.registry") == [], path


def test_research_agent_has_no_tool_profile_isinstance_switch() -> None:
    source = _RESEARCH_AGENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "isinstance":
            continue
        if len(node.args) < 2:
            continue
        type_arg = node.args[1]
        names: set[str] = set()
        if isinstance(type_arg, ast.Name):
            names.add(type_arg.id)
        elif isinstance(type_arg, ast.Tuple):
            for elt in type_arg.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        assert "ToolProfile" not in names, "ResearchAgent must not isinstance-switch on ToolProfile"


def test_tool_enablement_contract_stays_pure() -> None:
    forbidden = (
        "intergrax.tools",
        "intergrax.runtime",
    )
    imports = _module_imports(_TOOL_ENABLEMENT_CONTRACT_PATH)
    for name in imports:
        for prefix in forbidden:
            assert not (name == prefix or name.startswith(prefix + ".")), name
        assert ".registry." not in name, name


def test_custom_tool_enablement_injected_into_research_agent() -> None:
    from research.research_agent import ResearchAgent

    class ProbeEnablement:
        def is_tool_enabled(self, tool_id: str) -> bool:
            return tool_id == "probe.tool"

    agent = ResearchAgent(tool_profile=ProbeEnablement())
    assert agent._tool_enables("probe.tool") is True
    assert agent._tool_enables("other.tool") is False
    assert ResearchAgent()._tool_enables("probe.tool") is False


def test_catalog_tool_enablement_view_bundle_semantics() -> None:
    from intergrax.tools.registry.bootstrap import register_default_tools
    from intergrax.tools.registry.enablement import CatalogToolEnablementView

    register_default_tools()
    view = CatalogToolEnablementView(
        CanonicalToolProfile(enabled_bundles=["harness"]),
    )
    assert view.is_tool_enabled("harness.get_run") is True
    assert view.is_tool_enabled("missing.tool") is False


def test_catalog_tool_enablement_view_exact_id_and_register_all() -> None:
    from intergrax.tools.registry.enablement import CatalogToolEnablementView

    exact = CatalogToolEnablementView(CanonicalToolProfile(enabled=["echo.ping"]))
    assert exact.is_tool_enabled("echo.ping") is True
    assert exact.is_tool_enabled("echo.other") is False

    all_catalog = CatalogToolEnablementView(
        CanonicalToolProfile(register_all_catalog_bundles=True),
    )
    assert all_catalog.is_tool_enabled("any.tool.id") is True


# --- EBH-2D-B-R3: enablement / runtime materialization coherence ---

_BINDING_PATH = _REPO_ROOT / "intergrax/applications/_shared/tool_enablement_binding.py"
_REFERENCE_HARNESS_PATH = _REPO_ROOT / "intergrax/agents/reference_harness.py"
_BOUNDARY_DEMO_PATH = _REPO_ROOT / "agents/boundary_demo/boundary_demo_agent.py"


def test_research_agent_does_not_assign_tool_profile_onto_runtime_config() -> None:
    source = _RESEARCH_AGENT_PATH.read_text(encoding="utf-8")
    assert "config.tool_profile" not in source
    assert "runtime_context.config.tool_profile" not in source


def test_boundary_demo_does_not_assign_enablement_onto_runtime_tool_profile() -> None:
    source = _BOUNDARY_DEMO_PATH.read_text(encoding="utf-8")
    assert "config.tool_profile = self._tool_profile" not in source


def test_no_reverse_extraction_of_catalog_enablement_private_profile() -> None:
    forbidden_snippets = (
        "._profile",
        'getattr(',
        "hasattr(",
    )
    for path in (_RESEARCH_AGENT_PATH, _BOUNDARY_DEMO_PATH, _BINDING_PATH):
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in source, f"{path}: forbidden {snippet!r}"


def test_no_duplicate_tool_profile_dto_aliases() -> None:
    hits: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in ("RuntimeToolProfile", "AgentToolProfile", "ResolvedToolProfile"):
            if name in text:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert hits == []


def test_lab_harness_carries_prebuilt_tool_registry_into_runtime_config() -> None:
    from intergrax.agents.reference_harness import (
        LabHarnessContext,
        build_lab_agent_runtime_config,
    )
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.tools.registry.runtime import ToolRegistry
    from testing_support.builder import FakeLLMAdapter, build_runtime_request_for_tests

    registry = ToolRegistry()
    harness = LabHarnessContext(
        policy_bundle=RuntimePolicyBundle(),
        tool_registry=registry,
    )
    config = build_lab_agent_runtime_config(
        request=build_runtime_request_for_tests(
            seed="r3-harness-registry",
            tenant_id="t",
            agent_id="research",
            user_id="u",
            session_id="s",
            message="probe",
        ),
        llm_adapter=FakeLLMAdapter(),
        harness=harness,
    )
    assert config.tool_registry is registry


def test_composition_enablement_and_registry_share_canonical_tool_profile() -> None:
    from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.bootstrap import register_default_tools
    from research.research_agent import ResearchAgent
    from intergrax.agents.reference_harness import LabHarnessContext
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from testing_support.builder import build_runtime_request_for_tests

    register_default_tools()
    profile = CanonicalToolProfile(enabled_bundles=["harness"])
    wiring = build_application_tool_wiring(profile)
    enablement = resolve_tool_enablement(None, environment_tool_profile=profile)
    assert enablement is not None
    assert enablement.is_tool_enabled("harness.get_run") is True

    agent = ResearchAgent(
        LabHarnessContext(
            policy_bundle=RuntimePolicyBundle(),
            tool_wiring_context=wiring.wiring_context,
            tool_registry=wiring.registry,
        ),
        tool_profile=enablement,
    )
    assert agent._tool_enables("harness.get_run") is True
    ctx = agent.build_context(
        build_runtime_request_for_tests(
            seed="r3-coherence-bundle",
            tenant_id="t",
            agent_id="research",
            user_id="u",
            session_id="s",
            message="probe",
        )
    )
    assert ctx.config.tool_registry is wiring.registry
    assert wiring.registry.has("harness.get_run")


def test_composition_disabled_tool_not_enabled_and_not_registered() -> None:
    from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.bootstrap import register_default_tools
    from research.research_agent import ResearchAgent
    from intergrax.agents.reference_harness import LabHarnessContext
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from testing_support.builder import build_runtime_request_for_tests

    register_default_tools()
    profile = CanonicalToolProfile(enabled=["harness.get_run"])
    wiring = build_application_tool_wiring(profile)
    enablement = resolve_tool_enablement(None, environment_tool_profile=profile)
    assert enablement is not None
    assert enablement.is_tool_enabled("harness.compare_runs") is False
    assert wiring.registry.has("harness.get_run")
    assert not wiring.registry.has("harness.compare_runs")

    agent = ResearchAgent(
        LabHarnessContext(
            policy_bundle=RuntimePolicyBundle(),
            tool_registry=wiring.registry,
        ),
        tool_profile=enablement,
    )
    assert agent._tool_enables("harness.compare_runs") is False
    ctx = agent.build_context(
        build_runtime_request_for_tests(
            seed="r3-coherence-disabled",
            tenant_id="t",
            agent_id="research",
            user_id="u",
            session_id="s",
            message="probe",
        )
    )
    assert ctx.config.tool_registry is wiring.registry
    assert not ctx.config.tool_registry.has("harness.compare_runs")


def test_composition_register_all_parity() -> None:
    from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.bootstrap import register_default_tools

    register_default_tools()
    profile = CanonicalToolProfile(register_all_catalog_bundles=True)
    wiring = build_application_tool_wiring(profile)
    enablement = resolve_tool_enablement(None, environment_tool_profile=profile)
    assert enablement is not None
    assert enablement.is_tool_enabled("harness.get_run") is True
    assert wiring.registry.has("harness.get_run")


def test_custom_enablement_does_not_become_registry_authority() -> None:
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.bootstrap import register_default_tools
    from research.research_agent import ResearchAgent
    from intergrax.agents.reference_harness import LabHarnessContext
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from testing_support.builder import build_runtime_request_for_tests

    register_default_tools()
    profile = CanonicalToolProfile(enabled=["harness.get_run"])
    wiring = build_application_tool_wiring(profile)

    class CustomEnablement:
        def is_tool_enabled(self, tool_id: str) -> bool:
            return tool_id == "custom.only"

    agent = ResearchAgent(
        LabHarnessContext(
            policy_bundle=RuntimePolicyBundle(),
            tool_registry=wiring.registry,
        ),
        tool_profile=CustomEnablement(),
    )
    assert agent._tool_enables("custom.only") is True
    assert agent._tool_enables("harness.get_run") is False
    ctx = agent.build_context(
        build_runtime_request_for_tests(
            seed="r3-custom-enablement",
            tenant_id="t",
            agent_id="research",
            user_id="u",
            session_id="s",
            message="probe",
        )
    )
    assert ctx.config.tool_registry is wiring.registry
    assert wiring.registry.has("harness.get_run")
    assert not wiring.registry.has("custom.only")


def test_single_source_composition_path_reuses_same_tool_profile() -> None:
    from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.enablement import CatalogToolEnablementView
    from intergrax.tools.registry.bootstrap import register_default_tools

    register_default_tools()
    profile = CanonicalToolProfile(enabled_bundles=["harness"])
    wiring = build_application_tool_wiring(profile)
    enablement = resolve_tool_enablement(
        profile,
        environment_tool_profile=profile,
    )
    assert isinstance(enablement, CatalogToolEnablementView)
    assert enablement.is_tool_enabled("harness.get_run") is True
    assert wiring.registry.has("harness.get_run")
    assert wiring.profile is profile


# --- EBH-2D-B-R4: typed tool runtime registry boundary ---

_RUNTIME_CONFIG_PATH = _REPO_ROOT / "intergrax/runtime/nexus/config.py"
_REGISTRY_EXECUTOR_PATH = (
    _REPO_ROOT / "intergrax/runtime/nexus/tools/registry_tool_executor.py"
)
_RUNTIME_READ_BOUNDARY_PATHS = (
    _REFERENCE_HARNESS_PATH,
    _RUNTIME_CONFIG_PATH,
    _REGISTRY_EXECUTOR_PATH,
    _REPO_ROOT / "intergrax/runtime/nexus/engine/runtime_context.py",
)


def test_reference_harness_does_not_import_concrete_tool_registry() -> None:
    imports = _module_imports(_REFERENCE_HARNESS_PATH)
    assert "intergrax.tools.registry.runtime" not in imports
    forbidden = [n for n in imports if n.endswith("ToolRegistry") and "runtime" in n]
    assert forbidden == []


def test_runtime_config_tool_registry_uses_read_contract() -> None:
    source = _RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
    assert "ToolRegistryRead" in source
    assert "tool_registry: Optional[ToolRegistryRead]" in source
    assert "tool_registry: Optional[ToolRegistry]" not in source


def test_registry_tool_executor_constructor_uses_read_contract() -> None:
    source = _REGISTRY_EXECUTOR_PATH.read_text(encoding="utf-8")
    assert "ToolRegistryRead" in source
    tree = ast.parse(source)
    init_params: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                if arg.annotation is not None:
                    init_params.append(ast.unparse(arg.annotation))
    assert "ToolRegistryRead" in init_params
    assert "ToolRegistry" not in init_params


def test_runtime_read_boundary_modules_do_not_call_registry_mutation() -> None:
    forbidden_calls = (".register(", ".unregister(")
    for path in _RUNTIME_READ_BOUNDARY_PATHS:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_calls:
            assert snippet not in source, f"{path}: forbidden {snippet!r}"


def test_probe_registry_structural_acceptance_by_runtime_executor() -> None:
    from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
    from intergrax.tools.examples.custom_echo.plugin import CUSTOM_ECHO_TOOL_ID, CustomEchoInput
    from intergrax.tools.execution_models import ToolExecutionRequest
    from intergrax.tools.registry.factory import build_registry_from_profile
    from intergrax.tools.registry.plugin_register import register_tool_plugin
    from intergrax.tools.registry.runtime import ToolRegistry
    from intergrax.tools.examples.custom_echo import CustomEchoToolPlugin

    register_tool_plugin(CustomEchoToolPlugin, override=True)
    backing = ToolRegistry()
    build_registry_from_profile(
        CanonicalToolProfile(enabled_bundles=["custom_echo"]),
        registry=backing,
    )

    class ProbeRegistry:
        def has(self, tool_id: str) -> bool:
            return backing.has(tool_id)

        def get(self, tool_id: str):
            return backing.get(tool_id)

        def activation_metadata(self, tool_id: str):
            return backing.activation_metadata(tool_id)

    executor = RegistryToolExecutor(ProbeRegistry())
    result = executor.execute(
        ToolExecutionRequest(
            run_id="run/r4-probe",
            step_id="step/1",
            tool_id=CUSTOM_ECHO_TOOL_ID,
            input=CustomEchoInput(message="r4-probe"),
        ),
    )
    assert result.message == "r4-probe"


def test_probe_registry_via_runtime_config_harness_path() -> None:
    from intergrax.agents.reference_harness import (
        LabHarnessContext,
        build_lab_agent_runtime_config,
    )
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.tools.registry.bootstrap import register_default_tools
    from intergrax.tools.registry.factory import build_registry_from_profile
    from intergrax.tools.registry.runtime import ToolRegistry
    from testing_support.builder import FakeLLMAdapter, build_runtime_request_for_tests

    register_default_tools()
    backing = ToolRegistry()
    build_registry_from_profile(
        CanonicalToolProfile(enabled_bundles=["harness"]),
        registry=backing,
    )

    class ProbeRegistry:
        def has(self, tool_id: str) -> bool:
            return backing.has(tool_id)

        def get(self, tool_id: str):
            return backing.get(tool_id)

        def activation_metadata(self, tool_id: str):
            return backing.activation_metadata(tool_id)

    probe = ProbeRegistry()
    harness = LabHarnessContext(
        policy_bundle=RuntimePolicyBundle(),
        tool_registry=probe,
    )
    config = build_lab_agent_runtime_config(
        request=build_runtime_request_for_tests(
            seed="r4-probe-config",
            tenant_id="t",
            agent_id="research",
            user_id="u",
            session_id="s",
            message="probe",
        ),
        llm_adapter=FakeLLMAdapter(),
        harness=harness,
    )
    assert config.tool_registry is probe
