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
