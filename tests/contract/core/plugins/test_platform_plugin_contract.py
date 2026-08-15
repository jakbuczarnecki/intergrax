# © Artur Czarnecki. All rights reserved.

"""PLATFORM-PLUGIN-9 cross-stage conformance suite (PLUGIN-3..8 invariants)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from intergrax.utils import attribute_access
from intergrax.core import plugins as core_plugins
from intergrax.core.plugins import (
    EP_INTEGRATIONS,
    EP_POLICY_RULES,
    EP_SECURITY_DEFENSES,
    EP_SKILLS,
    EP_TOOL_INVOCATION_PATTERNS,
    EP_TOOLS,
    PlatformPluginLifecycleState,
    PlatformPluginTrustModel,
    PluginDeliverySource,
    PluginQualificationEvidenceKind,
    build_external_package_subject,
    build_host_embedded_capability_subject,
    build_platform_plugin_manifest,
    build_qualification_result,
    evaluate_package_production_admission,
    iter_entry_point_specs,
    load_entry_point_value,
    parse_platform_plugin_pyproject_toml,
)
from intergrax.core.qualification import QualificationStatus
from intergrax.core.plugins.discovery import (
    EP_CONTEXT,
    EP_MEMORY_STORES,
    EP_RAG_CHUNKERS,
    EP_RAG_RERANKERS,
    EP_RAG_RETRIEVERS,
    instantiate_entry_point_target,
)
from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.distribution import PlatformCompatibility
from intergrax.core.plugins.package_contract import (
    CapabilityDescriptor,
    reject_secret_like_keys,
)
from intergrax.core.distribution import (
    PlatformCompatibilityReason,
    check_platform_compatibility,
)
from intergrax.scaffold.application_extension_templates import (
    local_prefix_echo_plugin_py,
    tool_wiring_local_extension_block,
)
from intergrax.core.plugins import require_production_qualification
from intergrax.tools.examples.custom_echo import CustomEchoToolPlugin
from intergrax.tools.registry.plugin_register import register_tool_plugin

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.ci_smoke]

_FACTORY_WAS_INVOKED = False


def _factory_invocation_probe() -> None:
    global _FACTORY_WAS_INVOKED
    _FACTORY_WAS_INVOKED = True

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REFERENCE_PLUGIN = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_tool_plugin"
    / "src"
    / "intergrax_reference_tool_plugin"
    / "plugin.py"
)
_LOCAL_PLUGIN = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "local_embedded_tool_extension"
    / "local_prefix_echo_plugin.py"
)
_CANONICAL_EP_GROUPS = (
    EP_INTEGRATIONS,
    EP_TOOLS,
    EP_SKILLS,
    EP_CONTEXT,
    EP_MEMORY_STORES,
    EP_RAG_CHUNKERS,
    EP_RAG_RETRIEVERS,
    EP_RAG_RERANKERS,
    EP_SECURITY_DEFENSES,
    EP_POLICY_RULES,
    EP_TOOL_INVOCATION_PATTERNS,
    "intergrax.vendor_knowledge.providers",
)


def _load_plugin_class(path: Path, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    plugin_type = attribute_access.optional(module, class_name)
    assert isinstance(plugin_type, type)
    return plugin_type


# --- Package / manifest (1-4) ---


def test_manifest_remains_package_level_coordination_metadata() -> None:
    manifest = build_platform_plugin_manifest(
        name="acme-intergrax",
        version="1.0.0",
        intergrax_version=">=1,<2",
    )
    assert manifest.capabilities == ()
    assert manifest.package.name == "acme-intergrax"


def test_package_identity_must_agree_with_python_project_metadata() -> None:
    with pytest.raises(
        PlatformPluginManifestValidationError,
        match="manifest package name conflicts",
    ):
        parse_platform_plugin_pyproject_toml(
            """
            [project]
            name = "acme-intergrax"
            version = "1.0.0"

            [tool.intergrax.plugin]
            name = "other-plugin"
            version = "1.0.0"
            intergrax_version = ">=1"
            """
        )


def test_manifest_cannot_carry_runtime_config_or_secrets() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="secret-like manifest field"):
        reject_secret_like_keys({"api_token": "value"})


def test_multi_capability_packages_remain_supported() -> None:
    manifest = build_platform_plugin_manifest(
        name="acme-intergrax",
        version="1.0.0",
        intergrax_version=">=1,<2",
        capabilities=[
            CapabilityDescriptor(
                domain="integrations",
                entry_point_group=EP_INTEGRATIONS,
                entry_point_name="acme_foo",
            ),
            CapabilityDescriptor(
                domain="tools",
                entry_point_group=EP_TOOLS,
                entry_point_name="acme_tool",
            ),
        ],
    )
    assert len(manifest.capabilities) == 2


# --- Discovery (5-8) ---


def test_canonical_public_ep_groups_remain_domain_scoped() -> None:
    for group in _CANONICAL_EP_GROUPS:
        assert group.startswith("intergrax.")
        assert group.count(".") >= 1


def test_no_single_mandatory_global_platform_ep_group() -> None:
    exported = {name for name in dir(core_plugins) if name.startswith("EP_")}
    assert "EP_PLATFORM_PLUGINS" not in exported
    assert "intergrax.platform_plugins" not in _CANONICAL_EP_GROUPS


def test_discovery_loader_does_not_execute_callables() -> None:
    global _FACTORY_WAS_INVOKED
    _FACTORY_WAS_INVOKED = False
    target = load_entry_point_value(f"{__name__}:_factory_invocation_probe")
    assert callable(target)
    assert _FACTORY_WAS_INVOKED is False
    assert instantiate_entry_point_target(target) is target


def test_external_package_discovery_remains_entry_point_based() -> None:
    specs = iter_entry_point_specs(EP_TOOLS)
    assert isinstance(specs, tuple)
    for spec in specs:
        assert spec.group == EP_TOOLS
        assert spec.name
        assert spec.value


# --- Local embedded delivery (9-13) ---


def test_host_embedded_extension_does_not_require_wheel_or_entry_point() -> None:
    subject = build_host_embedded_capability_subject(
        domain="tools",
        capability_id="local_prefix_echo",
        host_registration_path="extensions/local_prefix_echo_plugin.py",
    )
    assert subject.delivery_source is PluginDeliverySource.HOST_EMBEDDED_EXTENSION
    assert subject.package_name is None
    assert subject.entry_point_name is None
    assert subject.entry_point_group is None


def test_local_path_uses_explicit_registration_helper() -> None:
    source = inspect.getsource(tool_wiring_local_extension_block)
    assert "register_tool_plugin" in source


def test_scaffold_local_plugin_has_no_import_time_registration() -> None:
    source = inspect.getsource(local_prefix_echo_plugin_py)
    assert "register_tool_plugin" not in source
    assert "require_production_qualification" not in source


def test_scaffold_qualification_precedes_registration() -> None:
    source = inspect.getsource(tool_wiring_local_extension_block)
    qual_idx = source.index("require_production_qualification")
    register_idx = source.index("register_tool_plugin")
    assert qual_idx < register_idx


# --- Qualification / trust (14-20) ---


def test_discovery_distinct_from_qualification_states() -> None:
    lifecycle = {state.value for state in PlatformPluginLifecycleState}
    qualification = {status.value for status in QualificationStatus}
    assert lifecycle.isdisjoint(qualification)


def test_compatible_does_not_imply_qualified() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=1,<2"),
        "1.0",
    )
    assert compatibility.compatible is True
    result = build_qualification_result(
        subject=build_external_package_subject(
            level=core_plugins.PluginQualificationLevel.PACKAGE,
            package_name="acme-intergrax",
            package_version="1.0.0",
        ),
        status=QualificationStatus.NOT_QUALIFIED,
        evidence=(),
        reason="compatible only",
    )
    assert result.status is QualificationStatus.NOT_QUALIFIED


def test_qualified_does_not_imply_production_qualified() -> None:
    result = build_qualification_result(
        subject=build_external_package_subject(
            level=core_plugins.PluginQualificationLevel.PACKAGE,
            package_name="acme-intergrax",
            package_version="1.0.0",
        ),
        status=QualificationStatus.QUALIFIED,
        evidence=(),
        reason="domain qualified",
    )
    assert result.production_allowed is False


def test_external_package_missing_compatibility_fails_closed() -> None:
    package_result = build_qualification_result(
        subject=build_external_package_subject(
            level=core_plugins.PluginQualificationLevel.PACKAGE,
            package_name="acme-intergrax",
            package_version="1.0.0",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="production qualified",
    )
    admission = evaluate_package_production_admission(package_result, compatibility=None)
    assert admission.admitted is False


def test_host_embedded_package_compatibility_not_fabricated() -> None:
    host_result = build_qualification_result(
        subject=build_host_embedded_capability_subject(
            domain="tools",
            capability_id="local_prefix_echo",
            host_registration_path="extensions/local_prefix_echo_plugin.py",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="host embedded",
    )
    assert all(
        evidence.kind.value != PluginQualificationEvidenceKind.PLATFORM_COMPATIBILITY.value
        for evidence in host_result.evidence
    )
    assert require_production_qualification(host_result) is host_result


def test_trust_model_is_in_process_only() -> None:
    assert list(PlatformPluginTrustModel) == [PlatformPluginTrustModel.TRUSTED_IN_PROCESS]


def test_no_sandbox_signing_or_isolation_states_claimed() -> None:
    trust_values = {item.value for item in PlatformPluginTrustModel}
    forbidden = {"sandboxed", "signed", "verified", "isolated", "process_isolated"}
    assert trust_values.isdisjoint(forbidden)
    qualification_values = {item.value for item in QualificationStatus}
    assert qualification_values.isdisjoint(forbidden)


# --- Domain convergence (21-24) ---


def test_external_and_local_tool_plugins_share_contract() -> None:
    reference_cls = _load_plugin_class(_REFERENCE_PLUGIN, "ReferencePrefixEchoToolPlugin")
    local_cls = _load_plugin_class(_LOCAL_PLUGIN, "LocalPrefixEchoToolPlugin")
    assert issubclass(reference_cls, object)
    assert hasattr(reference_cls, "tool_bundle_manifest")
    assert hasattr(reference_cls, "register_tools")
    assert hasattr(local_cls, "tool_bundle_manifest")
    assert hasattr(local_cls, "register_tools")


def test_both_delivery_modes_materialize_through_same_tool_registry() -> None:
    assert register_tool_plugin.__module__ == "intergrax.tools.registry.plugin_register"
    register_tool_plugin(CustomEchoToolPlugin)
    assert hasattr(CustomEchoToolPlugin, "tool_bundle_manifest")
    assert hasattr(CustomEchoToolPlugin, "register_tools")


def test_both_delivery_modes_use_tool_wiring_context() -> None:
    reference_cls = _load_plugin_class(_REFERENCE_PLUGIN, "ReferencePrefixEchoToolPlugin")
    local_cls = _load_plugin_class(_LOCAL_PLUGIN, "LocalPrefixEchoToolPlugin")
    ref_sig = inspect.signature(reference_cls.register_tools)
    local_sig = inspect.signature(local_cls.register_tools)
    assert "ctx" in ref_sig.parameters
    assert "ctx" in local_sig.parameters
    assert "ToolWiringContext" in str(ref_sig.parameters["ctx"].annotation)
    assert "ToolWiringContext" in str(local_sig.parameters["ctx"].annotation)


def test_runtime_invocation_path_is_shared_registry_invoker() -> None:
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
    from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor

    assert RuntimeToolInvoker.__module__ == "intergrax.runtime.nexus.tools.invoker"
    assert RegistryToolExecutor.__module__ == "intergrax.runtime.nexus.tools.registry_tool_executor"


# --- Architecture guards (25-30) ---


def test_no_platform_plugin_execute_universal_runtime_api() -> None:
    assert not hasattr(core_plugins, "PlatformPlugin")
    removed_plugin_exports = (
        "PluginPackageIdentity",
        "PluginQualificationStatus",
        "PlatformCompatibility",
        "PlatformCompatibilityResult",
        "InvalidPlatformVersionError",
        "PlatformIncompatibilityError",
    )
    for name in removed_plugin_exports:
        assert not hasattr(core_plugins, name), name
    plugin_modules = [
        "intergrax.core.plugins.discovery",
        "intergrax.core.plugins.platform_qualification",
        "intergrax.core.plugins.platform_semantics",
        "intergrax.core.plugins.package_contract",
    ]
    for module_name in plugin_modules:
        module = importlib.import_module(module_name)
        assert not hasattr(module, "PlatformPlugin")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Plugin") and name != "LoadedPlugin":
                assert not hasattr(obj, "execute"), f"{module_name}.{name} must not expose execute()"


def test_no_global_platform_plugin_manager() -> None:
    assert "PlatformPluginManager" not in core_plugins.__all__
    with pytest.raises(AttributeError):
        attribute_access.optional(core_plugins, "PlatformPluginManager")


def test_no_global_local_plugin_scanning_mechanism() -> None:
    discovery_source = inspect.getsource(importlib.import_module("intergrax.core.plugins.discovery"))
    forbidden = ("scan_local", "discover_local", "import_by_path", "load_plugins_from_path")
    assert not any(token in discovery_source for token in forbidden)


def test_no_global_secret_api_introduced_by_program() -> None:
    exported = set(core_plugins.__all__)
    forbidden = {
        "PlatformPluginSecrets",
        "PluginSecretResolver",
        "GlobalSecretStore",
        "resolve_plugin_secret",
    }
    assert exported.isdisjoint(forbidden)


def test_no_global_di_container_introduced_by_program() -> None:
    exported = set(core_plugins.__all__)
    forbidden = {
        "PlatformPluginContainer",
        "GlobalPluginDIContainer",
        "PluginDependencyContainer",
    }
    assert exported.isdisjoint(forbidden)


def test_vendor_knowledge_remains_domain_owned_not_tier0_catalog() -> None:
    tier0_groups = {
        EP_INTEGRATIONS,
        EP_TOOLS,
        EP_SKILLS,
        EP_CONTEXT,
        EP_MEMORY_STORES,
        EP_RAG_CHUNKERS,
        EP_RAG_RETRIEVERS,
        EP_RAG_RERANKERS,
    }
    assert "intergrax.vendor_knowledge.providers" not in tier0_groups
    vk_catalog = importlib.import_module("intergrax.runtime.vendor_knowledge.contribution_catalog")
    assert (
        vk_catalog.VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP
        == "intergrax.vendor_knowledge.providers"
    )


def test_incompatible_external_package_fails_closed_at_admission() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=2,<3"),
        "1.0",
    )
    assert compatibility.compatible is False
    assert compatibility.reason is PlatformCompatibilityReason.INCOMPATIBLE_VERSION
    package_result = build_qualification_result(
        subject=build_external_package_subject(
            level=core_plugins.PluginQualificationLevel.PACKAGE,
            package_name="acme-intergrax",
            package_version="1.0.0",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="production qualified",
    )
    admission = evaluate_package_production_admission(package_result, compatibility=compatibility)
    assert admission.admitted is False
