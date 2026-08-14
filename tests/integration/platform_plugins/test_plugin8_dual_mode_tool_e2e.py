# © Artur Czarnecki. All rights reserved.

"""PLATFORM-PLUGIN-8 executable dual-mode E2E proof (external wheel + host-embedded)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, distribution, distributions
from pathlib import Path
from typing import Any

import pytest
import tomllib

from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins import (
    EP_TOOLS,
    PluginDeliverySource,
    PluginQualificationEvidence,
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    PluginQualificationStatus,
    build_external_package_subject,
    build_host_embedded_capability_subject,
    build_qualification_result,
    compatibility_evidence,
    evaluate_package_production_admission,
    iter_entry_point_specs,
    load_entry_point_plugins,
    parse_platform_plugin_pyproject,
    require_production_qualification,
)
from intergrax.core.plugins.errors import ProductionQualificationRequiredError
from intergrax.core.plugins.package_contract import PlatformCompatibility
from intergrax.core.plugins.platform_semantics import check_platform_compatibility
from intergrax.tools.core.plugin import ToolPlugin
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = [pytest.mark.integration, pytest.mark.gate, pytest.mark.ci_smoke]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_PKG_DIR = _REPO_ROOT / "examples" / "platform_plugins" / "intergrax_reference_tool_plugin"
_LOCAL_PLUGIN_PATH = (
    _REPO_ROOT / "examples" / "platform_plugins" / "local_embedded_tool_extension" / "local_prefix_echo_plugin.py"
)
_EP_NAME = "reference_prefix_echo"
_PACKAGE_NAME = "intergrax-reference-tool-plugin"
_PACKAGE_VERSION = "0.1.0"
_HOST_PLATFORM_VERSION = "0.1.0"
_EXTRAS_PREFIX = "PLUGIN8"


@pytest.fixture(autouse=True)
def _clean_tool_catalog() -> Iterator[None]:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()


def _build_reference_wheel(build_dir: Path) -> Path:
    dist_dir = build_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            ["uv", "build", "--wheel", "-o", str(dist_dir)],
            cwd=str(_REFERENCE_PKG_DIR),
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "wheel", str(_REFERENCE_PKG_DIR), "-w", str(dist_dir), "--no-deps"],
            cwd=str(_REPO_ROOT),
        )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found: {wheels}"
    return wheels[0]


def _install_wheel_to_target(wheel_path: Path, target: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for isolated wheel installation in PLUGIN-8 E2E proof")
    subprocess.check_call(
        [
            uv,
            "pip",
            "install",
            str(wheel_path),
            "--target",
            str(target),
            "--no-deps",
            "--reinstall-package",
            _PACKAGE_NAME,
            "--python",
            sys.executable,
        ],
        cwd=str(_REPO_ROOT),
    )


def _load_local_prefix_echo_plugin() -> tuple[type[ToolPlugin], Any]:
    spec = importlib.util.spec_from_file_location("plugin8_local_prefix_echo_plugin", _LOCAL_PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plugin_type = module.LocalPrefixEchoToolPlugin
    assert isinstance(plugin_type, type)
    return plugin_type, module


def _invoke_tool(
    *,
    registry,
    tool_id: str,
    input_model: Any,
    message: str,
) -> Any:
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="plugin8")
    result = invoker.invoke(
        state=state,
        agent_id="agent",
        request=ToolExecutionRequest(
            run_id="plugin8",
            step_id="step/1",
            tool_id=tool_id,
            input=input_model(message=message),
        ),
    )
    assert result.success is True
    assert result.output is not None
    return result.output


def _production_package_result(*, compatibility) -> Any:
    subject = build_external_package_subject(
        level=PluginQualificationLevel.PACKAGE,
        package_name=_PACKAGE_NAME,
        package_version=_PACKAGE_VERSION,
        domain="tools",
        capability_id="reference_prefix_echo",
        entry_point_group=EP_TOOLS,
        entry_point_name=_EP_NAME,
    )
    return build_qualification_result(
        subject=subject,
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            compatibility_evidence(compatibility),
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="tools.reference.tests.passed",
                ref="tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py",
            ),
        ),
        reason="reference external tool package production-qualified",
    )


def _production_capability_result(*, capability_id: str, host_path: str | None = None) -> Any:
    if host_path is None:
        subject = build_external_package_subject(
            level=PluginQualificationLevel.CAPABILITY,
            package_name=_PACKAGE_NAME,
            package_version=_PACKAGE_VERSION,
            domain="tools",
            capability_id=capability_id,
            entry_point_group=EP_TOOLS,
            entry_point_name=capability_id,
        )
    else:
        subject = build_host_embedded_capability_subject(
            domain="tools",
            capability_id=capability_id,
            host_registration_path=host_path,
        )
    return build_qualification_result(
        subject=subject,
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="tools.capability.tests.passed",
                ref="tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py",
            ),
        ),
        reason=f"{capability_id} capability production-qualified",
    )


def test_external_reference_wheel_end_to_end(tmp_path: Path) -> None:
    assert _REFERENCE_PKG_DIR.is_dir()
    assert not str(_REFERENCE_PKG_DIR).replace("\\", "/").startswith("intergrax/")

    wheel_path = _build_reference_wheel(tmp_path)
    assert wheel_path.suffix == ".whl"
    assert wheel_path.is_file()

    install_target = tmp_path / "site-packages"
    install_target.mkdir()
    _install_wheel_to_target(wheel_path, install_target)

    inserted = str(install_target)
    sys.path.insert(0, inserted)
    reset_entry_point_spec_cache_for_tests()
    try:
        dist_names = {
            dist.metadata["Name"]
            for dist in distributions(path=[str(install_target)])
        }
        assert _PACKAGE_NAME in dist_names
        installed = distribution(_PACKAGE_NAME)
        assert installed.version == _PACKAGE_VERSION

        pyproject_data = tomllib.loads((_REFERENCE_PKG_DIR / "pyproject.toml").read_text(encoding="utf-8"))
        manifest = parse_platform_plugin_pyproject(pyproject_data)
        assert manifest.package.name == _PACKAGE_NAME
        assert manifest.package.version == _PACKAGE_VERSION
        assert manifest.capabilities
        capability = manifest.capabilities[0]
        assert capability.entry_point_group == EP_TOOLS
        assert capability.entry_point_name == _EP_NAME

        specs = [spec for spec in iter_entry_point_specs(EP_TOOLS) if spec.name == _EP_NAME]
        assert len(specs) == 1
        assert specs[0].group == EP_TOOLS
        assert specs[0].distribution in {_PACKAGE_NAME, None}

        loaded = [item for item in load_entry_point_plugins(EP_TOOLS) if item.name == _EP_NAME]
        assert len(loaded) == 1
        plugin_type = loaded[0].plugin_type
        assert issubclass(plugin_type, object)
        assert hasattr(plugin_type, "tool_bundle_manifest")
        assert hasattr(plugin_type, "register_tools")

        compatibility = check_platform_compatibility(
            manifest.platform_compatibility,
            _HOST_PLATFORM_VERSION,
        )
        assert compatibility.compatible is True

        package_result = _production_package_result(compatibility=compatibility)
        admission = evaluate_package_production_admission(package_result, compatibility=compatibility)
        assert admission.admitted is True

        capability_result = _production_capability_result(capability_id="reference_prefix_echo")
        require_production_qualification(capability_result)

        bootstrap_catalogs(
            register_shipped=False,
            discover_entry_points=True,
            tool_bundle_ids=["reference_prefix_echo"],
        )

        ctx = ToolWiringContext(extras={"echo_prefix": _EXTRAS_PREFIX})
        registry = build_registry_from_profile(
            ToolProfile(enabled_bundles=["reference_prefix_echo"]),
            ctx=ctx,
        )

        from intergrax_reference_tool_plugin.plugin import (
            REFERENCE_PREFIX_ECHO_TOOL_ID,
            ReferencePrefixEchoInput,
        )

        assert registry.has(REFERENCE_PREFIX_ECHO_TOOL_ID)
        output = _invoke_tool(
            registry=registry,
            tool_id=REFERENCE_PREFIX_ECHO_TOOL_ID,
            input_model=ReferencePrefixEchoInput,
            message="hello",
        )
        assert output.message == f"{_EXTRAS_PREFIX}:hello"
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        for module_name in list(sys.modules):
            if module_name == "intergrax_reference_tool_plugin" or module_name.startswith(
                "intergrax_reference_tool_plugin."
            ):
                del sys.modules[module_name]
        reset_entry_point_spec_cache_for_tests()
        try:
            distribution(_PACKAGE_NAME)
        except PackageNotFoundError:
            pass


def test_local_embedded_extension_end_to_end() -> None:
    local_plugin, module = _load_local_prefix_echo_plugin()
    host_path = str(_LOCAL_PLUGIN_PATH.relative_to(_REPO_ROOT)).replace("\\", "/")

    capability_result = _production_capability_result(
        capability_id="local_prefix_echo",
        host_path=host_path,
    )
    assert capability_result.subject.delivery_source is PluginDeliverySource.HOST_EMBEDDED_EXTENSION
    assert capability_result.subject.host_registration_path == host_path
    assert capability_result.subject.package_name is None
    assert capability_result.subject.entry_point_name is None
    require_production_qualification(capability_result)

    register_tool_plugin(local_plugin)
    ctx = ToolWiringContext(extras={"echo_prefix": _EXTRAS_PREFIX})
    registry = build_registry_from_profile(
        ToolProfile(enabled_bundles=["local_prefix_echo"]),
        ctx=ctx,
    )

    module = sys.modules[local_plugin.__module__]
    tool_id = module.LOCAL_PREFIX_ECHO_TOOL_ID
    input_model = module.LocalPrefixEchoInput

    assert registry.has(tool_id)
    output = _invoke_tool(
        registry=registry,
        tool_id=tool_id,
        input_model=input_model,
        message="local",
    )
    assert output.message == f"{_EXTRAS_PREFIX}:local"


def test_negative_production_gates() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=0.1,<2"),
        _HOST_PLATFORM_VERSION,
    )
    package_result = _production_package_result(compatibility=compatibility)

    missing_compat_admission = evaluate_package_production_admission(package_result, compatibility=None)
    assert missing_compat_admission.admitted is False

    qualified_only = build_qualification_result(
        subject=package_result.subject,
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="compatible but qualified only",
    )
    qualified_admission = evaluate_package_production_admission(qualified_only, compatibility=compatibility)
    assert qualified_admission.admitted is False
    with pytest.raises(ProductionQualificationRequiredError):
        require_production_qualification(qualified_only)

    local_qualified = build_qualification_result(
        subject=build_host_embedded_capability_subject(
            domain="tools",
            capability_id="local_prefix_echo",
            host_registration_path="extensions/local_prefix_echo_plugin.py",
        ),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(),
        reason="host-embedded qualified only",
    )
    with pytest.raises(ProductionQualificationRequiredError):
        require_production_qualification(local_qualified)


def test_external_and_local_share_tool_plugin_contract(tmp_path: Path) -> None:
    local_plugin, _module = _load_local_prefix_echo_plugin()
    assert hasattr(local_plugin, "tool_bundle_manifest")
    assert hasattr(local_plugin, "register_tools")

    wheel_path = _build_reference_wheel(tmp_path)
    install_target = tmp_path / "site-packages"
    install_target.mkdir()
    _install_wheel_to_target(wheel_path, install_target)

    inserted = str(install_target)
    sys.path.insert(0, inserted)
    try:
        pyproject_data = tomllib.loads((_REFERENCE_PKG_DIR / "pyproject.toml").read_text(encoding="utf-8"))
        ep_value = pyproject_data["project"]["entry-points"]["intergrax.tools"][_EP_NAME]
        from intergrax.core.plugins.discovery import load_entry_point_value

        external_target = load_entry_point_value(ep_value)
        assert isinstance(external_target, type)
        assert hasattr(external_target, "tool_bundle_manifest")
        assert hasattr(external_target, "register_tools")

        external_manifest = external_target.tool_bundle_manifest()
        local_manifest = local_plugin.tool_bundle_manifest()
        assert external_manifest.bundle_id != local_manifest.bundle_id
        assert external_manifest.tool_ids != local_manifest.tool_ids
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        for module_name in list(sys.modules):
            if module_name == "intergrax_reference_tool_plugin" or module_name.startswith(
                "intergrax_reference_tool_plugin."
            ):
                del sys.modules[module_name]
