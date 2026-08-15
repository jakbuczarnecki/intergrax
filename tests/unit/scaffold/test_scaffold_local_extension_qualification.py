# © Artur Czarnecki. All rights reserved.

"""Scaffold host-embedded extension qualification gate (PLATFORM-PLUGIN-8 review fix)."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from intergrax.core.plugins import (
    PluginQualificationEvidenceKind,
    build_host_embedded_capability_subject,
    build_qualification_result,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.core.plugins.errors import ProductionQualificationRequiredError
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, is_tool_bundle_registered
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.scaffold.new_application import create_application
from tests.unit.applications.scaffold_runtime_helper import purge_scaffold_package

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LOCAL_BUNDLE_ID = "local_prefix_echo"
_HOST_PATH = "extensions/local_prefix_echo_plugin.py"


def _prepare_tool_wiring_scaffold(
    tmp_path,
    *,
    name: str,
    profile: str,
    port: int,
    route_prefix: str,
) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "applications").mkdir(parents=True, exist_ok=True)
    target = create_application(
        name=name,
        agents=["echo"],
        profile=profile,
        root=root,
        port=port,
        route_prefix=route_prefix,
        full_scaffold=profile == "lab",
    )
    pkg = target.name
    short = pkg.removesuffix("_application")
    apps_dir = str(target.parent)
    import sys

    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    purge_scaffold_package(pkg)
    return target, pkg, short


def _host_embedded_result(*, status: QualificationStatus):
    subject = build_host_embedded_capability_subject(
        domain="tools",
        capability_id=_LOCAL_BUNDLE_ID,
        host_registration_path=_HOST_PATH,
    )
    return build_qualification_result(
        subject=subject,
        status=status,
        evidence=(
            QualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="tools.scaffold.tests.passed",
                ref="tests/unit/scaffold/test_scaffold_local_extension_qualification.py",
            ),
        ),
        reason="scaffold local extension qualification",
    )


@pytest.fixture(autouse=True)
def _clean_tool_catalog() -> Iterator[None]:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def _import_tool_wiring(pkg: str):
    purge_scaffold_package(pkg)
    return importlib.import_module(f"{pkg}.host.tool_wiring")


@pytest.mark.parametrize("profile", ["lab", "product"])
def test_generated_scaffold_exposes_explicit_registration_helper(tmp_path, profile: str) -> None:
    target, pkg, short = _prepare_tool_wiring_scaffold(
        tmp_path,
        name=f"qual_gate_{profile}",
        profile=profile,
        port=8201,
        route_prefix=f"/v1/qual_gate_{profile}",
    )
    assert (target / "extensions" / "local_prefix_echo_plugin.py").is_file()
    tool_wiring_src = (target / "host" / "tool_wiring.py").read_text(encoding="utf-8")
    assert f"def register_{short}_local_tool_extensions" in tool_wiring_src
    assert "require_production_qualification" in tool_wiring_src
    assert "register_tool_plugin(LocalPrefixEchoToolPlugin)" in tool_wiring_src


def test_importing_tool_wiring_does_not_register_local_plugin(tmp_path) -> None:
    _, pkg, _short = _prepare_tool_wiring_scaffold(
        tmp_path,
        name="qual_gate_import",
        profile="lab",
        port=8202,
        route_prefix="/v1/qual_gate_import",
    )
    _import_tool_wiring(pkg)
    assert not is_tool_bundle_registered(_LOCAL_BUNDLE_ID)


def test_production_qualified_result_allows_explicit_registration(tmp_path) -> None:
    _, pkg, short = _prepare_tool_wiring_scaffold(
        tmp_path,
        name="qual_gate_prod",
        profile="lab",
        port=8203,
        route_prefix="/v1/qual_gate_prod",
    )
    tool_wiring = _import_tool_wiring(pkg)
    register_fn = tool_wiring.__dict__[f"register_{short}_local_tool_extensions"]
    register_fn(local_prefix_echo_qualification=_host_embedded_result(status=QualificationStatus.PRODUCTION_QUALIFIED))
    assert is_tool_bundle_registered(_LOCAL_BUNDLE_ID)

    ctx = ToolWiringContext(extras={"echo_prefix": "scaffold"})
    registry = build_registry_from_profile(
        ToolProfile(enabled_bundles=[_LOCAL_BUNDLE_ID]),
        ctx=ctx,
    )
    plugin_mod = importlib.import_module(f"{pkg}.extensions.local_prefix_echo_plugin")
    assert registry.has(plugin_mod.LOCAL_PREFIX_ECHO_TOOL_ID)


def test_qualified_only_result_blocks_scaffold_registration(tmp_path) -> None:
    _, pkg, short = _prepare_tool_wiring_scaffold(
        tmp_path,
        name="qual_gate_block",
        profile="product",
        port=8204,
        route_prefix="/v1/qual_gate_block",
    )
    tool_wiring = _import_tool_wiring(pkg)
    register_fn = tool_wiring.__dict__[f"register_{short}_local_tool_extensions"]
    with pytest.raises(ProductionQualificationRequiredError):
        register_fn(
            local_prefix_echo_qualification=_host_embedded_result(status=QualificationStatus.QUALIFIED),
        )
    assert not is_tool_bundle_registered(_LOCAL_BUNDLE_ID)


def test_wire_tools_composes_qualification_registration_and_materialization(tmp_path) -> None:
    _, pkg, short = _prepare_tool_wiring_scaffold(
        tmp_path,
        name="qual_gate_wire",
        profile="lab",
        port=8205,
        route_prefix="/v1/qual_gate_wire",
    )
    tool_wiring = _import_tool_wiring(pkg)
    wire_fn = tool_wiring.__dict__[f"wire_{short}_tools"]
    wiring = wire_fn(
        local_prefix_echo_qualification=_host_embedded_result(
            status=QualificationStatus.PRODUCTION_QUALIFIED,
        ),
    )
    plugin_mod = importlib.import_module(f"{pkg}.extensions.local_prefix_echo_plugin")
    assert wiring.registry.has(plugin_mod.LOCAL_PREFIX_ECHO_TOOL_ID)
