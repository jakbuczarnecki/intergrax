"""Red-first NT-PLG probes (source gates; no optional SDK imports)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


class _NtPlg02ExternalStructuralPlugin:
    """Proof-only plugin type defined outside the intergrax package tree."""

    marker = "nt-plg-02-external"


def test_nt_plg_01_evaluation_wiring_exposes_tenant_scoped_registry_contract() -> None:
    """NT-PLG-01 / FRZ-PLG-01: extensible mechanisms expose platform contracts."""
    text = (_REPO / "intergrax/applications/_shared/evaluation_wiring.py").read_text(encoding="utf-8")
    assert "TenantScopedOnlineEvaluationRegistry" in text, (
        "NT-PLG-01 extensible mechanisms expose platform contracts: "
        "evaluation wiring must bind TenantScopedOnlineEvaluationRegistry for tenant-scoped hosts"
    )
    assert "tenant_id: str | None = None" in text, (
        "NT-PLG-01 extensible mechanisms expose platform contracts: "
        "wire_application_evaluation must accept tenant_id for registry scoping"
    )


def test_nt_plg_02_external_structural_plugin_registers_without_core_patch() -> None:
    """NT-PLG-02 / FRZ-PLG-02: external structural implementation works without core patch."""
    from intergrax.core.plugins.discovery import load_plugin_types, register_plugins

    registered: list[type] = []
    group = "intergrax.nt_plg_02_probe"
    count = register_plugins(
        group,
        lambda plugin_type: registered.append(plugin_type) or True,
        explicit=[_NtPlg02ExternalStructuralPlugin],
    )
    types = load_plugin_types(group, explicit=[_NtPlg02ExternalStructuralPlugin])
    assert count == 1, (
        "NT-PLG-02 external structural implementation works without core patch: "
        "register_plugins must accept an explicit external plugin class"
    )
    assert _NtPlg02ExternalStructuralPlugin in types, (
        "NT-PLG-02 external structural implementation works without core patch: "
        "load_plugin_types must surface the explicit external structural class"
    )
    assert registered == [_NtPlg02ExternalStructuralPlugin], (
        "NT-PLG-02 external structural implementation works without core patch: "
        "register_plugins must invoke the registration callback for the external class"
    )


def test_nt_plg_06_strict_manifest_gate_must_not_downgrade_product_to_lab() -> None:
    """NT-PLG-06 / FRZ-PLG-06: plugin package is not capability authority."""
    text = (_REPO / "intergrax/applications/_shared/strict_product_manifest_ci_gates.py").read_text(encoding="utf-8")
    assert "_gate_wiring_environment" not in text, (
        "NT-PLG-06 plugin package is not capability authority: "
        "STRICT product capability graph gate must not downgrade PRODUCT hosts to LAB integration profile"
    )
    assert "package_gate_qualification" in text, (
        "NT-PLG-06 plugin package is not capability authority: "
        "STRICT product gates must use package_gate_qualification for admission-qualified wiring"
    )


def test_nt_plg_07_capability_payload_must_not_exceed_declared_contract_after_policy() -> None:
    """NT-PLG-07 / FRZ-PLG-07: runtime extensions cannot self-expand authority."""
    from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
    from intergrax.runtime.nexus.tools.tool_runtime import (
        capability_payload_to_tool_ids,
        tool_invocation_plan_from_capability_payload,
    )

    declared = ("registry.contracted.tool",)
    extra = "runtime.self.granted.extra"
    payload = {"tool_ids": [declared[0], extra], "use_tools": True}
    ids = capability_payload_to_tool_ids(payload)
    assert extra in ids, (
        "NT-PLG-07 runtime extensions cannot self-expand authority: "
        "capability_payload_to_tool_ids must be exercised on the payload under test"
    )
    plan = tool_invocation_plan_from_capability_payload(payload)
    bounded = ToolAccessPolicy.apply(plan, allowed_tools=declared)
    assert extra not in bounded.tool_ids, (
        "NT-PLG-07 runtime extensions cannot self-expand authority: "
        "tool_invocation_plan_from_capability_payload plus ToolAccessPolicy "
        "must refuse tool ids outside the declared registry contract"
    )
