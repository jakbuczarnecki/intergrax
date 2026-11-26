"""Red-first NT-PLG probes (source gates; no optional SDK imports)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


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


def test_nt_plg_02_tier3_llm_resolver_must_not_construct_adapters_locally() -> None:
    """NT-PLG-02 / FRZ-PLG-02: external structural implementation works without core patch."""
    text = (_REPO / "intergrax/applications/_shared/llm_resolver.py").read_text(encoding="utf-8")
    assert "llm_profile_from_env().create_adapter()" not in text, (
        "NT-PLG-02 external structural implementation works without core patch: "
        "Tier-3 llm_resolver must delegate adapter construction to runtime wiring"
    )
    assert "runtime.wiring import llm_resolver" in text, (
        "NT-PLG-02 external structural implementation works without core patch: "
        "Tier-3 llm_resolver must delegate to canonical runtime wiring"
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

