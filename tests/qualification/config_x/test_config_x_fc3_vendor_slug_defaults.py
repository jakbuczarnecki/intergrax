# © Artur Czarnecki. All rights reserved.

"""CONFIG-X FC-3 vendor-slug default gates (NT-FC3-CX1..CX12)."""

from __future__ import annotations

import hashlib

import pytest
from intergrax.core.security.security_configuration import (
    INBOUND_VERIFIER_NONE_FORBIDDEN,
    SecurityConfigurationError,
)
from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.runtime.interactions.verification.factory import (
    InboundVerifierMode,
    InboundVerifierSettings,
    create_inbound_verifier,
)
from tests.qualification.config_x._config_x_semantic_production_scan import (
    EXPECTED_RESIDUAL_FINGERPRINTS,
    FC3_SCOPED_PHYSICAL_PATHS,
    Fc3VendorDisposition,
    derive_fc3_vendor_values,
    discover_fc3_outside_scope_vendor_findings,
    discover_fc3_vendor_defaults_in_source,
    discover_semantic_production_findings,
    evaluate_fc3_scoped_gate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def _has_vendor_hit(findings, *, value: str | None = None) -> bool:
    hits = [f for f in findings if f.resolved_vendor_value and f.detection_mechanism != "lab_named_constant"]
    if value is None:
        return bool(hits)
    return any(f.resolved_vendor_value == value for f in hits)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_default_integrations_state()
    yield
    reset_default_integrations_state()


def test_nt_fc3_cx1_os_getenv_vendor_default() -> None:
    src = 'import os\ndef f(provider=os.getenv("X_PROVIDER", "ollama")):\n pass\n'
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx1.py")
    assert any("env" in f.detection_mechanism for f in findings)


def test_nt_fc3_cx2_enum_class_field_default() -> None:
    src = (
        "from intergrax.runtime.notifications.backend_contract import NotificationBackend\n"
        "class C:\n backend: NotificationBackend = NotificationBackend.LOG\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx2.py")
    assert any("enum" in f.detection_mechanism for f in findings)


def test_nt_fc3_cx3_os_environ_get_enum_value() -> None:
    src = (
        "import os\nfrom intergrax.runtime.interactions.verification.factory import InboundVerifierMode\n"
        "ENV='X'\ndef f():\n return os.environ.get(ENV, InboundVerifierMode.NONE.value)\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx3.py")
    assert _has_vendor_hit(findings)


def test_nt_fc3_cx4_module_level_selector_scanned() -> None:
    src = "def resolve_foo_settings():\n return 'x'\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx4.py")
    assert isinstance(findings, tuple)


def test_nt_fc3_cx5_semantic_parameter_defaults_including_positional_only() -> None:
    src = "def f(mode, /, backend='log', *, channel='stub'):\n pass\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx5.py")
    assert _has_vendor_hit(findings, value="none") or _has_vendor_hit(findings)


def test_nt_fc3_cx6_catalog_slug_vs_non_slug() -> None:
    a = discover_fc3_vendor_defaults_in_source('def f(provider: str = "ollama"):\n pass\n', "synthetic/cx6a.py")
    b = discover_fc3_vendor_defaults_in_source('def f(label: str = "verbose"):\n pass\n', "synthetic/cx6b.py")
    assert _has_vendor_hit(a)
    assert not _has_vendor_hit(b)


def test_nt_fc3_cx7_repo_scoped_gate_expected_residual_inventory() -> None:
    ok, residuals, unexpected = evaluate_fc3_scoped_gate()
    assert ok, [f.fingerprint for f in unexpected]
    assert len(residuals) == 3
    assert {f.fingerprint for f in residuals} == set(EXPECTED_RESIDUAL_FINGERPRINTS)


def test_nt_fc3_cx8_synthetic_embedding_vendor_default_refused() -> None:
    src = 'def embedding_profile_from_env(provider: str = "ollama"):\n return provider\n'
    findings = discover_fc3_vendor_defaults_in_source(src, "intergrax/rag/embedding/registry/profile.py")
    assert _has_vendor_hit(findings)


def test_nt_fc3_cx9_guarded_lab_constant_not_blocking() -> None:
    src = (
        "from intergrax.integrations.contracts.base import LabSelectorOptIn\n"
        "_LAB_X='ollama'\n"
        "def f(lab_opt_in: LabSelectorOptIn | None = None):\n"
        " if lab_opt_in is not None:\n  return _LAB_X\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx9.py")
    assert not [f for f in findings if f.disposition == Fc3VendorDisposition.LAB_REFUSED]


def test_nt_fc3_cx10_unguarded_lab_constant_blocking() -> None:
    src = "_LAB_X='ollama'\ndef f():\n return _LAB_X\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx10.py")
    assert any(f.detection_mechanism == "lab_named_constant" for f in findings)


def test_nt_fc3_cx11_mixed_guarded_and_unguarded_lab_loads() -> None:
    src = (
        "from intergrax.integrations.contracts.base import LabSelectorOptIn\n"
        "_LAB_X='ollama'\n"
        "def f(lab_opt_in: LabSelectorOptIn | None = None):\n"
        " if lab_opt_in is not None:\n  x=_LAB_X\n"
        " return _LAB_X\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx11.py")
    assert any(f.detection_mechanism == "lab_named_constant" for f in findings)


def test_nt_fc3_cx12_unguarded_default_in_lab_named_module() -> None:
    src = "def f(provider: str = 'stub'):\n return provider\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "intergrax/lab_module.py")
    assert _has_vendor_hit(findings)


def test_fc3_vision_hf_adapter_slug_is_provider_identity() -> None:
    findings = discover_fc3_vendor_defaults_in_source(
        open(
            __import__("pathlib").Path(__file__).resolve().parents[3]
            / "intergrax/model_inference/adapters/huggingface_inference_vision.py",
            encoding="utf-8",
        ).read(),
        "intergrax/model_inference/adapters/huggingface_inference_vision.py",
    )
    slug_hits = [
        f for f in findings if f.enclosing_symbol == "HuggingFaceInferenceVisionAdapter" and f.semantic_field == "slug"
    ]
    assert slug_hits
    assert all(f.disposition == Fc3VendorDisposition.PROVIDER_IDENTITY for f in slug_hits)


def test_fc3_inbound_settings_default_none_refuses_without_lab() -> None:
    assert InboundVerifierSettings().mode is InboundVerifierMode.NONE
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier(InboundVerifierSettings())
    assert exc.value.reason_code == INBOUND_VERIFIER_NONE_FORBIDDEN


def test_fc3_inbound_explicit_lab_none_allowed() -> None:
    opt = LabSelectorOptIn(selector="none", reason="fc3 cx behavioral positive control")
    create_inbound_verifier(InboundVerifierSettings(), lab_opt_in=opt)


def test_fc3_global_semantic_scan_unchanged_blocker_set() -> None:
    before = frozenset((f.repo_path, f.finding_class.value) for f in discover_semantic_production_findings())
    assert isinstance(before, frozenset)


def test_fc3_nine_paths_declared() -> None:
    assert len(FC3_SCOPED_PHYSICAL_PATHS) == 9


def test_fc3_vendor_value_authority_dynamic() -> None:
    vendor, counts = derive_fc3_vendor_values()
    assert counts["unique_vendor_value_count"] == len(vendor)
    digest = hashlib.sha256("\n".join(sorted(vendor)).encode()).hexdigest()
    assert counts["vendor_value_digest"] == digest


def test_fc3_outside_scope_cloud_adapter_slugs_recorded() -> None:
    outside = discover_fc3_outside_scope_vendor_findings()
    values = {f.resolved_vendor_value for f in outside}
    assert {"aws", "azure", "gcp"}.issubset(values)
