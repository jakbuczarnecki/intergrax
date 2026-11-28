# © Artur Czarnecki. All rights reserved.

"""CONFIG-X FC-3 vendor-slug default gates (NT-FC3-CX1..CX12)."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from intergrax.core.security.security_configuration import (
    INBOUND_VERIFIER_NONE_FORBIDDEN,
    SecurityConfigurationError,
)
from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.runtime.interactions.verification.factory import (
    InboundVerifierMode,
    InboundVerifierSettings,
    create_inbound_verifier,
)
from tests.qualification.config_x import _config_x_semantic_production_scan as scan
from tests.qualification.config_x._config_x_semantic_production_scan import (
    EXPECTED_RESIDUAL_FINGERPRINTS,
    FC3_SCOPED_PHYSICAL_PATHS,
    Fc3VendorDisposition,
    derive_fc3_vendor_values,
    discover_fc3_outside_scope_vendor_findings,
    discover_fc3_vendor_defaults_in_source,
    discover_fc3_vendor_defaults_inventory,
    discover_semantic_production_findings,
    evaluate_fc3_scoped_gate,
    evaluate_fc3_scoped_gate_from_inventory,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def _has_vendor_hit(findings, *, value: str | None = None) -> bool:
    hits = [f for f in findings if f.resolved_vendor_value and f.detection_mechanism != "lab_named_constant"]
    if value is None:
        return bool(hits)
    return any(f.resolved_vendor_value == value for f in hits)


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
    ) + "ENV='X'\ndef f():\n return os.environ.get(ENV, InboundVerifierMode.NONE.value)\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx3.py")
    assert _has_vendor_hit(findings)


def test_nt_fc3_cx4_module_level_selector_scanned() -> None:
    src = "import os\ndef resolve_settings():\n provider = os.getenv('X_PROVIDER', 'ollama')\n return provider\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx4.py")
    assert [
        (f.enclosing_symbol, f.semantic_field, f.resolved_vendor_value, f.detection_mechanism) for f in findings
    ] == [("resolve_settings", "provider", "ollama", "env_get_default:string_literal")]


def test_nt_fc3_cx5_semantic_parameter_defaults_including_positional_only() -> None:
    src = "def f(mode='none', /, backend='log', *, channel='stub'):\n pass\ndef required(provider, /):\n pass\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/cx5.py")
    assert {(f.enclosing_symbol, f.semantic_field, f.resolved_vendor_value) for f in findings} == {
        ("f", "mode", "none"),
        ("f", "backend", "log"),
        ("f", "channel", "stub"),
    }


def test_nt_fc3_cx6_catalog_slug_vs_non_slug() -> None:
    a = discover_fc3_vendor_defaults_in_source('def f(provider: str = "ollama"):\n pass\n', "synthetic/cx6a.py")
    b = discover_fc3_vendor_defaults_in_source(
        'def f(provider: str = "not-a-registered-vendor"):\n pass\n', "synthetic/cx6b.py"
    )
    assert _has_vendor_hit(a)
    assert not _has_vendor_hit(b)


def test_nt_fc3_cx7_repo_scoped_gate_expected_residual_inventory() -> None:
    ok, residuals, unexpected = evaluate_fc3_scoped_gate()
    assert ok, [f.fingerprint for f in unexpected]
    assert len(residuals) == 3
    assert {f.fingerprint for f in residuals} == set(EXPECTED_RESIDUAL_FINGERPRINTS)


def test_nt_fc3_cx8_synthetic_embedding_vendor_default_refused() -> None:
    path = "intergrax/rag/embedding/registry/profile.py"
    original = (scan.repo_root() / path).read_text(encoding="utf-8")
    mutated = original + '\ndef mutation(provider="ollama"):\n return provider\n'
    inventory = tuple(f for f in discover_fc3_vendor_defaults_inventory() if f.repo_path != path)
    inventory += discover_fc3_vendor_defaults_in_source(mutated, path)
    ok, _, unexpected = evaluate_fc3_scoped_gate_from_inventory(inventory)
    assert ok is False
    assert any(f.enclosing_symbol == "mutation" and f.semantic_field == "provider" for f in unexpected)


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
    imported = discover_fc3_vendor_defaults_in_source(
        "from intergrax.rag.embedding.registry.provider_authority import _LAB_EMBEDDING_PROVIDER\n",
        "synthetic/cx11_import.py",
    )
    assert any("cross_module" in f.detail for f in imported)


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
    path = "tests/qualification/config_x/_config_x_semantic_production_scan.py"
    baseline = subprocess.run(
        ["git", "show", f"9efe643045c40255ce868e99400e9468d2fee4f4:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    old = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(baseline).body
        if isinstance(n, ast.FunctionDef)
    }
    current = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse((scan.repo_root() / path).read_text(encoding="utf-8-sig")).body
        if isinstance(n, ast.FunctionDef)
    }
    assert old
    assert {name: current[name] for name in old} == old
    discover_semantic_production_findings.cache_clear()
    assert scan.discover_semantic_i_blocker_paths() == frozenset()
    assert scan.discover_named_constant_semantic_blind_spot_paths() == frozenset()
    assert scan.discover_unclassified_provider_surface_paths() == frozenset()


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


def test_fc3_inbound_lab_none_returns_null_verifier() -> None:
    from intergrax.runtime.interactions.verification.contract import NullInboundRequestVerifier

    opt = LabSelectorOptIn(selector="none", reason="fc3 corrective lab positive control")
    verifier = create_inbound_verifier(InboundVerifierSettings(), lab_opt_in=opt)
    assert isinstance(verifier, NullInboundRequestVerifier)


def test_fc3_f01_lab_guard_else_branch_not_exempt() -> None:
    src = (
        "from intergrax.integrations.contracts.base import LabSelectorOptIn\n"
        "_LAB_X='ollama'\n"
        "def f(lab_opt_in: LabSelectorOptIn | None = None):\n"
        " if lab_opt_in is not None:\n"
        "  x=_LAB_X\n"
        " else:\n"
        "  return _LAB_X\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/f01.py")
    assert any(f.detection_mechanism == "lab_named_constant" for f in findings)


def test_fc3_f01_lab_param_requires_structural_annotation() -> None:
    src = "_LAB_X='ollama'\ndef f(lab_opt_in=None):\n if lab_opt_in is not None:\n  return _LAB_X\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/f01b.py")
    assert any(f.detection_mechanism == "lab_named_constant" for f in findings)


def test_fc3_f02_validation_only_by_ast_not_path_suffix() -> None:
    src = (
        "from intergrax.integrations.contracts.base import LabSelectorOptIn\n"
        "_LAB_Y='inmemory'\n"
        "def resolve_slug(lab_opt_in: LabSelectorOptIn) -> str:\n"
        " slug = lab_opt_in.selector.strip().lower()\n"
        " if slug != _LAB_Y:\n"
        "  raise ValueError(f'bad {slug!r} expected {_LAB_Y!r}')\n"
        " return slug\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/f02.py")
    lab_refs = [f for f in findings if "_LAB_Y" in f.expression or f.semantic_field == "_LAB_Y"]
    assert lab_refs
    assert all(f.disposition == Fc3VendorDisposition.VALIDATION_ONLY for f in lab_refs)


def test_fc3_f03_vision_identity_requires_import_binding() -> None:
    src = (
        "class VisionInferenceAdapterTypos:\n pass\n"
        "class Fake(VisionInferenceAdapterTypos):\n slug = 'huggingface_inference'\n"
    )
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/f03.py")
    slug_hits = [f for f in findings if f.semantic_field == "slug"]
    assert slug_hits
    assert all(f.disposition != Fc3VendorDisposition.PROVIDER_IDENTITY for f in slug_hits)


def test_fc3_f04_gate_multiset_from_inventory() -> None:
    inventory = discover_fc3_vendor_defaults_inventory()
    ok, residuals, unexpected = evaluate_fc3_scoped_gate_from_inventory(inventory)
    assert ok, [f.fingerprint for f in unexpected]
    assert Counter(f.fingerprint for f in residuals) == Counter(EXPECTED_RESIDUAL_FINGERPRINTS)


def test_fc3_f05_cross_module_lab_import_flagged() -> None:
    src = "from intergrax.rag.embedding.registry.provider_authority import _LAB_EMBEDDING_PROVIDER\n"
    findings = discover_fc3_vendor_defaults_in_source(src, "synthetic/f05.py")
    assert any(
        "cross_module" in (f.detail or "") or getattr(f, "detection_mechanism", "") == "lab_named_constant"
        for f in findings
    )


def test_fc3_f07_derive_vendor_values_does_not_pollute_catalog() -> None:
    from intergrax.integrations.registry.catalog import list_slugs

    before = tuple(sorted(list_slugs()))
    derive_fc3_vendor_values()
    after = tuple(sorted(list_slugs()))
    assert before == after


def test_fc3_f08_scan_context_isolation_between_modules() -> None:
    a = discover_fc3_vendor_defaults_in_source(
        ("from enum import StrEnum\nclass Mode(StrEnum):\n A='ollama'\ndef f(mode: str = Mode.A.value):\n pass\n"),
        "synthetic/f08a.py",
    )
    b = discover_fc3_vendor_defaults_in_source(
        ("from enum import StrEnum\nclass Mode(StrEnum):\n B='stub'\ndef f(mode: str = Mode.B.value):\n pass\n"),
        "synthetic/f08b.py",
    )
    assert any(f.resolved_vendor_value == "ollama" for f in a)
    assert any(f.resolved_vendor_value == "stub" for f in b)


LAB_IMPORT = "from intergrax.integrations.contracts.base import LabSelectorOptIn\n"
LAB_BODY = " if lab_opt_in is not None:\n  return _LAB_X\n"


@pytest.mark.parametrize(
    ("prefix", "annotation", "body", "exempt"),
    [
        (LAB_IMPORT, "LabSelectorOptIn | None", LAB_BODY, True),
        (LAB_IMPORT, "LabSelectorOptIn | None", " if lab_opt_in:\n  pass\n else:\n  return _LAB_X\n", False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " return _LAB_X if lab_opt_in else None\n", True),
        (LAB_IMPORT, "LabSelectorOptIn | None", " return None if lab_opt_in else _LAB_X\n", False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " return True and lab_opt_in and _LAB_X\n", True),
        (LAB_IMPORT, "LabSelectorOptIn | None", " return _LAB_X and lab_opt_in\n", False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " if lab_opt_in:\n  def nested():\n   return _LAB_X\n", False),
        (LAB_IMPORT, "list[LabSelectorOptIn]", LAB_BODY, False),
        (LAB_IMPORT, "LabSelectorOptIn | str", LAB_BODY, False),
        ("class LabSelectorOptIn: pass\n", "LabSelectorOptIn | None", LAB_BODY, False),
        ("import unrelated as custom\n", "custom.LabSelectorOptIn | None", LAB_BODY, False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " lab_opt_in = object()\n" + LAB_BODY, False),
        ("from intergrax.integrations.contracts.base import LabSelectorOptIn as Opt\n", "Opt | None", LAB_BODY, True),
        (LAB_IMPORT, "LabSelectorOptIn", LAB_BODY, False),
        (LAB_IMPORT + "import typing\n", "typing.Optional[LabSelectorOptIn]", LAB_BODY, True),
        (LAB_IMPORT + "from typing import Optional\n", "Optional[LabSelectorOptIn]", LAB_BODY, True),
        (LAB_IMPORT, "None | LabSelectorOptIn", LAB_BODY, True),
        (LAB_IMPORT, "dict[str, LabSelectorOptIn]", LAB_BODY, False),
        ("", "LabSelectorOptIn | None", LAB_BODY, False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " if lab_opt_in:\n  lab_opt_in = None\n  return _LAB_X\n", False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " if lab_opt_in:\n  return lambda: _LAB_X\n", False),
        (LAB_IMPORT, "LabSelectorOptIn | None", " if lab_opt_in:\n  class Nested:\n   slug = _LAB_X\n", False),
    ],
    ids=[f"LAB-{i:02}" for i in range(1, 14)]
    + [
        "LAB-BARE",
        "LAB-TYPING",
        "LAB-OPTIONAL",
        "LAB-NONE-FIRST",
        "LAB-DICT",
        "LAB-UNBOUND",
        "LAB-REASSIGN-USE",
        "LAB-LAMBDA",
        "LAB-CLASS",
    ],
)
def test_corrective_lab(prefix: str, annotation: str, body: str, exempt: bool) -> None:
    source = prefix + "_LAB_X='ollama'\n" + f"def f(lab_opt_in: {annotation} = None):\n" + body
    hits = [
        f
        for f in discover_fc3_vendor_defaults_in_source(source, "synthetic/lab.py")
        if f.detection_mechanism == "lab_named_constant"
    ]
    assert (not hits) is exempt, [(f.expression, f.detail) for f in hits]


VALIDATION = (
    " slug = lab_opt_in.selector.strip().lower()\n"
    " if slug != _LAB_X:\n"
    "  raise ValueError(f'expected {_LAB_X!r}')\n"
    " return slug\n"
)


@pytest.mark.parametrize(
    ("body", "prefix", "validation"),
    [
        (VALIDATION, LAB_IMPORT, True),
        (VALIDATION, LAB_IMPORT, True),
        (VALIDATION.replace("return slug", "return _LAB_X"), LAB_IMPORT, False),
        (VALIDATION.replace("return slug", "slug = _LAB_X\n return slug"), LAB_IMPORT, False),
        (VALIDATION.replace("raise ValueError", "provider = build(_LAB_X)\n  raise ValueError"), LAB_IMPORT, False),
        (VALIDATION.replace("raise ValueError(f'expected {_LAB_X!r}')", "pass"), LAB_IMPORT, False),
        (" return f'expected {_LAB_X}'\n", LAB_IMPORT, False),
        (VALIDATION.replace("raise ValueError", "ValueError"), LAB_IMPORT, False),
        (VALIDATION.replace("lab_opt_in.selector", "other.selector"), LAB_IMPORT, False),
        (VALIDATION.replace("return slug", "slug = 'ollama'\n return slug"), LAB_IMPORT, False),
        (VALIDATION, "class LabSelectorOptIn: pass\n", False),
    ],
    ids=[f"VAL-{i:02}" for i in range(1, 12)],
)
def test_corrective_validation(body: str, prefix: str, validation: bool, request: pytest.FixtureRequest) -> None:
    source = prefix + "_LAB_X='inmemory'\ndef f(lab_opt_in: LabSelectorOptIn):\n" + body
    hits = [
        f
        for f in discover_fc3_vendor_defaults_in_source(source, "synthetic/validation.py")
        if f.detection_mechanism == "lab_named_constant"
    ]
    assert hits
    if validation:
        line = 5 if request.node.callspec.id == "VAL-01" else 6
        assert any(f.source_line == line and f.disposition == Fc3VendorDisposition.VALIDATION_ONLY for f in hits)
        assert all(f.disposition == Fc3VendorDisposition.VALIDATION_ONLY for f in hits)
    else:
        assert all(f.disposition == Fc3VendorDisposition.LAB_REFUSED for f in hits), hits


VISION_IMPORT = "from intergrax.model_inference.contracts import VisionInferenceAdapter\n"


@pytest.mark.parametrize(
    ("source", "identity"),
    [
        ("@hf", True),
        ("@triton", True),
        (
            (
                "from intergrax.model_inference.contracts import VisionInferenceAd"
                "apter as V\nclass C(V):\n slug='vision_serving'\n"
            ),
            True,
        ),
        (
            (
                "import intergrax.model_inference.contracts as c\nclass C(c.VisionI"
                "nferenceAdapter):\n slug='vision_serving'\n"
            ),
            True,
        ),
        (("class VisionInferenceAdapter: pass\nclass C(VisionInferenceAdapter):\n slug='vision_serving'\n"), False),
        ("import custom\nclass C(custom.VisionInferenceAdapter):\n slug='vision_serving'\n", False),
        (
            (
                "from unrelated.module import VisionInferenceAdapter\nclass C(Visio"
                "nInferenceAdapter):\n slug='vision_serving'\n"
            ),
            False,
        ),
        (
            VISION_IMPORT + "VisionInferenceAdapter=object\nclass C(VisionInferenceAdapter):\n slug='vision_serving'\n",
            False,
        ),
        (VISION_IMPORT + "VALUE='vision_serving'\nclass C(VisionInferenceAdapter):\n slug=VALUE\n", False),
        (VISION_IMPORT + "class C(VisionInferenceAdapter):\n def f(self, provider='ollama'): pass\n", False),
    ],
    ids=[f"ID-{i:02}" for i in range(1, 11)],
)
def test_corrective_identity(source: str, identity: bool) -> None:
    rel = "synthetic/identity.py"
    if source.startswith("@"):
        name = "huggingface_inference_vision" if source == "@hf" else "triton_vision"
        rel = f"intergrax/model_inference/adapters/{name}.py"
        source = (scan.repo_root() / rel).read_text(encoding="utf-8")
    hits = discover_fc3_vendor_defaults_in_source(source, rel)
    assert hits
    assert any(f.disposition == Fc3VendorDisposition.PROVIDER_IDENTITY for f in hits) is identity
    if "provider=" in source:
        assert any(f.semantic_field == "provider" and f.resolved_vendor_value == "ollama" for f in hits)


@pytest.mark.parametrize("case", range(1, 8), ids=[f"MULTI-{i:02}" for i in range(1, 8)])
def test_corrective_multiset(case: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inventory = discover_fc3_vendor_defaults_inventory()
    residual = next(f for f in inventory if f.fingerprint == EXPECTED_RESIDUAL_FINGERPRINTS[0])
    if case == 2:
        inventory = tuple(f for f in inventory if f is not residual)
    elif case == 3:
        inventory += (residual,)
    elif case == 4:
        inventory = tuple(replace(f, normalized_expression="altered") if f is residual else f for f in inventory)
    elif case == 5:
        path = FC3_SCOPED_PHYSICAL_PATHS[1]
        inventory += discover_fc3_vendor_defaults_in_source("def f(provider='ollama'): pass\n", path)
    elif case in (6, 7):
        path = FC3_SCOPED_PHYSICAL_PATHS[1]
        if case == 7:
            target = tmp_path / path
            target.parent.mkdir(parents=True)
            target.write_text("def broken(:\n", encoding="utf-8")
        with monkeypatch.context() as m:
            m.setattr(scan, "repo_root", lambda: tmp_path)
            mutated = scan.discover_fc3_vendor_defaults_for_path(path)
        inventory = tuple(f for f in inventory if f.repo_path != path) + mutated
    ok, residuals, _ = evaluate_fc3_scoped_gate_from_inventory(inventory)
    assert ok is (case == 1)
    if case == 1:
        assert Counter(f.fingerprint for f in residuals) == Counter(EXPECTED_RESIDUAL_FINGERPRINTS)


@pytest.mark.parametrize("case", range(1, 9), ids=[f"LABIMPORT-{i:02}" for i in range(1, 9)])
def test_corrective_lab_imports(case: int, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = {"some.module": "_LAB_VENDOR='ollama'\nraise RuntimeError('must never execute')\n"}
    source = {
        1: "from some.module import _LAB_VENDOR\n",
        2: "from some.module import _LAB_VENDOR as alias\n",
        3: "import some.module as module\nx=module._LAB_VENDOR\n",
        4: "def f():\n from some.module import _LAB_VENDOR as local\n return local\n",
        5: "from missing.source import _LAB_UNKNOWN\n",
        6: "_LAB_VENDOR='ollama'\n__all__=['_LAB_VENDOR']\n",
        7: "from missing.source import _LAB_UNKNOWN\n",
        8: LAB_IMPORT
        + (
            "_LAB_VENDOR='ollama'\ndef f(lab_opt_in: LabSelectorOptIn | None):\n"
            " if lab_opt_in:\n  x=_LAB_VENDOR\n return _LAB_VENDOR\n"
        ),
    }[case]
    hits = discover_fc3_vendor_defaults_in_source(source, "synthetic/lab_import.py", module_sources=sources)
    assert len(hits) == 1, hits
    assert hits[0].disposition == Fc3VendorDisposition.LAB_REFUSED
    if case in (1, 2, 3, 4):
        assert hits[0].resolved_vendor_value == "ollama"
        assert "cross_module" in hits[0].detail
    elif case in (5, 7):
        assert hits[0].resolved_vendor_value == ""
        assert "unresolved" in hits[0].detail
    if case == 7:
        monkeypatch.setattr(scan, "discover_fc3_vendor_defaults_inventory", lambda: hits)
        assert scan.discover_fc3_outside_scope_vendor_findings() == hits


@pytest.mark.parametrize("case", range(1, 8), ids=[f"CAT-{i:02}" for i in range(1, 8)])
def test_corrective_catalog(case: int) -> None:
    script = """
import subprocess
from unittest.mock import patch
from intergrax.integrations.contracts.base import IntegrationEntry
from intergrax.integrations.registry import bootstrap, catalog
from tests.qualification.config_x import _config_x_semantic_production_scan as scan
catalog.register_integration(IntegrationEntry(slug="custom-corrective", categories=(), factory=lambda **kw: None))
bootstrap._BOOTSTRAPPED = True
before = catalog.catalog_snapshot()
with catalog.catalog_state_lock():
    revision = catalog.read_catalog_revision_under_lock()
flag = bootstrap._BOOTSTRAPPED
if CASE == 5:
    with patch.object(scan.subprocess, "run", side_effect=subprocess.CalledProcessError(9, ["child"])):
        try:
            scan.derive_fc3_vendor_values()
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError("failure was swallowed")
else:
    first = scan.derive_fc3_vendor_values()
    second = scan.derive_fc3_vendor_values()
    assert first == second
assert catalog.get_entry("custom-corrective") is before["custom-corrective"]
assert catalog.catalog_snapshot() == before
assert all(catalog.catalog_snapshot()[key] is value for key, value in before.items())
with catalog.catalog_state_lock():
    assert catalog.read_catalog_revision_under_lock() == revision
assert bootstrap._BOOTSTRAPPED is flag
"""
    subprocess.run(
        [sys.executable, "-c", f"CASE={case}\n" + script],
        cwd=scan.repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    if case == 7:
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8-sig"))
        assert not any(isinstance(n, ast.FunctionDef) and n.name == "_reset_registry" for n in tree.body)
        # Real test order in the child, with custom parent state already present.
        ordered = script.replace(
            "first = scan.derive_fc3_vendor_values()",
            (
                "test_nt_fc3_cx7_repo_scoped_gate_expected_residual_inventory()\n  "
                "  test_fc3_vendor_value_authority_dynamic()\n    first = scan.deri"
                "ve_fc3_vendor_values()"
            ),
        )
        ordered = (
            "from tests.qualification.config_x.test_config_x_fc3_vendor_slug_d"
            "efaults import test_nt_fc3_cx7_repo_scoped_gate_expected_residual"
            "_inventory, test_fc3_vendor_value_authority_dynamic\n"
        ) + ordered
        subprocess.run(
            [sys.executable, "-c", "CASE=7\n" + ordered],
            cwd=scan.repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "import os\ndef f(provider=os.getenv('X', 'ollama')): pass\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os\ndef f(provider=os.getenv('X', default='ollama')): pass\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os\ndef f(provider=os.environ.get('X', 'ollama')): pass\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os as env\ndef f(provider=env.getenv('X', 'ollama')): pass\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "from os import getenv as read\ndef f(provider=read('X', 'ollama')): pass\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os\ndef f():\n provider=(os.getenv('X') or os.getenv('Y')) or 'ollama'\n",
            [("f", "provider", "ollama", "env_or_fallback:string_literal")],
        ),
        (
            "import os\ndef f():\n provider=os.getenv('X').strip().lower() or 'ollama'\n",
            [("f", "provider", "ollama", "env_or_fallback:string_literal")],
        ),
        (
            "import os\nclass C:\n provider=os.getenv('X', 'ollama')\n",
            [("C", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os\nprovider=os.getenv('X', 'ollama')\n",
            [("<module>", "provider", "ollama", "env_get_default:string_literal")],
        ),
        (
            "import os\nasync def f():\n provider=os.getenv('X', 'ollama')\n",
            [("f", "provider", "ollama", "env_get_default:string_literal")],
        ),
        ("def f(mode='none', /): pass\n", [("f", "mode", "none", "positional_default:string_literal")]),
        ("def f(*, channel='stub'): pass\n", [("f", "channel", "stub", "keyword_only_default:string_literal")]),
        ("import os\ndef f(value):\n provider=(wrap(os.getenv('X')) or (value or 'ollama'))\n", []),
        ("import custom\ndef f():\n provider=custom.environ.get('X','ollama')\n", []),
        (("from unrelated.module import InboundVerifierMode\ndef f(mode=InboundVerifierMode.NONE): pass\n"), []),
        (
            (
                "from intergrax.runtime.interactions.verification.factory import I"
                "nboundVerifierMode as M\ndef f(mode=M.NONE): pass\n"
            ),
            [("f", "mode", "none", "positional_default:enum_member")],
        ),
        (
            "from enum import StrEnum\nclass Mode(StrEnum):\n A='ollama'\ndef f(mode=Mode.A): pass\n",
            [("f", "mode", "ollama", "positional_default:local_enum_member")],
        ),
        (
            (
                "import os\ndef f(*, backend=os.getenv('X','log')):\n channel=os.get"
                "env('Y','stub')\n return os.getenv('Z','none')\n"
            ),
            [
                ("f", "backend", "log", "env_get_default:string_literal"),
                ("f", "channel", "stub", "env_get_default:string_literal"),
                ("f", "<return:4>", "none", "env_get_default:string_literal"),
            ],
        ),
        (
            "import os\ndef outer():\n def inner():\n  provider=os.getenv('X','ollama')\n",
            [("outer.inner", "provider", "ollama", "env_get_default:string_literal")],
        ),
    ],
    ids=[f"ENV-{i:02}" for i in range(1, 20)],
)
def test_corrective_environment(
    source: str, expected: list[tuple[str, str, str, str]], request: pytest.FixtureRequest
) -> None:
    hits = discover_fc3_vendor_defaults_in_source(source, "synthetic/environment.py")
    actual = [(f.enclosing_symbol, f.semantic_field, f.resolved_vendor_value, f.detection_mechanism) for f in hits]
    assert sorted(actual) == sorted(expected)
    if request.node.callspec.id == "ENV-17":
        second = discover_fc3_vendor_defaults_in_source(source.replace("A='ollama'", "B='stub'"), "synthetic/second.py")
        assert second == (), second
