# © Artur Czarnecki. All rights reserved.

"""AC-6 Phase 5 architecture regression gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DISTRIBUTION_DIR = REPO_ROOT / "intergrax" / "agent_distribution"
ARCHITECTURE_DOC = (
    REPO_ROOT / "docs" / "project" / "architecture" / "AGENT_DISTRIBUTION.md"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EMERGENCY_MODULE = AGENT_DISTRIBUTION_DIR / "emergency_revocation_response.py"
_TRUST_MODULE = AGENT_DISTRIBUTION_DIR / "package_trust.py"
_ADMIN_MODULE = AGENT_DISTRIBUTION_DIR / "admin_service.py"
_DYNAMIC_ACQUISITION_MODULE = AGENT_DISTRIBUTION_DIR / "dynamic_acquisition.py"

_FORBIDDEN_EMERGENCY_ATTR_CALLS: tuple[tuple[str, str], ...] = (
    ("serving_store", "persist"),
    ("revision_store", "persist"),
    ("deployment_instance_store", "update"),
    ("deployment_adapter", "stop"),
)

_FORBIDDEN_EMERGENCY_NAMES = frozenset({"AgentRegistry"})

_TRUST_BOUNDARY_MODULES = (
    _ADMIN_MODULE,
    _DYNAMIC_ACQUISITION_MODULE,
    _EMERGENCY_MODULE,
)

_CRYPTO_PRIMITIVE_MARKERS = (
    "Ed25519PrivateKey",
    "Ed25519PublicKey",
    "cryptography.hazmat",
)

_SUSPICIOUS_TRUST_ENGINE_PATTERNS = (
    re.compile(r"\bTrustManager\b"),
    re.compile(r"\bTrustEngine\b"),
    re.compile(r"\bCertificationService\b"),
    re.compile(r"\bSecurityTrustService\b"),
    re.compile(r"\ballow_trusted\b"),
    re.compile(r"\btrusted\s*=\s*True\b"),
    re.compile(r"\bcertified\s*=\s*True\b"),
)


def _attribute_call_targets(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        targets.append((node.func.value.id, node.func.attr))
    return targets


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_emergency_revocation_service_has_no_direct_lifecycle_store_mutation() -> None:
    targets = _attribute_call_targets(_EMERGENCY_MODULE)
    violations = [
        f"{obj}.{attr}"
        for obj, attr in targets
        if any(
            obj == forbidden_obj and attr.startswith(forbidden_attr)
            for forbidden_obj, forbidden_attr in _FORBIDDEN_EMERGENCY_ATTR_CALLS
        )
    ]
    assert not violations, f"forbidden emergency lifecycle mutations: {violations}"


def test_emergency_revocation_service_has_no_registry_coupling() -> None:
    text = _EMERGENCY_MODULE.read_text(encoding="utf-8")
    imported = _imported_names(_EMERGENCY_MODULE)
    violations = sorted(
        name for name in _FORBIDDEN_EMERGENCY_NAMES if name in imported or name in text
    )
    assert not violations, f"forbidden emergency registry coupling: {violations}"


def test_trust_boundary_modules_delegate_to_package_trust_coordinator() -> None:
    admin_text = _ADMIN_MODULE.read_text(encoding="utf-8")
    emergency_text = _EMERGENCY_MODULE.read_text(encoding="utf-8")
    dynamic_text = _DYNAMIC_ACQUISITION_MODULE.read_text(encoding="utf-8")
    assert "AgentPackageTrustCoordinator" in admin_text
    assert "assert_install_admission" in admin_text
    assert "AgentPackageTrustCoordinator" in emergency_text
    assert (
        "assert_install_admission" in emergency_text
        or "package_trust_coordinator" in emergency_text
    )
    assert "install_agent" in dynamic_text
    assert "trust_record" in dynamic_text


def test_package_trust_coordinator_has_no_direct_crypto_primitives() -> None:
    text = _TRUST_MODULE.read_text(encoding="utf-8")
    modules = _imported_modules(_TRUST_MODULE)
    violations = [
        marker
        for marker in _CRYPTO_PRIMITIVE_MARKERS
        if marker in text or any(marker in module for module in modules)
    ]
    assert not violations, (
        f"trust coordinator must not embed crypto primitives: {violations}"
    )


def test_bounded_production_surfaces_have_no_duplicate_trust_engine_names() -> None:
    surfaces = (
        AGENT_DISTRIBUTION_DIR / "package_trust.py",
        AGENT_DISTRIBUTION_DIR / "admin_service.py",
        AGENT_DISTRIBUTION_DIR / "dynamic_acquisition.py",
        AGENT_DISTRIBUTION_DIR / "emergency_revocation_response.py",
        AGENT_DISTRIBUTION_DIR / "ed25519_package_attestation_verifier.py",
    )
    violations: list[str] = []
    for path in surfaces:
        text = path.read_text(encoding="utf-8")
        for pattern in _SUSPICIOUS_TRUST_ENGINE_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.name}: {pattern.pattern}")
    assert not violations, "\n".join(violations)


TESTING_SUPPORT_DIR = REPO_ROOT / "testing_support"


def _testing_support_imports_from_tests() -> list[str]:
    violations: list[str] = []
    for path in sorted(TESTING_SUPPORT_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            if module == "tests" or module.startswith("tests."):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {module}")
    return violations


def test_testing_support_does_not_import_tests() -> None:
    violations = _testing_support_imports_from_tests()
    assert not violations, (
        "testing_support must not import tests; violations:\n" + "\n".join(violations)
    )


def test_legacy_admin_test_aliases_reference_canonical_harness() -> None:
    from testing_support.agent_platform_admin_harness import (
        DeterministicAgentDistributionAdapter,
        FakeAgentCatalog,
        AgentProjectMetadataTestProvider,
    )
    from tests.unit.agent_distribution import (
        test_agent_platform_admin_service as admin_tests,
    )

    assert admin_tests._DeterministicAdapter is DeterministicAgentDistributionAdapter
    assert admin_tests._FakeCatalog is FakeAgentCatalog
    assert admin_tests._MetadataProvider is AgentProjectMetadataTestProvider


def test_ac6_frozen_architecture_section_present() -> None:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    required_markers = (
        "### 10.7 AC-6 Trust, Certification and Runtime Revocation - Frozen Architecture",
        "AgentPackageAttestationVerifier",
        "AgentPackageTrustCoordinator",
        "ActivationService.rollback",
        "QUALIFICATION != TRUST",
        "SIGNATURE != TRUST",
        "HISTORICAL ALLOW != CURRENT ALLOW",
        "INSTALLED != ADMISSIBLE",
        "FROZEN - Reference Production V1",
    )
    missing = [marker for marker in required_markers if marker not in text]
    assert not missing, f"missing AC-6 freeze markers: {missing}"
