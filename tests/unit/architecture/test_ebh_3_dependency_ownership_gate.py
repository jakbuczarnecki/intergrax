# © Artur Czarnecki. All rights reserved.

"""EBH-3 — dependency direction and Integrations composition ownership gates."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_SHARED_PREFIX = "intergrax.integrations._shared"
_INTEGRATIONS_PREFIX = "intergrax/integrations/"
_PLATFORM_SCAN_ROOT = _REPO_ROOT / "intergrax"

_CONTRACT_MODULE = _REPO_ROOT / "intergrax/integrations/contracts/circuit_breaker.py"
_CONTRACT_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.integrations._shared",
    "intergrax.integrations.registry",
    "intergrax.integrations.providers",
    "intergrax.runtime",
    "intergrax.applications",
)


def _platform_python_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(_PLATFORM_SCAN_ROOT.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith(_INTEGRATIONS_PREFIX):
            continue
        files.append(path)
    return files


def _module_imports_forbidden_shared(module: str | None) -> bool:
    if module is None:
        return False
    return module == _FORBIDDEN_SHARED_PREFIX or module.startswith(f"{_FORBIDDEN_SHARED_PREFIX}.")


def _collect_forbidden_shared_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _module_imports_forbidden_shared(node.module):
            hits.append(node.module or "")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_imports_forbidden_shared(alias.name):
                    hits.append(alias.name)
    return hits


def test_ebh_3_production_must_not_import_integrations_shared() -> None:
    violations: list[str] = []
    for path in _platform_python_files():
        hits = _collect_forbidden_shared_imports(path)
        if hits:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel} ({', '.join(sorted(set(hits)))})")
    assert not violations, (
        "Cross-domain production modules must use Integrations contracts/registry composition, not "
        f"{_FORBIDDEN_SHARED_PREFIX}.*: " + "; ".join(violations)
    )


def test_ebh_3_circuit_breaker_config_has_single_class_definition() -> None:
    definitions = 0
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except OSError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "IntegrationCircuitBreakerConfig":
                definitions += 1
    assert definitions == 1, f"expected exactly one IntegrationCircuitBreakerConfig, found {definitions}"


def test_ebh_3_circuit_breaker_contract_module_is_pure() -> None:
    hits = _collect_forbidden_shared_imports(_CONTRACT_MODULE)
    tree = ast.parse(_CONTRACT_MODULE.read_text(encoding="utf-8"), filename=str(_CONTRACT_MODULE))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in _CONTRACT_FORBIDDEN_IMPORT_PREFIXES:
                if node.module == prefix or node.module.startswith(f"{prefix}."):
                    hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in _CONTRACT_FORBIDDEN_IMPORT_PREFIXES:
                    if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                        hits.append(alias.name)
    assert not hits, f"circuit_breaker contract imports forbidden modules: {sorted(set(hits))}"


def test_ebh_3_applications_must_not_construct_private_integration_breaker() -> None:
    root = _REPO_ROOT / "intergrax/applications"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "IntegrationCircuitBreaker(" in text or "integrations._shared.circuit_breaker" in text:
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violations, violations


def test_ebh_3_rag_must_not_import_private_circuit_breaker_module() -> None:
    rag_root = _REPO_ROOT / "intergrax/rag"
    violations: list[str] = []
    for path in sorted(rag_root.rglob("*.py")):
        hits = _collect_forbidden_shared_imports(path)
        if any("circuit_breaker" in h for h in hits):
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violations, violations


def test_ebh_3_shared_health_module_imports_without_package_init_cycle() -> None:
    probe = textwrap.dedent(
        """
        import intergrax.integrations._shared.health as health

        if not callable(health.health_check_all):
            raise SystemExit(2)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_ebh_3_sanctioned_health_probes_surface_imports() -> None:
    from intergrax.integrations.registry import health_probes

    assert callable(health_probes.health_check_all)
    assert callable(health_probes.health_check_catalog_slugs)


def test_ebh_3_sanctioned_circuit_breaker_composition_surface() -> None:
    from intergrax.integrations.registry.circuit_breakers import create_integration_circuit_breaker
    from intergrax.integrations.contracts.circuit_breaker import IntegrationCircuitBreakerConfig

    breaker = create_integration_circuit_breaker(
        "ebh-3-gate",
        IntegrationCircuitBreakerConfig(failure_threshold=1),
    )
    assert breaker.call(lambda: "ok") == "ok"
