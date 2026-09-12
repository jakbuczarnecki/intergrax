"""Architecture gate: VPI scenario root must not accumulate operational Python scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs/scenarios/verified_product_identification"

_ALLOWED_ROOT_SCRIPTS = frozenset({"run_proof.py"})

_SCRIPT_PURPOSES = frozenset(
    {
        "dataset_generation",
        "dataset_validation",
        "operator_lifecycle",
        "diagnostics",
        "retrieval_tooling",
        "storage_tooling",
        "identity_tooling",
        "proof_tooling",
        "migration",
        "experimental",
    }
)

_DEPRECATED_FLAT_SCRIPT_DIRS = frozenset({"build", "diagnostics", "operator"})


def test_scenario_root_has_no_orphan_scripts() -> None:
    orphans = sorted(
        path.name
        for path in _VPI_ROOT.glob("*.py")
        if path.name not in _ALLOWED_ROOT_SCRIPTS
    )
    assert orphans == []


def test_operational_scripts_use_domain_and_purpose_layout() -> None:
    scripts_root = _VPI_ROOT / "scripts"
    violations: list[str] = []
    for path in sorted(scripts_root.rglob("*.py")):
        rel = path.relative_to(scripts_root)
        parts = rel.parts
        if parts[0] in _DEPRECATED_FLAT_SCRIPT_DIRS:
            if path.name != "__init__.py" and "Deprecated import path" not in path.read_text(
                encoding="utf-8"
            ):
                violations.append(f"{rel.as_posix()}: non-shim in deprecated flat folder")
            continue
        if parts[0] == "__pycache__":
            continue
        if len(parts) < 3:
            if path.name == "__init__.py" and len(parts) == 1:
                continue
            if path.name == "__init__.py" and len(parts) == 2:
                continue
            violations.append(f"{rel.as_posix()}: must live under scripts/<domain>/<purpose>/")
            continue
        purpose = parts[1]
        if purpose not in _SCRIPT_PURPOSES:
            violations.append(f"{rel.as_posix()}: unknown purpose '{purpose}'")
    assert violations == []
