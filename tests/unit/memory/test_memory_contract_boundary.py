# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from intergrax.memory.contracts import memory_models as canonical_models
from intergrax.memory import user_profile_memory as legacy_surface
from tests.unit.memory.memory_contract_boundary_ast import scan_memory_contracts_tree

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_ROOT = REPO_ROOT / "intergrax" / "memory" / "contracts"
_MEMORY_ROOT = REPO_ROOT / "intergrax" / "memory"


def test_memory_contracts_do_not_import_implementation_modules() -> None:
    violations = scan_memory_contracts_tree(CONTRACTS_ROOT)
    assert not violations, "\n".join(v.as_message() for v in violations)


def test_memory_package_does_not_import_applications_tier() -> None:
    forbidden = "intergrax.applications"
    for path in sorted(_MEMORY_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert forbidden not in text, f"{path} imports applications tier"


def test_public_user_profile_memory_reexports_canonical_models() -> None:
    assert legacy_surface.UserProfile is canonical_models.UserProfile
    assert (
        legacy_surface.UserProfileMemoryEntry is canonical_models.UserProfileMemoryEntry
    )
    assert legacy_surface.MemoryKind is canonical_models.MemoryKind
    assert legacy_surface.MemoryImportance is canonical_models.MemoryImportance
    assert legacy_surface.UserIdentity is canonical_models.UserIdentity
    assert legacy_surface.UserPreferences is canonical_models.UserPreferences


def _run_cold_import(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cold_import_contracts_package() -> None:
    result = _run_cold_import("import intergrax.memory.contracts")
    assert result.returncode == 0, result.stderr


def test_cold_import_memory_lifecycle_outcome() -> None:
    result = _run_cold_import(
        "from intergrax.memory.contracts.memory_lifecycle import MemoryLifecycleOutcome"
    )
    assert result.returncode == 0, result.stderr


def test_cold_import_user_profile_public_path() -> None:
    result = _run_cold_import(
        "from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry"
    )
    assert result.returncode == 0, result.stderr


def test_import_order_independence_subprocess_matrix() -> None:
    scripts = [
        "from intergrax.memory.contracts.memory_lifecycle import MemoryLifecycleOutcome; "
        "from intergrax.memory.user_profile_memory import UserProfile",
        "from intergrax.memory.user_profile_memory import UserProfile; "
        "from intergrax.memory.contracts.memory_lifecycle import MemoryLifecycleOutcome",
        "import intergrax.memory; "
        "from intergrax.memory.contracts.memory_control import MemoryControlPlaneScope",
    ]
    for script in scripts:
        result = _run_cold_import(script)
        assert result.returncode == 0, f"script failed: {script!r}\n{result.stderr}"
