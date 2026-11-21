# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-I — A4-007 seam b2: the migration-gate scripts refuse unregistered application packages.

Negative tests NT-A7C-09 and NT-A7C-10, control CTL-A7C-04 (plan section 8). The temporary
applications root holds one unregistered ``*_application`` package with a valid migration document
and a ``manifest.py`` whose import has a visible side effect (it writes a marker file). Before the
change the gate imports that module, so the side-effect assertion is what turns red.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import scripts.gates.check_application_production_gates as production_gates
from intergrax.applications.contracts.application_migration import (
    ApplicationMigration,
    MigrationStep,
    MigrationStepAction,
    MigrationStepTarget,
)
from scripts.maintenance import check_application_migrations as maintenance_gate

pytestmark = [pytest.mark.unit]

_SYNTH_PACKAGE = "zz_synth_app_application"
_EXPECTED_VIOLATION = f"{_SYNTH_PACKAGE}: migrations present but manifest could not be loaded"


@dataclass(frozen=True)
class _SyntheticTree:
    applications_root: Path
    side_effect_marker: Path


def _purge_synthetic_modules() -> None:
    for name in [key for key in sys.modules if key == _SYNTH_PACKAGE or key.startswith(_SYNTH_PACKAGE + ".")]:
        del sys.modules[name]


@pytest.fixture
def synthetic_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_SyntheticTree]:
    applications_root = tmp_path / "applications"
    package_dir = applications_root / _SYNTH_PACKAGE
    migrations_dir = package_dir / "migrations"
    migrations_dir.mkdir(parents=True)
    marker = tmp_path / "manifest_side_effect.txt"
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "manifest.py").write_text(
        f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    migration = ApplicationMigration(
        migration_id="zz_synth_1_0_to_1_1",
        from_app_version="1.0.0",
        to_app_version="1.1.0",
        steps=[
            MigrationStep(
                target=MigrationStepTarget.PROFILE,
                action=MigrationStepAction.VALIDATE_ONLY,
            ),
        ],
    )
    (migrations_dir / "zz_synth_1_0_to_1_1.json").write_text(
        json.dumps(migration.model_dump(mode="json")),
        encoding="utf-8",
    )
    _purge_synthetic_modules()
    monkeypatch.syspath_prepend(str(applications_root))
    yield _SyntheticTree(applications_root=applications_root, side_effect_marker=marker)
    _purge_synthetic_modules()


def _record_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real_import = importlib.import_module

    def _recording(name: str, package: str | None = None) -> ModuleType:
        calls.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _recording)
    return calls


def _synthetic(calls: list[str]) -> list[str]:
    return [name for name in calls if name.startswith(_SYNTH_PACKAGE)]


def test_nt_a7c_09_maintenance_gate_refuses_unregistered_application_package(
    synthetic_tree: _SyntheticTree,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(maintenance_gate, "APPLICATIONS_ROOT", synthetic_tree.applications_root)
    calls = _record_imports(monkeypatch)

    exit_code = maintenance_gate.main()
    output = capsys.readouterr().out

    assert not synthetic_tree.side_effect_marker.exists()
    assert _synthetic(calls) == []
    assert exit_code == 1
    assert _EXPECTED_VIOLATION in output


def test_nt_a7c_10_production_gate_migrations_refuses_unregistered_application_package(
    synthetic_tree: _SyntheticTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(production_gates, "APPLICATIONS_ROOT", synthetic_tree.applications_root)
    calls = _record_imports(monkeypatch)

    violations = production_gates.check_application_migrations()

    assert not synthetic_tree.side_effect_marker.exists()
    assert _synthetic(calls) == []
    assert violations == [_EXPECTED_VIOLATION]


def test_ctl_a7c_04_both_gate_entry_points_are_clean_over_the_real_tree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert maintenance_gate.main() == 0
    assert "application migrations gate: OK" in capsys.readouterr().out
    assert production_gates.check_application_migrations() == []
