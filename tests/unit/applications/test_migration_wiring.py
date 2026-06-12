# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-2 — migration wiring and manifest coverage checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.applications.contracts.application_migration import (
    ApplicationMigration,
    MigrationStep,
    MigrationStepAction,
    MigrationStepTarget,
    ProfileMigration,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications._shared.migration_wiring import (
    check_application_migrations,
    load_application_migration,
    validate_application_migration_document,
    validate_manifest_migration_coverage,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _write_migration(app_root: Path, migration: ApplicationMigration) -> Path:
    migrations_dir = app_root / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    path = migrations_dir / f"{migration.migration_id}.json"
    path.write_text(json.dumps(migration.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def test_load_and_validate_migration_with_script(tmp_path: Path) -> None:
    app_root = tmp_path / "demo_application"
    app_root.mkdir()
    scripts_dir = app_root / "migrations" / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "profile_v1_1.py"
    script.write_text("# migration script\n", encoding="utf-8")

    migration = ApplicationMigration(
        migration_id="demo_1_0_to_1_1",
        from_app_version=">=1.0.0,<1.1.0",
        to_app_version="1.1.0",
        steps=[
            MigrationStep(
                target=MigrationStepTarget.PROFILE,
                action=MigrationStepAction.TRANSFORM,
                script_ref="migrations/scripts/profile_v1_1.py",
                breaking=True,
            ),
        ],
        profile_migration=ProfileMigration(
            migration_id="demo_profile_v1_1",
            from_spec_version="1.0.0",
            to_spec_version="1.1.0",
            default_injection={"profile_id": "default"},
            breaking=True,
            golden_replay_ref="tests/fixtures/demo_profile_replay.json",
        ),
    )
    path = _write_migration(app_root, migration)
    loaded = load_application_migration(path)
    violations = validate_application_migration_document(loaded, app_root=app_root)
    assert violations == []


def test_manifest_coverage_requires_matching_to_version() -> None:
    manifest = ApplicationManifest.model_validate(
        {
            "app_id": "demo",
            "name": "Demo",
            "version": "1.1.0",
            "route_prefix": "/v1/demo",
            "env_prefix": "DEMO_",
        },
    )
    migration = ApplicationMigration(
        migration_id="demo_mismatch",
        from_app_version=">=1.1.0,<1.2.0",
        to_app_version="1.0.9",
        steps=[
            MigrationStep(
                target=MigrationStepTarget.PROFILE,
                action=MigrationStepAction.VALIDATE_ONLY,
            ),
        ],
    )
    violations = validate_manifest_migration_coverage("demo_application", manifest, [migration])
    assert any("does not match manifest.version" in item for item in violations)


def test_check_application_migrations_ignores_hosts_without_migrations(
    tmp_path: Path,
) -> None:
    applications_root = tmp_path / "applications"
    (applications_root / "empty_application").mkdir(parents=True)
    assert check_application_migrations(applications_root) == []
