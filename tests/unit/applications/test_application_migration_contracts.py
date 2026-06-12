# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-2/2b — ApplicationMigration and typed sub-migration contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications.contracts.application_migration import (
    ApplicationMigration,
    GraphSpecMigration,
    MigrationStep,
    MigrationStepAction,
    MigrationStepTarget,
    OrgEnvelopeMigration,
    ProfileMigration,
)
from intergrax.applications._shared.migration_wiring import (
    validate_application_migration_document,
    validate_graph_spec_migration,
    validate_org_envelope_migration,
    validate_profile_migration,
    version_matches_range,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sample_migration(*, breaking: bool = False) -> ApplicationMigration:
    return ApplicationMigration(
        migration_id="legal_1_0_to_1_1",
        from_app_version=">=1.0.0,<1.1.0",
        to_app_version="1.1.0",
        steps=[
            MigrationStep(
                target=MigrationStepTarget.PROFILE,
                action=MigrationStepAction.TRANSFORM,
                script_ref="migrations/scripts/profile_v1_1.py",
                breaking=breaking,
            ),
        ],
        profile_migration=ProfileMigration(
            migration_id="legal_profile_v1_1",
            from_spec_version="1.0.0",
            to_spec_version="1.1.0",
            default_injection={"cost_profile": {"enabled": True}},
            breaking=breaking,
        ),
        rollback_supported=False,
    )


def test_application_migration_round_trip() -> None:
    migration = _sample_migration()
    payload = migration.model_dump(mode="json")
    restored = ApplicationMigration.model_validate(payload)
    assert restored.migration_id == migration.migration_id
    assert restored.profile_migration is not None


def test_breaking_step_requires_script_ref() -> None:
    with pytest.raises(ValidationError, match="script_ref"):
        MigrationStep(
            target=MigrationStepTarget.PROFILE,
            action=MigrationStepAction.TRANSFORM,
            breaking=True,
        )


def test_profile_migration_requires_profile_step() -> None:
    with pytest.raises(ValidationError, match="profile MigrationStep"):
        ApplicationMigration(
            migration_id="missing_step",
            from_app_version="1.0.0",
            to_app_version="1.1.0",
            steps=[],
            profile_migration=ProfileMigration(
                migration_id="profile_only",
                from_spec_version="1.0.0",
                to_spec_version="1.1.0",
            ),
        )


def test_validate_profile_migration_breaking_requires_transform() -> None:
    migration = ProfileMigration(
        migration_id="profile_break",
        from_spec_version="1.0.0",
        to_spec_version="2.0.0",
        breaking=True,
    )
    violations = validate_profile_migration(migration)
    assert any("breaking ProfileMigration" in item for item in violations)


def test_validate_graph_spec_migration_node_rename() -> None:
    migration = GraphSpecMigration(
        migration_id="graph_break",
        from_graph_version="1.0.0",
        to_graph_version="2.0.0",
        node_renames={"": "next"},
        breaking=True,
    )
    violations = validate_graph_spec_migration(migration)
    assert any("node_renames" in item for item in violations)


def test_validate_org_envelope_migration_versions() -> None:
    migration = OrgEnvelopeMigration(
        migration_id="org_break",
        from_envelope_version="not-semver",
        to_envelope_version="1.1.0",
        breaking=True,
        playbook_id_map={"old": "new"},
    )
    violations = validate_org_envelope_migration(migration)
    assert any("invalid envelope version" in item for item in violations)


def test_version_matches_range_supports_comma_clauses() -> None:
    assert version_matches_range("1.1.0", ">=1.0.0,<2.0.0")
    assert not version_matches_range("2.0.0", ">=1.0.0,<2.0.0")


def test_step_order_validation() -> None:
    migration = ApplicationMigration(
        migration_id="bad_order",
        from_app_version="1.0.0",
        to_app_version="1.1.0",
        steps=[
            MigrationStep(
                target=MigrationStepTarget.GRAPH_SPEC,
                action=MigrationStepAction.VALIDATE_ONLY,
            ),
            MigrationStep(
                target=MigrationStepTarget.PROFILE,
                action=MigrationStepAction.VALIDATE_ONLY,
            ),
        ],
    )
    violations = validate_application_migration_document(migration)
    assert any("profile → graph_spec" in item for item in violations)
