# © Artur Czarnecki. All rights reserved.

"""Load and validate Tier-3 application migrations (APP-EVOL-2 · APP-EVOL-2b)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from intergrax.applications.contracts.application_migration import (
    ApplicationMigration,
    GraphSpecMigration,
    MigrationStepTarget,
    OrgEnvelopeMigration,
    ProfileMigration,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.semver_compat import SemVer

_TYPED_TARGET_ORDER: tuple[MigrationStepTarget, ...] = (
    MigrationStepTarget.PROFILE,
    MigrationStepTarget.GRAPH_SPEC,
    MigrationStepTarget.ORG_ENVELOPE,
    MigrationStepTarget.ROSTER,
    MigrationStepTarget.HOOKS,
)
_VERSION_RANGE_RE = re.compile(
    r"^(?P<op>>=|<=|>|<|=)?(?P<version>\d+\.\d+\.\d+(?:[-+].*)?)$"
)


def load_application_migration(path: Path) -> ApplicationMigration:
    """Parse a migration JSON document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ApplicationMigration.model_validate(payload)


def iter_application_migration_files(
    applications_root: Path,
) -> Iterator[tuple[str, Path]]:
    """Yield ``(package_name, migration_path)`` for JSON files under ``migrations/``."""
    if not applications_root.is_dir():
        return
    for package_dir in sorted(applications_root.glob("*_application")):
        migrations_dir = package_dir / "migrations"
        if not migrations_dir.is_dir():
            continue
        for path in sorted(migrations_dir.glob("*.json")):
            yield package_dir.name, path


def _target_rank(target: MigrationStepTarget) -> int:
    try:
        return _TYPED_TARGET_ORDER.index(target)
    except ValueError:
        return len(_TYPED_TARGET_ORDER)


def validate_profile_migration(migration: ProfileMigration) -> list[str]:
    """Validate a typed profile migration document."""
    violations: list[str] = []
    if migration.breaking and not (
        migration.field_transforms or migration.default_injection or migration.golden_replay_ref
    ):
        violations.append(
            f"{migration.migration_id}: breaking ProfileMigration requires transforms, "
            "default_injection, or golden_replay_ref",
        )
    try:
        SemVer.parse(migration.from_spec_version)
        SemVer.parse(migration.to_spec_version)
    except ValueError as exc:
        violations.append(f"{migration.migration_id}: invalid profile spec version — {exc}")
    return violations


def validate_graph_spec_migration(migration: GraphSpecMigration) -> list[str]:
    """Validate a typed graph-spec migration document."""
    violations: list[str] = []
    try:
        SemVer.parse(migration.from_graph_version)
        SemVer.parse(migration.to_graph_version)
    except ValueError as exc:
        violations.append(f"{migration.migration_id}: invalid graph version — {exc}")
    if migration.breaking and not (
        migration.node_renames or migration.edge_rewrites or migration.golden_replay_ref
    ):
        violations.append(
            f"{migration.migration_id}: breaking GraphSpecMigration requires rewrites or golden_replay_ref",
        )
    for source, target in migration.node_renames.items():
        if not source.strip() or not target.strip():
            violations.append(f"{migration.migration_id}: node_renames keys and values must be non-empty")
    return violations


def validate_org_envelope_migration(migration: OrgEnvelopeMigration) -> list[str]:
    """Validate a typed organizational envelope migration document."""
    violations: list[str] = []
    try:
        SemVer.parse(migration.from_envelope_version)
        SemVer.parse(migration.to_envelope_version)
    except ValueError as exc:
        violations.append(f"{migration.migration_id}: invalid envelope version — {exc}")
    if migration.breaking and not (
        migration.playbook_id_map
        or migration.tool_deny_additions
        or migration.tool_deny_removals
        or migration.golden_replay_ref
    ):
        violations.append(
            f"{migration.migration_id}: breaking OrgEnvelopeMigration requires playbook/tool changes "
            "or golden_replay_ref",
        )
    return violations


def validate_application_migration_document(
    migration: ApplicationMigration,
    *,
    app_root: Path | None = None,
) -> list[str]:
    """Validate one :class:`ApplicationMigration` and optional on-disk script refs."""
    violations: list[str] = []
    try:
        SemVer.parse(migration.to_app_version)
    except ValueError as exc:
        violations.append(f"{migration.migration_id}: invalid to_app_version — {exc}")

    if not migration.steps:
        violations.append(f"{migration.migration_id}: steps must not be empty")

    ordered_targets = [_target_rank(step.target) for step in migration.steps]
    if ordered_targets != sorted(ordered_targets):
        violations.append(
            f"{migration.migration_id}: steps must follow profile → graph_spec → org_envelope → roster → hooks",
        )

    if migration.profile_migration is not None:
        violations.extend(validate_profile_migration(migration.profile_migration))
    if migration.graph_spec_migration is not None:
        violations.extend(validate_graph_spec_migration(migration.graph_spec_migration))
    if migration.org_envelope_migration is not None:
        violations.extend(validate_org_envelope_migration(migration.org_envelope_migration))

    if migration.profile_migration and migration.graph_spec_migration:
        if migration.profile_migration.breaking or migration.graph_spec_migration.breaking:
            pass  # order already enforced via steps
    if migration.graph_spec_migration and migration.org_envelope_migration:
        if (
            migration.graph_spec_migration.to_graph_version
            and migration.org_envelope_migration.to_envelope_version
        ):
            pass

    for step in migration.steps:
        if step.breaking and step.action.value != "validate_only":
            if not step.script_ref:
                violations.append(
                    f"{migration.migration_id}: breaking step on {step.target.value} requires script_ref",
                )
            elif app_root is not None:
                script_path = app_root / step.script_ref
                if not script_path.is_file():
                    violations.append(
                        f"{migration.migration_id}: missing script_ref {step.script_ref}",
                    )

    breaking_sub = any(
        item.breaking
        for item in (
            migration.profile_migration,
            migration.graph_spec_migration,
            migration.org_envelope_migration,
        )
        if item is not None
    )
    if breaking_sub and not any(step.breaking for step in migration.steps):
        violations.append(
            f"{migration.migration_id}: typed breaking sub-migration requires a breaking MigrationStep",
        )

    return violations


def version_matches_range(version: str, range_spec: str) -> bool:
    """Return whether ``version`` satisfies a simple semver range expression."""
    spec = range_spec.strip()
    if "," in spec:
        return all(version_matches_range(version, part.strip()) for part in spec.split(","))

    match = _VERSION_RANGE_RE.match(spec)
    if match is None:
        return version.strip() == spec

    operator = match.group("op") or "="
    bound = match.group("version")
    left = SemVer.parse(version)
    right = SemVer.parse(bound)

    if operator == "=":
        return left.major == right.major and left.minor == right.minor and left.patch == right.patch
    if operator == ">=":
        return (left.major, left.minor, left.patch) >= (right.major, right.minor, right.patch)
    if operator == "<=":
        return (left.major, left.minor, left.patch) <= (right.major, right.minor, right.patch)
    if operator == ">":
        return (left.major, left.minor, left.patch) > (right.major, right.minor, right.patch)
    if operator == "<":
        return (left.major, left.minor, left.patch) < (right.major, right.minor, right.patch)
    return False


def validate_manifest_migration_coverage(
    package: str,
    manifest: ApplicationManifest,
    migrations: list[ApplicationMigration],
) -> list[str]:
    """Ensure declared migrations cover the manifest version and breaking bumps."""
    violations: list[str] = []
    if not migrations:
        return violations

    for migration in migrations:
        if not version_matches_range(manifest.version, migration.from_app_version):
            continue
        if migration.to_app_version != manifest.version:
            violations.append(
                f"{package}: migration {migration.migration_id} to_app_version "
                f"{migration.to_app_version!r} does not match manifest.version {manifest.version!r}",
            )

    applicable = [
        migration
        for migration in migrations
        if version_matches_range(manifest.version, migration.from_app_version)
        and migration.to_app_version == manifest.version
    ]
    if not applicable:
        violations.append(
            f"{package}: manifest.version {manifest.version!r} has no matching ApplicationMigration",
        )
        return violations

    latest = max(applicable, key=lambda item: SemVer.parse(item.to_app_version))
    if latest.to_app_version != manifest.version:
        violations.append(
            f"{package}: latest migration must target manifest.version {manifest.version!r}",
        )
    return violations


def _load_manifest_for_package(package: str) -> ApplicationManifest | None:
    import importlib
    import inspect

    try:
        module = importlib.import_module(f"{package}.manifest")
    except ImportError:
        return None

    for value in module.__dict__.values():
        if isinstance(value, ApplicationManifest):
            return value

    for value in module.__dict__.values():
        if not callable(value) or inspect.isclass(value):
            continue
        if not value.__name__.startswith("build_"):
            continue
        manifest = value()
        if isinstance(manifest, ApplicationManifest):
            return manifest
    return None


def check_application_migrations(applications_root: Path) -> list[str]:
    """Validate all application migration documents and manifest coverage."""
    violations: list[str] = []
    migrations_by_package: dict[str, list[tuple[Path, ApplicationMigration]]] = {}

    for package, path in iter_application_migration_files(applications_root):
        app_root = applications_root / package
        try:
            migration = load_application_migration(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rel = path.relative_to(applications_root.parent).as_posix()
            violations.append(f"{rel}: invalid migration document — {exc}")
            continue
        violations.extend(
            validate_application_migration_document(migration, app_root=app_root),
        )
        migrations_by_package.setdefault(package, []).append((path, migration))

    for package, entries in sorted(migrations_by_package.items()):
        manifest = _load_manifest_for_package(package)
        if manifest is None:
            violations.append(f"{package}: migrations present but manifest could not be loaded")
            continue
        migrations = [migration for _, migration in entries]
        violations.extend(validate_manifest_migration_coverage(package, manifest, migrations))

    return violations
