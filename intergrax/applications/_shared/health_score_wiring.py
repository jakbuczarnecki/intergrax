# © Artur Czarnecki. All rights reserved.

"""EnvironmentHealthScore computation from APP-PROD / APP-EVOL / APP-OPS gates (APP-OPS-3)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.agent_certification_wiring import (
    validate_strict_roster_agent_certification,
)
from intergrax.applications._shared.budget_wiring import check_manifest_budget_enforcement
from intergrax.applications._shared.capability_alias_wiring import (
    build_capability_alias_registry,
    check_manifest_lists_canonical_capabilities,
    validate_capability_governance_profile,
)
from intergrax.applications._shared.capability_graph_deploy_gate import (
    check_strict_product_capability_graph,
)
from intergrax.applications._shared.environment_snapshot_wiring import capture_environment_snapshot
from intergrax.applications._shared.migration_wiring import (
    iter_application_migration_files,
    load_application_migration,
    validate_application_migration_document,
    validate_manifest_migration_coverage,
)
from intergrax.applications._shared.ownership_wiring import check_manifest_operational_ownership
from intergrax.applications._shared.recovery_contract_wiring import (
    check_strict_product_recovery_contract,
)
from intergrax.applications._shared.tier3_scenario_matrix_wiring import (
    check_reference_host_scenario_matrix,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_health_score import (
    PRODUCTION_READY_THRESHOLD,
    ApplicationHealthScore,
    EnvironmentHealthScore,
    HealthDimension,
    HealthDimensionScore,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.utils.time_provider import SystemTimeProvider


def _binary_dimension(
    dimension: HealthDimension,
    violations: list[str],
    *,
    evidence_refs: list[str] | None = None,
) -> HealthDimensionScore:
    refs = list(evidence_refs or [])
    if violations:
        refs.extend(violations[:3])
        return HealthDimensionScore(dimension=dimension, score=0.0, evidence_refs=refs)
    refs.append(f"{dimension.value}:gate-green")
    return HealthDimensionScore(dimension=dimension, score=1.0, evidence_refs=refs)


def _deprecated_capability_violations(
    package: str,
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    violations: list[str] = []
    profile = env.capability_governance_profile
    violations.extend(validate_capability_governance_profile(profile))
    registry = build_capability_alias_registry(profile)
    violations.extend(
        check_manifest_lists_canonical_capabilities(package, manifest, registry),
    )
    for binding in manifest.enabled_agents():
        contract = binding.resolved_agent_type()().get_contract()
        if contract.lifecycle_state is AgentLifecycleState.DEPRECATED:
            violations.append(f"roster agent {contract.id} lifecycle is deprecated")
    return violations


def _stale_agent_violations(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    return validate_strict_roster_agent_certification(manifest, env)


def _migration_violations(
    package: str,
    manifest: ApplicationManifest,
    applications_root: Path,
) -> list[str]:
    violations: list[str] = []
    migrations: list = []
    for pkg, path in iter_application_migration_files(applications_root):
        if pkg != package:
            continue
        migration = load_application_migration(path)
        violations.extend(validate_application_migration_document(migration))
        migrations.append(migration)
    violations.extend(validate_manifest_migration_coverage(package, manifest, migrations))
    return violations


def _policy_coverage_violations(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []
    if env.execution_mode is not ExecutionMode.STRICT:
        return []
    has_rules = False
    if env.policy_rules is not None:
        has_rules = bool(env.policy_rules.inline_rules or env.policy_rules.rules_path)
    has_org = env.organizational_policy is not None
    has_policy_bundle = env.security_profile.prompt_defense_enabled
    if has_rules or has_org or has_policy_bundle:
        return []
    return ["STRICT product host missing policy rules, organizational envelope, or security profile"]


def _scenario_violations_for_package(repo_root: Path, package: str) -> list[str]:
    prefix = f"{package}:"
    return [
        item[len(prefix) :]
        for item in check_reference_host_scenario_matrix(repo_root)
        if item.startswith(prefix)
    ]


def compute_environment_health_score(
    product_id: str,
    manifest: ApplicationManifest,
    *,
    repo_root: Path,
    environment_id: str | None = None,
) -> EnvironmentHealthScore:
    """Compute platform-scoped health score for one product environment."""
    env = manifest.resolved_environment()
    package = f"{product_id}_application"
    applications_root = repo_root / "applications"
    env_id = environment_id or f"{manifest.app_id}-strict"
    snapshot = capture_environment_snapshot(manifest, env)

    dimension_inputs: list[tuple[HealthDimension, list[str]]] = [
        (
            HealthDimension.DEPRECATED_CAPABILITIES,
            _deprecated_capability_violations(package, manifest, env),
        ),
        (HealthDimension.STALE_AGENTS, _stale_agent_violations(manifest, env)),
        (
            HealthDimension.FAILED_MIGRATIONS,
            _migration_violations(package, manifest, applications_root),
        ),
        (HealthDimension.POLICY_COVERAGE, _policy_coverage_violations(manifest, env)),
        (
            HealthDimension.TEST_COVERAGE,
            _scenario_violations_for_package(repo_root, package),
        ),
        (
            HealthDimension.OWNERSHIP_COMPLETE,
            check_manifest_operational_ownership(product_id, manifest),
        ),
        (
            HealthDimension.CAPABILITY_GRAPH_VALID,
            check_strict_product_capability_graph(product_id, manifest),
        ),
        (
            HealthDimension.BUDGET_GOVERNANCE_CONFIGURED,
            check_manifest_budget_enforcement(product_id, manifest),
        ),
        (
            HealthDimension.RECOVERY_CONTRACT_DOCUMENTED,
            check_strict_product_recovery_contract(
                product_id,
                manifest,
                applications_root=applications_root,
            ),
        ),
    ]

    dimensions: list[HealthDimensionScore] = []
    blockers: list[str] = []
    warnings: list[str] = []

    for dimension, violations in dimension_inputs:
        scored = _binary_dimension(dimension, violations)
        dimensions.append(scored)
        if scored.score == 0.0:
            blockers.append(f"{dimension.value}: {violations[0] if violations else 'failed'}")
        elif violations:
            warnings.extend(violations)

    overall = sum(item.score for item in dimensions) / len(dimensions) if dimensions else 0.0

    return EnvironmentHealthScore(
        app_id=manifest.app_id,
        environment_id=env_id,
        snapshot_id=snapshot.snapshot_id,
        scored_at=SystemTimeProvider.utc_now(),
        overall=round(overall, 4),
        dimensions=dimensions,
        blockers=blockers,
        warnings=warnings,
    )


def build_application_health_score(
    product_id: str,
    manifest: ApplicationManifest,
    *,
    repo_root: Path,
) -> ApplicationHealthScore:
    """Roll up environment scores for one application."""
    env_score = compute_environment_health_score(product_id, manifest, repo_root=repo_root)
    production_ready = (
        env_score.overall >= PRODUCTION_READY_THRESHOLD and not env_score.blockers
    )
    worst = None
    if env_score.overall < PRODUCTION_READY_THRESHOLD:
        worst = env_score.environment_id
    return ApplicationHealthScore(
        app_id=manifest.app_id,
        environments=[env_score],
        worst_environment=worst,
        production_ready=production_ready,
    )


def format_environment_health_score(score: EnvironmentHealthScore) -> str:
    """Human-readable health summary for CLI output."""
    lines = [
        f"app:         {score.app_id}",
        f"environment: {score.environment_id}",
        f"snapshot:    {score.snapshot_id or 'n/a'}",
        f"overall:     {score.overall:.2f}",
        f"production_ready: {score.overall >= PRODUCTION_READY_THRESHOLD and not score.blockers}",
        "dimensions:",
    ]
    for item in score.dimensions:
        lines.append(f"  - {item.dimension.value}: {item.score:.2f}")
    if score.blockers:
        lines.append("blockers:")
        lines.extend(f"  - {item}" for item in score.blockers)
    if score.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {item}" for item in score.warnings)
    return "\n".join(lines)


def check_strict_product_health_scores(repo_root: Path) -> list[str]:
    """Return violations when any STRICT product host is below production threshold."""
    from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        rollup = build_application_health_score(product_id, manifest, repo_root=repo_root)
        env_score = rollup.environments[0]
        if not rollup.production_ready:
            violations.append(
                f"{product_id}: health overall {env_score.overall:.2f} below "
                f"{PRODUCTION_READY_THRESHOLD}",
            )
        if env_score.blockers:
            violations.append(f"{product_id}: blockers present ({len(env_score.blockers)})")
    return violations
