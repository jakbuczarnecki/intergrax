# © Artur Czarnecki. All rights reserved.

"""Canonical harness-host meaningful side-effect authorization composition (GR-10-R9 / P2D-R1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.collaborative_work.persistence import (
    collaborative_work_core_repositories,
    open_sqlite_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.runtime.governance.orchestration_meaningful_side_effect_composition import (
    build_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


def resolve_collaborative_work_sqlite_path_for_harness_host(
    checkpoints_db_path: Path | None,
) -> Path | None:
    """When checkpoint storage is explicit (harness/test), materialize CW repos on SQLite alongside it."""
    if checkpoints_db_path is None:
        return None
    return checkpoints_db_path.with_name("collaborative_work.db")


def build_harness_host_meaningful_side_effect_authorization_port(
    environment: ApplicationEnvironmentProfile,
    *,
    collaborative_work_sqlite_path: Path | None = None,
) -> MeaningfulSideEffectAuthorizationPort:
    """Build platform default MSE authorization for strict Tier-3 harness hosts."""
    if collaborative_work_sqlite_path is not None:
        bundle = open_sqlite_collaborative_work_repositories(
            str(collaborative_work_sqlite_path),
        )
    else:
        bundle = resolve_collaborative_work_repositories(environment.integration_profile)
    core = collaborative_work_core_repositories(bundle)
    return build_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=core.operation_profile,
        membership_repository=core.membership,
        principal_authority_repository=core.principal_authority,
        delegation_repository=core.delegation,
        collaborative_policy_repository=core.policy,
        runtime_policy_evaluator=RuntimePolicyEngine(),
    )


def resolve_harness_host_meaningful_side_effect_authorization_port(
    environment: ApplicationEnvironmentProfile,
    *,
    explicit: MeaningfulSideEffectAuthorizationPort | None = None,
    collaborative_work_sqlite_path: Path | None = None,
) -> MeaningfulSideEffectAuthorizationPort | None:
    """Resolve MSE port: injectable override, strict default, or absent in non-strict hosts."""
    if explicit is not None:
        return explicit
    if environment.execution_mode.value != "strict":
        return None
    return build_harness_host_meaningful_side_effect_authorization_port(
        environment,
        collaborative_work_sqlite_path=collaborative_work_sqlite_path,
    )


__all__ = [
    "build_harness_host_meaningful_side_effect_authorization_port",
    "resolve_collaborative_work_sqlite_path_for_harness_host",
    "resolve_harness_host_meaningful_side_effect_authorization_port",
]
