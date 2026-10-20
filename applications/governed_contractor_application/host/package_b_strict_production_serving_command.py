# © Artur Czarnecki. All rights reserved.

"""Sanctioned Package B strict production serve command (profile v1 + D4 uvicorn factory)."""

from __future__ import annotations

import os
import sys

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.single_node_profile_v1_strict_startup import (
    open_gca_strict_production_serving_session_from_single_node_production_profile_v1,
)
from governed_contractor_application.host.strict_production_serving import (
    build_governed_contractor_strict_production_uvicorn_factory,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.package_b_operator_production_governance_configuration import (
    PackageBOperatorProductionGovernanceConfigurationError,
    load_package_b_operator_production_control_plane_governance_from_env,
)
from intergrax.applications._shared.package_b_strict_production_serving_environment import (
    PackageBStrictProductionServingEnvironmentError,
    load_strict_production_serving_environment_from_env,
)
from intergrax.applications._shared.single_node_production_profile_v1 import (
    load_single_node_production_profile_v1_from_env,
    validate_configured_single_node_production_profile_v1,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
)

_LEGACY_STRICT_SERVE_MARKERS = (
    "governed_contractor_application.host.main:app",
    "host.main:app",
)


class PackageBStrictProductionServingCommandError(RuntimeError):
    """Package B strict production serving command cannot start safely."""


def _reject_legacy_strict_serve_invocation() -> None:
    joined = " ".join(sys.argv)
    for marker in _LEGACY_STRICT_SERVE_MARKERS:
        if marker in joined:
            raise PackageBStrictProductionServingCommandError(
                f"legacy strict production entry {marker!r} is forbidden; "
                "use governed_contractor_application.host.package_b_strict_production_serving_command",
            )


def _close_strict_session_when_command_still_owns_shutdown(
    session: StrictProductionServingSession,
) -> None:
    """
    Close the session when the command path still owns shutdown (before ASGI lifespan).

    ASGI lifespan may have already closed the session; idempotent for that case.
    Writer lock release remains owned by ``StrictProductionServingSession.close()``.
    """
    try:
        session.close()
    except StrictProductionServingLauncherError as exc:
        message = str(exc)
        if "already closed" in message or "close incomplete" in message:
            return
        raise


def _validate_operator_backend_settings(
    settings: GovernedContractorBackendSettings,
) -> None:
    if not settings.api_keys_map:
        raise PackageBStrictProductionServingCommandError(
            "production API credentials are required "
            "(GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON or bootstrap key)",
        )
    if settings.strict_external_work_provider_id is None:
        raise PackageBStrictProductionServingCommandError(
            "provider execution remains disabled without operator provider pin: "
            "GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_ID",
        )
    if settings.strict_external_work_provider_descriptor_digest is None:
        raise PackageBStrictProductionServingCommandError(
            "provider execution remains disabled without operator provider pin: "
            "GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST",
        )


def run_package_b_strict_production_serving_command() -> None:
    """Open profile v1 strict session and serve via the D4 uvicorn factory."""
    _reject_legacy_strict_serve_invocation()
    try:
        settings = GovernedContractorBackendSettings.from_env()
        _validate_operator_backend_settings(settings)
        governance = load_package_b_operator_production_control_plane_governance_from_env()
        environment = load_strict_production_serving_environment_from_env()
        profile_config = load_single_node_production_profile_v1_from_env()
        validate_configured_single_node_production_profile_v1(profile_config)
        deployment_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
            trusted_artifact_parent_roots=profile_config.trusted_artifact_parent_roots,
            allow_qualification_locators=False,
        )
        session, _owned_gca, _config, _writer_hold = (
            open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
                environment=environment,
                governance=governance,
                deployment_adapter=deployment_adapter,
            )
        )
    except (
        PackageBOperatorProductionGovernanceConfigurationError,
        PackageBStrictProductionServingEnvironmentError,
        StrictProductionServingLauncherError,
        ValueError,
    ) as exc:
        raise PackageBStrictProductionServingCommandError(str(exc)) from exc

    try:
        if not session.readiness.ready_for_traffic:
            raise PackageBStrictProductionServingCommandError(
                "strict production serving session is not ready for traffic",
            )

        factory = build_governed_contractor_strict_production_uvicorn_factory(
            session,
            settings=settings,
        )
        bind_host = os.environ.get(
            "GOVERNED_CONTRACTOR_BACKEND_BIND_HOST",
            "0.0.0.0",
        ).strip()
        import uvicorn

        uvicorn.run(
            factory,
            host=bind_host or "0.0.0.0",
            port=settings.backend_port,
            factory=True,
        )
    except PackageBStrictProductionServingCommandError:
        _close_strict_session_when_command_still_owns_shutdown(session)
        raise
    except (KeyboardInterrupt, SystemExit):
        _close_strict_session_when_command_still_owns_shutdown(session)
        raise
    except BaseException as exc:
        _close_strict_session_when_command_still_owns_shutdown(session)
        raise PackageBStrictProductionServingCommandError(str(exc)) from exc


def main() -> None:
    try:
        run_package_b_strict_production_serving_command()
    except PackageBStrictProductionServingCommandError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
