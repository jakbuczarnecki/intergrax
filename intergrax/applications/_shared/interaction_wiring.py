# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Slack / Teams / lab interaction intake for Tier-3 hosts (§18, B.12)."""

from __future__ import annotations

from intergrax.applications._shared.task_control_wiring import TaskEnricher
from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.security.security_configuration import (
    INBOUND_VERIFIER_NONE_FORBIDDEN,
    LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
    SecurityConfigurationError,
)
from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.runtime.interactions.factory import (
    create_interaction_adapter,
    resolve_interaction_settings,
)
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.task_executor import TaskExecutor
from intergrax.runtime.interactions.verification.factory import (
    ENV_INBOUND_VERIFIER,
    InboundVerifierMode,
    create_inbound_verifier,
)


def wire_interaction_intake_service(
    *,
    env: ApplicationEnvironmentProfile,
    interaction_surface: str = "auto",
    task_executor: TaskExecutor | None = None,
    task_enricher: TaskEnricher | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> InteractionIntakeService:
    """
    Build :class:`InteractionIntakeService` for ``POST /v1/interactions/intake``.

    ``interaction_surface`` follows ``LAB_INTERACTION_SURFACE`` semantics:
    ``auto`` | ``slack`` | ``teams`` | ``lab`` | ``lab_json``.
    """
    settings = resolve_interaction_settings(surface=interaction_surface)
    adapter = create_interaction_adapter(settings)

    if wiring_fails_closed(env):
        if lab_opt_in is not None:
            if env.application_profile is ApplicationProfile.PRODUCT:
                raise SecurityConfigurationError(
                    item="lab_opt_in",
                    reason_code=LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
                    message="laboratory opt-in is forbidden for PRODUCT application profile",
                )
            raise SecurityConfigurationError(
                item="lab_opt_in",
                reason_code=INBOUND_VERIFIER_NONE_FORBIDDEN,
                message="laboratory opt-in is forbidden under strict wiring posture",
            )
        verifier = create_inbound_verifier()
    else:
        derived_opt_in = lab_opt_in or LabSelectorOptIn(
            selector=InboundVerifierMode.NONE.value,
            reason="derived laboratory inbound verifier default for non-strict host wiring",
        )
        verifier = create_inbound_verifier(lab_opt_in=derived_opt_in)

    return InteractionIntakeService(
        task_executor=task_executor,
        adapter=adapter,
        verifier=verifier,
        task_enricher=task_enricher,
    )


__all__ = [
    "ENV_INBOUND_VERIFIER",
    "wire_interaction_intake_service",
]
