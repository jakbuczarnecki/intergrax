# © Artur Czarnecki. All rights reserved.

"""Typed strict production serving environment for Package B operator wiring (PB-S02)."""

from __future__ import annotations

from intergrax.applications._shared.settings_loader import EnvReader
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingEnvironment,
)

_ENV_PREFIX = "INTERGRAX_STRICT_PRODUCTION_SERVING_"
_DEFAULT_APPLICATION_ID = "governed_contractor"


class PackageBStrictProductionServingEnvironmentError(ValueError):
    """Strict production serving environment configuration is incomplete."""


def load_strict_production_serving_environment_from_env(
    *,
    env: EnvReader | None = None,
) -> StrictProductionServingEnvironment:
    reader = env or EnvReader(_ENV_PREFIX)
    application_id = reader.str("APPLICATION_ID", default=_DEFAULT_APPLICATION_ID).strip()
    application_environment_id = reader.optional_str("APPLICATION_ENVIRONMENT_ID")
    if not application_environment_id:
        raise PackageBStrictProductionServingEnvironmentError(
            f"{_ENV_PREFIX}APPLICATION_ENVIRONMENT_ID is required",
        )
    if not application_id:
        raise PackageBStrictProductionServingEnvironmentError(
            f"{_ENV_PREFIX}APPLICATION_ID must be non-empty when set",
        )
    return StrictProductionServingEnvironment(
        application_id=application_id,
        application_environment_id=application_environment_id,
    )


__all__ = [
    "PackageBStrictProductionServingEnvironmentError",
    "load_strict_production_serving_environment_from_env",
]
