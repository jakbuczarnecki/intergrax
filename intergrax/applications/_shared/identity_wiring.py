# © Artur Czarnecki. All rights reserved.

"""Apply identity profile to FastAPI harness hosts (Phase H-APP.2.2)."""

from __future__ import annotations

import os

from fastapi import FastAPI

from intergrax.applications._shared.harness_auth import apply_harness_auth_middleware
from intergrax.applications.contracts.environment_profile import IdentityProfile


def wire_application_identity(app: FastAPI, profile: IdentityProfile) -> None:
    """
    Configure harness API key middleware from ``IdentityProfile``.

    When ``require_api_key`` is true and no key is configured, startup should fail
    in the host factory (same guard as lab strict harness).
    """
    if profile.require_api_key:
        key = (os.environ.get(profile.api_key_env) or "").strip()
        if not key:
            raise ValueError(
                f"{profile.api_key_env} is required when identity_profile.require_api_key=true"
            )
    apply_harness_auth_middleware(app)
