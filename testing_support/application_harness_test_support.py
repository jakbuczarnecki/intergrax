# © Artur Czarnecki. All rights reserved.

"""Canonical harness host builders for application-layer unit tests."""

from __future__ import annotations

from typing import Any

from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest

CANONICAL_HARNESS_TEST_TENANT = "test-tenant"


def build_harness_host_runtime_for_tests(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    **kwargs: Any,
) -> HarnessHostRuntime:
    """``build_harness_host_runtime`` with explicit governance tenant identity."""
    tenant_id = kwargs.pop("tenant_id", CANONICAL_HARNESS_TEST_TENANT)
    return build_harness_host_runtime(
        manifest,
        environment,
        tenant_id=tenant_id,
        **kwargs,
    )
