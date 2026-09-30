# © Artur Czarnecki. All rights reserved.

"""Reference substrate qualification lifecycle errors."""

from __future__ import annotations


class ReferenceSubstrateLifecycleError(RuntimeError):
    """Aggregate qualification lifecycle or cleanup failure."""


class ReferenceSubstrateEndpointError(ReferenceSubstrateLifecycleError):
    """HTTP endpoint harness could not start in a controlled topology."""


class ReferenceSubstrateEndpointCleanupError(ReferenceSubstrateLifecycleError):
    """HTTP endpoint harness cleanup did not complete."""


class ReferenceSubstrateSecuritySetupLifecycleError(ReferenceSubstrateLifecycleError):
    """Secure session setup failed; owned topology cleanup may also have failed."""
