# © Artur Czarnecki. All rights reserved.

"""Application-owned profile resolution entrypoints (P1.1)."""

from intergrax.applications._shared.profile_resolution.engine import resolve_profile
from intergrax.applications._shared.profile_resolution.field_resolvers import (
    DEFAULT_FIELD_RESOLVERS,
    ProfileFieldResolver,
    ProfileFieldResolveResult,
)
from intergrax.applications._shared.profile_resolution.fingerprint import (
    compute_effective_profile_fingerprint,
)

__all__ = [
    "DEFAULT_FIELD_RESOLVERS",
    "ProfileFieldResolveResult",
    "ProfileFieldResolver",
    "compute_effective_profile_fingerprint",
    "resolve_profile",
]
