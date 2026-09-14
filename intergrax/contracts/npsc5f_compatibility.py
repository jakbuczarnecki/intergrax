# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed NPSC-5F v1/v2 compatibility failures (provider-neutral)."""

from __future__ import annotations


class Npsc5fCompatibilityError(Exception):
    """Base for deterministic fail-closed NPSC-5F compatibility failures."""


class LegacyCausalEvidenceIncompatibleError(Npsc5fCompatibilityError):
    """Legacy causal evidence cannot be interpreted or enriched safely."""


class AmbiguousLegacyExecutionIdentityError(Npsc5fCompatibilityError):
    """More than one canonical ExecutionId matches legacy correlation keys."""


class LegacyBackgroundExecutionIdentityIncompatibleError(Npsc5fCompatibilityError):
    """Legacy background identity record cannot yield canonical ExecutionId."""


class BackgroundExecutionIdentityConflictError(Npsc5fCompatibilityError):
    """Concurrent v1 and v2 durable identity records disagree."""


class ForbiddenPlatformCausalEvidenceV1WriteError(Npsc5fCompatibilityError):
    """Attempt to persist new platform_causal_evidence.v1 after cutover."""


class UnknownCausalEvidenceExportVersionError(Npsc5fCompatibilityError):
    """Requested causal evidence export schema version is not supported."""


class UnknownPlatformCausalEvidenceSchemaError(Npsc5fCompatibilityError):
    """Platform causal evidence payload schema_version is not supported."""
