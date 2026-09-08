# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability metering domain errors (CAPABILITY-CATALOG-1 Stage 13)."""


class CapabilityMeteringError(RuntimeError):
    """Base error for capability usage metering failures."""


class CapabilityUsageConflictError(CapabilityMeteringError):
    """Duplicate event_id with conflicting usage evidence."""
