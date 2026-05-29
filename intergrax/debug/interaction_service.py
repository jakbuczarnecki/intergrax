# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Debug API alias for :mod:`intergrax.runtime.interactions.intake_service`."""

from __future__ import annotations

from intergrax.runtime.interactions.intake_service import (
    InteractionIntakeResult,
    InteractionIntakeService,
)

DebugInteractionIntakeService = InteractionIntakeService

__all__ = ["DebugInteractionIntakeService", "InteractionIntakeResult", "InteractionIntakeService"]
