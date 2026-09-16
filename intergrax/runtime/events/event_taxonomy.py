# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Re-export contract-owned event taxonomy (OBS-EVOL-9)."""

from intergrax.contracts.event_taxonomy import (
    EventCategory,
    RetentionClass,
    category_for_event_kind,
)

__all__ = [
    "EventCategory",
    "RetentionClass",
    "category_for_event_kind",
]
