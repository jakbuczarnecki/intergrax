# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Teams interaction adapter — wraps ``TeamsActivityInteractionAdapter``."""

from __future__ import annotations

from intergrax.runtime.interactions.adapters.teams_activity_adapter import TeamsActivityInteractionAdapter


class _TeamsInteractionAdapter(TeamsActivityInteractionAdapter):
    """Catalog facade over ``TeamsActivityInteractionAdapter`` (channel ``teams``)."""
