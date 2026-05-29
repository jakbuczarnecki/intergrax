# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab JSON interaction adapter — wraps ``LabJsonInteractionAdapter``."""

from __future__ import annotations

from intergrax.runtime.interactions.adapters.lab_json_adapter import LabJsonInteractionAdapter


class LabJsonIntegrationAdapter(LabJsonInteractionAdapter):
    """Catalog facade over ``LabJsonInteractionAdapter`` (runtime channel ``lab``)."""
