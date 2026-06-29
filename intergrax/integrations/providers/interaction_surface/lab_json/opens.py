# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level lab JSON openers — internal to the lab_json integration package.

Only this module may construct ``LabJsonInteractionAdapter`` instances for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.interaction_surface.lab_json.adapter import _LabJsonIntegrationAdapter
from intergrax.integrations.providers.interaction_surface.lab_json.config import LabJsonIntegrationConfig
from intergrax.integrations.providers.interaction_surface.lab_json.integration import LabJsonInteractionSurfaceIntegration
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


def open_lab_json_interaction_surface(
    config: LabJsonIntegrationConfig,
    *,
    implementation: Optional[InteractionAdapter] = None,
) -> LabJsonInteractionSurfaceIntegration:
    del config
    if implementation is not None:
        if isinstance(implementation, LabJsonInteractionSurfaceIntegration):
            return implementation
        return LabJsonInteractionSurfaceIntegration.from_client(implementation)
    return LabJsonInteractionSurfaceIntegration.from_client(_LabJsonIntegrationAdapter())
