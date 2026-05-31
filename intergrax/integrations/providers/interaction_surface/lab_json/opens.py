# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level lab JSON openers — internal to the lab_json integration package.

Only this module may construct ``LabJsonInteractionAdapter`` instances for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.interaction_surface.lab_json.adapter import LabJsonIntegrationAdapter
from intergrax.integrations.providers.interaction_surface.lab_json.config import LabJsonIntegrationConfig
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


def open_lab_json_interaction_surface(
    config: LabJsonIntegrationConfig,
    *,
    implementation: Optional[InteractionAdapter] = None,
) -> InteractionAdapter:
    del config
    if implementation is not None:
        return implementation
    return LabJsonIntegrationAdapter()
