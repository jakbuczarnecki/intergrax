# © Artur Czarnecki. All rights reserved.

"""Capabilities for DIAG-FUNCTIONAL-Q4 model-routing qualification."""

from __future__ import annotations

MODEL_ROUTING_QUALIFICATION_CAPABILITY = "local.workspace.model_routing_qualification"

CAPABILITIES: tuple[str, ...] = (MODEL_ROUTING_QUALIFICATION_CAPABILITY,)

__all__ = ["CAPABILITIES", "MODEL_ROUTING_QUALIFICATION_CAPABILITY"]
