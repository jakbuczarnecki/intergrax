# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow errors (SELF-HEALING R2)."""

from __future__ import annotations


class SelfHealingWorkflowError(Exception):
    """Base workflow orchestration error."""


class SelfHealingWorkflowStateError(SelfHealingWorkflowError):
    """Illegal lifecycle transition or tenant mismatch."""


class SelfHealingWorkflowPluginError(SelfHealingWorkflowError):
    """Plugin failure — contained; must not impersonate success."""


class SelfHealingWorkflowPluginFailedError(SelfHealingWorkflowPluginError):
    """Plugin raised or timed out — status PLUGIN_FAILED."""


class SelfHealingWorkflowValidationError(SelfHealingWorkflowError):
    """Validation did not pass — rollback may be required."""


class SelfHealingWorkflowGovernanceError(SelfHealingWorkflowError):
    """Governance or admission blocked workflow progression."""


PLUGIN_FAILED = "PLUGIN_FAILED"

__all__ = [
    "PLUGIN_FAILED",
    "SelfHealingWorkflowError",
    "SelfHealingWorkflowGovernanceError",
    "SelfHealingWorkflowPluginFailedError",
    "SelfHealingWorkflowPluginError",
    "SelfHealingWorkflowStateError",
    "SelfHealingWorkflowValidationError",
]
