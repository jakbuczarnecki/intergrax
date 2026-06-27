# © Artur Czarnecki. All rights reserved.

"""Tier-3 execution mode — compatibility re-export."""

from __future__ import annotations

from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.runtime.policy.execution_mode_defaults import runtime_policies_for_execution_mode

__all__ = ["ExecutionMode", "runtime_policies_for_execution_mode"]
