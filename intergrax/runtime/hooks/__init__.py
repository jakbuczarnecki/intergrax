# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime hook system (architecture §42.3)."""

from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.hooks.parity import HookCoverage, HOOK_COVERAGE, hook_coverage

__all__ = [
    "HOOK_COVERAGE",
    "HookAction",
    "HookContext",
    "HookCoverage",
    "HookPoint",
    "HookRegistry",
    "HookResult",
    "hook_coverage",
]
