# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hook / middleware parity vs canon §42.3 and §42.20 (Appendix B.06)."""

from __future__ import annotations

from enum import Enum

from intergrax.runtime.hooks.hook_point import HookPoint


class HookCoverage(str, Enum):
    """Implementation status for a :class:`HookPoint`."""

    WIRED = "wired"
    PARTIAL = "partial"
    NOT_WIRED = "not_wired"


HOOK_COVERAGE: dict[HookPoint, HookCoverage] = {
    HookPoint.BEFORE_TASK_INTAKE: HookCoverage.WIRED,
    HookPoint.AFTER_TASK_INTAKE: HookCoverage.WIRED,
    HookPoint.BEFORE_CLASSIFICATION: HookCoverage.WIRED,
    HookPoint.AFTER_CLASSIFICATION: HookCoverage.WIRED,
    HookPoint.BEFORE_PLANNING: HookCoverage.WIRED,
    HookPoint.AFTER_PLANNING: HookCoverage.WIRED,
    HookPoint.BEFORE_AGENT_SELECTION: HookCoverage.WIRED,
    HookPoint.AFTER_AGENT_SELECTION: HookCoverage.WIRED,
    HookPoint.BEFORE_CONTEXT_BUILD: HookCoverage.WIRED,
    HookPoint.AFTER_CONTEXT_BUILD: HookCoverage.WIRED,
    HookPoint.BEFORE_STEP: HookCoverage.WIRED,
    HookPoint.AFTER_STEP: HookCoverage.WIRED,
    HookPoint.BEFORE_TOOL_CALL: HookCoverage.WIRED,
    HookPoint.AFTER_TOOL_CALL: HookCoverage.WIRED,
    HookPoint.BEFORE_VALIDATION: HookCoverage.WIRED,
    HookPoint.AFTER_VALIDATION: HookCoverage.WIRED,
    HookPoint.BEFORE_DECISION: HookCoverage.WIRED,
    HookPoint.AFTER_DECISION: HookCoverage.WIRED,
    HookPoint.BEFORE_INTERRUPT: HookCoverage.WIRED,
    HookPoint.AFTER_INTERRUPT: HookCoverage.WIRED,
    HookPoint.BEFORE_HUMAN_APPROVAL: HookCoverage.WIRED,
    HookPoint.AFTER_HUMAN_APPROVAL: HookCoverage.WIRED,
    HookPoint.BEFORE_RETRY: HookCoverage.WIRED,
    HookPoint.AFTER_RETRY: HookCoverage.WIRED,
    HookPoint.BEFORE_HANDOFF: HookCoverage.WIRED,
    HookPoint.AFTER_HANDOFF: HookCoverage.WIRED,
    HookPoint.BEFORE_FINALIZATION: HookCoverage.WIRED,
    HookPoint.AFTER_FINALIZATION: HookCoverage.WIRED,
    HookPoint.BEFORE_TRACE_PERSIST: HookCoverage.WIRED,
    HookPoint.AFTER_TRACE_PERSIST: HookCoverage.WIRED,
}


def hook_coverage(point: HookPoint) -> HookCoverage:
    return HOOK_COVERAGE.get(point, HookCoverage.NOT_WIRED)


def list_hook_points_by_coverage(coverage: HookCoverage) -> list[HookPoint]:
    return [point for point, status in HOOK_COVERAGE.items() if status == coverage]
