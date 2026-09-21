# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map CodeCraft orchestrator CraftResult codes to gap synthesis outcomes (UCA-6A-R)."""

from __future__ import annotations

from intergrax.codecraft.contracts import CraftResult
from intergrax.contracts.codecraft.gap_synthesis import CodeCraftGapSynthesisOutcome
from intergrax.runtime.codecraft.artifact_reference import artifact_reference_for_craft

_UNAVAILABLE_ERRORS: frozenset[str] = frozenset(
    {
        "codecraft_profile_missing",
        "sandbox_session_not_configured",
        "isolation_requirement_unsatisfied",
        "network_egress_requirement_unsatisfied",
        "network_egress_allowlist_requirement_unsatisfied",
        "hosted_substrate_unavailable",
        "codecraft_execution_scope_unavailable",
    },
)

_HITL_PENDING_ERRORS: frozenset[str] = frozenset({"hitl_pending"})

_HITL_DENIED_ERRORS: frozenset[str] = frozenset({"hitl_denied"})

_BLOCKED_ERRORS: frozenset[str] = frozenset(
    {
        "codecraft_mode_disabled",
        "craft_ownership_mismatch",
        "craft_session_not_found",
        "codecraft_tenant_mismatch",
        "codecraft_task_mismatch",
        "codecraft_run_mismatch",
    },
)


def gap_synthesis_outcome_for_craft_error(error: str) -> CodeCraftGapSynthesisOutcome:
    if error in _HITL_PENDING_ERRORS:
        return CodeCraftGapSynthesisOutcome.REQUIRES_HITL
    if error in _HITL_DENIED_ERRORS or error in _BLOCKED_ERRORS:
        return CodeCraftGapSynthesisOutcome.BLOCKED
    if error in _UNAVAILABLE_ERRORS:
        return CodeCraftGapSynthesisOutcome.UNAVAILABLE
    return CodeCraftGapSynthesisOutcome.FAILED


def gap_synthesis_outcome_for_iterate_result(
    result: CraftResult,
) -> CodeCraftGapSynthesisOutcome:
    error = result.error or ""
    if error:
        return gap_synthesis_outcome_for_craft_error(error)
    if result.success and result.verdict == "continue":
        return CodeCraftGapSynthesisOutcome.FAILED
    if result.success and result.verdict == "promote":
        return CodeCraftGapSynthesisOutcome.SUCCEEDED
    if result.verdict == "revise":
        return CodeCraftGapSynthesisOutcome.FAILED
    return CodeCraftGapSynthesisOutcome.FAILED


__all__ = [
    "artifact_reference_for_craft",
    "gap_synthesis_outcome_for_craft_error",
    "gap_synthesis_outcome_for_iterate_result",
]
