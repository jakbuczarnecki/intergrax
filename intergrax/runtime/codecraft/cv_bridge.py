# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CVL bridge — iteration verdict from gate/exec/test (ECC-2)."""

from __future__ import annotations

from intergrax.codecraft.contracts import CraftVerdict, StaticGateResult


def iteration_cvl_verdict(
    *,
    static_gate: StaticGateResult,
    exec_success: bool,
    test_passed: bool | None,
    iteration: int,
    max_iterations: int,
) -> CraftVerdict:
    """Map deterministic craft signals to continue/revise/promote/abort."""
    if not static_gate.passed:
        return "revise"
    if not exec_success:
        return "revise" if iteration < max_iterations else "abort"
    if test_passed is False:
        return "revise" if iteration < max_iterations else "abort"
    if iteration >= max_iterations:
        return "promote"
    return "continue"
