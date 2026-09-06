# © Artur Czarnecki. All rights reserved.

"""Pre-context assembly policy gate (CE-4.7)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment, ContextFragmentSource


@dataclass(frozen=True, slots=True)
class ContextAssemblyPolicyResult:
    allowed: bool
    errors: tuple[str, ...] = ()


def run_pre_context_policy_gate(
    request: ContextAssemblyRequest,
    *,
    collected: tuple[ContextFragment, ...] | None = None,
) -> ContextAssemblyPolicyResult:
    """
    Validate assembly policy before format/validate.

    Mirrors ``pre_context_policy_audit`` hook parity — no silent bypass of
    ``required_sources`` / ``excluded_sources`` on the request.

    ``required_sources`` are enforced only when ``collected`` is provided
    (post-collect gate). Pre-collect admission uses provider eligibility checks.
    """
    errors: list[str] = []
    if request.budget_policy.max_chars < 1:
        errors.append("budget_policy.max_chars must be >= 1")

    if collected is not None:
        present_sources = {fragment.source for fragment in collected}
        for required in request.required_sources:
            if required not in present_sources:
                errors.append(f"required source missing after collect: {required.value}")

        for fragment in collected:
            if fragment.source in request.excluded_sources:
                errors.append(f"excluded source present after collect: {fragment.source.value}")

    return ContextAssemblyPolicyResult(allowed=not errors, errors=tuple(errors))
