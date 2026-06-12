# © Artur Czarnecki. All rights reserved.

"""Context validation for assembled windows (CE-3.10)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.contracts import AssembledContext, ContextAssemblyRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight


@dataclass(frozen=True, slots=True)
class ContextValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


class DefaultContextValidator:
    """Runs never-overflow preflight before LLM invocation."""

    def validate(
        self,
        assembled: AssembledContext,
        request: ContextAssemblyRequest,
        *,
        runtime_config: RuntimeConfig,
        max_output_tokens: int | None = None,
    ) -> ContextValidationResult:
        try:
            verify_context_preflight(
                list(assembled.messages),
                runtime_config.llm_adapter,
                max_output_tokens=max_output_tokens,
            )
        except ValueError as exc:
            return ContextValidationResult(valid=False, errors=(str(exc),))
        return ContextValidationResult(valid=True)
