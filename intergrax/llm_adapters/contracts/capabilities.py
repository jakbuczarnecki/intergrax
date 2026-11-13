# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Declared model capabilities and fail-closed feature checks (ADR-PDC-03).

Capabilities are *declared*, never forced: an adapter states what it can do through its
``supports_*`` flag family, the provider registration states how an in-flight call can be
cancelled, and a requested feature that is not declared is denied *before* any provider call
with :class:`LLMCapabilityMismatchError`. A streamed tool call is never silently downgraded to
buffered output.

The ``LLMAdapter`` protocol is intentionally unchanged (adding a member would break structural
``isinstance`` checks for third-party adapters); ``supports_tool_streaming`` is therefore read
through ``getattr`` and an adapter that does not declare it is treated as *not supporting* it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationOutcome,
    TerminationResult,
)


class LLMFeature(StrEnum):
    """Feature an LLM call can request from an adapter."""

    STREAMING = "streaming"
    TOOLS = "tools"
    TOOL_STREAMING = "tool_streaming"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    STRICT_TOOL_ARGUMENTS = "strict_tool_arguments"


class CancelCapability(StrEnum):
    """What cancelling an in-flight provider call is declared to achieve."""

    TERMINATED = "terminated"
    TRANSPORT_CLOSED = "transport_closed"
    NOT_SUPPORTED = "not_supported"


#: ``LLMFeature`` -> name of the adapter method that declares it.
FEATURE_FLAG_METHODS: dict[LLMFeature, str] = {
    LLMFeature.STREAMING: "supports_streaming",
    LLMFeature.TOOLS: "supports_tools",
    LLMFeature.TOOL_STREAMING: "supports_tool_streaming",
    LLMFeature.STRUCTURED_OUTPUT: "supports_structured_output",
    LLMFeature.VISION: "supports_vision",
    LLMFeature.AUDIO_INPUT: "supports_audio_input",
    LLMFeature.AUDIO_OUTPUT: "supports_audio_output",
    LLMFeature.STRICT_TOOL_ARGUMENTS: "supports_strict_tool_argument_conformance",
}


class LLMCapabilityMismatchError(NotImplementedError):
    """A requested feature is not declared by the adapter; raised before the provider call.

    Subclasses ``NotImplementedError`` so callers (and tests) written against the former
    untyped base-class behaviour keep working.
    """

    def __init__(
        self,
        features: tuple[LLMFeature, ...] | LLMFeature,
        *,
        provider: str = "",
        model: str = "",
        detail: str = "",
    ) -> None:
        requested = (features,) if isinstance(features, LLMFeature) else tuple(features)
        self.features: tuple[LLMFeature, ...] = requested
        self.provider = provider
        self.model = model
        names = ", ".join(feature.value for feature in requested)
        message = detail or (f"capability mismatch: {names} not declared by provider={provider!r} model={model!r}")
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LLMDeclaredCapabilities:
    """Immutable snapshot of what an adapter and its provider registration declare."""

    streaming: bool = False
    tools: bool = False
    tool_streaming: bool = False
    structured_output: bool = False
    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False
    strict_tool_arguments: bool = False
    cancel: CancelCapability = CancelCapability.NOT_SUPPORTED

    def supports(self, feature: LLMFeature) -> bool:
        return bool(getattr(self, feature.value))

    def missing(self, features: tuple[LLMFeature, ...]) -> tuple[LLMFeature, ...]:
        return tuple(feature for feature in features if not self.supports(feature))


def read_feature_flag(adapter: object, feature: LLMFeature) -> bool:
    """Read one declared flag; an undeclared or non-``True`` value means *not supported*."""
    probe = getattr(adapter, FEATURE_FLAG_METHODS[feature], None)
    if not callable(probe):
        return False
    return probe() is True


def adapter_declared_flags(adapter: object) -> dict[LLMFeature, bool]:
    return {feature: read_feature_flag(adapter, feature) for feature in LLMFeature}


def declared_capabilities_of(
    adapter: object,
    *,
    cancel_for: Callable[[str], CancelCapability],
) -> LLMDeclaredCapabilities:
    """Assemble the declared snapshot; ``cancel_for`` resolves the provider's cancel capability."""
    flags = adapter_declared_flags(adapter)
    provider, _ = _identity(adapter)
    return LLMDeclaredCapabilities(
        streaming=flags[LLMFeature.STREAMING],
        tools=flags[LLMFeature.TOOLS],
        tool_streaming=flags[LLMFeature.TOOL_STREAMING],
        structured_output=flags[LLMFeature.STRUCTURED_OUTPUT],
        vision=flags[LLMFeature.VISION],
        audio_input=flags[LLMFeature.AUDIO_INPUT],
        audio_output=flags[LLMFeature.AUDIO_OUTPUT],
        strict_tool_arguments=flags[LLMFeature.STRICT_TOOL_ARGUMENTS],
        cancel=cancel_for(provider) if provider else CancelCapability.NOT_SUPPORTED,
    )


def cancel_capability_from_external_operation(
    capabilities: ExternalOperationCapabilities,
) -> CancelCapability:
    """Map the provider registration's cancellation affordances to the typed outcome.

    * native cancel declared          -> ``terminated`` (physical stop confirmed)
    * only a stream abort is declared -> ``transport_closed``
    * nothing declared                -> ``not_supported``
    """
    if capabilities.supports_native_cancel:
        return CancelCapability.TERMINATED
    if capabilities.supports_stream_abort:
        return CancelCapability.TRANSPORT_CLOSED
    return CancelCapability.NOT_SUPPORTED


def cancel_outcome_of(result: TerminationResult) -> CancelCapability:
    """Type an *observed* termination result as ``terminated`` / ``transport_closed`` / ``not_supported``.

    ``signal_failed`` is a runtime failure, not a cancel outcome, and is rejected.
    """
    outcome = result.outcome
    if outcome is TerminationOutcome.PHYSICAL_STOP_CONFIRMED:
        return CancelCapability.TERMINATED
    if outcome is TerminationOutcome.TRANSPORT_CLOSED:
        return CancelCapability.TRANSPORT_CLOSED
    if outcome is TerminationOutcome.NOT_SUPPORTED:
        return CancelCapability.NOT_SUPPORTED
    raise ValueError(f"termination outcome {outcome.value!r} is a failure, not a cancel outcome")


def _identity(adapter: object) -> tuple[str, str]:
    provider = getattr(adapter, "provider", "")
    provider_text = getattr(provider, "value", provider)
    return str(provider_text or ""), str(getattr(adapter, "model", "") or "")


def require_declared_features(adapter: object, *features: LLMFeature) -> None:
    """Fail closed before a provider call when ``adapter`` does not declare a requested feature."""
    missing = tuple(feature for feature in features if not read_feature_flag(adapter, feature))
    if missing:
        provider, model = _identity(adapter)
        raise LLMCapabilityMismatchError(missing, provider=provider, model=model)


__all__ = [
    "CancelCapability",
    "FEATURE_FLAG_METHODS",
    "LLMCapabilityMismatchError",
    "LLMDeclaredCapabilities",
    "LLMFeature",
    "adapter_declared_flags",
    "cancel_capability_from_external_operation",
    "cancel_outcome_of",
    "declared_capabilities_of",
    "read_feature_flag",
    "require_declared_features",
]
