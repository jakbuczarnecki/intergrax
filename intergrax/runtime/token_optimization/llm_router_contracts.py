# © Artur Czarnecki. All rights reserved.

"""LLM router contracts for Token Optimization configuration selection (TOKEN-9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineResult,
    TokenOptimizationRequest,
)
from intergrax.runtime.token_optimization.prompt_assembly import (
    CacheStablePromptAssemblyReport,
    CacheStablePromptState,
)


class TokenOptimizationRouterConfigurationId(StrEnum):
    NO_OPTIMIZATION = "no_optimization"
    EXACT_ONLY = "exact_only"
    EXTRACTIVE_ONLY = "extractive_only"
    PACKING_ONLY = "packing_only"
    EXACT_THEN_PACKING = "exact_then_packing"
    EXACT_THEN_EXTRACTIVE = "exact_then_extractive"
    EXTRACTIVE_THEN_EXACT = "extractive_then_exact"


class TokenOptimizationRouterReasonCode(StrEnum):
    CLEAN_NO_OP = "clean_no_op"
    EXACT_DUPLICATES = "exact_duplicates"
    NOISY_TOOL_OUTPUT = "noisy_tool_output"
    PRIORITY_PACKING = "priority_packing"
    MIXED_DEDUPLICATION_PACKING = "mixed_deduplication_packing"
    PROTECTED_OR_HIGH_RISK = "protected_or_high_risk"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class TokenOptimizationRouterRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TokenOptimizationRouterToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration_id: TokenOptimizationRouterConfigurationId
    reason_code: TokenOptimizationRouterReasonCode
    risk: TokenOptimizationRouterRisk
    review_required: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_review_rules(self) -> TokenOptimizationRouterToolInput:
        if self.risk is TokenOptimizationRouterRisk.HIGH and not self.review_required:
            raise ValueError("risk=HIGH requires review_required=True")
        if (
            self.reason_code is TokenOptimizationRouterReasonCode.PROTECTED_OR_HIGH_RISK
            and not self.review_required
        ):
            raise ValueError(
                "reason_code=PROTECTED_OR_HIGH_RISK requires review_required=True"
            )
        return self


class TokenOptimizationRouterTransport(StrEnum):
    NATIVE_TOOLS = "native_tools"
    STRUCTURED_OUTPUT = "structured_output"
    UNSUPPORTED = "unsupported"


class TokenOptimizationRouterStatus(StrEnum):
    ROUTED = "routed"
    NO_OPTIMIZATION = "no_optimization"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    INVALID_DECISION = "invalid_decision"
    UNSUPPORTED_ADAPTER = "unsupported_adapter"
    LLM_ERROR = "llm_error"


class TokenOptimizationRouterReason(StrEnum):
    POLICY_DISABLED = "policy_disabled"
    PROFILE_OFF = "profile_off"
    UNSUPPORTED_ADAPTER = "unsupported_adapter"
    CAPABILITY_RESOLUTION_FAILED = "capability_resolution_failed"
    NO_TOOL_CALL = "no_tool_call"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    UNEXPECTED_TOOL = "unexpected_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    UNKNOWN_CONFIGURATION = "unknown_configuration"
    SOURCE_TYPE_NOT_SUPPORTED = "source_type_not_supported"
    LOSSY_NOT_ALLOWED = "lossy_not_allowed"
    PACKING_INPUT_REQUIRED = "packing_input_required"
    PROTECTED_REGIONS_REQUIRE_REVIEW = "protected_regions_require_review"
    MODEL_REQUESTED_REVIEW = "model_requested_review"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    LLM_ERROR = "llm_error"
    PROMPT_ASSEMBLY_INTEGRITY_FAILED = "prompt_assembly_integrity_failed"


class TokenOptimizationPolicyOverrideReason(StrEnum):
    SECURITY_WARNING_REQUIRES_REVIEW = "security_warning_requires_review"


@dataclass(frozen=True, slots=True)
class TokenOptimizationLLMRouterPolicy:
    allow_structured_output_fallback: bool = True
    require_review_for_protected_lossy_content: bool = True
    minimum_confidence: float = 0.60
    execute_when_review_required: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0.0 and 1.0")
        if self.execute_when_review_required:
            raise ValueError("execute_when_review_required must remain False in TOKEN-9")


@dataclass(frozen=True, slots=True)
class TokenOptimizationLLMRouterRequest:
    request: TokenOptimizationRequest
    policy: TokenOptimizationLLMRouterPolicy
    request_id: str
    previous_prompt_cache_state: CacheStablePromptState | None = None


@dataclass(frozen=True, slots=True)
class TokenOptimizationLLMRouterResult:
    request_id: str
    status: TokenOptimizationRouterStatus
    reason: TokenOptimizationRouterReason | None
    transport: TokenOptimizationRouterTransport
    configuration_id: TokenOptimizationRouterConfigurationId | None
    reason_code: TokenOptimizationRouterReasonCode | None
    risk: TokenOptimizationRouterRisk | None
    review_required: bool | None
    confidence: float | None
    provider: str | None
    model: str | None
    tool_call_id: str | None
    pipeline_config: TokenOptimizationPipelineConfig | None
    pipeline_result: TokenOptimizationPipelineResult | None
    executed: bool
    prompt_cache_state: CacheStablePromptState | None = None
    prompt_assembly_report: CacheStablePromptAssemblyReport | None = None
    model_configuration_id: TokenOptimizationRouterConfigurationId | None = None
    model_reason_code: TokenOptimizationRouterReasonCode | None = None
    model_risk: TokenOptimizationRouterRisk | None = None
    model_review_required: bool | None = None
    policy_override_applied: bool = False
    policy_override_reason: TokenOptimizationPolicyOverrideReason | None = None
