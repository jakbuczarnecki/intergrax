# © Artur Czarnecki. All rights reserved.

"""Token Optimization shared contracts (Phase TOKEN-1A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class TokenOptimizationProfile(StrEnum):
    """Operator-facing optimization profile."""

    OFF = "off"
    MEASURE_ONLY = "measure_only"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class CompressionLevel(StrEnum):
    """Compression intensity for structural and semantic strategies."""

    OFF = "off"
    LIGHT = "light"
    MEDIUM = "medium"
    HIGH = "high"
    SEMANTIC = "semantic"


class OutputProfile(StrEnum):
    """Runtime-selected output verbosity profile."""

    MINIMAL = "minimal"
    TERSE = "terse"
    STANDARD = "standard"
    FULL = "full"
    AUDIT = "audit"
    MACHINE_RECEIPT = "machine_receipt"
    DEBUG_VERBOSE = "debug_verbose"


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """Output shaping policy resolved before model completion."""

    profile: OutputProfile = OutputProfile.STANDARD
    max_output_tokens: int | None = None
    require_sections: tuple[str, ...] = ()
    forbid_sections: tuple[str, ...] = ()
    preserve_exact: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationPolicy:
    """Top-level token optimization policy envelope."""

    enabled: bool = False
    profile: TokenOptimizationProfile = TokenOptimizationProfile.OFF
    compression_level: CompressionLevel = CompressionLevel.OFF
    allow_lossy: bool = False
    require_validation: bool = True
    fallback_on_validation_failure: bool = True
    emit_receipts: bool = True
    emit_observability: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TokenOptimizationSourceType(StrEnum):
    """Content source under optimization."""

    UNKNOWN = "unknown"
    PROMPT = "prompt"
    SYSTEM_POLICY = "system_policy"
    TOOL_CATALOG = "tool_catalog"
    TOOL_OUTPUT = "tool_output"
    TERMINAL_OUTPUT = "terminal_output"
    LOG_OUTPUT = "log_output"
    RAG_CONTEXT_PACK = "rag_context_pack"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
    MEMORY = "memory"
    CONVERSATION_HISTORY = "conversation_history"
    STRUCTURED_DATA = "structured_data"
    OUTPUT = "output"


class TokenCategory(StrEnum):
    """Token accounting category."""

    INPUT_CONTEXT = "input_context"
    TOOL_CATALOG = "tool_catalog"
    RAG_CONTEXT_PACK = "rag_context_pack"
    MEMORY = "memory"
    OUTPUT = "output"
    SYSTEM_POLICY = "system_policy"
    TOTAL = "total"


@dataclass(frozen=True, slots=True)
class TokenOptimizationAttribution:
    """Run/step attribution for measurements and telemetry."""

    run_id: str | None = None
    step_id: str | None = None
    workflow_id: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    model: str | None = None
    provider: str | None = None
    runtime_profile: str | None = None
    optimization_profile: TokenOptimizationProfile | None = None
    source_type: TokenOptimizationSourceType | None = None
    token_category: TokenCategory | None = None
    strategy_id: str | None = None
    plugin_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TokenOptimizationMechanism(StrEnum):
    """Policy-selectable optimization surface."""

    TOOL_OUTPUT_COMPACTION = "tool_output_compaction"
    TERMINAL_LOG_FILTERING = "terminal_log_filtering"
    TOOL_CATALOG_COMPACTION = "tool_catalog_compaction"
    RAG_CONTEXT_PACK_COMPRESSION = "rag_context_pack_compression"
    MEMORY_CONTEXT_PRUNING = "memory_context_pruning"
    CACHE_ALIGNMENT = "cache_alignment"
    OUTPUT_POLICY_SHAPING = "output_policy_shaping"
    STRUCTURED_DATA_COMPRESSION = "structured_data_compression"
    REVERSIBLE_M2M_REPRESENTATION = "reversible_m2m_representation"
    RETRIEVAL_ON_DEMAND = "retrieval_on_demand"
    DEDUPLICATION = "deduplication"


class TokenOptimizationStrategyKind(StrEnum):
    """Algorithm taxonomy applied by a mechanism."""

    LOSSLESS_NORMALIZATION = "lossless_normalization"
    LOSSLESS_STRUCTURAL_COMPRESSION = "lossless_structural_compression"
    DEDUPLICATION = "deduplication"
    EXTRACTIVE_FILTERING = "extractive_filtering"
    SCHEMA_MINIMIZATION = "schema_minimization"
    RANKING_PRUNING = "ranking_pruning"
    CACHE_PREFIX_STABILIZATION = "cache_prefix_stabilization"
    SAFE_LOSSY_SUMMARIZATION = "safe_lossy_summarization"
    SEMANTIC_COMPRESSION = "semantic_compression"
    REVERSIBLE_M2M_ENCODING = "reversible_m2m_encoding"
    RETRIEVAL_ON_DEMAND = "retrieval_on_demand"
    OUTPUT_VERBOSITY_SHAPING = "output_verbosity_shaping"


class StrategySafetyClass(StrEnum):
    """Safety classification for a strategy."""

    LOSSLESS = "lossless"
    LOSSY = "lossy"
    REVERSIBLE = "reversible"
    MEASUREMENT_ONLY = "measurement_only"
    POLICY_ONLY = "policy_only"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class TokenOptimizationStrategyRef:
    """Reference to a built-in or plugin strategy without execution logic."""

    strategy_id: str
    mechanism: TokenOptimizationMechanism
    kind: TokenOptimizationStrategyKind
    safety_class: StrategySafetyClass
    plugin_id: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationPluginCapability:
    """Declared capability of a custom optimizer plugin."""

    mechanism: TokenOptimizationMechanism
    strategy_kind: TokenOptimizationStrategyKind
    source_types: tuple[TokenOptimizationSourceType, ...] = ()
    lossless: bool = False
    lossy: bool = False
    reversible: bool = False
    requires_validation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationPluginDescriptor:
    """Descriptor-only contract for third-party optimizer registration."""

    plugin_id: str
    name: str
    version: str
    capabilities: tuple[TokenOptimizationPluginCapability, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TokenSavingsClaimConfidence(StrEnum):
    """Confidence level for savings claims."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    PROJECTED = "projected"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class TokenUsageMeasurement:
    """Single token count observation."""

    tokens: int
    category: TokenCategory
    source_type: TokenOptimizationSourceType
    attribution: TokenOptimizationAttribution | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class TokenSavingsMeasurement:
    """Baseline vs optimized token savings record."""

    baseline_tokens: int
    optimized_tokens: int
    saved_tokens: int
    saved_ratio: float
    confidence: TokenSavingsClaimConfidence
    category: TokenCategory
    source_type: TokenOptimizationSourceType
    strategy: TokenOptimizationStrategyRef | None = None
    attribution: TokenOptimizationAttribution | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.baseline_tokens < 0:
            raise ValueError("baseline_tokens cannot be negative")
        if self.optimized_tokens < 0:
            raise ValueError("optimized_tokens cannot be negative")
        expected_saved = self.baseline_tokens - self.optimized_tokens
        if self.saved_tokens != expected_saved:
            raise ValueError(
                f"saved_tokens must equal baseline_tokens - optimized_tokens "
                f"({expected_saved}), got {self.saved_tokens}"
            )
        if self.baseline_tokens > 0 and not 0.0 <= self.saved_ratio <= 1.0:
            raise ValueError("saved_ratio must be between 0.0 and 1.0 when baseline_tokens > 0")


class ProtectedRegionKind(StrEnum):
    """Exact-preservation region classification."""

    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    PATH = "path"
    URL = "url"
    ENV_VAR = "env_var"
    API_NAME = "api_name"
    CLASS_NAME = "class_name"
    FUNCTION_NAME = "function_name"
    COMMAND = "command"
    IDENTIFIER = "identifier"
    ENUM_VALUE = "enum_value"
    HASH = "hash"
    DATE = "date"
    VERSION = "version"
    EXACT_ERROR = "exact_error"
    SECURITY_WARNING = "security_warning"
    LEGAL_TEXT = "legal_text"
    POLICY_TEXT = "policy_text"
    EVIDENCE_REFERENCE = "evidence_reference"
    TENANT_IDENTIFIER = "tenant_identifier"


@dataclass(frozen=True, slots=True)
class ProtectedRegion:
    """Detected region that must be preserved during optimization."""

    kind: ProtectedRegionKind
    value: str
    start: int | None = None
    end: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProtectedRegionValidationStatus(StrEnum):
    """Outcome of protected-region validation."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ProtectedRegionValidationResult:
    """Validation outcome without parser/validator implementation."""

    status: ProtectedRegionValidationStatus
    regions_checked: int = 0
    regions_preserved: int = 0
    regions_failed: int = 0
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompressionReceiptRef:
    """Minimal receipt reference for later receipt builders."""

    receipt_id: str
    run_id: str | None = None
    step_id: str | None = None
    strategy_id: str | None = None
    original_hash: str | None = None
    optimized_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TokenOptimizationDecision(StrEnum):
    """Engine decision for an optimization attempt."""

    APPLY = "apply"
    BYPASS = "bypass"
    FALLBACK = "fallback"
    FAILED = "failed"
    MEASURE_ONLY = "measure_only"


class TokenOptimizationBypassReason(StrEnum):
    """Reason optimization was bypassed or fell back."""

    DISABLED = "disabled"
    UNSUPPORTED_SOURCE_TYPE = "unsupported_source_type"
    POLICY_DISALLOWED = "policy_disallowed"
    VALIDATION_REQUIRED = "validation_required"
    VALIDATION_FAILED = "validation_failed"
    NO_STRATEGY = "no_strategy"
    NO_SAVINGS = "no_savings"
    PROTECTED_REGION_RISK = "protected_region_risk"
    QUALITY_RISK = "quality_risk"
    PLUGIN_UNAVAILABLE = "plugin_unavailable"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class TokenOptimizationRequest:
    """Optimization request contract without execution logic."""

    content: str
    source_type: TokenOptimizationSourceType
    policy: TokenOptimizationPolicy = field(default_factory=TokenOptimizationPolicy)
    attribution: TokenOptimizationAttribution | None = None
    strategy: TokenOptimizationStrategyRef | None = None
    protected_regions: tuple[ProtectedRegion, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationResult:
    """Optimization result contract without execution logic."""

    content: str
    decision: TokenOptimizationDecision
    measurement: TokenSavingsMeasurement | None = None
    validation: ProtectedRegionValidationResult | None = None
    receipt_ref: CompressionReceiptRef | None = None
    strategy: TokenOptimizationStrategyRef | None = None
    fallback_used: bool = False
    bypass_reason: TokenOptimizationBypassReason | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _validate_non_negative_optional_token(
    value: int | None,
    field_name: str,
) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


class ContextFragmentPriority(StrEnum):
    """Priority tier for context packing decisions."""

    MUST_KEEP = "must_keep"
    HIGH_PRIORITY = "high_priority"
    COMPRESSIBLE = "compressible"
    DROPPABLE = "droppable"


class ContextPackingDecisionKind(StrEnum):
    """Per-fragment packing action aligned with receipt/report language."""

    KEEP = "keep"
    COMPACT = "compact"
    DEDUPLICATE = "deduplicate"
    DROP = "drop"
    TRUNCATE = "truncate"
    BYPASS = "bypass"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ContextPackingBudget:
    """Token budget envelope for context packing without counting logic."""

    max_input_tokens: int | None = None
    reserved_output_tokens: int | None = None
    target_context_tokens: int | None = None
    hard_context_limit: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_optional_token(self.max_input_tokens, "max_input_tokens")
        _validate_non_negative_optional_token(
            self.reserved_output_tokens, "reserved_output_tokens"
        )
        _validate_non_negative_optional_token(
            self.target_context_tokens, "target_context_tokens"
        )
        _validate_non_negative_optional_token(
            self.hard_context_limit, "hard_context_limit"
        )
        if (
            self.target_context_tokens is not None
            and self.hard_context_limit is not None
            and self.target_context_tokens > self.hard_context_limit
        ):
            raise ValueError(
                "target_context_tokens cannot exceed hard_context_limit"
            )
        if (
            self.reserved_output_tokens is not None
            and self.max_input_tokens is not None
            and self.reserved_output_tokens > self.max_input_tokens
        ):
            raise ValueError(
                "reserved_output_tokens cannot exceed max_input_tokens"
            )


@dataclass(frozen=True, slots=True)
class ContextPackingDecision:
    """Per-fragment packing decision for dedupe and budget-aware packing."""

    fragment_id: str
    decision: ContextPackingDecisionKind
    priority: ContextFragmentPriority
    reason: str | None = None
    original_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    related_fragment_ids: tuple[str, ...] = ()
    strategy: TokenOptimizationStrategyRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("fragment_id cannot be empty")
        _validate_non_negative_optional_token(self.original_tokens, "original_tokens")
        _validate_non_negative_optional_token(self.optimized_tokens, "optimized_tokens")
        _validate_non_negative_optional_token(self.saved_tokens, "saved_tokens")
        if (
            self.original_tokens is not None
            and self.optimized_tokens is not None
            and self.saved_tokens is not None
            and self.saved_tokens != self.original_tokens - self.optimized_tokens
        ):
            raise ValueError(
                "saved_tokens must equal original_tokens - optimized_tokens"
            )


@dataclass(frozen=True, slots=True)
class ContextDeduplicationMetadata:
    """Cross-fragment duplicate linkage without dedupe execution logic."""

    duplicate_of_fragment_id: str | None = None
    duplicate_fragment_ids: tuple[str, ...] = ()
    dedupe_key: str | None = None
    exact: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dedupe_key is not None and not self.dedupe_key:
            raise ValueError("dedupe_key cannot be empty")
        if any(not fragment_id for fragment_id in self.duplicate_fragment_ids):
            raise ValueError("duplicate_fragment_ids cannot contain empty strings")


@dataclass(frozen=True, slots=True)
class ContextFragmentPackingMetadata:
    """Per-fragment packing metadata for priority-tiered context packing."""

    priority: ContextFragmentPriority = ContextFragmentPriority.COMPRESSIBLE
    required: bool = False
    protected: bool = False
    budget: ContextPackingBudget | None = None
    deduplication: ContextDeduplicationMetadata | None = None
    decisions: tuple[ContextPackingDecision, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.required and self.priority is ContextFragmentPriority.DROPPABLE:
            raise ValueError(
                "required fragments cannot have droppable priority"
            )
        if self.protected and self.priority is ContextFragmentPriority.DROPPABLE:
            raise ValueError(
                "protected fragments cannot have droppable priority"
            )


@dataclass(frozen=True, slots=True)
class ContextPackingReceiptMetadata:
    """Receipt explanation metadata for context packing without receipt builders."""

    budget: ContextPackingBudget | None = None
    decisions: tuple[ContextPackingDecision, ...] = ()
    total_original_tokens: int | None = None
    total_optimized_tokens: int | None = None
    total_saved_tokens: int | None = None
    strategy_breakdown: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_optional_token(
            self.total_original_tokens, "total_original_tokens"
        )
        _validate_non_negative_optional_token(
            self.total_optimized_tokens, "total_optimized_tokens"
        )
        _validate_non_negative_optional_token(
            self.total_saved_tokens, "total_saved_tokens"
        )
        if (
            self.total_original_tokens is not None
            and self.total_optimized_tokens is not None
            and self.total_saved_tokens is not None
            and self.total_saved_tokens
            != self.total_original_tokens - self.total_optimized_tokens
        ):
            raise ValueError(
                "total_saved_tokens must equal total_original_tokens - total_optimized_tokens"
            )
        for strategy_id, saved in self.strategy_breakdown.items():
            if saved < 0:
                raise ValueError(
                    f"strategy_breakdown[{strategy_id!r}] cannot be negative"
                )
