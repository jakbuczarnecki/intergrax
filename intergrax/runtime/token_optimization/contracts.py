# © Artur Czarnecki. All rights reserved.

"""Token Optimization shared contracts (Phase TOKEN-1A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol


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


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")


def _validate_non_empty_id_tuple(ids: tuple[str, ...], field_name: str) -> None:
    if any(not layer_id for layer_id in ids):
        raise ValueError(f"{field_name} cannot contain empty strings")


def _validate_non_negative_optional_index(
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


class TokenOptimizationLayerDecision(StrEnum):
    """Per-layer outcome within a sequential optimization pipeline."""

    APPLY = "apply"
    BYPASS = "bypass"
    FALLBACK = "fallback"
    OVERRIDE_PREVIOUS = "override_previous"
    REVERT_TO_ORIGINAL = "revert_to_original"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TokenOptimizationLayerDescriptor:
    """Descriptor for a built-in, custom, or plugin optimization layer."""

    layer_id: str
    name: str
    version: str
    strategy: TokenOptimizationStrategyRef
    supported_source_types: tuple[TokenOptimizationSourceType, ...] = ()
    safety_class: StrategySafetyClass | None = None
    plugin_id: str | None = None
    built_in: bool = False
    requires_validation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.layer_id, "layer_id")
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.version, "version")
        if self.plugin_id is not None:
            _validate_non_empty_string(self.plugin_id, "plugin_id")
        for source_type in self.supported_source_types:
            if not isinstance(source_type, TokenOptimizationSourceType):
                raise ValueError(
                    "supported_source_types cannot contain invalid entries"
                )


@dataclass(frozen=True, slots=True)
class TokenOptimizationLayerContext:
    """Pipeline position and lineage for a single layer invocation."""

    pipeline_id: str | None = None
    layer_index: int | None = None
    previous_layer_ids: tuple[str, ...] = ()
    applied_layer_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_optional_index(self.layer_index, "layer_index")
        _validate_non_empty_id_tuple(self.previous_layer_ids, "previous_layer_ids")
        _validate_non_empty_id_tuple(self.applied_layer_ids, "applied_layer_ids")


@dataclass(frozen=True, slots=True)
class TokenOptimizationLayerRequest:
    """Layer input carrying immutable baseline and mutable working content."""

    original_content: str
    current_content: str
    source_type: TokenOptimizationSourceType
    policy: TokenOptimizationPolicy = field(default_factory=TokenOptimizationPolicy)
    attribution: TokenOptimizationAttribution | None = None
    layer_context: TokenOptimizationLayerContext | None = None
    strategy: TokenOptimizationStrategyRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original_content is None:
            raise ValueError("original_content must not be None")
        if self.current_content is None:
            raise ValueError("current_content must not be None")


@dataclass(frozen=True, slots=True)
class TokenOptimizationLayerResult:
    """Per-layer outcome with explicit override and fallback visibility."""

    layer_id: str
    output_content: str
    decision: TokenOptimizationLayerDecision
    measurement: TokenSavingsMeasurement | None = None
    validation: ProtectedRegionValidationResult | None = None
    receipt_metadata: Mapping[str, Any] = field(default_factory=dict)
    previous_changes_overridden: bool = False
    overridden_layer_ids: tuple[str, ...] = ()
    override_reason: str | None = None
    fallback_used: bool = False
    bypass_reason: TokenOptimizationBypassReason | None = None
    strategy: TokenOptimizationStrategyRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.layer_id, "layer_id")
        if self.output_content is None:
            raise ValueError("output_content must not be None")
        _validate_non_empty_id_tuple(self.overridden_layer_ids, "overridden_layer_ids")
        if self.previous_changes_overridden and not self.override_reason:
            raise ValueError(
                "override_reason should be provided when previous_changes_overridden is True"
            )
        if (
            self.decision is TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS
            and not self.previous_changes_overridden
        ):
            raise ValueError(
                "previous_changes_overridden must be True when decision is OVERRIDE_PREVIOUS"
            )
        if (
            self.decision is TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL
            and not self.previous_changes_overridden
        ):
            raise ValueError(
                "previous_changes_overridden must be True when decision is REVERT_TO_ORIGINAL"
            )


class TokenOptimizationLayer(Protocol):
    """Contract for built-in, custom, or plugin optimization layers."""

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        ...

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        ...


@dataclass(frozen=True, slots=True)
class TokenOptimizationLayerRef:
    """Ordered layer reference for pipeline composition."""

    layer_id: str
    plugin_id: str | None = None
    version: str | None = None
    enabled: bool = True
    order: int | None = None
    required: bool = False
    settings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.layer_id, "layer_id")
        if self.plugin_id is not None:
            _validate_non_empty_string(self.plugin_id, "plugin_id")
        if self.version is not None:
            _validate_non_empty_string(self.version, "version")
        _validate_non_negative_optional_index(self.order, "order")


class TokenOptimizationPipelineMode(StrEnum):
    """How a pipeline resolves its ordered layer list."""

    DEFAULT = "default"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class TokenOptimizationPipelineConfig:
    """Developer-configurable optimization pipeline composition."""

    pipeline_id: str
    mode: TokenOptimizationPipelineMode = TokenOptimizationPipelineMode.DEFAULT
    layers: tuple[TokenOptimizationLayerRef, ...] = ()
    allow_repeated_layers: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.pipeline_id, "pipeline_id")
        if self.mode is TokenOptimizationPipelineMode.REPLACE and not self.layers:
            raise ValueError("layers must not be empty when mode is REPLACE")
        for layer_ref in self.layers:
            _validate_non_negative_optional_index(layer_ref.order, "order")
        if not self.allow_repeated_layers:
            enabled_ids = [layer.layer_id for layer in self.layers if layer.enabled]
            if len(enabled_ids) != len(set(enabled_ids)):
                raise ValueError(
                    "enabled layer_id values must be unique when "
                    "allow_repeated_layers is False"
                )


@dataclass(frozen=True, slots=True)
class TokenOptimizationPipelineResult:
    """Aggregate outcome after sequential layer execution."""

    pipeline_id: str
    original_content: str
    final_content: str
    layer_results: tuple[TokenOptimizationLayerResult, ...] = ()
    applied_layer_ids: tuple[str, ...] = ()
    bypassed_layer_ids: tuple[str, ...] = ()
    failed_layer_ids: tuple[str, ...] = ()
    fallback_used: bool = False
    aggregate_measurement: TokenSavingsMeasurement | None = None
    receipt_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.pipeline_id, "pipeline_id")
        if self.original_content is None:
            raise ValueError("original_content must not be None")
        if self.final_content is None:
            raise ValueError("final_content must not be None")
        _validate_non_empty_id_tuple(self.applied_layer_ids, "applied_layer_ids")
        _validate_non_empty_id_tuple(self.bypassed_layer_ids, "bypassed_layer_ids")
        _validate_non_empty_id_tuple(self.failed_layer_ids, "failed_layer_ids")


class PromptCacheMode(StrEnum):
    """Requested prompt-cache behavior mode (provider-neutral)."""

    OFF = "off"
    PROVIDER_DEFAULT = "provider_default"
    EXPLICIT_BREAKPOINTS = "explicit_breakpoints"
    CACHE_KEY = "cache_key"
    SESSION_AFFINITY = "session_affinity"


class PromptCacheInvalidationReason(StrEnum):
    """Why a stable prompt-cache prefix became invalid (no raw content)."""

    NONE = "none"
    DISABLED = "disabled"
    UNSUPPORTED_PROVIDER = "unsupported_provider"
    PREFIX_CHANGED = "prefix_changed"
    TOOL_ENVELOPE_CHANGED = "tool_envelope_changed"
    DYNAMIC_DATA_IN_PREFIX = "dynamic_data_in_prefix"
    TTL_EXPIRED = "ttl_expired"
    CACHE_KEY_CHANGED = "cache_key_changed"
    SESSION_CHANGED = "session_changed"
    PROVIDER_NOT_REPORTED = "provider_not_reported"
    UNKNOWN = "unknown"


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _validate_optional_non_negative_float(value: float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _validate_stripped_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _validate_optional_stripped_non_empty(value: str | None, field_name: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


@dataclass(frozen=True, slots=True)
class PromptCacheProviderCapabilities:
    """Declared prompt-cache capabilities of an LLM provider."""

    provider: str
    supports_prompt_caching: bool = False
    supports_automatic_caching: bool = False
    supports_explicit_breakpoints: bool = False
    supports_cache_key: bool = False
    supports_cache_retention_ttl: bool = False
    supports_cache_usage_tokens: bool = False
    supports_cache_creation_tokens: bool = False
    supports_cache_read_tokens: bool = False
    requires_session_affinity: bool = False
    max_cache_breakpoints: int | None = None
    default_ttl_seconds: int | None = None
    max_ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        _validate_stripped_non_empty(self.provider, "provider")
        _validate_optional_non_negative_int(
            self.max_cache_breakpoints, "max_cache_breakpoints"
        )
        _validate_optional_non_negative_int(self.default_ttl_seconds, "default_ttl_seconds")
        _validate_optional_non_negative_int(self.max_ttl_seconds, "max_ttl_seconds")
        if (
            self.default_ttl_seconds is not None
            and self.max_ttl_seconds is not None
            and self.default_ttl_seconds > self.max_ttl_seconds
        ):
            raise ValueError(
                "default_ttl_seconds cannot exceed max_ttl_seconds"
            )
        if not self.supports_prompt_caching:
            feature_flags = (
                self.supports_automatic_caching,
                self.supports_explicit_breakpoints,
                self.supports_cache_key,
                self.supports_cache_retention_ttl,
                self.supports_cache_usage_tokens,
                self.supports_cache_creation_tokens,
                self.supports_cache_read_tokens,
                self.requires_session_affinity,
            )
            if any(feature_flags):
                raise ValueError(
                    "provider must not claim specific prompt-cache features "
                    "when supports_prompt_caching is False"
                )


@dataclass(frozen=True, slots=True)
class PromptCachePolicy:
    """Shared prompt-cache policy envelope (no provider API calls)."""

    enabled: bool = False
    mode: PromptCacheMode = PromptCacheMode.OFF
    stable_prefix_required: bool = False
    append_only_thread_required: bool = False
    allow_explicit_breakpoints: bool = False
    allow_cache_key: bool = False
    allow_session_affinity: bool = False
    cache_key_scope: str | None = None
    max_cache_breakpoints: int | None = None
    requested_ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not self.enabled and self.mode is not PromptCacheMode.OFF:
            raise ValueError("mode must be OFF when enabled is False")
        if self.enabled and self.mode is PromptCacheMode.OFF:
            raise ValueError("mode must not be OFF when enabled is True")
        _validate_optional_stripped_non_empty(self.cache_key_scope, "cache_key_scope")
        _validate_optional_non_negative_int(
            self.max_cache_breakpoints, "max_cache_breakpoints"
        )
        _validate_optional_non_negative_int(
            self.requested_ttl_seconds, "requested_ttl_seconds"
        )
        if (
            self.mode is PromptCacheMode.EXPLICIT_BREAKPOINTS
            and not self.allow_explicit_breakpoints
        ):
            raise ValueError(
                "EXPLICIT_BREAKPOINTS requires allow_explicit_breakpoints=True"
            )
        if self.mode is PromptCacheMode.CACHE_KEY:
            if not self.allow_cache_key:
                raise ValueError("CACHE_KEY requires allow_cache_key=True")
            if self.cache_key_scope is None:
                raise ValueError("CACHE_KEY requires cache_key_scope")
        if (
            self.mode is PromptCacheMode.SESSION_AFFINITY
            and not self.allow_session_affinity
        ):
            raise ValueError(
                "SESSION_AFFINITY requires allow_session_affinity=True"
            )

    @classmethod
    def disabled(cls) -> "PromptCachePolicy":
        return cls(enabled=False, mode=PromptCacheMode.OFF)


@dataclass(frozen=True, slots=True)
class PromptCacheUsageSnapshot:
    """Provider-reported prompt-cache usage signals (not content-reduction savings)."""

    provider: str
    model: str | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cached_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cache_hit_ratio: float | None = None
    cache_latency_delta_estimate_ms: float | None = None
    cache_discount_estimate: float | None = None

    def __post_init__(self) -> None:
        _validate_stripped_non_empty(self.provider, "provider")
        _validate_optional_non_negative_int(self.cache_read_tokens, "cache_read_tokens")
        _validate_optional_non_negative_int(
            self.cache_creation_tokens, "cache_creation_tokens"
        )
        _validate_optional_non_negative_int(
            self.cached_input_tokens, "cached_input_tokens"
        )
        _validate_optional_non_negative_int(
            self.uncached_input_tokens, "uncached_input_tokens"
        )
        if self.cache_hit_ratio is not None and not 0.0 <= self.cache_hit_ratio <= 1.0:
            raise ValueError("cache_hit_ratio must be between 0.0 and 1.0 inclusive")
        _validate_optional_non_negative_float(
            self.cache_latency_delta_estimate_ms, "cache_latency_delta_estimate_ms"
        )
        _validate_optional_non_negative_float(
            self.cache_discount_estimate, "cache_discount_estimate"
        )


@dataclass(frozen=True, slots=True)
class PromptCacheAttribution:
    """Attribution that keeps provider-cache usage separate from content reduction."""

    policy: PromptCachePolicy
    provider_capabilities: PromptCacheProviderCapabilities | None = None
    usage: PromptCacheUsageSnapshot | None = None
    prefix_hash: str | None = None
    prefix_stability_status: str | None = None
    invalidation_reason: PromptCacheInvalidationReason = PromptCacheInvalidationReason.NONE
    content_reduction_strategy: str | None = None
    content_saved_chars: int | None = None
    content_saved_tokens: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_stripped_non_empty(self.prefix_hash, "prefix_hash")
        _validate_optional_stripped_non_empty(
            self.prefix_stability_status, "prefix_stability_status"
        )
        _validate_optional_stripped_non_empty(
            self.content_reduction_strategy, "content_reduction_strategy"
        )
        _validate_optional_non_negative_int(
            self.content_saved_chars, "content_saved_chars"
        )
        _validate_optional_non_negative_int(
            self.content_saved_tokens, "content_saved_tokens"
        )
        if (
            self.provider_capabilities is not None
            and self.usage is not None
            and self.provider_capabilities.provider != self.usage.provider
        ):
            raise ValueError(
                "provider_capabilities.provider and usage.provider must match"
            )

    def has_provider_cache_usage(self) -> bool:
        if self.usage is None:
            return False
        return any(
            value is not None
            for value in (
                self.usage.cache_read_tokens,
                self.usage.cache_creation_tokens,
                self.usage.cached_input_tokens,
                self.usage.uncached_input_tokens,
                self.usage.cache_hit_ratio,
                self.usage.cache_latency_delta_estimate_ms,
                self.usage.cache_discount_estimate,
            )
        )

    def has_content_reduction(self) -> bool:
        return (
            self.content_reduction_strategy is not None
            or self.content_saved_chars is not None
            or self.content_saved_tokens is not None
        )


class CacheAwareCompactionTarget(StrEnum):
    """Where compaction would rewrite relative to a cacheable prefix."""

    STABLE_PREFIX = "stable_prefix"
    DYNAMIC_TAIL = "dynamic_tail"
    COLD_HISTORY = "cold_history"
    FULL_THREAD = "full_thread"
    UNKNOWN = "unknown"


class CacheAwareCompactionDecision(StrEnum):
    """Whether compaction should run, wait, skip, or require review."""

    RUN = "run"
    DEFER = "defer"
    BYPASS = "bypass"
    REQUIRE_MANUAL_REVIEW = "require_manual_review"


class CacheAwareCompactionReason(StrEnum):
    """Why a cache-aware compaction timing decision was chosen (no raw content)."""

    DYNAMIC_TAIL_SAFE_TO_REDUCE = "dynamic_tail_safe_to_reduce"
    COLD_HISTORY_SAFE_TO_COMPACT = "cold_history_safe_to_compact"
    CACHE_INVALIDATION_COST_TOO_HIGH = "cache_invalidation_cost_too_high"
    PREFIX_NOT_STABLE = "prefix_not_stable"
    CACHE_NEAR_EXPIRY = "cache_near_expiry"
    LOW_CONTENT_REDUCTION_BENEFIT = "low_content_reduction_benefit"
    FULL_THREAD_REWRITE_RISK = "full_thread_rewrite_risk"
    PROTECTED_OR_SEMANTIC_RISK = "protected_or_semantic_risk"
    INSUFFICIENT_SIGNALS = "insufficient_signals"


@dataclass(frozen=True, slots=True)
class CacheAwareCompactionTimingInput:
    """Signals for cache-aware compaction timing (helper/policy only)."""

    target: CacheAwareCompactionTarget
    prefix_stability_status: str | None = None
    invalidation_reason: PromptCacheInvalidationReason = PromptCacheInvalidationReason.NONE
    cache_hot: bool | None = None
    ttl_seconds_remaining: int | None = None
    near_expiry_threshold_seconds: int = 60
    estimated_content_reduction_chars: int | None = None
    estimated_cache_invalidation_cost_tokens: int | None = None
    protected_or_semantic_risk: bool = False
    dynamic_tail_reduction_available: bool = False

    def __post_init__(self) -> None:
        _validate_optional_non_negative_int(
            self.ttl_seconds_remaining, "ttl_seconds_remaining"
        )
        if self.near_expiry_threshold_seconds < 0:
            raise ValueError("near_expiry_threshold_seconds cannot be negative")
        _validate_optional_non_negative_int(
            self.estimated_content_reduction_chars,
            "estimated_content_reduction_chars",
        )
        _validate_optional_non_negative_int(
            self.estimated_cache_invalidation_cost_tokens,
            "estimated_cache_invalidation_cost_tokens",
        )
        _validate_optional_stripped_non_empty(
            self.prefix_stability_status, "prefix_stability_status"
        )


@dataclass(frozen=True, slots=True)
class CacheAwareCompactionTimingDecision:
    """Deterministic cache-aware compaction timing outcome (no raw content)."""

    decision: CacheAwareCompactionDecision
    reason: CacheAwareCompactionReason
    target: CacheAwareCompactionTarget
    cache_hot: bool | None
    ttl_seconds_remaining: int | None
    estimated_content_reduction_chars: int | None
    estimated_cache_invalidation_cost_tokens: int | None
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        _validate_optional_non_negative_int(
            self.ttl_seconds_remaining, "ttl_seconds_remaining"
        )
        _validate_optional_non_negative_int(
            self.estimated_content_reduction_chars,
            "estimated_content_reduction_chars",
        )
        _validate_optional_non_negative_int(
            self.estimated_cache_invalidation_cost_tokens,
            "estimated_cache_invalidation_cost_tokens",
        )


def _validate_optional_unit_ratio(value: float | None, field_name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0 inclusive")


class TokenOptimizationRecommendationAction(StrEnum):
    """Advisory recommendation action (policy-only; no auto-apply)."""

    KEEP_CURRENT = "keep_current"
    USE_CONSERVATIVE_PROFILE = "use_conservative_profile"
    USE_BALANCED_PROFILE = "use_balanced_profile"
    ESCALATE_TO_FULL_CONTEXT = "escalate_to_full_context"
    ENABLE_STRATEGY = "enable_strategy"
    DISABLE_STRATEGY = "disable_strategy"
    PREFER_DYNAMIC_TAIL_REDUCTION = "prefer_dynamic_tail_reduction"
    PRESERVE_CACHEABLE_PREFIX = "preserve_cacheable_prefix"
    REQUIRE_MANUAL_REVIEW = "require_manual_review"
    INSUFFICIENT_DATA = "insufficient_data"


class TokenOptimizationRecommendationReason(StrEnum):
    """Why an advisory recommendation was chosen (no raw content)."""

    QUALITY_REGRESSION_RISK = "quality_regression_risk"
    PROTECTED_REGION_RISK = "protected_region_risk"
    HIGH_FALLBACK_RATE = "high_fallback_rate"
    LOW_OR_NO_SAVINGS = "low_or_no_savings"
    MEASURED_SAFE_SAVINGS = "measured_safe_savings"
    CACHE_PREFIX_HOT = "cache_prefix_hot"
    CACHE_PREFIX_UNSTABLE = "cache_prefix_unstable"
    DYNAMIC_TAIL_CAN_BE_REDUCED = "dynamic_tail_can_be_reduced"
    REGRESSION_GATE_FAILED = "regression_gate_failed"
    REGRESSION_GATE_PASSED = "regression_gate_passed"
    INSUFFICIENT_SIGNALS = "insufficient_signals"
    POLICY_REQUIRES_REVIEW = "policy_requires_review"


class TokenOptimizationRecommendationConfidence(StrEnum):
    """Confidence level for an advisory recommendation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_ENOUGH_DATA = "not_enough_data"


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisorySignal:
    """Redaction-safe scalar inputs for advisory recommendations."""

    source_type: TokenOptimizationSourceType
    strategy_kind: TokenOptimizationStrategyKind | None = None
    current_profile: TokenOptimizationProfile | None = None
    sample_count: int = 0
    measured_saved_ratio: float | None = None
    validation_pass_rate: float | None = None
    fallback_rate: float | None = None
    quality_regression_detected: bool = False
    protected_region_failure_detected: bool = False
    regression_gate_passed: bool | None = None
    cache_prefix_stability_status: str | None = None
    cache_hot: bool | None = None
    dynamic_tail_reduction_available: bool = False
    policy_review_required: bool = False

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        _validate_optional_unit_ratio(
            self.measured_saved_ratio, "measured_saved_ratio"
        )
        _validate_optional_unit_ratio(
            self.validation_pass_rate, "validation_pass_rate"
        )
        _validate_optional_unit_ratio(self.fallback_rate, "fallback_rate")
        _validate_optional_stripped_non_empty(
            self.cache_prefix_stability_status, "cache_prefix_stability_status"
        )


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryRecommendation:
    """Advisory recommendation outcome (policy-only; no auto-apply)."""

    action: TokenOptimizationRecommendationAction
    reason: TokenOptimizationRecommendationReason
    confidence: TokenOptimizationRecommendationConfidence
    source_type: TokenOptimizationSourceType
    strategy_kind: TokenOptimizationStrategyKind | None = None
    recommended_profile: TokenOptimizationProfile | None = None
    auto_apply_allowed: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.auto_apply_allowed:
            raise ValueError("auto_apply_allowed must remain False")
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        if self.action in (
            TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
            TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
        ) and self.strategy_kind is None:
            raise ValueError(
                "strategy_kind must be present when action is ENABLE_STRATEGY "
                "or DISABLE_STRATEGY"
            )
        if (
            self.action is TokenOptimizationRecommendationAction.USE_CONSERVATIVE_PROFILE
            and self.recommended_profile is not TokenOptimizationProfile.CONSERVATIVE
        ):
            raise ValueError(
                "recommended_profile must be CONSERVATIVE when action is "
                "USE_CONSERVATIVE_PROFILE"
            )
        if (
            self.action is TokenOptimizationRecommendationAction.USE_BALANCED_PROFILE
            and self.recommended_profile is not TokenOptimizationProfile.BALANCED
        ):
            raise ValueError(
                "recommended_profile must be BALANCED when action is "
                "USE_BALANCED_PROFILE"
            )
