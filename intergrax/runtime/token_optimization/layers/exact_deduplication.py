# © Artur Czarnecki. All rights reserved.

"""Built-in exact line-based deduplication optimization layer (TOKEN-OPT-3C-B)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.protected_regions import validate_protected_regions

_LAYER_ID = "builtin.exact_deduplication"
_WHITESPACE_RE = re.compile(r"\s+")

_BUILTIN_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id=_LAYER_ID,
    mechanism=TokenOptimizationMechanism.DEDUPLICATION,
    kind=TokenOptimizationStrategyKind.DEDUPLICATION,
    safety_class=StrategySafetyClass.LOSSLESS,
    version="1",
)

_SUPPORTED_SOURCE_TYPES = (
    TokenOptimizationSourceType.PROMPT,
    TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
    TokenOptimizationSourceType.CONVERSATION_HISTORY,
    TokenOptimizationSourceType.TOOL_OUTPUT,
)


@dataclass(frozen=True, slots=True)
class ExactDeduplicationLayerConfig:
    """Pipeline-level defaults for exact line deduplication."""

    case_sensitive: bool = True
    normalize_whitespace: bool = True
    preserve_first_occurrence: bool = True
    min_duplicate_length: int = 1

    def __post_init__(self) -> None:
        if self.min_duplicate_length < 1:
            raise ValueError("min_duplicate_length must be >= 1")
        if not self.preserve_first_occurrence:
            raise ValueError(
                "preserve_first_occurrence must remain True for this implementation"
            )


@dataclass(frozen=True, slots=True)
class _DuplicateGroup:
    representative_line_index: int
    duplicate_line_indices: tuple[int, ...]
    dedupe_key_hash: str


class ExactDeduplicationLayer:
    """Deterministic exact line deduplication as a standalone optimization layer."""

    def __init__(
        self,
        *,
        config: ExactDeduplicationLayerConfig | None = None,
    ) -> None:
        self._config = config or ExactDeduplicationLayerConfig()

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return TokenOptimizationLayerDescriptor(
            layer_id=_LAYER_ID,
            name="Exact Deduplication",
            version="1",
            strategy=_BUILTIN_STRATEGY,
            supported_source_types=_SUPPORTED_SOURCE_TYPES,
            safety_class=StrategySafetyClass.LOSSLESS,
            built_in=True,
            requires_validation=True,
        )

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        base_config = self._config
        effective_config = base_config
        config_overrides: dict[str, Any] = {}

        if request.source_type not in _SUPPORTED_SOURCE_TYPES:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE,
            )

        if not _policy_allows_optimization(request.policy):
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=_policy_bypass_reason(request.policy),
            )

        current_content = request.current_content
        if not current_content:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.NOT_APPLICABLE,
            )

        lines = current_content.splitlines()
        original_line_count = len(lines)
        kept_lines, duplicate_groups, duplicates_removed = _dedupe_lines(
            lines,
            effective_config,
        )

        if duplicates_removed == 0:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.NO_SAVINGS,
                original_line_count=original_line_count,
            )

        output_content = "\n".join(kept_lines)
        dedupe_saved_chars = len(current_content) - len(output_content)

        validation = validate_protected_regions(
            request.original_content,
            output_content,
        )
        if validation.status is ProtectedRegionValidationStatus.FAILED:
            metadata = _build_metadata(
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                duplicates_removed=duplicates_removed,
                duplicate_groups=duplicate_groups,
                dedupe_saved_chars=0,
                kept_line_count=original_line_count,
                original_line_count=original_line_count,
                fallback_reason="protected_region_validation_failed",
            )
            return TokenOptimizationLayerResult(
                layer_id=_LAYER_ID,
                output_content=current_content,
                decision=TokenOptimizationLayerDecision.FALLBACK,
                validation=validation,
                receipt_metadata=metadata,
                fallback_used=True,
                strategy=_BUILTIN_STRATEGY,
                metadata=metadata,
            )

        metadata = _build_metadata(
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            duplicates_removed=duplicates_removed,
            duplicate_groups=duplicate_groups,
            dedupe_saved_chars=dedupe_saved_chars,
            kept_line_count=len(kept_lines),
            original_line_count=original_line_count,
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=output_content,
            decision=TokenOptimizationLayerDecision.APPLY,
            validation=validation,
            receipt_metadata=metadata,
            strategy=_BUILTIN_STRATEGY,
            metadata=metadata,
        )

    def _bypass_result(
        self,
        *,
        request: TokenOptimizationLayerRequest,
        base_config: ExactDeduplicationLayerConfig,
        effective_config: ExactDeduplicationLayerConfig,
        config_overrides: dict[str, Any],
        bypass_reason: TokenOptimizationBypassReason,
        original_line_count: int = 0,
    ) -> TokenOptimizationLayerResult:
        metadata = _build_metadata(
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            duplicates_removed=0,
            duplicate_groups=(),
            dedupe_saved_chars=0,
            kept_line_count=original_line_count,
            original_line_count=original_line_count,
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
            bypass_reason=bypass_reason,
            strategy=_BUILTIN_STRATEGY,
            receipt_metadata=metadata,
            metadata=metadata,
        )


def _policy_allows_optimization(policy: TokenOptimizationPolicy) -> bool:
    if not policy.enabled:
        return False
    return policy.profile not in (
        TokenOptimizationProfile.OFF,
        TokenOptimizationProfile.MEASURE_ONLY,
    )


def _policy_bypass_reason(
    policy: TokenOptimizationPolicy,
) -> TokenOptimizationBypassReason:
    if not policy.enabled:
        return TokenOptimizationBypassReason.DISABLED
    return TokenOptimizationBypassReason.POLICY_DISALLOWED


def _dedupe_key(line: str, config: ExactDeduplicationLayerConfig) -> str:
    key = line.strip()
    if config.normalize_whitespace:
        key = _WHITESPACE_RE.sub(" ", key)
    if not config.case_sensitive:
        key = key.lower()
    return key


def _dedupe_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _is_meaningful_for_dedupe(
    key: str,
    config: ExactDeduplicationLayerConfig,
) -> bool:
    if not key:
        return False
    return len(key) >= config.min_duplicate_length


def _dedupe_lines(
    lines: list[str],
    config: ExactDeduplicationLayerConfig,
) -> tuple[list[str], tuple[_DuplicateGroup, ...], int]:
    seen_keys: set[str] = set()
    key_to_representative: dict[str, int] = {}
    group_duplicates: dict[str, list[int]] = {}
    kept_lines: list[str] = []
    duplicates_removed = 0

    for index, line in enumerate(lines):
        key = _dedupe_key(line, config)
        if not _is_meaningful_for_dedupe(key, config):
            kept_lines.append(line)
            continue

        if key in seen_keys:
            duplicates_removed += 1
            group_duplicates.setdefault(key, []).append(index)
            continue

        seen_keys.add(key)
        key_to_representative[key] = index
        kept_lines.append(line)

    duplicate_groups = tuple(
        _DuplicateGroup(
            representative_line_index=key_to_representative[key],
            duplicate_line_indices=tuple(duplicate_indices),
            dedupe_key_hash=_dedupe_key_hash(key),
        )
        for key, duplicate_indices in sorted(
            group_duplicates.items(),
            key=lambda item: key_to_representative[item[0]],
        )
    )
    return kept_lines, duplicate_groups, duplicates_removed


def _config_mapping(config: ExactDeduplicationLayerConfig) -> dict[str, Any]:
    return dict(asdict(config))


def _build_metadata(
    *,
    base_config: ExactDeduplicationLayerConfig,
    effective_config: ExactDeduplicationLayerConfig,
    config_overrides: dict[str, Any],
    duplicates_removed: int,
    duplicate_groups: tuple[_DuplicateGroup, ...],
    dedupe_saved_chars: int,
    kept_line_count: int,
    original_line_count: int,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "base_config": _config_mapping(base_config),
        "effective_config": _config_mapping(effective_config),
        "config_overrides": dict(config_overrides),
        "dedupe_unit": "line",
        "case_sensitive": effective_config.case_sensitive,
        "normalize_whitespace": effective_config.normalize_whitespace,
        "preserve_first_occurrence": effective_config.preserve_first_occurrence,
        "duplicates_removed": duplicates_removed,
        "duplicate_groups": [
            {
                "representative_line_index": group.representative_line_index,
                "duplicate_line_indices": list(group.duplicate_line_indices),
                "dedupe_key_hash": group.dedupe_key_hash,
            }
            for group in duplicate_groups
        ],
        "dedupe_saved_chars": dedupe_saved_chars,
        "kept_line_count": kept_line_count,
        "original_line_count": original_line_count,
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = fallback_reason
    return metadata
