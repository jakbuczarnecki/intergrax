# © Artur Czarnecki. All rights reserved.

"""Built-in extractive filtering optimization layer for tool/terminal/log output (TOKEN-OPT-4A)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
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

_LAYER_ID = "builtin.extractive_filtering"
_STRATEGY_NAME = "extractive_filtering"

_BUILTIN_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id=_LAYER_ID,
    mechanism=TokenOptimizationMechanism.TERMINAL_LOG_FILTERING,
    kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
    safety_class=StrategySafetyClass.LOSSY,
    version="1",
)

_SUPPORTED_SOURCE_TYPES = (
    TokenOptimizationSourceType.TOOL_OUTPUT,
    TokenOptimizationSourceType.TERMINAL_OUTPUT,
    TokenOptimizationSourceType.LOG_OUTPUT,
)

_TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
_TRACEBACK_MAX_LINES = 50

_IMPORTANT_SUBSTRINGS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "warning",
    "warn",
    "exit code",
    "return code",
)

_IMPORTANT_MARKER_SUBSTRINGS = (
    "FAILED",
    "AssertionError",
    "BUILD FAILED",
    "ERROR:",
    "E       ",
)

_OMISSION_MARKER_TEMPLATE = (
    "[... omitted {count} non-critical lines by intergrax extractive filtering ...]"
)
_REPEATED_MARKER_TEMPLATE = "[... repeated {count}x: {preview} ...]"
_REPEATED_MARKER_HASH_TEMPLATE = "[... repeated {count}x: line_hash={line_hash} ...]"


@dataclass(frozen=True, slots=True)
class ExtractiveFilteringLayerConfig:
    enabled: bool = True
    max_output_chars: int = 4000
    head_lines: int = 40
    tail_lines: int = 80
    min_lines_before_filtering: int = 120
    preserve_error_lines: bool = True
    preserve_warning_lines: bool = True
    preserve_traceback_blocks: bool = True
    collapse_repeated_lines: bool = True
    repeated_line_threshold: int = 3
    case_sensitive_repeated_lines: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be > 0")
        if self.head_lines < 0:
            raise ValueError("head_lines must be >= 0")
        if self.tail_lines < 0:
            raise ValueError("tail_lines must be >= 0")
        if self.min_lines_before_filtering < 0:
            raise ValueError("min_lines_before_filtering must be >= 0")
        if self.repeated_line_threshold < 1:
            raise ValueError("repeated_line_threshold must be >= 1")


@dataclass(frozen=True, slots=True)
class _RepeatedLineGroup:
    representative_line_index: int
    repeated_line_indices: tuple[int, ...]
    line_hash: str
    repeat_count: int


class ExtractiveFilteringLayer:
    """Deterministic extractive filtering for noisy tool, terminal, and log output."""

    def __init__(
        self,
        *,
        config: ExtractiveFilteringLayerConfig | None = None,
    ) -> None:
        self._config = config or ExtractiveFilteringLayerConfig()

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return TokenOptimizationLayerDescriptor(
            layer_id=_LAYER_ID,
            name="Extractive Filtering",
            version="1",
            strategy=_BUILTIN_STRATEGY,
            supported_source_types=_SUPPORTED_SOURCE_TYPES,
            safety_class=StrategySafetyClass.LOSSY,
            built_in=True,
            requires_validation=True,
            metadata={
                "strategy_id": _LAYER_ID,
                "supported_tool_output_compaction": True,
            },
        )

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        base_config = self._config
        effective_config = base_config
        config_overrides: dict[str, Any] = {}
        protected_regions = _resolve_protected_regions(request)

        if not effective_config.enabled:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=TokenOptimizationBypassReason.DISABLED,
            )

        if request.source_type not in _SUPPORTED_SOURCE_TYPES:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE,
            )

        if not _policy_allows_optimization(request.policy):
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=_policy_bypass_reason(request.policy),
            )

        current_content = request.current_content
        if not current_content:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=TokenOptimizationBypassReason.NOT_APPLICABLE,
            )

        raw_lines = current_content.splitlines(keepends=True)
        input_line_count = len(raw_lines)
        if (
            input_line_count < effective_config.min_lines_before_filtering
            and len(current_content) <= effective_config.max_output_chars
        ):
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=TokenOptimizationBypassReason.NO_SAVINGS,
                input_line_count=input_line_count,
            )

        keep_indices, important_line_count, traceback_block_count = _select_keep_indices(
            raw_lines,
            effective_config,
        )
        output_content, omitted_line_count, repeated_line_groups = _build_filtered_output(
            raw_lines,
            keep_indices,
            effective_config,
        )

        if output_content == current_content:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                protected_regions=protected_regions,
                bypass_reason=TokenOptimizationBypassReason.NO_SAVINGS,
                input_line_count=input_line_count,
            )

        if not _protected_regions_preserved(output_content, protected_regions):
            metadata = _build_metadata(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                original_content=current_content,
                output_content=current_content,
                input_line_count=input_line_count,
                output_line_count=input_line_count,
                omitted_line_count=0,
                repeated_line_groups=(),
                important_line_count=important_line_count,
                traceback_block_count=traceback_block_count,
                protected_regions=protected_regions,
                char_budget_satisfied=True,
            )
            return TokenOptimizationLayerResult(
                layer_id=_LAYER_ID,
                output_content=current_content,
                decision=TokenOptimizationLayerDecision.FALLBACK,
                receipt_metadata=metadata,
                fallback_used=True,
                bypass_reason=TokenOptimizationBypassReason.PROTECTED_REGION_RISK,
                strategy=_BUILTIN_STRATEGY,
                metadata=metadata,
            )

        output_line_count = output_content.count("\n")
        if output_content and not output_content.endswith("\n"):
            output_line_count += 1

        char_budget_satisfied = len(output_content) <= effective_config.max_output_chars
        metadata = _build_metadata(
            request=request,
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            original_content=current_content,
            output_content=output_content,
            input_line_count=input_line_count,
            output_line_count=output_line_count,
            omitted_line_count=omitted_line_count,
            repeated_line_groups=repeated_line_groups,
            important_line_count=important_line_count,
            traceback_block_count=traceback_block_count,
            protected_regions=protected_regions,
            char_budget_satisfied=char_budget_satisfied,
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=output_content,
            decision=TokenOptimizationLayerDecision.APPLY,
            receipt_metadata=metadata,
            strategy=_BUILTIN_STRATEGY,
            metadata=metadata,
        )

    def _bypass_result(
        self,
        *,
        request: TokenOptimizationLayerRequest,
        base_config: ExtractiveFilteringLayerConfig,
        effective_config: ExtractiveFilteringLayerConfig,
        config_overrides: dict[str, Any],
        protected_regions: tuple[ProtectedRegion, ...],
        bypass_reason: TokenOptimizationBypassReason,
        input_line_count: int = 0,
    ) -> TokenOptimizationLayerResult:
        current_content = request.current_content
        output_line_count = 0
        if current_content:
            output_line_count = current_content.count("\n")
            if not current_content.endswith("\n"):
                output_line_count += 1
        metadata = _build_metadata(
            request=request,
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            original_content=current_content,
            output_content=current_content,
            input_line_count=input_line_count or output_line_count,
            output_line_count=output_line_count,
            omitted_line_count=0,
            repeated_line_groups=(),
            important_line_count=0,
            traceback_block_count=0,
            protected_regions=protected_regions,
            char_budget_satisfied=True,
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
            bypass_reason=bypass_reason,
            strategy=_BUILTIN_STRATEGY,
            receipt_metadata=metadata,
            metadata=metadata,
        )


def _resolve_protected_regions(
    request: TokenOptimizationLayerRequest,
) -> tuple[ProtectedRegion, ...]:
    regions = getattr(request, "protected_regions", None)
    if regions:
        return tuple(regions)
    meta_regions = request.metadata.get("protected_regions")
    if not meta_regions:
        return ()
    if isinstance(meta_regions, ProtectedRegion):
        return (meta_regions,)
    return tuple(meta_regions)


def _protected_regions_preserved(
    output_content: str,
    protected_regions: tuple[ProtectedRegion, ...],
) -> bool:
    for region in protected_regions:
        if region.value and region.value not in output_content:
            return False
    return True


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


def _line_body(raw_line: str) -> str:
    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith("\n") or raw_line.endswith("\r"):
        return raw_line[:-1]
    return raw_line


def _is_important_line(body: str, config: ExtractiveFilteringLayerConfig) -> bool:
    lowered = body.lower()
    if config.preserve_error_lines and any(
        marker in body for marker in _IMPORTANT_MARKER_SUBSTRINGS
    ):
        return True
    if config.preserve_error_lines and any(
        token in lowered
        for token in ("error", "failed", "failure", "exception", "traceback")
    ):
        return True
    if config.preserve_warning_lines and any(
        token in lowered for token in ("warning", "warn")
    ):
        return True
    if any(token in lowered for token in ("exit code", "return code")):
        return True
    return False


def _traceback_blocks(raw_lines: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    index = 0
    line_count = len(raw_lines)
    while index < line_count:
        body = _line_body(raw_lines[index])
        if _TRACEBACK_START_RE.search(body):
            start = index
            index += 1
            while index < line_count and index < start + _TRACEBACK_MAX_LINES:
                if not _line_body(raw_lines[index]).strip() and index > start:
                    break
                index += 1
            blocks.append((start, index))
        else:
            index += 1
    return blocks


def _select_keep_indices(
    raw_lines: list[str],
    config: ExtractiveFilteringLayerConfig,
) -> tuple[set[int], int, int]:
    line_count = len(raw_lines)
    keep_indices: set[int] = set()

    for index in range(min(config.head_lines, line_count)):
        keep_indices.add(index)
    tail_start = max(0, line_count - config.tail_lines)
    for index in range(tail_start, line_count):
        keep_indices.add(index)

    important_line_count = 0
    body_start = min(config.head_lines, line_count)
    body_end = max(body_start, line_count - config.tail_lines)
    for index in range(body_start, body_end):
        if _is_important_line(_line_body(raw_lines[index]), config):
            keep_indices.add(index)
            important_line_count += 1

    traceback_block_count = 0
    if config.preserve_traceback_blocks:
        for start, end in _traceback_blocks(raw_lines):
            traceback_block_count += 1
            for index in range(start, end):
                keep_indices.add(index)

    return keep_indices, important_line_count, traceback_block_count


def _comparison_key(
    raw_line: str,
    config: ExtractiveFilteringLayerConfig,
) -> str:
    body = _line_body(raw_line)
    if config.case_sensitive_repeated_lines:
        return body
    return body.lower()


def _line_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _repeated_marker(count: int, preview_line: str) -> str:
    preview = preview_line.strip()
    if not preview or len(preview) > 80:
        return _REPEATED_MARKER_HASH_TEMPLATE.format(
            count=count,
            line_hash=_line_hash(preview_line),
        )
    return _REPEATED_MARKER_TEMPLATE.format(count=count, preview=preview)


def _build_filtered_output(
    raw_lines: list[str],
    keep_indices: set[int],
    config: ExtractiveFilteringLayerConfig,
) -> tuple[str, int, tuple[_RepeatedLineGroup, ...]]:
    output_parts: list[str] = []
    omitted_run = 0
    omitted_line_count = 0
    repeated_line_groups: list[_RepeatedLineGroup] = []

    pending_kept: list[tuple[int, str]] = []

    def flush_omitted_run() -> None:
        nonlocal omitted_run
        if omitted_run > 0:
            output_parts.append(
                _OMISSION_MARKER_TEMPLATE.format(count=omitted_run) + "\n"
            )
            omitted_run = 0

    def flush_pending_kept() -> None:
        nonlocal pending_kept
        if not pending_kept:
            return
        if not config.collapse_repeated_lines:
            for _, raw_line in pending_kept:
                output_parts.append(raw_line)
            pending_kept = []
            return

        run_key: str | None = None
        run_indices: list[int] = []
        run_lines: list[str] = []

        def flush_repeat_run() -> None:
            nonlocal run_key, run_indices, run_lines
            if not run_indices:
                return
            output_parts.append(run_lines[0])
            extra = len(run_indices) - 1
            if extra >= config.repeated_line_threshold:
                repeated_line_groups.append(
                    _RepeatedLineGroup(
                        representative_line_index=run_indices[0],
                        repeated_line_indices=tuple(run_indices[1:]),
                        line_hash=_line_hash(run_key or ""),
                        repeat_count=extra,
                    )
                )
                output_parts.append(
                    _repeated_marker(extra, _line_body(run_lines[0])) + "\n"
                )
            else:
                for raw_line in run_lines[1:]:
                    output_parts.append(raw_line)
            run_key = None
            run_indices = []
            run_lines = []

        for line_index, raw_line in pending_kept:
            key = _comparison_key(raw_line, config)
            if run_key is None:
                run_key = key
                run_indices = [line_index]
                run_lines = [raw_line]
                continue
            if key == run_key:
                run_indices.append(line_index)
                run_lines.append(raw_line)
                continue
            flush_repeat_run()
            run_key = key
            run_indices = [line_index]
            run_lines = [raw_line]
        flush_repeat_run()
        pending_kept = []

    for index, raw_line in enumerate(raw_lines):
        if index in keep_indices:
            if omitted_run > 0:
                flush_omitted_run()
            pending_kept.append((index, raw_line))
            continue
        omitted_run += 1
        omitted_line_count += 1
        flush_pending_kept()

    flush_pending_kept()
    flush_omitted_run()
    return "".join(output_parts), omitted_line_count, tuple(repeated_line_groups)


def _config_mapping(config: ExtractiveFilteringLayerConfig) -> dict[str, Any]:
    payload = dict(asdict(config))
    payload["metadata"] = dict(config.metadata)
    return payload


def _build_metadata(
    *,
    request: TokenOptimizationLayerRequest,
    base_config: ExtractiveFilteringLayerConfig,
    effective_config: ExtractiveFilteringLayerConfig,
    config_overrides: dict[str, Any],
    original_content: str,
    output_content: str,
    input_line_count: int,
    output_line_count: int,
    omitted_line_count: int,
    repeated_line_groups: tuple[_RepeatedLineGroup, ...],
    important_line_count: int,
    traceback_block_count: int,
    protected_regions: tuple[ProtectedRegion, ...],
    char_budget_satisfied: bool,
) -> dict[str, Any]:
    original_chars = len(original_content)
    output_chars = len(output_content)
    return {
        "strategy": _STRATEGY_NAME,
        "budget_unit": "chars",
        "original_chars": original_chars,
        "output_chars": output_chars,
        "saved_chars": max(0, original_chars - output_chars),
        "input_line_count": input_line_count,
        "output_line_count": output_line_count,
        "omitted_line_count": omitted_line_count,
        "repeated_line_groups": [
            {
                "representative_line_index": group.representative_line_index,
                "repeated_line_indices": list(group.repeated_line_indices),
                "line_hash": group.line_hash,
                "repeat_count": group.repeat_count,
            }
            for group in repeated_line_groups
        ],
        "important_line_count": important_line_count,
        "traceback_block_count": traceback_block_count,
        "protected_region_count": len(protected_regions),
        "char_budget_satisfied": char_budget_satisfied,
        "source_type": request.source_type.value,
        "base_config": _config_mapping(base_config),
        "effective_config": _config_mapping(effective_config),
        "config_overrides": dict(config_overrides),
    }
