# © Artur Czarnecki. All rights reserved.

"""Protected-region detection and validation (Phase TOKEN-1B)."""

from __future__ import annotations

import re
from typing import Iterable

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
)

_PREVIEW_MAX_LEN = 40

_FENCED_CODE_RE = re.compile(
    r"```[^\n]*\n[\s\S]*?```|```[\s\S]*?```",
    re.MULTILINE,
)
_INLINE_CODE_RE = re.compile(r"(?<![`\\])`([^`\n]+)`")
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_UNIX_PATH_RE = re.compile(
    r"(?<![\w./])(?:/[a-zA-Z0-9_][a-zA-Z0-9_./-]*[a-zA-Z0-9_.-])"
)
_WINDOWS_PATH_RE = re.compile(
    r"(?<![\w:])(?:[A-Za-z]:\\(?:[\w.-]+\\)*[\w.-]+)"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![\w./])(?:\./[\w./-]+|\.\./[\w./-]+)"
)
_ENV_VAR_RE = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:_(?:URL|TOKEN|KEY|SECRET|PASSWORD|API_KEY))\b"
    r"|\b[A-Z][A-Z0-9_]{2,}\b"
)
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b")
_EXACT_ERROR_RE = re.compile(
    r"(?:ValueError|RuntimeError|TypeError|KeyError|OSError|FileNotFoundError)"
    r"\([^)]{1,200}\)"
    r"|Error:\s*[^\n]{1,200}"
)
_COMMAND_RE = re.compile(
    r"(?:^|\n)(?:\$ )?"
    r"(?:(?:sudo|env)\s+)?"
    r"(?:git|uv|pytest|docker|npm|cargo|make|curl|wget)\s+[^\n]{1,200}",
    re.MULTILINE,
)
_RUN_ID_RE = re.compile(r"\brun_[0-9a-fA-F]{8,}\b")
_EVIDENCE_REF_RE = re.compile(
    r"\bevidence[_-][0-9a-zA-Z][0-9a-zA-Z_-]{7,}\b",
    re.IGNORECASE,
)

_DETECTOR_SPECS: tuple[tuple[re.Pattern[str], ProtectedRegionKind], ...] = (
    (_URL_RE, ProtectedRegionKind.URL),
    (_UNIX_PATH_RE, ProtectedRegionKind.PATH),
    (_WINDOWS_PATH_RE, ProtectedRegionKind.PATH),
    (_RELATIVE_PATH_RE, ProtectedRegionKind.PATH),
    (_ENV_VAR_RE, ProtectedRegionKind.ENV_VAR),
    (_HASH_RE, ProtectedRegionKind.HASH),
    (_DATE_RE, ProtectedRegionKind.DATE),
    (_VERSION_RE, ProtectedRegionKind.VERSION),
    (_EXACT_ERROR_RE, ProtectedRegionKind.EXACT_ERROR),
    (_COMMAND_RE, ProtectedRegionKind.COMMAND),
    (_RUN_ID_RE, ProtectedRegionKind.IDENTIFIER),
    (_EVIDENCE_REF_RE, ProtectedRegionKind.EVIDENCE_REFERENCE),
)


def detect_protected_regions(content: str) -> tuple[ProtectedRegion, ...]:
    """Detect protected regions in *content* using conservative deterministic patterns."""
    if not content:
        return ()

    regions: list[ProtectedRegion] = []
    excluded_spans: list[tuple[int, int]] = []

    for match in _FENCED_CODE_RE.finditer(content):
        regions.append(
            ProtectedRegion(
                kind=ProtectedRegionKind.CODE_BLOCK,
                value=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
        excluded_spans.append((match.start(), match.end()))

    for match in _INLINE_CODE_RE.finditer(content):
        if _span_overlaps_excluded(match.start(), match.end(), excluded_spans):
            continue
        regions.append(
            ProtectedRegion(
                kind=ProtectedRegionKind.INLINE_CODE,
                value=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
        excluded_spans.append((match.start(), match.end()))

    for pattern, kind in _DETECTOR_SPECS:
        for match in pattern.finditer(content):
            if _span_overlaps_excluded(match.start(), match.end(), excluded_spans):
                continue
            value = match.group(0)
            if kind is ProtectedRegionKind.COMMAND and value.startswith("\n"):
                value = value[1:]
            regions.append(
                ProtectedRegion(
                    kind=kind,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return _deduplicate_regions(regions)


def validate_protected_regions(
    original_content: str,
    optimized_content: str,
    *,
    regions: tuple[ProtectedRegion, ...] | None = None,
) -> ProtectedRegionValidationResult:
    """Verify that all protected region values appear exactly in *optimized_content*."""
    checked_regions = regions if regions is not None else detect_protected_regions(original_content)
    if not checked_regions:
        return ProtectedRegionValidationResult(
            status=ProtectedRegionValidationStatus.NOT_APPLICABLE,
            regions_checked=0,
            regions_preserved=0,
            regions_failed=0,
        )

    preserved = 0
    failures: list[str] = []

    for region in checked_regions:
        if region.value in optimized_content:
            preserved += 1
            continue
        failures.append(_format_failure(region))

    failed = len(failures)
    checked = len(checked_regions)
    if failed == 0:
        status = ProtectedRegionValidationStatus.PASSED
    else:
        status = ProtectedRegionValidationStatus.FAILED

    return ProtectedRegionValidationResult(
        status=status,
        regions_checked=checked,
        regions_preserved=preserved,
        regions_failed=failed,
        failures=tuple(failures),
    )


def _span_overlaps_excluded(start: int, end: int, excluded: Iterable[tuple[int, int]]) -> bool:
    for ex_start, ex_end in excluded:
        if start < ex_end and end > ex_start:
            return True
    return False


def _deduplicate_regions(regions: list[ProtectedRegion]) -> tuple[ProtectedRegion, ...]:
    """Drop exact duplicate values; prefer earliest span when values collide."""
    seen_values: set[str] = set()
    unique: list[ProtectedRegion] = []
    for region in sorted(
        regions,
        key=lambda item: (
            item.start if item.start is not None else -1,
            -(item.end if item.end is not None else -1),
        ),
    ):
        if region.value in seen_values:
            continue
        seen_values.add(region.value)
        unique.append(region)
    return tuple(unique)


def _format_failure(region: ProtectedRegion) -> str:
    preview = region.value
    if len(preview) > _PREVIEW_MAX_LEN:
        preview = f"{preview[: _PREVIEW_MAX_LEN - 3]}..."
    parts = [f"missing {region.kind.value}", f"preview={preview!r}"]
    if region.start is not None:
        parts.append(f"start={region.start}")
    if region.end is not None:
        parts.append(f"end={region.end}")
    return " ".join(parts)
