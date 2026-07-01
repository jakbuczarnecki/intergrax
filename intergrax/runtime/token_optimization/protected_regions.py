# © Artur Czarnecki. All rights reserved.

"""Protected-region detection and validation (Phase TOKEN-1B / TOKEN-1B-R)."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
)

_PREVIEW_MAX_LEN = 40

PROTECTED_TERMS_ENV_VAR = "INTERGRAX_TOKEN_OPTIMIZATION_PROTECTED_TERMS"
MAX_ENV_PROTECTED_TERMS = 200
MAX_PROTECTED_TERM_LENGTH = 128

BUILT_IN_PROTECTED_TERMS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "DATABASE_URL",
    "REDIS_URL",
    "QDRANT_URL",
    "ELASTICSEARCH_URL",
    "JWT_SECRET",
    "CLIENT_SECRET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

_TERM_SEP_RE = re.compile(r"[,;\n]")

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
    (_HASH_RE, ProtectedRegionKind.HASH),
    (_DATE_RE, ProtectedRegionKind.DATE),
    (_VERSION_RE, ProtectedRegionKind.VERSION),
    (_EXACT_ERROR_RE, ProtectedRegionKind.EXACT_ERROR),
    (_COMMAND_RE, ProtectedRegionKind.COMMAND),
    (_RUN_ID_RE, ProtectedRegionKind.IDENTIFIER),
    (_EVIDENCE_REF_RE, ProtectedRegionKind.EVIDENCE_REFERENCE),
)


def parse_protected_terms(value: str) -> tuple[str, ...]:
    """Parse a delimiter-separated protected-terms string with deterministic normalization."""
    seen: set[str] = set()
    result: list[str] = []
    for part in _TERM_SEP_RE.split(value):
        term = part.strip()
        if not term:
            continue
        if len(term) > MAX_PROTECTED_TERM_LENGTH:
            continue
        if term in seen:
            continue
        seen.add(term)
        result.append(term)
        if len(result) >= MAX_ENV_PROTECTED_TERMS:
            break
    return tuple(result)


def get_env_protected_terms(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return protected terms from the env extension variable, if set."""
    source = os.environ if env is None else env
    raw = source.get(PROTECTED_TERMS_ENV_VAR, "")
    if not raw:
        return ()
    return parse_protected_terms(raw)


def resolve_protected_terms(
    *,
    protected_terms: Iterable[str] = (),
    include_env: bool = True,
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Merge built-in, env-extended, and explicit protected terms (env extends built-ins)."""
    combined: list[str] = list(BUILT_IN_PROTECTED_TERMS)
    if include_env:
        combined.extend(get_env_protected_terms(env))
    combined.extend(protected_terms)
    return _deduplicate_terms(combined)


def detect_protected_regions(
    content: str,
    *,
    protected_terms: Iterable[str] = (),
    include_env_protected_terms: bool = True,
) -> tuple[ProtectedRegion, ...]:
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

    resolved_terms = resolve_protected_terms(
        protected_terms=protected_terms,
        include_env=include_env_protected_terms,
    )
    regions.extend(
        _detect_env_var_regions(content, resolved_terms, excluded_spans)
    )

    return _deduplicate_regions(regions)


def validate_protected_regions(
    original_content: str,
    optimized_content: str,
    *,
    regions: tuple[ProtectedRegion, ...] | None = None,
    protected_terms: Iterable[str] = (),
    include_env_protected_terms: bool = True,
) -> ProtectedRegionValidationResult:
    """Verify that all protected region values appear exactly in *optimized_content*."""
    if regions is not None:
        checked_regions = regions
    else:
        checked_regions = detect_protected_regions(
            original_content,
            protected_terms=protected_terms,
            include_env_protected_terms=include_env_protected_terms,
        )
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


def _detect_env_var_regions(
    content: str,
    protected_terms: tuple[str, ...],
    excluded_spans: list[tuple[int, int]],
) -> list[ProtectedRegion]:
    regions: list[ProtectedRegion] = []
    for term in protected_terms:
        start = 0
        while start < len(content):
            idx = content.find(term, start)
            if idx == -1:
                break
            end = idx + len(term)
            if not _span_overlaps_excluded(idx, end, excluded_spans):
                regions.append(
                    ProtectedRegion(
                        kind=ProtectedRegionKind.ENV_VAR,
                        value=term,
                        start=idx,
                        end=end,
                    )
                )
            start = idx + len(term)
    return regions


def _deduplicate_terms(terms: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if not term or term in seen:
            continue
        if len(term) > MAX_PROTECTED_TERM_LENGTH:
            continue
        seen.add(term)
        unique.append(term)
    return tuple(unique)


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
