# © Artur Czarnecki. All rights reserved.

"""Execution qualification parallelism resolution (explicit argument, environment, qualified default)."""

from __future__ import annotations

import os

from testing_support.execution_qualification.contracts import QualificationManifestError

EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV = "INTERGRAX_EXECUTION_QUALIFICATION_MAX_PARALLEL"

# Performance-qualified default (R3); operators may override via explicit argument or ENV.
EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL = 2


def _validate_positive_max_parallel(value: int, *, source: str) -> int:
    if value < 1:
        raise QualificationManifestError(f"{source} must be >= 1, got {value}")
    return value


def _parse_env_max_parallel(raw: str) -> int:
    stripped = raw.strip()
    if not stripped:
        raise QualificationManifestError(
            f"{EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV} must be a positive integer, "
            "got empty value after stripping whitespace"
        )
    try:
        parsed = int(stripped)
    except ValueError:
        raise QualificationManifestError(
            f"{EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV} must be a positive integer, got {raw!r}"
        ) from None
    return _validate_positive_max_parallel(
        parsed,
        source=EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV,
    )


def resolve_execution_qualification_max_parallel(
    *,
    explicit_value: int | None,
) -> int:
    """Resolve max_parallel: explicit argument, then environment, then qualified default."""
    if explicit_value is not None:
        return _validate_positive_max_parallel(explicit_value, source="max_parallel")

    env_raw = os.environ.get(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV)
    if env_raw is None:
        return EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL

    return _parse_env_max_parallel(env_raw)
