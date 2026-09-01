# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed identifiers for functional diagnostic specifications (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

import re
from typing import NewType

from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_SPECIFICATION_VERSION,
)

FunctionalDiagnosticCheckId = NewType("FunctionalDiagnosticCheckId", str)
FunctionalDiagnosticSpecificationId = NewType("FunctionalDiagnosticSpecificationId", str)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_CHECK_PREFIX = "fdcheck_"
_SPEC_PREFIX = "fdspec_"


def validate_functional_diagnostic_check_id(value: object) -> FunctionalDiagnosticCheckId:
    if type(value) is not str:
        raise TypeError(f"FunctionalDiagnosticCheckId must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError("FunctionalDiagnosticCheckId must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError("FunctionalDiagnosticCheckId must not contain leading or trailing whitespace")
    if not value.startswith(_CHECK_PREFIX):
        raise ValueError("FunctionalDiagnosticCheckId must start with 'fdcheck_'")
    suffix = value[len(_CHECK_PREFIX) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError("FunctionalDiagnosticCheckId suffix must match [0-9a-f]{32}")
    return FunctionalDiagnosticCheckId(value)


def validate_functional_diagnostic_specification_id(
    value: object,
) -> FunctionalDiagnosticSpecificationId:
    if type(value) is not str:
        raise TypeError(
            f"FunctionalDiagnosticSpecificationId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "FunctionalDiagnosticSpecificationId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "FunctionalDiagnosticSpecificationId must not contain leading or trailing whitespace",
        )
    if not value.startswith(_SPEC_PREFIX):
        raise ValueError("FunctionalDiagnosticSpecificationId must start with 'fdspec_'")
    suffix = value[len(_SPEC_PREFIX) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError("FunctionalDiagnosticSpecificationId suffix must match [0-9a-f]{32}")
    return FunctionalDiagnosticSpecificationId(value)


def validate_functional_diagnostic_specification_version(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError("FunctionalDiagnosticSpecificationVersion must be int")
    if value < 1:
        raise ValueError("FunctionalDiagnosticSpecificationVersion must be >= 1")
    if value > MAX_FUNCTIONAL_DIAGNOSTIC_SPECIFICATION_VERSION:
        raise ValueError(
            f"FunctionalDiagnosticSpecificationVersion must be <= "
            f"{MAX_FUNCTIONAL_DIAGNOSTIC_SPECIFICATION_VERSION}",
        )
    return value


__all__ = [
    "FunctionalDiagnosticCheckId",
    "FunctionalDiagnosticSpecificationId",
    "validate_functional_diagnostic_check_id",
    "validate_functional_diagnostic_specification_id",
    "validate_functional_diagnostic_specification_version",
]
