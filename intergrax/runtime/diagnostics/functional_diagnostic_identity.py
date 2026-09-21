# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Compatibility re-export — canonical IDs live in ``intergrax.contracts.diagnostics``."""

from __future__ import annotations

from intergrax.contracts.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
    validate_functional_diagnostic_check_id,
    validate_functional_diagnostic_specification_id,
    validate_functional_diagnostic_specification_version,
)

__all__ = [
    "FunctionalDiagnosticCheckId",
    "FunctionalDiagnosticSpecificationId",
    "validate_functional_diagnostic_check_id",
    "validate_functional_diagnostic_specification_id",
    "validate_functional_diagnostic_specification_version",
]
