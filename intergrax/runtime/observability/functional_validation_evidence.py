# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Compatibility re-export for functional validation evidence contract."""

from intergrax.contracts.functional_validation_evidence import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA,
)

__all__ = [
    "DiagnosticExecutionCorrelation",
    "ExpectedActualRelation",
    "FunctionalValidationEvidence",
    "FunctionalValidationKind",
    "FunctionalValidationOutcome",
    "FunctionalValidatorRef",
    "PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA",
]
