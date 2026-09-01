# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded lookup for functional validation evidence (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.observability.functional_validation_evidence import (
    FunctionalValidationEvidence,
)


@dataclass(frozen=True, slots=True)
class FunctionalValidationEvidenceLookup:
    """
    Bounded, analysis-scoped validation evidence index.

    Analysis requests supply only validations within the execution scope.
    """

    validations: tuple[FunctionalValidationEvidence, ...]

    def get(self, validation_id: EventId) -> FunctionalValidationEvidence | None:
        for validation in self.validations:
            if validation.validation_id == validation_id:
                return validation
        return None


__all__ = ["FunctionalValidationEvidenceLookup"]
