# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform diagnostic evidence contributor SPI (frozen R1 names).

Contributors emit typed, tenant-scoped evidence into canonical stores. They do
not mint Problems, assign root cause, or override central diagnostic authority.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticEvidenceContext,
    DiagnosticExtensionEvidence,
)


@runtime_checkable
class DiagnosticEvidenceContributor(Protocol):
    """Emit versioned domain evidence into approved canonical evidence ports."""

    @property
    def contributor_id(self) -> str:
        """Stable contributor identity for registry ordering and audit."""

    @property
    def evidence_namespace(self) -> str:
        """Namespaced evidence family (e.g. decision, integration, application)."""

    @property
    def priority(self) -> int:
        """Lower values run earlier; ties broken by namespace then contributor_id."""

    def collect(
        self,
        context: DiagnosticEvidenceContext,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        """Collect typed evidence for one bounded diagnostic execution scope."""


@runtime_checkable
class DecisionEvidenceContributor(DiagnosticEvidenceContributor, Protocol):
    """Decision System evidence plugin — decision facts only, never causality."""
