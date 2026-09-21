# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Success evidence invariants for capability acquisition (UCA-3)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)


def validate_acquisition_success_evidence(
    evidence: CapabilityAcquisitionEvidence | None,
) -> None:
    """Require typed handoff reference — not HOST_AVAILABLE projection."""
    if evidence is None:
        raise ValueError("SUCCEEDED requires acquisition evidence")
    if not evidence.domain_handoff_reference and not evidence.artifact_reference:
        raise ValueError(
            "SUCCEEDED requires domain_handoff_reference or artifact_reference",
        )


__all__ = ["validate_acquisition_success_evidence"]
