"""Typed arena candidate specification."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaCandidateEligibility,
    EmbeddingLicenseClassification,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputPolicyRef,
)


@dataclass(frozen=True, slots=True)
class EmbeddingArenaCandidate:
    """Immutable arena candidate — model behavior via typed configuration only."""

    candidate_id: str
    provider: str
    model: str
    expected_dimension: int
    license_classification: EmbeddingLicenseClassification
    license_identifier: str
    license_reference: str
    license_reason: str
    query_instruction_policy: EmbeddingInputPolicyRef
    document_instruction_policy: EmbeddingInputPolicyRef
    semantic_input_policy_id: str
    max_sequence_length: int | None
    trust_remote_code_required: bool
    normalization_expected: bool
    eligibility_status: EmbeddingArenaCandidateEligibility
    is_baseline: bool
    fixed_provider_batch_size: int | None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            msg = "candidate_id must be non-empty"
            raise ValueError(msg)
        if not self.provider.strip():
            msg = "provider must be non-empty"
            raise ValueError(msg)
        if not self.model.strip():
            msg = "model must be non-empty"
            raise ValueError(msg)
        if self.expected_dimension <= 0:
            msg = "expected_dimension must be > 0"
            raise ValueError(msg)
