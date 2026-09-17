# © Artur Czarnecki. All rights reserved.

"""Validation boundary for compaction results (CE-02-R1)."""

from __future__ import annotations

from intergrax.contracts.data_classification import DataClassification
from intergrax.context.budget.compaction import ContextCompactionResult
from intergrax.context.contracts import ContextFragment, content_hash_for_text
from intergrax.context.errors import ContextProviderContractViolationError

_SENSITIVITY_RANK: dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


def validate_compaction_result(
    original: ContextFragment,
    result: ContextCompactionResult,
) -> None:
    """Fail closed when compaction breaks governance invariants."""
    compacted = result.fragment
    if compacted.fragment_id != original.fragment_id:
        raise ContextProviderContractViolationError("compaction.fragment_id_changed")
    if compacted.source_id != original.source_id:
        raise ContextProviderContractViolationError("compaction.source_id_changed")
    if compacted.scope_ref != original.scope_ref:
        raise ContextProviderContractViolationError("compaction.scope_ref_changed")
    if compacted.authority_class != original.authority_class:
        raise ContextProviderContractViolationError("compaction.authority_changed")
    if _SENSITIVITY_RANK[compacted.sensitivity] < _SENSITIVITY_RANK[original.sensitivity]:
        raise ContextProviderContractViolationError("compaction.sensitivity_relaxed")
    if compacted.provider_provenance != original.provider_provenance:
        raise ContextProviderContractViolationError("compaction.provider_provenance_changed")

    expected_input_hash = original.content_hash or content_hash_for_text(original.content)
    if result.provenance.input_content_hash != expected_input_hash:
        raise ContextProviderContractViolationError("compaction.input_hash_mismatch")
    output_hash = compacted.content_hash or content_hash_for_text(compacted.content)
    if result.provenance.output_content_hash != output_hash:
        raise ContextProviderContractViolationError("compaction.output_hash_mismatch")
    if original.fragment_id not in result.provenance.source_fragment_ids:
        raise ContextProviderContractViolationError("compaction.provenance_fragment_mismatch")
