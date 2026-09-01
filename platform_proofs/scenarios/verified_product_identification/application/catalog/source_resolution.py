"""Resolve immutable source truth from derived candidates."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    SourceRecordFetchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ProductCandidate,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceRecord,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    SourceRecordFetchPort,
)


class SourceTruthResolutionError(RuntimeError):
    """Raised when a candidate cannot be resolved to immutable source truth."""


def resolve_source_record(
    candidate: ProductCandidate,
    source_port: SourceRecordFetchPort,
) -> ProductSourceRecord:
    """Follow candidate provenance to immutable source truth and preserve offer identity."""
    fetch_result: SourceRecordFetchResult = source_port.fetch(candidate.source_ref.offer_id)
    if fetch_result.failure is not None:
        raise SourceTruthResolutionError(fetch_result.failure.message)
    record = fetch_result.record
    if record is None:
        raise SourceTruthResolutionError("source record not found for candidate")
    if record.offer_id != candidate.offer_id:
        raise SourceTruthResolutionError("source record identity does not match candidate")
    if record.offer_id != candidate.source_ref.offer_id:
        raise SourceTruthResolutionError("source record identity does not match source reference")
    return record
