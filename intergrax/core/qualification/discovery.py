# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification run discovery over persisted ProofReceipt evidence (PROVIDER-QUAL-4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from intergrax.core.qualification.persistence import (
    PROVIDER_QUALIFICATION_APPLICATION_ID,
    PROVIDER_QUALIFICATION_PROOF_KIND,
    ProviderQualificationPersistenceIntegrityError,
    proof_receipt_to_provider_qualification_run,
)
from intergrax.core.qualification.provider import ProviderQualificationRun
from intergrax.core.qualification.status import QualificationStatus
from intergrax.integrations.contracts.document_store import (
    DocumentRecord,
    DocumentStore,
    validate_document_query_limit,
)
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_partition_key,
    proof_receipt_row_key_prefix,
)

_PROVIDER_QUALIFICATION_SCAN_PAGE_SIZE = 500
_DISCOVERY_CURSOR_PREFIX = "offset:"


@dataclass(frozen=True, slots=True)
class ProviderQualificationRunFilter:
    """Exact-match discovery filter over persisted provider qualification runs."""

    provider_id: str | None = None
    provider_version: str | None = None
    capability_id: str | None = None
    domain: str | None = None
    qualification_suite_id: str | None = None
    qualification_suite_version: str | None = None
    environment_id: str | None = None
    status: QualificationStatus | None = None

    def has_any_criterion(self) -> bool:
        return any(
            value is not None
            for value in (
                self.provider_id,
                self.provider_version,
                self.capability_id,
                self.domain,
                self.qualification_suite_id,
                self.qualification_suite_version,
                self.environment_id,
                self.status,
            )
        )


@dataclass(frozen=True, slots=True)
class ProviderQualificationRunDiscoveryPage:
    """One page of discovery results with deterministic ordering."""

    runs: tuple[ProviderQualificationRun, ...]
    next_cursor: str | None = None


class ProviderQualificationDiscoveryError(ValueError):
    """Raised when discovery input is invalid."""


def _decode_discovery_cursor(cursor: str) -> int:
    if not cursor.startswith(_DISCOVERY_CURSOR_PREFIX):
        raise ProviderQualificationDiscoveryError("invalid provider qualification discovery cursor")
    suffix = cursor[len(_DISCOVERY_CURSOR_PREFIX) :]
    if not suffix.isdigit():
        raise ProviderQualificationDiscoveryError("invalid provider qualification discovery cursor")
    return int(suffix)


def _encode_discovery_cursor(offset: int) -> str:
    return f"{_DISCOVERY_CURSOR_PREFIX}{offset}"


def run_matches_filter(
    run: ProviderQualificationRun,
    query_filter: ProviderQualificationRunFilter,
) -> bool:
    """Return True when ``run`` matches every supplied exact filter field."""
    subject = run.subject
    if query_filter.provider_id is not None and subject.provider_id != query_filter.provider_id:
        return False
    if (
        query_filter.provider_version is not None
        and subject.provider_version != query_filter.provider_version
    ):
        return False
    if query_filter.capability_id is not None and subject.capability_id != query_filter.capability_id:
        return False
    if query_filter.domain is not None and subject.domain != query_filter.domain:
        return False
    if (
        query_filter.qualification_suite_id is not None
        and subject.qualification_suite_id != query_filter.qualification_suite_id
    ):
        return False
    if (
        query_filter.qualification_suite_version is not None
        and subject.qualification_suite_version != query_filter.qualification_suite_version
    ):
        return False
    if (
        query_filter.environment_id is not None
        and subject.environment_id != query_filter.environment_id
    ):
        return False
    if query_filter.status is not None and run.status != query_filter.status:
        return False
    return True


def sort_provider_qualification_runs(
    runs: tuple[ProviderQualificationRun, ...] | list[ProviderQualificationRun],
) -> tuple[ProviderQualificationRun, ...]:
    """Sort runs by ``executed_at`` descending, then ``qualification_run_id`` descending."""
    return tuple(
        sorted(
            runs,
            key=lambda run: (run.executed_at, str(run.qualification_run_id)),
            reverse=True,
        ),
    )


def _iter_provider_qualification_documents(
    document_store: DocumentStore,
) -> Iterator[DocumentRecord]:
    partition_key = proof_receipt_partition_key(PROVIDER_QUALIFICATION_APPLICATION_ID)
    row_key_prefix = proof_receipt_row_key_prefix(PROVIDER_QUALIFICATION_PROOF_KIND)
    page_size = validate_document_query_limit(_PROVIDER_QUALIFICATION_SCAN_PAGE_SIZE)
    cursor: str | None = None
    while True:
        page = document_store.query(
            partition_key,
            limit=page_size,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )
        yield from page.documents
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


def _document_to_provider_qualification_run(document: DocumentRecord) -> ProviderQualificationRun:
    try:
        receipt = proof_receipt_from_document(document)
    except ValueError as exc:
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid persisted proof receipt for provider qualification discovery",
        ) from exc
    return proof_receipt_to_provider_qualification_run(receipt)


def discover_provider_qualification_runs(
    document_store: DocumentStore,
    query_filter: ProviderQualificationRunFilter,
    *,
    limit: int = 100,
    cursor: str | None = None,
) -> ProviderQualificationRunDiscoveryPage:
    """
    Discover persisted provider qualification runs via bounded DocumentStore scans.

    Scans only the provider-qualification ProofReceipt partition (not the whole platform),
    applies exact-match filters, and returns ``ProviderQualificationRun`` domain records.
    """
    if not query_filter.has_any_criterion():
        raise ProviderQualificationDiscoveryError(
            "provider qualification discovery requires at least one filter criterion",
        )
    validated_limit = validate_document_query_limit(limit)
    offset = _decode_discovery_cursor(cursor) if cursor is not None else 0
    if offset < 0:
        raise ProviderQualificationDiscoveryError("invalid provider qualification discovery cursor")

    matches: list[ProviderQualificationRun] = []
    for document in _iter_provider_qualification_documents(document_store):
        run = _document_to_provider_qualification_run(document)
        if run_matches_filter(run, query_filter):
            matches.append(run)

    ordered = sort_provider_qualification_runs(matches)
    page_runs = ordered[offset : offset + validated_limit]
    next_offset = offset + validated_limit
    next_cursor = (
        _encode_discovery_cursor(next_offset) if next_offset < len(ordered) else None
    )
    return ProviderQualificationRunDiscoveryPage(runs=page_runs, next_cursor=next_cursor)
