# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification run discovery over persisted ProofReceipt evidence (PROVIDER-QUAL-4)."""

from __future__ import annotations

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
    DocumentDataEquality,
    DocumentDataSort,
    DocumentRecord,
    DocumentStore,
    validate_document_query_limit,
)
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_partition_key,
    proof_receipt_row_key_prefix,
)

_QUALIFICATION_DISCOVERY_SORT: tuple[DocumentDataSort, ...] = (
    DocumentDataSort(path="recorded_at", direction="desc"),
    DocumentDataSort(path="run_id", direction="desc"),
    DocumentDataSort(path="$row_key", direction="desc"),
)


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


def _filter_to_data_equalities(
    query_filter: ProviderQualificationRunFilter,
) -> tuple[DocumentDataEquality, ...]:
    equalities: list[DocumentDataEquality] = []
    if query_filter.provider_id is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.provider_id",
                query_filter.provider_id,
            ),
        )
    if query_filter.provider_version is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.provider_version",
                query_filter.provider_version,
            ),
        )
    if query_filter.capability_id is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.capability_id",
                query_filter.capability_id,
            ),
        )
    if query_filter.domain is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.domain",
                query_filter.domain,
            ),
        )
    if query_filter.qualification_suite_id is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.qualification_suite_id",
                query_filter.qualification_suite_id,
            ),
        )
    if query_filter.qualification_suite_version is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.qualification_suite_version",
                query_filter.qualification_suite_version,
            ),
        )
    if query_filter.environment_id is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.subject.environment_id",
                query_filter.environment_id,
            ),
        )
    if query_filter.status is not None:
        equalities.append(
            DocumentDataEquality(
                "domain_evidence.status",
                query_filter.status.value,
            ),
        )
    return tuple(equalities)


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
    Discover persisted provider qualification runs via storage-bounded DocumentStore queries.

    Storage narrows the candidate set using generic data-path equality filters before receipts
    are decoded into ``ProviderQualificationRun`` domain records.
    """
    if not query_filter.has_any_criterion():
        raise ProviderQualificationDiscoveryError(
            "provider qualification discovery requires at least one filter criterion",
        )
    validated_limit = validate_document_query_limit(limit)
    partition_key = proof_receipt_partition_key(PROVIDER_QUALIFICATION_APPLICATION_ID)
    row_key_prefix = proof_receipt_row_key_prefix(PROVIDER_QUALIFICATION_PROOF_KIND)
    data_equalities = _filter_to_data_equalities(query_filter)

    page = document_store.query(
        partition_key,
        limit=validated_limit,
        row_key_prefix=row_key_prefix,
        cursor=cursor,
        data_equalities=data_equalities,
        sort=_QUALIFICATION_DISCOVERY_SORT,
    )

    runs: list[ProviderQualificationRun] = []
    for document in page.documents:
        runs.append(_document_to_provider_qualification_run(document))

    ordered = sort_provider_qualification_runs(runs)
    if ordered != tuple(runs):
        raise ProviderQualificationPersistenceIntegrityError(
            "provider qualification discovery storage sort mismatch",
        )
    return ProviderQualificationRunDiscoveryPage(runs=ordered, next_cursor=page.next_cursor)
