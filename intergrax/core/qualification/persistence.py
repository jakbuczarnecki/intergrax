# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification run persistence via ProofReceipt (PROVIDER-QUAL-3C)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.provider import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
)
from intergrax.core.qualification.status import QualificationStatus
from intergrax.core.qualification.validity import (
    QualificationRunId,
    validate_qualification_run_id,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentStore,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_lookup_row_key,
    proof_receipt_partition_key,
    proof_receipt_to_document,
)

PROVIDER_QUALIFICATION_PROOF_KIND = "provider_qualification"
PROVIDER_QUALIFICATION_APPLICATION_ID = "intergrax.provider_qualification"
_PROVIDER_QUALIFICATION_RUN_SCHEMA = "intergrax.provider_qualification_run.v1"
_FORBIDDEN_PERSISTENCE_KEYS = frozenset(
    {
        "password",
        "token",
        "secret",
        "api_key",
        "dsn",
        "credentials",
    },
)


class ProviderQualificationPersistenceConflictError(Exception):
    """Raised when the same qualification_run_id already stores different content."""


class ProviderQualificationPersistenceIntegrityError(Exception):
    """Raised when stored qualification evidence is malformed or inconsistent."""


def _qualification_status_to_receipt_result(status: QualificationStatus) -> ProofReceiptResult:
    if status in (
        QualificationStatus.PRODUCTION_QUALIFIED,
        QualificationStatus.QUALIFIED,
    ):
        return ProofReceiptResult.PASS
    if status == QualificationStatus.REJECTED:
        return ProofReceiptResult.FAIL
    return ProofReceiptResult.FAIL


def _proof_id_for_run(qualification_run_id: QualificationRunId) -> str:
    return f"provider_qualification_receipt:{qualification_run_id}"


def _encode_optional_text(value: str | None) -> str | None:
    return value


def _encode_evidence_item(
    item: QualificationEvidence[ProviderQualificationEvidenceKind],
) -> dict[str, str | None]:
    return {
        "kind": item.kind.value,
        "code": item.code,
        "ref": _encode_optional_text(item.ref),
        "label": _encode_optional_text(item.label),
    }


def _decode_evidence_item(
    data: object,
) -> QualificationEvidence[ProviderQualificationEvidenceKind]:
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification evidence item",
        )
    kind_raw = data.get("kind")
    code = data.get("code")
    if not isinstance(kind_raw, str) or not isinstance(code, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification evidence item fields",
        )
    ref = data.get("ref")
    label = data.get("label")
    if ref is not None and not isinstance(ref, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification evidence ref",
        )
    if label is not None and not isinstance(label, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification evidence label",
        )
    return QualificationEvidence(
        kind=ProviderQualificationEvidenceKind(kind_raw),
        code=code,
        ref=ref,
        label=label,
    )


def _encode_subject(subject: ProviderQualificationSubject) -> dict[str, str | None]:
    return {
        "provider_id": subject.provider_id,
        "provider_version": subject.provider_version,
        "capability_id": subject.capability_id,
        "domain": subject.domain,
        "intergrax_revision": subject.intergrax_revision,
        "qualification_suite_id": subject.qualification_suite_id,
        "qualification_suite_version": subject.qualification_suite_version,
        "environment_id": subject.environment_id,
        "adapter_identity": _encode_optional_text(subject.adapter_identity),
        "package_name": _encode_optional_text(subject.package_name),
        "package_version": _encode_optional_text(subject.package_version),
        "entry_point_group": _encode_optional_text(subject.entry_point_group),
        "entry_point_name": _encode_optional_text(subject.entry_point_name),
        "host_registration_path": _encode_optional_text(subject.host_registration_path),
        "delivery_source": _encode_optional_text(subject.delivery_source),
        "integration_kind": _encode_optional_text(subject.integration_kind),
    }


def _decode_subject(data: object) -> ProviderQualificationSubject:
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification subject",
        )
    required_text_fields = (
        "provider_id",
        "provider_version",
        "capability_id",
        "domain",
        "intergrax_revision",
        "qualification_suite_id",
        "qualification_suite_version",
        "environment_id",
    )
    values: dict[str, str | None] = {}
    for field_name in required_text_fields:
        value = data.get(field_name)
        if not isinstance(value, str):
            raise ProviderQualificationPersistenceIntegrityError(
                f"invalid provider qualification subject field {field_name!r}",
            )
        values[field_name] = value
    optional_fields = (
        "adapter_identity",
        "package_name",
        "package_version",
        "entry_point_group",
        "entry_point_name",
        "host_registration_path",
        "delivery_source",
        "integration_kind",
    )
    for field_name in optional_fields:
        value = data.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ProviderQualificationPersistenceIntegrityError(
                f"invalid provider qualification subject field {field_name!r}",
            )
        values[field_name] = value
    return ProviderQualificationSubject(**values)


def _encode_executor(executor: ProviderQualificationExecutor) -> dict[str, str | None]:
    return {
        "executor_kind": executor.executor_kind,
        "executor_id": executor.executor_id,
        "executor_version": _encode_optional_text(executor.executor_version),
    }


def _decode_executor(data: object) -> ProviderQualificationExecutor:
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification executor",
        )
    executor_kind = data.get("executor_kind")
    executor_id = data.get("executor_id")
    if not isinstance(executor_kind, str) or not isinstance(executor_id, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification executor fields",
        )
    executor_version = data.get("executor_version")
    if executor_version is not None and not isinstance(executor_version, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification executor_version",
        )
    return ProviderQualificationExecutor(
        executor_kind=executor_kind,
        executor_id=executor_id,
        executor_version=executor_version,
    )


def _encode_result_summary(
    summary: ProviderQualificationResultSummary,
) -> dict[str, int | str | None]:
    return {
        "passed": summary.passed,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "label": _encode_optional_text(summary.label),
    }


def _decode_result_summary(data: object) -> ProviderQualificationResultSummary:
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification result_summary",
        )
    counts: dict[str, int] = {}
    for field_name in ("passed", "failed", "skipped"):
        value = data.get(field_name)
        if type(value) is not int:
            raise ProviderQualificationPersistenceIntegrityError(
                f"invalid provider qualification result_summary field {field_name!r}",
            )
        counts[field_name] = value
    label = data.get("label")
    if label is not None and not isinstance(label, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification result_summary label",
        )
    return ProviderQualificationResultSummary(label=label, **counts)


def _encode_environment_metadata(
    metadata: ProviderQualificationEnvironmentMetadata,
) -> dict[str, bool | str | None]:
    return {
        "real_backend": metadata.real_backend,
        "mocks": metadata.mocks,
        "sqlite_substitution": metadata.sqlite_substitution,
        "bounded_environment": _encode_optional_text(metadata.bounded_environment),
    }


def _decode_environment_metadata(
    data: object,
) -> ProviderQualificationEnvironmentMetadata:
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification environment_metadata",
        )
    bool_fields = ("real_backend", "mocks", "sqlite_substitution")
    values: dict[str, bool | str | None] = {}
    for field_name in bool_fields:
        value = data.get(field_name)
        if type(value) is not bool:
            raise ProviderQualificationPersistenceIntegrityError(
                f"invalid provider qualification environment_metadata field {field_name!r}",
            )
        values[field_name] = value
    bounded_environment = data.get("bounded_environment")
    if bounded_environment is not None and not isinstance(bounded_environment, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification bounded_environment",
        )
    values["bounded_environment"] = bounded_environment
    return ProviderQualificationEnvironmentMetadata(**values)  # type: ignore[arg-type]


def encode_provider_qualification_run(run: ProviderQualificationRun) -> dict[str, Any]:
    """Serialize a provider qualification run into ProofReceipt domain evidence."""
    return {
        "schema_version": _PROVIDER_QUALIFICATION_RUN_SCHEMA,
        "qualification_run_id": str(run.qualification_run_id),
        "subject": _encode_subject(run.subject),
        "status": run.status.value,
        "executed_at": run.executed_at.isoformat(),
        "executor": _encode_executor(run.executor),
        "result_summary": _encode_result_summary(run.result_summary),
        "evidence": [_encode_evidence_item(item) for item in run.evidence],
        "reproducibility": _encode_optional_text(run.reproducibility),
        "limitations": list(run.limitations),
        "source_revision": run.source_revision,
        "environment_metadata": _encode_environment_metadata(run.environment_metadata),
    }


def decode_provider_qualification_run(data: object) -> ProviderQualificationRun:
    """Rehydrate a provider qualification run from ProofReceipt domain evidence."""
    if not isinstance(data, dict):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification run payload",
        )
    schema_version = data.get("schema_version")
    if schema_version != _PROVIDER_QUALIFICATION_RUN_SCHEMA:
        raise ProviderQualificationPersistenceIntegrityError(
            f"unsupported provider qualification run schema_version: {schema_version!r}",
        )
    qualification_run_id = validate_qualification_run_id(data.get("qualification_run_id"))
    status_raw = data.get("status")
    if not isinstance(status_raw, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification status",
        )
    executed_at_raw = data.get("executed_at")
    if not isinstance(executed_at_raw, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification executed_at",
        )
    executed_at = datetime.fromisoformat(executed_at_raw)
    evidence_raw = data.get("evidence")
    if not isinstance(evidence_raw, list):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification evidence list",
        )
    limitations_raw = data.get("limitations")
    if not isinstance(limitations_raw, list) or not all(
        isinstance(item, str) for item in limitations_raw
    ):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification limitations",
        )
    source_revision = data.get("source_revision")
    if not isinstance(source_revision, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification source_revision",
        )
    reproducibility = data.get("reproducibility")
    if reproducibility is not None and not isinstance(reproducibility, str):
        raise ProviderQualificationPersistenceIntegrityError(
            "invalid provider qualification reproducibility",
        )
    return ProviderQualificationRun(
        qualification_run_id=qualification_run_id,
        subject=_decode_subject(data.get("subject")),
        status=QualificationStatus(status_raw),
        executed_at=executed_at,
        executor=_decode_executor(data.get("executor")),
        result_summary=_decode_result_summary(data.get("result_summary")),
        evidence=tuple(_decode_evidence_item(item) for item in evidence_raw),
        reproducibility=reproducibility,
        limitations=tuple(limitations_raw),
        source_revision=source_revision,
        environment_metadata=_decode_environment_metadata(data.get("environment_metadata")),
    )


def runs_semantically_equal(
    left: ProviderQualificationRun,
    right: ProviderQualificationRun,
) -> bool:
    """Return True when two runs represent the same historical qualification fact."""
    return left == right


def provider_qualification_run_to_proof_receipt(
    run: ProviderQualificationRun,
) -> ProofReceipt:
    """Project a provider qualification run onto the platform ProofReceipt contract."""
    qualification_run_id = str(run.qualification_run_id)
    return ProofReceipt(
        proof_id=_proof_id_for_run(run.qualification_run_id),
        proof_kind=PROVIDER_QUALIFICATION_PROOF_KIND,
        application_id=PROVIDER_QUALIFICATION_APPLICATION_ID,
        result=_qualification_status_to_receipt_result(run.status),
        recorded_at=run.executed_at,
        run_id=qualification_run_id,
        domain_evidence=encode_provider_qualification_run(run),
        metadata={"qualification_run_id": qualification_run_id},
    )


def proof_receipt_to_provider_qualification_run(
    receipt: ProofReceipt,
) -> ProviderQualificationRun:
    """Reconstruct a provider qualification run from a persisted ProofReceipt."""
    if receipt.proof_kind != PROVIDER_QUALIFICATION_PROOF_KIND:
        raise ProviderQualificationPersistenceIntegrityError(
            "proof receipt is not a provider qualification receipt",
        )
    if receipt.application_id != PROVIDER_QUALIFICATION_APPLICATION_ID:
        raise ProviderQualificationPersistenceIntegrityError(
            "proof receipt application_id does not match provider qualification scope",
        )
    run = decode_provider_qualification_run(dict(receipt.domain_evidence))
    metadata_run_id = receipt.metadata.get("qualification_run_id")
    if metadata_run_id != str(run.qualification_run_id):
        raise ProviderQualificationPersistenceIntegrityError(
            "proof receipt metadata qualification_run_id mismatch",
        )
    if receipt.run_id != str(run.qualification_run_id):
        raise ProviderQualificationPersistenceIntegrityError(
            "proof receipt run_id does not match qualification_run_id",
        )
    return run


def _assert_safe_persistence_payload(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in _FORBIDDEN_PERSISTENCE_KEYS:
                raise ProviderQualificationPersistenceIntegrityError(
                    f"unsafe persistence key {key!r}",
                )
            _assert_safe_persistence_payload(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_safe_persistence_payload(item)


class DocumentStoreProviderQualificationPersistence:
    """ConditionalDocumentStore-backed durable provider qualification evidence."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "provider qualification persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def persist(self, run: ProviderQualificationRun) -> ProviderQualificationRun:
        """Persist an immutable provider qualification run with idempotent semantics."""
        receipt = provider_qualification_run_to_proof_receipt(run)
        _assert_safe_persistence_payload(receipt.model_dump(mode="json"))
        document = proof_receipt_to_document(receipt)

        if self._document_store.put_if_absent(document):
            return run

        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise RuntimeError("provider qualification persistence append failed")
        return self._resolve_existing_record(existing, run)

    def get_by_qualification_run_id(
        self,
        qualification_run_id: QualificationRunId | str,
    ) -> ProviderQualificationRun | None:
        """Load a persisted provider qualification run by authoritative run identity."""
        validated_run_id = validate_qualification_run_id(qualification_run_id)
        partition_key = proof_receipt_partition_key(PROVIDER_QUALIFICATION_APPLICATION_ID)
        row_key = proof_receipt_lookup_row_key(
            PROVIDER_QUALIFICATION_PROOF_KIND,
            str(validated_run_id),
        )
        document = self._document_store.get(partition_key, row_key)
        if document is None:
            return None
        return self._document_to_run(document)

    def close(self) -> None:
        """Release adapter resources; DocumentStore lifecycle remains caller-owned."""

    def _document_to_run(self, document: object) -> ProviderQualificationRun:
        from intergrax.integrations.contracts.document_store import DocumentRecord

        if not isinstance(document, DocumentRecord):
            raise ProviderQualificationPersistenceIntegrityError(
                "invalid provider qualification document record",
            )
        try:
            receipt = proof_receipt_from_document(document)
        except ValueError as exc:
            raise ProviderQualificationPersistenceIntegrityError(
                "invalid persisted proof receipt for provider qualification",
            ) from exc
        return proof_receipt_to_provider_qualification_run(receipt)

    def _resolve_existing_record(
        self,
        existing: object,
        incoming: ProviderQualificationRun,
    ) -> ProviderQualificationRun:
        stored = self._document_to_run(existing)
        if not runs_semantically_equal(stored, incoming):
            raise ProviderQualificationPersistenceConflictError(
                "conflicting provider qualification run for qualification_run_id",
            )
        return stored


def wire_provider_qualification_persistence(
    *,
    document_store: DocumentStore | None = None,
) -> DocumentStoreProviderQualificationPersistence:
    """Platform composition boundary: DocumentStore → provider qualification persistence."""
    if document_store is None:
        raise ValueError("wire_provider_qualification_persistence requires document_store")
    return DocumentStoreProviderQualificationPersistence(document_store)
