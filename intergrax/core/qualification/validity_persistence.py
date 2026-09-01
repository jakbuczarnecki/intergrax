# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification validity persistence via ProofReceipt (PROVIDER-QUAL-5)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from intergrax.core.qualification.persistence import (
    PROVIDER_QUALIFICATION_APPLICATION_ID,
    _PROVIDER_QUALIFICATION_PERSISTENCE_SECRET_POLICY,
)
from intergrax.core.qualification.validity import (
    ProviderQualificationValidityContext,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationValidityInterpretation,
    QualificationValidityRecord,
    ValidityEvaluationId,
    validate_qualification_run_id,
    validate_validity_evaluation_id,
)
from intergrax.core.qualification.validity_evaluation import (
    establish_current_qualification_validity,
    interpret_latest_qualification_validity,
)
from intergrax.core.security import (
    SecretSafetyValidationError,
    validate_secret_safe_value,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentRecord,
    DocumentStore,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_partition_key,
    proof_receipt_to_document,
)

PROVIDER_QUALIFICATION_VALIDITY_PROOF_KIND = "provider_qualification_validity"
_PROVIDER_QUALIFICATION_VALIDITY_SCHEMA = "intergrax.provider_qualification_validity.v1"
_VALIDITY_DISCOVERY_SORT: tuple[DocumentDataSort, ...] = (
    DocumentDataSort(path="recorded_at", direction="desc"),
    DocumentDataSort(path="run_id", direction="desc"),
    DocumentDataSort(path="$row_key", direction="desc"),
)


class ProviderQualificationValidityPersistenceConflictError(Exception):
    """Raised when the same validity_evaluation_id already stores different content."""


class ProviderQualificationValidityPersistenceIntegrityError(Exception):
    """Raised when stored validity evidence is malformed or inconsistent."""


class ProviderQualificationValidityPersistenceSafetyError(Exception):
    """Raised when validity evidence contains unsafe credential-bearing data."""


def _proof_id_for_validity(
    qualification_run_id: QualificationRunId,
    validity_evaluation_id: ValidityEvaluationId,
) -> str:
    return (
        f"provider_qualification_validity_receipt:"
        f"{qualification_run_id}:{validity_evaluation_id}"
    )


def _run_id_for_validity(
    qualification_run_id: QualificationRunId,
    validity_evaluation_id: ValidityEvaluationId,
) -> str:
    return f"{qualification_run_id}#{validity_evaluation_id}"


def _validity_row_key(
    qualification_run_id: QualificationRunId,
    validity_evaluation_id: ValidityEvaluationId,
) -> str:
    return (
        f"proof/{PROVIDER_QUALIFICATION_VALIDITY_PROOF_KIND}/"
        f"{qualification_run_id}/{validity_evaluation_id}"
    )


def _validity_row_key_prefix(qualification_run_id: QualificationRunId) -> str:
    return f"proof/{PROVIDER_QUALIFICATION_VALIDITY_PROOF_KIND}/{qualification_run_id}/"


def _encode_optional_text(value: str | None) -> str | None:
    return value


def _encode_validity_context(
    context: ProviderQualificationValidityContext | None,
) -> dict[str, str | None] | None:
    if context is None:
        return None
    return {
        "provider_id": context.provider_id,
        "provider_version": context.provider_version,
        "capability_id": context.capability_id,
        "domain": context.domain,
        "intergrax_revision": context.intergrax_revision,
        "qualification_suite_id": context.qualification_suite_id,
        "qualification_suite_version": context.qualification_suite_version,
        "environment_id": context.environment_id,
        "source_revision": context.source_revision,
        "adapter_identity": _encode_optional_text(context.adapter_identity),
    }


def _decode_validity_context(data: object) -> ProviderQualificationValidityContext | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity evaluation_context",
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
        "source_revision",
    )
    values: dict[str, str | None] = {}
    for field_name in required_text_fields:
        value = data.get(field_name)
        if not isinstance(value, str):
            raise ProviderQualificationValidityPersistenceIntegrityError(
                f"invalid provider qualification validity context field {field_name!r}",
            )
        values[field_name] = value
    adapter_identity = data.get("adapter_identity")
    if adapter_identity is not None and not isinstance(adapter_identity, str):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity adapter_identity",
        )
    values["adapter_identity"] = adapter_identity
    return ProviderQualificationValidityContext(**values)  # type: ignore[arg-type]


def encode_qualification_validity_record(
    record: QualificationValidityRecord,
) -> dict[str, Any]:
    """Serialize a validity evaluation into ProofReceipt domain evidence."""
    return {
        "schema_version": _PROVIDER_QUALIFICATION_VALIDITY_SCHEMA,
        "qualification_run_id": str(record.qualification_run_id),
        "validity_evaluation_id": str(record.validity_evaluation_id),
        "validity": record.validity.value,
        "evaluated_at": record.evaluated_at.isoformat(),
        "reason": _encode_optional_text(record.reason),
        "evaluation_context": _encode_validity_context(record.evaluation_context),
    }


def decode_qualification_validity_record(data: object) -> QualificationValidityRecord:
    """Rehydrate a validity evaluation from ProofReceipt domain evidence."""
    if not isinstance(data, dict):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity payload",
        )
    schema_version = data.get("schema_version")
    if schema_version != _PROVIDER_QUALIFICATION_VALIDITY_SCHEMA:
        raise ProviderQualificationValidityPersistenceIntegrityError(
            f"unsupported provider qualification validity schema_version: {schema_version!r}",
        )
    qualification_run_id = validate_qualification_run_id(data.get("qualification_run_id"))
    validity_evaluation_id = validate_validity_evaluation_id(
        data.get("validity_evaluation_id"),
    )
    validity_raw = data.get("validity")
    if not isinstance(validity_raw, str):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity value",
        )
    evaluated_at_raw = data.get("evaluated_at")
    if not isinstance(evaluated_at_raw, str):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity evaluated_at",
        )
    evaluated_at = datetime.fromisoformat(evaluated_at_raw)
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "invalid provider qualification validity reason",
        )
    return QualificationValidityRecord(
        qualification_run_id=qualification_run_id,
        validity_evaluation_id=validity_evaluation_id,
        validity=QualificationEvidenceValidity(validity_raw),
        evaluated_at=evaluated_at,
        reason=reason,
        evaluation_context=_decode_validity_context(data.get("evaluation_context")),
    )


def validity_records_semantically_equal(
    left: QualificationValidityRecord,
    right: QualificationValidityRecord,
) -> bool:
    """Return True when two records represent the same validity evaluation fact."""
    return left == right


def qualification_validity_record_to_proof_receipt(
    record: QualificationValidityRecord,
) -> ProofReceipt:
    """Project a validity evaluation onto the platform ProofReceipt contract."""
    qualification_run_id = str(record.qualification_run_id)
    validity_evaluation_id = str(record.validity_evaluation_id)
    result = (
        ProofReceiptResult.PASS
        if record.validity is QualificationEvidenceValidity.CURRENT
        else ProofReceiptResult.FAIL
    )
    return ProofReceipt(
        proof_id=_proof_id_for_validity(
            record.qualification_run_id,
            record.validity_evaluation_id,
        ),
        proof_kind=PROVIDER_QUALIFICATION_VALIDITY_PROOF_KIND,
        application_id=PROVIDER_QUALIFICATION_APPLICATION_ID,
        result=result,
        recorded_at=record.evaluated_at,
        run_id=_run_id_for_validity(
            record.qualification_run_id,
            record.validity_evaluation_id,
        ),
        domain_evidence=encode_qualification_validity_record(record),
        metadata={
            "qualification_run_id": qualification_run_id,
            "validity_evaluation_id": validity_evaluation_id,
        },
    )


def proof_receipt_to_qualification_validity_record(
    receipt: ProofReceipt,
) -> QualificationValidityRecord:
    """Reconstruct a validity evaluation from a persisted ProofReceipt."""
    if receipt.proof_kind != PROVIDER_QUALIFICATION_VALIDITY_PROOF_KIND:
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "proof receipt is not a provider qualification validity receipt",
        )
    if receipt.application_id != PROVIDER_QUALIFICATION_APPLICATION_ID:
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "proof receipt application_id does not match provider qualification scope",
        )
    record = decode_qualification_validity_record(dict(receipt.domain_evidence))
    metadata_run_id = receipt.metadata.get("qualification_run_id")
    metadata_eval_id = receipt.metadata.get("validity_evaluation_id")
    if metadata_run_id != str(record.qualification_run_id):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "proof receipt metadata qualification_run_id mismatch",
        )
    if metadata_eval_id != str(record.validity_evaluation_id):
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "proof receipt metadata validity_evaluation_id mismatch",
        )
    expected_run_id = _run_id_for_validity(
        record.qualification_run_id,
        record.validity_evaluation_id,
    )
    if receipt.run_id != expected_run_id:
        raise ProviderQualificationValidityPersistenceIntegrityError(
            "proof receipt run_id does not match validity evaluation identity",
        )
    return record


def _assert_safe_validity_payload(payload: object) -> None:
    try:
        validate_secret_safe_value(
            payload,
            policy=_PROVIDER_QUALIFICATION_PERSISTENCE_SECRET_POLICY,
            context_label="provider qualification validity evidence",
        )
    except SecretSafetyValidationError as exc:
        raise ProviderQualificationValidityPersistenceSafetyError(str(exc)) from exc


class DocumentStoreProviderQualificationValidityPersistence:
    """ConditionalDocumentStore-backed append-only qualification validity evidence."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "provider qualification validity persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def append_evaluation(
        self,
        record: QualificationValidityRecord,
    ) -> QualificationValidityRecord:
        """Append one validity evaluation with idempotent semantics per evaluation id."""
        receipt = qualification_validity_record_to_proof_receipt(record)
        _assert_safe_validity_payload(receipt.model_dump(mode="json"))
        document = proof_receipt_to_document(receipt)
        document = DocumentRecord(
            partition_key=document.partition_key,
            row_key=_validity_row_key(
                record.qualification_run_id,
                record.validity_evaluation_id,
            ),
            data=document.data,
            ttl_seconds=document.ttl_seconds,
        )

        if self._document_store.put_if_absent(document):
            return record

        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise RuntimeError("provider qualification validity append failed")
        return self._resolve_existing_record(existing, record)

    def list_evaluations(
        self,
        qualification_run_id: QualificationRunId | str,
    ) -> tuple[QualificationValidityRecord, ...]:
        """Return all persisted validity evaluations for one qualification run."""
        validated_run_id = validate_qualification_run_id(qualification_run_id)
        partition_key = proof_receipt_partition_key(PROVIDER_QUALIFICATION_APPLICATION_ID)
        prefix = _validity_row_key_prefix(validated_run_id)
        records: list[QualificationValidityRecord] = []
        cursor: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=100,
                row_key_prefix=prefix,
                cursor=cursor,
                sort=_VALIDITY_DISCOVERY_SORT,
            )
            for document in page.documents:
                records.append(self._document_to_record(document))
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return tuple(
            sorted(
                records,
                key=lambda item: (item.evaluated_at, str(item.validity_evaluation_id)),
            )
        )

    def get_current_validity(
        self,
        qualification_run_id: QualificationRunId | str,
    ) -> QualificationValidityInterpretation | None:
        """Return the latest validity interpretation when evaluations exist."""
        validated_run_id = validate_qualification_run_id(qualification_run_id)
        records = self.list_evaluations(validated_run_id)
        if not records:
            return None
        return interpret_latest_qualification_validity(validated_run_id, records)

    def establish_current_validity(
        self,
        qualification_run_id: QualificationRunId | str,
    ) -> QualificationValidityInterpretation:
        """Fail-closed current validity resolution from persisted evaluations."""
        validated_run_id = validate_qualification_run_id(qualification_run_id)
        records = self.list_evaluations(validated_run_id)
        return establish_current_qualification_validity(validated_run_id, records)

    def close(self) -> None:
        """Release adapter resources; DocumentStore lifecycle remains caller-owned."""

    def _document_to_record(self, document: object) -> QualificationValidityRecord:
        from intergrax.integrations.contracts.document_store import DocumentRecord

        if not isinstance(document, DocumentRecord):
            raise ProviderQualificationValidityPersistenceIntegrityError(
                "invalid provider qualification validity document record",
            )
        try:
            receipt = proof_receipt_from_document(document)
        except ValueError as exc:
            raise ProviderQualificationValidityPersistenceIntegrityError(
                "invalid persisted proof receipt for provider qualification validity",
            ) from exc
        return proof_receipt_to_qualification_validity_record(receipt)

    def _resolve_existing_record(
        self,
        existing: object,
        incoming: QualificationValidityRecord,
    ) -> QualificationValidityRecord:
        stored = self._document_to_record(existing)
        if not validity_records_semantically_equal(stored, incoming):
            raise ProviderQualificationValidityPersistenceConflictError(
                "conflicting provider qualification validity evaluation for validity_evaluation_id",
            )
        return stored


def wire_provider_qualification_validity_persistence(
    *,
    document_store: DocumentStore | None = None,
) -> DocumentStoreProviderQualificationValidityPersistence:
    """Platform composition boundary: DocumentStore → validity persistence."""
    if document_store is None:
        raise ValueError(
            "wire_provider_qualification_validity_persistence requires document_store",
        )
    return DocumentStoreProviderQualificationValidityPersistence(document_store)
