"""Deterministic bounded qualification sample selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    DataPackDocumentEmbeddingInputPolicyPort,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    load_vpi_embedding_materialization_config,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.semantic_text_sampler import (
    sample_semantic_texts,
)


@dataclass(frozen=True, slots=True)
class BoundedQualificationDocuments:
    dataset_path: str
    record_count: int
    selection: str
    full_semantic_texts: tuple[str, ...]
    bounded_document_texts: tuple[str, ...]


def load_bounded_qualification_documents(
    *,
    dataset_path: Path,
    record_count: int,
    policy: DataPackDocumentEmbeddingInputPolicyPort,
    artifact_dir: Path,
) -> BoundedQualificationDocuments:
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    config = load_vpi_embedding_materialization_config(
        max_records_override=record_count,
        artifact_dir_override=artifact_dir,
    )
    resolved_config = replace(
        config,
        dataset_path=dataset_path,
        max_records=record_count,
    )
    full_semantic_texts = sample_semantic_texts(
        resolved_config,
        record_count=record_count,
    )
    bounded_document_texts = tuple(
        policy.apply_document(text) for text in full_semantic_texts
    )
    return BoundedQualificationDocuments(
        dataset_path=str(dataset_path),
        record_count=len(bounded_document_texts),
        selection="deterministic_first_n_real_offers",
        full_semantic_texts=full_semantic_texts,
        bounded_document_texts=bounded_document_texts,
    )
