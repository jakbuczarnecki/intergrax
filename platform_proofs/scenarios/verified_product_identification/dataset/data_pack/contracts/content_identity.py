"""Deterministic semantic content identity for VPI data packs."""

from __future__ import annotations

import hashlib
import json

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
)


def compute_data_pack_content_identity(
    *,
    source_dataset: SourceDatasetIdentity,
    derivation_version: str,
    semantic_text_version: str,
    embedding_identity: EmbeddingPackIdentity,
    relational_schema_version: str,
    embedding_schema_version: str,
) -> str:
    """Hash immutable semantic inputs; excludes timestamps and binary checksums."""
    revision = embedding_identity.resolved_model_identity()
    payload = {
        "source_dataset_sha256": source_dataset.dataset_sha256,
        "derivation_version": derivation_version,
        "semantic_text_version": semantic_text_version,
        "embedding_provider": embedding_identity.provider,
        "embedding_model": embedding_identity.model,
        "embedding_model_revision": revision,
        "embedding_dimension": embedding_identity.dimension,
        "relational_schema_version": relational_schema_version,
        "embedding_schema_version": embedding_schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_identity_from_manifest(manifest: DataPackManifest) -> str:
    return compute_data_pack_content_identity(
        source_dataset=manifest.source_dataset,
        derivation_version=manifest.derivation_version,
        semantic_text_version=manifest.semantic_text_version,
        embedding_identity=manifest.embedding_identity,
        relational_schema_version=manifest.relational_schema_version,
        embedding_schema_version=manifest.embedding_schema_version,
    )
