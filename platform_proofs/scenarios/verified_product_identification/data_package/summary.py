"""Deterministic descriptor summaries for VPI Data Pack distribution."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.proof_data.descriptor import ProofDataPackageDescriptor

from platform_proofs.scenarios.verified_product_identification.data_package.content_policy import (
    DistributableFileRole,
)


@dataclass(frozen=True, slots=True)
class DataPackDescriptorSummary:
    total_files: int
    total_bytes: int
    relational_file_count: int
    embedding_file_count: int
    manifest_file_count: int
    checksum_file_count: int
    evidence_file_count: int


def summarize_descriptor(descriptor: ProofDataPackageDescriptor) -> DataPackDescriptorSummary:
    relational_file_count = 0
    embedding_file_count = 0
    manifest_file_count = 0
    checksum_file_count = 0
    evidence_file_count = 0
    total_bytes = 0

    for file_descriptor in descriptor.files:
        total_bytes += file_descriptor.size_bytes
        role = file_descriptor.role
        if role == DistributableFileRole.RELATIONAL_SHARD.value:
            relational_file_count += 1
        elif role == DistributableFileRole.EMBEDDING_SHARD.value:
            embedding_file_count += 1
        elif role == DistributableFileRole.MANIFEST.value:
            manifest_file_count += 1
        elif role == DistributableFileRole.CHECKSUMS.value:
            checksum_file_count += 1
        elif role == DistributableFileRole.PROOF_REPORT.value:
            evidence_file_count += 1

    return DataPackDescriptorSummary(
        total_files=len(descriptor.files),
        total_bytes=total_bytes,
        relational_file_count=relational_file_count,
        embedding_file_count=embedding_file_count,
        manifest_file_count=manifest_file_count,
        checksum_file_count=checksum_file_count,
        evidence_file_count=evidence_file_count,
    )
