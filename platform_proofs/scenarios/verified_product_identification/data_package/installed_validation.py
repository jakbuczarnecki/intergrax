"""Post-install validation for distributed VPI Data Packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.proof_data.checksum import verify_file_integrity
from intergrax.proof_data.descriptor import ProofDataPackageDescriptor

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    verify_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    read_shard_index_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)


@dataclass(frozen=True, slots=True)
class InstalledDataPackValidationResult:
    package_id: str
    package_version: str
    install_root: Path
    manifest_status: str
    relational_shard_count: int
    embedding_shard_count: int
    descriptor_file_count: int


def validate_installed_distributed_data_pack(
    install_root: Path,
    descriptor: ProofDataPackageDescriptor,
) -> InstalledDataPackValidationResult:
    root = install_root.resolve()
    paths = resolve_data_pack_paths(root)

    for file_descriptor in descriptor.files:
        destination = root / file_descriptor.relative_path
        if not destination.is_file():
            raise VpiDataPackageCompatibilityError(
                f"missing installed file: {file_descriptor.relative_path}"
            )
        verify_file_integrity(
            destination,
            expected_sha256=file_descriptor.sha256,
            expected_size_bytes=file_descriptor.size_bytes,
        )

    verify_sha256sums(paths.checksums_file, root)

    manifest = read_manifest_file(paths.manifest_file)
    if manifest.status is not DataPackStatus.READY:
        raise VpiDataPackageCompatibilityError(
            f"installed manifest status must be READY, got {manifest.status.value}"
        )

    shard_index = read_shard_index_file(paths.shards_index_file)
    relational_shard_count = len(shard_index.relational_shards)
    embedding_shard_count = len(shard_index.embedding_shards)
    if relational_shard_count != embedding_shard_count:
        raise VpiDataPackageCompatibilityError("installed shard index is inconsistent")

    return InstalledDataPackValidationResult(
        package_id=descriptor.package_id,
        package_version=descriptor.package_version,
        install_root=root,
        manifest_status=manifest.status.value,
        relational_shard_count=relational_shard_count,
        embedding_shard_count=embedding_shard_count,
        descriptor_file_count=len(descriptor.files),
    )
