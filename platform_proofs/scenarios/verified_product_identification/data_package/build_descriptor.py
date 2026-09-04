"""Build immutable VPI package descriptors from trusted local files."""

from __future__ import annotations

from pathlib import Path

from intergrax.proof_data import (
    DataPackageFileDescriptor,
    ProofDataPackageDescriptor,
    PublicationStatus,
    dump_proof_data_package_descriptor,
    sha256_file,
)

from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    VPI_PACKAGE_ID,
    VPI_PACKAGE_VERSION,
)


def build_descriptor_from_files(
    *,
    files: tuple[tuple[str, str, Path], ...],
    description: str,
    provenance_ref: str,
    redistribution_status: PublicationStatus,
    output_path: Path,
) -> ProofDataPackageDescriptor:
    descriptors: list[DataPackageFileDescriptor] = []
    total_size = 0
    for relative_path, role, file_path in files:
        if not file_path.is_file():
            raise FileNotFoundError(f"missing package file: {file_path}")
        size_bytes = file_path.stat().st_size
        descriptors.append(
            DataPackageFileDescriptor(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=sha256_file(file_path),
                role=role,
            )
        )
        total_size += size_bytes

    descriptor = ProofDataPackageDescriptor(
        package_id=VPI_PACKAGE_ID,
        package_version=VPI_PACKAGE_VERSION,
        description=description,
        files=tuple(descriptors),
        total_size_bytes=total_size,
        provenance_ref=provenance_ref,
        redistribution_status=redistribution_status,
    )
    dump_proof_data_package_descriptor(descriptor, output_path)
    return descriptor
