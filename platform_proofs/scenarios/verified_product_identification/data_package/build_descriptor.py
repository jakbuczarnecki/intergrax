"""Build immutable VPI package descriptors from trusted local files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.proof_data import (
    DataPackageFileDescriptor,
    ProofDataPackageDescriptor,
    PublicationStatus,
    dump_proof_data_package_descriptor,
    sha256_file,
)

from platform_proofs.scenarios.verified_product_identification.data_package.content_policy import (
    collect_distributable_files,
)
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    VPI_CANONICAL_DATA_PACK_PACKAGE_VERSION,
    VPI_PACKAGE_ID,
    VPI_PACKAGE_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.data_package.publication import (
    assert_publication_permitted_with_qualification,
)
from platform_proofs.scenarios.verified_product_identification.data_package.redistribution_qualification import (
    load_default_redistribution_qualification,
)
from platform_proofs.scenarios.verified_product_identification.data_package.validation_handoff import (
    assert_descriptor_generation_preconditions,
    load_descriptor_validation_gate,
)


def build_descriptor_from_files(
    *,
    files: tuple[tuple[str, str, Path], ...],
    description: str,
    provenance_ref: str,
    redistribution_status: PublicationStatus,
    output_path: Path,
    package_version: str = VPI_PACKAGE_VERSION,
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
        package_version=package_version,
        description=description,
        files=tuple(descriptors),
        total_size_bytes=total_size,
        provenance_ref=provenance_ref,
        redistribution_status=redistribution_status,
    )
    dump_proof_data_package_descriptor(descriptor, output_path)
    return descriptor


def build_data_pack_descriptor(
    *,
    artifact_root: Path,
    validation_report_path: Path,
    output_path: Path,
    description: str,
    provenance_ref: str,
    redistribution_status: PublicationStatus,
    package_version: str = VPI_CANONICAL_DATA_PACK_PACKAGE_VERSION,
) -> ProofDataPackageDescriptor:
    resolved_root = artifact_root.resolve()
    if redistribution_status is PublicationStatus.PUBLICATION_APPROVED:
        qualification = load_default_redistribution_qualification()
        assert_publication_permitted_with_qualification(
            redistribution_status,
            qualification,
        )
    gate = load_descriptor_validation_gate(validation_report_path)
    assert_descriptor_generation_preconditions(gate, resolved_root)
    distributable_files = collect_distributable_files(resolved_root)
    file_inputs = tuple(
        (entry.relative_path, entry.role.value, entry.path) for entry in distributable_files
    )
    return build_descriptor_from_files(
        files=file_inputs,
        description=description,
        provenance_ref=provenance_ref,
        redistribution_status=redistribution_status,
        output_path=output_path,
        package_version=package_version,
    )


def descriptor_identity(descriptor: ProofDataPackageDescriptor) -> tuple[str, str, tuple[tuple[str, str, int], ...]]:
    return (
        descriptor.package_id,
        descriptor.package_version,
        tuple(
            (file_descriptor.relative_path, file_descriptor.sha256, file_descriptor.size_bytes)
            for file_descriptor in descriptor.files
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable VPI Data Pack distribution descriptor",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package-version",
        default=VPI_CANONICAL_DATA_PACK_PACKAGE_VERSION,
    )
    parser.add_argument(
        "--redistribution-status",
        default=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED.value,
    )
    parser.add_argument(
        "--description",
        default="Verified Product Identification canonical portable Data Pack",
    )
    parser.add_argument(
        "--provenance-ref",
        default="evidence/proof-report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        status = PublicationStatus(args.redistribution_status)
    except ValueError:
        print(f"error=invalid redistribution status: {args.redistribution_status}", file=sys.stderr)
        return 2
    try:
        descriptor = build_data_pack_descriptor(
            artifact_root=args.artifact_root,
            validation_report_path=args.validation_report,
            output_path=args.output,
            description=args.description,
            provenance_ref=args.provenance_ref,
            redistribution_status=status,
            package_version=args.package_version,
        )
    except VpiDataPackageDescriptorBuildError as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1
    print(f"package_id={descriptor.package_id}")
    print(f"package_version={descriptor.package_version}")
    print(f"files_total={len(descriptor.files)}")
    print(f"total_size_bytes={descriptor.total_size_bytes}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
