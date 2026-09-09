"""Publication and redistribution gate helpers."""

from __future__ import annotations

from intergrax.proof_data.descriptor import PublicationStatus

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)


def is_publication_approved(status: PublicationStatus) -> bool:
    return status is PublicationStatus.PUBLICATION_APPROVED


def assert_publication_permitted(status: PublicationStatus) -> None:
    if not is_publication_approved(status):
        raise VpiDataPackageDescriptorBuildError(
            f"public publication blocked for redistribution status {status.value}"
        )
