"""Publication and redistribution gate helpers."""

from __future__ import annotations

from intergrax.proof_data.descriptor import PublicationStatus

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.redistribution_qualification import (
    RedistributionAssetType,
    RedistributionQualificationRecord,
    assert_public_redistribution_permitted,
    assert_publication_permitted_with_qualification,
    can_publish,
    load_default_redistribution_qualification,
    resolve_publication_status,
)


def is_publication_approved(status: PublicationStatus) -> bool:
    return status is PublicationStatus.PUBLICATION_APPROVED


def is_public_redistribution_permitted(
    qualification: RedistributionQualificationRecord,
) -> bool:
    return can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification)


def assert_publication_permitted(status: PublicationStatus) -> None:
    if not is_publication_approved(status):
        raise VpiDataPackageDescriptorBuildError(
            f"public publication blocked for redistribution status {status.value}"
        )


def assert_publication_permitted_for_qualification(
    qualification: RedistributionQualificationRecord | None = None,
) -> None:
    record = qualification or load_default_redistribution_qualification()
    assert_public_redistribution_permitted(record)


def effective_publication_status(
    qualification: RedistributionQualificationRecord | None = None,
) -> PublicationStatus:
    record = qualification or load_default_redistribution_qualification()
    return resolve_publication_status(record)


__all__ = [
    "assert_publication_permitted",
    "assert_publication_permitted_for_qualification",
    "assert_publication_permitted_with_qualification",
    "effective_publication_status",
    "is_public_redistribution_permitted",
    "is_publication_approved",
]
