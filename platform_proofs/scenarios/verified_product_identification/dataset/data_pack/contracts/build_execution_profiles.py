"""Scenario-owned execution profiles for VPI Data Pack production builds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VPI_EMBEDDING_EXECUTION_ENV_PREFIX,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    VPI_CANONICAL_EMBEDDING_MODEL,
)

PRODUCTION_LOCAL_GPU_PROFILE_ID = "production-local-gpu"


class DataPackBuildExecutionProfileId(StrEnum):
    PRODUCTION_LOCAL_GPU = PRODUCTION_LOCAL_GPU_PROFILE_ID


@dataclass(frozen=True, slots=True)
class DataPackBuildExecutionProfile:
    profile_id: DataPackBuildExecutionProfileId
    device: str
    provider_batch_size: int
    model: str


_DATA_PACK_BUILD_EXECUTION_PROFILES: dict[DataPackBuildExecutionProfileId, DataPackBuildExecutionProfile] = {
    DataPackBuildExecutionProfileId.PRODUCTION_LOCAL_GPU: DataPackBuildExecutionProfile(
        profile_id=DataPackBuildExecutionProfileId.PRODUCTION_LOCAL_GPU,
        device="cuda",
        provider_batch_size=1,
        model=VPI_CANONICAL_EMBEDDING_MODEL,
    ),
}


def list_data_pack_build_execution_profile_ids() -> tuple[str, ...]:
    return tuple(profile.value for profile in DataPackBuildExecutionProfileId)


def resolve_data_pack_build_execution_profile(profile_id: str) -> DataPackBuildExecutionProfile:
    normalized = profile_id.strip()
    try:
        resolved_id = DataPackBuildExecutionProfileId(normalized)
    except ValueError as exc:
        allowed = ", ".join(list_data_pack_build_execution_profile_ids())
        raise VpiDataPackBuildError(
            f"unknown data pack execution profile: {profile_id!r}; allowed: {allowed}"
        ) from exc
    return _DATA_PACK_BUILD_EXECUTION_PROFILES[resolved_id]


def apply_data_pack_build_execution_profile(
    profile: DataPackBuildExecutionProfile,
    *,
    prefix: str = VPI_EMBEDDING_EXECUTION_ENV_PREFIX,
) -> None:
    """Apply provider-neutral execution tuning via scenario-owned environment variables."""
    os.environ[f"{prefix}_DEVICE"] = profile.device
    os.environ[f"{prefix}_PROVIDER_BATCH_SIZE"] = str(profile.provider_batch_size)
