"""Deterministic fast-resume boundary policy (build-state only)."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackResumeError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.resume_boundary import (
    DataPackResumeBoundary,
)


def validate_contiguous_ready_prefix(state: DataPackBuildState) -> None:
    """READY shards must be exactly ordinals 1..completed_shards with no gaps."""
    prefix_length = state.completed_shards
    for index, shard in enumerate(state.shards):
        if index < prefix_length:
            if shard.status is not DataPackShardStatus.READY:
                raise VpiDataPackResumeError(
                    "contiguous READY prefix violated: "
                    f"shard {shard.ordinal} expected READY (prefix length {prefix_length})"
                )
            if shard.ordinal != index + 1:
                raise VpiDataPackResumeError(
                    f"contiguous READY prefix violated: unexpected ordinal {shard.ordinal} "
                    f"at prefix index {index}"
                )
        elif shard.status is DataPackShardStatus.READY:
            raise VpiDataPackResumeError(
                "contiguous READY prefix violated: "
                f"shard {shard.ordinal} is READY beyond completed prefix {prefix_length}"
            )


def resolve_resume_boundary(state: DataPackBuildState) -> DataPackResumeBoundary:
    """Resolve resume position from persisted build-state only."""
    validate_contiguous_ready_prefix(state)
    prefix_length = state.completed_shards
    build_complete = prefix_length == state.shard_count
    if build_complete:
        last_ready = state.shard_count if state.shard_count > 0 else None
        return DataPackResumeBoundary(
            last_ready_ordinal=last_ready,
            next_shard_ordinal=None,
            completed_shards=prefix_length,
            build_complete=True,
        )
    last_ready = prefix_length if prefix_length > 0 else None
    next_shard = prefix_length + 1
    return DataPackResumeBoundary(
        last_ready_ordinal=last_ready,
        next_shard_ordinal=next_shard,
        completed_shards=prefix_length,
        build_complete=False,
    )
