"""Arena execution profiles — composition-level resource budgets."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    DEFAULT_BATCH_CANDIDATES,
    DEFAULT_STAGE_A_RECORDS,
    DEFAULT_STAGE_B_RECORDS,
    DEFAULT_STAGE_C_RECORDS,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_budget import (
    EmbeddingArenaExecutionBudget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_environment import (
    ArenaAcceleratorRequirement,
)

STANDARD_ARENA_PROFILE_ID = "standard"
SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID = "safe-local-gpu"

# RTX 4080 Laptop 12 GB — sustained VRAM guardrail (~11.5 GiB).
_SAFE_LOCAL_GPU_MAX_VRAM_BYTES = int(11.5 * 1024**3)

STANDARD_ARENA_EXECUTION_BUDGET = EmbeddingArenaExecutionBudget(
    profile_id=STANDARD_ARENA_PROFILE_ID,
    accelerator_requirement=ArenaAcceleratorRequirement.ANY,
    stage_a_records=DEFAULT_STAGE_A_RECORDS,
    stage_b_records=DEFAULT_STAGE_B_RECORDS,
    stage_c_records=DEFAULT_STAGE_C_RECORDS,
    max_stage_c_finalists=3,
    candidate_timeout_seconds=7200.0,
    default_batch_size=16,
    fallback_batch_size=8,
    batch_sweep_sizes=DEFAULT_BATCH_CANDIDATES,
    isolate_candidates=False,
    screening_mode=False,
    max_vram_bytes=None,
    query_latency_repetitions=5,
    query_latency_query_count=5,
)

SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET = EmbeddingArenaExecutionBudget(
    profile_id=SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID,
    accelerator_requirement=ArenaAcceleratorRequirement.CUDA,
    stage_a_records=20,
    stage_b_records=50,
    stage_c_records=100,
    max_stage_c_finalists=2,
    candidate_timeout_seconds=900.0,
    default_batch_size=16,
    fallback_batch_size=8,
    batch_sweep_sizes=(),
    isolate_candidates=True,
    screening_mode=True,
    max_vram_bytes=_SAFE_LOCAL_GPU_MAX_VRAM_BYTES,
    query_latency_repetitions=3,
    query_latency_query_count=3,
)

_EXECUTION_PROFILES: dict[str, EmbeddingArenaExecutionBudget] = {
    STANDARD_ARENA_PROFILE_ID: STANDARD_ARENA_EXECUTION_BUDGET,
    SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID: SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
}


def resolve_execution_budget(profile_id: str) -> EmbeddingArenaExecutionBudget:
    try:
        return _EXECUTION_PROFILES[profile_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_EXECUTION_PROFILES))
        msg = f"unknown arena profile {profile_id!r}; supported: {supported}"
        raise ValueError(msg) from exc


def list_execution_profile_ids() -> tuple[str, ...]:
    return tuple(sorted(_EXECUTION_PROFILES))
