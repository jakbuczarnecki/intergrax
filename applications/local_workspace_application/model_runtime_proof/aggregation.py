# © Artur Czarnecki. All rights reserved.

"""Fail-closed proof result aggregation helpers."""

from __future__ import annotations

from local_workspace_application.model_runtime_proof.contracts import (
    IndexInvarianceResult,
    ProviderQualificationResult,
    StageStatus,
)


def _stage_passes(status: StageStatus) -> bool:
    return status is StageStatus.PASS


def provider_qualification_passes(result: ProviderQualificationResult) -> bool:
    if not result.resolved_through_canonical_resolver:
        return False
    if not result.ask_run_persisted:
        return False
    stages = result.stages
    for status in (
        stages.health,
        stages.generation,
        stages.structured_plan,
        stages.tool_call,
        stages.tool_execution,
        stages.grounded_ask,
        stages.citation,
    ):
        if not _stage_passes(status):
            return False
    return True


def index_invariance_passes(result: IndexInvarianceResult) -> bool:
    for status in result.model_dump().values():
        if status is not StageStatus.PASS:
            return False
    return True
