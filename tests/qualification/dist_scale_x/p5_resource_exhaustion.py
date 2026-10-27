# © Artur Czarnecki. All rights reserved.

"""P5 dimension L — admission and bounded resource behavior."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyOverloadMode,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)
from intergrax.tools.providers.http.idempotent_post_qual import HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)

from tests.qualification.dist_scale_x._p5_support import (
    new_run_id,
    p5_record,
    require_p5_infra,
    write_p5_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot

pytestmark = [pytest.mark.qualification, pytest.mark.integration]


def test_p5_l_admission_reject_when_saturated() -> None:
    require_p5_infra()
    config = tool_dependency_concurrency_admission_configuration(
        HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        max_concurrent_calls=1,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
    )
    boundary = materialize_tool_dependency_attempt_boundary(config, production_mode=True)
    assert boundary is not None
    binding = config.bindings[0]
    admission = LocalDependencyConcurrencyAdmission(
        policies={binding.identity: binding.policy},
    )
    request = DependencyConcurrencyAdmissionRequest(
        dependency=binding.identity,
        tenant_id=None,
    )

    async def _double_acquire() -> None:
        permit = await admission.acquire(request)
        try:
            with pytest.raises(DependencyConcurrencyExceededError):
                await admission.acquire(request)
        finally:
            await permit.release()

    asyncio.run(_double_acquire())

    run_id = new_run_id("p5-l-adm")
    topo = "L admission REJECT when max_concurrent_calls=1 and slot held"
    write_p5_manifest(
        run_id=run_id,
        topology_description=topo,
        records=(
            p5_record(
                record_id=f"{run_id}-adm",
                workload_id="WL-4",
                dimension_ids=("L",),
                duration_seconds=1.0,
                topology_description=topo,
                metrics=DistScaleXMetricsSnapshot(extra={"max_concurrent_calls": 1.0}),
                fault_description="second acquire while permit held",
            ),
        ),
    )
