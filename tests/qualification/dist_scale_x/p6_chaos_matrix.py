# © Artur Czarnecki. All rights reserved.

"""P6 dimension N — seeded chaos matrix (qual-only)."""

from __future__ import annotations

import pytest

from tests.qualification.dist_scale_x._p5_support import new_run_id, require_p5_docker, require_p5_infra
from tests.qualification.dist_scale_x._p6_support import p6_record, write_p6_manifest
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot

pytestmark = [pytest.mark.qualification, pytest.mark.integration]

CHAOS_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("N-01", "worker_sigkill"),
    ("N-02", "pg_pause"),
    ("N-03", "rabbit_stop"),
    ("N-05", "event_store_deny"),
)


@pytest.mark.parametrize(("seed", "fault_class"), CHAOS_SCENARIOS)
def test_p6_n_chaos_matrix_delegate(seed: str, fault_class: str) -> None:
    require_p5_infra()
    if fault_class in {"pg_pause", "rabbit_stop"}:
        require_p5_docker()
    if fault_class == "worker_sigkill":
        from tests.qualification.dist_scale_x.p4_physical_crash import (
            test_p4_g_crash_boundary_durable_state,
        )
        test_p4_g_crash_boundary_durable_state("G-01")
    elif fault_class == "pg_pause":
        from tests.qualification.dist_scale_x.p5_network_unavailable import (
            test_p5_i_pg_pause_bounded_errors_recover,
        )

        test_p5_i_pg_pause_bounded_errors_recover()
    elif fault_class == "rabbit_stop":
        from tests.qualification.dist_scale_x.p5_network_unavailable import (
            test_p5_i_rabbit_outage_bounded_recovery,
        )

        test_p5_i_rabbit_outage_bounded_recovery()
    elif fault_class == "event_store_deny":
        from tests.qualification.dist_scale_x.p4_physical_faults import (
            test_p4_f_event_injected_append_outage_fail_closed,
        )

        test_p4_f_event_injected_append_outage_fail_closed()
    run_id = new_run_id(f"chaos-{seed}")
    write_p6_manifest(
        run_id=run_id,
        chaos_seed=seed,
        topology_description=f"chaos {seed} {fault_class}",
        records=(
            p6_record(
                record_id=f"{run_id}-n",
                workload_id="WL-7",
                dimension_ids=("N",),
                duration_seconds=1.0,
                topology_description=fault_class,
                metrics=DistScaleXMetricsSnapshot(extra={"seed": seed}),
                fault_description=fault_class,
            ),
        ),
    )
