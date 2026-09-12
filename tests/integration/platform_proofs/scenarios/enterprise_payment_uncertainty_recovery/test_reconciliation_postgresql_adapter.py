# © Artur Czarnecki. All rights reserved.

"""PostgreSQL external-reality lookup for reconciliation plugin (lab)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.postgresql_external_reality_lookup import (
    PostgreSqlExternalRealityLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.probe_result import (
    map_snapshot_to_probe_result,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    EXTERNAL_EFFECT_SOR_PROBE_REF,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
)
from intergrax.contracts.enterprise_reliability.reconciliation_execution import ReconciliationProbeRequest
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.connection import (
    connect,
    load_connection_settings,
    postgres_lab_available,
    required_tables_present,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    load_scenario_package,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materialization import (
    row_payloads,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materializer import (
    cleanup_state,
    materialize_package,
)

pytestmark = pytest.mark.integration

_DATASET_ROOT = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/dataset"
)


@pytest.mark.parametrize(
    ("variant_id", "expected_verdict"),
    [
        ("payment_completed_after_unknown", ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS),
        ("payment_failed_after_unknown", ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE),
        ("payment_truth_unavailable", ExternalEffectEvidenceVerdict.INSUFFICIENT),
    ],
)
@pytest.mark.skipif(not postgres_lab_available(), reason="ERL-QUAL-004 PostgreSQL lab not available")
def test_postgresql_lookup_maps_variant_truth(
    variant_id: str,
    expected_verdict: ExternalEffectEvidenceVerdict,
) -> None:
    settings = load_connection_settings()
    assert settings is not None
    package = load_scenario_package(
        dataset_package_root=_DATASET_ROOT,
        qualification_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id=variant_id,
    )
    payloads = row_payloads(package)
    correlation_id = str(payloads["external_sor.external_reality"]["correlation_id"])

    conn = connect(settings)
    state = None
    try:
        assert required_tables_present(conn)
        state = materialize_package(conn, package)
        lookup = PostgreSqlExternalRealityLookup(conn)
        snapshot = lookup.lookup_by_correlation_id(correlation_id)
        result = map_snapshot_to_probe_result(
            request=ReconciliationProbeRequest(
                tenant_id="tenant-lab",
                correlation_id=correlation_id,
                contract_id="contract-lab",
                probe_ref=EXTERNAL_EFFECT_SOR_PROBE_REF,
                plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
            ),
            snapshot=snapshot,
        )
        assert result.verdict is expected_verdict
    finally:
        if state is not None:
            cleanup_state(conn, state)
            conn.commit()
        conn.close()
