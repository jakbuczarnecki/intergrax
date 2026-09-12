"""PostgreSQL external_sor persistence when lab infrastructure is available."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.dataset_profile_loader import (
    load_variant_execution_profile,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.postgresql_persistence import (
    PostgreSqlExternalRealityStore,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.capture import (
    PaymentCaptureCommand,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.services.capture_service import (
    ExternalPaymentCaptureService,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.connection import (
    connect,
    load_connection_settings,
    postgres_lab_available,
    required_tables_present,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    load_scenario_package,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materializer import (
    cleanup_state,
    materialize_package,
)

pytestmark = pytest.mark.integration

_DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"


@pytest.mark.skipif(not postgres_lab_available(), reason="ERL-QUAL-004 PostgreSQL lab not available")
def test_capture_upserts_external_sor_rows() -> None:
    settings = load_connection_settings()
    assert settings is not None
    package = load_scenario_package(
        dataset_package_root=_DATASET_ROOT,
        qualification_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id="payment_completed_after_unknown",
    )

    conn = connect(settings)
    state = None
    try:
        assert required_tables_present(conn)
        state = materialize_package(conn, package)

        store = PostgreSqlExternalRealityStore(conn)
        service = ExternalPaymentCaptureService(store)
        profile = load_variant_execution_profile("payment_completed_after_unknown")
        command = PaymentCaptureCommand(
            correlation_id="corr-pg-lab",
            external_business_reference="PAY-20260912-8F31A",
            idempotency_key="idem-pg",
            merchant_order_reference="PO-2026-004872",
            amount=Decimal("12500.00"),
            currency="EUR",
            request_timestamp=datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC),
        )
        service.process_capture(command, profile, payment_intent_id=state.payment_intent_id)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT terminal_outcome, funds_captured
                FROM external_sor.external_reality
                WHERE external_payment_effect_id = %s
                """,
                (state.external_payment_effect_id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "PAYMENT_COMPLETED"
        assert row[1] is True
    finally:
        if state is not None:
            cleanup_state(conn, state)
            conn.commit()
        conn.close()
