# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 14 — stale durable serving must not grant authority after physical-unit loss.

Physical-loss models (qualification, not subprocess crash recovery):
- **STOPPED / invalidated** — same ``VenvBundleColocatedRuntimeDeploymentAdapter`` instance;
  ``stop()`` invalidates runtime resources and marks the unit STOPPED (unit table entry remains).
- **Unknown unit** — adapter instance with no matching unit (DIM14-06 / DIM13-04 slice); distinct from STOPPED.

Durable SQLite serving state is intentionally preserved across physical loss; reconciliation and
strict session open must fail closed. Grant/reopen cannot resurrect stopped or unknown units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from intergrax.agent_distribution.deployment import (
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionServingReconciliationError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
    open_strict_production_serving_session,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_c_dim13_strict_process_restart_recovery_gate import (
    _activate_durable_traffic_serving,
    _attest_handoff_unit,
    _authority_snapshot,
    _build_strict_http_app,
    _open_strict_session,
    _physical_unit_count,
    _simulate_process_runtime_restart,
    _strict_http_get_ok,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _physical_in_flight,
    _run_request_json,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "5e2d45c2f629d6824dc0855a5b156f9a6fbd7d86"


@pytest.fixture(autouse=True)
def _isolate_production_process_lifecycle_slot() -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    active = ppl._active_production_process_lifecycle
    if active is not None and not active._process_closed:
        active.close_process_resources()
    yield
    active_after = ppl._active_production_process_lifecycle
    if active_after is not None and not active_after._process_closed:
        active_after.close_process_resources()


def test_c_dim14_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _assert_production_slot_released() -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    assert ppl._active_production_process_lifecycle is None


def _attestation_request_for_handoff(session: StrictProductionServingSession):
    handoff = session.serving_runtime_handoff
    assert handoff is not None
    identity = handoff.identity
    stores = session.composition.agent_platform_runtime.stores
    revision = stores.revision_store.get_revision(identity.runtime_revision_id)
    record = stores.materialization_store.get_by_revision(identity.runtime_revision_id)
    assert revision is not None and record is not None
    return (
        revision,
        ServingUnitAttestationRequest(
            application_id=identity.application_id,
            application_environment_id=identity.application_environment_id,
            runtime_revision_id=identity.runtime_revision_id,
            serving_unit_ref=identity.serving_unit_ref,
            materialization_artifact_digest=identity.materialization_artifact_digest,
            artifact_locator=record.artifact_locator,
        ),
        identity.serving_unit_ref,
    )


def _stop_colocated_physical_unit(
    session: StrictProductionServingSession,
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
) -> str:
    revision, _request, unit_ref = _attestation_request_for_handoff(session)
    adapter.stop(revision, serving_unit_ref=unit_ref)
    return unit_ref


def _install_strict_http_wiring(monkeypatch: pytest.MonkeyPatch, d4) -> None:
    from testing_support.host_fixture_wiring import (
        install_diagnostic_cursor_secret,
        install_host_llm_stub,
    )

    from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
        _patch_gca_strict_http_wiring,
    )

    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)


def test_c_dim14_01_valid_physical_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM14-01 — D4 baseline: durable ACTIVE, attestation, handoff, authorized HTTP."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    _install_strict_http_wiring(monkeypatch, d4)
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-01-baseline"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-baseline",
    )
    before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    )
    assert before.traffic_revision_id == revision_id
    assert revision_id in before.active_revision_ids
    assert before.physical_unit_count >= 1

    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    try:
        assert session.readiness.ready_for_traffic is True
        assert_identity_continuity(session, revision_id=revision_id)
        _attest_handoff_unit(session, adapter)
        app, run_url = _build_strict_http_app(d4, session)
        _strict_http_get_ok(
            monkeypatch,
            app,
            run_url,
            environment.application_environment_id,
        )
    finally:
        if not session._closed:
            session.close()


def test_c_dim14_02_physical_unit_stopped_durable_serving_unchanged(
    tmp_path: Path,
) -> None:
    """DIM14-02 — ``stop()`` on same adapter; durable traffic-serving row unchanged; attestation denied."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-02-stopped"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-stopped",
    )
    pointer_before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    )

    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    try:
        unit_ref = _stop_colocated_physical_unit(session, adapter)
        snapshot_after_loss = _authority_snapshot(
            stores, adapter, environment=environment, revision_id=revision_id
        )
        assert snapshot_after_loss.traffic_revision_id == pointer_before.traffic_revision_id
        assert (
            snapshot_after_loss.serving_pointer_revision
            == pointer_before.serving_pointer_revision
        )
        assert revision_id in snapshot_after_loss.active_revision_ids
        assert _physical_unit_count(adapter) == pointer_before.physical_unit_count

        revision, request, _ = _attestation_request_for_handoff(session)
        with pytest.raises(ServingUnitAttestationError, match="not live") as exc_info:
            adapter.attest_serving_unit(revision, request=request)
        assert exc_info.value.reason_code == "unit_not_live"

        with pytest.raises(VenvBundleColocatedDeploymentError, match="stopped"):
            adapter.reserve_unit_execution(unit_ref)
    finally:
        session.close()


def test_c_dim14_03_strict_startup_fails_after_physical_loss(
    tmp_path: Path,
) -> None:
    """DIM14-03 — reopen strict session against stale durable state after physical stop."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-03-fail-open"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-fail-open",
    )
    pointer_before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    ).serving_pointer_revision
    active_before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    ).active_revision_ids

    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    _stop_colocated_physical_unit(session, adapter)
    _simulate_process_runtime_restart(session)

    with patch(
        "intergrax.applications._shared.production_lifecycle.ProductionLifecycleLauncher."
        "deploy_and_activate",
        side_effect=AssertionError("unauthorized activation after physical unit loss"),
    ):
        with pytest.raises(StrictProductionServingLauncherError, match="attestation"):
            open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance,
                deployment_adapter=adapter,
                composition_builder=composition_builder,
                missing_continuation_file_policy="empty",
            )

    _assert_production_slot_released()
    after = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    )
    assert after.traffic_revision_id == revision_id
    assert after.serving_pointer_revision == pointer_before
    assert after.active_revision_ids == active_before
    assert len(after.active_revision_ids) == 1


def test_c_dim14_04_stale_asgi_cannot_admit_new_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM14-04 — HTTP on pre-built strict ASGI after physical stop; no inner execution."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    _install_strict_http_wiring(monkeypatch, d4)
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-04-http"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-http",
    )
    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    app, run_url = _build_strict_http_app(d4, session)
    unit_ref = _stop_colocated_physical_unit(session, adapter)
    inner_mock = AsyncMock()

    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(app) as client:
                response = client.post(
                    run_url,
                    json=_run_request_json(environment.application_environment_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 502
        assert response.json()["detail"].startswith("run_error:")
        inner_mock.assert_not_called()
        assert _physical_in_flight(adapter, unit_ref) == 0
        assert _authority_snapshot(
            stores, adapter, environment=environment, revision_id=revision_id
        ).traffic_revision_id == revision_id
    finally:
        if not session._closed:
            session.close()


def test_c_dim14_05_repeated_restart_attempts_remain_fail_closed(
    tmp_path: Path,
) -> None:
    """DIM14-05 — repeated strict open attempts do not recreate units or advance pointers."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-05-repeat"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-repeat",
    )
    pointer_before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    ).serving_pointer_revision
    units_before = _physical_unit_count(adapter)

    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    _stop_colocated_physical_unit(session, adapter)
    _simulate_process_runtime_restart(session)

    with patch.object(adapter, "prepare", wraps=adapter.prepare) as prepare_spy:
        for _ in range(3):
            with pytest.raises(StrictProductionServingLauncherError, match="attestation"):
                open_strict_production_serving_session(
                    storage=storage,
                    environment=environment,
                    governance=governance,
                    deployment_adapter=adapter,
                    composition_builder=composition_builder,
                    missing_continuation_file_policy="empty",
                )
            _assert_production_slot_released()

    assert prepare_spy.call_count == 0
    after = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    )
    assert after.serving_pointer_revision == pointer_before
    assert _physical_unit_count(adapter) == units_before


def test_c_dim14_06_fresh_adapter_cannot_attest_durable_active(
    tmp_path: Path,
) -> None:
    """DIM14-06 — DIM13-04 slice: unknown unit on fresh adapter; durable ACTIVE unchanged."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim14-06-unknown"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim14-unknown",
    )
    serving_before = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert serving_before is not None
    assert serving_before.traffic_serving_revision_id == revision_id

    revisions = stores.revision_store.list_revisions_for_environment(
        environment.application_id,
        environment.application_environment_id,
    )
    active_count = sum(
        1 for r in revisions if r.revision_state is RuntimeRevisionState.ACTIVE
    )
    assert active_count == 1

    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    assert _physical_unit_count(fresh_adapter) == 0

    with pytest.raises(StrictProductionServingLauncherError, match="attestation"):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=fresh_adapter,
            composition_builder=composition_builder,
            missing_continuation_file_policy="empty",
        )

    with pytest.raises(ProductionServingReconciliationError, match="attestation"):
        open_production_process_lifecycle(
            agent_distribution_db=storage.agent_distribution_db,
            continuation_state_file=tmp_path / "dim14-06-continuation.json",
            deployment_adapter=fresh_adapter,
            governance=governance,
            trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
            missing_file_policy="empty",
            provider_invocation_db=storage.provider_invocation_db,
            composition_builder=composition_builder,
            serving_reconciliation_targets=(
                (environment.application_id, environment.application_environment_id),
            ),
        )

    serving_after = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert serving_after is not None
    assert serving_after.traffic_serving_revision_id == revision_id
    assert serving_after.serving_pointer_revision == serving_before.serving_pointer_revision
