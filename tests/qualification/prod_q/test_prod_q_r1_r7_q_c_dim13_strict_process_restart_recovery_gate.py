# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 13 — strict production serving recovery across process-runtime restart.

Process boundary (qualification model):
- **Process-owned** (released on ``close_process_resources``): continuation store, provider
  invocation store, production lifecycle slot.
- **Durable** (SQLite agent distribution DB): revision, materialization, serving pointer,
  deployment instance records.
- **Physical colocated units** live in the injected ``VenvBundleColocatedRuntimeDeploymentAdapter``
  (out-of-process memory). Recovery after restart requires the same adapter instance with a still-
  attestable unit, or reconciliation fails closed (colocated unit loss — DIM14 scope).

Event **A** (controlled strict session close / drain) is not used as the restart surrogate here.
Event **B** uses lifecycle resource close without strict-session drain.
Event **C** (fresh adapter / unit table loss) is exercised as a negative DIM13-04 slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
from intergrax.agent_distribution.deployment import ServingUnitAttestationRequest
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionServingReconciliationError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    RECONCILIATION_OUTCOME_READY,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
    open_strict_production_serving_session,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    VenvBundleRuntimeAgentFactoryResolver,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
    with_qualified_strict_external_work_provider_pin,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
    with_strict_http_api_keys,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _RUN_ROUTE_SUFFIX,
    _gr6_production_policy_bundle,
    _patch_gca_strict_http_wiring,
    _run_request_json,
)
pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "82b9c9ec05abece8ae74269413ec44a32ec5c8cc"


@dataclass(frozen=True, slots=True)
class _AuthoritySnapshot:
    traffic_revision_id: str | None
    serving_pointer_revision: int
    active_revision_ids: tuple[str, ...]
    physical_unit_count: int

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


def test_c_dim13_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _physical_unit_count(adapter: VenvBundleColocatedRuntimeDeploymentAdapter) -> int:
    return len(adapter._units)


def _authority_snapshot(
    stores,
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    *,
    environment,
    revision_id: str,
) -> _AuthoritySnapshot:
    serving = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    revisions = stores.revision_store.list_revisions_for_environment(
        environment.application_id,
        environment.application_environment_id,
    )
    active_ids = tuple(
        r.runtime_revision_id
        for r in revisions
        if r.revision_state is RuntimeRevisionState.ACTIVE
    )
    return _AuthoritySnapshot(
        traffic_revision_id=(
            serving.traffic_serving_revision_id if serving is not None else None
        ),
        serving_pointer_revision=(
            serving.serving_pointer_revision if serving is not None else 0
        ),
        active_revision_ids=active_ids,
        physical_unit_count=_physical_unit_count(adapter),
    )


def _activate_durable_traffic_serving(
    d4,
    tmp_path: Path,
    *,
    environment,
    revision_id: str,
    marker: str,
) -> tuple[object, ...]:
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    composition_builder = d4._composition_builder(tmp_path)
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id=revision_id,
            marker=marker,
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle,
            locator,
            principal=governance.principal,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores
    return storage, adapter, governance, composition_builder, stores


def _open_strict_session(
    *,
    storage,
    environment,
    governance,
    adapter,
    composition_builder,
) -> StrictProductionServingSession:
    return open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=adapter,
        composition_builder=composition_builder,
        missing_continuation_file_policy="empty",
    )


def _simulate_process_runtime_restart(session: StrictProductionServingSession) -> None:
    """Event B — release process-owned resources without strict-session drain (not event A)."""
    session.lifecycle.close_process_resources()
    session._closed = True


def _build_strict_http_app(d4, session: StrictProductionServingSession) -> tuple[FastAPI, str]:
    env = session.environment
    tenant_id = env.application_environment_id
    settings = with_qualified_strict_external_work_provider_pin(
        with_strict_http_api_keys(
            replace(
                GovernedContractorBackendSettings.from_env(),
                include_mcp=False,
                include_scheduler=False,
                include_interaction_routes=False,
                include_task_control=False,
                default_agent_id="search",
                runtime_policy_bundle=_gr6_production_policy_bundle(),
            ),
            execution_tenant_id=tenant_id,
            wrong_tenant_id=f"{tenant_id}-unauthorized-caller",
        ),
    )
    host_runtime = d4._host_runtime_for_startup(settings)
    app = create_governed_contractor_strict_production_asgi_app(
        session,
        settings=settings,
        host_runtime=host_runtime,
    )
    run_url = f"{settings.route_prefix}{_RUN_ROUTE_SUFFIX}"
    return app, run_url


def _strict_http_get_ok(
    monkeypatch: pytest.MonkeyPatch,
    app: FastAPI,
    run_url: str,
    tenant_id: str,
) -> None:
    from intergrax.runtime.task.task import TaskResult, TaskState
    from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
        terminal_task_result_exposure_no_decision_gate,
    )
    from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

    completed = TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="c-dim13-restart-http",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )

    async def _stub_execute(*_args, **_kwargs):
        return completed

    with patch(_INNER_DELEGATE, new=AsyncMock(side_effect=_stub_execute)):
        client = TestClient(app)
        response = client.post(
            run_url,
            json=_run_request_json(tenant_id),
            headers=strict_http_auth_headers(),
        )
    assert response.status_code == 200


def _attest_handoff_unit(
    session: StrictProductionServingSession,
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
) -> None:
    handoff = session.serving_runtime_handoff
    identity = handoff.identity
    stores = session.composition.agent_platform_runtime.stores
    revision = stores.revision_store.get_revision(identity.runtime_revision_id)
    record = stores.materialization_store.get_by_revision(identity.runtime_revision_id)
    assert revision is not None and record is not None
    attestation = adapter.attest_serving_unit(
        revision,
        request=ServingUnitAttestationRequest(
            application_id=identity.application_id,
            application_environment_id=identity.application_environment_id,
            runtime_revision_id=identity.runtime_revision_id,
            serving_unit_ref=identity.serving_unit_ref,
            materialization_artifact_digest=identity.materialization_artifact_digest,
            artifact_locator=record.artifact_locator,
        ),
    )
    assert attestation.physical_unit_live is True
    assert attestation.runtime_revision_id == identity.runtime_revision_id


def test_c_dim13_01_pre_restart_serving_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM13-01 — durable ACTIVE serving, attestation, strict HTTP before restart."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    from testing_support.host_fixture_wiring import (
        install_diagnostic_cursor_secret,
        install_host_llm_stub,
    )

    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim13-01-baseline"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim13-baseline",
    )
    before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    )
    assert before.traffic_revision_id == revision_id
    assert revision_id in before.active_revision_ids

    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    try:
        assert session.readiness.ready_for_traffic is True
        handoff = session.serving_runtime_handoff
        assert_identity_continuity(session, revision_id=revision_id)
        _attest_handoff_unit(session, adapter)
        app, run_url = _build_strict_http_app(d4, session)
        _strict_http_get_ok(monkeypatch, app, run_url, environment.application_environment_id)
    finally:
        if not session._closed:
            session.close()


def test_c_dim13_02_03_recovery_attestation_handoff_and_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM13-02/03 — reconcile after process-runtime restart; HTTP only after recovery."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    from testing_support.host_fixture_wiring import (
        install_diagnostic_cursor_secret,
        install_host_llm_stub,
    )

    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim13-02-recovery"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim13-recovery",
    )
    pointer_before = _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    ).serving_pointer_revision
    units_before_restart = _physical_unit_count(adapter)

    session_before = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    handoff_before = session_before.serving_runtime_handoff
    unit_ref_before = handoff_before.identity.serving_unit_ref
    materialization_digest_before = handoff_before.identity.materialization_artifact_digest
    _simulate_process_runtime_restart(session_before)

    with patch(
        "intergrax.applications._shared.production_lifecycle.ProductionLifecycleLauncher."
        "deploy_and_activate",
        side_effect=AssertionError("unauthorized activation during restart recovery"),
    ):
        session_after = _open_strict_session(
            storage=storage,
            environment=environment,
            governance=governance,
            adapter=adapter,
            composition_builder=composition_builder,
        )
    try:
        assert session_after.reconciliation.evidence.outcome == RECONCILIATION_OUTCOME_READY
        stores_after = session_after.composition.agent_platform_runtime.stores
        snapshot_after = _authority_snapshot(
            stores_after,
            adapter,
            environment=environment,
            revision_id=revision_id,
        )
        assert snapshot_after.traffic_revision_id == revision_id
        assert snapshot_after.serving_pointer_revision == pointer_before
        assert snapshot_after.physical_unit_count == units_before_restart
        handoff = session_after.serving_runtime_handoff
        assert handoff.identity.serving_unit_ref == unit_ref_before
        assert handoff.identity.materialization_artifact_digest == materialization_digest_before
        resolver = handoff.runtime_agent_factory_resolver
        assert isinstance(resolver, VenvBundleRuntimeAgentFactoryResolver)
        _attest_handoff_unit(session_after, adapter)
        app, run_url = _build_strict_http_app(d4, session_after)
        _strict_http_get_ok(
            monkeypatch,
            app,
            run_url,
            environment.application_environment_id,
        )
    finally:
        session_after.close()


def test_c_dim13_04_no_traffic_before_completed_reconciliation(
    tmp_path: Path,
) -> None:
    """DIM13-04 — durable ACTIVE/READY in DB cannot admit strict HTTP without physical authority."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim13-04-stale-physical"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim13-no-traffic",
    )
    assert _authority_snapshot(
        stores, adapter, environment=environment, revision_id=revision_id
    ).traffic_revision_id == revision_id

    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    assert _physical_unit_count(fresh_adapter) == 0

    with pytest.raises(StrictProductionServingLauncherError, match="attestation"):
        _open_strict_session(
            storage=storage,
            environment=environment,
            governance=governance,
            adapter=fresh_adapter,
            composition_builder=composition_builder,
        )

    with pytest.raises(ProductionServingReconciliationError, match="attestation"):
        open_production_process_lifecycle(
            agent_distribution_db=storage.agent_distribution_db,
            continuation_state_file=tmp_path / "dim13-04-continuation.json",
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


def test_c_dim13_05_recovery_reconciliation_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM13-05 — repeated reconciliation without duplicate physical preparation or pointer move."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    from testing_support.host_fixture_wiring import (
        install_diagnostic_cursor_secret,
        install_host_llm_stub,
    )

    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim13-05-idempotent"
    storage, adapter, governance, composition_builder, stores = _activate_durable_traffic_serving(
        d4,
        tmp_path,
        environment=environment,
        revision_id=revision_id,
        marker="dim13-idempotent",
    )
    session_first = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    _simulate_process_runtime_restart(session_first)
    session = _open_strict_session(
        storage=storage,
        environment=environment,
        governance=governance,
        adapter=adapter,
        composition_builder=composition_builder,
    )
    try:
        pointer = _authority_snapshot(
            session.composition.agent_platform_runtime.stores,
            adapter,
            environment=environment,
            revision_id=revision_id,
        ).serving_pointer_revision
        units = _physical_unit_count(adapter)
        with patch.object(adapter, "prepare", wraps=adapter.prepare) as prepare_spy:
            first = session.lifecycle.reconcile_serving_environment(
                environment.application_id,
                environment.application_environment_id,
                expected_serving_pointer_revision=pointer,
            )
            second = session.lifecycle.reconcile_serving_environment(
                environment.application_id,
                environment.application_environment_id,
                expected_serving_pointer_revision=pointer,
            )
        assert prepare_spy.call_count == 0
        assert first.evidence.outcome == RECONCILIATION_OUTCOME_READY
        assert second.evidence.outcome == RECONCILIATION_OUTCOME_READY
        assert first.attestation is not None and second.attestation is not None
        assert (
            first.attestation.serving_unit_ref == second.attestation.serving_unit_ref
        )
        snapshot = _authority_snapshot(
            session.composition.agent_platform_runtime.stores,
            adapter,
            environment=environment,
            revision_id=revision_id,
        )
        assert snapshot.serving_pointer_revision == pointer
        assert snapshot.physical_unit_count == units
        assert snapshot.active_revision_ids == (revision_id,)
    finally:
        session.close()
