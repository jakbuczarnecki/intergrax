# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1 — strict HTTP invocation identity qualification (DIM20 parent evidence)."""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
)
from applications.governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from applications.governed_contractor_application.serving.schemas import (
    GovernedContractorRunRequestV1,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryRequest,
    evaluate_provider_invocation_recovery,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionPorts,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationPersistenceError,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import repo_root
from tests.qualification.prod_q import _prod_q_strict_http_invocation_identity_support as _strict_http_support
from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
    assert_no_leaked_owned_strict_http_invocation_sessions,
    build_sanctioned_external_work_runtime,
    build_strict_http_invocation_qual_fixture,
    close_registered_owned_strict_http_invocation_sessions,
    owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan,
    owned_strict_http_invocation_session_registration_count,
    register_owned_strict_http_invocation_close_target_for_regression,
    register_owned_strict_http_invocation_session,
    execute_sanctioned_create_with_invocation,
    external_work_adapter_metadata,
    external_work_http_metadata,
    _QUAL_T0,
    fixed_provider_invocation,
    list_intent_invocation_ids,
    post_strict_http_run,
    run_request_json,
    strict_http_qualification_asgi_lifespan,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    _WRONG_TENANT_KEY,
    strict_http_auth_headers,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _completed_task_result,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative import (
    _INVOCATION_ID as _IG_F01_INVOCATION_ID,
)
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)
from applications.governed_contractor_application.tests.host.test_gr7_a7_provider_recovery import (
    _CREATE_CONTRACT,
    _RecordingRepeatPort,
    _SelectIdempotentRepeatRecoveryPolicy,
    _invocation as _recovery_invocation,
    _outcome as _recovery_outcome,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityRequest,
)
from applications.governed_contractor_application.tests.host.test_gr7_a7_provider_recovery import (
    _AllowRepeatPolicy,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "e5e5d326184b23d3eb93786e60db7a4c2154499d"


@pytest.fixture(autouse=True)
def _owned_strict_http_invocation_session_lifecycle() -> None:
    assert_no_leaked_owned_strict_http_invocation_sessions()
    yield
    close_registered_owned_strict_http_invocation_sessions()


def test_g00_qualification_baseline_ancestor() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_h01_http_contract_has_no_caller_supplied_invocation_id() -> None:
    fields = GovernedContractorRunRequestV1.model_fields
    assert "invocation_id" not in fields
    assert "run_id" not in fields
    assert "task_id" not in fields


def test_h01_baseline_strict_http_and_sanctioned_external_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h01-baseline",
    )
    identity = default_gr3_identity_bundle()
    production_runtime, pis, identity = build_sanctioned_external_work_runtime(
        fixture,
        identity=identity,
    )
    inv_id = "inv-h01-sanctioned"
    execute_sanctioned_create_with_invocation(
        production_runtime,
        fixture,
        invocation_id=inv_id,
        idempotency_key="idem-h01-sanctioned",
        identity=identity,
    )
    assert fixture.external_work_fake.create_calls == 1
    assert pis.get_invocation(inv_id) is not None
    assert list_intent_invocation_ids(fixture.provider_invocation_db) == (inv_id,)

    from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
        _run_request_json,
    )

    with patch(
        _INNER_DELEGATE,
        new_callable=AsyncMock,
        return_value=_completed_task_result(),
    ):
        body = post_strict_http_run(fixture, _run_request_json(fixture.tenant_id))
    assert body["task_id"]
    assert body["run_id"]
    assert body["task_id"] != inv_id
    assert body["run_id"] != inv_id


def test_h02_same_canonical_invocation_gate_replay_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h02-replay",
    )
    identity = default_gr3_identity_bundle()
    production_runtime, pis, identity = build_sanctioned_external_work_runtime(
        fixture,
        identity=identity,
    )
    adapter = production_runtime.adapter
    task_id, run_id, attempt_id, execution_id = identity
    invocation = fixed_provider_invocation(
        str(task_id),
        str(run_id),
        _IG_F01_INVOCATION_ID,
    )
    request = adapter.build_create_request(
        task_id=str(task_id),
        run_id=str(run_id),
        metadata=external_work_adapter_metadata(
            tenant_id=fixture.tenant_id,
            idempotency_key="idem-h02-replay",
        ),
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
        governance_tenant_id=fixture.tenant_id,
        governance_workspace_id="http-inv-qual-ws",
        governance_principal_id="http-inv-qual-principal",
    ):
        first = adapter.create_and_map(
            request,
            principal_id="http-inv-qual-principal",
            tenant_id=fixture.tenant_id,
            provider_invocation=invocation,
        )
        assert first.used is True
        second = adapter.create_and_map(
            request,
            principal_id="http-inv-qual-principal",
            tenant_id=fixture.tenant_id,
            provider_invocation=invocation,
        )
    assert fixture.external_work_fake.create_calls == 1
    assert second.used is False
    assert second.error_message and "dispatch_not_admitted" in second.error_message
    assert pis.get_outcome(_IG_F01_INVOCATION_ID) is None


def test_h03_unresolved_external_effect_no_ordinary_redispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h03-ambiguous",
    )
    production_runtime, pis, _identity = build_sanctioned_external_work_runtime(fixture)
    inv = _recovery_invocation(invocation_id="inv-h03-ambiguous")
    pis.insert_invocation_if_absent(inv)
    outcome = _recovery_outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    pis.put_outcome(outcome)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
    )
    decision = evaluate_provider_invocation_recovery(request)
    assert decision.action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 0
    assert result.provider_mutation_count == 0
    gate = GovernedProviderInvocationDispatchGate(store=pis, clock=lambda: _QUAL_T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(ProviderInvocationPersistenceError, match="dispatch_not_admitted"):
        gate.dispatch_after_intent_persisted(inv, _execute)
    assert calls == 0


def test_h04_independent_identical_http_requests_distinct_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
        _run_request_json,
    )

    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h04-dup-payload",
    )
    payload = _run_request_json(fixture.tenant_id)

    async def _inner_fresh(*_args, **_kwargs):
        return _completed_task_result()

    with patch(
        _INNER_DELEGATE,
        new_callable=AsyncMock,
        side_effect=_inner_fresh,
    ):
        with strict_http_qualification_asgi_lifespan(fixture) as client:
            first_resp = client.post(
                fixture.run_url,
                json=payload,
                headers=strict_http_auth_headers(),
            )
            second_resp = client.post(
                fixture.run_url,
                json=payload,
                headers=strict_http_auth_headers(),
            )
    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    first = first_resp.json()
    second = second_resp.json()
    assert first["task_id"] != second["task_id"]
    assert first["run_id"] != second["run_id"]


def test_h05_tenant_isolation_http_and_pis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h05-tenant",
    )
    from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
        _run_request_json,
    )

    identity = default_gr3_identity_bundle()
    production_runtime, pis, identity = build_sanctioned_external_work_runtime(
        fixture,
        identity=identity,
    )
    inv_a = fixed_provider_invocation("task-a", "run-a", "inv-h05-tenant-a")
    pis.insert_invocation_if_absent(inv_a)
    assert pis.get_invocation("inv-h05-tenant-a") is not None
    assert pis.get_invocation("inv-h05-other-env") is None
    execute_sanctioned_create_with_invocation(
        production_runtime,
        fixture,
        invocation_id="inv-h05-tenant-b",
        idempotency_key="idem-h05-b",
        identity=identity,
    )
    assert list_intent_invocation_ids(fixture.provider_invocation_db) == (
        "inv-h05-tenant-a",
        "inv-h05-tenant-b",
    )

    with strict_http_qualification_asgi_lifespan(fixture) as client:
        denied = client.post(
            fixture.run_url,
            json=_run_request_json(fixture.tenant_id),
            headers=strict_http_auth_headers(_WRONG_TENANT_KEY),
        )
    assert denied.status_code == 403


def test_h06_conflicting_invocation_identity_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h06-conflict",
    )
    _, pis, _identity = build_sanctioned_external_work_runtime(fixture)
    inv = fixed_provider_invocation("task-1", "run-1", "inv-h06-conflict")
    pis.insert_invocation_if_absent(inv)
    conflict = inv.model_copy(update={"run_id": "run-other"})
    with pytest.raises(ProviderInvocationConflictError):
        pis.insert_invocation_if_absent(conflict)
    gate = GovernedProviderInvocationDispatchGate(store=pis, clock=lambda: _QUAL_T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(ProviderInvocationConflictError):
        gate.dispatch_after_intent_persisted(conflict, _execute)
    assert calls == 0


def test_h07_concurrent_gate_single_execute_on_session_pis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h07-concurrent",
    )
    _, pis, _identity = build_sanctioned_external_work_runtime(fixture)
    inv = fixed_provider_invocation("task-c", "run-c", "inv-h07-concurrent")
    gate = GovernedProviderInvocationDispatchGate(store=pis, clock=lambda: _QUAL_T0)
    calls = 0
    lock = threading.Lock()

    def _execute() -> str:
        nonlocal calls
        with lock:
            calls += 1
        return "ok"

    errors: list[BaseException] = []

    def _dispatch() -> None:
        try:
            gate.dispatch_after_intent_persisted(inv, _execute)
        except ProviderInvocationPersistenceError as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_dispatch) for _ in range(12)]
        for future in as_completed(futures):
            future.result()

    assert calls == 1
    assert len(errors) == 11


def test_h08_strict_session_sqlite_pis_and_dispatch_gate_wiring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h08-composition",
    )
    pis = fixture.session.composition.provider_invocation_store
    assert isinstance(pis, SQLiteProviderInvocationStore)
    assert pis.is_durable is True
    production_runtime, _, _identity = build_sanctioned_external_work_runtime(fixture)
    adapter = production_runtime.adapter
    assert adapter._invocation_dispatch is not None
    assert adapter._reliability_aware_invocation_dispatch is not None
    gate = adapter._invocation_dispatch
    assert type(gate).__name__ == "GovernedProviderInvocationDispatchGate"
    assert gate.store is pis


def test_h03b_sanctioned_recovery_repeat_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h03b-recovery",
    )
    _, pis, _identity = build_sanctioned_external_work_runtime(fixture)
    inv = _recovery_invocation(invocation_id="inv-h03b-failed")
    outcome = _recovery_outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    pis.insert_invocation_if_absent(inv)
    pis.put_outcome(outcome)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=_CREATE_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=eligibility,
        repeat_execution_supported=True,
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 1
    assert result.provider_mutation_count == 1
    assert repeat_port.last_idempotency_key == inv.idempotency_key


@dataclass
class _QualOwnedSessionDouble:
    """Deterministic close double for owned-session registry regression tests."""

    label: str
    close_error: Exception | None = None
    close_calls: int = 0
    close_order: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.close_calls += 1
        self.close_order.append(self.label)
        if self.close_error is not None:
            raise self.close_error


def test_r01_owned_session_registry_releases_after_successful_close() -> None:
    session = _QualOwnedSessionDouble(label="r01")
    register_owned_strict_http_invocation_close_target_for_regression(session)
    assert owned_strict_http_invocation_session_registration_count() == 1
    close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 0
    assert session.close_calls == 1


def test_r02_failed_close_keeps_registry_evidence_and_blocks_next_test() -> None:
    session = _QualOwnedSessionDouble(
        label="r02",
        close_error=StrictProductionServingLauncherError("r02 close failed"),
    )
    register_owned_strict_http_invocation_close_target_for_regression(session)
    with pytest.raises(StrictProductionServingLauncherError, match="r02 close failed"):
        close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 1
    with pytest.raises(AssertionError, match="leaked from a prior test"):
        assert_no_leaked_owned_strict_http_invocation_sessions()
    session.close_error = None
    close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 0


def test_r03_already_closed_message_does_not_suppress_close_failure() -> None:
    session = _QualOwnedSessionDouble(
        label="r03",
        close_error=StrictProductionServingLauncherError(
            "strict production serving session already closed",
        ),
    )
    register_owned_strict_http_invocation_close_target_for_regression(session)
    with pytest.raises(StrictProductionServingLauncherError, match="already closed"):
        close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 1
    session.close_error = None
    close_registered_owned_strict_http_invocation_sessions()


def test_r04_multiple_owned_sessions_close_in_reverse_registration_order() -> None:
    order: list[str] = []
    first = _QualOwnedSessionDouble(label="first", close_order=order)
    second = _QualOwnedSessionDouble(label="second", close_order=order)
    third = _QualOwnedSessionDouble(label="third", close_order=order)
    register_owned_strict_http_invocation_close_target_for_regression(first)
    register_owned_strict_http_invocation_close_target_for_regression(second)
    register_owned_strict_http_invocation_close_target_for_regression(third)
    close_registered_owned_strict_http_invocation_sessions()
    assert order == ["third", "second", "first"]
    assert owned_strict_http_invocation_session_registration_count() == 0


def test_r05_qualification_support_never_resets_global_production_lifecycle_slot() -> None:
    support_path = (
        Path(__file__).resolve().parent / "_prod_q_strict_http_invocation_identity_support.py"
    )
    text = support_path.read_text(encoding="utf-8")
    assert "production_process_lifecycle" not in text
    assert "_active_production_process_lifecycle" not in text
    assert "_process_closed" not in text
    assert "._closed" not in text
    assert "close_process_resources" not in text
    assert "__getattr__" not in text
    assert "getattr(" not in text


def _owned_strict_http_invocation_registry_support_source() -> str:
    support_path = (
        Path(__file__).resolve().parent / "_prod_q_strict_http_invocation_identity_support.py"
    )
    text = support_path.read_text(encoding="utf-8")
    start = text.index("_owned_strict_http_invocation_sessions")
    end = text.index("@dataclass(frozen=True, slots=True)\nclass StrictHttpInvocationQualFixture")
    return text[start:end]


def test_r07_strict_http_fixture_passes_concrete_strict_production_session_to_asgi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r07-concrete-session",
    )
    assert isinstance(fixture.session, StrictProductionServingSession)
    assert type(fixture.session) is StrictProductionServingSession


def test_r08_qualification_support_has_no_reflection_or_typing_suppression_at_boundary() -> None:
    text = _owned_strict_http_invocation_registry_support_source()
    assert "__getattr__" not in text
    assert "getattr(" not in text
    assert "setattr(" not in text
    assert "type: ignore" not in text
    assert " cast(" not in text
    assert ": Any" not in text


def test_r09_asgi_lifespan_close_does_not_fail_owned_session_registry_teardown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r09-lifespan-close",
    )
    assert owned_strict_http_invocation_session_registration_count() == 1
    with strict_http_qualification_asgi_lifespan(fixture):
        pass
    assert owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
    close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 0
    assert not owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )


def test_r10_asgi_lifespan_exit_failure_does_not_record_completed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r10-lifespan-exit-failure",
    )
    assert not owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )

    def _close_raises() -> None:
        raise StrictProductionServingLauncherError("r10 lifespan close failed")

    with patch.object(fixture.session, "close", _close_raises):
        with pytest.raises(
            StrictProductionServingLauncherError,
            match="r10 lifespan close failed",
        ):
            with strict_http_qualification_asgi_lifespan(fixture):
                pass

    assert not owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
    assert owned_strict_http_invocation_session_registration_count() == 1


def test_r11_asgi_lifespan_body_failure_still_records_completed_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r11-lifespan-body-failure",
    )
    assert owned_strict_http_invocation_session_registration_count() == 1
    with pytest.raises(ValueError, match="r11 body failed"):
        with strict_http_qualification_asgi_lifespan(fixture):
            raise ValueError("r11 body failed")
    assert owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
    close_registered_owned_strict_http_invocation_sessions()
    assert owned_strict_http_invocation_session_registration_count() == 0


def test_r12_asgi_lifespan_body_and_shutdown_failure_preserves_both_without_completion_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r12-lifespan-dual-failure",
    )

    def _close_raises() -> None:
        raise StrictProductionServingLauncherError("r12 lifespan close failed")

    with patch.object(fixture.session, "close", _close_raises):
        with pytest.raises(ExceptionGroup) as exc_info:
            with strict_http_qualification_asgi_lifespan(fixture):
                raise ValueError("r12 body failed")
    group = exc_info.value
    assert len(group.exceptions) == 2
    assert isinstance(group.exceptions[0], ValueError)
    assert isinstance(group.exceptions[1], StrictProductionServingLauncherError)

    assert not owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
    assert owned_strict_http_invocation_session_registration_count() == 1


def test_r13_asgi_lifespan_fatal_shutdown_propagates_without_invalid_exception_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-r13-lifespan-fatal-shutdown",
    )

    class _FakeTestClient:
        def __init__(self, _app: object) -> None:
            pass

        def __enter__(self) -> object:
            return object()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: object,
        ) -> bool:
            if exc_type is None:
                raise KeyboardInterrupt("r13 fatal shutdown during unwind")
            return False

    monkeypatch.setattr(_strict_http_support, "TestClient", _FakeTestClient)
    with pytest.raises(KeyboardInterrupt, match="r13 fatal shutdown"):
        with strict_http_qualification_asgi_lifespan(fixture):
            raise ValueError("r13 body failed")
    assert not owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
