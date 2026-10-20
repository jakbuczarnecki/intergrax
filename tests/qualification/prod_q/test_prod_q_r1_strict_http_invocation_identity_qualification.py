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
    execute_sanctioned_create_with_invocation,
    external_work_adapter_metadata,
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

_EXPECTED_START_HEAD: Final = "8cf15daa9fffd61b579d5d9d4732e5eca87560f2"


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


def test_h09_http_strict_post_traverses_reflex_sanctioned_pis_and_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
        assert_identity_continuity,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        _QUAL_PROVIDER,
    )

    revision_id = "rev-h09-http-reflex-create"
    idempotency_key = "idem-h09-http-reflex-create"
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id=revision_id,
        default_agent_id="external_contractor_adapter",
    )
    assert_identity_continuity(fixture.session, revision_id=revision_id)
    pis = fixture.session.composition.provider_invocation_store
    assert isinstance(pis, SQLiteProviderInvocationStore)

    intents_before = list_intent_invocation_ids(fixture.provider_invocation_db)
    baseline_create_calls = fixture.external_work_fake.create_calls
    expected_intent_count_after_admission = len(intents_before) + 1

    original_create_work = fixture.external_work_fake.create_work

    def _create_work_requires_prior_pis_intent(request: object) -> object:
        assert (
            len(list_intent_invocation_ids(fixture.provider_invocation_db))
            == expected_intent_count_after_admission
        )
        assert fixture.external_work_fake.create_calls == baseline_create_calls
        return original_create_work(request)  # type: ignore[arg-type]

    fixture.external_work_fake.create_work = _create_work_requires_prior_pis_intent  # type: ignore[method-assign]

    payload = run_request_json(fixture.tenant_id, idempotency_key=idempotency_key)
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        response = client.post(
            fixture.run_url,
            json=payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["agent_id"] == "external_contractor_adapter"
        task_id = str(body["task_id"])
        run_id = str(body["run_id"])
        assert task_id and run_id

        assert fixture.external_work_fake.create_calls == baseline_create_calls + 1

        intents_after = list_intent_invocation_ids(fixture.provider_invocation_db)
        new_invocation_ids = [
            inv_id for inv_id in intents_after if inv_id not in intents_before
        ]
        assert len(new_invocation_ids) == 1
        invocation_id = new_invocation_ids[0]
        assert invocation_id.startswith("gca-pinv:")
        assert invocation_id not in {task_id, run_id}

        inv = pis.get_invocation(invocation_id)
        assert inv is not None
        assert inv.task_id == task_id
        assert inv.run_id == run_id
        assert inv.provider_id == _QUAL_PROVIDER
        assert inv.idempotency_key == idempotency_key
        assert inv.operation == "create_work"

        outcome = pis.get_outcome(invocation_id)
        assert outcome is not None
        assert outcome.status is ProviderInvocationStatus.SUCCEEDED

    response_meta = body.get("metadata") or {}
    domain_summary = response_meta.get("domain_summary") or response_meta.get(
        "external_work",
    )
    if isinstance(domain_summary, dict):
        assert domain_summary.get("used") is True

    assert owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )


def test_h10_http_strict_post_accept_quote_traverses_reflex_sanctioned_pis_and_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
        assert_identity_continuity,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        _QUAL_PROVIDER,
        follow_up_strict_http_run_request_json,
        post_strict_http_create_prerequisite,
    )
    from intergrax.contracts.sanctioned_external_work_operation import (
        SanctionedExternalWorkMutationOperation,
    )

    revision_id = "rev-h10-http-reflex-accept"
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id=revision_id,
        default_agent_id="external_contractor_adapter",
    )
    assert_identity_continuity(fixture.session, revision_id=revision_id)
    pis = fixture.session.composition.provider_invocation_store
    assert isinstance(pis, SQLiteProviderInvocationStore)

    with strict_http_qualification_asgi_lifespan(fixture) as client:
        prerequisite = post_strict_http_create_prerequisite(
            client,
            fixture,
            idempotency_key="idem-h10-setup-create",
        )
        intents_before_accept = list_intent_invocation_ids(fixture.provider_invocation_db)
        expected_intents_after_admission = len(intents_before_accept) + 1
        baseline_accept_calls = fixture.external_work_fake.accept_calls

        original_accept = fixture.external_work_fake.submit_quote_acceptance

        def _accept_requires_prior_pis_intent(*args: object, **kwargs: object) -> object:
            assert (
                len(list_intent_invocation_ids(fixture.provider_invocation_db))
                == expected_intents_after_admission
            )
            assert fixture.external_work_fake.accept_calls == baseline_accept_calls
            return original_accept(*args, **kwargs)  # type: ignore[arg-type]

        fixture.external_work_fake.submit_quote_acceptance = _accept_requires_prior_pis_intent  # type: ignore[method-assign]

        accept_payload = follow_up_strict_http_run_request_json(
            fixture,
            prerequisite,
            operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
            idempotency_key="idem-h10-http-reflex-accept",
        )
        response = client.post(
            fixture.run_url,
            json=accept_payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert fixture.external_work_fake.accept_calls == baseline_accept_calls + 1
        assert "mapped" in str(body.get("answer") or "")
        assert fixture.external_work_fake.create_calls == prerequisite.baseline_create_calls + 1

        intents_after = list_intent_invocation_ids(fixture.provider_invocation_db)
        new_invocation_ids = [
            inv_id for inv_id in intents_after if inv_id not in prerequisite.baseline_intent_ids
        ]
        assert len(new_invocation_ids) == 2
        accept_invocation_id = [
            inv_id for inv_id in new_invocation_ids if inv_id not in intents_before_accept
        ][0]
        assert accept_invocation_id.startswith("inv-")

        inv = pis.get_invocation(accept_invocation_id)
        assert inv is not None
        assert inv.operation == "submit_quote_acceptance"
        assert inv.provider_id == _QUAL_PROVIDER
        assert inv.idempotency_key == "idem-h10-http-reflex-accept"

        outcome = pis.get_outcome(accept_invocation_id)
        assert outcome is not None
        assert outcome.status is ProviderInvocationStatus.SUCCEEDED

    assert owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )


def test_qual_strict_http_policy_bundle_allows_cancel_runtime_policy() -> None:
    from external_contractor_adapter.side_effect_actions import ACTION_CANCEL_EXTERNAL_WORK
    from intergrax.contracts.meaningful_side_effect import (
        MeaningfulSideEffectKind,
        MeaningfulSideEffectRequest,
    )
    from intergrax.contracts.runtime_policy import PolicyAction
    from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
        RuntimePolicyBundleEvaluator,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        _QUAL_PRINCIPAL,
        _qual_strict_http_policy_bundle,
    )
    from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    bundle = _qual_strict_http_policy_bundle()
    evaluator = RuntimePolicyBundleEvaluator(bundle)
    evaluated = evaluator.evaluate(
        MeaningfulSideEffectRequest(
            action=ACTION_CANCEL_EXTERNAL_WORK,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="idem-policy-smoke",
            task_id=str(task_id),
            run_id=str(run_id),
            attempt_id=str(attempt_id),
            execution_id=str(execution_id),
            principal_id=_QUAL_PRINCIPAL,
            tenant_id="env-prod",
        ),
    )
    assert evaluated.decision.action is PolicyAction.ALLOW


def test_h11_http_strict_post_cancel_traverses_reflex_sanctioned_pis_and_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
        assert_identity_continuity,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        _QUAL_PROVIDER,
        follow_up_strict_http_run_request_json,
        post_strict_http_create_prerequisite,
    )
    from intergrax.contracts.sanctioned_external_work_operation import (
        SanctionedExternalWorkMutationOperation,
    )

    revision_id = "rev-h11-http-reflex-cancel"
    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id=revision_id,
        default_agent_id="external_contractor_adapter",
    )
    assert_identity_continuity(fixture.session, revision_id=revision_id)
    pis = fixture.session.composition.provider_invocation_store
    assert isinstance(pis, SQLiteProviderInvocationStore)

    with strict_http_qualification_asgi_lifespan(fixture) as client:
        prerequisite = post_strict_http_create_prerequisite(
            client,
            fixture,
            idempotency_key="idem-h11-setup-create",
        )
        intents_before_cancel = list_intent_invocation_ids(fixture.provider_invocation_db)
        expected_intents_after_admission = len(intents_before_cancel) + 1
        baseline_cancel_calls = fixture.external_work_fake.cancel_calls

        cancel_payload = follow_up_strict_http_run_request_json(
            fixture,
            prerequisite,
            operation=SanctionedExternalWorkMutationOperation.CANCEL,
            idempotency_key="idem-h11-http-reflex-cancel",
        )
        response = client.post(
            fixture.run_url,
            json=cancel_payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert fixture.external_work_fake.cancel_calls == baseline_cancel_calls + 1, {
            "answer": body.get("answer"),
        }
        assert fixture.external_work_fake.create_calls == prerequisite.baseline_create_calls + 1

        intents_after = list_intent_invocation_ids(fixture.provider_invocation_db)
        new_invocation_ids = [
            inv_id for inv_id in intents_after if inv_id not in prerequisite.baseline_intent_ids
        ]
        assert len(new_invocation_ids) == 2
        cancel_invocation_id = [
            inv_id for inv_id in new_invocation_ids if inv_id not in intents_before_cancel
        ][0]
        assert cancel_invocation_id.startswith("inv-")

        inv = pis.get_invocation(cancel_invocation_id)
        assert inv is not None
        assert inv.operation == "cancel_work"
        assert inv.provider_id == _QUAL_PROVIDER
        assert inv.idempotency_key == "idem-h11-http-reflex-cancel"

        outcome = pis.get_outcome(cancel_invocation_id)
        assert outcome is not None
        assert outcome.status is ProviderInvocationStatus.SUCCEEDED

    assert owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )


def test_h10n_accept_without_prior_create_fail_closed_zero_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from intergrax.contracts.sanctioned_external_work_operation import (
        META_PRIOR_EXECUTION_ID,
        META_SANCTIONED_MUTATION_OPERATION,
        SanctionedExternalWorkMutationOperation,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        external_work_http_metadata,
        quote_acceptance_evidence_for_qual,
    )
    from external_contractor_adapter.external_work_adapter import (
        META_ACCEPTANCE_IDEMPOTENCY_KEY,
        META_QUOTE_ACCEPTANCE,
    )
    from intergrax.contracts.execution_identity import mint_execution_id

    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h10n-missing-prereq",
        default_agent_id="external_contractor_adapter",
    )
    baseline_accept = fixture.external_work_fake.accept_calls
    meta = external_work_http_metadata(
        tenant_id=fixture.tenant_id,
        idempotency_key="idem-h10n-missing",
    )
    meta[META_SANCTIONED_MUTATION_OPERATION] = (
        SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE.value
    )
    meta[META_PRIOR_EXECUTION_ID] = str(mint_execution_id())
    meta[META_QUOTE_ACCEPTANCE] = quote_acceptance_evidence_for_qual(
        "quote-missing",
        tenant_id=fixture.tenant_id,
        scope_digest="sha256:" + ("aa" * 32),
    )
    meta[META_ACCEPTANCE_IDEMPOTENCY_KEY] = "idem-h10n-missing"
    payload = run_request_json(
        fixture.tenant_id,
        idempotency_key="idem-h10n-http",
        metadata=meta,
    )
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        response = client.post(
            fixture.run_url,
            json=payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
    assert fixture.external_work_fake.accept_calls == baseline_accept


def test_h11n_cancel_without_prior_create_fail_closed_zero_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from intergrax.contracts.sanctioned_external_work_operation import (
        META_PRIOR_EXECUTION_ID,
        META_SANCTIONED_MUTATION_OPERATION,
        SanctionedExternalWorkMutationOperation,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        external_work_http_metadata,
    )
    from intergrax.contracts.execution_identity import mint_execution_id

    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h11n-missing-prereq",
        default_agent_id="external_contractor_adapter",
    )
    baseline_cancel = fixture.external_work_fake.cancel_calls
    meta = external_work_http_metadata(
        tenant_id=fixture.tenant_id,
        idempotency_key="idem-h11n-missing",
    )
    meta[META_SANCTIONED_MUTATION_OPERATION] = (
        SanctionedExternalWorkMutationOperation.CANCEL.value
    )
    meta[META_PRIOR_EXECUTION_ID] = str(mint_execution_id())
    payload = run_request_json(
        fixture.tenant_id,
        idempotency_key="idem-h11n-http",
        metadata=meta,
    )
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        response = client.post(
            fixture.run_url,
            json=payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
    assert fixture.external_work_fake.cancel_calls == baseline_cancel


def test_h10_h11n_unsupported_sanctioned_operation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from intergrax.contracts.sanctioned_external_work_operation import (
        META_SANCTIONED_MUTATION_OPERATION,
    )
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        external_work_http_metadata,
    )

    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h10-h11n-unsupported-op",
        default_agent_id="external_contractor_adapter",
    )
    baseline = (
        fixture.external_work_fake.create_calls,
        fixture.external_work_fake.accept_calls,
        fixture.external_work_fake.cancel_calls,
    )
    meta = external_work_http_metadata(
        tenant_id=fixture.tenant_id,
        idempotency_key="idem-unsupported-op",
    )
    meta[META_SANCTIONED_MUTATION_OPERATION] = "terminate_physical"
    payload = run_request_json(
        fixture.tenant_id,
        metadata=meta,
    )
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        response = client.post(
            fixture.run_url,
            json=payload,
            headers=strict_http_auth_headers(),
        )
        assert response.status_code == 200, response.text
    assert (
        fixture.external_work_fake.create_calls,
        fixture.external_work_fake.accept_calls,
        fixture.external_work_fake.cancel_calls,
    ) == baseline


def test_h10n_wrong_tenant_accept_denied_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.qualification.prod_q._prod_q_strict_http_invocation_identity_support import (
        follow_up_strict_http_run_request_json,
        post_strict_http_create_prerequisite,
    )
    from intergrax.contracts.sanctioned_external_work_operation import (
        SanctionedExternalWorkMutationOperation,
    )

    fixture = build_strict_http_invocation_qual_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-h10n-wrong-tenant",
        default_agent_id="external_contractor_adapter",
    )
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        prerequisite = post_strict_http_create_prerequisite(
            client,
            fixture,
            idempotency_key="idem-h10n-tenant-setup",
        )
        baseline_accept = fixture.external_work_fake.accept_calls
        accept_payload = follow_up_strict_http_run_request_json(
            fixture,
            prerequisite,
            operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
            idempotency_key="idem-h10n-wrong-tenant-accept",
        )
        response = client.post(
            fixture.run_url,
            json=accept_payload,
            headers=strict_http_auth_headers(_WRONG_TENANT_KEY),
        )
        assert response.status_code == 403, response.text
    assert fixture.external_work_fake.accept_calls == baseline_accept

