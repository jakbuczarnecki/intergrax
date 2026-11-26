# © Artur Czarnecki. All rights reserved.

"""Cross-process helper for ARCH-FREEZE package F idempotency proofs."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_task_id,
)
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    InvocationUncertaintyError,
    PreEffectSuspendedWorkRecoveryAuthority,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from pydantic import BaseModel
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
)

_RUN_ID = canonical_run_id_for_tests("arch-freeze-f-run")
_TENANT = "test-tenant"
_KEY = os.environ.get("ARCH_FREEZE_F_KEY", "arch-freeze-f-idem")


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    value: int = 1


def _contract() -> ToolContract:
    return ToolContract(
        tool_id="arch-freeze-f-effect",
        name="arch-freeze-f-effect",
        description="package F cross-process proof",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )


def _coordinator() -> tuple[IdempotencyPreEffectCoordinator, str]:
    db_path = os.environ["ARCH_FREEZE_F_DB"]
    store = SQLiteIdempotencyStore(db_path)
    return IdempotencyPreEffectCoordinator(idempotency_store=store), db_path


def _request() -> ToolExecutionRequest[_In]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id="arch-freeze-f-effect",
        input=_In(),
        idempotency_key=_KEY,
    )


def _bump_backend_counter(counter_path: Path) -> None:
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    current = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
    counter_path.write_text(str(current + 1), encoding="utf-8")


def _process_a_after_admit_crash() -> None:
    coordinator, _ = _coordinator()
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    counter = Path(os.environ["ARCH_FREEZE_F_COUNTER"])
    coordination = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert coordination.claim_context is not None
    coordinator.admit_external_effect_may_have_started(claim_context=coordination.claim_context)
    _bump_backend_counter(counter)
    coordinator.on_post_claim_exception(
        claim_context=coordination.claim_context,
        contract=contract,
        effect_may_have_started=True,
    )


def _process_a_before_admit_exit() -> None:
    coordinator, _ = _coordinator()
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    coordination = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert coordination.claim_context is not None
    auth_path = Path(os.environ["ARCH_FREEZE_F_AUTHORITY"])
    auth_path.write_text(
        f"{coordination.claim_context.claim.owner_id},{coordination.claim_context.claim.fence}",
        encoding="utf-8",
    )
    os._exit(0)  # noqa: PLW1509 — deliberate orphan STARTED row without admit bit


def _process_a_complete() -> None:
    coordinator, _ = _coordinator()
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    counter = Path(os.environ["ARCH_FREEZE_F_COUNTER"])
    coordination = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert coordination.claim_context is not None
    coordinator.admit_external_effect_may_have_started(claim_context=coordination.claim_context)
    _bump_backend_counter(counter)
    coordinator.after_external_effect(
        claim_context=coordination.claim_context,
        contract=contract,
        result=ToolExecutionResult.ok(_Out(value=1)),
    )


def _process_b_observe_uncertain() -> None:
    coordinator, _ = _coordinator()
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    counter = Path(os.environ["ARCH_FREEZE_F_COUNTER"])
    before = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    try:
        coordinator.before_external_effect(
            state=state,
            contract=contract,
            request=request,
        )
    except InvocationUncertaintyError:
        after = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
        if after != before:
            raise SystemExit("backend counter increased on uncertain replay")
        print("uncertain")
        return
    raise SystemExit("expected InvocationUncertaintyError")


def _process_b_reconcile_and_call_once() -> None:
    coordinator, _ = _coordinator()
    store = coordinator._store
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    counter = Path(os.environ["ARCH_FREEZE_F_COUNTER"])
    operation_identity = compute_invocation_operation_identity(request.tool_id, request.input)
    auth_path = Path(os.environ["ARCH_FREEZE_F_AUTHORITY"])
    owner_id, fence_s = auth_path.read_text(encoding="utf-8").split(",", 1)
    authority = PreEffectSuspendedWorkRecoveryAuthority(
        owner_id=owner_id,
        fence=int(fence_s),
    )
    if not store.reconcile_abandoned_pre_effect_not_started(
        _TENANT,
        _KEY,
        operation_identity,
        recovery_authority=authority,
    ):
        raise SystemExit("reconcile did not clear pre-admit orphan")
    before = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    coordination = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    if coordination.claim_context is None:
        raise SystemExit("expected fresh claim after reconcile")
    coordinator.admit_external_effect_may_have_started(claim_context=coordination.claim_context)
    _bump_backend_counter(counter)
    coordinator.after_external_effect(
        claim_context=coordination.claim_context,
        contract=contract,
        result=ToolExecutionResult.ok(_Out(value=1)),
    )
    after = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    if after != before + 1:
        raise SystemExit("expected exactly one backend call after reconcile")


def _process_b_replay_completed() -> None:
    coordinator, _ = _coordinator()
    state = build_runtime_state_for_tests(run_id="arch-freeze-f-run")
    contract = _contract()
    request = _request()
    counter = Path(os.environ["ARCH_FREEZE_F_COUNTER"])
    before = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    coordination = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    if coordination.outcome is not ClaimOutcome.REPLAY_COMPLETED:
        raise SystemExit(f"expected replay, got {coordination.outcome}")
    after = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    if after != before:
        raise SystemExit("backend counter increased on completed replay")


def main() -> None:
    mode = os.environ.get("ARCH_FREEZE_F_MODE", "").strip()
    handlers = {
        "process_a_after_admit_crash": _process_a_after_admit_crash,
        "process_a_before_admit_exit": _process_a_before_admit_exit,
        "process_a_complete": _process_a_complete,
        "process_b_observe_uncertain": _process_b_observe_uncertain,
        "process_b_reconcile_and_call_once": _process_b_reconcile_and_call_once,
        "process_b_replay_completed": _process_b_replay_completed,
    }
    bind_active_execution_identity(
        run_id=_RUN_ID,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        task_id=mint_task_id(),
    )
    handler = handlers.get(mode)
    if handler is None:
        raise SystemExit(f"unknown ARCH_FREEZE_F_MODE: {mode!r}")
    handler()


if __name__ == "__main__":
    main()
