# © Artur Czarnecki. All rights reserved.

"""OD-R2-03 subprocess worker — fresh-process and abrupt-termination qualification phases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.governed_execution_result import GovernedExecutionResultV2

from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import ContinuationReason

from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor

from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2 import (
    _Identity,
    _Seed,
    _T0,
    _close_pair,
    _create,
    _expected_invocation_id,
    _inject_controlled_gca_execution_loss,
    _inject_controlled_pis_outcome_loss,
    _mint,
    _open_pair,
    _reopen,
    _runtime,
    _seed,
)

_EXIT_OK = 0
_EXIT_ERROR = 1
_MANIFEST = "od_r2_03_manifest.json"
_RC_CROSS_MANIFEST = "rc_cross_manifest.json"
_READY_FLAG = "od_r2_03_committed_ready.flag"
_RC_CONTINUATION_REQUEST_ID = "gcr-impl01-rc-cross"


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _manifest_path(work_dir: Path) -> Path:
    return work_dir / _MANIFEST


def _seed_to_manifest(seed: _Seed) -> dict[str, str]:
    return {
        "gca_path": str(seed.gca_path),
        "pis_path": str(seed.pis_path),
        "execution_id": seed.identity.execution_key,
        "task_id": str(seed.identity.task_id),
        "run_id": str(seed.identity.run_id),
        "attempt_id": str(seed.identity.attempt_id),
        "invocation_id": seed.invocation_id,
    }


def _manifest_to_seed(data: dict[str, str]) -> _Seed:
    from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2 import (
        _Identity,
    )

    identity = _Identity(
        task_id=data["task_id"],
        run_id=data["run_id"],
        attempt_id=data["attempt_id"],
        execution_id=data["execution_id"],
    )
    return _Seed(
        gca_path=Path(data["gca_path"]),
        pis_path=Path(data["pis_path"]),
        identity=identity,
        invocation_id=data["invocation_id"],
    )


def _cmd_dur_proc_01_seed(work_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    seed = _seed(work_dir, "dur-proc-01")
    payload = _seed_to_manifest(seed)
    _manifest_path(work_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit({"phase": "dur-proc-01-seed", "status": "ok", **payload})
    return _EXIT_OK


def _verify_reopen(seed: _Seed) -> dict[str, Any]:
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        ger = gca.bundle.execution_store.get_result(seed.identity.execution_key)
        follow_on = gca.bundle.execution_store.get_create_follow_on_record(
            seed.identity.execution_key,
        )
        assert isinstance(ger, GovernedExecutionResultV2)
        assert follow_on is not None
        assert follow_on.create_provider_invocation_id == seed.invocation_id
        assert pis.get_outcome(seed.invocation_id) is not None
        second = _create(runtime, seed.identity)
        return {
            "status": "ok",
            "create_calls_after_retry": fake.create_calls,
            "retry_reason": second.reason,
            "invocation_id": seed.invocation_id,
            "expected_invocation_id": _expected_invocation_id(seed.identity),
            "ger_action": ger.action,
            "follow_on_present": follow_on is not None,
        }
    finally:
        _close_pair(gca, pis)


def _cmd_dur_proc_01_verify(work_dir: Path) -> int:
    raw = json.loads(_manifest_path(work_dir).read_text(encoding="utf-8"))
    seed = _manifest_to_seed(raw)
    result = _verify_reopen(seed)
    if result["create_calls_after_retry"] != 0:
        _emit({**result, "phase": "dur-proc-01-verify", "status": "failed"})
        return _EXIT_ERROR
    if result["retry_reason"] != "create_redispatch_prohibited":
        _emit({**result, "phase": "dur-proc-01-verify", "status": "failed"})
        return _EXIT_ERROR
    _emit({"phase": "dur-proc-01-verify", **result})
    return _EXIT_OK


def _cmd_dur_proc_02_committed_wait(work_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    seed = _seed(work_dir, "dur-proc-02-committed")
    payload = _seed_to_manifest(seed)
    _manifest_path(work_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (work_dir / _READY_FLAG).write_text("committed_closed", encoding="utf-8")
    _emit({"phase": "dur-proc-02-committed-wait", "status": "ready", **payload})
    while True:
        time.sleep(0.25)


def _cmd_dur_proc_02_incomplete_seed(work_dir: Path) -> int:
    """Controlled ambiguity: PIS intent without outcome and no GCA GER (post-close inject)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    seed = _seed(work_dir, "dur-proc-02-incomplete")
    _inject_controlled_pis_outcome_loss(seed.pis_path)
    _inject_controlled_gca_execution_loss(seed.gca_path, seed.identity.execution_key)
    payload = _seed_to_manifest(seed)
    payload["ambiguity"] = "pis_intent_without_outcome_and_no_ger"
    _manifest_path(work_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit({"phase": "dur-proc-02-incomplete-seed", "status": "ok", **payload})
    return _EXIT_OK


def _continuation_for_identity(identity: _Identity) -> GovernedContinuationRequest:
    return GovernedContinuationRequest(
        reason=ContinuationReason.QUOTE,
        task_id=identity.task_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        source_agent_id="impl-01-rc-cross",
        prompt="rc-cross continuation qualification",
        continuation_request_id=_RC_CONTINUATION_REQUEST_ID,
    )


def _receipt_execution_id(receipt: object) -> str:
    event = receipt.execution_boundary_event  # type: ignore[attr-defined]
    execution = getattr(event, "execution", None)
    if execution is not None:
        return str(execution.execution_id)
    return str(event.execution_id)


def _rc_cross_manifest_path(work_dir: Path) -> Path:
    return work_dir / _RC_CROSS_MANIFEST


def _runtime_with_attestor(fake: object, identity: _Identity, gca: object, pis: object):
    from applications.governed_contractor_application.host.production_external_work_composition import (
        build_governed_external_work_production_runtime,
    )
    from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
        gr6_seeded_collaborative_work_repositories,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        RecordingAdmissionPort,
        _PRINCIPAL,
        _PROVIDER,
        _TENANT,
        _WORKSPACE,
        _policy_bundle,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from intergrax.contracts.external_work_provider_capabilities import (
        quote_first_partner_capability_fixture,
    )
    from tests.unit.runtime.governance.gr3_test_support import StaticActiveTaskScope

    assert isinstance(fake, DeterministicExternalWorkFake)
    bundle = gca.bundle
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    return build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(identity.task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),
        collaborative_work_repositories=cw,
        execution_store=bundle.execution_store,
        receipt_store=bundle.receipt_store,
        bundle_store=bundle.bundle_store,
        continuation_store=bundle.continuation_store,
        provider_invocation_store=pis,
        reliability_bridge=bridge,
        clock=lambda: _T0,
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
    )


def _cmd_rc_cross_seed(work_dir: Path) -> int:
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    identity = _mint()
    gca_path = work_dir / "rc-cross-gca.sqlite3"
    pis_path = work_dir / "rc-cross-pis.sqlite3"
    fake = DeterministicExternalWorkFake()
    gca, pis = _open_pair(gca_path, pis_path)
    continuation = _continuation_for_identity(identity)
    try:
        runtime = _runtime_with_attestor(fake, identity, gca, pis)
        step = _create(runtime, identity)
        assert step.governed_result is not None
        receipt = gca.bundle.receipt_store.get_receipt(identity.execution_key)
        assert receipt is not None
        gca.bundle.continuation_store.put_continuation(
            str(identity.task_id),
            continuation,
        )
        payload = {
            "gca_path": str(gca_path),
            "execution_id": identity.execution_key,
            "task_id": str(identity.task_id),
            "run_id": str(identity.run_id),
            "attempt_id": str(identity.attempt_id),
            "receipt_id": receipt.receipt_id,
            "receipt_execution_id": _receipt_execution_id(receipt),
            "continuation_request_id": continuation.continuation_request_id,
        }
    finally:
        _close_pair(gca, pis)
    _rc_cross_manifest_path(work_dir).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _emit({"phase": "rc-cross-seed", "status": "ok", **payload})
    return _EXIT_OK


def _cmd_rc_cross_verify(work_dir: Path) -> int:
    from applications.governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
        SQLiteGovernedExternalWorkHostStoreBackend,
    )
    from applications.governed_contractor_application.host.stores import (
        InMemoryContinuationStateStore,
        InMemoryProofReceiptStore,
    )

    raw = json.loads(_rc_cross_manifest_path(work_dir).read_text(encoding="utf-8"))
    gca = SQLiteGovernedExternalWorkHostStoreBackend(Path(raw["gca_path"]))
    try:
        receipt_store = gca.bundle.receipt_store
        continuation_store = gca.bundle.continuation_store
        if isinstance(receipt_store, InMemoryProofReceiptStore):
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "in_memory_receipt_fallback",
                },
            )
            return _EXIT_ERROR
        if isinstance(continuation_store, InMemoryContinuationStateStore):
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "in_memory_continuation_fallback",
                },
            )
            return _EXIT_ERROR
        if not receipt_store.is_durable or not continuation_store.is_durable:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "store_not_durable",
                },
            )
            return _EXIT_ERROR
        receipt = receipt_store.get_receipt(raw["execution_id"])
        continuation = continuation_store.get_continuation(raw["task_id"])
        if receipt is None or continuation is None:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "missing_record",
                    "receipt_present": receipt is not None,
                    "continuation_present": continuation is not None,
                },
            )
            return _EXIT_ERROR
        if receipt.receipt_id != raw["receipt_id"]:
            _emit({"phase": "rc-cross-verify", "status": "failed", "reason": "receipt_id_mismatch"})
            return _EXIT_ERROR
        if _receipt_execution_id(receipt) != raw["receipt_execution_id"]:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "receipt_execution_binding_mismatch",
                },
            )
            return _EXIT_ERROR
        if continuation.continuation_request_id != raw["continuation_request_id"]:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "continuation_request_id_mismatch",
                },
            )
            return _EXIT_ERROR
        if str(continuation.execution_id) != raw["execution_id"]:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "continuation_execution_binding_mismatch",
                },
            )
            return _EXIT_ERROR
        if continuation_store.get_continuation("wrong-task-id") is not None:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "unexpected_continuation_for_wrong_task",
                },
            )
            return _EXIT_ERROR
        if receipt_store.get_receipt("wrong-execution-id") is not None:
            _emit(
                {
                    "phase": "rc-cross-verify",
                    "status": "failed",
                    "reason": "unexpected_receipt_for_wrong_execution",
                },
            )
            return _EXIT_ERROR
        outcome = {
            "phase": "rc-cross-verify",
            "status": "ok",
            "receipt_id": receipt.receipt_id,
            "continuation_request_id": continuation.continuation_request_id,
            "receipt_durable": receipt_store.is_durable,
            "continuation_durable": continuation_store.is_durable,
        }
    finally:
        gca.close()
    _emit(outcome)
    return _EXIT_OK


def _cmd_dur_proc_02_verify_incomplete(work_dir: Path) -> int:
    raw = json.loads(_manifest_path(work_dir).read_text(encoding="utf-8"))
    seed = _manifest_to_seed(raw)
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        assert pis.get_invocation(seed.invocation_id) is not None
        assert pis.get_outcome(seed.invocation_id) is None
        second = _create(runtime, seed.identity)
        outcome = {
            "phase": "dur-proc-02-verify-incomplete",
            "status": "ok",
            "create_calls": fake.create_calls,
            "retry_reason": second.reason,
            "host_state": second.state.value if second.state is not None else None,
        }
    finally:
        _close_pair(gca, pis)
    if outcome["create_calls"] != 0 or outcome["retry_reason"] != (
        "create_invocation_intent_without_persisted_outcome"
    ):
        outcome["status"] = "failed"
        _emit(outcome)
        return _EXIT_ERROR
    _emit(outcome)
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "dur-proc-01-seed",
            "dur-proc-01-verify",
            "dur-proc-02-committed-wait",
            "dur-proc-02-incomplete-seed",
            "dur-proc-02-verify-incomplete",
            "rc-cross-seed",
            "rc-cross-verify",
        ),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "dur-proc-01-seed":
        return _cmd_dur_proc_01_seed(args.work_dir)
    if args.command == "dur-proc-01-verify":
        return _cmd_dur_proc_01_verify(args.work_dir)
    if args.command == "dur-proc-02-committed-wait":
        return _cmd_dur_proc_02_committed_wait(args.work_dir)
    if args.command == "dur-proc-02-incomplete-seed":
        return _cmd_dur_proc_02_incomplete_seed(args.work_dir)
    if args.command == "dur-proc-02-verify-incomplete":
        return _cmd_dur_proc_02_verify_incomplete(args.work_dir)
    if args.command == "rc-cross-seed":
        return _cmd_rc_cross_seed(args.work_dir)
    if args.command == "rc-cross-verify":
        return _cmd_rc_cross_verify(args.work_dir)
    return _EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
