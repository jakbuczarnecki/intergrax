# © Artur Czarnecki. All rights reserved.

"""OD-R2-03 subprocess worker — fresh-process and abrupt-termination qualification phases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.governed_execution_result import GovernedExecutionResultV2

from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2 import (
    _Seed,
    _close_pair,
    _create,
    _expected_invocation_id,
    _inject_controlled_gca_execution_loss,
    _inject_controlled_pis_outcome_loss,
    _open_pair,
    _reopen,
    _seed,
)

_EXIT_OK = 0
_EXIT_ERROR = 1
_MANIFEST = "od_r2_03_manifest.json"
_READY_FLAG = "od_r2_03_committed_ready.flag"


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
    return _EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
