# © Artur Czarnecki. All rights reserved.

"""OD-R2-03 — restart safety, cross-tenant denial, subprocess durability (IMPL-01-R2)."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    open_sqlite_governed_external_work_host_store_bundle_at,
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
    _create_meta,
    _policy_bundle,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.orchestrator import GovernedExternalWorkOrchestrator
from governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from governed_contractor_application.host.sanctioned_external_work_operation_port import (
    GovernedSanctionedExternalWorkOperationPort,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkAcceptQuoteRequest,
    SanctionedExternalWorkMutationOperation,
    SanctionedExternalWorkOperationRequest,
)
from tests.unit.applications.governed_contractor.impl_01_r2_od_r2_03_restart_worker import (
    _MANIFEST,
    _RC_CROSS_MANIFEST,
    _READY_FLAG,
)

_WORKER_MODULE = (
    "tests.unit.applications.governed_contractor.impl_01_r2_od_r2_03_restart_worker"
)
from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2 import (
    _Identity,
    _Seed,
    _T0,
    _close_pair,
    _create,
    _open_pair,
    _reopen,
)
from tests.unit.contracts.test_external_work import _acceptance
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SUBPROCESS_TIMEOUT_SEC = 45

_TENANT_B = "od-r2-03-tenant-b"
_PRINCIPAL_B = "od-r2-03-principal-b"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _subprocess_env() -> dict[str, str]:
    root = _repo_root()
    paths = [str(root), str(root / "agents"), str(root / "applications")]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([existing] if existing else []),
    )
    return env


def _worker_cmd(command: str, work_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        _WORKER_MODULE,
        command,
        "--work-dir",
        str(work_dir),
    ]


def _run_worker(
    command: str,
    work_dir: Path,
    *,
    timeout_sec: int = _SUBPROCESS_TIMEOUT_SEC,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _worker_cmd(command, work_dir),
        cwd=_repo_root(),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def _runtime_for_tenant(
    fake: DeterministicExternalWorkFake,
    identity: object,
    gca: object,
    pis: object,
    *,
    tenant_id: str,
    principal_id: str,
):
    bundle = gca.bundle
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        principal_id=principal_id,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    runtime = build_governed_external_work_production_runtime(
        fake,
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        principal_id=principal_id,
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
    )
    assert isinstance(runtime.orchestrator, GovernedExternalWorkOrchestrator)
    gate = runtime.adapter._invocation_dispatch
    assert isinstance(gate, GovernedProviderInvocationDispatchGate)
    return runtime


def test_dur_proc_01_fresh_python_process_graceful_restart(tmp_path: Path) -> None:
    work_dir = tmp_path / "dur-proc-01"
    seed_proc = _run_worker("dur-proc-01-seed", work_dir)
    assert seed_proc.returncode == 0, seed_proc.stderr + seed_proc.stdout
    assert (work_dir / _MANIFEST).is_file()
    verify_proc = _run_worker("dur-proc-01-verify", work_dir)
    assert verify_proc.returncode == 0, verify_proc.stderr + verify_proc.stdout
    tail = verify_proc.stdout.strip().splitlines()[-1]
    payload = json.loads(tail)
    assert payload["phase"] == "dur-proc-01-verify"
    assert payload["create_calls_after_retry"] == 0


def test_dur_proc_02_committed_state_survives_abrupt_child_termination(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "dur-proc-02-committed"
    child = subprocess.Popen(
        _worker_cmd("dur-proc-02-committed-wait", work_dir),
        cwd=_repo_root(),
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = _SUBPROCESS_TIMEOUT_SEC
        import time

        start = time.monotonic()
        while not (work_dir / _READY_FLAG).is_file():
            if time.monotonic() - start > deadline:
                child.kill()
                raise AssertionError("child never wrote committed ready flag")
            time.sleep(0.05)
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
    verify_proc = _run_worker("dur-proc-01-verify", work_dir)
    assert verify_proc.returncode == 0, verify_proc.stderr + verify_proc.stdout


def test_dur_proc_02_incomplete_ambiguity_fail_closed_across_process_boundary(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "dur-proc-02-incomplete"
    seed_proc = _run_worker("dur-proc-02-incomplete-seed", work_dir)
    assert seed_proc.returncode == 0, seed_proc.stderr + seed_proc.stdout
    verify_proc = _run_worker("dur-proc-02-verify-incomplete", work_dir)
    assert verify_proc.returncode == 0, verify_proc.stderr + verify_proc.stdout
    payload = json.loads(verify_proc.stdout.strip().splitlines()[-1])
    assert payload["create_calls"] == 0
    assert payload["retry_reason"] == "create_invocation_intent_without_persisted_outcome"


def test_dur_ten_01_cross_tenant_recovery_denied_through_sanctioned_boundary(
    tmp_path: Path,
) -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    identity = _Identity(task_id, run_id, attempt_id, execution_id)
    gca_path = tmp_path / "ten-gca.sqlite3"
    pis_path = tmp_path / "ten-pis.sqlite3"
    fake_a = DeterministicExternalWorkFake()
    gca, pis = _open_pair(gca_path, pis_path)
    runtime_a = _runtime_for_tenant(
        fake_a,
        identity,
        gca,
        pis,
        tenant_id=_TENANT,
        principal_id=_PRINCIPAL,
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
        governance_tenant_id=_TENANT,
        governance_workspace_id=_WORKSPACE,
        governance_principal_id=_PRINCIPAL,
    ):
        step = _create(runtime_a, identity)
    assert fake_a.create_calls == 1
    assert step.governed_result is not None
    invocation_id = step.governed_result.provider_invocation.invocation_id
    _close_pair(gca, pis)

    seed = _Seed(gca_path, pis_path, identity, invocation_id)
    fake_b = DeterministicExternalWorkFake()
    gca_b, pis_b, runtime_b = _reopen(seed, fake_b)
    try:
        follow_on_b = runtime_b.orchestrator.resolve_create_adapter_result_for_follow_on(
            identity.execution_key,
            tenant_id=_TENANT_B,
            principal_id=_PRINCIPAL_B,
        )
        assert follow_on_b is None
        port = GovernedSanctionedExternalWorkOperationPort(runtime_b.orchestrator)
        with patch.object(runtime_b.orchestrator, "accept") as accept_mutator:
            denied = port.execute(
                SanctionedExternalWorkOperationRequest(
                    operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
                    accept_quote=SanctionedExternalWorkAcceptQuoteRequest(
                        execution_id=identity.execution_key,
                        principal_id=_PRINCIPAL_B,
                        tenant_id=_TENANT_B,
                        acceptance=_acceptance(),
                        acceptance_idempotency_key="ten-deny-accept",
                    ),
                ),
            )
        assert denied.admitted is False
        assert denied.reason == "prior_create_snapshot_missing"
        assert denied.provider_mutation_dispatched is False
        accept_mutator.assert_not_called()
        assert _TENANT not in (denied.domain_summary.reason or "")
        runtime_b_cross = _runtime_for_tenant(
            fake_b,
            identity,
            gca_b,
            pis_b,
            tenant_id=_TENANT_B,
            principal_id=_PRINCIPAL_B,
        )
        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            task_id=task_id,
            governance_tenant_id=_TENANT_B,
            governance_workspace_id=_WORKSPACE,
            governance_principal_id=_PRINCIPAL_B,
        ):
            retry = _create(runtime_b_cross, identity)
        assert fake_b.create_calls == 0
        assert retry.reason == "create_redispatch_prohibited"
        ger_a = gca_b.bundle.execution_store.get_result(identity.execution_key)
        assert ger_a is not None
        assert str(ger_a.tenant_id) == _TENANT
    finally:
        _close_pair(gca_b, pis_b)


def test_rc_cross_process_receipt_and_continuation_survive_graceful_restart(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "rc-cross"
    seed_proc = _run_worker("rc-cross-seed", work_dir)
    assert seed_proc.returncode == 0, seed_proc.stderr + seed_proc.stdout
    assert (work_dir / _RC_CROSS_MANIFEST).is_file()
    verify_proc = _run_worker("rc-cross-verify", work_dir)
    assert verify_proc.returncode == 0, verify_proc.stderr + verify_proc.stdout
    payload = json.loads(verify_proc.stdout.strip().splitlines()[-1])
    assert payload["phase"] == "rc-cross-verify"
    assert payload["status"] == "ok"
    assert payload["receipt_durable"] is True
    assert payload["continuation_durable"] is True


def test_rc_neg_missing_receipt_returns_none(tmp_path: Path) -> None:
    work_dir = tmp_path / "rc-missing"
    assert _run_worker("rc-cross-seed", work_dir).returncode == 0
    manifest = json.loads((work_dir / _RC_CROSS_MANIFEST).read_text(encoding="utf-8"))
    conn = sqlite3.connect(manifest["gca_path"])
    try:
        conn.execute(
            "DELETE FROM gca_proof_receipts WHERE execution_id = ?",
            (manifest["execution_id"],),
        )
        conn.commit()
    finally:
        conn.close()
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        Path(manifest["gca_path"]),
    )
    try:
        assert backend.bundle.receipt_store.get_receipt(manifest["execution_id"]) is None
    finally:
        backend.close()


def test_rc_neg_corrupt_receipt_fail_closed(tmp_path: Path) -> None:
    work_dir = tmp_path / "rc-corrupt-receipt"
    assert _run_worker("rc-cross-seed", work_dir).returncode == 0
    manifest = json.loads((work_dir / _RC_CROSS_MANIFEST).read_text(encoding="utf-8"))
    conn = sqlite3.connect(manifest["gca_path"])
    try:
        conn.execute(
            "UPDATE gca_proof_receipts SET payload = ? WHERE execution_id = ?",
            ("not-valid-json", manifest["execution_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        Path(manifest["gca_path"]),
    )
    try:
        with pytest.raises(ValueError, match="corrupted_receipt_artifact"):
            backend.bundle.receipt_store.get_receipt(manifest["execution_id"])
    finally:
        backend.close()


def test_rc_neg_corrupt_continuation_fail_closed(tmp_path: Path) -> None:
    work_dir = tmp_path / "rc-corrupt-continuation"
    assert _run_worker("rc-cross-seed", work_dir).returncode == 0
    manifest = json.loads((work_dir / _RC_CROSS_MANIFEST).read_text(encoding="utf-8"))
    conn = sqlite3.connect(manifest["gca_path"])
    try:
        conn.execute(
            "UPDATE gca_continuations SET payload = ? WHERE task_id = ?",
            ("not-valid-json", manifest["task_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        Path(manifest["gca_path"]),
    )
    try:
        with pytest.raises(ValueError, match="corrupted_continuation_artifact"):
            backend.bundle.continuation_store.get_continuation(manifest["task_id"])
    finally:
        backend.close()


def test_dur_cons_01_q2_regression_ger_without_f02_remains_fail_closed(
    tmp_path: Path,
) -> None:
    from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2 import (
        test_q2_n03_ger_without_f02_denies_follow_on_and_create_redispatch,
    )

    test_q2_n03_ger_without_f02_denies_follow_on_and_create_redispatch(tmp_path)
