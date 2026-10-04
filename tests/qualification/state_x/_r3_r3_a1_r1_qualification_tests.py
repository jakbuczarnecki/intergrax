# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R3-A1-R1 — CodeCraft authority shortcut removal."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.tools.providers.codecraft.service import codecraft_iterate
from intergrax.runtime.codecraft.ownership import (
    CodeCraftSessionOwnership,
    codecraft_exec_hitl_notes,
    resolve_codecraft_exec_authorization,
)
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    WiringCodeCraftBoundCapabilityExecution,
)
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceValidationError
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftIterateToolInput,
    CodeCraftRunToolInput,
)
from intergrax.tools.providers.codecraft.service import codecraft_run
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import codecraft_sandbox_execution_profile
from tests.qualification.state_x._r3_r3_support import (
    SHARED_TASK,
    TENANT_A,
    TENANT_B,
    human_decision_store_factories,
    sample_record,
)
from tests.unit.autonomous_work.uca6c_r5_tool_runtime_fixtures import build_sandbox_session
from tests.unit.runtime.human.test_gr5_r3_r1_canonical_first_atomic_projection import (
    _ATTEMPT,
    _EXECUTION,
    _HR,
    _PAUSE,
    _RUN,
    _waiting_setup,
)
from tests.unit.runtime.codecraft.test_identity_governance import (
    RUN_A,
    TASK_X,
    TENANT_A as GOV_TENANT_A,
    _approve,
    _bind_run,
    _ctx,
    _open_session,
    _reset_run,
    _sandbox,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWNERSHIP = _REPO_ROOT / "intergrax/runtime/codecraft/ownership.py"
_WIRING = _REPO_ROOT / "intergrax/runtime/codecraft/wiring_bound_capability_execution.py"
_CODECRAFT_ROOT = _REPO_ROOT / "intergrax/runtime/codecraft"
_PROVIDER_ROOT = _REPO_ROOT / "intergrax/tools/providers/codecraft"
_INTAKE = _REPO_ROOT / "intergrax/runtime/nexus/orchestration/intake_runner.py"


def test_r3_r3_a1_r1_q01_no_upstream_canonical_hitl_satisfied_in_production() -> None:
    for path in _REPO_ROOT.joinpath("intergrax").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "upstream_canonical_hitl_satisfied" not in text, path.as_posix()


def _supervised_wiring_port(
    tmp_path: Path,
    *,
    hitl_store: InMemoryHumanDecisionPersistence | SQLiteHumanDecisionStore | None = None,
    run_id: str | None = None,
) -> tuple[WiringCodeCraftBoundCapabilityExecution, _RecordingCatalogInvoker, str, str]:
    sandbox = build_sandbox_session(
        tmp_path,
        tenant_id=TENANT_A,
        task_id=SHARED_TASK,
    )
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    craft_id = "craft-r1-supervised"
    effective_run_id = run_id if run_id is not None else str(mint_run_id())
    ownership = CodeCraftSessionOwnership(
        tenant_id=TENANT_A,
        task_id=SHARED_TASK,
        run_id=effective_run_id,
    )
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="supervised",
        craft_id=craft_id,
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print('blocked')\n"}),
        ownership,
    )
    registry.for_craft(craft_id).register("ephemeral.r1.supervised")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="supervised",
                require_hitl_before_exec=True,
                require_tests=False,
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    recording = _RecordingCatalogInvoker()
    port = WiringCodeCraftBoundCapabilityExecution(ctx, catalog_tool_invoker=recording)
    return port, recording, craft_id, effective_run_id


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = "qualification-r1"
    calls: int = 0

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]:
        self.calls += 1
        _ = request
        return ToolExecutionResult.fail("injected", "injected")


def _persist_approve(
    store: InMemoryHumanDecisionPersistence | SQLiteHumanDecisionStore,
    *,
    craft_id: str,
    run_id: str,
    verdict: HumanResponseVerdict = HumanResponseVerdict.APPROVE,
) -> None:
    record = build_human_decision_record(
        task_id=SHARED_TASK,
        tenant_id=TENANT_A,
        approver=local_development_approver_evidence(tenant_id=TENANT_A, actor_id="op"),
        verdict=verdict,
        response_text="evidence",
        run_id=run_id,
        notes=codecraft_exec_hitl_notes(craft_id),
    ).model_copy(update={"decision_id": f"hdec-r1-{verdict.value}-{craft_id}"})
    store.record(record)


def test_r3_r3_a1_r1_q02_supervised_wiring_bound_fails_closed(tmp_path: Path) -> None:
    port, invoker, craft_id, run_id = _supervised_wiring_port(tmp_path)
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=TENANT_A,
                task_id=SHARED_TASK,
                run_id=run_id,
                execution_id=execution_id,
                execution_request_id="r1-q02",
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED
    assert "hitl" in (result.reason_detail or "")
    assert invoker.calls == 0
    assert port.runtime_execution_calls == 0


def test_r3_r3_a1_r1_q03_supervised_wiring_bound_does_not_invoke_catalog(tmp_path: Path) -> None:
    port, invoker, craft_id, run_id = _supervised_wiring_port(tmp_path)
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=TENANT_A,
                task_id=SHARED_TASK,
                run_id=run_id,
                execution_id=execution_id,
                execution_request_id="r1-q03",
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert invoker.calls == 0


@pytest.mark.parametrize(
    "verdict",
    [
        HumanResponseVerdict.APPROVE,
        HumanResponseVerdict.REJECT,
        HumanResponseVerdict.ESCALATE,
    ],
)
def test_r3_r3_a1_r1_q04_q05_persisted_verdict_cannot_authorize_wiring_bound(
    tmp_path: Path,
    verdict: HumanResponseVerdict,
) -> None:
    store = InMemoryHumanDecisionPersistence()
    port, invoker, craft_id, run_id = _supervised_wiring_port(tmp_path, hitl_store=store)
    _persist_approve(store, craft_id=craft_id, run_id=run_id, verdict=verdict)
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=TENANT_A,
                task_id=SHARED_TASK,
                run_id=run_id,
                execution_id=execution_id,
                execution_request_id=f"r1-q04-{verdict.value}",
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED
    assert invoker.calls == 0
    assert port.runtime_execution_calls == 0


def test_r3_r3_a1_r1_q06_orchestrator_supervised_standalone_fail_closed(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=GOV_TENANT_A, task_id=TASK_X), profile=profile, hitl_store=store)
    token = _bind_run()
    try:
        craft_id = _open_session(ctx, tenant_id=GOV_TENANT_A, task_id=TASK_X)
        _approve(store, tenant_id=GOV_TENANT_A, task_id=TASK_X, craft_id=craft_id, run_id=str(RUN_A))
        with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
            out = codecraft_iterate(
                ctx,
                CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=GOV_TENANT_A, task_id=TASK_X),
            )
            mocked_exec.assert_not_called()
    finally:
        _reset_run(token)
    assert out.result.error == "hitl_pending"


def test_r3_r3_a1_r1_q07_codecraft_run_supervised_standalone_fail_closed(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=GOV_TENANT_A, task_id=TASK_X), profile=profile, hitl_store=store)
    with patch("intergrax.tools.providers.codecraft.service.code_exec") as mocked_exec:
        out = codecraft_run(
            ctx,
            CodeCraftRunToolInput(
                code="print('nope')\n",
                tenant_id=GOV_TENANT_A,
                task_id=TASK_X,
            ),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "hitl_pending"


def test_r3_r3_a1_r1_q08_autonomous_wiring_bound_reaches_catalog_invoker(tmp_path: Path) -> None:
    sandbox = build_sandbox_session(tmp_path, tenant_id=TENANT_A, task_id=SHARED_TASK)
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    craft_id = "craft-r1-autonomous"
    ownership = CodeCraftSessionOwnership(tenant_id=TENANT_A, task_id=SHARED_TASK)
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    sessions.save_owned(session.model_copy(update={"code": "print(1)\n"}), ownership)
    registry.for_craft(craft_id).register("ephemeral.r1.auto")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous",
                require_hitl_before_exec=False,
                require_tests=False,
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    recording = _RecordingCatalogInvoker()
    port = WiringCodeCraftBoundCapabilityExecution(ctx, catalog_tool_invoker=recording)
    execution_id = mint_execution_id()
    run_id = mint_run_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=TENANT_A,
                task_id=SHARED_TASK,
                run_id=None,
                execution_id=execution_id,
                execution_request_id="r1-q08",
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert recording.calls == 1
    assert port.runtime_execution_calls == 1


def test_r3_r3_a1_r1_q09_wiring_static_gate_catalog_invoker_boundary() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    assert "self._catalog_tool_invoker.invoke" in source
    assert "PolicyAction.ALLOW" not in source
    assert "MeaningfulSideEffectAuthorization" not in source


def test_r3_r3_a1_r1_q10_resolve_has_no_human_decision_reads() -> None:
    tree = ast.parse(_OWNERSHIP.read_text(encoding="utf-8"))
    resolve_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_codecraft_exec_authorization"
    )
    source_segment = ast.get_source_segment(_OWNERSHIP.read_text(encoding="utf-8"), resolve_fn) or ""
    lowered = "\n".join(
        line for line in source_segment.splitlines() if not line.strip().startswith('"""')
    )
    assert "HumanDecisionRecord" not in lowered
    assert "HumanResponseVerdict" not in lowered
    assert "human_decision_store" not in lowered
    assert "upstream_canonical_hitl_satisfied" not in lowered


def test_r3_r3_a1_r1_q11_codecraft_exec_hitl_notes_not_authority() -> None:
    text = _OWNERSHIP.read_text(encoding="utf-8")
    assert "codecraft_exec_hitl_notes" in text
    assert "if notes" not in text
    assert "notes.startswith" not in text


def test_r3_r3_a1_r1_q12_no_session_hitl_approved_bypass() -> None:
    orchestrator = (_REPO_ROOT / "intergrax/runtime/codecraft/orchestrator.py").read_text(
        encoding="utf-8"
    )
    service = (_REPO_ROOT / "intergrax/tools/providers/codecraft/service.py").read_text(
        encoding="utf-8"
    )
    ownership = _OWNERSHIP.read_text(encoding="utf-8")
    assert "if session.hitl_approved" not in orchestrator
    assert "if params.hitl_approved" not in service
    assert "hitl_approved" not in ownership


def test_r3_r3_a1_r1_q13_codecraft_execution_entrypoints_classified() -> None:
    required = {
        "intergrax/runtime/codecraft/orchestrator.py",
        "intergrax/tools/providers/codecraft/service.py",
        "intergrax/runtime/codecraft/wiring_bound_capability_execution.py",
    }
    hits: set[str] = set()
    for path in (_CODECRAFT_ROOT).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "resolve_codecraft_exec_authorization" in text:
            hits.add(path.relative_to(_REPO_ROOT).as_posix())
    for path in (_PROVIDER_ROOT).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "resolve_codecraft_exec_authorization" in text:
            hits.add(path.relative_to(_REPO_ROOT).as_posix())
    assert required <= hits


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_a1_r1_q14_persistence_validation_regression(
    store_factory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    valid = sample_record(decision_id="hdec-r1-q14-valid")
    invalid = valid.model_copy(
        update={"approver": local_development_approver_evidence(tenant_id=TENANT_B, actor_id="x")}
    )
    with pytest.raises(HumanDecisionPersistenceValidationError):
        store.record(invalid)
    assert store.get_decision("hdec-r1-q14-valid", TENANT_A) is None


def test_r3_r3_a1_r1_q15_canonical_first_persistence_ordering() -> None:
    text = _INTAKE.read_text(encoding="utf-8")
    approve_idx = text.index("if verdict == HumanResponseVerdict.APPROVE:")
    segment = text[approve_idx : approve_idx + 8000]
    resolve_idx = segment.index("resolve_human_response_and_apply_canonical")
    persist_idx = segment.index("self.hitl.persist_human_decision")
    assert resolve_idx < persist_idx


def test_r3_r3_a1_r1_q16_persistence_failure_observable() -> None:
    task, port, _waiting = _waiting_setup()
    store = InMemoryHumanDecisionPersistence()
    approver = local_development_approver_evidence(tenant_id=task.tenant_id, actor_id="op")
    from intergrax.runtime.nexus.orchestration.human_response import HumanDecisionPersistenceError
    from intergrax.runtime.human.pause import HumanPauseCoordinator

    HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=port,
        pause_id=_PAUSE,
        human_request_id=_HR,
        run_id=str(_RUN),
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )

    def boom(record):  # noqa: ANN001
        raise HumanDecisionPersistenceValidationError(
            "forced",
            decision_id=record.decision_id,
            tenant_id=record.tenant_id,
        )

    with patch.object(store, "record", side_effect=boom):
        with pytest.raises(HumanDecisionPersistenceValidationError):
            persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)


def test_r3_r3_a1_r1_q17_uca_comparison_recorded_placeholder() -> None:
    """Evidence for R1-Q17 is captured in the session report (START_HEAD vs FINAL pytest)."""
    assert Path("tests/unit/autonomous_work/test_uca6c_r5_canonical_tool_runtime.py").is_file()


def test_r3_r3_a1_r1_q18_targeted_pyright_scope_files_exist() -> None:
    assert _OWNERSHIP.is_file()
    assert _WIRING.is_file()


def test_r3_r3_a1_r1_q19_a1_regression_module_importable() -> None:
    import tests.qualification.state_x._r3_r3_a1_qualification_tests as a1  # noqa: F401


def test_r3_r3_a1_r1_q20_r3_qualification_module_importable() -> None:
    import tests.qualification.state_x._r3_r3_qualification_tests as r3  # noqa: F401


def test_r3_r3_a1_r1_static_no_caller_bool_authorize_branch() -> None:
    tree = ast.parse(_OWNERSHIP.read_text(encoding="utf-8"))
    resolve_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_codecraft_exec_authorization"
    )
    param_names = [arg.arg for arg in resolve_fn.args.args] + [
        arg.arg for arg in resolve_fn.args.kwonlyargs
    ]
    assert "upstream_canonical_hitl_satisfied" not in param_names
    allow_returns = 0
    deny_returns = 0
    for node in ast.walk(resolve_fn):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "CodeCraftExecAuthorization"
        ):
            for kw in node.value.keywords:
                if kw.arg == "authorized" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is True:
                        allow_returns += 1
                    if kw.value.value is False:
                        deny_returns += 1
    assert allow_returns == 1
    assert deny_returns >= 1
