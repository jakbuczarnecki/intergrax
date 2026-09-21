# © Artur Czarnecki. All rights reserved.

"""UCA-6A-R — production CodeCraftOrchestratorGapSynthesisPort."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.codecraft.codegen_adapter import CodeGenerationAdapter
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.codecraft.gap_synthesis import (
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisRequest,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.codecraft.acquisition import (
    CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
    CodeCraftGapCapabilityAcquisitionStrategy,
    CodeCraftOrchestratorGapSynthesisPort,
)
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator
from intergrax.runtime.codecraft.ownership import codecraft_exec_hitl_notes
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.human.models import (
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.persistence_contract import (
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import (
    codecraft_sandbox_execution_profile,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PORT_PATH = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "codecraft"
    / "acquisition"
    / "orchestrator_gap_synthesis_port.py"
)
_TENANT = "tenant-uca6"
_TASK = "task-uca6"


def _sandbox(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id=_TENANT,
        task_id=_TASK,
        allowed_operations=frozenset(
            {
                "echo",
                "write_file",
                "read_file",
                "list_files",
                "run_python",
                "run_script",
            },
        ),
    )


def _ctx(
    sandbox: SandboxSession,
    *,
    profile: CodeCraftProfile,
    codegen: CodeGenerationAdapter | None = None,
    hitl_store: InMemoryHumanDecisionPersistence | None = None,
) -> ToolWiringContext:
    extras: dict[str, object] = {
        "codecraft_profile": profile,
        "codecraft_session_manager": CodeCraftSessionManager(),
        "codecraft_ephemeral_registry": EphemeralToolRegistryStore(),
        "effective_environment_profile": codecraft_sandbox_execution_profile(),
    }
    if codegen is not None:
        extras["codecraft_codegen_adapter"] = codegen
    return ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras=extras,
    )


def _gap_request(
    operation_id: str = "op-deterministic-uca6",
    *,
    correlation_id: str | None = "corr-gap",
    causation_id: str | None = "cause-gap",
) -> CodeCraftGapSynthesisRequest:
    need = CapabilityNeed(
        need_id="need-gap",
        kinds=(CapabilityKind.TOOL,),
        intent_summary="synthesize helper tool",
    )
    return CodeCraftGapSynthesisRequest(
        operation_id=operation_id,
        gap_id="gap-uca6",
        canonical_discovery_correlation_id="discovery-corr",
        capability_need=need,
        synthesis_goal=need.intent_summary,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


class FailingCodegenAdapter:
    def generate(
        self, *, goal: str, constraints: str = "", language: str = "python"
    ) -> str:
        del goal, constraints, language
        return "import os\nos.system('noop')\n"

    def patch(
        self,
        *,
        goal: str,
        code: str,
        diagnostics: str,
        language: str = "python",
    ) -> str:
        del goal, diagnostics, language
        return code


def test_production_port_avoids_aw_imports() -> None:
    tree = ast.parse(_PORT_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    for module in imports:
        assert not module.startswith("intergrax.autonomous_work")
        assert not module.startswith("intergrax.marketplace")
        assert not module.startswith("intergrax.runtime.execution")


def test_deterministic_craft_id_from_operation_id(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="autonomous", require_tests=False, forbidden_imports=["os"]
    )
    port = CodeCraftOrchestratorGapSynthesisPort(_ctx(sandbox, profile=profile))
    operation_id = "uca6-op-craft-binding"
    started: list[str] = []
    real_factory = CodeCraftOrchestrator

    def factory(ctx: ToolWiringContext, run_id: str) -> CodeCraftOrchestrator:
        orch = real_factory(ctx, run_id=run_id)
        original_start = orch.start

        def wrapped_start(**kwargs: object) -> object:
            started.append(str(kwargs.get("craft_id")))
            return original_start(**kwargs)

        orch.start = wrapped_start  # type: ignore[method-assign]
        return orch

    port._orchestrator_factory = factory  # type: ignore[attr-defined]
    request = _gap_request(operation_id)
    result = port.synthesize_from_gap(request)
    assert started == [operation_id]
    if result.outcome is CodeCraftGapSynthesisOutcome.SUCCEEDED:
        assert result.artifact_reference == f"codecraft:artifact:{operation_id}"


def test_terminal_success_through_real_orchestrator(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="autonomous", require_tests=False, forbidden_imports=["os"]
    )
    port = CodeCraftOrchestratorGapSynthesisPort(_ctx(sandbox, profile=profile))
    operation_id = "uca6-success-op"
    result = port.synthesize_from_gap(_gap_request(operation_id))
    assert result.outcome is CodeCraftGapSynthesisOutcome.SUCCEEDED
    assert result.artifact_reference == f"codecraft:artifact:{operation_id}"
    assert result.codecraft_operation_correlation_id == operation_id
    assert result.correlation_id == "corr-gap"
    assert result.causation_id == "cause-gap"


def test_profile_missing_is_unavailable(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    ctx = ToolWiringContext(sandbox_session=sandbox, extras={})
    port = CodeCraftOrchestratorGapSynthesisPort(ctx)
    result = port.synthesize_from_gap(_gap_request())
    assert result.outcome is CodeCraftGapSynthesisOutcome.UNAVAILABLE
    assert result.reason_detail == "codecraft_profile_missing"


def test_hitl_pending_requires_hitl(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="supervised", require_hitl_before_exec=True, require_tests=False
    )
    store = InMemoryHumanDecisionPersistence()
    port = CodeCraftOrchestratorGapSynthesisPort(
        _ctx(sandbox, profile=profile, hitl_store=store)
    )
    result = port.synthesize_from_gap(_gap_request("hitl-pending-op"))
    assert result.outcome is CodeCraftGapSynthesisOutcome.REQUIRES_HITL
    assert result.reason_detail == "hitl_pending"


def test_hitl_denied_is_blocked(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="supervised",
        isolation_tier="local",
        require_hitl_before_exec=True,
        require_tests=False,
    )
    store = InMemoryHumanDecisionPersistence()
    craft_id = "hitl-denied-op"
    run_id = mint_run_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        store.record(
            build_human_decision_record(
                task_id=_TASK,
                tenant_id=_TENANT,
                approver=local_development_approver_evidence(
                    tenant_id=_TENANT, actor_id="operator"
                ),
                verdict=HumanResponseVerdict.REJECT,
                response_text="no",
                run_id=str(run_id),
                notes=codecraft_exec_hitl_notes(craft_id),
            ),
        )
        port = CodeCraftOrchestratorGapSynthesisPort(
            _ctx(sandbox, profile=profile, hitl_store=store)
        )
        result = port.synthesize_from_gap(_gap_request(craft_id))
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftGapSynthesisOutcome.BLOCKED
    assert result.reason_detail == "hitl_denied"


def test_max_iterations_failed(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="autonomous",
        require_tests=False,
        max_iterations=2,
        forbidden_imports=["os"],
    )
    port = CodeCraftOrchestratorGapSynthesisPort(
        _ctx(sandbox, profile=profile, codegen=FailingCodegenAdapter()),
    )
    result = port.synthesize_from_gap(_gap_request("max-iter-op"))
    assert result.outcome is CodeCraftGapSynthesisOutcome.FAILED
    assert "max_iterations" in result.reason_detail or result.reason_detail


def test_promote_success_without_ephemeral_tools_fails(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    port = CodeCraftOrchestratorGapSynthesisPort(_ctx(sandbox, profile=profile))
    mock_orch = MagicMock(spec=CodeCraftOrchestrator)
    from intergrax.codecraft.contracts import CraftResult, StaticGateResult

    gate = StaticGateResult(passed=True)
    mock_orch.start.return_value = (MagicMock(craft_id="op-mock"), None)
    mock_orch.iterate.return_value = (
        None,
        CraftResult(
            craft_id="op-mock",
            success=True,
            mode="autonomous",
            static_gate=gate,
            verdict="promote",
        ),
    )
    mock_orch.promote.return_value = CraftResult(
        craft_id="op-mock",
        success=True,
        mode="autonomous",
        static_gate=gate,
        verdict="promote",
    )

    port._orchestrator_factory = lambda _ctx, _run: mock_orch  # type: ignore[attr-defined]
    result = port.synthesize_from_gap(_gap_request("op-mock"))
    assert result.outcome is CodeCraftGapSynthesisOutcome.FAILED
    assert result.reason_detail == "synthesis_artifact_missing"


def test_production_port_qualification_compatible(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from intergrax.contracts.capability_acquisition.acquisition_outcome import (
        CapabilityAcquisitionOutcome,
    )
    from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
    from intergrax.contracts.capability_catalog.discovery_completion import (
        build_discovery_completion,
    )
    from intergrax.contracts.capability_catalog.federation import (
        CapabilityCatalogFederationCompleteness,
    )
    from intergrax.contracts.capability_acquisition.acquisition_request import (
        CapabilityAcquisitionRequest,
        derive_capability_acquisition_request_id,
    )

    sandbox = _sandbox(tmp_path)
    profile = CodeCraftProfile(
        mode="autonomous", require_tests=False, forbidden_imports=["os"]
    )
    port = CodeCraftOrchestratorGapSynthesisPort(_ctx(sandbox, profile=profile))
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    created = datetime(2026, 9, 21, 8, 0, tzinfo=UTC)
    completion = build_discovery_completion(
        need_id="need-qual",
        discovery_correlation_id="disc-qual",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=created,
    )
    gap = CapabilityGap.from_discovery_completion(completion)
    need = CapabilityNeed(
        need_id=gap.need_id,
        kinds=(CapabilityKind.TOOL,),
        intent_summary="synthesize helper tool",
    )
    operation_id = derive_capability_acquisition_request_id(
        gap_id=gap.gap_id,
        request_nonce="qual-nonce",
    )
    req = CapabilityAcquisitionRequest(
        request_id=operation_id,
        request_nonce="qual-nonce",
        capability_gap=gap,
        capability_need=need,
        correlation_id="corr-qual",
        causation_id="cause-qual",
        requested_at=created,
    )
    acquisition = strategy.acquire(req)
    assert acquisition.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert acquisition.evidence is not None
    assert (
        acquisition.evidence.artifact_reference == f"codecraft:artifact:{operation_id}"
    )
    qual = CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=req.request_id,
            qualification_nonce="q-nonce",
        ),
        qualification_nonce="q-nonce",
        acquisition_request_id=req.request_id,
        gap_id=gap.gap_id,
        strategy_id=CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
        acquisition_result=acquisition,
        correlation_id=req.correlation_id,
        causation_id=req.causation_id,
        requested_at=created,
    )
    assert qual.acquisition_result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
