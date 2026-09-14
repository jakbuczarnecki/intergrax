# © Artur Czarnecki. All rights reserved.

"""ACP execution budget — ExecutionBudgetLedgerFactory injection (HARDENING-9.2A)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.authoring.acp_run import run_acp_session
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.contracts.execution_identity_authority import MintedExecutionIdentity
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    ExecutionBudgetLedgerFactory,
    create_execution_budget_ledger,
    fixed_execution_budget_ledger_factory,
)
from intergrax.runtime.execution.identity_authority import RootTaskIdentity
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from tests.unit.agents.conftest import make_acp_host_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ACP_RUN = _REPO_ROOT / "intergrax" / "agents" / "authoring" / "acp_run.py"

_ACP_MINT_PATCH = (
    "intergrax.agents.authoring.acp_run.default_execution_identity_authority.mint_execution_identity"
)


@dataclass
class _RecordingLedgerFactory:
    """Test factory that records create_ledger identity context."""

    inner: ExecutionBudgetLedgerFactory
    calls: list[tuple[str | None, str | None, str | None]] = field(default_factory=list)

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        self.calls.append(
            (
                tenant_id,
                str(run_id) if run_id is not None else None,
                str(attempt_id) if attempt_id is not None else None,
            ),
        )
        return self.inner.create_ledger(
            run_budget,
            tenant_id=tenant_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )


class _BudgetProbeAgent(IntergraxAgent):
    contract_id = "budget-probe"
    capabilities = ("demo.budget",)
    agent_name = "Budget Probe"
    agent_description = "ACP budget factory probe"
    risk_level = AgentRiskLevel.LOW
    max_steps = 2

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        _ = step_ctx
        active = peek_active_execution_budget()
        assert active is not None
        return StepOutcome.complete(
            output={"ledger": active.ledger is sentinel_ledger},
            terminal_reason=TerminalReason.GOAL_MET,
        )


sentinel_ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=7))


def test_acp_run_does_not_construct_concrete_execution_budget_ledger() -> None:
    tree = ast.parse(_ACP_RUN.read_text(encoding="utf-8"), filename=str(_ACP_RUN))
    forbidden = frozenset({"create_execution_budget_ledger"})
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in forbidden:
            violations.append(f"{_ACP_RUN.name}:{node.lineno}")
    assert violations == []


def _minted_identity(root: RootTaskIdentity) -> MintedExecutionIdentity:
    return MintedExecutionIdentity(
        run_id=root.run_id,
        attempt_id=root.attempt_id,
        execution_id=root.execution_id,
    )


@pytest.mark.asyncio
async def test_acp_session_uses_injected_execution_budget_ledger_factory() -> None:
    minted_run = mint_run_id()
    minted_attempt = mint_attempt_id()
    minted_execution = mint_execution_id()
    root = RootTaskIdentity(
        run_id=minted_run,
        attempt_id=minted_attempt,
        execution_id=minted_execution,
    )
    recording = _RecordingLedgerFactory(
        inner=fixed_execution_budget_ledger_factory(sentinel_ledger),
    )
    host_ctx = make_acp_host_context(
        ApplicationEnvironmentProfile.lab_defaults(profile_id="budget.factory"),
        execution_budget_ledger_factory=recording,
    )
    request = AgentRunRequest(
        input="probe",
        identity=RequestIdentity(tenant_id="tenant-budget", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    agent = _BudgetProbeAgent()
    with (
        patch(
            _ACP_MINT_PATCH,
            return_value=_minted_identity(root),
        ),
        patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"),
        patch(
            "intergrax.runtime.wiring.llm_resolver.resolve_llm_adapter",
            return_value=FakeLLMAdapter(),
        ),
    ):
        result = await run_acp_session(agent, request)

    assert result.output == {"ledger": True}
    assert len(recording.calls) == 1
    tenant_id, run_id, attempt_id = recording.calls[0]
    assert tenant_id == "tenant-budget"
    assert run_id == str(minted_run)
    assert attempt_id == str(minted_attempt)
