# © Artur Czarnecki. All rights reserved.

"""Runtime worker HTTP server — canonical RuntimeToolInvoker entry."""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    IdempotencyOperationConflictError,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
    DeclarativePolicyViolationError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.integrations.providers.key_value_cache.redis.bundle import (
    create_redis_idempotency_store,
)
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from tests.system.tools_side_effect_safety.shared.contracts import (
    PROOF_GOVERNANCE_TOOL,
    PROOF_TOOL_BAD_OUTPUT,
    PROOF_TOOL_CHARGE,
    PROOF_TOOL_CHARGE_ALT,
    PROOF_TOOL_FAIL_BEFORE,
    PROOF_TOOL_SLOW_CHARGE,
    ChargeInput,
    ChargeOutput,
    InvokeRequest,
    InvokeResponse,
)
from tests.system.tools_side_effect_safety.shared.runtime_state import ProofRuntimeState
from tests.system.tools_side_effect_safety.shared.tool_handlers import (
    BadOutputHandler,
    ChargeHandler,
    FailBeforeHandler,
    resolve_effect_service_url,
)


class LedgerQueryResponse(BaseModel):
    status: str | None


def _build_idempotency_store():
    topology = os.environ.get("PERSISTENCE_TOPOLOGY", "durable_single_host")
    if topology == "shared_multi_host":
        redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        return create_redis_idempotency_store(url=redis_url)
    db_path = os.environ.get("SQLITE_DB_PATH", "/data/idempotency.db")
    from pathlib import Path

    from intergrax.integrations.providers.relational_store.sqlite.bundle import (
        create_sqlite_idempotency_store,
    )

    return create_sqlite_idempotency_store(db_path=Path(db_path))


def _register_tools(registry: ToolRegistry, worker_source: str) -> None:
    effect_url = resolve_effect_service_url()
    charge = ChargeHandler(effect_service_url=effect_url, worker_source=worker_source)
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_TOOL_CHARGE,
            name=PROOF_TOOL_CHARGE,
            description="Proof charge tool",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={ValueError: RuntimeErrorCode.VALIDATION_ERROR},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=5_000,
        ),
        handler=charge,
    )
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_TOOL_CHARGE_ALT,
            name=PROOF_TOOL_CHARGE_ALT,
            description="Alternate proof charge tool",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=5_000,
        ),
        handler=ChargeHandler(effect_service_url=effect_url, worker_source=worker_source),
    )
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_TOOL_SLOW_CHARGE,
            name=PROOF_TOOL_SLOW_CHARGE,
            description="Slow charge tool for timeout proof",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={ValueError: RuntimeErrorCode.VALIDATION_ERROR},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=500,
        ),
        handler=charge,
    )
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_TOOL_FAIL_BEFORE,
            name=PROOF_TOOL_FAIL_BEFORE,
            description="Fails before external effect",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=5_000,
        ),
        handler=FailBeforeHandler(),
    )
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_TOOL_BAD_OUTPUT,
            name=PROOF_TOOL_BAD_OUTPUT,
            description="Commits then fails output validation",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={ValueError: RuntimeErrorCode.VALIDATION_ERROR},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=5_000,
        ),
        handler=BadOutputHandler(effect_service_url=effect_url, worker_source=worker_source),
    )
    registry.register(
        contract=ToolContract(
            tool_id=PROOF_GOVERNANCE_TOOL,
            name=PROOF_GOVERNANCE_TOOL,
            description="Governance proof tool",
            input_schema=ChargeInput,
            output_schema=ChargeOutput,
            error_mapping={},
            side_effects=True,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            timeout_ms=5_000,
        ),
        handler=charge,
    )


def _policy_bundle(*, action: str, tool_id: str, rule_id: str) -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"proof.{action}")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": rule_id,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": tool_id,
                "action": action,
            }
        ],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _hitl_grant(*, bundle: object, key: str, run_id: str, task_id: str, tool_id: str, rule_id: str) -> DeclarativeHitlApprovalGrant:
    provenance = bundle.declarative_policy_runtime.provenance.rules_digest_sha256
    return DeclarativeHitlApprovalGrant(
        grant_id=f"grant-{key}",
        invocation_scope_id=f"scope-{key}",
        task_id=task_id,
        run_id=run_id,
        step_id="step1",
        tool_id=tool_id,
        agent_id="proof-agent",
        idempotency_key=key,
        matched_rule_ids=(rule_id,),
        human_request_id=f"hr-{key}",
        policy_provenance_digest=provenance,
        pause_id=f"pause-{key}",
        approved_at="2026-08-30T00:00:00+00:00",
    )


class RuntimeWorkerApp:
    def __init__(self) -> None:
        self.worker_source = os.environ.get("WORKER_SOURCE", "runtime-single")
        self.store = _build_idempotency_store()
        self.registry = ToolRegistry()
        _register_tools(self.registry, self.worker_source)
        coordinator = IdempotencyPreEffectCoordinator(
            idempotency_store=self.store,
            lease_seconds=int(os.environ.get("IDEMPOTENCY_LEASE_SECONDS", "300")),
        )
        self.invoker = RuntimeToolInvoker(
            registry=self.registry,
            executor=RegistryToolExecutor(self.registry),
            pre_effect_coordinator=coordinator,
        )
        self.app = FastAPI(title=f"Proof Runtime Worker ({self.worker_source})")
        self.app.get("/health")(self.health)
        self.app.post("/invoke", response_model=InvokeResponse)(self.invoke)
        self.app.get("/ledger/{tenant_id}/{key}", response_model=LedgerQueryResponse)(self.ledger)

    def health(self) -> dict[str, str]:
        return {"status": "ok", "worker": self.worker_source}

    def ledger(self, tenant_id: str, key: str) -> LedgerQueryResponse:
        status = self.store.get_status(tenant_id, key)
        return LedgerQueryResponse(status=status.value if status is not None else None)

    def invoke(self, body: InvokeRequest) -> InvokeResponse:
        policy_bundle = None
        hitl_grant = None
        tool_id = body.tool_id
        if body.governance_action:
            policy_bundle = _policy_bundle(
                action=body.governance_action,
                tool_id=PROOF_GOVERNANCE_TOOL,
                rule_id=body.governance_rule_id,
            )
            tool_id = PROOF_GOVERNANCE_TOOL
        elif body.require_hitl:
            policy_bundle = _policy_bundle(
                action="require_hitl",
                tool_id=PROOF_GOVERNANCE_TOOL,
                rule_id=body.governance_rule_id,
            )
            tool_id = PROOF_GOVERNANCE_TOOL
            if body.hitl_resume:
                hitl_grant = _hitl_grant(
                    bundle=policy_bundle,
                    key=body.idempotency_key,
                    run_id=body.run_id,
                    task_id=body.run_id,
                    tool_id=PROOF_GOVERNANCE_TOOL,
                    rule_id=body.governance_rule_id,
                )

        state = ProofRuntimeState(
            tenant_id=body.tenant_id,
            run_id=body.run_id,
            task_id=body.run_id,
            policy_bundle=policy_bundle,
            declarative_hitl_grant=hitl_grant,
        )
        request = ToolExecutionRequest(
            run_id=body.run_id,
            step_id=body.step_id,
            tool_id=tool_id,
            input=ChargeInput(
                business_operation_id=body.business_operation_id,
                amount=body.amount,
                proof_mode=body.proof_mode,
                proof_delay_ms=body.proof_delay_ms,
                http_timeout_s=120.0,
            ),
            idempotency_key=body.idempotency_key,
        )
        if body.require_hitl and body.hitl_resume and hitl_grant is not None:
            request = maybe_assign_declarative_hitl_scope(
                request,
                state=state,
                assignment_state=DeclarativeHitlScopeAssignmentState(),
                unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
                request_index=0,
            )

        try:
            result = self.invoker.invoke(state=state, agent_id="proof-agent", request=request)
        except IdempotencyOperationConflictError as exc:
            return InvokeResponse(
                success=False,
                blocked=True,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )
        except InvocationUncertaintyError as exc:
            return InvokeResponse(
                success=False,
                uncertain=True,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )
        except ActiveInvocationClaimError as exc:
            return InvokeResponse(
                success=False,
                blocked=True,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )
        except DeclarativePolicyViolationError as exc:
            return InvokeResponse(
                success=False,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )
        except DeclarativePolicyHitlRequiredError as exc:
            return InvokeResponse(
                success=False,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )
        except Exception as exc:
            return InvokeResponse(
                success=False,
                error_type=type(exc).__name__,
                ledger_status=self._ledger_value(body.tenant_id, body.idempotency_key),
            )

        ledger_status = self._ledger_value(body.tenant_id, body.idempotency_key)
        return InvokeResponse(
            success=result.success,
            replay=False,
            uncertain=ledger_status == InvocationStatus.UNCERTAIN.value,
            error_code=result.error.error_code.value if result.error else None,
            ledger_status=ledger_status,
            output=result.output.model_dump() if result.output is not None else None,
        )

    def _ledger_value(self, tenant_id: str, key: str) -> str | None:
        status = self.store.get_status(tenant_id, key)
        return status.value if status is not None else None


def create_app() -> FastAPI:
    return RuntimeWorkerApp().app


def main() -> None:
    host = os.environ.get("RUNTIME_HOST", "0.0.0.0")
    port = int(os.environ.get("RUNTIME_PORT", "8090"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
