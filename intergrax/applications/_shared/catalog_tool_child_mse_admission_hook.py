# © Artur Czarnecki. All rights reserved.

"""MSE admission hook for catalog child invoke — runs under child-bound execution identity."""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp.exceptions import ToolError

from intergrax.applications._shared.mcp_catalog_side_effect_enforcement import (
    build_catalog_side_effect_enforcement_request,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationConsistencyError,
    MeaningfulSideEffectAuthorizationPort,
    assert_consistent_meaningful_side_effect_authorization_result,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.catalog_tool_child_work_port import (
    CatalogToolChildExecutionRequest,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding


def _tool_error(code: str, detail: str) -> ToolError:
    return ToolError(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CatalogToolChildMeaningfulSideEffectAdmissionHook:
    """Authorize meaningful side effects using child ``execution_id`` (post child bind)."""

    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None
    binding: ActiveTaskBinding
    caller: RequestIdentity
    tool_id: str
    step_id: str
    resolved_agent_id: str
    side_effects: bool

    async def admit(self, request: CatalogToolChildExecutionRequest) -> None:
        _ = request
        if not self.side_effects:
            return
        port = self.meaningful_side_effect_authorization
        if port is None:
            raise _tool_error(
                "unauthorized_side_effect_tool",
                "side-effect catalog tool requires host meaningful side-effect authorization",
            )
        token = bind_governed_execution_task(self.binding.task)
        try:
            enforcement_request = build_catalog_side_effect_enforcement_request(
                binding=self.binding,
                tool_id=self.tool_id,
                step_id=self.step_id,
                caller=self.caller,
            )
            if enforcement_request is None:
                raise _tool_error(
                    "unauthorized_side_effect_tool",
                    "side-effect catalog invoke lacks active execution governance identity binding",
                )
            try:
                authorization = port.authorize(
                    enforcement_request,
                    source_agent_id=self.resolved_agent_id,
                    source_step_id=self.step_id,
                )
                assert_consistent_meaningful_side_effect_authorization_result(authorization)
            except MeaningfulSideEffectAuthorizationConsistencyError as exc:
                raise _tool_error(
                    "side_effect_authorization_denied",
                    f"meaningful side-effect authorization inconsistent: {exc}",
                ) from exc
            if not authorization.permitted or authorization.decision.action is not PolicyAction.ALLOW:
                raise _tool_error(
                    "side_effect_authorization_denied",
                    "meaningful side-effect authorization denied for catalog tool invoke",
                )
        finally:
            reset_governed_execution_task(token)


__all__ = ["CatalogToolChildMeaningfulSideEffectAdmissionHook"]
