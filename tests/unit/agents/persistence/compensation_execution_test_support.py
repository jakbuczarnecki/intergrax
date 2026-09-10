# © Artur Czarnecki. All rights reserved.

"""Test doubles for admitted compensation side-effect execution."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)


def build_test_admitted_compensation_side_effect_execution(
    invoker: DeclarativeToolInvoker,
    *,
    authority: ParentExecutionAuthority | None = None,
) -> CompensationSideEffectExecutionPort:
    return build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(invoker),
        authority=authority or ParentExecutionAuthority.unrestricted_root(),
    )
