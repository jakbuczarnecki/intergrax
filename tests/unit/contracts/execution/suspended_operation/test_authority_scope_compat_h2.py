# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
    UnknownInvocationScopeError,
    infer_authority_scope_from_invocation,
    invocation_scope_matches_authority_scope,
    recognize_authority_scope_from_invocation,
)

pytestmark = pytest.mark.unit


def test_recognize_agent_scope() -> None:
    scope = recognize_authority_scope_from_invocation("agr_deadbeef")
    assert scope is SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE


def test_recognize_declarative_scope() -> None:
    scope = recognize_authority_scope_from_invocation("dhr_deadbeef")
    assert scope is SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE


def test_recognize_canonical_mse_scope() -> None:
    scope = recognize_authority_scope_from_invocation(
        CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
    )
    assert scope is SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT
    scoped = recognize_authority_scope_from_invocation(
        f"{CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID}:tool-a",
    )
    assert scoped is SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT


def test_unknown_prefix_fails_closed() -> None:
    with pytest.raises(UnknownInvocationScopeError):
        infer_authority_scope_from_invocation("unknown_scope")


def test_empty_and_whitespace_fail_closed() -> None:
    with pytest.raises(UnknownInvocationScopeError):
        infer_authority_scope_from_invocation("")
    with pytest.raises(UnknownInvocationScopeError):
        infer_authority_scope_from_invocation("   ")


def test_match_unknown_scope_is_false() -> None:
    assert not invocation_scope_matches_authority_scope(
        "random_scope",
        SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT,
    )
