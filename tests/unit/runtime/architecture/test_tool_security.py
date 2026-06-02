from __future__ import annotations

from intergrax.runtime.architecture.tool_security import (
    ToolInvocationPolicy,
    ToolInvocationRequest,
    evaluate_tool_invocation_security,
)


def test_tool_security_denies_blocked_token_in_arguments() -> None:
    decision = evaluate_tool_invocation_security(
        request=ToolInvocationRequest(
            tool_id="rag.retrieve",
            arguments={"query": "dump data; DROP TABLE users"},
            capability_ids=["rag.retrieve"],
        ),
        policy=ToolInvocationPolicy(
            allowed_tool_ids=["rag.retrieve"],
            blocked_argument_tokens=["DROP TABLE"],
            require_explicit_capability_match=True,
        ),
    )
    assert decision.allowed is False
    assert any("Blocked token" in reason for reason in decision.reasons)


def test_tool_security_denies_missing_capability_match() -> None:
    decision = evaluate_tool_invocation_security(
        request=ToolInvocationRequest(
            tool_id="rag.retrieve",
            arguments={"query": "safe prompt"},
            capability_ids=["websearch.query"],
        ),
        policy=ToolInvocationPolicy(
            allowed_tool_ids=["rag.retrieve"],
            blocked_argument_tokens=[],
            require_explicit_capability_match=True,
        ),
    )
    assert decision.allowed is False
    assert any("capability match" in reason for reason in decision.reasons)
