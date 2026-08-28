# © Artur Czarnecki. All rights reserved.

"""Extract UAEP-promoted domain payload from agent execution results."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_execution_result import AgentExecutionResult


def domain_payload_from_execution(execution: AgentExecutionResult) -> dict[str, Any]:
    structured = dict(execution.structured_data)
    domain_summary = structured.get("domain_summary")
    if isinstance(domain_summary, dict):
        return dict(domain_summary)
    return structured
