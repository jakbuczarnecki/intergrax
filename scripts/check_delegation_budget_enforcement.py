#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-10.2 — budget delegation enforcement on reference hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.delegation_budget_wiring import resolve_delegation_budget_policy
from intergrax.applications._shared.graph_spec_to_plan import application_graph_spec_to_nexus_plan
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphEdge, GraphEdgeKind, GraphNode
from intergrax.runtime.task.task import Task, TaskContext


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    policy = resolve_delegation_budget_policy(env)
    if not policy.enforcement_enabled:
        print("product host must enforce delegation budgets", file=sys.stderr)
        return 1
    if policy.max_llm_calls is None or policy.max_tool_calls is None:
        print("product host must declare delegation llm/tool budgets", file=sys.stderr)
        return 1

    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="parent"), GraphNode(agent_id="child")],
        edges=[
            GraphEdge(
                source_agent_id="parent",
                target_agent_id="child",
                kind=GraphEdgeKind.DELEGATES_TO,
            )
        ],
    )
    task = Task(tenant_id="t1", user_id="u1", message="m", context=TaskContext(capability="x.pipeline"))
    plan = application_graph_spec_to_nexus_plan(
        spec,
        task,
        classification="multi_agent",
        delegation_budget=policy,
    )
    delegated = next(step for step in plan.steps if step.delegation is not None)
    if delegated.delegation is None:
        print("delegation step missing", file=sys.stderr)
        return 1
    if delegated.delegation.max_llm_calls != policy.max_llm_calls:
        print("delegation max_llm_calls not applied", file=sys.stderr)
        return 1
    if delegated.delegation.max_tool_calls != policy.max_tool_calls:
        print("delegation max_tool_calls not applied", file=sys.stderr)
        return 1

    print("OK: delegation budget enforcement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
