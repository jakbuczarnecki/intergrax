#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-10.1 — evaluator-loop standard node in product graph specs."""

from __future__ import annotations

import sys

from intergrax.applications._shared.evaluator_loop_graph_templates import evaluator_loop_graph_template
from intergrax.applications._shared.graph_spec_to_plan import application_graph_spec_to_nexus_plan
from intergrax.runtime.task.task import Task, TaskContext


def main() -> int:
    graph = evaluator_loop_graph_template(
        producer_agent_id="producer",
        evaluator_agent_id="evaluator",
        revise_agent_id="revise",
    )
    if graph.evaluator_loop is None:
        print("evaluator loop template must include binding", file=sys.stderr)
        return 1

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="m",
        context=TaskContext(capability="evaluator_loop.pipeline"),
    )
    plan = application_graph_spec_to_nexus_plan(graph, task, classification="multi_agent")
    if "evaluator_loop.v1" not in plan.plan_metadata:
        print("graph plan missing evaluator_loop metadata", file=sys.stderr)
        return 1

    print("OK: evaluator-loop graph template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
