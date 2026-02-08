# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import json

from pydantic import BaseModel

from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tools_agent import ToolsAgent
from tests._support.builder import FakeLLMAdapter


def make_contract(tool_id: str, input_model, output_model):
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=f"{tool_id} description",
        input_schema=input_model,
        output_schema=output_model,
        error_mapping={},
        side_effects=False,
    )


# ============================================================
# FAKE LLMs
# ============================================================

class NativeLLM(FakeLLMAdapter):
    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools, **kwargs):
        # Always call sum_tool with a=2, b=3
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "sum_tool",
                        "arguments": json.dumps({"a": 2, "b": 3}),
                    },
                }
            ],
        }


class PlannerLLM(FakeLLMAdapter):
    def supports_tools(self) -> bool:
        return False

    def generate_messages(self, messages, **kwargs):
        return json.dumps(
            {"call_tool": {"name": "sum_tool", "arguments": {"a": 4, "b": 5}}}
        )


# ============================================================
# MODELS
# ============================================================

class SumInput(BaseModel):
    a: int
    b: int


class SumOutput(BaseModel):
    result: int


# ============================================================
# HANDLERS
# ============================================================

class SumHandler:
    def execute(self, request: ToolExecutionRequest) -> SumOutput:
        return SumOutput(result=request.input.a + request.input.b)


# ============================================================
# TESTS
# ============================================================

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        make_contract("sum_tool", SumInput, SumOutput),
        SumHandler(),
    )
    return registry


def test_native_tool_execution():
    agent = ToolsAgent(NativeLLM(), build_registry())
    result = agent.run("Add numbers", run_id="r1")

    assert result.tool_traces
    assert result.tool_traces[0].output["result"] == 5


def test_planner_mode():
    agent = ToolsAgent(PlannerLLM(), build_registry())
    decision = agent.plan_tools("Add")

    assert decision.tool_plan.calls[0].tool_id == "sum_tool"


def test_output_model_mapping():
    agent = ToolsAgent(NativeLLM(), build_registry())
    result = agent.run("Add", run_id="r2", output_model=SumOutput)

    assert isinstance(result.output_structure, SumOutput)
    assert result.output_structure.result == 5


def test_anti_loop_detection():
    class LoopLLM(NativeLLM):
        def generate_with_tools(self, messages, tools, **kwargs):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "sum_tool",
                            "arguments": json.dumps({"a": 1, "b": 1}),
                        },
                    }
                ],
            }

    agent = ToolsAgent(LoopLLM(), build_registry())
    result = agent.run("Loop", run_id="r3")

    assert "Stopped repeated identical tool call" in result.final_answer


def test_iteration_limit():
    class EndlessLLM(NativeLLM):
        def generate_with_tools(self, messages, tools, **kwargs):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "sum_tool",
                            "arguments": json.dumps({"a": 1, "b": 2}),
                        },
                    }
                ],
            }

    agent = ToolsAgent(EndlessLLM(), build_registry())
    agent.cfg.max_tool_iters = 1
    result = agent.run("Endless", run_id="r4")

    assert "iteration limit" in result.final_answer.lower()


def test_tool_selection_from_many_tools_native_mode():
    registry = build_registry()

    class SubOutput(BaseModel):
        result: int

    class SubHandler:
        def execute(self, request: ToolExecutionRequest) -> SubOutput:
            return SubOutput(result=request.input.a - request.input.b)

    class SubInput(BaseModel):
        a: int
        b: int

    registry.register(
        make_contract("sub_tool", SubInput, SubOutput),
        SubHandler(),
    )

    agent = ToolsAgent(NativeLLM(), registry)
    result = agent.run("Add numbers", run_id="r5")

    assert result.tool_traces[0].tool == "sum_tool"


def test_planner_tool_selection_from_many_tools_native_mode():
    agent = ToolsAgent(PlannerLLM(), build_registry())
    decision = agent.plan_tools("Add numbers")

    assert decision.tool_plan.calls[0].tool_id == "sum_tool"
