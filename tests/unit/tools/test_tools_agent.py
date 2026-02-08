import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.tools_agent import ToolsAgent, ToolTrace
from intergrax.tools.tools_base import ToolRegistry


# ============================================================
# FAKE TOOL
# ============================================================

class SumInput(BaseModel):
    a: int
    b: int


class SumTool:
    name = "sum_tool"
    description = "Adds numbers"
    schema_model = SumInput

    def validate_args(self, args):
        return SumInput.model_validate(args).model_dump()

    def get_parameters(self):
        return SumInput.model_json_schema()

    def run(self, **kwargs):
        return {"result": kwargs["a"] + kwargs["b"]}


# ============================================================
# FAKE REGISTRY
# ============================================================

class FakeRegistry(ToolRegistry):
    def __init__(self):
        self.tool = SumTool()

    def get(self, name):
        return self.tool

    def list(self):
        return [self.tool]

    def to_openai_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "sum_tool",
                "description": "Adds numbers",
                "parameters": SumInput.model_json_schema()
            }
        }]


# ============================================================
# FAKE LLMs
# ============================================================

class NativeLLM:
    def supports_tools(self): return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):
        return {
            "content": "",
            "tool_calls": [{
                "id": "1",
                "function": {"name": "sum_tool", "arguments": '{"a":2,"b":3}'}
            }]
        }


class PlannerLLM:
    def supports_tools(self): return False

    def generate_messages(self, messages, **kwargs):
        return json.dumps({
            "call_tool": {
                "name": "sum_tool",
                "arguments": {"a": 5, "b": 7}
            }
        })


# ============================================================
# TESTS
# ============================================================

def test_native_tool_execution():
    agent = ToolsAgent(NativeLLM(), FakeRegistry())
    result = agent.run("Add numbers")

    assert isinstance(result.tool_traces[0], ToolTrace)
    assert result.tool_traces[0].output["result"] == 5


def test_planner_mode():
    agent = ToolsAgent(PlannerLLM(), FakeRegistry())
    decision = agent.plan_tools("Add")

    assert decision.tool_plan.calls[0].tool_id == "sum_tool"
    assert decision.tool_plan.calls[0].input.a == 5


def test_output_model_mapping():
    class Out(BaseModel):
        result: int

    agent = ToolsAgent(NativeLLM(), FakeRegistry())
    res = agent.run("Add", output_model=Out)

    assert isinstance(res.output_structure, Out)
    assert res.output_structure.result == 5


def test_anti_loop_detection():
    class LoopLLM(NativeLLM):
        pass  # returns same call repeatedly

    agent = ToolsAgent(LoopLLM(), FakeRegistry())
    result = agent.run("loop")

    assert "Stopped repeated identical tool call" in result.final_answer


def test_iteration_limit():
    class InfinitePlanner(PlannerLLM):
        def generate_messages(self, messages, **kwargs):
            return json.dumps({
                "call_tool": {"name": "sum_tool", "arguments": {"a": 1, "b": 1}}
            })

    agent = ToolsAgent(InfinitePlanner(), FakeRegistry())
    result = agent.run("loop")

    assert "iteration limit" in result.final_answer.lower()


import json
from typing import Any, Dict, List

from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.tools.tools_agent import ToolsAgent
from intergrax.tools.tools_base import ToolBase, ToolRegistry


def test_tool_selection_from_many_tools_native_mode():
    # -----------------------------
    # Tool schemas (pydantic)
    # -----------------------------
    class AddInput(BaseModel):
        a: int
        b: int

    class MulInput(BaseModel):
        x: int
        y: int

    # -----------------------------
    # Concrete tools (ToolBase)
    # -----------------------------
    class AddTool(ToolBase):
        name = "add_tool"
        description = "Adds two numbers"
        schema_model = AddInput

        def run(self, **kwargs) -> Any:
            return {"result": kwargs["a"] + kwargs["b"]}

    class MulTool(ToolBase):
        name = "mul_tool"
        description = "Multiplies two numbers"
        schema_model = MulInput

        def run(self, **kwargs) -> Any:
            return {"result": kwargs["x"] * kwargs["y"]}

    # Add some noise tools to simulate a larger registry
    class NoiseTool(ToolBase):
        name = "noise_tool"
        description = "Does something unrelated"
        schema_model = None

        def run(self, **kwargs) -> Any:
            return {"ok": True}

    # -----------------------------
    # Proper registry usage
    # -----------------------------
    registry = ToolRegistry()
    registry.register(AddTool())
    registry.register(MulTool())
    registry.register(NoiseTool())

    # -----------------------------
    # Fake LLM that "selects" tool
    # (we don't test the real model,
    #  we test: ToolsAgent wiring + schema + registry)
    # -----------------------------
    class SmartLLM:
        def supports_tools(self) -> bool:
            return True

        def generate_with_tools(self, messages: List[ChatMessage], tools_schema: List[Dict[str, Any]], **kwargs):
            user_text = (messages[-1].content or "").lower()

            if "add" in user_text:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "function": {"name": "add_tool", "arguments": '{"a":2,"b":3}'},
                        }
                    ],
                }

            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "mul_tool", "arguments": '{"x":2,"y":3}'},
                    }
                ],
            }

    agent = ToolsAgent(SmartLLM(), registry)

    res_add = agent.run("Please add 2 and 3")
    res_mul = agent.run("Please multiply 2 and 3")

    assert res_add.tool_traces[0].tool == "add_tool"
    assert res_add.tool_traces[0].output["result"] == 5

    assert res_mul.tool_traces[0].tool == "mul_tool"
    assert res_mul.tool_traces[0].output["result"] == 6


