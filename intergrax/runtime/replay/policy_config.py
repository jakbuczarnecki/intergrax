from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionPolicyConfig:
    max_total_tokens: int | None = None
    max_llm_call_delta: int | None = None
    min_tool_calls: int | None = None
    max_steps: int | None = None
    fail_on_answer_change: bool = False
