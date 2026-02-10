from dataclasses import dataclass


@dataclass(slots=True)
class HistoryPolicyConfig:
    llm_spike_ratio: float | None = None
    tool_drop_ratio: float | None = None
    step_spike_ratio: float | None = None
