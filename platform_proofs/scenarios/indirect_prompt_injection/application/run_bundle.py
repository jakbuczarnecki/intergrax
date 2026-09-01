"""Application-owned runtime bundle for order assistant execution."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.indirect_prompt_injection.application.agent import OrderAssistantAgent
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ScenarioRuntimeComposition,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import WorkflowKind


@dataclass(frozen=True, slots=True)
class OrderAssistantRunBundle:
    workflow: WorkflowKind
    provider_client: OrderProviderClient
    agent: OrderAssistantAgent
    runtime_composition: ScenarioRuntimeComposition
    order_id: str
    user_message: str
