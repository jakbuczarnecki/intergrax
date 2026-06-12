# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed RuntimeEvent payload models (OBS-BUS-1)."""

from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextCandidatePayloadV1,
    DecisionPayloadV1,
    GraphNodePayloadV1,
    DelegationGrantedPayloadV1,
    HandoffPayloadV1,
    HumanPayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    SkillResolvedPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    TraceBridgePayloadV1,
    ValidationPayloadV1,
)

CANONICAL_PAYLOAD_TYPES: tuple[type[RuntimeEventPayload], ...] = (
    DelegationGrantedPayloadV1,
    DecisionPayloadV1,
    ToolPayloadV1,
    ValidationPayloadV1,
    InterruptPayloadV1,
    HumanPayloadV1,
    HandoffPayloadV1,
    AgentSelectionPayloadV1,
    GraphNodePayloadV1,
    LlmCallPayloadV1,
    TraceBridgePayloadV1,
    SkillResolvedPayloadV1,
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextCandidatePayloadV1,
    TaskLifecyclePayloadV1,
)

__all__ = [
    "AgentSelectionPayloadV1",
    "CANONICAL_PAYLOAD_TYPES",
    "ContextAssemblyPayloadV1",
    "ContextAssemblyPayloadV2",
    "ContextCandidatePayloadV1",
    "DecisionPayloadV1",
    "GraphNodePayloadV1",
    "DelegationGrantedPayloadV1",
    "HandoffPayloadV1",
    "HumanPayloadV1",
    "InterruptPayloadV1",
    "LlmCallPayloadV1",
    "RuntimeEventPayload",
    "SkillResolvedPayloadV1",
    "TaskLifecyclePayloadV1",
    "ToolPayloadV1",
    "TraceBridgePayloadV1",
    "ValidationPayloadV1",
]
