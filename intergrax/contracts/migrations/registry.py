# © Artur Czarnecki. All rights reserved.

"""Canonical schema_version registry (architecture §40.11 · ACP-PROD-11)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContractSchemaEntry:
    contract_name: str
    module_path: str
    current_version: str


CONTRACT_SCHEMA_REGISTRY: tuple[ContractSchemaEntry, ...] = (
    ContractSchemaEntry("AgentRunRequest", "intergrax.contracts.agent_run", "agent_run.v1"),
    ContractSchemaEntry("AgentRunResult", "intergrax.contracts.agent_run", "agent_run.v1"),
    ContractSchemaEntry("AgentRunTrace", "intergrax.contracts.agent_run_trace", "agent_run_trace.v1"),
    ContractSchemaEntry("AcpSessionState", "intergrax.contracts.acp_state", "acp.state.v1"),
    ContractSchemaEntry("ArtifactRef", "intergrax.contracts.artifact_ref", "artifact_ref.v1"),
    ContractSchemaEntry("AgentRunCheckpoint", "intergrax.contracts.side_effect", "agent_run_checkpoint.v1"),
    ContractSchemaEntry("SideEffectRecord", "intergrax.contracts.side_effect", "side_effect.v1"),
    ContractSchemaEntry(
        "OrganizationalPolicyEnvelope",
        "intergrax.contracts.org_policy",
        "org_policy_envelope.v1",
    ),
)
