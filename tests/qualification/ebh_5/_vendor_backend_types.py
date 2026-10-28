# © Artur Czarnecki. All rights reserved.

"""EBH-5 VBN record types (VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE §2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RoadmapVendorDomain(StrEnum):
    RELATIONAL_PERSISTENCE = "1-relational-persistence"
    NOSQL_DOCUMENT = "2-nosql-document"
    KEY_VALUE = "3-key-value"
    CACHE = "4-cache-distributed"
    VECTOR = "5-vector-database"
    GRAPH = "6-graph-database"
    OBJECT_BLOB = "7-object-blob"
    EVENT_STREAMING = "8-event-streaming"
    QUEUES_WORKERS = "9-queues-workers"
    DISTRIBUTED_COORDINATION = "10-distributed-coordination"
    TRANSACTION_IDEMPOTENCY = "11-transaction-idempotency"
    CHECKPOINTS_EXECUTION = "12-checkpoints-execution-state"
    AGENT_MEMORY = "13-agent-memory"
    RAG_RETRIEVAL = "14-rag-retrieval"
    LLM_EMBEDDING = "15-llm-embedding"
    AGENT_RUNTIMES = "16-agent-runtimes"
    HOSTING_SUBSTRATE = "17-hosting-substrate"
    EXTERNAL_MCP = "18-external-integrations-mcp"
    LOGGING = "19-logging"
    METRICS = "20-metrics"
    TRACING = "21-tracing"
    DIAGNOSTICS = "22-diagnostics"
    ALERTING = "23-alerting"
    AUDIT_EVIDENCE = "24-audit-evidence"
    SECRETS = "25-secrets"
    AUTH_IDENTITY = "26-auth-identity"
    CONFIG_PROFILE = "27-config-profile"
    NETWORK_HTTP = "28-network-http"
    TELEMETRY_OTLP = "29-telemetry-otlp"
    EVAL_ANALYTICS = "30-evaluation-analytics"
    DISCOVERED = "31-discovered"


class ExtensibilityPromised(StrEnum):
    YES = "yes"
    NO = "no"
    PARTIAL = "partial — WITH EVIDENCE"


class ExtensibilityClassification(StrEnum):
    REPLACEABLE_QUALIFIED = "REPLACEABLE — QUALIFIED"
    REPLACEABLE_NOT_YET = "REPLACEABLE — NOT YET QUALIFIED"
    FIXED_SUBSTRATE = "FIXED SUBSTRATE — RATIFIED WITH EVIDENCE"
    INTERNAL = "INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED"
    ARCHITECTURE_BLOCKER = "ARCHITECTURE BLOCKER"


class CoreChangeToSwitch(StrEnum):
    NO = "no"
    YES_BLOCKER = "yes — BLOCKER"
    ADAPTER_ONLY = "adapter-only"
    CONFIG_ONLY = "config-only"


class ConcernConclusion(StrEnum):
    ACCEPTED = "ACCEPTED"
    DEBT = "DEBT"
    BLOCKER = "BLOCKER"
    REQUIRES_ADR = "REQUIRES ADR"


class RplGateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NA = "N/A — WITH EVIDENCE"


@dataclass(frozen=True, slots=True)
class VendorBackendConcernRecord:
    concern_id: str
    domain: RoadmapVendorDomain
    semantic_owner: str
    canonical_contract_or_port: str
    composition_selection_owner: str
    concrete_implementation: str
    extensibility_promised: ExtensibilityPromised
    configuration_profile_source: str
    lifecycle_activation_authority: str
    vendor_code_location: str
    core_consumers: tuple[str, ...]
    core_change_required_to_switch: CoreChangeToSwitch
    fallback_behavior: str
    structural_replaceability_evidence_id: str
    behavioral_qualification_evidence_id: str
    frz_criteria: tuple[str, ...]
    extensibility_classification: ExtensibilityClassification
    conclusion: ConcernConclusion
    outstanding_debt: str
    config_x_concern_id: str | None = None
    production_effective: bool = True


@dataclass(frozen=True, slots=True)
class ReplaceabilityMatrixRow:
    concern_id: str
    rpl_a: RplGateStatus
    rpl_b: RplGateStatus
    rpl_c: RplGateStatus
    rpl_d: RplGateStatus
    rpl_e: RplGateStatus
    rpl_f: RplGateStatus
    rpl_g: RplGateStatus
    rpl_h: ExtensibilityClassification
    alternate_implementation_ref: str
    behavioral_proof_ref: str
    regression_gate_ids: tuple[str, ...]
