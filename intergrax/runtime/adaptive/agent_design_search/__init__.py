# © Artur Czarnecki. All rights reserved.

"""Agent Design Search (ADAS) contracts and candidate archive foundation."""

from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchive,
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
    InMemoryAgentDesignArchive,
)
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    ADAS_CONTRACT_SCHEMA_VERSION,
    AdasOperationalEnvelope,
    AgentCandidateDraft,
    AgentCandidateEvaluationResult,
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    AgentDesignObjective,
    AgentDesignSearchRun,
    AggregateResourceLimits,
    BaselineConfigurationRef,
    ConfigurationDelta,
    ResourceAccountingEvidence,
    SearchSelectionOutcome,
    canonical_record_json,
    compute_evidence_bundle_digest,
    verify_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.agent_design_search.sqlite_archive import SQLiteAgentDesignArchive

__all__ = [
    "ADAS_CONTRACT_SCHEMA_VERSION",
    "AdasOperationalEnvelope",
    "AgentCandidateDraft",
    "AgentCandidateEvaluationResult",
    "AgentCandidateEvidenceBundle",
    "AgentCandidateRecord",
    "AgentCandidateStatus",
    "AgentDesignArchive",
    "AgentDesignArchiveError",
    "AgentDesignArchiveScopeError",
    "AgentDesignObjective",
    "AgentDesignSearchRun",
    "AggregateResourceLimits",
    "BaselineConfigurationRef",
    "ConfigurationDelta",
    "InMemoryAgentDesignArchive",
    "ResourceAccountingEvidence",
    "SQLiteAgentDesignArchive",
    "SearchSelectionOutcome",
    "canonical_record_json",
    "compute_evidence_bundle_digest",
    "verify_evidence_bundle_digest",
]
