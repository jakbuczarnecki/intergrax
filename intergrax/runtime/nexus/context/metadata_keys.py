# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime metadata keys written by ContextManager (§28)."""

from enum import StrEnum


class AgentContextMetadataKey(StrEnum):
    """Flat metadata keys injected by ``ContextManager.apply_to_task``."""

    AGENT_CONTEXT = "agent_context"
    AGENT_CONTEXT_BUNDLE = "agent_context_bundle"
    CONTEXT_PROVENANCE = "context_provenance"
    SHARED_CONTEXT_READS = "shared_context_reads"
    PRIOR_AGENT_OUTPUTS = "prior_agent_outputs"


HANDOFF_STRUCTURED_OUTPUT_PREFIX = "handoff:"

# Backward-compatible aliases.
AGENT_CONTEXT_KEY = AgentContextMetadataKey.AGENT_CONTEXT
AGENT_CONTEXT_BUNDLE_KEY = AgentContextMetadataKey.AGENT_CONTEXT_BUNDLE
CONTEXT_PROVENANCE_KEY = AgentContextMetadataKey.CONTEXT_PROVENANCE
SHARED_CONTEXT_READS_KEY = AgentContextMetadataKey.SHARED_CONTEXT_READS
PRIOR_AGENT_OUTPUTS_KEY = AgentContextMetadataKey.PRIOR_AGENT_OUTPUTS
