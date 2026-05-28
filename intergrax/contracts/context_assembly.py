# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed context assembly options for task intake (§28)."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Dict, Mapping

from pydantic import BaseModel, Field


class ContextAssemblyMetadataKey(StrEnum):
    """Flat metadata keys for context assembly (prefer ``TaskExecutionOptions.context``)."""

    POLICY = "context_assembly_policy"
    SUMMARY_TIER = "context_summary_tier"
    MAX_PRIOR_CHARS = "context_max_prior_chars"
    MAX_PRIOR_ENTRIES = "context_max_prior_entries"
    INCLUDE_SHARED_HANDOFFS = "context_include_shared_handoffs"
    INCLUDE_SHARED_ARTIFACTS = "context_include_shared_artifacts"


CONTEXT_ASSEMBLY_LEGACY_METADATA_KEYS: frozenset[str] = frozenset(
    key.value for key in ContextAssemblyMetadataKey
)

# Backward-compatible aliases.
CONTEXT_ASSEMBLY_POLICY_KEY = ContextAssemblyMetadataKey.POLICY
CONTEXT_SUMMARY_TIER_KEY = ContextAssemblyMetadataKey.SUMMARY_TIER
CONTEXT_MAX_PRIOR_CHARS_KEY = ContextAssemblyMetadataKey.MAX_PRIOR_CHARS
CONTEXT_MAX_PRIOR_ENTRIES_KEY = ContextAssemblyMetadataKey.MAX_PRIOR_ENTRIES
CONTEXT_INCLUDE_SHARED_HANDOFFS_KEY = ContextAssemblyMetadataKey.INCLUDE_SHARED_HANDOFFS
CONTEXT_INCLUDE_SHARED_ARTIFACTS_KEY = ContextAssemblyMetadataKey.INCLUDE_SHARED_ARTIFACTS


class ContextSummaryTier(str, Enum):
    """How much prior-agent narrative is injected into the agent message."""

    FULL = "full"
    SUMMARY_ONLY = "summary_only"
    STRUCTURED_ONLY = "structured_only"
    MINIMAL = "minimal"


class TaskContextAssemblyOptions(BaseModel):
    """Tier-3 / task-scoped rules for bounded context assembly."""

    summary_tier: ContextSummaryTier = ContextSummaryTier.FULL
    max_prior_chars: int = Field(default=4000, ge=1)
    max_prior_entries: int = Field(default=32, ge=1)
    include_shared_handoffs: bool = True
    include_shared_artifacts: bool = False
    schema_version: str = "task_context_assembly.v1"


def context_assembly_options_from_metadata(
    metadata: Mapping[str, Any],
) -> TaskContextAssemblyOptions:
    """Parse context assembly options from flat task metadata (legacy bridge path)."""
    raw = metadata.get(ContextAssemblyMetadataKey.POLICY)
    if isinstance(raw, TaskContextAssemblyOptions):
        return raw
    if isinstance(raw, dict):
        return TaskContextAssemblyOptions.model_validate(raw)

    payload: Dict[str, Any] = {}
    if metadata.get(ContextAssemblyMetadataKey.SUMMARY_TIER) is not None:
        payload["summary_tier"] = metadata[ContextAssemblyMetadataKey.SUMMARY_TIER]
    if metadata.get(ContextAssemblyMetadataKey.MAX_PRIOR_CHARS) is not None:
        payload["max_prior_chars"] = metadata[ContextAssemblyMetadataKey.MAX_PRIOR_CHARS]
    if metadata.get(ContextAssemblyMetadataKey.MAX_PRIOR_ENTRIES) is not None:
        payload["max_prior_entries"] = metadata[ContextAssemblyMetadataKey.MAX_PRIOR_ENTRIES]
    if metadata.get(ContextAssemblyMetadataKey.INCLUDE_SHARED_HANDOFFS) is not None:
        payload["include_shared_handoffs"] = metadata[
            ContextAssemblyMetadataKey.INCLUDE_SHARED_HANDOFFS
        ]
    if metadata.get(ContextAssemblyMetadataKey.INCLUDE_SHARED_ARTIFACTS) is not None:
        payload["include_shared_artifacts"] = metadata[
            ContextAssemblyMetadataKey.INCLUDE_SHARED_ARTIFACTS
        ]
    if payload:
        return TaskContextAssemblyOptions.model_validate(payload)
    return TaskContextAssemblyOptions()


def sync_context_assembly_metadata(
    metadata: Dict[str, Any],
    options: TaskContextAssemblyOptions,
) -> None:
    """Write typed context options to flat metadata for JSON/API compatibility."""
    metadata[ContextAssemblyMetadataKey.SUMMARY_TIER] = options.summary_tier.value
    metadata[ContextAssemblyMetadataKey.MAX_PRIOR_CHARS] = options.max_prior_chars
    metadata[ContextAssemblyMetadataKey.MAX_PRIOR_ENTRIES] = options.max_prior_entries
    metadata[ContextAssemblyMetadataKey.INCLUDE_SHARED_HANDOFFS] = options.include_shared_handoffs
    metadata[ContextAssemblyMetadataKey.INCLUDE_SHARED_ARTIFACTS] = options.include_shared_artifacts
    metadata[ContextAssemblyMetadataKey.POLICY] = options.model_dump(mode="json")
