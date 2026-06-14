# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Context Engineering contracts (Phase CE-1.1–CE-1.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

CONTEXT_CONTRACTS_SCHEMA = "context_contracts.v1"
ASSEMBLED_CONTEXT_SCHEMA = "assembled_context.v1"

ContextAssemblyScope = Literal["uaep_turn", "graph_node", "delegation_child", "acp_step"]


class ContextFragmentSource(str, Enum):
    """Normative fragment origins for CE assembly (architecture §7.2)."""

    TASK_MESSAGE = "task_message"
    SYSTEM_INSTRUCTIONS = "system_instructions"
    SESSION_HISTORY = "session_history"
    SESSION_HISTORY_SEMANTIC = "session_history_semantic"
    LONGTERM_MEMORY = "longterm_memory"
    RAG = "rag"
    WEBSEARCH = "websearch"
    TOOL_OUTPUT = "tool_output"
    GRAPH_PRIOR = "graph_prior"
    SHARED_CONTEXT = "shared_context"
    ATTACHMENT = "attachment"
    POLICY_OVERLAY = "policy_overlay"
    WORKSPACE = "workspace"
    CUSTOM = "custom"


def content_hash_for_text(text: str) -> str:
    """Stable dedup key for fragment content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextDecisionSnapshot:
    """Portable decision policy for assembly requests (mirrors ``ContextDecisionProfile``)."""

    include_session_history: bool = True
    prefer_longterm_memory: bool = True
    prefer_rag_when_enabled: bool = True
    max_memory_entries_in_context: int = 8


@dataclass(frozen=True, slots=True)
class ContextBudgetSnapshot:
    """Portable budget policy for assembly requests (mirrors ``ContextBudgetPolicy``)."""

    max_chars: int = 16_000
    max_tokens_estimate: int = 4_000
    summary_tier: ContextSummaryTier = ContextSummaryTier.FULL

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if self.max_tokens_estimate < 1:
            raise ValueError("max_tokens_estimate must be >= 1")


@dataclass(frozen=True, slots=True)
class ContextFragment:
    fragment_id: str
    source: ContextFragmentSource
    source_id: str
    content: str
    token_estimate: int
    relevance_score: float
    freshness_score: float
    confidence_score: float
    mandatory: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_hash_for_text(self.content))
        for name in ("relevance_score", "freshness_score", "confidence_score"):
            value = object.__getattribute__(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.token_estimate < 0:
            raise ValueError("token_estimate must be >= 0")


@dataclass(frozen=True, slots=True)
class ContextAssemblyProvenance:
    """Lineage record for an included or excluded fragment."""

    source_type: str
    source_id: str
    fragment_id: str = ""
    schema_version: str = "context_assembly_provenance.v1"


@dataclass(frozen=True, slots=True)
class ContextAssemblyRequest:
    """Serializable input for one ``ContextEngine.assemble`` call (CE-1.2)."""

    trace_id: str
    run_id: str
    task_id: str
    tenant_id: str
    assembly_scope: ContextAssemblyScope
    objective: str
    decision_profile: ContextDecisionSnapshot
    budget_policy: ContextBudgetSnapshot
    assembly_options: TaskContextAssemblyOptions
    step_index: int | None = None
    graph_node_id: str | None = None
    step_kind: str | None = None
    required_sources: frozenset[ContextFragmentSource] = frozenset()
    excluded_sources: frozenset[ContextFragmentSource] = frozenset()
    schema_version: str = CONTEXT_CONTRACTS_SCHEMA

    def __repr__(self) -> str:
        return (
            f"ContextAssemblyRequest(scope={self.assembly_scope!r}, "
            f"task_id={self.task_id!r}, trace_id={self.trace_id!r}, "
            f"step_index={self.step_index!r}, step_kind={self.step_kind!r})"
        )


@dataclass
class ContextProviderContext:
    """Runtime handles for provider ``collect`` — not serialized or logged at INFO."""

    engine_id: str = "default"
    plugin_ids: tuple[str, ...] = ()
    handles: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ContextProviderContext(engine_id={self.engine_id!r}, "
            f"plugin_ids={self.plugin_ids!r}, handle_keys={sorted(self.handles)!r})"
        )


@dataclass(frozen=True, slots=True)
class BudgetAllocationResult:
    included: tuple[ContextFragment, ...]
    excluded: tuple[tuple[ContextFragment, str], ...]
    total_tokens: int
    budget_tokens: int
    degradation_steps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssembledContext:
    messages: tuple[ChatMessage, ...]
    fragments_included: tuple[ContextFragment, ...]
    fragments_excluded: tuple[tuple[ContextFragment, str], ...]
    provenance: tuple[ContextAssemblyProvenance, ...]
    total_tokens: int
    budget_tokens: int
    degradation_steps: tuple[str, ...] = ()
    schema_version: str = ASSEMBLED_CONTEXT_SCHEMA
