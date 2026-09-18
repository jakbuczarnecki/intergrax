# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Context Engineering contracts (Phase CE-1.1–CE-1.2)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Self

from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.contracts.data_classification import DataClassification
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

if TYPE_CHECKING:
    from intergrax.context.budget.compaction import ContextCompactionProvenance
    from intergrax.context.budget.contracts import ResolvedModelContextBudget
    from intergrax.context.planning import ContextPlan
    from intergrax.context.source_inputs import ContextProviderSourceInputs
    from intergrax.runtime.nexus.context.assembly_runtime_deps import (
        ContextAssemblyRuntimeDependencies,
    )

CONTEXT_CONTRACTS_SCHEMA = "context_contracts.v1"
ASSEMBLED_CONTEXT_SCHEMA = "assembled_context.v1"
CONTEXT_PROVIDER_DESCRIPTOR_SCHEMA = "context_provider_descriptor.v1"
CONTEXT_PROVIDER_SET_SNAPSHOT_SCHEMA = "context_provider_set_snapshot.v1"
BUILTIN_PROVIDER_VERSION = "1.0.0"

ContextProviderCollectionStatus = Literal["success", "skipped", "failed", "degraded"]

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


def canonicalize_fragment_content(text: str) -> str:
    """Deterministic minimal canonicalization before content hashing (no semantic rewrite)."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def content_hash_for_text(text: str) -> str:
    """Stable dedup key for fragment content."""
    canonical = canonicalize_fragment_content(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ContextAuthorityClass(str, Enum):
    """Typed authority for cross-source policy (MEM-XINT-2-R)."""

    UNASSIGNED = "unassigned"
    SYSTEM_CONTEXT = "system_context"
    CANONICAL_MEMORY = "canonical_memory"
    DERIVED_MEMORY = "derived_memory"
    RAG_EVIDENCE = "rag_evidence"
    TOOL_OBSERVATION = "tool_observation"
    SESSION_EPISODIC = "session_episodic"


PROVIDER_FORBIDDEN_AUTHORITY_CLASSES: frozenset[ContextAuthorityClass] = frozenset(
    {
        ContextAuthorityClass.SYSTEM_CONTEXT,
        ContextAuthorityClass.CANONICAL_MEMORY,
    }
)


class ContextPolicyStage(str, Enum):
    SCOPE_ISOLATION = "scope_isolation"
    CANONICALIZE = "canonicalize"
    EXACT_DEDUP = "exact_dedup"
    NORMALIZE = "normalize"
    SEMANTIC_DEDUP = "semantic_dedup"
    CONFLICT = "conflict"
    RANK = "rank"
    BUDGET = "budget"


class ContextPolicyReasonCode(str, Enum):
    SCOPE_INCOMPATIBLE = "scope_incompatible"
    EXACT_DUPLICATE_IDENTITY = "exact_duplicate_identity"
    EXACT_DUPLICATE_CONTENT = "exact_duplicate_content"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    CONFLICT_RESOLVED = "conflict_resolved"
    BUDGET_EXCLUDED = "budget_excluded"
    QUALITY_THRESHOLD = "quality_threshold"


class ContextConflictAction(str, Enum):
    KEEP_BOTH = "keep_both"
    PREFER_LEFT = "prefer_left"
    PREFER_RIGHT = "prefer_right"
    DOWNRANK_LEFT = "downrank_left"
    DOWNRANK_RIGHT = "downrank_right"
    ANNOTATE = "annotate"


@dataclass(frozen=True, slots=True)
class ContextFragmentScopeRef:
    tenant_id: str
    user_id: str = ""
    execution_scope_key: str = ""

    def __post_init__(self) -> None:
        tenant = self.tenant_id.strip()
        if not tenant:
            raise ValueError("tenant_id must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)


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
class IterativeToolOutputBlock:
    """Typed carrier for iterative native tool feedback into CE (UE-6C)."""

    content: str
    tool_call_id: str
    tool_name: str
    step_id: str | None


@dataclass(frozen=True, slots=True)
class ContextProviderDescriptor:
    """Immutable semantic identity for one registered context source provider."""

    provider_id: str
    provider_version: str
    supported_sources: frozenset[ContextFragmentSource]
    origin: str = "builtin"
    trusted_authority_class: ContextAuthorityClass | None = None
    allowed_authority_classes: frozenset[ContextAuthorityClass] = frozenset(
        {ContextAuthorityClass.UNASSIGNED},
    )
    schema_version: str = CONTEXT_PROVIDER_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        normalized_id = self.provider_id.strip().lower()
        if not normalized_id:
            raise ValueError("provider_id must be non-empty")
        object.__setattr__(self, "provider_id", normalized_id)
        version = self.provider_version.strip()
        if not version:
            raise ValueError("provider_version must be non-empty")
        forbidden = {"unknown", "latest", "current", "0"}
        if version.lower() in forbidden:
            raise ValueError(f"provider_version must be explicit, got {version!r}")
        object.__setattr__(self, "provider_version", version)
        origin = self.origin.strip()
        if not origin:
            raise ValueError("origin must be non-empty")
        object.__setattr__(self, "origin", origin)
        if not self.supported_sources:
            raise ValueError("supported_sources must be non-empty")


@dataclass(frozen=True, slots=True)
class ContextProviderProvenance:
    """Canonical provider lineage attached to fragments and assembly provenance."""

    provider_id: str
    provider_version: str
    origin: str = "builtin"
    schema_version: str = "context_provider_provenance.v1"

    @classmethod
    def from_descriptor(cls, descriptor: ContextProviderDescriptor) -> Self:
        return cls(
            provider_id=descriptor.provider_id,
            provider_version=descriptor.provider_version,
            origin=descriptor.origin,
        )


class ContextPolicyInvariantViolationCode(str, Enum):
    """Hard invariant failures for replaceable policy pipelines (MEM-XINT-5-R2)."""

    DUPLICATE_FRAGMENT_ID = "duplicate_fragment_id"
    UNKNOWN_FRAGMENT = "unknown_fragment"
    SOURCE_CHANGED = "source_changed"
    SOURCE_ID_CHANGED = "source_id_changed"
    PROVENANCE_CHANGED = "provenance_changed"
    AUTHORITY_CHANGED = "authority_changed"
    SENSITIVITY_CHANGED = "sensitivity_changed"
    SCOPE_CHANGED = "scope_changed"
    CONTENT_CHANGED = "content_changed"
    RAW_RELEVANCE_CHANGED = "raw_relevance_changed"
    INVALID_DECISION_REFERENCE = "invalid_decision_reference"


@dataclass(frozen=True, slots=True)
class ContextFragmentInvariantSnapshot:
    """Immutable source facts captured before replaceable policy stages."""

    fragment_id: str
    source: ContextFragmentSource
    source_id: str
    provider_provenance: ContextProviderProvenance | None
    authority_class: ContextAuthorityClass
    sensitivity: DataClassification
    scope_ref: ContextFragmentScopeRef | None
    canonical_content_hash: str
    raw_relevance_signal: float


@dataclass(frozen=True, slots=True)
class ContextProviderSetSnapshot:
    """Deterministic provider-set identity — descriptors only, no provider objects."""

    engine_id: str
    providers: tuple[ContextProviderDescriptor, ...]
    fingerprint: str
    schema_version: str = CONTEXT_PROVIDER_SET_SNAPSHOT_SCHEMA


@dataclass(frozen=True, slots=True)
class ContextProviderCollectionOutcome:
    """Bounded provider contribution result for assembly inspection."""

    descriptor: ContextProviderDescriptor
    status: ContextProviderCollectionStatus
    fragment_count: int = 0
    failure_reason: str = ""
    reason_code: str = ""


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
    provider_provenance: ContextProviderProvenance | None = None
    authority_class: ContextAuthorityClass = ContextAuthorityClass.UNASSIGNED
    trust_score: float = 0.75
    sensitivity: DataClassification = DataClassification.INTERNAL
    scope_ref: ContextFragmentScopeRef | None = None
    raw_relevance_signal: float | None = None
    normalized_relevance_score: float | None = None
    conflict_key: str = ""
    semantic_fingerprint: str = ""

    def __post_init__(self) -> None:
        canonical_content = canonicalize_fragment_content(self.content)
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                hashlib.sha256(canonical_content.encode("utf-8")).hexdigest(),
            )
        for name in ("freshness_score", "confidence_score", "trust_score"):
            value = object.__getattribute__(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        raw_signal = self.raw_relevance_signal
        if raw_signal is None:
            raw_signal = self.relevance_score
        if math.isnan(raw_signal) or math.isinf(raw_signal):
            raw_signal = 0.0
        object.__setattr__(self, "raw_relevance_signal", raw_signal)
        normalized = self.normalized_relevance_score
        if normalized is None:
            normalized = _clamp_unit_interval(self.relevance_score)
        else:
            normalized = _clamp_unit_interval(normalized)
        object.__setattr__(self, "normalized_relevance_score", normalized)
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(f"relevance_score must be in [0, 1], got {self.relevance_score}")
        if self.token_estimate < 0:
            raise ValueError("token_estimate must be >= 0")
        if not self.semantic_fingerprint:
            fingerprint_source = canonical_content.casefold()
            object.__setattr__(
                self,
                "semantic_fingerprint",
                hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
            )


@dataclass(frozen=True, slots=True)
class ContextAssemblyProvenance:
    """Lineage record for an included or excluded fragment."""

    source_type: str
    source_id: str
    fragment_id: str = ""
    provider_id: str = ""
    provider_version: str = ""
    provider_origin: str = ""
    content_hash: str = ""
    schema_version: str = "context_assembly_provenance.v2"


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
    workspace_id: str | None = None
    step_index: int | None = None
    graph_node_id: str | None = None
    step_kind: str | None = None
    user_id: str = ""
    required_sources: frozenset[ContextFragmentSource] = frozenset()
    excluded_sources: frozenset[ContextFragmentSource] = frozenset()
    execution_scope: ModelCallExecutionScope = ModelCallExecutionScope.PRIMARY_MODEL_CALL
    schema_version: str = CONTEXT_CONTRACTS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.execution_scope, ModelCallExecutionScope):
            raise ValueError("execution_scope must be ModelCallExecutionScope")
        if self.workspace_id is not None:
            stripped = self.workspace_id.strip()
            if not stripped:
                raise ValueError("workspace_id must be non-empty when provided")
            if stripped != self.workspace_id:
                object.__setattr__(self, "workspace_id", stripped)

    def __repr__(self) -> str:
        return (
            f"ContextAssemblyRequest(scope={self.assembly_scope!r}, "
            f"task_id={self.task_id!r}, trace_id={self.trace_id!r}, "
            f"step_index={self.step_index!r}, step_kind={self.step_kind!r})"
        )


def _clamp_unit_interval(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True, slots=True)
class ContextNormalizationInput:
    fragment: ContextFragment
    request: ContextAssemblyRequest


@dataclass(frozen=True, slots=True)
class ContextSemanticDedupDecision:
    kept_fragment_id: str
    suppressed_fragment_ids: tuple[str, ...]
    reason_code: ContextPolicyReasonCode
    strategy_id: str


@dataclass(frozen=True, slots=True)
class ContextConflictDecision:
    left_fragment_id: str
    right_fragment_id: str
    action: ContextConflictAction
    kept_fragment_ids: tuple[str, ...]
    reason_code: ContextPolicyReasonCode
    strategy_id: str


@dataclass(frozen=True, slots=True)
class ContextPolicyDecision:
    stage: ContextPolicyStage
    strategy_id: str
    input_fragment_ids: tuple[str, ...]
    output_fragment_ids: tuple[str, ...]
    reason_code: ContextPolicyReasonCode
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ContextPolicyPipelineResult:
    fragments: tuple[ContextFragment, ...]
    excluded: tuple[tuple[ContextFragment, str], ...]
    decisions: tuple[ContextPolicyDecision, ...]
    semantic_dedup_decisions: tuple[ContextSemanticDedupDecision, ...] = ()
    conflict_decisions: tuple[ContextConflictDecision, ...] = ()


def replace_context_fragment(fragment: ContextFragment, **updates: object) -> ContextFragment:
    """Return a copy of ``fragment`` with selective field overrides."""
    current = {
        "fragment_id": fragment.fragment_id,
        "source": fragment.source,
        "source_id": fragment.source_id,
        "content": fragment.content,
        "token_estimate": fragment.token_estimate,
        "relevance_score": fragment.relevance_score,
        "freshness_score": fragment.freshness_score,
        "confidence_score": fragment.confidence_score,
        "mandatory": fragment.mandatory,
        "metadata": dict(fragment.metadata),
        "content_hash": fragment.content_hash,
        "provider_provenance": fragment.provider_provenance,
        "authority_class": fragment.authority_class,
        "trust_score": fragment.trust_score,
        "sensitivity": fragment.sensitivity,
        "scope_ref": fragment.scope_ref,
        "raw_relevance_signal": fragment.raw_relevance_signal,
        "normalized_relevance_score": fragment.normalized_relevance_score,
        "conflict_key": fragment.conflict_key,
        "semantic_fingerprint": fragment.semantic_fingerprint,
    }
    current.update(updates)
    return ContextFragment(**current)  # type: ignore[arg-type]


def _default_provider_source_inputs() -> ContextProviderSourceInputs:
    from intergrax.context.source_inputs import ContextProviderSourceInputs

    return ContextProviderSourceInputs()


@dataclass
class ContextProviderContext:
    """Runtime context for provider ``collect`` — not serialized or logged at INFO.

    ``sources`` carries canonical typed semantic inputs. ``runtime`` carries typed assembly
    dependencies for ``ContextEngine.assemble``. ``handles`` is legacy auxiliary compatibility
    only (workspace files, provider pinning, session bridge writers). Canonical engine paths
    must not read semantic assembly inputs from ``handles``.
    """

    engine_id: str = "default"
    plugin_ids: tuple[str, ...] = ()
    sources: ContextProviderSourceInputs = field(default_factory=_default_provider_source_inputs)
    runtime: ContextAssemblyRuntimeDependencies | None = None
    handles: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._hydrate_sources_from_legacy_session_handles()

    def _hydrate_sources_from_legacy_session_handles(self) -> None:
        """Writer-side compatibility: move session snapshot handles into ``sources`` once."""
        if self.sources.session is not None:
            return
        from intergrax.context.session_history import (
            SESSION_HISTORY_CONTEXT_SCOPE_HANDLE,
            SESSION_HISTORY_REVISION_HANDLE,
            SESSION_HISTORY_SNAPSHOT_HANDLE,
            SessionHistorySnapshot,
        )
        from intergrax.context.source_inputs import ContextSessionSourceInput

        raw = self.handles.get(SESSION_HISTORY_SNAPSHOT_HANDLE)
        if type(raw) is not SessionHistorySnapshot:
            return
        scope = self.handles.get(SESSION_HISTORY_CONTEXT_SCOPE_HANDLE)
        revision = self.handles.get(SESSION_HISTORY_REVISION_HANDLE)
        if type(scope) is not str or not scope.strip():
            scope = raw.context_scope_id
        if type(revision) is not str or not revision.strip():
            revision = raw.revision_id
        self.sources = self.sources.with_session(
            ContextSessionSourceInput(
                snapshot=raw,
                binding_context_scope_id=scope.strip(),
                binding_revision_id=revision.strip(),
            )
        )

    def __repr__(self) -> str:
        return (
            f"ContextProviderContext(engine_id={self.engine_id!r}, "
            f"plugin_ids={self.plugin_ids!r}, sources=({self.sources.summary_repr()}), "
            f"runtime={'set' if self.runtime is not None else 'none'}, "
            f"handle_keys={sorted(self.handles)!r})"
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
    context_plan: ContextPlan | None = None
    provider_outcomes: tuple[ContextProviderCollectionOutcome, ...] = ()
    provider_set_snapshot: ContextProviderSetSnapshot | None = None
    policy_decisions: tuple[ContextPolicyDecision, ...] = ()
    policy_semantic_dedup: tuple[ContextSemanticDedupDecision, ...] = ()
    policy_conflicts: tuple[ContextConflictDecision, ...] = ()
    resolved_model_budget: ResolvedModelContextBudget | None = None
    compaction_provenance: tuple[ContextCompactionProvenance, ...] = ()
    token_counter_strategy_id: str = ""
    compaction_strategy_id: str = ""
    degradation_policy_id: str = ""
    schema_version: str = ASSEMBLED_CONTEXT_SCHEMA
