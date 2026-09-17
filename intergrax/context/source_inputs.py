# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed semantic source inputs for builtin context providers (MEM-XINT-6-R)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intergrax.context.contracts import IterativeToolOutputBlock
from intergrax.context.session_history import SessionHistorySnapshot
from intergrax.llm.messages import ChatMessage


@dataclass(frozen=True, slots=True)
class ContextMemoryEntryInput:
    entry_id: str
    content: str
    kind: str | None = None
    title: str | None = None
    session_id: str | None = None
    importance: float | None = None
    deleted: bool = False
    raw_relevance_signal: float | None = None


@dataclass(frozen=True, slots=True)
class ContextRagCitationField:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class ContextRagChunkInput:
    chunk_id: str
    content: str
    citation_fields: tuple[ContextRagCitationField, ...] = ()
    extra_metadata_keys: tuple[tuple[str, str], ...] = ()
    raw_relevance_signal: float | None = None


@dataclass(frozen=True, slots=True)
class ContextWebSearchResultInput:
    source_id: str
    content: str
    url: str | None = None
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSystemInstructionsInput:
    text: str


@dataclass(frozen=True, slots=True)
class ContextPolicyOverlayInput:
    overlay_id: str
    content: str
    priority: int = 100


@dataclass(frozen=True, slots=True)
class ContextAttachmentSummaryInput:
    attachment_id: str
    summary: str
    mime_type: str | None = None
    filename: str | None = None
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSharedContextReadInput:
    entry_key: str
    content: str


@dataclass(frozen=True, slots=True)
class ContextPriorOutputInput:
    node_id: str
    content: str
    agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSessionSourceInput:
    snapshot: SessionHistorySnapshot
    binding_context_scope_id: str
    binding_revision_id: str


@dataclass(frozen=True, slots=True)
class ContextProviderSourceInputs:
    """Immutable aggregate of semantic inputs — runtime-only, not logged at INFO."""

    memory: tuple[ContextMemoryEntryInput, ...] = ()
    rag: tuple[ContextRagChunkInput, ...] = ()
    tools: tuple[IterativeToolOutputBlock, ...] = ()
    web: tuple[ContextWebSearchResultInput, ...] = ()
    session: ContextSessionSourceInput | None = None
    system: ContextSystemInstructionsInput | None = None
    attachments: tuple[ContextAttachmentSummaryInput, ...] = ()
    shared_context: tuple[ContextSharedContextReadInput, ...] = ()
    policy_overlay: tuple[ContextPolicyOverlayInput, ...] = ()
    graph_prior: tuple[ContextPriorOutputInput, ...] = ()

    def with_session(self, session: ContextSessionSourceInput | None) -> ContextProviderSourceInputs:
        return replace(self, session=session)

    def merge(self, other: ContextProviderSourceInputs) -> ContextProviderSourceInputs:
        """Compose sources without mutating either side."""
        return ContextProviderSourceInputs(
            memory=other.memory if other.memory else self.memory,
            rag=other.rag if other.rag else self.rag,
            tools=other.tools if other.tools else self.tools,
            web=other.web if other.web else self.web,
            session=other.session if other.session is not None else self.session,
            system=other.system if other.system is not None else self.system,
            attachments=other.attachments if other.attachments else self.attachments,
            shared_context=other.shared_context if other.shared_context else self.shared_context,
            policy_overlay=other.policy_overlay if other.policy_overlay else self.policy_overlay,
            graph_prior=other.graph_prior if other.graph_prior else self.graph_prior,
        )

    def summary_repr(self) -> str:
        return (
            f"memory={len(self.memory)} rag={len(self.rag)} tools={len(self.tools)} "
            f"web={len(self.web)} session={'yes' if self.session else 'no'} "
            f"system={'yes' if self.system else 'no'} attachments={len(self.attachments)} "
            f"shared={len(self.shared_context)} policy={len(self.policy_overlay)} "
            f"graph_prior={len(self.graph_prior)}"
        )


@dataclass(frozen=True, slots=True)
class ContextTaskMessageAuxiliaryInput:
    """Optional chat turns for task-message fallback when objective is empty."""

    messages: tuple[ChatMessage, ...] = ()
