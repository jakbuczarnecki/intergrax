# © Artur Czarnecki. All rights reserved.

"""Durable citation-reference resolution for conversational inspect actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from local_workspace_application.workspaces.ask_models import AskCitation, WorkspaceAskRun
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationCitationContextV1,
    ConversationExecutionContextV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)


class ConversationCitationContextError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ResolvedConversationCitation:
    citation: AskCitation
    run_id: str
    workspace_id: str


class ConversationCitationContextService:
    """Persist and resolve ordinal citation references against durable Ask runs."""

    def __init__(
        self,
        *,
        context_repository: ConversationContextRepository,
        ask_repository: WorkspaceAskRepository,
        clock: Callable[[], datetime] | None = None,
        max_conflict_retries: int = 3,
    ) -> None:
        if isinstance(max_conflict_retries, bool) or not 1 <= max_conflict_retries <= 3:
            raise ValueError("max_conflict_retries must be between 1 and 3")
        self._context_repository = context_repository
        self._ask_repository = ask_repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_conflict_retries = max_conflict_retries

    def record_ask_run(
        self,
        *,
        context: ConversationExecutionContextV1,
        run_id: str,
        workspace_id: str,
    ) -> None:
        normalized_run_id = run_id.strip()
        normalized_workspace_id = workspace_id.strip()
        if not normalized_run_id or not normalized_workspace_id:
            return
        now = self._clock()
        replacement = ConversationCitationContextV1(
            tenant_id=context.tenant_id,
            conversation_context_binding_id=context.conversation_context_binding_id,
            workspace_id=normalized_workspace_id,
            last_ask_run_id=normalized_run_id,
            configuration_version=1,
            updated_at=now,
        )
        try:
            current = self._context_repository.get_citation_context(
                tenant_id=context.tenant_id,
                conversation_context_binding_id=context.conversation_context_binding_id,
            )
        except ConversationContextRepositoryError as exc:
            raise ConversationCitationContextError(
                "conversation_context_storage_unavailable"
            ) from exc

        if current is None:
            try:
                if not self._context_repository.put_citation_context_if_absent(replacement):
                    current = self._context_repository.get_citation_context(
                        tenant_id=context.tenant_id,
                        conversation_context_binding_id=context.conversation_context_binding_id,
                    )
            except ConversationContextRepositoryError as exc:
                raise ConversationCitationContextError(
                    "conversation_context_storage_unavailable"
                ) from exc

        if current is None:
            return

        for attempt in range(self._max_conflict_retries):
            next_replacement = replacement.model_copy(
                update={
                    "configuration_version": current.configuration_version + 1,
                    "updated_at": self._clock(),
                }
            )
            try:
                if self._context_repository.replace_citation_context_if_match(
                    expected=current,
                    replacement=next_replacement,
                ):
                    return
            except ConversationContextRepositoryError as exc:
                raise ConversationCitationContextError(
                    "conversation_context_storage_unavailable"
                ) from exc
            try:
                current = self._context_repository.get_citation_context(
                    tenant_id=context.tenant_id,
                    conversation_context_binding_id=context.conversation_context_binding_id,
                )
            except ConversationContextRepositoryError as exc:
                raise ConversationCitationContextError(
                    "conversation_context_storage_unavailable"
                ) from exc
            if current is None:
                return
        raise ConversationCitationContextError("conversation_context_storage_unavailable")

    def resolve_citation(
        self,
        *,
        context: ConversationExecutionContextV1,
        workspace_id: str,
        citation_ordinal: int,
    ) -> ResolvedConversationCitation:
        if citation_ordinal < 1:
            raise ConversationCitationContextError("citation_ordinal_invalid")
        try:
            stored = self._context_repository.get_citation_context(
                tenant_id=context.tenant_id,
                conversation_context_binding_id=context.conversation_context_binding_id,
            )
        except ConversationContextRepositoryError as exc:
            raise ConversationCitationContextError(
                "conversation_context_storage_unavailable"
            ) from exc
        if stored is None or stored.workspace_id != workspace_id:
            raise ConversationCitationContextError("citation_context_not_found")

        run = self._ask_repository.get_run(
            tenant_id=context.tenant_id,
            run_id=stored.last_ask_run_id,
        )
        if run is None:
            raise ConversationCitationContextError("citation_not_available")
        if run.workspace_id != workspace_id:
            raise ConversationCitationContextError("citation_not_available")

        citations = _citations_from_run(run)
        if citation_ordinal > len(citations):
            raise ConversationCitationContextError("citation_ordinal_invalid")
        citation = citations[citation_ordinal - 1]
        if citation.document_id.strip() == "":
            raise ConversationCitationContextError("citation_not_available")
        return ResolvedConversationCitation(
            citation=citation,
            run_id=run.run_id,
            workspace_id=run.workspace_id,
        )


def _citations_from_run(run: WorkspaceAskRun | Any) -> list[AskCitation]:
    raw = getattr(run, "citations", None)
    if not isinstance(raw, list):
        return []
    citations: list[AskCitation] = []
    for item in raw:
        if isinstance(item, AskCitation):
            citations.append(item)
            continue
        try:
            citations.append(AskCitation.model_validate(item))
        except Exception:
            continue
    return citations
