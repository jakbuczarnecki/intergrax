# © Artur Czarnecki. All rights reserved.

"""Application-facing bounded Conversation Context thread-memory boundary."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningTurn,
)
from local_workspace_application.workspaces.conversation_context_memory import (
    ConversationThreadMemoryError,
    ConversationThreadMemorySnapshotV1,
    SessionHistorySnapshotConversationThreadMemoryAdapter,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
    ConversationThreadMemoryLimitsV1,
    ConversationThreadMemoryMessageRole,
    ConversationThreadMemoryMessageV1,
)


class ConversationThreadMemoryServiceError(RuntimeError):
    """Stable application-facing memory failure."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ConversationThreadMemoryService:
    """Converts the canonical adapter model into safe planner turns."""

    def __init__(
        self,
        *,
        adapter: SessionHistorySnapshotConversationThreadMemoryAdapter,
        limits: ConversationThreadMemoryLimitsV1,
        clock: Callable[[], datetime] | None = None,
        max_conflict_retries: int = 3,
    ) -> None:
        if isinstance(max_conflict_retries, bool) or not 1 <= max_conflict_retries <= 3:
            raise ValueError("max_conflict_retries must be between 1 and 3")
        self._adapter = adapter
        self._limits = limits
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_conflict_retries = max_conflict_retries

    def load_recent_turns(
        self,
        *,
        context: ConversationExecutionContextV1,
        now: datetime | None = None,
    ) -> tuple[ConversationPlanningTurn, ...]:
        observed_at = now or self._clock()
        try:
            messages = self._adapter.load_bounded_history_from_port(
                context=context,
                limits=self._limits,
                now=observed_at,
            )
        except ConversationThreadMemoryError as exc:
            raise ConversationThreadMemoryServiceError(
                "conversation_thread_memory_load_failed"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - storage boundary normalization
            raise ConversationThreadMemoryServiceError(
                "conversation_thread_memory_load_failed"
            ) from exc

        turns: list[ConversationPlanningTurn] = []
        for message in messages:
            if message.role not in {
                ConversationThreadMemoryMessageRole.USER,
                ConversationThreadMemoryMessageRole.ASSISTANT,
            }:
                continue
            role = (
                "user"
                if message.role is ConversationThreadMemoryMessageRole.USER
                else "assistant"
            )
            turns.append(
                ConversationPlanningTurn(
                    role=role,
                    text=message.content,
                )
            )
        return tuple(turns)

    def append_exchange(
        self,
        *,
        context: ConversationExecutionContextV1,
        user_text: str,
        assistant_text: str,
        user_created_at: datetime,
        assistant_created_at: datetime,
        exchange_id: str,
    ) -> ConversationThreadMemorySnapshotV1:
        if assistant_created_at < user_created_at:
            raise ValueError("assistant_created_at must not precede user_created_at")
        user_message = ConversationThreadMemoryMessageV1(
            role=ConversationThreadMemoryMessageRole.USER,
            content=user_text,
            created_at=user_created_at,
        )
        assistant_message = ConversationThreadMemoryMessageV1(
            role=ConversationThreadMemoryMessageRole.ASSISTANT,
            content=assistant_text,
            created_at=assistant_created_at,
        )
        for attempt in range(self._max_conflict_retries):
            try:
                return self._adapter.append_exchange(
                    context=context,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    exchange_id=exchange_id,
                )
            except ConversationThreadMemoryError as exc:
                if (
                    exc.error_code != "THREAD_MEMORY_REVISION_CONFLICT"
                    or attempt + 1 >= self._max_conflict_retries
                ):
                    if exc.error_code == "THREAD_MEMORY_REVISION_CONFLICT":
                        raise ConversationThreadMemoryServiceError(
                            "conversation_thread_memory_conflict"
                        ) from exc
                    raise ConversationThreadMemoryServiceError(
                        "conversation_thread_memory_append_failed"
                    ) from exc
            except Exception as exc:  # noqa: BLE001 - storage boundary normalization
                raise ConversationThreadMemoryServiceError(
                    "conversation_thread_memory_append_failed"
                ) from exc
        raise AssertionError("unreachable memory conflict loop")


__all__ = [
    "ConversationThreadMemoryService",
    "ConversationThreadMemoryServiceError",
]
