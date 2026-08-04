# © Artur Czarnecki. All rights reserved.

"""Durable personal conversation workspace selection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from re import compile
from typing import NoReturn

from pydantic import BaseModel, ConfigDict, Field, field_validator

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    PersonalConversationStateV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService

_REF_RE = compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_IDENTITY_MISMATCH_ERRORS = frozenset(
    {
        "conversation_context_malformed_record",
        "conversation_context_record_identity_mismatch",
    }
)


class ConversationWorkspaceSelectionError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ConversationWorkspaceSelectionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    conversation_context_binding_id: str
    owner_principal_ref: str
    selected_workspace_id: str
    previous_workspace_id: str | None
    configuration_version: int = Field(..., ge=1)
    changed: bool
    updated_at: datetime

    @field_validator(
        "tenant_id",
        "conversation_context_binding_id",
        "owner_principal_ref",
        "selected_workspace_id",
        "previous_workspace_id",
    )
    @classmethod
    def _validate_reference(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        field_name = getattr(info, "field_name", "reference")
        if not normalized:
            raise ValueError(f"{field_name}_must_be_non_blank")
        if _REF_RE.fullmatch(normalized) is None:
            raise ValueError(f"{field_name}_invalid")
        return normalized

    @field_validator("updated_at")
    @classmethod
    def _validate_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("datetime_must_be_timezone_aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("datetime_must_be_utc")
        return value


class ConversationWorkspaceSelectionService:
    def __init__(
        self,
        repository: ConversationContextRepository,
        managed_workspace_service: ManagedWorkspaceService,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._repository = repository
        self._managed_workspace_service = managed_workspace_service
        self._clock = clock

    def select_personal_workspace(
        self,
        *,
        execution_context: ConversationExecutionContextV1,
        workspace_id: str,
    ) -> ConversationWorkspaceSelectionResultV1:
        if execution_context.audience_mode is not ConversationAudienceMode.PERSONAL:
            raise ConversationWorkspaceSelectionError(
                "workspace_selection_personal_context_required"
            )

        if (
            ConversationProductCapability.WORKSPACE_SELECTION
            not in execution_context.allowed_product_capabilities
        ):
            raise ConversationWorkspaceSelectionError("workspace_selection_not_allowed")

        try:
            workspace = self._managed_workspace_service.get_workspace(
                tenant_id=execution_context.tenant_id,
                workspace_id=workspace_id,
            )
        except Exception as exc:
            raise ConversationWorkspaceSelectionError("workspace_not_found") from exc
        if workspace is None:
            raise ConversationWorkspaceSelectionError("workspace_not_found")

        current = self._get_personal_state(execution_context)
        if current is None:
            return self._create_selection(
                execution_context=execution_context,
                workspace_id=workspace_id,
            )

        if not isinstance(current, PersonalConversationStateV1):
            raise ConversationWorkspaceSelectionError(
                "conversation_context_identity_mismatch"
            )
        if not self._matches_context_identity(current, execution_context):
            raise ConversationWorkspaceSelectionError(
                "conversation_context_identity_mismatch"
            )

        if current.selected_workspace_id == workspace_id:
            return ConversationWorkspaceSelectionResultV1(
                tenant_id=current.tenant_id,
                conversation_context_binding_id=current.conversation_context_binding_id,
                owner_principal_ref=current.owner_principal_ref,
                selected_workspace_id=current.selected_workspace_id,
                previous_workspace_id=current.selected_workspace_id,
                configuration_version=current.configuration_version,
                changed=False,
                updated_at=current.updated_at,
            )

        replacement = PersonalConversationStateV1(
            tenant_id=current.tenant_id,
            conversation_context_binding_id=current.conversation_context_binding_id,
            owner_principal_ref=current.owner_principal_ref,
            selected_workspace_id=workspace_id,
            configuration_version=current.configuration_version + 1,
            updated_at=self._read_clock(),
        )
        result = ConversationWorkspaceSelectionResultV1(
            tenant_id=replacement.tenant_id,
            conversation_context_binding_id=replacement.conversation_context_binding_id,
            owner_principal_ref=replacement.owner_principal_ref,
            selected_workspace_id=replacement.selected_workspace_id,
            previous_workspace_id=current.selected_workspace_id,
            configuration_version=replacement.configuration_version,
            changed=True,
            updated_at=replacement.updated_at,
        )
        if not self._replace_personal_state(current, replacement):
            raise ConversationWorkspaceSelectionError("workspace_selection_conflict")
        return result

    def _create_selection(
        self,
        *,
        execution_context: ConversationExecutionContextV1,
        workspace_id: str,
    ) -> ConversationWorkspaceSelectionResultV1:
        state = PersonalConversationStateV1(
            tenant_id=execution_context.tenant_id,
            conversation_context_binding_id=execution_context.conversation_context_binding_id,
            owner_principal_ref=execution_context.principal_ref,
            selected_workspace_id=workspace_id,
            configuration_version=1,
            updated_at=self._read_clock(),
        )
        result = ConversationWorkspaceSelectionResultV1(
            tenant_id=state.tenant_id,
            conversation_context_binding_id=state.conversation_context_binding_id,
            owner_principal_ref=state.owner_principal_ref,
            selected_workspace_id=state.selected_workspace_id,
            previous_workspace_id=None,
            configuration_version=state.configuration_version,
            changed=True,
            updated_at=state.updated_at,
        )
        if not self._put_personal_state(state):
            raise ConversationWorkspaceSelectionError("workspace_selection_conflict")
        return result

    def _read_clock(self) -> datetime:
        timestamp = self._clock()
        if not isinstance(timestamp, datetime):
            raise ValueError("clock_must_produce_datetime")
        if timestamp.tzinfo is None:
            raise ValueError("clock_must_produce_timezone_aware_utc")
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("clock_must_produce_timezone_aware_utc")
        return timestamp

    @staticmethod
    def _matches_context_identity(
        state: PersonalConversationStateV1,
        execution_context: ConversationExecutionContextV1,
    ) -> bool:
        return (
            state.tenant_id == execution_context.tenant_id
            and state.conversation_context_binding_id
            == execution_context.conversation_context_binding_id
            and state.owner_principal_ref == execution_context.principal_ref
        )

    def _get_personal_state(
        self,
        execution_context: ConversationExecutionContextV1,
    ) -> PersonalConversationStateV1 | None:
        try:
            return self._repository.get_personal_state(
                tenant_id=execution_context.tenant_id,
                conversation_context_binding_id=(
                    execution_context.conversation_context_binding_id
                ),
                owner_principal_ref=execution_context.principal_ref,
            )
        except ConversationContextRepositoryError as exc:
            self._raise_repository_error(exc)
        except Exception as exc:
            raise ConversationWorkspaceSelectionError(
                "conversation_context_storage_unavailable"
            ) from exc

    def _put_personal_state(self, state: PersonalConversationStateV1) -> bool:
        try:
            return self._repository.put_personal_state_if_absent(state)
        except ConversationContextRepositoryError as exc:
            self._raise_repository_error(exc)
        except Exception as exc:
            raise ConversationWorkspaceSelectionError(
                "conversation_context_storage_unavailable"
            ) from exc

    def _replace_personal_state(
        self,
        expected: PersonalConversationStateV1,
        replacement: PersonalConversationStateV1,
    ) -> bool:
        try:
            return self._repository.replace_personal_state_if_match(
                expected=expected,
                replacement=replacement,
            )
        except ConversationContextRepositoryError as exc:
            self._raise_repository_error(exc)
        except Exception as exc:
            raise ConversationWorkspaceSelectionError(
                "conversation_context_storage_unavailable"
            ) from exc

    @staticmethod
    def _raise_repository_error(exc: ConversationContextRepositoryError) -> NoReturn:
        if exc.error_code in _IDENTITY_MISMATCH_ERRORS:
            raise ConversationWorkspaceSelectionError(
                "conversation_context_identity_mismatch"
            ) from exc
        raise ConversationWorkspaceSelectionError(
            "conversation_context_storage_unavailable"
        ) from exc
