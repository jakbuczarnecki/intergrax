# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for Conversation Context (LKW-CONVERSATION-CONTEXT-1A)."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationContextBindingV1,
    PersonalConversationStateV1,
    WorkspaceConversationAudiencePolicyV1,
)

_ENTITY_BINDING = "conversation_context_binding"
_ENTITY_WORKSPACE_AUDIENCE = "conversation_context_workspace_audience"
_ENTITY_PERSONAL_STATE = "conversation_context_personal_state"
_SEMANTIC_IDENTITY_SEPARATOR = "\x1e"
_BINDING_SCAN_LIMIT = 100


class ConversationContextRepositoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _partition(tenant_id: str, entity: str) -> str:
    return f"lkw.conversation_context:{tenant_id}:{entity}"


def _binding_row_key(
    *,
    conversation_connection_ref: str,
    opaque_conversation_ref: str,
    conversation_context_binding_id: str,
) -> str:
    return (
        f"{conversation_connection_ref}{_SEMANTIC_IDENTITY_SEPARATOR}"
        f"{opaque_conversation_ref}{_SEMANTIC_IDENTITY_SEPARATOR}"
        f"{conversation_context_binding_id}"
    )


def _binding_semantic_prefix(
    *,
    conversation_connection_ref: str,
    opaque_conversation_ref: str,
) -> str:
    return (
        f"{conversation_connection_ref}{_SEMANTIC_IDENTITY_SEPARATOR}"
        f"{opaque_conversation_ref}{_SEMANTIC_IDENTITY_SEPARATOR}"
    )


def _binding_id_from_row_key(
    *,
    row_key: str,
    conversation_connection_ref: str,
    opaque_conversation_ref: str,
) -> str:
    prefix = _binding_semantic_prefix(
        conversation_connection_ref=conversation_connection_ref,
        opaque_conversation_ref=opaque_conversation_ref,
    )
    if not row_key.startswith(prefix):
        raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
    return row_key[len(prefix) :]


def _assert_binding_record_identity(
    binding: ConversationContextBindingV1,
    *,
    tenant_id: str,
    conversation_connection_ref: str,
    opaque_conversation_ref: str,
    conversation_context_binding_id: str | None = None,
    row_key: str | None = None,
) -> None:
    if binding.tenant_id != tenant_id:
        raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
    if binding.conversation_connection_ref != conversation_connection_ref:
        raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
    if binding.opaque_conversation_ref != opaque_conversation_ref:
        raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
    if (
        conversation_context_binding_id is not None
        and binding.conversation_context_binding_id != conversation_context_binding_id
    ):
        raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
    if row_key is not None:
        row_key_binding_id = _binding_id_from_row_key(
            row_key=row_key,
            conversation_connection_ref=conversation_connection_ref,
            opaque_conversation_ref=opaque_conversation_ref,
        )
        if row_key_binding_id != binding.conversation_context_binding_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")


def _assert_binding_immutable_fields_unchanged(
    expected: ConversationContextBindingV1,
    replacement: ConversationContextBindingV1,
) -> None:
    if (
        expected.conversation_connection_ref != replacement.conversation_connection_ref
        or expected.opaque_conversation_ref != replacement.opaque_conversation_ref
        or expected.frontend_provider_id != replacement.frontend_provider_id
        or expected.audience_mode != replacement.audience_mode
        or expected.created_at != replacement.created_at
    ):
        raise ConversationContextRepositoryError(
            "conversation_context_binding_immutable_field_changed"
        )


def _personal_state_row_key(
    *,
    conversation_context_binding_id: str,
    owner_principal_ref: str,
) -> str:
    return (
        f"{conversation_context_binding_id}{_SEMANTIC_IDENTITY_SEPARATOR}"
        f"{owner_principal_ref}"
    )


def _to_document_record(
    model: BaseModel,
    *,
    partition_key: str,
    row_key: str,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=partition_key,
        row_key=row_key,
        data=model.model_dump(mode="json"),
    )


class ConversationContextRepository:
    """Tenant-scoped Conversation Context persistence over DocumentStore."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def _require_conditional_store(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConversationContextRepositoryError("conversation_context_conditional_store_required")
        return self._store

    def _put_if_absent(self, model: BaseModel, *, partition_key: str, row_key: str) -> bool:
        store = self._require_conditional_store()
        return store.put_if_absent(
            _to_document_record(model, partition_key=partition_key, row_key=row_key)
        )

    def _replace_if_match(
        self,
        *,
        expected: BaseModel,
        replacement: BaseModel,
        partition_key: str,
        row_key: str,
    ) -> bool:
        store = self._require_conditional_store()
        return store.replace_if_match(
            expected=_to_document_record(expected, partition_key=partition_key, row_key=row_key),
            replacement=_to_document_record(
                replacement,
                partition_key=partition_key,
                row_key=row_key,
            ),
        )

    def _parse_model(self, model_type: type[BaseModel], data: object) -> BaseModel:
        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            raise ConversationContextRepositoryError("conversation_context_malformed_record") from exc

    def _parse_binding(self, data: object) -> ConversationContextBindingV1:
        model = self._parse_model(ConversationContextBindingV1, data)
        if not isinstance(model, ConversationContextBindingV1):
            raise ConversationContextRepositoryError("conversation_context_malformed_record")
        return model

    def _parse_workspace_audience_policy(
        self,
        data: object,
    ) -> WorkspaceConversationAudiencePolicyV1:
        model = self._parse_model(WorkspaceConversationAudiencePolicyV1, data)
        if not isinstance(model, WorkspaceConversationAudiencePolicyV1):
            raise ConversationContextRepositoryError("conversation_context_malformed_record")
        return model

    def _parse_personal_state(self, data: object) -> PersonalConversationStateV1:
        model = self._parse_model(PersonalConversationStateV1, data)
        if not isinstance(model, PersonalConversationStateV1):
            raise ConversationContextRepositoryError("conversation_context_malformed_record")
        return model

    def put_binding_if_absent(self, binding: ConversationContextBindingV1) -> bool:
        partition_key = _partition(binding.tenant_id, _ENTITY_BINDING)
        row_key = _binding_row_key(
            conversation_connection_ref=binding.conversation_connection_ref,
            opaque_conversation_ref=binding.opaque_conversation_ref,
            conversation_context_binding_id=binding.conversation_context_binding_id,
        )
        return self._put_if_absent(binding, partition_key=partition_key, row_key=row_key)

    def get_binding(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
        opaque_conversation_ref: str,
        conversation_context_binding_id: str,
    ) -> ConversationContextBindingV1 | None:
        partition_key = _partition(tenant_id, _ENTITY_BINDING)
        row_key = _binding_row_key(
            conversation_connection_ref=conversation_connection_ref,
            opaque_conversation_ref=opaque_conversation_ref,
            conversation_context_binding_id=conversation_context_binding_id,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        binding = self._parse_binding(dict(record.data))
        _assert_binding_record_identity(
            binding,
            tenant_id=tenant_id,
            conversation_connection_ref=conversation_connection_ref,
            opaque_conversation_ref=opaque_conversation_ref,
            conversation_context_binding_id=conversation_context_binding_id,
            row_key=row_key,
        )
        return binding

    def replace_binding_if_match(
        self,
        *,
        expected: ConversationContextBindingV1,
        replacement: ConversationContextBindingV1,
    ) -> bool:
        if expected.conversation_context_binding_id != replacement.conversation_context_binding_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        if expected.tenant_id != replacement.tenant_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        if replacement.configuration_version != expected.configuration_version + 1:
            raise ConversationContextRepositoryError(
                "conversation_context_configuration_version_invalid"
            )
        _assert_binding_immutable_fields_unchanged(expected, replacement)
        partition_key = _partition(expected.tenant_id, _ENTITY_BINDING)
        row_key = _binding_row_key(
            conversation_connection_ref=expected.conversation_connection_ref,
            opaque_conversation_ref=expected.opaque_conversation_ref,
            conversation_context_binding_id=expected.conversation_context_binding_id,
        )
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def list_bindings_for_semantic_identity(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
        opaque_conversation_ref: str,
    ) -> list[ConversationContextBindingV1]:
        partition_key = _partition(tenant_id, _ENTITY_BINDING)
        prefix = _binding_semantic_prefix(
            conversation_connection_ref=conversation_connection_ref,
            opaque_conversation_ref=opaque_conversation_ref,
        )
        result = self._store.query(
            partition_key,
            limit=_BINDING_SCAN_LIMIT + 1,
            row_key_prefix=prefix,
        )
        if len(result.documents) > _BINDING_SCAN_LIMIT:
            raise ConversationContextRepositoryError("conversation_context_binding_scan_limit_exceeded")
        bindings: list[ConversationContextBindingV1] = []
        for doc in result.documents:
            binding = self._parse_binding(dict(doc.data))
            _assert_binding_record_identity(
                binding,
                tenant_id=tenant_id,
                conversation_connection_ref=conversation_connection_ref,
                opaque_conversation_ref=opaque_conversation_ref,
                row_key=doc.row_key,
            )
            bindings.append(binding)
        return bindings

    def put_workspace_audience_policy_if_absent(
        self,
        policy: WorkspaceConversationAudiencePolicyV1,
    ) -> bool:
        partition_key = _partition(policy.tenant_id, _ENTITY_WORKSPACE_AUDIENCE)
        row_key = policy.workspace_id
        return self._put_if_absent(policy, partition_key=partition_key, row_key=row_key)

    def get_workspace_audience_policy(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceConversationAudiencePolicyV1 | None:
        partition_key = _partition(tenant_id, _ENTITY_WORKSPACE_AUDIENCE)
        record = self._store.get(partition_key, workspace_id)
        if record is None:
            return None
        policy = self._parse_workspace_audience_policy(dict(record.data))
        if policy.tenant_id != tenant_id or policy.workspace_id != workspace_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        return policy

    def replace_workspace_audience_policy_if_match(
        self,
        *,
        expected: WorkspaceConversationAudiencePolicyV1,
        replacement: WorkspaceConversationAudiencePolicyV1,
    ) -> bool:
        if expected.tenant_id != replacement.tenant_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        if expected.workspace_id != replacement.workspace_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_WORKSPACE_AUDIENCE)
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=expected.workspace_id,
        )

    def put_personal_state_if_absent(self, state: PersonalConversationStateV1) -> bool:
        partition_key = _partition(state.tenant_id, _ENTITY_PERSONAL_STATE)
        row_key = _personal_state_row_key(
            conversation_context_binding_id=state.conversation_context_binding_id,
            owner_principal_ref=state.owner_principal_ref,
        )
        return self._put_if_absent(state, partition_key=partition_key, row_key=row_key)

    def get_personal_state(
        self,
        *,
        tenant_id: str,
        conversation_context_binding_id: str,
        owner_principal_ref: str,
    ) -> PersonalConversationStateV1 | None:
        partition_key = _partition(tenant_id, _ENTITY_PERSONAL_STATE)
        row_key = _personal_state_row_key(
            conversation_context_binding_id=conversation_context_binding_id,
            owner_principal_ref=owner_principal_ref,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        state = self._parse_personal_state(dict(record.data))
        if (
            state.tenant_id != tenant_id
            or state.conversation_context_binding_id != conversation_context_binding_id
            or state.owner_principal_ref != owner_principal_ref
        ):
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        return state

    def replace_personal_state_if_match(
        self,
        *,
        expected: PersonalConversationStateV1,
        replacement: PersonalConversationStateV1,
    ) -> bool:
        if expected.tenant_id != replacement.tenant_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        if expected.conversation_context_binding_id != replacement.conversation_context_binding_id:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        if expected.owner_principal_ref != replacement.owner_principal_ref:
            raise ConversationContextRepositoryError("conversation_context_record_identity_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_PERSONAL_STATE)
        row_key = _personal_state_row_key(
            conversation_context_binding_id=expected.conversation_context_binding_id,
            owner_principal_ref=expected.owner_principal_ref,
        )
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def delete_personal_state(
        self,
        *,
        tenant_id: str,
        conversation_context_binding_id: str,
        owner_principal_ref: str,
    ) -> None:
        partition_key = _partition(tenant_id, _ENTITY_PERSONAL_STATE)
        row_key = _personal_state_row_key(
            conversation_context_binding_id=conversation_context_binding_id,
            owner_principal_ref=owner_principal_ref,
        )
        self._store.delete(partition_key, row_key)
