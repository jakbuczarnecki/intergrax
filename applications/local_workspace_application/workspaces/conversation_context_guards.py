# © Artur Czarnecki. All rights reserved.

"""Deterministic conversation evidence and execution guards (LKW-CONVERSATION-CONTEXT-1B2)."""

from __future__ import annotations

import hashlib
import json

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationApprovedModelInputV1,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationExecutionGuardReceiptV1,
    ConversationModelInputKindV1,
    ConversationOutboundGuardReceiptV1,
    ConversationOutboundTargetV1,
    ConversationProductCapability,
    ConversationScopedModelInputV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
)

_EXECUTION_GUARD_SCHEMA = "lkw.conversation_execution_guard.v1"
_OUTBOUND_GUARD_SCHEMA = "lkw.conversation_outbound_guard.v1"
_EXECUTION_RECEIPT_PREFIX = "lkw-conversation-guard:v1:"
_OUTBOUND_RECEIPT_PREFIX = "lkw-conversation-outbound-guard:v1:"


class ConversationExecutionGuardError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _raise_guard(error_code: str) -> None:
    raise ConversationExecutionGuardError(error_code)


def _approved_input_payload(
    approved: ConversationApprovedModelInputV1,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "audience_eligibility": approved.audience_eligibility.value,
        "input_id": approved.input_id,
        "input_kind": approved.input_kind.value,
    }
    if approved.origin_audience_mode is not None:
        payload["origin_audience_mode"] = approved.origin_audience_mode.value
    return payload


def _derive_receipt_id(*, prefix: str, payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def _execution_receipt_payload(
    *,
    tenant_id: str,
    conversation_context_binding_id: str,
    audience_mode: ConversationAudienceMode,
    workspace_id: str,
    canonical_thread_ref: str,
    requested_capability: ConversationProductCapability,
    approved_inputs: tuple[ConversationApprovedModelInputV1, ...],
) -> dict[str, object]:
    return {
        "approved_inputs": [_approved_input_payload(item) for item in approved_inputs],
        "audience_mode": audience_mode.value,
        "canonical_thread_ref": canonical_thread_ref,
        "conversation_context_binding_id": conversation_context_binding_id,
        "requested_capability": requested_capability.value,
        "schema_version": _EXECUTION_GUARD_SCHEMA,
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
    }


def _outbound_receipt_payload(
    *,
    execution_receipt_id: str,
    tenant_id: str,
    conversation_context_binding_id: str,
    audience_mode: ConversationAudienceMode,
    workspace_id: str,
    canonical_thread_ref: str,
    used_input_ids: tuple[str, ...],
    citation_input_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "audience_mode": audience_mode.value,
        "canonical_thread_ref": canonical_thread_ref,
        "citation_input_ids": list(citation_input_ids),
        "conversation_context_binding_id": conversation_context_binding_id,
        "execution_receipt_id": execution_receipt_id,
        "schema_version": _OUTBOUND_GUARD_SCHEMA,
        "tenant_id": tenant_id,
        "used_input_ids": list(used_input_ids),
        "workspace_id": workspace_id,
    }


def _build_approved_input(
    model_input: ConversationScopedModelInputV1,
) -> ConversationApprovedModelInputV1:
    origin_audience_mode = model_input.origin_audience_mode
    if model_input.input_kind is not ConversationModelInputKindV1.THREAD_MEMORY:
        origin_audience_mode = None
    return ConversationApprovedModelInputV1(
        input_id=model_input.input_id,
        input_kind=model_input.input_kind,
        audience_eligibility=model_input.audience_eligibility,
        origin_audience_mode=origin_audience_mode,
    )


def _verify_execution_receipt_integrity(
    receipt: ConversationExecutionGuardReceiptV1,
) -> None:
    if receipt.schema_version != _EXECUTION_GUARD_SCHEMA:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    expected_id = _derive_receipt_id(
        prefix=_EXECUTION_RECEIPT_PREFIX,
        payload=_execution_receipt_payload(
            tenant_id=receipt.tenant_id,
            conversation_context_binding_id=receipt.conversation_context_binding_id,
            audience_mode=receipt.audience_mode,
            workspace_id=receipt.workspace_id,
            canonical_thread_ref=receipt.canonical_thread_ref,
            requested_capability=receipt.requested_capability,
            approved_inputs=receipt.approved_inputs,
        ),
    )
    if receipt.receipt_id != expected_id:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")


def _verify_outbound_receipt_integrity(
    receipt: ConversationOutboundGuardReceiptV1,
) -> None:
    if receipt.schema_version != _OUTBOUND_GUARD_SCHEMA:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    expected_id = _derive_receipt_id(
        prefix=_OUTBOUND_RECEIPT_PREFIX,
        payload=_outbound_receipt_payload(
            execution_receipt_id=receipt.execution_receipt_id,
            tenant_id=receipt.tenant_id,
            conversation_context_binding_id=receipt.conversation_context_binding_id,
            audience_mode=receipt.audience_mode,
            workspace_id=receipt.workspace_id,
            canonical_thread_ref=receipt.canonical_thread_ref,
            used_input_ids=receipt.used_input_ids,
            citation_input_ids=receipt.citation_input_ids,
        ),
    )
    if receipt.receipt_id != expected_id:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")


def _validate_unique_ids(
    input_ids: tuple[str, ...],
    *,
    duplicate_error: str,
) -> None:
    seen: set[str] = set()
    for input_id in input_ids:
        if input_id in seen:
            _raise_guard(duplicate_error)
        seen.add(input_id)


def authorize_conversation_model_inputs(
    *,
    context: ConversationExecutionContextV1,
    requested_capability: ConversationProductCapability,
    inputs: tuple[ConversationScopedModelInputV1, ...],
) -> ConversationExecutionGuardReceiptV1:
    if requested_capability not in context.allowed_product_capabilities:
        _raise_guard("CONVERSATION_GUARD_CAPABILITY_NOT_ALLOWED")
    if context.audience_mode is ConversationAudienceMode.SHARED:
        if requested_capability is not ConversationProductCapability.READ_ONLY_ASK:
            _raise_guard("CONVERSATION_GUARD_CAPABILITY_NOT_ALLOWED")

    approved_inputs: list[ConversationApprovedModelInputV1] = []
    seen_input_ids: set[str] = set()

    for model_input in inputs:
        if model_input.input_id in seen_input_ids:
            _raise_guard("CONVERSATION_GUARD_DUPLICATE_INPUT")
        seen_input_ids.add(model_input.input_id)

        if model_input.tenant_id != context.tenant_id:
            _raise_guard("CONVERSATION_GUARD_TENANT_MISMATCH")
        if model_input.workspace_id != context.workspace_id:
            _raise_guard("CONVERSATION_GUARD_WORKSPACE_MISMATCH")
        if model_input.source_active is not True:
            _raise_guard("CONVERSATION_GUARD_SOURCE_INACTIVE")

        if context.audience_mode is ConversationAudienceMode.SHARED:
            if (
                model_input.audience_eligibility
                is not KnowledgeAudienceEligibilityV1.SHARED_ALLOWED
            ):
                _raise_guard("CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED")

        if model_input.input_kind is ConversationModelInputKindV1.THREAD_MEMORY:
            if model_input.origin_audience_mode != context.audience_mode:
                _raise_guard("CONVERSATION_GUARD_MEMORY_AUDIENCE_MISMATCH")
            if (
                model_input.conversation_context_binding_id
                != context.conversation_context_binding_id
            ):
                _raise_guard("CONVERSATION_GUARD_MEMORY_BINDING_MISMATCH")
            if model_input.canonical_thread_ref != context.canonical_thread_ref:
                _raise_guard("CONVERSATION_GUARD_MEMORY_THREAD_MISMATCH")

        approved_inputs.append(_build_approved_input(model_input))

    approved_tuple = tuple(approved_inputs)
    receipt_payload = _execution_receipt_payload(
        tenant_id=context.tenant_id,
        conversation_context_binding_id=context.conversation_context_binding_id,
        audience_mode=context.audience_mode,
        workspace_id=context.workspace_id,
        canonical_thread_ref=context.canonical_thread_ref,
        requested_capability=requested_capability,
        approved_inputs=approved_tuple,
    )
    receipt_id = _derive_receipt_id(
        prefix=_EXECUTION_RECEIPT_PREFIX,
        payload=receipt_payload,
    )
    return ConversationExecutionGuardReceiptV1(
        receipt_id=receipt_id,
        tenant_id=context.tenant_id,
        conversation_context_binding_id=context.conversation_context_binding_id,
        audience_mode=context.audience_mode,
        workspace_id=context.workspace_id,
        canonical_thread_ref=context.canonical_thread_ref,
        requested_capability=requested_capability,
        approved_inputs=approved_tuple,
    )


def authorize_conversation_outbound_delivery(
    *,
    context: ConversationExecutionContextV1,
    execution_receipt: ConversationExecutionGuardReceiptV1,
    target: ConversationOutboundTargetV1,
    used_input_ids: tuple[str, ...],
    citation_input_ids: tuple[str, ...],
) -> ConversationOutboundGuardReceiptV1:
    if target.tenant_id != context.tenant_id:
        _raise_guard("CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH")
    if target.conversation_context_binding_id != context.conversation_context_binding_id:
        _raise_guard("CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH")
    if target.audience_mode != context.audience_mode:
        _raise_guard("CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH")
    if target.workspace_id != context.workspace_id:
        _raise_guard("CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH")
    if target.canonical_thread_ref != context.canonical_thread_ref:
        _raise_guard("CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH")

    _verify_execution_receipt_integrity(execution_receipt)

    if execution_receipt.tenant_id != context.tenant_id:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    if execution_receipt.conversation_context_binding_id != context.conversation_context_binding_id:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    if execution_receipt.audience_mode != context.audience_mode:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    if execution_receipt.workspace_id != context.workspace_id:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    if execution_receipt.canonical_thread_ref != context.canonical_thread_ref:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")
    if execution_receipt.requested_capability not in context.allowed_product_capabilities:
        _raise_guard("CONVERSATION_GUARD_RECEIPT_INVALID")

    _validate_unique_ids(
        used_input_ids,
        duplicate_error="CONVERSATION_GUARD_DUPLICATE_USED_INPUT",
    )
    _validate_unique_ids(
        citation_input_ids,
        duplicate_error="CONVERSATION_GUARD_DUPLICATE_CITATION",
    )

    approved_by_id = {
        approved.input_id: approved for approved in execution_receipt.approved_inputs
    }

    for used_input_id in used_input_ids:
        if used_input_id not in approved_by_id:
            _raise_guard("CONVERSATION_GUARD_INPUT_NOT_APPROVED")

    for citation_input_id in citation_input_ids:
        if citation_input_id not in approved_by_id:
            _raise_guard("CONVERSATION_GUARD_CITATION_NOT_APPROVED")
        if citation_input_id not in used_input_ids:
            _raise_guard("CONVERSATION_GUARD_CITATION_NOT_USED")

    if context.audience_mode is ConversationAudienceMode.SHARED:
        for used_input_id in used_input_ids:
            approved = approved_by_id[used_input_id]
            if (
                approved.audience_eligibility
                is not KnowledgeAudienceEligibilityV1.SHARED_ALLOWED
            ):
                _raise_guard("CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED")
            if approved.input_kind is ConversationModelInputKindV1.THREAD_MEMORY:
                if approved.origin_audience_mode is ConversationAudienceMode.PERSONAL:
                    _raise_guard("CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED")

    outbound_payload = _outbound_receipt_payload(
        execution_receipt_id=execution_receipt.receipt_id,
        tenant_id=target.tenant_id,
        conversation_context_binding_id=target.conversation_context_binding_id,
        audience_mode=target.audience_mode,
        workspace_id=target.workspace_id,
        canonical_thread_ref=target.canonical_thread_ref,
        used_input_ids=used_input_ids,
        citation_input_ids=citation_input_ids,
    )
    outbound_receipt_id = _derive_receipt_id(
        prefix=_OUTBOUND_RECEIPT_PREFIX,
        payload=outbound_payload,
    )
    outbound_receipt = ConversationOutboundGuardReceiptV1(
        receipt_id=outbound_receipt_id,
        execution_receipt_id=execution_receipt.receipt_id,
        tenant_id=target.tenant_id,
        conversation_context_binding_id=target.conversation_context_binding_id,
        audience_mode=target.audience_mode,
        workspace_id=target.workspace_id,
        canonical_thread_ref=target.canonical_thread_ref,
        used_input_ids=used_input_ids,
        citation_input_ids=citation_input_ids,
    )
    _verify_outbound_receipt_integrity(outbound_receipt)
    return outbound_receipt
