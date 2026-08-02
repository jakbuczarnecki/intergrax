# © Artur Czarnecki. All rights reserved.

"""Unit tests for conversation evidence and execution guards."""

from __future__ import annotations

import pytest

from local_workspace_application.workspaces.conversation_context_guards import (
    ConversationExecutionGuardError,
    _EXECUTION_RECEIPT_PREFIX,
    _derive_receipt_id,
    _execution_receipt_payload,
    authorize_conversation_model_inputs,
    authorize_conversation_outbound_delivery,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationApprovedModelInputV1,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationExecutionGuardReceiptV1,
    ConversationModelInputKindV1,
    ConversationOutboundGuardReceiptV1,
    ConversationOutboundTargetV1,
    ConversationProductCapability,
    ConversationScopedModelInputV1,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_BINDING = "binding-1"
_THREAD = "thread.alpha"
_PRINCIPAL = "principal.alice"


def _personal_context(
    **overrides: object,
) -> ConversationExecutionContextV1:
    payload = {
        "tenant_id": _TENANT,
        "conversation_context_binding_id": _BINDING,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_id": _WORKSPACE,
        "principal_ref": _PRINCIPAL,
        "canonical_thread_ref": _THREAD,
        "activation_policy": ConversationActivationPolicy.ALWAYS,
        "thread_context_policy": ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        "allowed_product_capabilities": frozenset(
            {
                ConversationProductCapability.READ_ONLY_ASK,
                ConversationProductCapability.SOURCE_DISCOVERY,
            }
        ),
    }
    payload.update(overrides)
    return ConversationExecutionContextV1(**payload)  # type: ignore[arg-type]


def _shared_context(**overrides: object) -> ConversationExecutionContextV1:
    return _personal_context(
        audience_mode=ConversationAudienceMode.SHARED,
        allowed_product_capabilities=frozenset({ConversationProductCapability.READ_ONLY_ASK}),
        **overrides,
    )


def _scoped_input(**overrides: object) -> ConversationScopedModelInputV1:
    payload = {
        "input_id": "input-1",
        "input_kind": ConversationModelInputKindV1.INDEXED_EVIDENCE,
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "audience_eligibility": KnowledgeAudienceEligibilityV1.PERSONAL_ONLY,
        "source_active": True,
        "source_ref": "source.ref",
    }
    payload.update(overrides)
    return ConversationScopedModelInputV1(**payload)  # type: ignore[arg-type]


def _memory_input(**overrides: object) -> ConversationScopedModelInputV1:
    payload = {
        "input_kind": ConversationModelInputKindV1.THREAD_MEMORY,
        "origin_audience_mode": ConversationAudienceMode.PERSONAL,
        "conversation_context_binding_id": _BINDING,
        "canonical_thread_ref": _THREAD,
    }
    payload.update(overrides)
    return _scoped_input(**payload)


def _outbound_target(**overrides: object) -> ConversationOutboundTargetV1:
    payload = {
        "tenant_id": _TENANT,
        "conversation_context_binding_id": _BINDING,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_id": _WORKSPACE,
        "canonical_thread_ref": _THREAD,
    }
    payload.update(overrides)
    return ConversationOutboundTargetV1(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("context_factory", "capability", "inputs", "error_code"),
    [
        (
            _personal_context,
            ConversationProductCapability.READ_ONLY_ASK,
            (_scoped_input(),),
            None,
        ),
        (
            _personal_context,
            ConversationProductCapability.WORKSPACE_DISCOVERY,
            (),
            "CONVERSATION_GUARD_CAPABILITY_NOT_ALLOWED",
        ),
        (
            _shared_context,
            ConversationProductCapability.READ_ONLY_ASK,
            (_scoped_input(audience_eligibility=KnowledgeAudienceEligibilityV1.SHARED_ALLOWED),),
            None,
        ),
        (
            _shared_context,
            ConversationProductCapability.SOURCE_DISCOVERY,
            (),
            "CONVERSATION_GUARD_CAPABILITY_NOT_ALLOWED",
        ),
    ],
    ids=[
        "personal-allowed-capability",
        "personal-missing-capability",
        "shared-read-only-ask",
        "shared-mutation-capability",
    ],
)
def test_capability_guard(
    context_factory: object,
    capability: ConversationProductCapability,
    inputs: tuple[ConversationScopedModelInputV1, ...],
    error_code: str | None,
) -> None:
    context = context_factory()  # type: ignore[operator]
    if error_code is None:
        receipt = authorize_conversation_model_inputs(
            context=context,
            requested_capability=capability,
            inputs=inputs,
        )
        assert receipt.requested_capability is capability
        return
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_model_inputs(
            context=context,
            requested_capability=capability,
            inputs=inputs,
        )
    assert exc_info.value.error_code == error_code


@pytest.mark.parametrize(
    ("context_factory", "input_overrides", "error_code"),
    [
        (
            _personal_context,
            {},
            None,
        ),
        (
            _shared_context,
            {"audience_eligibility": KnowledgeAudienceEligibilityV1.SHARED_ALLOWED},
            None,
        ),
        (
            _personal_context,
            {"tenant_id": "tenant-other"},
            "CONVERSATION_GUARD_TENANT_MISMATCH",
        ),
        (
            _personal_context,
            {"workspace_id": "workspace-other"},
            "CONVERSATION_GUARD_WORKSPACE_MISMATCH",
        ),
        (
            _personal_context,
            {"source_active": False},
            "CONVERSATION_GUARD_SOURCE_INACTIVE",
        ),
        (
            _shared_context,
            {"audience_eligibility": KnowledgeAudienceEligibilityV1.PERSONAL_ONLY},
            "CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED",
        ),
    ],
    ids=[
        "personal-active-input",
        "shared-shared-allowed-input",
        "tenant-mismatch",
        "workspace-mismatch",
        "inactive-source",
        "shared-personal-only-input",
    ],
)
def test_evidence_scope_guard(
    context_factory: object,
    input_overrides: dict[str, object],
    error_code: str | None,
) -> None:
    context = context_factory()  # type: ignore[operator]
    model_input = _scoped_input(**input_overrides)
    if error_code is None:
        receipt = authorize_conversation_model_inputs(
            context=context,
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(model_input,),
        )
        assert len(receipt.approved_inputs) == 1
        return
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_model_inputs(
            context=context,
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(model_input,),
        )
    assert exc_info.value.error_code == error_code


def test_duplicate_input_id_fails() -> None:
    duplicate = _scoped_input(input_id="dup-1")
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_model_inputs(
            context=_personal_context(),
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(duplicate, duplicate),
        )
    assert exc_info.value.error_code == "CONVERSATION_GUARD_DUPLICATE_INPUT"


def test_empty_input_list_is_valid() -> None:
    receipt = authorize_conversation_model_inputs(
        context=_personal_context(),
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(),
    )
    assert receipt.approved_inputs == ()


def test_input_order_is_preserved() -> None:
    first = _scoped_input(input_id="input-a")
    second = _scoped_input(input_id="input-b")
    receipt = authorize_conversation_model_inputs(
        context=_personal_context(),
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(first, second),
    )
    assert tuple(item.input_id for item in receipt.approved_inputs) == ("input-a", "input-b")


def test_reordered_inputs_produce_different_receipt_id() -> None:
    first = _scoped_input(input_id="input-a")
    second = _scoped_input(input_id="input-b")
    receipt_ab = authorize_conversation_model_inputs(
        context=_personal_context(),
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(first, second),
    )
    receipt_ba = authorize_conversation_model_inputs(
        context=_personal_context(),
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(second, first),
    )
    assert receipt_ab.receipt_id != receipt_ba.receipt_id


def test_caller_private_permission_cannot_expand_shared_eligibility() -> None:
    personal_only = _scoped_input(
        audience_eligibility=KnowledgeAudienceEligibilityV1.PERSONAL_ONLY,
    )
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_model_inputs(
            context=_shared_context(),
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(personal_only,),
        )
    assert exc_info.value.error_code == "CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("context_factory", "memory_overrides", "error_code"),
    [
        (
            _personal_context,
            {"origin_audience_mode": ConversationAudienceMode.PERSONAL},
            None,
        ),
        (
            _shared_context,
            {
                "origin_audience_mode": ConversationAudienceMode.SHARED,
                "audience_eligibility": KnowledgeAudienceEligibilityV1.SHARED_ALLOWED,
            },
            None,
        ),
        (
            _shared_context,
            {
                "origin_audience_mode": ConversationAudienceMode.PERSONAL,
                "audience_eligibility": KnowledgeAudienceEligibilityV1.SHARED_ALLOWED,
            },
            "CONVERSATION_GUARD_MEMORY_AUDIENCE_MISMATCH",
        ),
        (
            _personal_context,
            {
                "origin_audience_mode": ConversationAudienceMode.SHARED,
                "audience_eligibility": KnowledgeAudienceEligibilityV1.PERSONAL_ONLY,
            },
            "CONVERSATION_GUARD_MEMORY_AUDIENCE_MISMATCH",
        ),
        (
            _personal_context,
            {"conversation_context_binding_id": "binding-other"},
            "CONVERSATION_GUARD_MEMORY_BINDING_MISMATCH",
        ),
        (
            _personal_context,
            {"canonical_thread_ref": "thread-other"},
            "CONVERSATION_GUARD_MEMORY_THREAD_MISMATCH",
        ),
    ],
    ids=[
        "personal-memory-in-personal",
        "shared-memory-in-shared",
        "personal-memory-in-shared",
        "shared-memory-in-personal",
        "binding-mismatch",
        "thread-mismatch",
    ],
)
def test_memory_isolation(
    context_factory: object,
    memory_overrides: dict[str, object],
    error_code: str | None,
) -> None:
    context = context_factory()  # type: ignore[operator]
    model_input = _memory_input(**memory_overrides)
    if error_code is None:
        receipt = authorize_conversation_model_inputs(
            context=context,
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(model_input,),
        )
        assert receipt.approved_inputs[0].input_kind is ConversationModelInputKindV1.THREAD_MEMORY
        return
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_model_inputs(
            context=context,
            requested_capability=ConversationProductCapability.READ_ONLY_ASK,
            inputs=(model_input,),
        )
    assert exc_info.value.error_code == error_code


def test_same_canonical_request_produces_same_receipt_id() -> None:
    context = _personal_context()
    inputs = (_scoped_input(),)
    first = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=inputs,
    )
    second = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=inputs,
    )
    assert first.receipt_id == second.receipt_id


def test_changed_capability_produces_different_receipt_id() -> None:
    context = _personal_context()
    inputs = (_scoped_input(),)
    ask_receipt = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=inputs,
    )
    discovery_receipt = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.SOURCE_DISCOVERY,
        inputs=inputs,
    )
    assert ask_receipt.receipt_id != discovery_receipt.receipt_id


def test_changed_input_scope_produces_different_receipt_id() -> None:
    context = _personal_context()
    narrow = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(_scoped_input(input_id="input-a"),),
    )
    wider = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(
            _scoped_input(input_id="input-a"),
            _scoped_input(input_id="input-b"),
        ),
    )
    assert narrow.receipt_id != wider.receipt_id


def test_tampered_execution_receipt_id_is_rejected_on_outbound() -> None:
    context = _personal_context()
    execution_receipt = authorize_conversation_model_inputs(
        context=context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=(_scoped_input(),),
    )
    tampered = execution_receipt.model_copy(update={"receipt_id": "lkw-conversation-guard:v1:deadbeef"})
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_outbound_delivery(
            context=context,
            execution_receipt=tampered,
            target=_outbound_target(),
            used_input_ids=("input-1",),
            citation_input_ids=(),
        )
    assert exc_info.value.error_code == "CONVERSATION_GUARD_RECEIPT_INVALID"


def _authorized_outbound(
    *,
    context: ConversationExecutionContextV1 | None = None,
    inputs: tuple[ConversationScopedModelInputV1, ...] | None = None,
) -> tuple[ConversationExecutionContextV1, ConversationExecutionGuardReceiptV1]:
    resolved_context = context or _personal_context()
    model_inputs = inputs or (_scoped_input(),)
    execution_receipt = authorize_conversation_model_inputs(
        context=resolved_context,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        inputs=model_inputs,
    )
    return resolved_context, execution_receipt


def test_outbound_matching_target_and_inputs_succeeds() -> None:
    context, execution_receipt = _authorized_outbound()
    outbound = authorize_conversation_outbound_delivery(
        context=context,
        execution_receipt=execution_receipt,
        target=_outbound_target(),
        used_input_ids=("input-1",),
        citation_input_ids=("input-1",),
    )
    assert outbound.execution_receipt_id == execution_receipt.receipt_id
    assert outbound.used_input_ids == ("input-1",)
    assert outbound.citation_input_ids == ("input-1",)


@pytest.mark.parametrize(
    ("target_overrides", "error_code"),
    [
        ({"tenant_id": "tenant-other"}, "CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH"),
        ({"workspace_id": "workspace-other"}, "CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH"),
        ({"conversation_context_binding_id": "binding-other"}, "CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH"),
        ({"audience_mode": ConversationAudienceMode.SHARED}, "CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH"),
        ({"canonical_thread_ref": "thread-other"}, "CONVERSATION_GUARD_OUTBOUND_TARGET_MISMATCH"),
    ],
    ids=[
        "tenant-target-mismatch",
        "workspace-target-mismatch",
        "binding-target-mismatch",
        "audience-target-mismatch",
        "thread-target-mismatch",
    ],
)
def test_outbound_target_mismatch(
    target_overrides: dict[str, object],
    error_code: str,
) -> None:
    context, execution_receipt = _authorized_outbound()
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_outbound_delivery(
            context=context,
            execution_receipt=execution_receipt,
            target=_outbound_target(**target_overrides),
            used_input_ids=("input-1",),
            citation_input_ids=(),
        )
    assert exc_info.value.error_code == error_code


@pytest.mark.parametrize(
    ("used_input_ids", "citation_input_ids", "error_code"),
    [
        (("input-missing",), (), "CONVERSATION_GUARD_INPUT_NOT_APPROVED"),
        (("input-1", "input-1"), (), "CONVERSATION_GUARD_DUPLICATE_USED_INPUT"),
        (("input-1",), ("input-missing",), "CONVERSATION_GUARD_CITATION_NOT_APPROVED"),
        (("input-1",), ("input-1", "input-1"), "CONVERSATION_GUARD_DUPLICATE_CITATION"),
        (("input-1",), ("input-2",), "CONVERSATION_GUARD_CITATION_NOT_USED"),
    ],
    ids=[
        "unapproved-used-input",
        "duplicate-used-input",
        "unapproved-citation",
        "duplicate-citation",
        "citation-not-used",
    ],
)
def test_outbound_input_and_citation_validation(
    used_input_ids: tuple[str, ...],
    citation_input_ids: tuple[str, ...],
    error_code: str,
) -> None:
    context, execution_receipt = _authorized_outbound(
        inputs=(
            _scoped_input(input_id="input-1"),
            _scoped_input(input_id="input-2"),
        ),
    )
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_outbound_delivery(
            context=context,
            execution_receipt=execution_receipt,
            target=_outbound_target(),
            used_input_ids=used_input_ids,
            citation_input_ids=citation_input_ids,
        )
    assert exc_info.value.error_code == error_code


def test_shared_outbound_rechecks_audience_eligibility() -> None:
    context = _shared_context()
    tampered_approved = (
        ConversationApprovedModelInputV1(
            input_id="input-1",
            input_kind=ConversationModelInputKindV1.INDEXED_EVIDENCE,
            audience_eligibility=KnowledgeAudienceEligibilityV1.PERSONAL_ONLY,
        ),
    )
    receipt_payload = _execution_receipt_payload(
        tenant_id=context.tenant_id,
        conversation_context_binding_id=context.conversation_context_binding_id,
        audience_mode=context.audience_mode,
        workspace_id=context.workspace_id,
        canonical_thread_ref=context.canonical_thread_ref,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        approved_inputs=tampered_approved,
    )
    execution_receipt = ConversationExecutionGuardReceiptV1(
        receipt_id=_derive_receipt_id(
            prefix=_EXECUTION_RECEIPT_PREFIX,
            payload=receipt_payload,
        ),
        tenant_id=context.tenant_id,
        conversation_context_binding_id=context.conversation_context_binding_id,
        audience_mode=context.audience_mode,
        workspace_id=context.workspace_id,
        canonical_thread_ref=context.canonical_thread_ref,
        requested_capability=ConversationProductCapability.READ_ONLY_ASK,
        approved_inputs=tampered_approved,
    )
    with pytest.raises(ConversationExecutionGuardError) as exc_info:
        authorize_conversation_outbound_delivery(
            context=context,
            execution_receipt=execution_receipt,
            target=_outbound_target(audience_mode=ConversationAudienceMode.SHARED),
            used_input_ids=("input-1",),
            citation_input_ids=(),
        )
    assert exc_info.value.error_code == "CONVERSATION_GUARD_SHARED_INPUT_NOT_ALLOWED"


def test_outbound_receipt_is_deterministic_and_contains_no_answer_content() -> None:
    context, execution_receipt = _authorized_outbound()
    first = authorize_conversation_outbound_delivery(
        context=context,
        execution_receipt=execution_receipt,
        target=_outbound_target(),
        used_input_ids=("input-1",),
        citation_input_ids=(),
    )
    second = authorize_conversation_outbound_delivery(
        context=context,
        execution_receipt=execution_receipt,
        target=_outbound_target(),
        used_input_ids=("input-1",),
        citation_input_ids=(),
    )
    assert first.receipt_id == second.receipt_id
    forbidden = {"content", "answer", "snippet", "message", "credential", "token"}
    for model in (ConversationExecutionGuardReceiptV1, ConversationOutboundGuardReceiptV1):
        assert forbidden.isdisjoint(set(model.model_fields))
