# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 adversarial probes (production contracts; fixture/unit only)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import (
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewRequest,
    ContextViewScope,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.functional_evidence.tenant import (
    FunctionalEvidenceTenantIdentityError,
    require_tenant_id_from_exec_ctx,
)
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionTenantMismatchError,
    bootstrap_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.events.payloads.canonical import ContextAssemblyPayloadV4, LlmCallPayloadV3
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.llm.model_context_attribution import try_attribute_model_call_to_context
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    encode_vendor_knowledge_sync_job,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncMode
from intergrax.runtime.vendor_knowledge.sync_runtime import build_vendor_knowledge_sync_runtime
from intergrax.runtime.vendor_knowledge.sync_worker import make_vendor_knowledge_sync_worker_handler


def _kv() -> DistributedKVStore:
    class _KV(DistributedKVStore):
        def __init__(self) -> None:
            self._data: dict[tuple[str, str], bytes] = {}

        def get(self, tenant_id: str, key: str) -> bytes | None:
            return self._data.get((tenant_id, key))

        def set(
            self,
            tenant_id: str,
            key: str,
            value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> None:
            self._data[(tenant_id, key)] = value

        def delete(self, tenant_id: str, key: str) -> None:
            self._data.pop((tenant_id, key), None)

        def compare_and_set(
            self,
            tenant_id: str,
            key: str,
            expected: bytes | None,
            new_value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> bool:
            current = self.get(tenant_id, key)
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
            return True

    return _KV()


def _paired_context_llm_events(
    *,
    context_tenant: str,
    llm_tenant: str,
    shared_hash: str = "abc123hash",
    event_id: str | None = None,
) -> tuple[RuntimeEvent, RuntimeEvent]:
    eid = event_id or str(mint_event_id())
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    ctx_payload = ContextAssemblyPayloadV4(
        node_id="n1",
        summary_tier=None,
        context_original_chars=1,
        context_final_chars=1,
        trimmed=False,
        engine_id="",
        model_input_messages_hash=shared_hash,
        context_decision_evidence_fingerprint="fp-" + shared_hash,
    )
    ctx = runtime_event_with_payload(
        RuntimeEvent(
            tenant_id=context_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            correlation_id=str(task_id),
        ),
        ctx_payload,
    )
    ctx = ctx.model_copy(update={"event_id": eid})
    llm_payload = LlmCallPayloadV3(
        model="m",
        provider="openai",
        model_input_messages_hash=shared_hash,
        context_assembly_event_id=eid,
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    )
    llm = runtime_event_with_payload(
        RuntimeEvent(
            tenant_id=llm_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.LLM_CALL,
            phase=ExecutionPhase.STEP_EXECUTION,
            correlation_id=str(task_id),
        ),
        llm_payload,
    )
    return ctx, llm


def assert_context_llm_cross_tenant_attribution_rejects() -> None:
    ctx, llm = _paired_context_llm_events(context_tenant="tenant-a", llm_tenant="tenant-b")
    verdict = try_attribute_model_call_to_context(ctx, llm)
    assert verdict.attributable is False
    assert verdict.reason == "tenant_mismatch"


def assert_context_view_request_authority_mismatch_rejects() -> None:
    scope = ContextViewScope(tenant_id="tenant-a", workspace_id="ws-a")
    with pytest.raises(ValidationError):
        ContextViewRequest(
            scope=scope,
            acting_principal_id="principal-a",
            operation_id="op-1",
            requested_categories=(ContextViewCategory.MEMORY,),
            authority_request=EffectiveAuthorityRequest(
                tenant_id="tenant-b",
                workspace_id="ws-a",
                acting_principal_id="principal-a",
                requested_authority_scopes=("context_view.read",),
            ),
        )


def assert_background_transport_task_tenant_mismatch_rejects() -> None:
    persistence = KvBackgroundExecutionIdentityPersistence(_kv())
    transport = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="document_store",
        transport_task_id="t-1",
    )
    with pytest.raises(BackgroundExecutionTenantMismatchError):
        bootstrap_background_execution(
            transport_ref=transport,
            identity_persistence=persistence,
            task_tenant_id="tenant-b",
        )


def assert_background_blank_transport_rejects() -> None:
    persistence = KvBackgroundExecutionIdentityPersistence(_kv())
    transport = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="document_store",
        transport_task_id="t-1",
    )
    with pytest.raises(ValueError, match="task_tenant_id"):
        bootstrap_background_execution(
            transport_ref=transport,
            identity_persistence=persistence,
            task_tenant_id="   ",
        )


def assert_vendor_queue_request_tenant_mismatch_rejects() -> None:
    store = InMemoryDocumentStore()
    runtime = build_vendor_knowledge_sync_runtime(
        document_store=store,
        binding_service=Mock(),
        facade=Mock(),
        sink=Mock(),
        owner_id="owner",
        tenant_id="tenant-a",
    )
    queue = runtime.task_queue
    with pytest.raises(ValueError, match="tenant mismatch"):
        queue.enqueue(
            TaskRequest(
                tenant_id="tenant-b",
                run_id="run-1",
                task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                payload=b"{}",
                idempotency_key="k1",
            )
        )


def assert_vendor_worker_payload_job_tenant_mismatch_rejects() -> None:
    job_b = VendorKnowledgeSyncJob(
        tenant_id="tenant-b",
        binding_id="b1",
        operation_id="op",
        mode=KnowledgeSyncMode.INCREMENTAL,
        restart=False,
        page_size=10,
        recovery_attempt=0,
    )
    handler = make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=lambda tenant_id, run_id: (_ for _ in ()).throw(RuntimeError("skip")),
        scheduler=Mock(),
        main_loop_provider=lambda: None,
    )
    from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity

    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    result = handler(
        tenant_id="tenant-a",
        run_id="run-1",
        payload=encode_vendor_knowledge_sync_job(job_b),
        execution_identity=identity,
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == "vendor_knowledge_sync_tenant_mismatch"


def assert_functional_evidence_missing_tenant_rejects() -> None:
    ctx = RuntimeExecutionContext(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="agent-1",
        canonical_request_identity=None,
    )
    with pytest.raises(FunctionalEvidenceTenantIdentityError):
        require_tenant_id_from_exec_ctx(ctx)


def assert_context_view_request_membership_mismatch_rejects() -> None:
    scope = ContextViewScope(tenant_id="tenant-a", workspace_id="ws-a")
    with pytest.raises(ValidationError):
        ContextViewRequest(
            scope=scope,
            acting_principal_id="principal-a",
            operation_id="op-1",
            requested_categories=(ContextViewCategory.MEMORY,),
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=WorkspaceMembership(
                tenant_id="tenant-b",
                workspace_id="ws-b",
                membership_id="mem-1",
                principal_id="principal-a",
                role=WorkspaceMembershipRole.MEMBER,
                revision=0,
            ),
        )
