# © Artur Czarnecki. All rights reserved.

"""Scenario-oriented product acceptance proof for Local Workspace Knowledge."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from local_workspace_application.workspaces.knowledge_administration_service import (
    DeterministicKnowledgeAdministrationIntentInterpreter,
    HmacKnowledgeAdministrationConfirmationCodec,
    KnowledgeAdministrationActionV1,
    KnowledgeAdministrationConfirmationV1,
    KnowledgeAdministrationIntentV1,
    KnowledgeAdministrationService,
    KnowledgeAdministrationStatusV1,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    KnowledgeOperationCommandV1,
    KnowledgeOperationError,
    KnowledgeOperationV1,
    KnowledgeOperationsService,
    indexed_knowledge_item_id,
    live_knowledge_item_id,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    LiveWorkspaceCitationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV2,
)

from applications.local_workspace_application.tests.host.test_deployment_operations import (
    test_readiness_reports_capabilities_and_liveness_without_secret as _accepted_readiness_proof,
)
from applications.local_workspace_application.tests.workspaces.test_hybrid_ask_service import (
    _LIVE_ID,
    _RecordingLLM,
    _indexed_evidence,
    _command,
    _service,
)
from applications.local_workspace_application.tests.workspaces.rag_e2e_support import (
    _LATEST,
    _MARKER_ROOT,
    _OLDEST,
    _PREFIX,
    _TENANT,
    _WORKSPACE,
)
pytestmark = pytest.mark.integration

_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)
_SECRET = "acceptance-sentinel-hmac-secret"
_HASH = "a" * 64


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _drain(runtime: Any, repo: Any, operation_id: str) -> None:
    for _ in range(128):
        runtime.worker.drain_once()
        operation = repo.get_operation(
            tenant_id=_TENANT,
            operation_id=operation_id,
        )
        if operation is not None and operation.status.value in {"completed", "failed"}:
            return
    raise AssertionError("indexed_sync_timeout")


def _operation(
    *,
    item_id: str,
    operation: KnowledgeOperationV1,
    revision: int,
    request_id: str,
    workspace_id: str = _WORKSPACE,
) -> KnowledgeOperationCommandV1:
    return KnowledgeOperationCommandV1(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        knowledge_item_id=item_id,
        operation=operation,
        expected_revision=revision,
        idempotency_key_hash=_hash(request_id),
    )


def test_indexed_user_journey(rag_e2e_env: dict[str, Any]) -> None:
    """The product-facing Indexed path proves committed visibility, not receipt only."""

    client = rag_e2e_env["client"]
    repo = rag_e2e_env["repo"]
    runtime = rag_e2e_env["runtime"]
    llm = rag_e2e_env["llm"]

    discovered = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/"
        "conn.slack/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    )
    assert discovered.status_code == 200, discovered.text
    candidate = discovered.json()["items"][0]["opaque_candidate_ref"]

    created = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/1",
            "Idempotency-Key": "acceptance-indexed-create",
        },
        json={
            "connection_ref": "conn.slack",
            "opaque_candidate_ref": candidate,
            "root_oldest": _OLDEST,
            "root_latest": _LATEST,
        },
    )
    assert created.status_code == 201, created.text
    created_body = created.json()
    indexed_binding_id = created_body["indexed_source_binding_id"]
    source_id = created_body["source_id"]

    sync = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{indexed_binding_id}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert sync.status_code == 202, sync.text
    _drain(runtime, repo, sync.json()["operation_id"])

    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert refs
    assert any(
        ref.source_id == source_id and ref.materialization_ownership is not None
        for ref in refs
    )
    inventory_service = client.app.state.lkw_knowledge_inspection_service
    inventory = inventory_service.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    item = next(
        item
        for item in inventory.items
        if item.indexed_source_binding_id == indexed_binding_id
    )
    assert item.lifecycle_state == "active"
    assert item.sync_state == "succeeded"
    assert item.knowledge_item_id == indexed_knowledge_item_id(indexed_binding_id)

    ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "acceptance-indexed-ask"},
        json={"question": _MARKER_ROOT},
    )
    assert ask.status_code == 200, ask.text
    answer = ask.json()
    assert answer["citations"]
    assert any(citation["source_id"] == source_id for citation in answer["citations"])
    assert any(
        _MARKER_ROOT in content
        for message in llm.messages
        for _, content in message
    )

    operations = client.app.state.lkw_knowledge_operations_service
    disabled = asyncio.run(
        operations.execute(
            _operation(
                item_id=item.knowledge_item_id,
                operation=KnowledgeOperationV1.DISABLE,
                revision=item.revision,
                request_id="acceptance-indexed-disable",
            )
        )
    )
    assert disabled.item.lifecycle_state == "disabled"
    disabled_inventory = inventory_service.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    disabled_item = next(
        candidate
        for candidate in disabled_inventory.items
        if candidate.knowledge_item_id == item.knowledge_item_id
    )
    assert disabled_item.lifecycle_state == "disabled"

    disabled_ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={
            "X-Tenant-Id": _TENANT,
            "Idempotency-Key": "acceptance-indexed-disabled-ask",
        },
        json={"question": _MARKER_ROOT},
    )
    assert disabled_ask.status_code == 200, disabled_ask.text
    assert all(
        citation["source_id"] != source_id
        for citation in disabled_ask.json().get("citations", ())
    )

    enabled = asyncio.run(
        operations.execute(
            _operation(
                item_id=item.knowledge_item_id,
                operation=KnowledgeOperationV1.ENABLE,
                revision=disabled.item.revision,
                request_id="acceptance-indexed-enable",
            )
        )
    )
    assert enabled.item.knowledge_item_id == item.knowledge_item_id
    assert enabled.item.lifecycle_state == "active"
    assert len(repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == len(
        refs
    )

    enabled_search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": _MARKER_ROOT, "limit": 10},
    )
    assert enabled_search.status_code == 200, enabled_search.text
    assert any(
        hit["source_id"] == source_id for hit in enabled_search.json()["results"]
    )

    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        asyncio.run(
            operations.execute(
                _operation(
                    item_id=item.knowledge_item_id,
                    operation=KnowledgeOperationV1.DISABLE,
                    revision=item.revision,
                    request_id="acceptance-indexed-stale",
                )
            )
        )


class _AcceptanceConfiguration:
    def __init__(
        self,
        *,
        indexed: tuple[SimpleNamespace, ...],
        live: tuple[SimpleNamespace, ...],
    ) -> None:
        self.configuration = SimpleNamespace(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_sources=indexed,
            live_access_bindings=live,
            updated_at=_NOW,
        )

    def get_configuration(self, *, tenant_id: str, workspace_id: str) -> object | None:
        if tenant_id != _TENANT or workspace_id != _WORKSPACE:
            return None
        return self.configuration


class _AcceptanceLifecycle:
    def __init__(self, views: dict[str, SimpleNamespace], *, live: bool) -> None:
        self.views = views
        self.live = live
        self.effects: list[tuple[str, str]] = []
        self.replays: dict[tuple[str, str], SimpleNamespace] = {}

    def get(self, command: object = None, **kwargs: object) -> SimpleNamespace:
        binding_id = (
            getattr(command, "live_access_binding_id", None)
            if self.live
            else kwargs.get("indexed_source_binding_id")
        )
        if not isinstance(binding_id, str) or binding_id not in self.views:
            raise RuntimeError("knowledge_item_not_found")
        return self.views[binding_id]

    def _mutate(self, command: object, operation: str) -> SimpleNamespace:
        binding_id = (
            getattr(command, "live_access_binding_id", None)
            if self.live
            else getattr(command, "indexed_source_binding_id", None)
        )
        key = (str(getattr(command, "idempotency_key_hash")), operation)
        if key in self.replays:
            return self.replays[key]
        if not isinstance(binding_id, str) or binding_id not in self.views:
            raise RuntimeError("knowledge_item_not_found")
        view = self.views[binding_id]
        if operation != "sync":
            if self.live:
                view.configuration_revision += 1
            else:
                view.lifecycle_revision += 1
        if operation == "disable":
            view.enabled = False
            view.lifecycle_state = "disabled"
        elif operation == "enable":
            view.enabled = True
            view.lifecycle_state = "active"
        elif operation == "detach":
            view.enabled = False
            view.detached = True
            view.lifecycle_state = "detached"
        result = SimpleNamespace(
            operation_id=f"operation-{operation}-{binding_id}",
            mutation_id=f"mutation-{operation}-{binding_id}",
        )
        self.effects.append((binding_id, operation))
        self.replays[key] = result
        return result

    def request_sync(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "sync")

    def retry_sync(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "sync")

    def disable(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "disable")

    def enable(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "enable")

    def detach(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "detach")

    def resume_detach(self, command: object) -> SimpleNamespace:
        return self._mutate(command, "detach")


def _admin_stack() -> tuple[
    KnowledgeInspectionService,
    KnowledgeOperationsService,
    KnowledgeAdministrationService,
    _AcceptanceLifecycle,
    _AcceptanceLifecycle,
]:
    indexed_bindings = tuple(
        SimpleNamespace(
            indexed_source_binding_id=f"indexed-{index}",
            cached_safe_display_label=(
                "Shared" if index == 0 else "Indexed Disabled" if index == 1 else f"Indexed {index:03d}"
            ),
        )
        for index in range(100)
    )
    live_bindings = (
        SimpleNamespace(
            live_access_binding_id="live-active",
            derived_safe_display_label="Shared",
            allowed_capability_ids=("vendor.acceptance.read",),
        ),
        SimpleNamespace(
            live_access_binding_id="live-unavailable",
            derived_safe_display_label="Unavailable",
            allowed_capability_ids=("vendor.acceptance.read",),
        ),
    )
    indexed_views = {
        binding.indexed_source_binding_id: SimpleNamespace(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=f"source-{binding.indexed_source_binding_id}",
            indexed_source_binding_id=binding.indexed_source_binding_id,
            knowledge_source_binding_ref=f"ref-{binding.indexed_source_binding_id}",
            lifecycle_state="disabled" if binding.indexed_source_binding_id == "indexed-1" else "active",
            lifecycle_revision=1,
            enabled=binding.indexed_source_binding_id != "indexed-1",
            detached=False,
            sync_state="succeeded",
            last_successful_sync_at=_NOW,
            last_error_code=None,
            updated_at=_NOW,
        )
        for binding in indexed_bindings
    }
    live_views = {
        "live-active": SimpleNamespace(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            live_access_binding_id="live-active",
            connection_ref="connection-active",
            lifecycle_state="active",
            configuration_revision=1,
            enabled=True,
            detached=False,
            runtime_available=True,
            last_error_code=None,
            updated_at=_NOW,
        ),
        "live-unavailable": SimpleNamespace(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            live_access_binding_id="live-unavailable",
            connection_ref="connection-unavailable",
            lifecycle_state="active",
            configuration_revision=1,
            enabled=True,
            detached=False,
            runtime_available=False,
            last_error_code="connection_unavailable",
            updated_at=_NOW,
        ),
    }
    configuration = _AcceptanceConfiguration(
        indexed=indexed_bindings,
        live=live_bindings,
    )
    indexed_lifecycle = _AcceptanceLifecycle(indexed_views, live=False)
    live_lifecycle = _AcceptanceLifecycle(live_views, live=True)
    inspection = KnowledgeInspectionService(
        configuration_service=cast(Any, configuration),
        indexed_source_lifecycle_service=cast(Any, indexed_lifecycle),
        live_access_lifecycle_service=cast(Any, live_lifecycle),
    )
    operations = KnowledgeOperationsService(
        inspection_service=inspection,
        indexed_source_lifecycle_service=cast(Any, indexed_lifecycle),
        live_access_lifecycle_service=cast(Any, live_lifecycle),
    )
    administration = KnowledgeAdministrationService(
        inspection_service=inspection,
        operations_service=operations,
        interpreter=_AcceptanceInterpreter(),
        idempotency_key_factory=_IdempotencyFactory(),
        confirmation_port=HmacKnowledgeAdministrationConfirmationCodec(
            secret=_SECRET.encode()
        ),
    )
    return inspection, operations, administration, indexed_lifecycle, live_lifecycle


class _AcceptanceInterpreter(DeterministicKnowledgeAdministrationIntentInterpreter):
    async def interpret(self, *, utterance: str, context: Any):
        if utterance.casefold().strip() == "list knowledge sources":
            return KnowledgeAdministrationIntentV1(
                action=KnowledgeAdministrationActionV1.LIST,
            )
        return await super().interpret(utterance=utterance, context=context)


class _IdempotencyFactory:
    def create(self, **fields: object) -> str:
        return _hash("\x1f".join(str(fields[key]) for key in sorted(fields)))


@pytest.mark.asyncio
async def test_knowledge_administration_journey() -> None:
    inspection, operations, administration, indexed, live = _admin_stack()
    inventory = inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(inventory.items) == 102
    assert inventory.summary.total == 102
    assert inventory.summary.indexed == 100
    assert inventory.summary.live == 2
    assert inventory.summary.attention_required == 1
    assert [item.mode for item in inventory.items[:2]] == [
        KnowledgeAccessModeV1.INDEXED,
        KnowledgeAccessModeV1.INDEXED,
    ]
    assert indexed_knowledge_item_id("indexed-0") in {
        item.knowledge_item_id for item in inventory.items
    }
    assert inventory.items[-1].knowledge_item_id == live_knowledge_item_id("live-unavailable")

    listed = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-list",
        utterance="list knowledge sources",
    )
    assert listed.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert listed.inventory is not None
    assert len(listed.inventory.items) == 102

    synced = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-sync",
        utterance="sync indexed Shared",
    )
    assert synced.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert indexed.effects[-1] == ("indexed-0", "sync")
    replay_command = _operation(
        item_id=indexed_knowledge_item_id("indexed-4"),
        operation=KnowledgeOperationV1.SYNC,
        revision=1,
        request_id="admin-sync-replay",
    )
    await operations.execute(replay_command)
    await operations.execute(replay_command)
    assert indexed.effects.count(("indexed-4", "sync")) == 1

    disabled = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-disable-live",
        utterance="disable live Shared",
    )
    assert disabled.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert live.effects[-1] == ("live-active", "disable")

    ambiguous = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-ambiguous",
        utterance="detach Shared",
    )
    assert ambiguous.status is KnowledgeAdministrationStatusV1.AMBIGUOUS
    effects_before = len(indexed.effects) + len(live.effects)
    assert len(indexed.effects) + len(live.effects) == effects_before

    item_id = indexed_knowledge_item_id("indexed-0")
    pending = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-detach-pending",
        utterance=f"detach {item_id}",
    )
    assert pending.status is KnowledgeAdministrationStatusV1.CONFIRMATION_REQUIRED
    assert pending.confirmation_token
    effects_before = len(indexed.effects)

    tampered = ("A" if pending.confirmation_token[0] != "A" else "B") + (
        pending.confirmation_token[1:]
    )
    invalid = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-tampered",
        utterance=f"detach {item_id}",
        confirmation_token=tampered,
    )
    assert invalid.message_code == "knowledge_admin_confirmation_invalid"
    assert len(indexed.effects) == effects_before

    confirmed = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-detach-confirm",
        utterance=f"detach {item_id}",
        confirmation_token=pending.confirmation_token,
    )
    assert confirmed.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert len(indexed.effects) == effects_before + 1
    assert indexed.views["indexed-0"].detached is True

    codec = HmacKnowledgeAdministrationConfirmationCodec(secret=_SECRET.encode())
    second_item_id = indexed_knowledge_item_id("indexed-2")
    expired = codec.issue(
        KnowledgeAdministrationConfirmationV1(
            token="",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id=second_item_id,
            operation=KnowledgeOperationV1.DETACH,
            expected_revision=1,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    expired_result = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-expired",
        utterance=f"detach {second_item_id}",
        confirmation_token=expired,
    )
    assert expired_result.message_code == "knowledge_admin_confirmation_expired"

    unknown = await administration.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-unknown",
        utterance="disable missing-item",
    )
    assert unknown.status is KnowledgeAdministrationStatusV1.NOT_FOUND
    wrong_workspace = await administration.handle(
        tenant_id=_TENANT,
        workspace_id="workspace-other",
        request_id="admin-other-workspace",
        utterance="list knowledge sources",
    )
    assert wrong_workspace.status is KnowledgeAdministrationStatusV1.REJECTED

    restarted = KnowledgeAdministrationService(
        inspection_service=inspection,
        operations_service=operations,
        interpreter=_AcceptanceInterpreter(),
        idempotency_key_factory=_IdempotencyFactory(),
        confirmation_port=codec,
    )
    new_pending = await restarted.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-restart-pending",
        utterance=f"detach {indexed_knowledge_item_id('indexed-2')}",
    )
    assert new_pending.status is KnowledgeAdministrationStatusV1.CONFIRMATION_REQUIRED
    assert new_pending.confirmation_token
    restart_confirmed = await restarted.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-restart-confirm",
        utterance=f"detach {indexed_knowledge_item_id('indexed-2')}",
        confirmation_token=new_pending.confirmation_token,
    )
    assert restart_confirmed.status is KnowledgeAdministrationStatusV1.COMPLETED

    rotation_item_id = indexed_knowledge_item_id("indexed-3")
    rotation_pending = await restarted.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-rotation-pending",
        utterance=f"detach {rotation_item_id}",
    )
    assert rotation_pending.confirmation_token
    rotated = KnowledgeAdministrationService(
        inspection_service=inspection,
        operations_service=operations,
        interpreter=_AcceptanceInterpreter(),
        idempotency_key_factory=_IdempotencyFactory(),
        confirmation_port=HmacKnowledgeAdministrationConfirmationCodec(
            secret=b"rotated-secret"
        ),
    )
    rejected_after_rotation = await rotated.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="admin-rotated",
        utterance=f"detach {rotation_item_id}",
        confirmation_token=rotation_pending.confirmation_token,
    )
    assert rejected_after_rotation.message_code == "knowledge_admin_confirmation_invalid"

    stale = _operation(
        item_id=live_knowledge_item_id("live-active"),
        operation=KnowledgeOperationV1.ENABLE,
        revision=1,
        request_id="stale-live",
    )
    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        await operations.execute(stale)
    with pytest.raises(KnowledgeOperationError, match="knowledge_item_not_found"):
        await operations.execute(
            _operation(
                item_id=item_id,
                operation=KnowledgeOperationV1.ENABLE,
                revision=1,
                request_id="wrong-workspace",
                workspace_id="workspace-other",
            )
        )

    serialized = json.dumps(
        [
            inventory.model_dump(mode="json"),
            listed.model_dump(mode="json"),
            confirmed.model_dump(mode="json"),
        ],
        sort_keys=True,
    )
    assert _SECRET not in serialized
    assert "credential" not in serialized.casefold()
    assert "access_token" not in serialized


@pytest.mark.asyncio
async def test_live_user_journey_and_fail_closed_policy() -> None:
    llm = _RecordingLLM([_LIVE_ID])
    service, _, live_executor, _ = _service(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )
    result = await service.ask(_command(QueryPolicyModeV2.LIVE_ONLY))
    assert result.status.value == "completed"
    assert result.answer
    assert result.citations
    assert isinstance(result.citations[0], LiveWorkspaceCitationV1)
    assert live_executor.calls == 1

    indexed_only_llm = _RecordingLLM([_indexed_evidence().evidence_id])
    indexed_only, _, denied_live_executor, _ = _service(
        QueryPolicyModeV2.INDEXED_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        indexed_only_llm,
    )
    denied = await indexed_only.ask(_command(QueryPolicyModeV2.INDEXED_ONLY))
    assert denied.status.value == "completed"
    assert denied_live_executor.calls == 0
    assert indexed_only_llm.calls == 1


def test_readiness_and_startup_contract_without_secret(tmp_path: Any) -> None:
    _accepted_readiness_proof(tmp_path)


@pytest.mark.asyncio
async def test_restart_recovery_journey() -> None:
    inspection, operations, first, indexed, _ = _admin_stack()
    item_id = indexed_knowledge_item_id("indexed-10")
    token_result = await first.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="restart-token",
        utterance=f"detach {item_id}",
    )
    assert token_result.confirmation_token
    reconstructed = KnowledgeAdministrationService(
        inspection_service=inspection,
        operations_service=operations,
        interpreter=_AcceptanceInterpreter(),
        idempotency_key_factory=_IdempotencyFactory(),
        confirmation_port=HmacKnowledgeAdministrationConfirmationCodec(
            secret=_SECRET.encode()
        ),
    )
    confirmed = await reconstructed.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="restart-confirm",
        utterance=f"detach {item_id}",
        confirmation_token=token_result.confirmation_token,
    )
    assert confirmed.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert indexed.views["indexed-10"].detached is True
    before = tuple(item.knowledge_item_id for item in inspection.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ).items)
    after = tuple(item.knowledge_item_id for item in inspection.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ).items)
    assert after == before


def test_isolation_and_security_journey() -> None:
    inspection, operations, administration, _, live = _admin_stack()
    with pytest.raises(KnowledgeInventoryError, match="knowledge_item_not_found"):
        inspection.list_items(tenant_id="tenant-other", workspace_id=_WORKSPACE)
    with pytest.raises(KnowledgeOperationError, match="knowledge_item_not_found"):
        asyncio.run(
            operations.execute(
                _operation(
                    item_id=live_knowledge_item_id("live-active"),
                    operation=KnowledgeOperationV1.DISABLE,
                    revision=1,
                    request_id="cross-tenant",
                ).model_copy(update={"tenant_id": "tenant-other"})
            )
        )
    inventory = inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    serialized = json.dumps(
        [
            inventory.model_dump(mode="json"),
            administration._context(inventory=inventory, request_id="safe").model_dump(
                mode="json"
            ),
        ],
        sort_keys=True,
    )
    assert _SECRET not in serialized
    assert "access_token" not in serialized
    assert "api_key" not in serialized
