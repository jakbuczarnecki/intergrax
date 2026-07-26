# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for DocumentStore knowledge source binding repository."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
    binding_from_document,
    binding_partition_key,
    binding_row_key,
    binding_to_document,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingAlreadyExists,
    KnowledgeSourceBindingCorruptRecord,
    KnowledgeSourceBindingNotFound,
    KnowledgeSourceBindingStatus,
    KnowledgeSourceBindingVersionConflict,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from tests.unit.runtime.vendor_knowledge._fakes import InMemoryDocumentStore


def _scope(*, remote_scope_id: str = "scope-1") -> KnowledgeSourceScope:
    return KnowledgeSourceScope(
        remote_scope_id=remote_scope_id,
        remote_scope_type="project",
        safe_display_name="Example Project",
        parameters={},
    )


def _binding(
    *,
    binding_id: str = "bind-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "example",
    source_kind: str = "issues",
    connection_ref: str = "conn-1",
    status: KnowledgeSourceBindingStatus = KnowledgeSourceBindingStatus.ACTIVE,
    configuration_version: int = 1,
    safe_display_name: str = "Example binding",
) -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=source_kind,
        connection_ref=connection_ref,
        credential_ref="cred-1",
        safe_display_name=safe_display_name,
        scope=_scope(),
        status=status,
        configuration_version=configuration_version,
        broad_scope=False,
        scope_approval_ref=None,
    )


@pytest.mark.unit
def test_create_get_round_trip() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    binding = _binding()
    repo.create(binding)
    loaded = repo.get(tenant_id="tenant-1", binding_id="bind-1")
    assert loaded == binding


@pytest.mark.unit
def test_duplicate_create_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    with pytest.raises(KnowledgeSourceBindingAlreadyExists):
        repo.create(_binding(safe_display_name="Other"))


@pytest.mark.unit
def test_tenant_partition_isolation() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding(tenant_id="tenant-a", binding_id="shared"))
    repo.create(_binding(tenant_id="tenant-b", binding_id="shared", connection_ref="conn-b"))
    assert repo.get(tenant_id="tenant-a", binding_id="shared") is not None
    assert repo.get(tenant_id="tenant-b", binding_id="shared") is not None
    assert repo.get(tenant_id="tenant-a", binding_id="shared").connection_ref == "conn-1"
    assert repo.get(tenant_id="tenant-c", binding_id="shared") is None


@pytest.mark.unit
def test_deterministic_row_key() -> None:
    assert binding_partition_key("tenant-1") == "vendor_knowledge_bindings:tenant-1"
    assert binding_row_key("bind-1") == "binding:bind-1"
    document = binding_to_document(_binding())
    assert document.partition_key == "vendor_knowledge_bindings:tenant-1"
    assert document.row_key == "binding:bind-1"


@pytest.mark.unit
def test_deterministic_list_order() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding(binding_id="bind-c"))
    repo.create(_binding(binding_id="bind-a", connection_ref="conn-a"))
    repo.create(_binding(binding_id="bind-b", connection_ref="conn-b"))
    listed = repo.list(tenant_id="tenant-1")
    assert [item.binding_id for item in listed] == ["bind-a", "bind-b", "bind-c"]


@pytest.mark.unit
def test_status_filtering() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding(binding_id="active-1"))
    repo.create(
        _binding(
            binding_id="disabled-1",
            connection_ref="conn-d",
            status=KnowledgeSourceBindingStatus.DISABLED,
        )
    )
    listed = repo.list(
        tenant_id="tenant-1",
        status=KnowledgeSourceBindingStatus.DISABLED,
    )
    assert len(listed) == 1
    assert listed[0].binding_id == "disabled-1"


@pytest.mark.unit
def test_update_success() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    updated = _binding(
        configuration_version=2,
        safe_display_name="Updated",
        connection_ref="conn-2",
        status=KnowledgeSourceBindingStatus.DISABLED,
    )
    repo.update(updated, expected_configuration_version=1)
    loaded = repo.get(tenant_id="tenant-1", binding_id="bind-1")
    assert loaded is not None
    assert loaded.safe_display_name == "Updated"
    assert loaded.connection_ref == "conn-2"
    assert loaded.status is KnowledgeSourceBindingStatus.DISABLED
    assert loaded.configuration_version == 2


@pytest.mark.unit
def test_stale_expected_version_conflict() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    with pytest.raises(KnowledgeSourceBindingVersionConflict):
        repo.update(
            _binding(configuration_version=2, safe_display_name="Nope"),
            expected_configuration_version=0,
        )


@pytest.mark.unit
def test_skipped_configuration_version_conflict() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    with pytest.raises(KnowledgeSourceBindingVersionConflict):
        repo.update(
            _binding(configuration_version=3, safe_display_name="Skip"),
            expected_configuration_version=1,
        )


@pytest.mark.unit
def test_immutable_identity_change_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    with pytest.raises(KnowledgeSourceBindingVersionConflict):
        repo.update(
            _binding(
                provider_id="changed",
                configuration_version=2,
            ),
            expected_configuration_version=1,
        )
    assert repo.get(tenant_id="tenant-1", binding_id="bind-1").provider_id == "example"


@pytest.mark.unit
def test_corrupt_record_rejected() -> None:
    store = InMemoryDocumentStore()
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge_bindings:tenant-1",
            row_key="binding:bind-1",
            data={"binding_id": "bind-1"},
        )
    )
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    with pytest.raises(KnowledgeSourceBindingCorruptRecord):
        repo.get(tenant_id="tenant-1", binding_id="bind-1")


@pytest.mark.unit
def test_mismatched_partition_tenant_rejected() -> None:
    binding = _binding(tenant_id="tenant-1")
    document = binding_to_document(binding)
    bad = DocumentRecord(
        partition_key="vendor_knowledge_bindings:tenant-other",
        row_key=document.row_key,
        data=dict(document.data),
    )
    with pytest.raises(KnowledgeSourceBindingCorruptRecord):
        binding_from_document(bad)


@pytest.mark.unit
def test_document_does_not_contain_raw_secret_fields() -> None:
    document = binding_to_document(_binding())
    for key in (
        "access_token",
        "refresh_token",
        "api_key",
        "password",
        "client_secret",
        "authorization_header",
        "signed_download_url",
    ):
        assert key not in document.data
    assert "credential_ref" in document.data
    assert document.data["credential_ref"] == "cred-1"

    with pytest.raises(KnowledgeSourceBindingCorruptRecord):
        binding_from_document(
            DocumentRecord(
                partition_key="vendor_knowledge_bindings:tenant-1",
                row_key="binding:bind-1",
                data={
                    **dict(document.data),
                    "access_token": "leak",
                },
            )
        )


@pytest.mark.unit
def test_repository_does_not_close_document_store() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    repo.create(_binding())
    repo.get(tenant_id="tenant-1", binding_id="bind-1")
    repo.list(tenant_id="tenant-1")
    assert store.close_calls == 0
    assert store.closed is False
    assert "close" not in DocumentStoreKnowledgeSourceBindingRepository.__dict__


@pytest.mark.unit
def test_update_missing_binding_not_found() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    with pytest.raises(KnowledgeSourceBindingNotFound):
        repo.update(
            _binding(configuration_version=2),
            expected_configuration_version=1,
        )
