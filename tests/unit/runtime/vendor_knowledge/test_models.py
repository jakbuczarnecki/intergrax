# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for vendor-knowledge facade models."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgePrincipal,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
    KnowledgeVisibility,
)


def _scope(**overrides: object) -> KnowledgeSourceScope:
    payload: dict[str, object] = {
        "remote_scope_id": "space-1",
        "remote_scope_type": "space",
        "safe_display_name": "Engineering",
        "parameters": {"project": "ENG"},
    }
    payload.update(overrides)
    return KnowledgeSourceScope.model_validate(payload)


def _source(**overrides: object) -> KnowledgeSourceRef:
    payload: dict[str, object] = {
        "tenant_id": "tenant-1",
        "provider_id": "example",
        "integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "source_kind": "pages",
        "connection_ref": "conn-1",
        "scope": _scope(),
    }
    payload.update(overrides)
    return KnowledgeSourceRef.model_validate(payload)


def _identity(**overrides: object) -> KnowledgeItemIdentity:
    payload: dict[str, object] = {"remote_id": "item-1", "parent_remote_id": None, "logical_key": None}
    payload.update(overrides)
    return KnowledgeItemIdentity.model_validate(payload)


def _revision(**overrides: object) -> KnowledgeItemRevision:
    payload: dict[str, object] = {
        "version": "3",
        "etag": "etag-1",
        "content_hash": "hash-1",
        "acl_hash": "acl-1",
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    payload.update(overrides)
    return KnowledgeItemRevision.model_validate(payload)


def _provenance(**overrides: object) -> KnowledgeItemProvenance:
    payload: dict[str, object] = {
        "provider_id": "example",
        "source_kind": "pages",
        "remote_id": "item-1",
        "web_url": "https://example.test/item-1",
        "safe_locator": "pages/item-1",
    }
    payload.update(overrides)
    return KnowledgeItemProvenance.model_validate(payload)


def _descriptor(**overrides: object) -> KnowledgeItemDescriptor:
    payload: dict[str, object] = {
        "identity": _identity(),
        "revision": _revision(),
        "title": "Spec",
        "item_type": "page",
        "content_mode": KnowledgeContentMode.RICH_TEXT,
        "content_available": True,
        "provenance": _provenance(),
        "metadata": {"lang": "en"},
    }
    payload.update(overrides)
    return KnowledgeItemDescriptor.model_validate(payload)


@pytest.mark.unit
def test_descriptor_repr_hides_title() -> None:
    descriptor = _descriptor(title="Secret Subject Line")
    rendered = repr(descriptor)
    assert "Secret Subject Line" not in rendered
    assert descriptor.model_dump()["title"] == "Secret Subject Line"


@pytest.mark.unit
def test_content_repr_hides_payloads() -> None:
    content = KnowledgeContent(
        mode=KnowledgeContentMode.BINARY,
        binary=b"secret-bytes",
        mime_type="application/pdf",
    )
    rendered = repr(content)
    assert "secret-bytes" not in rendered
    assert content.model_dump()["binary"] == b"secret-bytes"

    rich = KnowledgeContent(mode=KnowledgeContentMode.RICH_TEXT, rich_text="secret body")
    rich_rendered = repr(rich)
    assert "secret body" not in rich_rendered
    assert rich.model_dump()["rich_text"] == "secret body"

    structured = KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={"secret": "value"},
    )
    structured_rendered = repr(structured)
    assert "secret" not in structured_rendered
    assert structured.model_dump()["structured_record"] == {"secret": "value"}


@pytest.mark.unit
def test_models_construct_successfully() -> None:
    source = _source()
    descriptor = _descriptor()
    principal = KnowledgePrincipal(
        principal_type="user",
        principal_id="u-1",
        provider_id="example",
    )
    permissions = KnowledgePermissions(
        visibility=KnowledgeVisibility.RESTRICTED,
        allowed_principals=(principal,),
        denied_principals=(),
        inherited=True,
        acl_version="v1",
    )
    content = KnowledgeContent(mode=KnowledgeContentMode.RICH_TEXT, rich_text="hello")
    cursor = KnowledgeCursor(value="cursor-1", version="1")
    change = KnowledgeChange(
        kind=KnowledgeChangeKind.UPSERT,
        descriptor=descriptor,
        remote_id="item-1",
    )
    page = KnowledgePage(
        changes=(change,),
        next_cursor=cursor,
        proposed_checkpoint=cursor,
        has_more=True,
    )
    scope_info = KnowledgeScopeInfo(
        source=source,
        capabilities=KnowledgeAdapterCapabilities(content_fetch=True),
        safe_display_name="Engineering",
    )

    assert source.tenant_id == "tenant-1"
    assert descriptor.identity.remote_id == "item-1"
    assert permissions.visibility is KnowledgeVisibility.RESTRICTED
    assert content.mode is KnowledgeContentMode.RICH_TEXT
    assert page.has_more is True
    assert scope_info.capabilities.content_fetch is True


@pytest.mark.unit
def test_rejects_empty_identifiers() -> None:
    with pytest.raises(ValidationError):
        KnowledgeItemIdentity(remote_id="")
    with pytest.raises(ValidationError):
        KnowledgeSourceRef(
            tenant_id=" ",
            provider_id="example",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            scope=_scope(),
        )
    with pytest.raises(ValidationError):
        KnowledgePrincipal(principal_type="user", principal_id="")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: KnowledgeSourceRef(
                tenant_id=" ",
                provider_id="example",
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind="issues",
                scope=_scope(),
            ),
            "tenant_id",
        ),
        (
            lambda: KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id="",
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind="issues",
                scope=_scope(),
            ),
            "provider_id",
        ),
        (lambda: _scope(remote_scope_id=""), "remote_scope_id"),
        (lambda: KnowledgeItemRevision(version=" "), "version"),
        (lambda: _descriptor(title=""), "title"),
        (
            lambda: KnowledgePrincipal(principal_type="user", principal_id=""),
            "principal_id",
        ),
        (
            lambda: KnowledgeContent(
                mode=KnowledgeContentMode.RICH_TEXT,
                rich_text="body",
                mime_type=" ",
            ),
            "mime_type",
        ),
    ],
)
def test_empty_field_validation_names_the_field(
    factory: Callable[[], object], field_name: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        factory()
    assert field_name in str(exc_info.value)


@pytest.mark.unit
def test_identity_is_separate_from_revision() -> None:
    identity = _identity(remote_id="stable-id")
    provenance = _provenance(remote_id="stable-id")
    revision_a = KnowledgeItemRevision(content_hash="hash-a")
    revision_b = KnowledgeItemRevision(content_hash="hash-b", version="2")
    left = _descriptor(identity=identity, revision=revision_a, provenance=provenance)
    right = _descriptor(identity=identity, revision=revision_b, provenance=provenance)

    assert left.identity.remote_id == right.identity.remote_id == "stable-id"
    assert left.provenance.remote_id == "stable-id"
    assert left.revision.content_hash != right.revision.content_hash
    assert left.identity.model_dump() == right.identity.model_dump()


@pytest.mark.unit
def test_rejects_mismatched_identity_and_provenance_remote_ids() -> None:
    with pytest.raises(ValidationError):
        _descriptor(
            identity=_identity(remote_id="id-a"),
            provenance=_provenance(remote_id="id-b"),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "payload"),
    [
        (KnowledgeContentMode.BINARY, {"binary": b"pdf"}),
        (KnowledgeContentMode.RICH_TEXT, {"rich_text": "body"}),
        (KnowledgeContentMode.STRUCTURED_RECORD, {"structured_record": {"k": "v"}}),
    ],
)
def test_content_mode_accepts_matching_payload(
    mode: KnowledgeContentMode, payload: dict[str, object]
) -> None:
    content = KnowledgeContent(mode=mode, **payload)
    assert content.mode is mode


@pytest.mark.unit
def test_content_mode_rejects_mismatched_or_multiple_payloads() -> None:
    with pytest.raises(ValidationError):
        KnowledgeContent(mode=KnowledgeContentMode.BINARY, rich_text="x")
    with pytest.raises(ValidationError):
        KnowledgeContent(
            mode=KnowledgeContentMode.RICH_TEXT,
            rich_text="x",
            binary=b"y",
        )
    with pytest.raises(ValidationError):
        KnowledgeContent(mode=KnowledgeContentMode.STRUCTURED_RECORD)


@pytest.mark.unit
def test_rejects_secrets_in_parameters_and_metadata() -> None:
    with pytest.raises(ValidationError):
        _scope(parameters={"Token": "secret-value"})
    with pytest.raises(ValidationError):
        _scope(parameters={"nested": {"api_key": "x"}})
    with pytest.raises(ValidationError):
        _descriptor(metadata={"authorization": "Bearer x"})
    with pytest.raises(ValidationError):
        _descriptor(metadata={"items": [{"password": "p"}]})


@pytest.mark.unit
def test_accepts_safe_credential_ref() -> None:
    scope = _scope(parameters={"credential_ref": "vault://connection/1", "region": "eu"})
    descriptor = _descriptor(metadata={"credential_ref": "vault://item/1"})
    assert scope.parameters["credential_ref"] == "vault://connection/1"
    assert descriptor.metadata["credential_ref"] == "vault://item/1"


@pytest.mark.unit
def test_accepts_ordinary_text_containing_secret_words() -> None:
    scope = _scope(parameters={"note": "Reset the password using the token from email"})
    descriptor = _descriptor(metadata={"hint": "do not paste an api_key into chat"})
    assert "password" in str(scope.parameters["note"])
    assert "token" in str(scope.parameters["note"])
    assert "api_key" in str(descriptor.metadata["hint"])


@pytest.mark.unit
def test_rejects_credentials_in_nested_parameters_url() -> None:
    with pytest.raises(ValidationError):
        _scope(
            parameters={
                "nested": {"endpoint": "https://user:pass@example.test/api"},
            }
        )


@pytest.mark.unit
def test_rejects_token_query_in_nested_metadata_url() -> None:
    with pytest.raises(ValidationError):
        _descriptor(
            metadata={
                "links": [{"href": "https://example.test/item?token=abc"}],
            }
        )


@pytest.mark.unit
def test_rejects_unsafe_url_in_safe_locator() -> None:
    with pytest.raises(ValidationError):
        _provenance(safe_locator="https://user:secret@example.test/item")
    with pytest.raises(ValidationError):
        _provenance(safe_locator="https://example.test/item?api_key=x")


@pytest.mark.unit
def test_accepts_safe_non_url_locator() -> None:
    provenance = _provenance(safe_locator="pages/item-1")
    assert provenance.safe_locator == "pages/item-1"


@pytest.mark.unit
def test_rejects_url_with_embedded_password_or_token_query() -> None:
    with pytest.raises(ValidationError):
        _provenance(web_url="https://user:pass@example.test/item")
    with pytest.raises(ValidationError):
        _provenance(web_url="https://example.test/item?access_token=abc")
    with pytest.raises(ValidationError):
        _provenance(web_url="https://example.test/item?TOKEN=abc")


@pytest.mark.unit
def test_active_change_kinds_require_descriptor() -> None:
    for kind in (
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
        KnowledgeChangeKind.PERMISSIONS_CHANGED,
    ):
        with pytest.raises(ValidationError):
            KnowledgeChange(kind=kind, remote_id="item-1", descriptor=None)


@pytest.mark.unit
def test_deleted_and_revoked_allow_missing_descriptor() -> None:
    deleted = KnowledgeChange(kind=KnowledgeChangeKind.DELETED, remote_id="item-1")
    revoked = KnowledgeChange(kind=KnowledgeChangeKind.REVOKED, remote_id="item-1")
    assert deleted.descriptor is None
    assert revoked.descriptor is None


@pytest.mark.unit
def test_change_remote_id_must_match_descriptor() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id="other-id",
            descriptor=_descriptor(),
        )


@pytest.mark.unit
def test_has_more_requires_next_cursor() -> None:
    with pytest.raises(ValidationError):
        KnowledgePage(changes=(), has_more=True, next_cursor=None)


@pytest.mark.unit
def test_immutable_tuples_for_changes_and_principals() -> None:
    principal = KnowledgePrincipal(principal_type="user", principal_id="u-1")
    permissions = KnowledgePermissions(
        visibility=KnowledgeVisibility.PRIVATE,
        allowed_principals=(principal, principal),
        denied_principals=(
            KnowledgePrincipal(principal_type="group", principal_id="g-1"),
            KnowledgePrincipal(principal_type="group", principal_id="g-1"),
        ),
    )
    change = KnowledgeChange(
        kind=KnowledgeChangeKind.DELETED,
        remote_id="item-1",
    )
    page = KnowledgePage(changes=[change])  # type: ignore[arg-type]

    assert isinstance(permissions.allowed_principals, tuple)
    assert isinstance(permissions.denied_principals, tuple)
    assert len(permissions.allowed_principals) == 1
    assert len(permissions.denied_principals) == 1
    assert isinstance(page.changes, tuple)
    assert not hasattr(permissions.allowed_principals, "append")
    assert not hasattr(page.changes, "append")


@pytest.mark.unit
def test_capabilities_default_to_false() -> None:
    capabilities = KnowledgeAdapterCapabilities()
    dumped = capabilities.model_dump()
    assert dumped
    assert all(value is False for value in dumped.values())


@pytest.mark.unit
def test_model_dump_has_no_secret_bearing_fields() -> None:
    source = _source(
        scope=_scope(parameters={"credential_ref": "vault://x", "folder": "docs"})
    )
    descriptor = _descriptor(metadata={"credential_ref": "vault://y", "title_hint": "a"})
    page = KnowledgePage(
        changes=(
            KnowledgeChange(
                kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=descriptor,
            ),
        )
    )
    dumped = {
        "source": source.model_dump(mode="json"),
        "page": page.model_dump(mode="json"),
        "content": KnowledgeContent(
            mode=KnowledgeContentMode.BINARY,
            binary=b"abc",
            mime_type="application/pdf",
        ).model_dump(mode="json"),
    }
    serialized = repr(dumped).lower()
    for forbidden in (
        "access_token",
        "refresh_token",
        "api_key",
        "password",
        "authorization",
        "bearer ",
    ):
        assert forbidden not in serialized
    assert "credential_ref" in dumped["source"]["scope"]["parameters"]
