# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace Docs knowledge source adapter."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_NATIVE_MIME_TYPE,
    GOOGLE_DOCS_SOURCE_KIND,
    GoogleDocsDocument,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
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
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE = "google_workspace_docs_document"

GOOGLE_DOCS_CURSOR_VERSION = "google_workspace.docs.cursor.v1"

GOOGLE_DOCS_ITEM_METADATA_VERSION = "google_workspace.docs.item.v1"

GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA = "google_workspace.docs.document.knowledge.v1"

GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE = (
    "application/vnd.intergrax.google-workspace-docs+json"
)

_GOOGLE_DOCS_ITEM_TYPE = "google_workspace_docs_document"

_INVALID_SCOPE_MESSAGE = "Google Workspace Docs knowledge source scope is invalid"
_INVALID_CURSOR_MESSAGE = "Google Workspace Docs knowledge cursor is invalid"
_COMPLETE_CURSOR_MESSAGE = (
    "Google Workspace Docs reconciliation cursor is complete; restart reconciliation"
)
_INVALID_PROVIDER_RESPONSE_MESSAGE = (
    "Google Workspace Docs knowledge provider response is invalid"
)
_INVALID_DESCRIPTOR_MESSAGE = "Google Workspace Docs document descriptor is invalid"
_CONFIGURATION_ERROR_MESSAGE = "Google Workspace Docs knowledge page limit is invalid"
_DEPENDENCY_UNAVAILABLE_MESSAGE = (
    "Google Workspace Docs knowledge dependency is unavailable"
)
_CONTENT_HASH_MISMATCH_MESSAGE = (
    "Google Workspace Docs document content changed since descriptor creation"
)
_UNSUPPORTED_PERMISSIONS_MESSAGE = (
    "Authoritative Google Docs permissions projection is not implemented"
)
_INTEGRATION_REQUIRED_MESSAGE = (
    "Google Workspace Docs knowledge adapter requires "
    "Google Workspace collaboration-suite integration"
)

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_MAX_DOCUMENT_ID_LENGTH = 1024
_MAX_ENCODED_CURSOR_LENGTH = 24_576
_CURSOR_ALPHABET = re.compile(r"^[A-Za-z0-9_-]+$")

_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "structured_record_schema",
        "native_mime_type",
        "tab_count",
    }
)

_T = TypeVar("_T")


class _GoogleDocsReconciliationCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["google_workspace.docs.cursor.v1"]
    scope_fingerprint: str = Field(repr=False)
    complete: Literal[True]


class GoogleWorkspaceDocsKnowledgeAdapter:
    """Thin mapping from Google Workspace Docs integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return GOOGLE_DOCS_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
            content_fetch=True,
            binary_content=False,
            rich_text_content=False,
            structured_content=True,
            permissions=False,
            tombstones=False,
            remote_versions=False,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        return KnowledgeScopeInfo(
            source=validated_source,
            capabilities=self.capabilities,
            safe_display_name=validated_source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        google_integration = self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        document_id = validated_source.scope.remote_scope_id
        self._validate_limit(limit)
        decoded = self._decode_cursor(cursor, document_id=document_id)
        if decoded is not None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=_COMPLETE_CURSOR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

        document = await self._invoke_integration(
            lambda: google_integration.read_docs_document(document_id=document_id),
        )
        validated_document = self._reconstruct_document(
            document,
            expected_document_id=document_id,
        )
        record = _build_structured_record(validated_document)
        content_hash = _compute_content_hash(record)
        descriptor = self._document_to_descriptor(validated_document, content_hash=content_hash)
        upsert = KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=validated_document.document_id,
            descriptor=descriptor,
        )
        checkpoint = self._encode_complete_cursor(document_id=document_id)
        return KnowledgePage(
            changes=(upsert,),
            next_cursor=None,
            proposed_checkpoint=checkpoint,
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        google_integration = self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        document_id = validated_source.scope.remote_scope_id
        descriptor = self._reconstruct_descriptor(item)
        self._validate_descriptor(descriptor, source=validated_source, document_id=document_id)

        document = await self._invoke_integration(
            lambda: google_integration.read_docs_document(document_id=document_id),
        )
        validated_document = self._reconstruct_document(
            document,
            expected_document_id=document_id,
        )
        record = _build_structured_record(validated_document)
        content_hash = _compute_content_hash(record)
        expected_hash = descriptor.revision.content_hash
        if expected_hash is None or content_hash != expected_hash:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_CONTENT_HASH_MISMATCH_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            )
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record=record,
            mime_type=GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE,
            content_hash=content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        document_id = validated_source.scope.remote_scope_id
        descriptor = self._reconstruct_descriptor(item)
        self._validate_descriptor(descriptor, source=validated_source, document_id=document_id)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=_UNSUPPORTED_PERMISSIONS_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_scope_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_SCOPE_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_cursor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_CURSOR,
            safe_message=_INVALID_CURSOR_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_provider_response_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_descriptor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_DESCRIPTOR_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _integration_required_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INTEGRATION_REQUIRED_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _require_google_integration(
        self,
        integration: object,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        if not isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration):
            raise self._integration_required_error()
        return integration

    def _reconstruct_source(self, source: object) -> KnowledgeSourceRef:
        try:
            if type(source) is not KnowledgeSourceRef:
                raise ValueError("invalid source type")
            scope_raw = source.scope
            if type(scope_raw.safe_display_name) is not str:
                raise ValueError("invalid safe_display_name")
            snapshot = source.model_dump(mode="python")
            scope_data = snapshot.get("scope")
            if isinstance(scope_data, dict):
                snapshot["scope"] = KnowledgeSourceScope(**scope_data)
            return KnowledgeSourceRef(**snapshot)
        except Exception:
            raise self._invalid_scope_error() from None

    def _validate_scope_document_id(self, value: object) -> str:
        if type(value) is not str:
            raise ValueError("invalid document id type")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("blank document id")
        if _ASCII_CONTROL.search(trimmed):
            raise ValueError("control character in document id")
        if len(trimmed) > _MAX_DOCUMENT_ID_LENGTH:
            raise ValueError("document id too long")
        if "/" in trimmed or "\\" in trimmed:
            raise ValueError("path separator in document id")
        return trimmed

    def _validate_source(self, source: KnowledgeSourceRef) -> KnowledgeSourceRef:
        reconstructed = self._reconstruct_source(source)
        if (
            reconstructed.provider_id != self.provider_id
            or reconstructed.integration_kind != self.integration_kind
            or reconstructed.source_kind != self.source_kind
        ):
            raise self._invalid_scope_error()
        scope = reconstructed.scope
        if scope.remote_scope_type != GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE:
            raise self._invalid_scope_error()
        if scope.parameters:
            raise self._invalid_scope_error()
        try:
            validated_id = self._validate_scope_document_id(scope.remote_scope_id)
        except ValueError:
            raise self._invalid_scope_error() from None
        if validated_id != scope.remote_scope_id:
            raise self._invalid_scope_error()
        return KnowledgeSourceRef(
            tenant_id=reconstructed.tenant_id,
            provider_id=reconstructed.provider_id,
            integration_kind=reconstructed.integration_kind,
            source_kind=reconstructed.source_kind,
            connection_ref=reconstructed.connection_ref,
            scope=KnowledgeSourceScope(
                remote_scope_id=validated_id,
                remote_scope_type=scope.remote_scope_type,
                safe_display_name=scope.safe_display_name,
                parameters={},
            ),
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _scope_fingerprint(self, document_id: str) -> str:
        payload = f"google_workspace\x00docs\x00{document_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _encode_complete_cursor(self, *, document_id: str) -> KnowledgeCursor:
        cursor = _GoogleDocsReconciliationCursor(
            schema_version=GOOGLE_DOCS_CURSOR_VERSION,
            scope_fingerprint=self._scope_fingerprint(document_id),
            complete=True,
        )
        return self._encode_cursor(cursor)

    def _encode_cursor(self, cursor: _GoogleDocsReconciliationCursor) -> KnowledgeCursor:
        payload = cursor.model_dump()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=GOOGLE_DOCS_CURSOR_VERSION)

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        document_id: str,
    ) -> _GoogleDocsReconciliationCursor | None:
        if cursor is None:
            return None
        try:
            if type(cursor) is not KnowledgeCursor:
                raise ValueError("invalid cursor type")
            snapshot = cursor.model_dump(mode="python")
            reconstructed_cursor = KnowledgeCursor(**snapshot)
            outer_value = reconstructed_cursor.value
            if type(outer_value) is not str:
                raise ValueError("invalid cursor value type")
            if not outer_value:
                raise ValueError("blank cursor value")
            if outer_value != outer_value.strip():
                raise ValueError("cursor whitespace")
            if len(outer_value) > _MAX_ENCODED_CURSOR_LENGTH:
                raise ValueError("cursor too long")
            if "=" in outer_value:
                raise ValueError("cursor padding")
            if _CURSOR_ALPHABET.fullmatch(outer_value) is None:
                raise ValueError("cursor alphabet")
            if reconstructed_cursor.version != GOOGLE_DOCS_CURSOR_VERSION:
                raise ValueError("cursor version mismatch")
            padding = "=" * (-len(outer_value) % 4)
            raw = base64.b64decode(outer_value + padding, altchars=b"-_", validate=True)
            data = json.loads(raw.decode("utf-8"))
            if type(data) is not dict:
                raise ValueError("cursor must be object")
            decoded = _GoogleDocsReconciliationCursor.model_validate(data)
            canonical = self._encode_cursor(decoded)
            if (
                canonical.value != reconstructed_cursor.value
                or canonical.version != reconstructed_cursor.version
            ):
                raise ValueError("noncanonical cursor")
        except Exception:
            raise self._invalid_cursor_error() from None
        if decoded.scope_fingerprint != self._scope_fingerprint(document_id):
            raise self._invalid_cursor_error()
        return decoded

    def _reconstruct_document(
        self,
        document: object,
        *,
        expected_document_id: str,
    ) -> GoogleDocsDocument:
        try:
            if type(document) is not GoogleDocsDocument:
                raise ValueError("invalid document type")
            snapshot = document.model_dump(mode="python")
            validated = GoogleDocsDocument(**snapshot)
        except Exception:
            raise self._invalid_provider_response_error() from None
        if validated.document_id != expected_document_id:
            raise self._invalid_provider_response_error()
        return validated

    def _document_to_descriptor(
        self,
        document: GoogleDocsDocument,
        *,
        content_hash: str,
    ) -> KnowledgeItemDescriptor:
        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=document.document_id,
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=None,
                etag=None,
                content_hash=content_hash,
                acl_hash=None,
                updated_at=None,
            ),
            title=document.title,
            item_type=_GOOGLE_DOCS_ITEM_TYPE,
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=document.document_id,
                web_url=None,
                safe_locator=None,
            ),
            metadata={
                "schema_version": GOOGLE_DOCS_ITEM_METADATA_VERSION,
                "structured_record_schema": GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
                "native_mime_type": GOOGLE_DOCS_NATIVE_MIME_TYPE,
                "tab_count": len(document.tabs),
            },
        )

    def _reconstruct_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        try:
            if type(item) is not KnowledgeItemDescriptor:
                raise ValueError("invalid descriptor type")
            snapshot = item.model_dump(mode="python")
            identity_data = snapshot.get("identity")
            if isinstance(identity_data, dict):
                snapshot["identity"] = KnowledgeItemIdentity(**identity_data)
            revision_data = snapshot.get("revision")
            if isinstance(revision_data, dict):
                snapshot["revision"] = KnowledgeItemRevision(**revision_data)
            provenance_data = snapshot.get("provenance")
            if isinstance(provenance_data, dict):
                snapshot["provenance"] = KnowledgeItemProvenance(**provenance_data)
            return KnowledgeItemDescriptor(**snapshot)
        except Exception:
            raise self._invalid_descriptor_error() from None

    def _validate_descriptor(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        document_id: str,
    ) -> None:
        try:
            if item.identity.remote_id != document_id:
                raise ValueError("identity remote id mismatch")
            if item.provenance.remote_id != document_id:
                raise ValueError("provenance remote id mismatch")
            if item.provenance.provider_id != self.provider_id:
                raise ValueError("provider mismatch")
            if item.provenance.source_kind != self.source_kind:
                raise ValueError("source kind mismatch")
            if item.identity.parent_remote_id is not None:
                raise ValueError("parent must be None")
            if item.identity.logical_key is not None:
                raise ValueError("logical key must be None")
            if item.item_type != _GOOGLE_DOCS_ITEM_TYPE:
                raise ValueError("item type mismatch")
            if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
                raise ValueError("content mode mismatch")
            if not item.content_available:
                raise ValueError("content unavailable")
            revision = item.revision
            if revision.version is not None:
                raise ValueError("version must be None")
            if revision.etag is not None:
                raise ValueError("etag must be None")
            if revision.acl_hash is not None:
                raise ValueError("acl hash must be None")
            if revision.updated_at is not None:
                raise ValueError("updated_at must be None")
            content_hash = revision.content_hash
            if type(content_hash) is not str or _SHA256_HEX_RE.fullmatch(content_hash) is None:
                raise ValueError("invalid content hash")
            if item.provenance.web_url is not None:
                raise ValueError("web url must be None")
            if item.provenance.safe_locator is not None:
                raise ValueError("safe locator must be None")
            metadata = item.metadata
            if not isinstance(metadata, dict):
                raise ValueError("invalid metadata")
            if set(metadata.keys()) != _METADATA_KEYS:
                raise ValueError("metadata keys mismatch")
            if metadata.get("schema_version") != GOOGLE_DOCS_ITEM_METADATA_VERSION:
                raise ValueError("metadata schema mismatch")
            if metadata.get("structured_record_schema") != GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA:
                raise ValueError("structured record schema mismatch")
            if metadata.get("native_mime_type") != GOOGLE_DOCS_NATIVE_MIME_TYPE:
                raise ValueError("native mime mismatch")
            tab_count = metadata.get("tab_count")
            if type(tab_count) is not int:
                raise ValueError("tab count type mismatch")
            if (
                item.provenance.provider_id != source.provider_id
                or item.provenance.source_kind != source.source_kind
            ):
                raise ValueError("provenance source mismatch")
        except Exception:
            raise self._invalid_descriptor_error() from None

    def _safe_message_for_code(self, code: VendorKnowledgeErrorCode) -> str:
        if code is VendorKnowledgeErrorCode.INVALID_CURSOR:
            return _INVALID_CURSOR_MESSAGE
        if code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR:
            return _CONFIGURATION_ERROR_MESSAGE
        if code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE:
            return _DEPENDENCY_UNAVAILABLE_MESSAGE
        if code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY:
            return _UNSUPPORTED_PERMISSIONS_MESSAGE
        return _INVALID_PROVIDER_RESPONSE_MESSAGE

    def _map_google_api_error(self, exc: GoogleWorkspaceApiError) -> VendorKnowledgeError:
        if exc.kind is GoogleWorkspaceErrorKind.AUTHENTICATION:
            code = VendorKnowledgeErrorCode.AUTHENTICATION_FAILED
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.AUTHORIZATION:
            code = VendorKnowledgeErrorCode.AUTHORIZATION_DENIED
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.NOT_FOUND:
            code = VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.RATE_LIMITED:
            code = VendorKnowledgeErrorCode.RATE_LIMITED
            retryable = True
        elif exc.kind is GoogleWorkspaceErrorKind.TEMPORARY:
            code = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
            retryable = True
        elif exc.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE:
            code = VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT:
            code = VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE:
            code = VendorKnowledgeErrorCode.CONFIGURATION_ERROR
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST:
            code = VendorKnowledgeErrorCode.CONFIGURATION_ERROR
            retryable = False
        else:
            code = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
            retryable = True
        return VendorKnowledgeError(
            code=code,
            safe_message=self._safe_message_for_code(code),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=retryable,
        )

    async def _invoke_integration(
        self,
        operation: Callable[[], _T],
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except GoogleWorkspaceApiError as exc:
            raise self._map_google_api_error(exc) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_provider_response_error() from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def _build_structured_record(document: GoogleDocsDocument) -> dict[str, Any]:
    return {
        "schema_version": GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
        "document_id": document.document_id,
        "title": document.title,
        "suggestions_view_mode": document.suggestions_view_mode,
        "tabs": [tab.model_dump(mode="json") for tab in document.tabs],
    }


def _compute_content_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def register_google_workspace_docs_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> GoogleWorkspaceDocsKnowledgeAdapter:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    registry.register(adapter)
    return adapter
