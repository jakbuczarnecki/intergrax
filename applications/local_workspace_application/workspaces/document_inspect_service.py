# © Artur Czarnecki. All rights reserved.

"""Host-mediated provider-neutral document inspect boundary (LKW-PRODUCT-3E)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from local_workspace_application.serving.source_projection import safe_source_label
from local_workspace_application.workspaces.models import (
    KnowledgeInputKind,
    WorkspaceDocumentReference,
    WorkspaceSource,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_MAX_PREVIEW_CHARS = 1200
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_TRAVERSAL_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)|(^|[\\/])\.\.?$")
_UNSAFE_DOCUMENT_ID_RE = re.compile(r"[\\/]|^\.\.?$")


class DocumentInspectLocationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    logical_location: str | None = None


class DocumentInspectViewV1(BaseModel):
    """Safe, provider-neutral document projection for inspect/open clients."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str
    display_name: str
    source_type: str
    source_label: str
    logical_location: str | None = None
    location: DocumentInspectLocationV1 | None = None
    preview: str | None = None
    external_url: str | None = None


class DocumentInspectError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _safe_text(value: object, *, limit: int) -> str:
    text = _CONTROL_RE.sub(" ", str(value or ""))
    text = " ".join(text.split()).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _bound_preview(value: str | None) -> str | None:
    if value is None:
        return None
    bounded = _safe_text(value, limit=_MAX_PREVIEW_CHARS)
    return bounded or None


def _validate_document_id(document_id: str) -> str:
    trimmed = document_id.strip()
    if not trimmed:
        raise DocumentInspectError("document_not_found")
    if _UNSAFE_DOCUMENT_ID_RE.search(trimmed) is not None:
        raise DocumentInspectError("document_not_found")
    if _PATH_TRAVERSAL_RE.search(trimmed) is not None:
        raise DocumentInspectError("document_not_found")
    return trimmed


class DocumentInspectService:
    """Resolve indexed documents through canonical workspace ownership only."""

    def __init__(self, *, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    def inspect(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        preview_hint: str | None = None,
        page: int | None = None,
        logical_location_hint: str | None = None,
    ) -> DocumentInspectViewV1:
        normalized_document_id = _validate_document_id(document_id)
        reference = self._repository.get_document_ref(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_id=normalized_document_id,
        )
        if reference is None:
            raise DocumentInspectError("document_not_found")
        if (
            reference.tenant_id != tenant_id
            or reference.workspace_id != workspace_id
        ):
            raise DocumentInspectError("document_forbidden")

        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=reference.source_id,
        )
        if source is None:
            raise DocumentInspectError("document_not_found")

        display_name = _safe_text(reference.file_name, limit=200) or "Document"
        source_label = safe_source_label(
            source_type=source.source_type.value,
            path=source.path or None,
        )
        source_type = source.source_type.value
        logical_location = _safe_text(
            logical_location_hint or display_name,
            limit=200,
        ) or None
        external_url = self._resolve_external_url(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source=source,
        )
        preview = _bound_preview(preview_hint)
        location = DocumentInspectLocationV1(
            page=page,
            logical_location=logical_location,
        )
        if location.page is None and location.logical_location is None:
            location = None

        return DocumentInspectViewV1(
            document_id=reference.document_id,
            source_id=reference.source_id,
            display_name=display_name,
            source_type=source_type,
            source_label=source_label,
            logical_location=logical_location,
            location=location,
            preview=preview,
            external_url=external_url,
        )

    def _resolve_external_url(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source: WorkspaceSource,
    ) -> str | None:
        if source.source_type is WorkspaceSourceType.WEB_RESOURCE:
            return self._resolve_web_url_open_target(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source.source_id,
            )
        return None

    def _resolve_web_url_open_target(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> str | None:
        input_id: str | None = None
        for knowledge_input in self._repository.list_knowledge_inputs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            if (
                knowledge_input.source_id == source_id
                and knowledge_input.input_kind is KnowledgeInputKind.WEB_URL
            ):
                input_id = knowledge_input.input_id
                break
        if input_id is None:
            return None
        for locator in self._repository.list_web_url_locators(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            if locator.input_id != input_id:
                continue
            target = locator.final_safe_display_url or locator.safe_display_url
            return _safe_text(target, limit=500) or None
        return None


def managed_file_display_name(
    repository: ManagedWorkspaceRepository,
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
) -> str | None:
    for managed in repository.list_managed_files(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    ):
        if managed.source_id == source_id:
            return _safe_text(managed.safe_file_name, limit=200) or None
    return None


def inspect_reference_metadata(
    reference: WorkspaceDocumentReference,
) -> tuple[str, str | None]:
    display_name = _safe_text(reference.file_name, limit=200) or "Document"
    return display_name, None
