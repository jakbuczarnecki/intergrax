# © Artur Czarnecki. All rights reserved.

"""Confluence provider-owned Indexed materialization strategies."""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
    CONFLUENCE_SPACE_SCOPE_TYPE,
    validate_confluence_space_id,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    VendorKnowledgeMaterializationError,
    build_materialized_connected_source_document,
    validate_materializer_source,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

CONFLUENCE_PAGES_RICH_TEXT_SCHEMA = "application/vnd.atlassian.confluence.storage+xml"

_MAX_CONFLUENCE_MATERIALIZED_CHARS = 8_000_000
_REMOTE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "ul",
    }
)
_IGNORED_CONTENT_TAGS = frozenset(
    {
        "ac:parameter",
        "ri:attachment",
        "ri:url",
        "script",
        "style",
        "title",
    }
)
_CONFLUENCE_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
    integration_category=IntegrationCategory.WIKI_KNOWLEDGE,
    source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
)


class ConfluencePageRichTextMaterializer:
    """Materialize safe text from one accepted Confluence storage body."""

    identity = _CONFLUENCE_IDENTITY
    runtime_ref = "indexed-source:confluence:pages"
    schema_name = CONFLUENCE_PAGES_RICH_TEXT_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        validate_materializer_source(self.identity, source)
        self._validate_identity(
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
        )
        self._validate_scope(source)
        self._validate_revision(revision)
        if content.mode is not KnowledgeContentMode.RICH_TEXT:
            raise VendorKnowledgeMaterializationError(
                "connected_source_content_mode_invalid"
            )
        if content.mime_type != self.schema_name or content.encoding != "utf-8":
            raise VendorKnowledgeMaterializationError(
                "connected_source_schema_unsupported"
            )
        if not isinstance(content.rich_text, str):
            raise VendorKnowledgeMaterializationError(
                "connected_source_rich_text_invalid"
            )
        markdown = _render_confluence_storage(content.rich_text)
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"confluence-page-{_remote_hash_prefix(remote_id)}.md",
            revision=revision,
            permissions=permissions,
        )

    @staticmethod
    def _validate_identity(
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
    ) -> None:
        if source.tenant_id != tenant_id or any(
            not isinstance(value, str) or not value.strip()
            for value in (tenant_id, workspace_id, binding_id, source_id)
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_identity_invalid"
            )
        if not isinstance(remote_id, str) or not _REMOTE_ID_RE.fullmatch(remote_id):
            raise VendorKnowledgeMaterializationError(
                "connected_source_remote_id_mismatch"
            )

    @staticmethod
    def _validate_scope(source: KnowledgeSourceRef) -> None:
        try:
            valid_space_id = validate_confluence_space_id(source.scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeMaterializationError(
                "connected_source_scope_invalid"
            ) from None
        if (
            source.scope.remote_scope_type != CONFLUENCE_SPACE_SCOPE_TYPE
            or source.scope.parameters
            or source.scope.remote_scope_id != valid_space_id
        ):
            raise VendorKnowledgeMaterializationError("connected_source_scope_invalid")

    @staticmethod
    def _validate_revision(revision: KnowledgeItemRevision | None) -> None:
        if (
            revision is None
            or not isinstance(revision.version, str)
            or not _REMOTE_ID_RE.fullmatch(revision.version)
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_revision_invalid"
            )


class _ConfluenceStorageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._heading_parts: list[str] = []
        self._heading_count = 0
        self._heading_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        tag = tag.lower()
        if self._ignored_depth:
            self._ignored_depth += 1
            return
        if tag in _IGNORED_CONTENT_TAGS:
            self._ignored_depth = 1
            return
        if tag == "h1":
            self._heading_count += 1
            self._heading_depth = 1
            self._heading_parts = []
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS or tag in {"table", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag == "h1":
            self._heading_depth = 0
        elif tag in _BLOCK_TAGS or tag in {"table", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._heading_depth:
            self._heading_parts.append(data)
        else:
            self._parts.append(data)

    @property
    def heading(self) -> str:
        return _normalize_line("".join(self._heading_parts))

    @property
    def body(self) -> str:
        lines = [_normalize_line(line) for line in "".join(self._parts).splitlines()]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        normalized: list[str] = []
        for line in lines:
            if not line and normalized and not normalized[-1]:
                continue
            normalized.append(line)
        return "\n".join(normalized)


def _render_confluence_storage(storage: str) -> str:
    if len(storage) > _MAX_CONFLUENCE_MATERIALIZED_CHARS:
        raise VendorKnowledgeMaterializationError("connected_source_content_too_large")
    parser = _ConfluenceStorageTextParser()
    try:
        parser.feed(storage)
        parser.close()
    except (TypeError, ValueError):
        raise VendorKnowledgeMaterializationError(
            "connected_source_rich_text_invalid"
        ) from None
    title = parser.heading
    body = parser.body
    if parser._heading_count != 1 or not title or not body:
        raise VendorKnowledgeMaterializationError(
            "connected_source_meaningful_content_missing"
        )
    markdown = f"# {title}\n\n{body}"
    if len(markdown) > _MAX_CONFLUENCE_MATERIALIZED_CHARS:
        raise VendorKnowledgeMaterializationError("connected_source_content_too_large")
    return markdown


def _normalize_line(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()


def _remote_hash_prefix(remote_id: str) -> str:
    return hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CONFLUENCE_PAGES_RICH_TEXT_SCHEMA",
    "ConfluencePageRichTextMaterializer",
]
