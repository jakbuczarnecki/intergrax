# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.tools.providers.document.contracts import (
    DocumentFragmentOutput,
    DocumentParseInput,
    DocumentParseOutput,
    DocumentParsePreviewInput,
    DocumentParsePreviewOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

DOCUMENT_PARSE_TOOL_ID = "document.parse"
DOCUMENT_PARSE_PREVIEW_TOOL_ID = "document.parse_preview"


def _require_parser(ctx: ToolWiringContext) -> DocumentParser:
    parser = ctx.document_parser
    if parser is None:
        raise RuntimeError("document_parser_not_configured")
    return parser


def _to_fragment(fragment: ParsedDocumentFragment) -> DocumentFragmentOutput:
    return DocumentFragmentOutput(text=fragment.text, metadata=dict(fragment.metadata))


def document_parse(ctx: ToolWiringContext, params: DocumentParseInput) -> DocumentParseOutput:
    parser = _require_parser(ctx)
    if not parser.is_available():
        raise RuntimeError("document_parser_not_available")
    fragments = [_to_fragment(item) for item in parser.parse_file(params.source_path.strip())]
    return DocumentParseOutput(
        parser_id=parser.parser_id(),
        fragments=fragments,
        fragment_count=len(fragments),
    )


def document_parse_preview(
    ctx: ToolWiringContext,
    params: DocumentParsePreviewInput,
) -> DocumentParsePreviewOutput:
    parser = _require_parser(ctx)
    if not parser.is_available():
        raise RuntimeError("document_parser_not_available")
    raw_fragments = parser.parse_file(params.source_path.strip())
    truncated = len(raw_fragments) > params.max_fragments
    fragments: list[DocumentFragmentOutput] = []
    for item in raw_fragments[: params.max_fragments]:
        text = item.text
        if len(text) > params.max_chars_per_fragment:
            text = text[: params.max_chars_per_fragment]
            truncated = True
        fragments.append(DocumentFragmentOutput(text=text, metadata=dict(item.metadata)))
    return DocumentParsePreviewOutput(
        parser_id=parser.parser_id(),
        fragments=fragments,
        fragment_count=len(fragments),
        truncated=truncated,
    )
