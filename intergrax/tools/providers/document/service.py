# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.tools.providers.document.contracts import (
    DocumentFragmentOutput,
    DocumentParseInput,
    DocumentParseOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

DOCUMENT_PARSE_TOOL_ID = "document.parse"


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
