# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Docling openers — only this module may import ``docling`` / ``httpx`` for Docling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig, DoclingMode

_DOC_CONVERTER: Any = None


def _build_local_converter(config: DoclingIntegrationConfig) -> Any:
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.pipeline.simple_pipeline import SimplePipeline

    if config.simple_pdf_mode:
        return DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=DoclingParseDocumentBackend,
                )
            }
        )
    return DocumentConverter()


def open_docling_local_converter(config: DoclingIntegrationConfig) -> Any:
    global _DOC_CONVERTER
    if _DOC_CONVERTER is None:
        _DOC_CONVERTER = _build_local_converter(config)
    return _DOC_CONVERTER


def parse_file_local(config: DoclingIntegrationConfig, source: str) -> list[ParsedDocumentFragment]:
    converter = open_docling_local_converter(config)
    result = converter.convert(Path(source))
    doc = result.document
    text = doc.export_to_markdown() or ""
    return [
        ParsedDocumentFragment(
            text=text,
            metadata={"parser_backend": "docling.local"},
            native_handle=doc,
        )
    ]


def parse_file_server(config: DoclingIntegrationConfig, source: str) -> list[ParsedDocumentFragment]:
    import httpx

    url = config.server_url.rstrip("/") + config.server_path
    with open(source, "rb") as handle:
        response = httpx.post(
            url,
            files={"file": (Path(source).name, handle)},
            timeout=config.timeout_seconds,
        )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("markdown") or payload.get("text") or ""
    return [
        ParsedDocumentFragment(
            text=text,
            metadata={"parser_backend": "docling.server"},
            native_handle=None,
        )
    ]


def parse_docling_file(
    config: DoclingIntegrationConfig,
    source: str,
    *,
    converter: Optional[Any] = None,
) -> list[ParsedDocumentFragment]:
    if config.mode is DoclingMode.NONE:
        raise RuntimeError("docling_not_configured")
    if config.mode is DoclingMode.SERVER:
        return parse_file_server(config, source)
    if converter is not None:
        global _DOC_CONVERTER
        _DOC_CONVERTER = converter
    return parse_file_local(config, source)
