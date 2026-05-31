# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment


def parse_unstructured_html(source: str) -> list[ParsedDocumentFragment]:
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    docs = UnstructuredHTMLLoader(source).load()
    return [
        ParsedDocumentFragment(
            text=doc.page_content or "",
            metadata={"parser_backend": "unstructured", **(doc.metadata or {})},
        )
        for doc in docs
    ]
