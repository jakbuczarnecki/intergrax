# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment


def parse_unstructured_html(source: str) -> list[ParsedDocumentFragment]:
    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
    except ModuleNotFoundError as exc:
        if exc.name == "langchain_community":
            raise RuntimeError(
                "Provider 'unstructured' requires optional dependency group "
                "'rag-langchain-loaders'. Install Intergrax with "
                "'rag-langchain-loaders'."
            ) from exc
        raise

    docs = UnstructuredHTMLLoader(source).load()
    return [
        ParsedDocumentFragment(
            text=doc.page_content or "",
            metadata={"parser_backend": "unstructured", **(doc.metadata or {})},
        )
        for doc in docs
    ]
