# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.unstructured.opens import parse_unstructured_html


class UnstructuredDocumentParser:
    def parser_id(self) -> str:
        return "unstructured"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_unstructured_html(source)


def create_unstructured_document_parser(**_: object) -> DocumentParser:
    return UnstructuredDocumentParser()
