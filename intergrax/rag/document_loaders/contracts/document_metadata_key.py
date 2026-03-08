# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


class DocumentMetadataKey(str, Enum):
    """
    Strongly typed metadata keys for document loader pipeline.
    """

    SOURCE = "source"
    PARSER = "parser"
    DOCUMENT_ID = "document_id"
    POSITION = "position"

    SOURCE_PATH = "source_path"
    SOURCE_NAME = "source_name"
    EXTENSION = "ext"

    PAGE_INDEX = "page_index"

    PARENT_ID = "parent_id"

    DOCLING_DOCUMENT_META = "_docling_document"