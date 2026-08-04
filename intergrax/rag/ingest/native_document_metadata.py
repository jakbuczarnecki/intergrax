from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.document import RESERVED_METADATA_KEYS


def filter_native_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only metadata that belongs in the public native metadata mapping."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in RESERVED_METADATA_KEYS
    }


def add_native_metadata(
    document: KnowledgeDocument,
    metadata: Mapping[str, Any],
) -> KnowledgeDocument:
    """Return a fully validated document with non-reserved metadata added."""
    payload = document.model_dump(mode="json")
    merged_metadata = dict(payload["metadata"])
    merged_metadata.update(filter_native_metadata(metadata))
    payload["metadata"] = merged_metadata
    return KnowledgeDocument.model_validate(payload)
