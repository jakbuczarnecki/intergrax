# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, List, Optional

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.validation import JsonValue


MetadataCallback = Callable[
    [KnowledgeDocument, Path | str],
    Mapping[str, JsonValue],
]


class BaseDocumentsLoader(ABC):

    @abstractmethod
    def load_document(
        self,
        source: str,
        *,
        tenant_id: str,
        namespace: str | None = None,
        use_default_metadata: bool = True,
        call_custom_metadata: Optional[MetadataCallback] = None,
    ) -> List[KnowledgeDocument]:
        """
        Load a single source (path/http/s3/etc.) using handler registry + metadata pipeline.

        Args:
            source: URI or path passed to the resolved handler.
            tenant_id: Required tenant scope for produced KnowledgeDocument instances.
            namespace: Optional namespace within the tenant scope.
            use_default_metadata: When True, run the configured metadata pipeline after normalize.
            call_custom_metadata: Optional callback ``(doc, path) -> dict`` merged into each
                document's metadata (after default enrichment when enabled).

        NOTE:
        - DocumentsLoader does NOT validate source correctness.
        - Handler is responsible for interpreting and validating the source.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Load directory
    # ---------------------------------------------------------

    @abstractmethod
    def load_documents(
        self,
        directory_path: str,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> List[KnowledgeDocument]:

        raise NotImplementedError
