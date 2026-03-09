# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document



class BaseDocumentsLoader(ABC):
    
    @abstractmethod
    def load_document(self, source: str) -> List[Document]:
        """
        Load a single source (path/http/s3/etc.) using handler registry + metadata pipeline.

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
    ) -> List[Document]:

        raise NotImplementedError