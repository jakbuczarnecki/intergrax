# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document


class BaseContextBuilder(ABC):

    @abstractmethod
    def build(
        self,
        documents: List[Document],
        tokenizer_id: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:

        raise NotImplementedError