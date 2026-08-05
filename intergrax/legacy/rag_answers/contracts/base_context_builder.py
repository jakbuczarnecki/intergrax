# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from intergrax.knowledge.contracts import KnowledgeDocument


class BaseContextBuilder(ABC):

    @abstractmethod
    def build(
        self,
        documents: Sequence[KnowledgeDocument],
        tokenizer_id: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:

        raise NotImplementedError