# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTokenizerManager(ABC):
    
    @abstractmethod
    def count_tokens(
        self,
        text: str,
        *,
        tokenizer_id: str | None = None,
    ) -> int:
        raise NotImplementedError