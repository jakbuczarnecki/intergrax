# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod



class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        *,
        query: str,
        context: str,
    ) -> str:

        raise NotImplementedError