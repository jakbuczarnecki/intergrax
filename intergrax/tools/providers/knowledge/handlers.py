# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.knowledge.contracts import (
    KnowledgeGetPageInput,
    KnowledgePageOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from intergrax.tools.providers.knowledge.service import knowledge_get_page, knowledge_search


class KnowledgeGetPageHandler(ServiceToolHandler[KnowledgeGetPageInput, KnowledgePageOutput]):
    _service = knowledge_get_page


class KnowledgeSearchHandler(ServiceToolHandler[KnowledgeSearchInput, KnowledgeSearchOutput]):
    _service = knowledge_search
