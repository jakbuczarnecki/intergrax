# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Optional, List
from pydantic import BaseModel, ConfigDict

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.session.session_manager import SessionManager

class LegalAgentConfig(BaseModel):
    """
    Full configuration for the Legal Agent (tier-2).

    This is a single source of truth for the Legal Agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_manager: SessionManager
    llm_adapter: LLMAdapter

    production_mode: bool = True
    
    enable_websearch: bool = False


    enable_rag: bool = False
    embedding_manager: Optional[BaseEmbeddingManager] = None
    vectorstore_manager: Optional[BaseVectorstoreManager] = None
    documents_loader: Optional[BaseDocumentsLoader] = None
    documents_splitter: Optional[BaseDocumentsSplitter] = None

    governance_service: Optional[GovernanceService] = None
