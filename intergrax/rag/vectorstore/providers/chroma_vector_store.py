# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible shim — ``chromadb`` imports live in Integration Library ``rag_store``."""

from intergrax.integrations.providers.vector_store.chroma.rag_store import ChromaConfig, ChromaVectorStore

__all__ = ["ChromaConfig", "ChromaVectorStore"]
