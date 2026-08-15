#!/usr/bin/env python3
"""Qualify a default Intergrax installation without LangChain or LangGraph."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import importlib.abc
import os
import site
import sys
import tomllib
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from intergrax.utils import attribute_access


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PREFIXES = ("langchain", "langgraph")
CHECKED_DISTRIBUTIONS = (
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",
    "langchain-ollama",
    "langchain-text-splitters",
    "langgraph",
)


class GateFailure(RuntimeError):
    """A fail-closed qualification failure with a reportable reason."""


def _normal_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _is_forbidden_name(name: str) -> bool:
    normalized = name.lower().replace("_", "-").replace(".", "-")
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}-")
        for prefix in FORBIDDEN_PREFIXES
    )


def _installed_distribution_names() -> list[str]:
    return sorted(
        {
            _normal_distribution_name(dist.metadata["Name"])
            for dist in metadata.distributions()
            if dist.metadata.get("Name")
        }
    )


def _print_distribution_inventory() -> None:
    installed = set(_installed_distribution_names())
    for name in CHECKED_DISTRIBUTIONS:
        state = "INSTALLED" if name in installed else "NOT INSTALLED"
        print(f"[gate] distribution {name}: {state}")
    forbidden = sorted(name for name in installed if _is_forbidden_name(name))
    if forbidden:
        raise GateFailure(
            "forbidden LangChain/LangGraph distributions installed: "
            + ", ".join(forbidden)
        )


def _check_project_dependencies() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    dependencies = [str(item) for item in project.get("dependencies", [])]
    leaked = [
        dependency
        for dependency in dependencies
        if _is_forbidden_name(dependency.split()[0].split("[", 1)[0])
    ]
    if leaked:
        raise GateFailure(
            "[project].dependencies contains LangChain/LangGraph: "
            + ", ".join(leaked)
        )
    print("[gate] project.dependencies LangChain/LangGraph entries: none")


class _ForbiddenImportFinder(importlib.abc.MetaPathFinder):
    """Prevent optional packages from being available through another path."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any = None,
    ) -> Any:
        del path, target
        if _is_forbidden_name(fullname):
            raise ModuleNotFoundError(
                f"blocked optional import during clean-install gate: {fullname}",
                name=fullname,
            )
        return None


def _install_import_blockade() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != user_site
    ]
    for module_name in list(sys.modules):
        if _is_forbidden_name(module_name):
            del sys.modules[module_name]
    sys.meta_path.insert(0, _ForbiddenImportFinder())
    print("[gate] import blockade langchain*/langgraph*: active")


def _run_core_imports() -> None:
    import intergrax
    import intergrax.llm_adapters.contracts
    import intergrax.llm.messages
    from intergrax.knowledge.contracts import KnowledgeDocument

    origin = Path(intergrax.__file__ or "").resolve()
    if REPO_ROOT in origin.parents or origin == REPO_ROOT:
        raise GateFailure(f"core imported from checkout instead of installed package: {origin}")
    if KnowledgeDocument.__module__ != "intergrax.knowledge.contracts.document":
        raise GateFailure("KnowledgeDocument resolved from an unexpected module")
    print(f"[gate] installed package origin: {origin}")


def _run_registry() -> None:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

    registered = set(LLMAdapterRegistry.registered_providers())
    required = {LLMProvider.OLLAMA.value, LLMProvider.OPENAI.value}
    if not required.issubset(registered):
        raise GateFailure(f"native providers missing from registry: {sorted(required - registered)}")
    if any(_is_forbidden_name(name) for name in registered):
        raise GateFailure("registry exposed a forbidden optional provider")
    print(f"[gate] registered providers: {len(registered)} (native Ollama/OpenAI present)")


def _run_native_ollama() -> None:
    from intergrax.llm.messages import ChatMessage
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.registry.catalog_capabilities import (
        unwrap_catalog_capability_adapter,
    )

    class _FakeNativeOllamaClient:
        def chat(self, **kwargs: object) -> object:
            if kwargs.get("stream") is not False:
                raise AssertionError("deterministic smoke expected non-streaming chat")
            return {
                "message": {"content": "native ollama gate response"},
                "prompt_eval_count": 3,
                "eval_count": 2,
            }

    adapter = LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        client=_FakeNativeOllamaClient(),
        model="gate-native-model",
    )
    concrete = unwrap_catalog_capability_adapter(adapter)
    if not isinstance(concrete, NativeOllamaAdapter):
        raise GateFailure(f"registry returned unexpected Ollama adapter: {type(concrete)!r}")
    response = concrete.generate_messages(
        [ChatMessage(role="user", content="native installation gate")]
    )
    if response.content != "native ollama gate response":
        raise GateFailure(f"unexpected Ollama response content: {response.content!r}")
    if response.usage is None or response.usage.input_tokens != 3:
        raise GateFailure(f"unexpected Ollama usage shape: {response.usage!r}")
    print("[gate] NativeOllamaAdapter construction and deterministic ABI smoke: PASS")


def _run_native_openai() -> None:
    module = importlib.import_module(
        "intergrax.llm_adapters.providers.openai_responses_adapter"
    )
    adapter_class = attribute_access.optional(module, "OpenAIChatResponsesAdapter", None)
    if adapter_class is None:
        raise GateFailure("OpenAIChatResponsesAdapter is not exported by native module")
    print("[gate] native OpenAI module/class import: PASS (no network, no credentials)")


def _make_document(document_id: str, content: str, *, tenant_id: str) -> Any:
    from intergrax.knowledge.contracts import KnowledgeDocument

    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": tenant_id,
                "namespace": "clean-install-gate",
            },
            "content": content,
            "metadata": {"gate": "langchain-free"},
            "provenance": {"source_kind": "gate", "source_id": document_id},
        }
    )


def _run_knowledge_document() -> None:
    document = _make_document(
        "gate-document",
        "The native Intergrax core works without optional compatibility packages.",
        tenant_id="gate-tenant",
    )
    if (
        document.scope.tenant_id != "gate-tenant"
        or document.scope.namespace != "clean-install-gate"
        or "native Intergrax" not in document.content
        or document.identity.document_id != "gate-document"
    ):
        raise GateFailure("KnowledgeDocument identity/scope/content contract failed")
    print("[gate] KnowledgeDocument identity, tenant, optional scope and content: PASS")


def _run_native_rag_read_only() -> None:
    try:
        from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
            RecursiveChunkingStrategy,
        )
        from intergrax.rag.profiles.rag_profile import RagProfile
        from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
        from intergrax.rag.retrieval.retrieval_service import RetrievalService
        from intergrax.rag.retrievers.contracts.base_retriever import (
            RetrieverQuery,
        )
        from intergrax.rag.retrievers.contracts.base_retriever_manager import (
            BaseRetrieverManager,
        )
        from intergrax.rag.retrievers.providers.vector_similarity_retriever import (
            VectorSimilarityRetriever,
        )
        from intergrax.rag.vectorstore.contracts.native_vectorstore import (
            MetadataFilter,
            VectorStoreScope,
        )
        from intergrax.rag.vectorstore.providers.inmemory_vectorstore import (
            InMemoryVectorStore,
        )
        from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
    except ModuleNotFoundError as exc:
        if exc.name and _is_forbidden_name(exc.name):
            raise GateFailure(
                "RAG_CONSUMER_REGRESSION_FOUND: "
                f"canonical native RAG imported optional module {exc.name}"
            ) from exc
        raise

    tenant_id = "gate-tenant"
    scope = VectorStoreScope(tenant_id=tenant_id, namespace="clean-install-gate")
    source = _make_document(
        "rag-gate-source",
        "Known native retrieval marker: LangChain-free Intergrax retrieval.",
        tenant_id=tenant_id,
    )
    chunks = RecursiveChunkingStrategy(chunk_size=256, chunk_overlap=0).chunk([source])
    if not chunks:
        raise GateFailure("canonical native splitter returned no chunks")

    vector_manager = VectorstoreManager(InMemoryVectorStore(tenant_id=tenant_id))
    vector_manager.add_documents(
        chunks,
        [[1.0, 0.0] for _chunk in chunks],
        scope=scope,
    )

    class _FixedEmbeddingFixture:
        def embed_one(self, text: str) -> tuple[float, float]:
            del text
            return (1.0, 0.0)

    class _NativeRetrieverManager(BaseRetrieverManager):
        def __init__(self, retriever: VectorSimilarityRetriever) -> None:
            self._retriever = retriever

        @property
        def supports_scoped_retrieval(self) -> bool:
            return True

        def retrieve(
            self,
            query_text: str,
            *,
            retriever_id: str,
            query_embedding: Sequence[float] | None = None,
            top_k: int = 5,
            metadata_filter: MetadataFilter | None = None,
            scope: Any = None,
            include_embeddings: bool = False,
        ) -> list[Any]:
            if retriever_id != "vector_similarity":
                raise ValueError(f"unexpected retriever id: {retriever_id}")
            return list(
                self._retriever.retrieve(
                    RetrieverQuery(
                        query_text=query_text,
                        query_embedding=query_embedding,
                        top_k=top_k,
                        metadata_filter=metadata_filter,
                        scope=scope,
                        include_embeddings=include_embeddings,
                    )
                )
            )

        def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> list[Any]:
            return list(self._retriever.retrieve(query)) if retriever_id == "vector_similarity" else []

    retriever = VectorSimilarityRetriever(
        vector_store=vector_manager,
        embedding_manager=_FixedEmbeddingFixture(),  # type: ignore[arg-type]
    )
    service = RetrievalService(
        retriever_manager=_NativeRetrieverManager(retriever),
        profile=RagProfile(
            retriever_id="vector_similarity",
            fast_retriever_id="vector_similarity",
            enable_rerank=False,
            route_mode="off",
            native_hybrid_enabled=False,
        ),
    )
    result = service.retrieve(
        RetrievalRequest(
            query="native retrieval marker",
            retriever_id="vector_similarity",
            route_tier_override="fast",
            final_top_k=1,
            prefetch_k=1,
            scope=scope,
        )
    )
    if not result.used or not result.chunks:
        raise GateFailure(f"native RAG retrieval returned no result: {result.reason}")
    if "Known native retrieval marker" not in result.chunks[0].text:
        raise GateFailure("native RAG returned an unexpected document")
    print("[gate] native RAG read-only splitter -> embedding fixture -> vector -> retrieval: PASS")


def _run_nexus_harness() -> None:
    from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import (
        InMemoryArtifactStore,
    )
    from intergrax.runtime.nexus.artifacts.models import Artifact

    store = InMemoryArtifactStore()
    artifact = Artifact(
        tenant_id="gate-tenant",
        artifact_id="gate-artifact",
        run_id="gate-run",
        step_id=None,
        kind="gate",
        mime_type="text/plain",
        created_at_utc=datetime.now(timezone.utc),
        data=b"core",
        size_bytes=4,
    )
    store.put(artifact)
    if store.get("gate-tenant", "gate-artifact").data != b"core":
        raise GateFailure("Nexus in-memory artifact round-trip failed")
    if len(list(store.list_for_run("gate-tenant", "gate-run"))) != 1:
        raise GateFailure("Nexus in-memory artifact listing failed")
    print("[gate] Nexus/Harness in-memory artifact smoke: PASS")


def _stage(name: str, operation: Callable[[], None]) -> None:
    print(f"[gate] stage={name}")
    try:
        operation()
    except GateFailure:
        raise
    except ModuleNotFoundError as exc:
        if exc.name and _is_forbidden_name(exc.name):
            raise GateFailure(
                f"CORE_DEPENDENCY_LEAK_FOUND: stage={name} imported {exc.name}"
            ) from exc
        raise GateFailure(
            f"stage={name} missing runtime dependency {exc.name!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise GateFailure(
            f"stage={name} failed with {type(exc).__name__}: {exc}"
        ) from exc
    print(f"[gate] stage={name}: PASS")


def main() -> int:
    print(
        "[gate] LangChain-free core installation gate "
        f"platform={sys.platform} python={sys.version.split()[0]}"
    )
    try:
        _stage("project-metadata", _check_project_dependencies)
        _stage("installed-distributions-before-smoke", _print_distribution_inventory)
        _stage("import-blockade", _install_import_blockade)
        _stage("core-imports", _run_core_imports)
        _stage("llm-registry", _run_registry)
        _stage("native-ollama", _run_native_ollama)
        _stage("native-openai", _run_native_openai)
        _stage("knowledge-document", _run_knowledge_document)
        _stage("rag-read-only-consumer", _run_native_rag_read_only)
        _stage("nexus-harness", _run_nexus_harness)
        _stage("installed-distributions-after-smoke", _print_distribution_inventory)
    except GateFailure as exc:
        print(f"[gate] VERDICT=FAIL {exc}")
        return 1
    print("[gate] CLI/core entrypoint: NOT_APPLICABLE (not required for this gate)")
    print("[gate] VERDICT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
