#!/usr/bin/env python3
"""Qualify one isolated optional LangChain/LangGraph compatibility install."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import re
import site
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "llm-langchain-ollama",
    "rag-langchain-loaders",
    "rag-langchain-embeddings",
    "rag-langchain-splitters",
    "langgraph-legacy",
)
COMPATIBILITY_PREFIXES = ("langchain", "langgraph")


class GateFailure(RuntimeError):
    """A fail-closed qualification failure."""


def _normal_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _dependency_name(requirement: str) -> str:
    return _normal_name(re.split(r"[<>=!~\s;]", requirement, maxsplit=1)[0])


def _is_compatibility_name(name: str) -> bool:
    normalized = _normal_name(name)
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}-")
        for prefix in COMPATIBILITY_PREFIXES
    )


def _project() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _installed_distributions() -> set[str]:
    return {
        _normal_name(dist.metadata["Name"])
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }


def _check_environment() -> None:
    if sys.version_info[:2] != (3, 12):
        raise GateFailure(
            f"BLOCKED_ENVIRONMENT: Python 3.12 required, got {sys.version.split()[0]}"
        )
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise GateFailure("BLOCKED_ENVIRONMENT: an isolated virtual environment is required")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise GateFailure("BLOCKED_ENVIRONMENT: PYTHONNOUSERSITE=1 is required")
    user_site = Path(site.getusersitepackages()).resolve()
    if user_site in {Path(entry).resolve() for entry in sys.path if entry}:
        raise GateFailure("BLOCKED_ENVIRONMENT: user site is present on sys.path")


def _check_installed_origin() -> None:
    import intergrax

    origin = Path(intergrax.__file__ or "").resolve()
    if REPO_ROOT in origin.parents or origin == REPO_ROOT:
        raise GateFailure(
            f"BLOCKED_ENVIRONMENT: intergrax imported from checkout: {origin}"
        )
    if Path(sys.prefix).resolve() not in origin.parents:
        raise GateFailure(
            f"BLOCKED_ENVIRONMENT: intergrax is outside isolated env: {origin}"
        )
    print(f"[gate] installed package origin: {origin}")


def _check_project_core() -> None:
    dependencies = [str(item) for item in _project().get("dependencies", [])]
    leaked = [
        dependency
        for dependency in dependencies
        if _is_compatibility_name(_dependency_name(dependency))
    ]
    if leaked:
        raise GateFailure(
            "COMPATIBILITY_PACKAGING_DEFECT: [project].dependencies contains "
            + ", ".join(leaked)
        )
    print("[gate] project core dependencies: LangChain/LangGraph-free")


def _declared_direct_dependencies(family: str) -> set[str]:
    optional = _project().get("optional-dependencies", {})
    if family not in optional:
        raise GateFailure(
            f"COMPATIBILITY_PACKAGING_DEFECT: missing optional extra {family!r}"
        )
    return {_dependency_name(str(item)) for item in optional[family]}


def _check_distribution_install(family: str) -> None:
    direct = _declared_direct_dependencies(family)
    installed = _installed_distributions()
    missing = sorted(direct - installed)
    if missing:
        raise GateFailure(
            "COMPATIBILITY_PACKAGING_DEFECT: "
            f"{family} did not install declared distributions: {', '.join(missing)}"
        )
    closure = sorted(name for name in installed if _is_compatibility_name(name))
    print(f"[gate] extra={family} direct distributions: {', '.join(sorted(direct))}")
    print(f"[gate] extra={family} compatibility closure: {', '.join(closure)}")
    if family != "langgraph-legacy" and "langgraph" in closure:
        raise GateFailure(
            "COMPATIBILITY_PACKAGING_DEFECT: LangGraph entered a non-LangGraph "
            f"compatibility environment ({family})"
        )
    if family == "langgraph-legacy" and "langgraph" not in installed:
        raise GateFailure(
            "COMPATIBILITY_PACKAGING_DEFECT: langgraph distribution is absent"
        )


def _make_document(document_id: str, content: str) -> Any:
    from intergrax.knowledge.contracts import KnowledgeDocument

    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": "langchain-compatibility-gate",
                "namespace": "installation-gate",
            },
            "content": content,
            "metadata": {"gate": "langchain-compatibility"},
            "provenance": {"source_kind": "gate", "source_id": document_id},
        }
    )


def _run_llm_ollama() -> None:
    from langchain_core.messages import AIMessage
    from pydantic import BaseModel

    from intergrax.llm.messages import ChatMessage
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import (
        OllamaModelCapabilityResolver,
    )
    from intergrax.llm_adapters.registry.catalog_capabilities import (
        unwrap_catalog_capability_adapter,
    )

    class Answer(BaseModel):
        answer: str

    class FakeBoundChat:
        model = "compatibility-model"

        def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"query": "compatibility"},
                        "id": "gate-call-1",
                        "type": "tool_call",
                    }
                ],
            )

    class FakeStructuredChat:
        def invoke(self, _messages: object, **_kwargs: object) -> dict[str, object]:
            return {
                "raw": AIMessage(content='{"answer":"structured compatibility"}'),
                "parsed": Answer(answer="structured compatibility"),
                "parsing_error": None,
            }

    class FakeChatOllama:
        model = "compatibility-model"

        def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
            return AIMessage(content="plain compatibility")

        def stream(self, _messages: object, **_kwargs: object) -> list[AIMessage]:
            return [AIMessage(content="stream "), AIMessage(content="compatibility")]

        def bind_tools(self, _tools: object, **_kwargs: object) -> FakeBoundChat:
            return FakeBoundChat()

        def with_structured_output(
            self,
            _schema: object,
            **_kwargs: object,
        ) -> FakeStructuredChat:
            return FakeStructuredChat()

    resolver = OllamaModelCapabilityResolver(
        show_model=lambda _model: SimpleNamespace(capabilities=["tools"])
    )
    adapter = LangChainOllamaAdapter(
        chat=FakeChatOllama(),
        model="compatibility-model",
        capability_resolver=resolver,
    )
    messages = [ChatMessage(role="user", content="compatibility gate")]

    plain = adapter.generate_messages(messages)
    if not isinstance(plain, LLMAdapterResponse) or plain.content != "plain compatibility":
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: LangChain Ollama plain ABI failed"
        )
    print("[gate] LangChainOllamaAdapter import/construction/plain: PASS")

    tools = adapter.generate_with_tools(
        messages,
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    if (
        tools.finish_reason is not LLMFinishReason.TOOL_CALLS
        or len(tools.tool_calls) != 1
        or tools.tool_calls[0].name != "lookup"
    ):
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: LangChain Ollama tools ABI failed"
        )
    print("[gate] LangChain Ollama tools compatibility: PASS")

    structured = adapter.generate_structured(messages, Answer)
    if structured.parsed.answer != "structured compatibility":
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: LangChain Ollama structured ABI failed"
        )
    print("[gate] LangChain Ollama structured compatibility: PASS")

    events = list(adapter.stream_messages(messages))
    if not events or events[-1].kind != "final" or events[-1].response is None:
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: LangChain Ollama stream ABI failed"
        )
    if events[-1].response.content != "stream compatibility":
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: LangChain Ollama stream content failed"
        )
    print("[gate] LangChain Ollama stream baseline: PASS")

    class FakeNativeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            if kwargs.get("stream") is not False:
                raise AssertionError("native compatibility smoke requires non-streaming chat")
            return {"message": {"content": "native default"}}

    native = LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        client=FakeNativeClient(),
        model="compatibility-native-model",
    )
    concrete = unwrap_catalog_capability_adapter(native)
    if not isinstance(concrete, NativeOllamaAdapter):
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: native Ollama ownership changed"
        )
    print("[gate] Native Ollama remains registry default: PASS")


def _run_loaders() -> None:
    import langchain_community.document_loaders as loaders

    from intergrax.integrations.providers.document_parser.unstructured.opens import (
        parse_unstructured_html,
    )

    class FakeDocument:
        page_content = "loader compatibility"
        metadata = {"source": "compatibility.html"}

    class FakeLoader:
        def __init__(self, source: str) -> None:
            self.source = source

        def load(self) -> list[FakeDocument]:
            return [FakeDocument()]

    with patch.object(loaders, "UnstructuredHTMLLoader", FakeLoader):
        loader = loaders.UnstructuredHTMLLoader("compatibility.html")
        fragments = parse_unstructured_html("compatibility.html")
    if loader is None or not fragments or fragments[0].text != "loader compatibility":
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: LangChain loader ABI failed"
        )
    if fragments[0].metadata["parser_backend"] != "unstructured":
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: loader metadata ABI failed"
        )
    print("[gate] langchain-community import/provider loader smoke: PASS")


def _run_embeddings() -> None:
    import langchain_ollama
    import numpy as np

    from intergrax.rag.embedding.providers.ollama_embedding_provider import (
        OllamaEmbeddingProvider,
    )

    class FakeEmbeddings:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def embed_query(self, text: str) -> list[float]:
            if text != "probe-dimension":
                raise AssertionError("unexpected dimension probe")
            return [1.0, 2.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(index), 2.0] for index, _text in enumerate(texts)]

    with patch.object(langchain_ollama, "OllamaEmbeddings", FakeEmbeddings):
        provider = OllamaEmbeddingProvider(model_name="compatibility-model")
        if provider.dimension() != 2:
            raise GateFailure(
                "RAG_COMPATIBILITY_REGRESSION_FOUND: embedding dimension ABI failed"
            )
        vectors = provider.embed(["one", "two"])
    if vectors.dtype != np.float32 or vectors.shape != (2, 2):
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: embedding vector ABI failed"
        )
    print("[gate] langchain-ollama embedding provider deterministic smoke: PASS")


def _run_splitters() -> None:
    from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
        create_default_document_splitter,
    )
    from intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy import (
        LangChainRecursiveChunkingStrategy,
    )

    strategy = LangChainRecursiveChunkingStrategy(chunk_size=10, chunk_overlap=2)
    source = _make_document("splitter-source", "abcdefghij" * 3)
    chunks = strategy.chunk([source])
    if not chunks or not all(
        chunk.metadata["chunk_strategy"] == "langchain_recursive" for chunk in chunks
    ):
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: LangChain splitter ABI failed"
        )
    print("[gate] LangChainRecursiveChunkingStrategy import/construction/chunk: PASS")

    native_splitter = create_default_document_splitter(discover_entry_points=False)
    native_chunks = native_splitter.split_documents([source], strategy_id="recursive")
    if not native_chunks or not all(
        chunk.metadata["chunk_strategy"] == "recursive" for chunk in native_chunks
    ):
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: native splitter default changed"
        )
    try:
        native_splitter.split_documents([source], strategy_id="langchain_recursive")
    except RuntimeError as exc:
        if "not registered" not in str(exc):
            raise GateFailure(
                "RAG_COMPATIBILITY_REGRESSION_FOUND: unexpected splitter registration"
            ) from exc
    else:
        raise GateFailure(
            "RAG_COMPATIBILITY_REGRESSION_FOUND: LangChain splitter became implicit default"
        )
    print("[gate] native RAG splitter remains default/explicit registration: PASS")


def _run_langgraph() -> None:
    langgraph = importlib.import_module("langgraph")
    legacy_module = importlib.import_module("intergrax.websearch.integration.langgraph_nodes")
    if not hasattr(legacy_module, "add_messages"):
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: legacy LangGraph module boundary failed"
        )
    if langgraph.__name__ != "langgraph":
        raise GateFailure(
            "COMPATIBILITY_REGRESSION_FOUND: legacy LangGraph import boundary failed"
        )
    print("[gate] langgraph distribution and legacy import boundary: PASS")


def _run_missing_extra_control() -> None:
    installed = sorted(
        name for name in _installed_distributions() if _is_compatibility_name(name)
    )
    if installed:
        raise GateFailure(
            "BASE_CONTROL_FAILED: compatibility distributions present: "
            + ", ".join(installed)
        )
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.rag.embedding.providers.ollama_embedding_provider import (
        OllamaEmbeddingProvider,
    )

    try:
        LangChainOllamaAdapter(model="missing-extra-model")
    except RuntimeError as exc:
        if "llm-langchain-ollama" not in str(exc):
            raise GateFailure(
                "BASE_CONTROL_FAILED: Ollama missing-extra error is unstable"
            ) from exc
    else:
        raise GateFailure(
            "BASE_CONTROL_FAILED: Ollama compatibility construction unexpectedly worked"
        )

    try:
        OllamaEmbeddingProvider(model_name="missing-extra-model").dimension()
    except RuntimeError as exc:
        if "rag-langchain-embeddings" not in str(exc):
            raise GateFailure(
                "BASE_CONTROL_FAILED: RAG missing-extra error is unstable"
            ) from exc
    else:
        raise GateFailure(
            "BASE_CONTROL_FAILED: RAG compatibility construction unexpectedly worked"
        )
    print("[gate] missing-extra Ollama and representative RAG controls: PASS")


def _run_family(family: str) -> None:
    _check_distribution_install(family)
    if family == "llm-langchain-ollama":
        _run_llm_ollama()
    elif family == "rag-langchain-loaders":
        _run_loaders()
    elif family == "rag-langchain-embeddings":
        _run_embeddings()
    elif family == "rag-langchain-splitters":
        _run_splitters()
    elif family == "langgraph-legacy":
        _run_langgraph()
    print(f"[gate] family={family} VERDICT=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", choices=FAMILIES)
    group.add_argument("--missing-extra", action="store_true")
    args = parser.parse_args()
    print(
        "[gate] LangChain compatibility installation gate "
        f"platform={sys.platform} python={sys.version.split()[0]}"
    )
    try:
        _check_environment()
        _check_installed_origin()
        _check_project_core()
        if args.missing_extra:
            _run_missing_extra_control()
            print("[gate] missing-extra control VERDICT=PASS")
        else:
            _run_family(args.family)
    except GateFailure as exc:
        print(f"[gate] VERDICT=FAIL {exc}")
        return 1
    except Exception as exc:
        print(f"[gate] VERDICT=FAIL COMPATIBILITY_REGRESSION_FOUND: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
