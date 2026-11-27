# © Artur Czarnecki. All rights reserved.

"""EDER-F-0091 (EQOA-A3-011 / FRZ-TYP-04): vectorstore hasattr ratchet and behavioral contracts.

Locks the exact classified baseline of residual ``hasattr(...)`` calls under
``intergrax/rag/vectorstore/`` on current HEAD so that zero new unclassified reflection
probes can leak into the package, while exercising the typed behavior of the three
cited modules (``vector_store.py``, ``lexical_hybrid.py``, ``prod_slo.py``).
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.hybrid.lexical_hybrid import LexicalHybridSupport
from intergrax.rag.vectorstore.soak.prod_slo import SoakConfig, run_beta_adapter_soak
from numpy.typing import NDArray

pytestmark = pytest.mark.unit

# Exact current-HEAD baseline for EDER-F-0091 / EQOA-A3-011:
# (repo-relative posix path, enclosing qualified function name, probed attribute name) -> exact call count
_CLASSIFIED_RESIDUAL_HASATTR_SITES: Counter[tuple[str, str, str]] = Counter(
    {
        (
            "intergrax/rag/vectorstore/contracts/vector_store.py",
            "VectorStore.list_collections",
            "cfg",
        ): 1,
        (
            "intergrax/rag/vectorstore/hybrid/lexical_hybrid.py",
            "LexicalHybridSupport._lexical_hits",
            "_payloads",
        ): 1,
        (
            "intergrax/rag/vectorstore/hybrid/lexical_hybrid.py",
            "LexicalHybridSupport.query_hybrid",
            "_payloads",
        ): 1,
        (
            "intergrax/rag/vectorstore/soak/prod_slo.py",
            "run_beta_adapter_soak",
            "add_records",
        ): 1,
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _collect_hasattr_sites(package_dir: Path, repo_root: Path) -> Counter[tuple[str, str, str]]:
    sites: Counter[tuple[str, str, str]] = Counter()
    for py_path in sorted(package_dir.rglob("*.py")):
        rel = py_path.relative_to(repo_root).as_posix()
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self._scope: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._scope.append(node.name)
                self.generic_visit(node)
                self._scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._scope.append(node.name)
                self.generic_visit(node)
                self._scope.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._scope.append(node.name)
                self.generic_visit(node)
                self._scope.pop()

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
                    attr_name = "<dynamic>"
                    if (
                        len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        attr_name = node.args[1].value
                    qualname = ".".join(self._scope) if self._scope else "<module>"
                    sites[(rel, qualname, attr_name)] += 1
                self.generic_visit(node)

        _Visitor().visit(tree)
    return sites


def test_vectorstore_contracts_hybrid_and_soak_have_no_unclassified_hasattr_dispatch() -> None:
    """Ratchet gate: zero NEW unclassified hasattr() probes anywhere in intergrax/rag/vectorstore/."""
    root = _repo_root()
    package_dir = root / "intergrax" / "rag" / "vectorstore"
    observed = _collect_hasattr_sites(package_dir, root)

    unexpected = observed - _CLASSIFIED_RESIDUAL_HASATTR_SITES
    assert not unexpected, (
        "FRZ-TYP-04 / EDER-F-0091 ratchet violation: new or duplicate hasattr() sites detected: "
        f"{sorted(unexpected.items())}"
    )
    assert observed == _CLASSIFIED_RESIDUAL_HASATTR_SITES, (
        "Classified EDER-F-0091 residual hasattr() baseline drifted; update register and ratchet: "
        f"observed={sorted(observed.items())}"
    )


class _MinimalVectorStore(VectorStore):
    def __init__(
        self,
        *,
        collection_name: str | None = None,
        cfg: object | None = None,
        tenant_id: str = "tenant-a",
    ) -> None:
        if collection_name is not None:
            self.collection_name = collection_name
        if cfg is not None:
            self.cfg = cfg
        self._tenant_id = tenant_id

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        return [r.vector_id for r in records]

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return []

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        return None

    def count(self, *, scope: VectorStoreScope) -> int:
        return 0


@dataclass(frozen=True)
class _CollectionCfg:
    collection_name: str


def test_vector_store_list_collections_resolves_explicit_cfg_and_default_paths() -> None:
    """Exercise VectorStore.list_collections across explicit, cfg-backed, and fallback paths."""
    explicit = _MinimalVectorStore(collection_name="docs_primary")
    assert explicit.list_collections() == ["docs_primary"]

    cfg_backed = _MinimalVectorStore(cfg=_CollectionCfg(collection_name="docs_from_cfg"))
    assert cfg_backed.list_collections() == ["docs_from_cfg"]

    fallback = _MinimalVectorStore(tenant_id="tenant-z")
    assert fallback.list_collections() == ["inmemory:tenant-z"]


class _MinimalHybridStore(LexicalHybridSupport):
    def __init__(self, dense_hits: list[VectorStoreHit]) -> None:
        super().__init__()
        self._dense_hits = dense_hits
        self._documents = {hit.vector_id: hit.document for hit in dense_hits}

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return list(self._dense_hits[:top_k])


def test_lexical_hybrid_support_fuses_hits_without_payloads_attribute() -> None:
    """LexicalHybridSupport.query_hybrid works cleanly on subclasses without _payloads."""
    scope = VectorStoreScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": "deterministic hybrid retrieval verification text",
            "metadata": {"tag": "guard"},
            "provenance": {"source_kind": "unit", "source_id": "doc-1"},
        }
    )
    dense_hit = VectorStoreHit(vector_id="doc-1", document=doc, similarity_score=0.85, rank=0)
    store = _MinimalHybridStore([dense_hit])
    store._index_lexical("doc-1", doc.content)

    results = store.query_hybrid(
        [1.0, 0.0],
        "deterministic hybrid",
        scope=scope,
        top_k=3,
    )
    assert store.supports_native_hybrid_search() is False
    assert len(results) == 1
    assert results[0].vector_id == "doc-1"


def test_run_beta_adapter_soak_falls_back_to_harness_store_on_non_vectorstore_factory() -> None:
    """run_beta_adapter_soak falls back cleanly when factory returns an object without add_records."""

    def _non_store_factory(**_kwargs: Any) -> object:
        return object()

    result = run_beta_adapter_soak(
        _non_store_factory,
        slug="pinecone",
        config=SoakConfig(document_count=4, query_rounds=1, top_k=2, embedding_dim=4),
    )
    assert result.passed is True
    assert result.slug == "pinecone"
    assert result.documents_indexed == 4
    assert result.queries_executed == 1
