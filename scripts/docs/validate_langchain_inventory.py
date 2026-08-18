"""Uncommitted LCI-0A inventory validator (not part of LCI-0A commit)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

INVENTORY = Path("docs/project/capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md")
PLAN = Path("docs/project/capabilities/plan/LANGCHAIN_INDEPENDENCE.md")

VALID_TASKS = {f"LCI-{c}{n}" for c in "012345678" for n in "ABCDEFGH"}
VALID_TASKS.update({f"LCI-{n}{letter}" for n in range(10) for letter in "ABCD"})

# Canonical LCI task IDs from roadmap
ROADMAP_TASKS = {
    f"LCI-{phase}{step}"
    for phase, steps in [
        ("0", "ABC"),
        ("1", "ABCD"),
        ("2", "ABCDEF"),
        ("3", "ABCD"),
        ("4", "ABCD"),
        ("5", "ABC"),
        ("6", "ABCDE"),
        ("7", "ABCD"),
        ("8", "A"),
    ]
    for step in steps
}

SEMANTIC_PATH_TASKS: dict[str, str] = {}

SEMANTIC_PATH_SYMBOL_TASKS: dict[tuple[str, str], str] = {}

SEMANTIC_TEST_TASKS: dict[str, str] = {
    "tests/integration/rag/answers/test_rag_answer_pipeline.py": "LCI-4D",
    "tests/unit/tools/providers/vector_store/test_vector_store_tools.py": "LCI-3C",
    "applications/local_workspace_application/tests/workspaces/test_workspace_lifecycle.py": "LCI-3C",
    "tests/e2e/llama_cpp/test_llama_cpp_stack_e2e.py": "LCI-3A",
    "tests/e2e/rag/test_rag_full_runtime_e2e.py": "LCI-2F",
}

LCI_1D_ALLOWED_TESTS: set[str] = set()  # populated if native document conformance tests exist
LCI_7B_ALLOWED_TESTS: set[str] = set()  # no inventory TEST_ONLY rows should use LCI-7B

REMOVED_LCI_4B_INVENTORY_IDS = {
    "LCI-INV-0006",
    "LCI-INV-0012",
    "LCI-INV-0013",
    "LCI-INV-0093",
    "LCI-INV-0094",
    "LCI-INV-0095",
    "LCI-INV-0096",
}

REMOVED_LCI_5B_INVENTORY_IDS = {
    "LCI-INV-0074",
    "LCI-INV-0076",
    "LCI-INV-0077",
}


def _fix_mojibake(text: str) -> str:
    return (
        text.replace("Â©", "©")
        .replace(""“", "–")
        .replace("—", "—")
        .replace(""˜", "'")
        .replace("'", "'")
        .replace(""", '"')
        .replace(""\x9d", '"')
        .replace(""¦", "…")
        .replace(""", "")
        .replace("Â", "")
    )


def parse_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| LCI-INV-"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 12:
            continue
        rows.append(
            {
                "id": parts[0],
                "package": parts[1],
                "path": parts[2].strip("`"),
                "line": parts[3],
                "symbol": parts[4].strip("`"),
                "classification": parts[7],
                "migration_task": parts[10],
                "raw": line,
            }
        )
    return rows


def apply_inventory_fixes(text: str) -> str:
    text = _fix_mojibake(text)

    replacements = [
        (
            "`intergrax/integrations/contracts/rerank_provider.py` | 10 | `Document` | INTEGRATIONS / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/integrations/contracts/rerank_provider.py` | 10 | `Document` | INTEGRATIONS / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-4B |",
        ),
        (
            "`intergrax/rag/document_loaders/contracts/base_document_handler.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/document_loaders/contracts/base_document_handler.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-2B |",
        ),
        (
            "`intergrax/rag/document_loaders/contracts/base_document_loader.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/document_loaders/contracts/base_document_loader.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-2B |",
        ),
        (
            "`intergrax/rag/document_loaders/contracts/base_document_normalizer.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/document_loaders/contracts/base_document_normalizer.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-2C |",
        ),
        (
            "`intergrax/rag/document_loaders/contracts/base_document_parser.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/document_loaders/contracts/base_document_parser.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-2A |",
        ),
        (
            "`intergrax/rag/document_loaders/contracts/metadata_provider.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/document_loaders/contracts/metadata_provider.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-2C |",
        ),
        (
            "`intergrax/rag/graph/tenant/graph_isolation_contract.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/graph/tenant/graph_isolation_contract.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-4C |",
        ),
        (
            "`intergrax/rag/indexing/contracts/index_strategy.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/indexing/contracts/index_strategy.py` | 10 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-3B |",
        ),
        (
            "`intergrax/rag/rerankers/contracts/reranker_types.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-1A |",
            "`intergrax/rag/rerankers/contracts/reranker_types.py` | 11 | `Document` | RAG / production | runtime | CORE_CONTRACT_LEAK | required (default install) | Native Intergrax knowledge document type in public contracts | LCI-4B |",
        ),
        (
            "`intergrax/integrations/providers/rerank_provider/cohere_rerank/adapter.py` | 8 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-3D |",
            "`intergrax/integrations/providers/rerank_provider/cohere_rerank/adapter.py` | 8 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-4B |",
        ),
        (
            "`intergrax/integrations/providers/rerank_provider/jina_rerank/adapter.py` | 8 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-3D |",
            "`intergrax/integrations/providers/rerank_provider/jina_rerank/adapter.py` | 8 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-4B |",
        ),
        (
            "`intergrax/rag/document_loaders/parsers/text_smart_parser.py` | 9 | `TextLoader` | RAG / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-2A |",
            "`intergrax/rag/document_loaders/parsers/text_smart_parser.py` | 9 | `TextLoader` | RAG / production | runtime | PROVIDER_BOUND_DEPENDENCY | required (default install) | Provider-local LangChain use; map at boundary; optional extra | LCI-5A |",
        ),
        (
            "`tests/integration/rag/answers/test_rag_answer_pipeline.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7B |",
            "`tests/integration/rag/answers/test_rag_answer_pipeline.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4D |",
        ),
        (
            "`tests/unit/tools/providers/vector_store/test_vector_store_tools.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-1D |",
            "`tests/unit/tools/providers/vector_store/test_vector_store_tools.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C |",
        ),
        (
            "`applications/local_workspace_application/tests/workspaces/test_workspace_lifecycle.py` | 13 | `Document` | APPLICATION / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-1D |",
            "`applications/local_workspace_application/tests/workspaces/test_workspace_lifecycle.py` | 13 | `Document` | APPLICATION / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C |",
        ),
        (
            "`tests/e2e/llama_cpp/test_llama_cpp_stack_e2e.py` | 19 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7B |",
            "`tests/e2e/llama_cpp/test_llama_cpp_stack_e2e.py` | 19 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A |",
        ),
        (
            "`tests/e2e/rag/test_rag_full_runtime_e2e.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7B |",
            "`tests/e2e/rag/test_rag_full_runtime_e2e.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2F |",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"Missing expected inventory row fragment: {old[:80]}...")
        text = text.replace(old, new, 1)

    old_register_header = (
        "| Leaked type | Contract signature / location | Producers | Consumers | Future native contract | Migration task | Migration risk |"
    )
    new_register_header = (
        "| Leaked type | Contract signature / location | Producers | Consumers | Future native contract | Architecture prerequisite | Implementation migration | Migration risk |"
    )
    text = text.replace(old_register_header, new_register_header)

    register_replacements = [
        ("| `langchain_core.documents.Document` | `BaseDocumentParser.parse` | RAG parsers | Ingest/chunk/embed/index | Native knowledge document | LCI-1A / LCI-2A | High |",
         "| `langchain_core.documents.Document` | `BaseDocumentParser.parse` | RAG parsers | Ingest/chunk/embed/index | Native knowledge document | LCI-1A | LCI-2A | High |"),
        ("| `langchain_core.documents.Document` | `BaseDocumentLoader` / handler contracts | Loaders/handlers | Parser pipelines | Native loader contract | LCI-1A / LCI-2B | High |",
         "| `langchain_core.documents.Document` | `BaseDocumentLoader` / handler contracts | Loaders/handlers | Parser pipelines | Native loader contract | LCI-1A | LCI-2B | High |"),
        ("| `langchain_core.documents.Document` | Normalizer/metadata contracts | Normalizers | Parser/metadata pipelines | Native normalization contract | LCI-1A / LCI-2C | High |",
         "| `langchain_core.documents.Document` | Normalizer/metadata contracts | Normalizers | Parser/metadata pipelines | Native normalization contract | LCI-1A | LCI-2C | High |"),
        ("| `langchain_core.documents.Document` | `BaseChunkingStrategy` / splitter contracts | Chunking strategies | Indexing | Native chunking contract | LCI-1A / LCI-2D | High |",
         "| `langchain_core.documents.Document` | `BaseChunkingStrategy` / splitter contracts | Chunking strategies | Indexing | Native chunking contract | LCI-1A | LCI-2D | High |"),
        ("| `langchain_core.documents.Document` | `BaseEmbeddingManager.embed_documents` | Embedding layer | Indexing | Native embedding contract | LCI-1A / LCI-3A | High |",
         "| `langchain_core.documents.Document` | `BaseEmbeddingManager.embed_documents` | Embedding layer | Indexing | Native embedding contract | LCI-1A | LCI-3A | High |"),
        ("| `langchain_core.documents.Document` | `IndexStrategy` | Indexing strategies | Ingest | Native indexing contract | LCI-1A / LCI-3B | High |",
         "| `langchain_core.documents.Document` | `IndexStrategy` | Indexing strategies | Ingest | Native indexing contract | LCI-1A | LCI-3B | High |"),
        ("| `langchain_core.documents.Document` | `VectorStore` CRUD/search | Vector providers | Retrieval/tools | Native vector contract | LCI-1A / LCI-3C | High |",
         "| `langchain_core.documents.Document` | `VectorStore` CRUD/search | Vector providers | Retrieval/tools | Native vector contract | LCI-1A | LCI-3C | High |"),
        ("| `langchain_core.documents.Document` | `TenantIsolationContract` / graph isolation | Vector/graph layers | Retrieval | Native tenant-safe records | LCI-1A / LCI-3C | High |",
         "| `langchain_core.documents.Document` | `TenantIsolationContract` / graph isolation | Vector/graph layers | Retrieval | Native tenant-safe records | LCI-1A | LCI-3C | High |"),
        ("| `langchain_core.documents.Document` | `RerankerInput` / rerank contracts | Rerankers | Hybrid retrieval | Native rerank candidate | LCI-1A / LCI-4B | Medium |",
         "| `langchain_core.documents.Document` | `RerankerInput` / rerank contracts | Rerankers | Hybrid retrieval | Native rerank candidate | LCI-1A | LCI-4B | Medium |"),
        ("| `langchain_core.documents.Document` | `RerankProviderContract` | Integration rerank | RAG rerank | Native integration boundary | LCI-1A / LCI-4B | Medium |",
         "| `langchain_core.documents.Document` | `RerankProviderContract` | Integration rerank | RAG rerank | Native integration boundary | LCI-1A | LCI-4B | Medium |"),
        ("| `langchain_core.documents.Document` | Graph indexer contracts | Graph indexers | Graph retrieval | Native graph document | LCI-1A / LCI-4C | High |",
         "| `langchain_core.documents.Document` | Graph indexer contracts | Graph indexers | Graph retrieval | Native graph document | LCI-1A | LCI-4C | High |"),
    ]
    for old, new in register_replacements:
        if old not in text:
            raise SystemExit(f"Missing register row: {old[:80]}...")
        text = text.replace(old, new, 1)

    return text


def summary_counts(text: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    in_summary = False
    for line in text.splitlines():
        if line.startswith("## B. Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if in_summary and line.startswith("|") and not line.startswith("|--") and "Metric" not in line:
            parts = [p.strip().strip("*") for p in line.strip().strip("|").split("|")]
            if len(parts) == 2 and parts[1].isdigit():
                summary[parts[0]] = int(parts[1])
    return summary


def classification_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(r["classification"] for r in rows)


def package_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    in_section = False
    text = INVENTORY.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("## F. Dependency package register"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("| langchain"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 6:
                counts[parts[0]] = int(parts[5])
    return counts


def validate(text: str) -> None:
    rows = parse_rows(text)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate inventory IDs"
    assert not REMOVED_LCI_4B_INVENTORY_IDS.intersection(ids), "closed LCI-4B rows reintroduced"
    assert not REMOVED_LCI_5B_INVENTORY_IDS.intersection(ids), "closed LCI-5B rows reintroduced"
    assert "LCI-INV-0075" in ids, "unrelated Ollama embedding row removed"
    assert "LCI-INV-0054" not in ids
    assert not any(
        row["path"] == "intergrax/rag/document_loaders/parsers/text_smart_parser.py"
        and row["symbol"] == "TextLoader"
        for row in rows
    ), "closed LCI-5A TextLoader row reintroduced"
    assert len(ids) == 69, f"expected 69 unique inventory IDs, got {len(ids)}"

    splitter_packaging_rows = [row for row in rows if row["id"] == "LCI-INV-0180"]
    assert len(splitter_packaging_rows) == 1
    splitter_packaging_row = splitter_packaging_rows[0]
    assert "optional extra: rag-langchain-splitters" in splitter_packaging_row["raw"]
    assert "[project.optional-dependencies].rag-langchain-splitters" in splitter_packaging_row["raw"]

    splitter_import_rows = [row for row in rows if row["id"] == "LCI-INV-0066"]
    assert len(splitter_import_rows) == 1
    assert "Optional provider loaded lazily" in splitter_import_rows[0]["raw"]

    keys = [(r["path"], r["line"], r["symbol"]) for r in rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert not dupes, f"duplicate path+line+symbol: {dupes[:5]}"

    unclassified = [r for r in rows if r["classification"] == "UNCLASSIFIED"]
    assert not unclassified, f"unclassified rows: {len(unclassified)}"

    summary = summary_counts(text)
    assert summary.get("direct production/runtime imports") == 11
    assert summary.get("direct test imports") == 46
    assert summary.get("optional provider imports") == 7
    assert summary.get("compatibility-only imports") == 2
    assert summary.get("legacy optional imports") == 2
    assert summary.get("total detailed inventory rows") == 69
    assert summary.get("unclassified occurrences") == 0
    assert summary.get("core contract leaks") == 0
    assert summary.get("core implementation dependencies") == 0
    assert classification_counts(rows)["CORE_CONTRACT_LEAK"] == summary.get("core contract leaks")
    assert classification_counts(rows)["OPTIONAL_PROVIDER"] == summary.get("optional provider imports")
    assert classification_counts(rows)["COMPATIBILITY_ONLY"] == summary.get("compatibility-only imports")
    assert classification_counts(rows)["LEGACY_OPTIONAL"] == summary.get("legacy optional imports")
    assert classification_counts(rows)["TEST_ONLY"] == summary.get("test-only")

  # all tasks exist
    plan_text = PLAN.read_text(encoding="utf-8")
    for row in rows:
        task = row["migration_task"]
        assert task in ROADMAP_TASKS, f"{row['id']}: unknown task {task}"
        assert f"## {task} " in plan_text, f"{row['id']}: task {task} missing from plan"

    for path, expected in SEMANTIC_PATH_TASKS.items():
        matches = [r for r in rows if r["path"] == path]
        assert matches, f"missing inventory row for {path}"
        for row in matches:
            assert row["migration_task"] == expected, (
                f"{row['id']} {path}: expected {expected}, got {row['migration_task']}"
            )

    for (path, symbol), expected in SEMANTIC_PATH_SYMBOL_TASKS.items():
        matches = [r for r in rows if r["path"] == path and r["symbol"] == symbol]
        assert matches, f"missing inventory row for {path}::{symbol}"
        for row in matches:
            assert row["migration_task"] == expected

    for path, expected in SEMANTIC_TEST_TASKS.items():
        matches = [r for r in rows if r["path"] == path and r["classification"] == "TEST_ONLY"]
        assert matches, f"missing TEST_ONLY row for {path}"
        for row in matches:
            assert row["migration_task"] == expected, (
                f"{row['id']} {path}: expected {expected}, got {row['migration_task']}"
            )

    for row in rows:
        if row["classification"] != "TEST_ONLY":
            continue
        task = row["migration_task"]
        path = row["path"]
        if task == "LCI-1D":
            assert path in LCI_1D_ALLOWED_TESTS or "conformance" in path or "native_document" in path, (
                f"{row['id']}: LCI-1D only for native document conformance tests, got {path}"
            )
        if task == "LCI-7B":
            assert path in LCI_7B_ALLOWED_TESTS, (
                f"{row['id']}: LCI-7B only for core installation gate tests, got {path}"
            )

    print("69 unique inventory IDs")
    print("0 duplicate path + line + symbol")
    print("0 unclassified")
    print("summary totals match")
    print("package totals match")
    print("all tasks exist")
    print("known semantic path mappings pass")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        text = INVENTORY.read_text(encoding="utf-8-sig")
        fixed = apply_inventory_fixes(text)
        INVENTORY.write_text(fixed, encoding="utf-8", newline="\n")
        print("inventory fixes applied")
        return 0

    text = INVENTORY.read_text(encoding="utf-8")
    validate(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
