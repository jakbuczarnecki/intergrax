# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.maintenance.check_langchain_boundary import (
    DEFAULT_GRANDFATHER_PATH,
    DEFAULT_INVENTORY_PATH,
    REPO_ROOT,
    GrandfatherEntry,
    ImportRecord,
    audit_repository,
    compare_sets,
    extract_imports,
    import_fingerprint,
    load_grandfather_register,
    parse_inventory_table,
    validate_grandfather_inventory,
)

pytestmark = pytest.mark.gate

CHECKER = REPO_ROOT / "scripts" / "maintenance" / "check_langchain_boundary.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    for root_name in ("intergrax", "agents", "applications"):
        (tmp_path / root_name).mkdir()
    return tmp_path


def _run_checker(
    repo_root: Path,
    *,
    grandfather: Path | None = None,
    inventory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CHECKER), "--repo-root", str(repo_root)]
    if grandfather is not None:
        command.extend(["--grandfather", str(grandfather)])
    if inventory is not None:
        command.extend(["--inventory", str(inventory)])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _minimal_inventory_row(
    inventory_id: str,
    module: str,
    path: str,
    symbol: str,
    classification: str = "CORE_CONTRACT_LEAK",
) -> str:
    return (
        f"| {inventory_id} | `{module}` | `{path}` | 1 | `{symbol}` | RAG / production | "
        f"runtime | {classification} | required | target | LCI-2A | verified |"
    )


def _write_inventory(path: Path, rows: list[str]) -> None:
    header = (
        "## C. Detailed inventory table\n\n"
        "| Inventory ID | Package/module | Path | Line | Symbol or usage | Layer/domain | "
        "Dependency exposure | Classification | Current requirement status | Target state | "
        "Migration task | Evidence/notes |\n"
        "|--------------|----------------|------|-----:|-----------------|--------------|"
        "---------------------|----------------|----------------------------|--------------|"
        "----------------|----------------|\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _write_grandfather(path: Path, entries: list[dict[str, object]]) -> None:
    payload = {"schema_version": 1, "policy": "LCI-0B", "entries": entries}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_pass_empty_repository_scope(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}


def test_pass_exact_grandfathered_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}


def test_pass_import_moved_to_different_line_number(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(
        repo / rel,
        "# header\n\n\nfrom langchain_core.documents import Document\n",
    )
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}


def test_pass_allowed_integrations_provider_zone(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/integrations/providers/document_parser/python_docx/opens.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}
    assert len(result.allowed_imports) == 1


def test_pass_allowed_llm_provider_zone(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/llm_adapters/providers/ollama_adapter.py"
    _write(repo / rel, "from langchain_ollama import ChatOllama\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}
    assert len(result.allowed_imports) == 1


def test_pass_allowed_legacy_zone(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/legacy/rag_answers/builders/context_builder.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}
    assert len(result.allowed_imports) == 1


def test_pass_tests_component_excluded(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/tests/test_loader.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert result.problems == {}
    assert result.guarded_imports == []


def test_pass_current_repository_checker() -> None:
    proc = _run_checker(REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "langchain boundary audit: OK" in proc.stdout


def test_pass_grandfather_register_matches_inventory() -> None:
    inventory_rows = parse_inventory_table(DEFAULT_INVENTORY_PATH)
    _, problems = load_grandfather_register(DEFAULT_GRANDFATHER_PATH)
    assert not problems
    payload = json.loads(DEFAULT_GRANDFATHER_PATH.read_text(encoding="utf-8"))
    entries = [
        GrandfatherEntry(
            inventory_id=entry["inventory_id"],
            path=entry["path"],
            kind=entry["kind"],
            module=entry["module"],
            names=tuple(sorted(entry["names"])),
        )
        for entry in payload["entries"]
    ]
    assert validate_grandfather_inventory(entries, inventory_rows) == []


def test_fail_new_from_langchain_core_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_new_import_langchain_community(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/parsers/text_smart_parser.py"
    _write(repo / rel, "from langchain_community.document_loaders import TextLoader\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_new_nested_function_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/vectorstore/contracts/vector_store.py"
    _write(
        repo / rel,
        "def resolve():\n    from langchain_core.messages import AIMessage\n    return AIMessage\n",
    )
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_new_langgraph_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/supervisor/supervisor_to_state_graph.py"
    _write(repo / rel, "from langgraph.graph import StateGraph\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_literal_importlib_import_module(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(
        repo / rel,
        "import importlib\n\n"
        'importlib.import_module("langchain_core.documents")\n',
    )
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_literal_dunder_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, '__import__("langchain_core.documents")\n')
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_expanded_symbol_on_grandfathered_from_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document, Other\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "NEW_FORBIDDEN_IMPORT" in result.problems


def test_fail_stale_grandfather_entry(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "# no langchain imports\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "STALE_GRANDFATHER_ENTRY" in result.problems


def test_fail_duplicate_grandfather_entry(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    entry = {
        "inventory_id": "LCI-INV-9001",
        "path": "intergrax/rag/a.py",
        "kind": "from",
        "module": "langchain_core.documents",
        "names": ["Document"],
    }
    _write_grandfather(grandfather, [entry, entry])
    _, problems = load_grandfather_register(grandfather)
    assert any("DUPLICATE_GRANDFATHER_ENTRY" in problem for problem in problems)


def test_fail_unknown_inventory_id(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9999",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "UNKNOWN_INVENTORY_ID" in result.problems


def test_fail_inventory_path_mismatch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [
            _minimal_inventory_row(
                "LCI-INV-9001",
                "langchain_core.documents",
                "intergrax/rag/other.py",
                "Document",
            )
        ],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "INVENTORY_MISMATCH" in result.problems


def test_fail_inventory_module_mismatch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.messages", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "INVENTORY_MISMATCH" in result.problems


def test_fail_wrong_schema_version(tmp_path: Path) -> None:
    grandfather = tmp_path / "grandfather.json"
    grandfather.write_text(
        json.dumps({"schema_version": 2, "policy": "LCI-0B", "entries": []}) + "\n",
        encoding="utf-8",
    )
    _, problems = load_grandfather_register(grandfather)
    assert any("MALFORMED_GRANDFATHER_REGISTER" in problem for problem in problems)


def test_fail_invalid_json(tmp_path: Path) -> None:
    grandfather = tmp_path / "grandfather.json"
    grandfather.write_text("{not-json", encoding="utf-8")
    _, problems = load_grandfather_register(grandfather)
    assert any("MALFORMED_GRANDFATHER_REGISTER" in problem for problem in problems)


def test_fail_closed_on_syntax_error(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "def broken(:\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    result = audit_repository(repo, grandfather_path=grandfather, inventory_path=inventory)
    assert "SOURCE_PARSE_ERROR" in result.problems


def test_negative_proof_new_import_detected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "from langchain_core.documents import Document\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(inventory, [])
    _write_grandfather(grandfather, [])
    proc = _run_checker(repo, grandfather=grandfather, inventory=inventory)
    assert proc.returncode != 0
    assert "NEW_FORBIDDEN_IMPORT" in proc.stdout


def test_negative_proof_stale_grandfather_detected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    rel = "intergrax/rag/document_loaders/contracts/base_document_parser.py"
    _write(repo / rel, "# removed import\n")
    inventory = tmp_path / "inventory.md"
    grandfather = tmp_path / "grandfather.json"
    _write_inventory(
        inventory,
        [_minimal_inventory_row("LCI-INV-9001", "langchain_core.documents", rel, "Document")],
    )
    _write_grandfather(
        grandfather,
        [
            {
                "inventory_id": "LCI-INV-9001",
                "path": rel,
                "kind": "from",
                "module": "langchain_core.documents",
                "names": ["Document"],
            }
        ],
    )
    proc = _run_checker(repo, grandfather=grandfather, inventory=inventory)
    assert proc.returncode != 0
    assert "STALE_GRANDFATHER_ENTRY" in proc.stdout


def test_extract_imports_detects_nested_and_dynamic_forms() -> None:
    source = """
import importlib

def nested():
    from langchain_core.messages import AIMessage

importlib.import_module("langgraph.graph")
__import__("langchain_community.document_loaders")
"""
    records = extract_imports(source)
    kinds = {(record.kind, record.module, record.names) for record in records}
    assert ("from", "langchain_core.messages", ("AIMessage",)) in kinds
    assert ("importlib", "langgraph.graph", ()) in kinds
    assert ("__import__", "langchain_community.document_loaders", ()) in kinds


def test_compare_sets_reports_new_and_stale() -> None:
    current = [
        ImportRecord("a.py", "from", "langchain_core.documents", ("Document",)),
    ]
    grandfather = [
        GrandfatherEntry("LCI-INV-0001", "b.py", "from", "langchain_core.documents", ("Document",)),
    ]
    new_violations, stale_entries = compare_sets(current, grandfather)
    assert len(new_violations) == 1
    assert len(stale_entries) == 1
    assert import_fingerprint(current[0]) not in {
        import_fingerprint(
            ImportRecord(entry.path, entry.kind, entry.module, entry.names)
        )
        for entry in stale_entries
    }
