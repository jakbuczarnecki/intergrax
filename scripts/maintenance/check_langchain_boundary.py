#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LCI-0B — LangChain/LangGraph architecture boundary guard for production imports."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRANDFATHER_PATH = Path(__file__).with_name("langchain_boundary_grandfather.json")
DEFAULT_INVENTORY_PATH = (
    REPO_ROOT
    / "docs/project/capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md"
)

SCAN_ROOTS = ("intergrax", "agents", "applications")

ALLOWED_ZONE_PREFIXES = (
    "intergrax/compat/langchain/",
    "intergrax/integrations/providers/",
    "intergrax/llm_adapters/providers/",
    "intergrax/legacy/",
)

EXCLUDED_CLASSIFICATIONS = frozenset(
    {
        "TEST_ONLY",
        "TOOLING_DEPENDENCY",
        "PACKAGING_DEPENDENCY",
        "GENERATED_LOCK_ENTRY",
    }
)

GRANDFATHER_ENTRY_KEYS = frozenset({"inventory_id", "path", "kind", "module", "names"})


@dataclass(frozen=True)
class ImportRecord:
    path: str
    kind: str
    module: str
    names: tuple[str, ...]
    nested: bool = False
    deferred: bool = False


@dataclass(frozen=True)
class InventoryRow:
    inventory_id: str
    module: str
    path: str
    symbols: tuple[str, ...]
    classification: str


@dataclass(frozen=True)
class GrandfatherEntry:
    inventory_id: str
    path: str
    kind: str
    module: str
    names: tuple[str, ...]


@dataclass
class AuditResult:
    scanned_files: int
    allowed_imports: list[ImportRecord]
    guarded_imports: list[ImportRecord]
    problems: dict[str, list[str]]


def is_langchain_root(root: str) -> bool:
    return root == "langchain" or root.startswith("langchain_") or root == "langgraph"


def is_langchain_module(module: str) -> bool:
    return is_langchain_root(module.split(".", 1)[0])


def normalize_posix_path(path: str | Path) -> str:
    return Path(path).as_posix()


def is_allowed_zone(path: str) -> bool:
    normalized = normalize_posix_path(path)
    return any(normalized.startswith(prefix) for prefix in ALLOWED_ZONE_PREFIXES)


def should_scan_file(path: Path) -> bool:
    parts = path.parts
    if path.suffix != ".py":
        return False
    if "__pycache__" in parts:
        return False
    if "tests" in parts:
        return False
    if "docker" in parts and "runtime-context" in parts:
        return False
    return True


def iter_scan_files(repo_root: Path) -> Iterator[Path]:
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if should_scan_file(path):
                yield path


def _sorted_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({name for name in names if name}))


def import_fingerprint(record: ImportRecord) -> tuple[str, str, str, tuple[str, ...]]:
    return (record.path, record.kind, record.module, record.names)


def grandfather_fingerprint(entry: GrandfatherEntry) -> tuple[str, str, str, tuple[str, ...]]:
    return (entry.path, entry.kind, entry.module, entry.names)


def inventory_fingerprint(row: InventoryRow, *, kind: str) -> tuple[str, str, str, tuple[str, ...]]:
    return (row.path, kind, row.module, row.symbols)


def parse_inventory_symbols(symbol: str) -> tuple[str, ...]:
    return _sorted_names(part.strip() for part in symbol.split(","))


def parse_inventory_table(inventory_path: Path) -> dict[str, InventoryRow]:
    rows: dict[str, InventoryRow] = {}
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| LCI-INV-"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 8:
            continue
        inventory_id = columns[0]
        module = columns[1].strip("`")
        path = normalize_posix_path(columns[2].strip("`"))
        symbol = columns[4].strip("`")
        classification = columns[7]
        rows[inventory_id] = InventoryRow(
            inventory_id=inventory_id,
            module=module,
            path=path,
            symbols=parse_inventory_symbols(symbol),
            classification=classification,
        )
    return rows


def _literal_module(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_importlib_module(call: ast.Call) -> str | None:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id != "importlib" or func.attr != "import_module":
        return None
    if not call.args:
        return None
    return _literal_module(call.args[0])


def _extract_dunder_import_module(call: ast.Call) -> str | None:
    func = call.func
    if not isinstance(func, ast.Name) or func.id != "__import__":
        return None
    if not call.args:
        return None
    return _literal_module(call.args[0])


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.records: list[ImportRecord] = []
        self._function_depth = 0
        self._type_checking_depth = 0

    def _record(self, kind: str, module: str, names: Iterable[str]) -> None:
        self.records.append(
            ImportRecord(
                path="",
                kind=kind,
                module=module,
                names=_sorted_names(names),
                nested=self._function_depth > 0,
                deferred=self._type_checking_depth > 0,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_langchain_module(alias.name):
                self._record("import", alias.name, ())

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and is_langchain_module(node.module):
            self._record("from", node.module, (alias.name for alias in node.names))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        is_type_checking_guard = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        )
        if is_type_checking_guard:
            self._type_checking_depth += 1
            for statement in node.body:
                self.visit(statement)
            self._type_checking_depth -= 1
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        module = _extract_importlib_module(node)
        if module and is_langchain_module(module):
            self._record("importlib", module, ())
        module = _extract_dunder_import_module(node)
        if module and is_langchain_module(module):
            self._record("__import__", module, ())
        self.generic_visit(node)


def extract_imports(source: str) -> list[ImportRecord]:
    tree = ast.parse(source)
    visitor = _ImportVisitor()
    visitor.visit(tree)
    return visitor.records


def scan_imports(repo_root: Path) -> tuple[int, list[ImportRecord], list[str]]:
    records: list[ImportRecord] = []
    parse_errors: list[str] = []
    scanned_files = 0
    for path in iter_scan_files(repo_root):
        scanned_files += 1
        rel_path = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8-sig")
            file_imports = extract_imports(source)
        except SyntaxError as exc:
            parse_errors.append(
                f"SOURCE_PARSE_ERROR: {rel_path}: {exc.msg} (line {exc.lineno})"
            )
            continue
        for item in file_imports:
            records.append(
                ImportRecord(
                    path=rel_path,
                    kind=item.kind,
                    module=item.module,
                    names=item.names,
                    nested=item.nested,
                    deferred=item.deferred,
                )
            )
    return scanned_files, records, parse_errors


def partition_imports(records: Sequence[ImportRecord]) -> tuple[list[ImportRecord], list[ImportRecord]]:
    allowed: list[ImportRecord] = []
    guarded: list[ImportRecord] = []
    for record in records:
        if is_allowed_zone(record.path):
            allowed.append(record)
        else:
            guarded.append(record)
    return allowed, guarded


def provider_eager_imports(records: Sequence[ImportRecord]) -> list[ImportRecord]:
    return [
        record
        for record in records
        if record.path.startswith("intergrax/integrations/providers/")
        and not record.nested
        and not record.deferred
    ]


def inventory_rows_for_guarded(
    inventory_rows: dict[str, InventoryRow],
) -> dict[str, InventoryRow]:
    return {
        inventory_id: row
        for inventory_id, row in inventory_rows.items()
        if row.classification not in EXCLUDED_CLASSIFICATIONS and not is_allowed_zone(row.path)
    }


def match_inventory_row(
    record: ImportRecord,
    inventory_rows: dict[str, InventoryRow],
) -> InventoryRow | None:
    candidates = [
        row
        for row in inventory_rows.values()
        if row.path == record.path and row.module == record.module and row.symbols == record.names
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_grandfather_register(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"MALFORMED_GRANDFATHER_REGISTER: {path}: {exc.msg}"]

    if not isinstance(payload, dict):
        return None, [f"MALFORMED_GRANDFATHER_REGISTER: {path}: root must be an object"]
    if payload.get("schema_version") != 1:
        problems.append(
            f"MALFORMED_GRANDFATHER_REGISTER: {path}: schema_version must be 1"
        )
    if payload.get("policy") != "LCI-0B":
        problems.append(f"MALFORMED_GRANDFATHER_REGISTER: {path}: policy must be LCI-0B")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        problems.append(f"MALFORMED_GRANDFATHER_REGISTER: {path}: entries must be a list")
        return payload, problems

    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(
                f"MALFORMED_GRANDFATHER_REGISTER: {path}: entries[{index}] must be an object"
            )
            continue
        missing = GRANDFATHER_ENTRY_KEYS - entry.keys()
        extra = entry.keys() - GRANDFATHER_ENTRY_KEYS
        if missing or extra:
            problems.append(
                "MALFORMED_GRANDFATHER_REGISTER: "
                f"{path}: entries[{index}] keys must be exactly {sorted(GRANDFATHER_ENTRY_KEYS)}"
            )
            continue
        names = entry.get("names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            problems.append(
                f"MALFORMED_GRANDFATHER_REGISTER: {path}: entries[{index}].names must be a string list"
            )
            continue
        fingerprint = (
            normalize_posix_path(str(entry["path"])),
            str(entry["kind"]),
            str(entry["module"]),
            tuple(sorted(names)),
        )
        if fingerprint in seen:
            problems.append(
                "DUPLICATE_GRANDFATHER_ENTRY: "
                f"{entry['path']}: {entry['kind']} {entry['module']} {sorted(names)}"
            )
        seen.add(fingerprint)
    return payload, problems


def parse_grandfather_entries(payload: dict[str, Any]) -> list[GrandfatherEntry]:
    entries: list[GrandfatherEntry] = []
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        return entries
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        names = entry.get("names")
        if not isinstance(names, list):
            continue
        entries.append(
            GrandfatherEntry(
                inventory_id=str(entry["inventory_id"]),
                path=normalize_posix_path(str(entry["path"])),
                kind=str(entry["kind"]),
                module=str(entry["module"]),
                names=tuple(sorted(str(name) for name in names)),
            )
        )
    return entries


def validate_grandfather_inventory(
    entries: Sequence[GrandfatherEntry],
    inventory_rows: dict[str, InventoryRow],
) -> list[str]:
    problems: list[str] = []
    guarded_inventory = inventory_rows_for_guarded(inventory_rows)
    for entry in entries:
        row = inventory_rows.get(entry.inventory_id)
        if row is None:
            problems.append(
                f"UNKNOWN_INVENTORY_ID: {entry.inventory_id} ({entry.path})"
            )
            continue
        if row.classification in EXCLUDED_CLASSIFICATIONS:
            problems.append(
                f"INVENTORY_MISMATCH: {entry.inventory_id}: classification {row.classification} is excluded"
            )
        if is_allowed_zone(row.path):
            problems.append(
                f"INVENTORY_MISMATCH: {entry.inventory_id}: allowed-zone path {row.path}"
            )
        if row.path != entry.path:
            problems.append(
                "INVENTORY_MISMATCH: "
                f"{entry.inventory_id}: path {entry.path!r} != inventory {row.path!r}"
            )
        if row.module != entry.module:
            problems.append(
                "INVENTORY_MISMATCH: "
                f"{entry.inventory_id}: module {entry.module!r} != inventory {row.module!r}"
            )
        if row.symbols != entry.names:
            problems.append(
                "INVENTORY_MISMATCH: "
                f"{entry.inventory_id}: names {list(entry.names)} != inventory {list(row.symbols)}"
            )
        if entry.inventory_id not in guarded_inventory:
            problems.append(
                f"INVENTORY_MISMATCH: {entry.inventory_id}: not a guarded production inventory row"
            )
    return problems


def compare_sets(
    current_guarded: Sequence[ImportRecord],
    grandfather_entries: Sequence[GrandfatherEntry],
) -> tuple[list[ImportRecord], list[GrandfatherEntry]]:
    current_by_fp = {import_fingerprint(record): record for record in current_guarded}
    grandfather_by_fp = {
        grandfather_fingerprint(entry): entry for entry in grandfather_entries
    }
    new_violations = [
        current_by_fp[fingerprint]
        for fingerprint in sorted(current_by_fp.keys() - grandfather_by_fp.keys())
    ]
    stale_entries = [
        grandfather_by_fp[fingerprint]
        for fingerprint in sorted(grandfather_by_fp.keys() - current_by_fp.keys())
    ]
    return new_violations, stale_entries


def format_import_problem(code: str, record: ImportRecord) -> str:
    names = ", ".join(record.names) if record.names else "-"
    return f"{code}: {record.path}: {record.kind} {record.module} [{names}]"


def format_stale_problem(entry: GrandfatherEntry) -> str:
    names = ", ".join(entry.names) if entry.names else "-"
    return (
        "STALE_GRANDFATHER_ENTRY: "
        f"{entry.inventory_id}: {entry.path}: {entry.kind} {entry.module} [{names}]"
    )


def audit_repository(
    repo_root: Path,
    *,
    grandfather_path: Path,
    inventory_path: Path,
) -> AuditResult:
    inventory_rows = parse_inventory_table(inventory_path)
    scanned_files, imports, parse_errors = scan_imports(repo_root)
    allowed, guarded = partition_imports(imports)

    problems: dict[str, list[str]] = {}
    if parse_errors:
        problems["SOURCE_PARSE_ERROR"] = parse_errors
    eager_provider_imports = provider_eager_imports(allowed)
    if eager_provider_imports:
        problems["EAGER_PROVIDER_IMPORT"] = [
            format_import_problem("EAGER_PROVIDER_IMPORT", record)
            for record in eager_provider_imports
        ]

    payload, register_problems = load_grandfather_register(grandfather_path)
    for problem in register_problems:
        code = problem.split(":", 1)[0]
        problems.setdefault(code, []).append(problem)

    grandfather_entries: list[GrandfatherEntry] = []
    if payload is not None and not any(
        code.startswith("MALFORMED_GRANDFATHER_REGISTER") or code == "DUPLICATE_GRANDFATHER_ENTRY"
        for code in problems
    ):
        grandfather_entries = parse_grandfather_entries(payload)
        for problem in validate_grandfather_inventory(grandfather_entries, inventory_rows):
            code = problem.split(":", 1)[0]
            problems.setdefault(code, []).append(problem)

        new_violations, stale_entries = compare_sets(guarded, grandfather_entries)
        if new_violations:
            problems["NEW_FORBIDDEN_IMPORT"] = [
                format_import_problem("NEW_FORBIDDEN_IMPORT", record) for record in new_violations
            ]
        if stale_entries:
            problems["STALE_GRANDFATHER_ENTRY"] = [
                format_stale_problem(entry) for entry in stale_entries
            ]

    return AuditResult(
        scanned_files=scanned_files,
        allowed_imports=sorted(allowed, key=lambda item: import_fingerprint(item)),
        guarded_imports=sorted(guarded, key=lambda item: import_fingerprint(item)),
        problems=problems,
    )


def build_grandfather_baseline(
    repo_root: Path,
    inventory_path: Path,
) -> tuple[list[GrandfatherEntry], list[str]]:
    inventory_rows = parse_inventory_table(inventory_path)
    guarded_inventory = inventory_rows_for_guarded(inventory_rows)
    _, imports, parse_errors = scan_imports(repo_root)
    _, guarded = partition_imports(imports)

    problems = list(parse_errors)
    entries: list[GrandfatherEntry] = []
    for record in guarded:
        row = match_inventory_row(record, guarded_inventory)
        if row is None:
            problems.append(format_import_problem("NEW_FORBIDDEN_IMPORT", record))
            continue
        entries.append(
            GrandfatherEntry(
                inventory_id=row.inventory_id,
                path=record.path,
                kind=record.kind,
                module=record.module,
                names=record.names,
            )
        )

    entries.sort(
        key=lambda entry: (
            entry.path,
            entry.kind,
            entry.module,
            entry.names,
            entry.inventory_id,
        )
    )
    return entries, problems


def write_grandfather_register(path: Path, entries: Sequence[GrandfatherEntry]) -> None:
    payload = {
        "schema_version": 1,
        "policy": "LCI-0B",
        "entries": [
            {
                "inventory_id": entry.inventory_id,
                "path": entry.path,
                "kind": entry.kind,
                "module": entry.module,
                "names": list(entry.names),
            }
            for entry in entries
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_report(result: AuditResult, *, verbose: bool) -> None:
    if verbose:
        print("allowed-zone imports:")
        for record in result.allowed_imports:
            print(f"  {record.path}: {record.kind} {record.module} {list(record.names)}")
        print("grandfathered guarded imports:")
        for record in result.guarded_imports:
            print(f"  {record.path}: {record.kind} {record.module} {list(record.names)}")

    if result.problems:
        for code in sorted(result.problems):
            print(f"[{code}]")
            for message in result.problems[code]:
                print(f"  {message}")
        return

    print("langchain boundary audit: OK")
    print(f"scanned production files: {result.scanned_files}")
    print(f"allowed-zone imports: {len(result.allowed_imports)}")
    print(f"grandfathered guarded imports: {len(result.guarded_imports)}")
    print("new forbidden imports: 0")
    print("stale grandfather entries: 0")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: auto-detected)",
    )
    parser.add_argument(
        "--grandfather",
        type=Path,
        default=DEFAULT_GRANDFATHER_PATH,
        help="Grandfather register JSON path",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="LCI dependency inventory markdown path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print allowed-zone and guarded import diagnostics (read-only)",
    )
    args = parser.parse_args(argv)

    result = audit_repository(
        args.repo_root.resolve(),
        grandfather_path=args.grandfather.resolve(),
        inventory_path=args.inventory.resolve(),
    )
    print_report(result, verbose=args.verbose)
    return 1 if result.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
