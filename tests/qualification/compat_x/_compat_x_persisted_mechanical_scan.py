# © Artur Czarnecki. All rights reserved.

"""Mechanical durable-boundary candidate discovery (production AST walk)."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_ast_signals import parse_module
from tests.qualification.compat_x._compat_x_persisted_types import (
    DurableAccessDirection,
    DurableBoundaryCandidate,
    DurableStorageSignal,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS: Final[tuple[str, ...]] = ("intergrax", "agents", "applications")

_EXCLUDE_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "tests",
        "testing_support",
        "scripts",
        "docs",
        "__pycache__",
        ".pytest_cache",
        "build",
        ".venv",
        "node_modules",
        "docker",
        "scaffold",
        ".intergrax-artifacts",
        "runtime-context",
    }
)

_EXCLUDE_PATH_FRAGMENTS: Final[tuple[str, ...]] = (
    "/tests/",
    "\\tests\\",
    "/testing_support/",
    "\\testing_support\\",
    "/scripts/",
    "\\scripts\\",
    "/docs/",
    "\\docs\\",
)

_CREATE_TABLE_PAYLOAD_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\([^)]*payload_json",
    re.IGNORECASE | re.DOTALL,
)

_CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS\s+(\w+)",
    re.IGNORECASE,
)

_SQL_TABLE_REF_RE = re.compile(
    r"(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)",
    re.IGNORECASE,
)

_DURABLE_CALL_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "execute",
        "executemany",
        "write_text",
        "write_bytes",
        "put",
        "put_if_absent",
        "replace_if_match",
        "persist",
        "append",
        "checkpoint",
        "commit",
    }
)

_PREFILTER_MARKERS: Final[tuple[str, ...]] = (
    "sqlite3",
    "payload_json",
    "DocumentRecord",
    "ConditionalDocumentStore",
    "model_dump_json",
    "model_validate_json",
    "CREATE TABLE",
    "executemany",
    ".execute(",
    "psycopg",
    "asyncpg",
    "postgresql",
    ".write_text(",
    ".write_bytes(",
    ".read_text(",
    "json.dump(",
    "json.dumps(",
    "json.loads(",
    "pickle.dumps",
    "pickle.loads",
    "put_if_absent",
    "replace_if_match",
    "persist_package_contract_authority",
    "register_script",
    "import redis",
    "from redis",
    ".hset(",
    ".hget(",
    ".hgetall(",
    'redis.call("SET"',
    'redis.call("HSET"',
)

_REDIS_WRITE_ATTRS: Final[frozenset[str]] = frozenset({"set", "hset", "delete"})
_REDIS_READ_ATTRS: Final[frozenset[str]] = frozenset({"get", "hget", "hgetall"})

_WIRE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "model_dump_json",
        "model_dump",
        "model_validate_json",
        "model_validate",
        "json.dumps",
    }
)


def discovery_root_paths() -> tuple[str, ...]:
    roots: list[str] = []
    for name in _DISCOVERY_ROOTS:
        path = _REPO_ROOT / name
        if path.is_dir():
            roots.append(name)
    return tuple(roots)


def _normalize_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _should_scan_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for fragment in _EXCLUDE_PATH_FRAGMENTS:
        if fragment in normalized:
            return False
    for part in path.parts:
        if part in _EXCLUDE_DIR_NAMES:
            return False
    return True


def iter_production_python_modules() -> tuple[str, ...]:
    modules: list[str] = []
    for root_name in discovery_root_paths():
        root = _REPO_ROOT / root_name
        for path in sorted(root.rglob("*.py")):
            if _should_scan_file(path):
                modules.append(_normalize_path(path))
    return tuple(modules)


def _enclosing_function(tree: ast.Module, lineno: int) -> str | None:
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.lineno <= lineno and (
                best is None or node.lineno >= best.lineno
            ):
                if hasattr(node, "end_lineno") and node.end_lineno and lineno <= node.end_lineno:
                    best = node
                elif best is None or node.lineno > best.lineno:
                    best = node
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.lineno <= lineno:
                        end = getattr(item, "end_lineno", None) or item.lineno + 500
                        if lineno <= end:
                            return f"{node.name}.{item.name}"
    if best is not None:
        return best.name
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _module_imports_sqlite(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3":
                    return True
        if isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
            return True
    return False


def _module_has_redis(tree: ast.Module, source: str) -> bool:
    if "register_script" in source:
        return True
    if 'redis.call("' in source or "redis.call('" in source:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "redis" or alias.name.startswith("redis."):
                    return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "redis" or node.module.startswith("redis."):
                return True
    return False


def _module_has_pickle(source: str) -> bool:
    return "pickle.dumps" in source or "pickle.loads" in source


def _module_has_file_durable_semantics(source: str) -> bool:
    lowered = source.lower()
    if "durable persist" in lowered or "durable persistence" in lowered:
        return True
    if "Durable single-host ledger" in source:
        return True
    atomic = ".replace(" in source and (".write_text(" in source or ".write_bytes(" in source)
    roundtrip = (
        (".read_text(" in source or "json.loads(" in source or "json.load(" in source)
        and (".write_text(" in source or ".write_bytes(" in source or "json.dump(" in source)
        and ("json.dumps(" in source or "json.dump(" in source)
    )
    return atomic or roundtrip


def _module_has_postgres(tree: ast.Module, source: str) -> bool:
    if "psycopg" in source or "asyncpg" in source:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "postgres" in node.module:
            return True
    return False


def module_may_contain_durable_signals(source: str) -> bool:
    return any(marker in source for marker in _PREFILTER_MARKERS)


def discover_durable_candidates_in_source(
    module_path: str,
    source: str,
) -> list[DurableBoundaryCandidate]:
    if not module_may_contain_durable_signals(source):
        return []
    try:
        tree = parse_module(module_path, source)
    except SyntaxError:
        return []
    found: list[DurableBoundaryCandidate] = []
    has_sqlite = _module_imports_sqlite(tree) or "sqlite3" in source
    has_postgres = _module_has_postgres(tree, source)
    has_document_store = "DocumentRecord" in source or "ConditionalDocumentStore" in source
    has_redis = _module_has_redis(tree, source)
    has_pickle = _module_has_pickle(source)
    has_file_persist = _module_has_file_durable_semantics(source)

    if has_sqlite:
        for match in _CREATE_TABLE_RE.finditer(source):
            table = match.group(1)
            if "payload_json" in match.group(0):
                continue
            cid = f"sqlite.table.schema::{module_path}::{table}"
            found.append(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=None,
                    storage_signal=DurableStorageSignal.SQLITE_PAYLOAD_TABLE,
                    write_or_read=DurableAccessDirection.READ_WRITE,
                    payload_symbol=table,
                    evidence=(f"CREATE TABLE {table}",),
                )
            )

    for match in _CREATE_TABLE_PAYLOAD_RE.finditer(source):
        table = match.group(1)
        cid = f"sqlite.table.payload::{module_path}::{table}"
        found.append(
            DurableBoundaryCandidate(
                candidate_id=cid,
                module_path=module_path,
                class_name=None,
                function_name=None,
                storage_signal=DurableStorageSignal.SQLITE_PAYLOAD_TABLE,
                write_or_read=DurableAccessDirection.READ_WRITE,
                payload_symbol=table,
                evidence=(f"CREATE TABLE {table} with payload_json",),
            )
        )

    wire_without_durable = (
        has_sqlite
        or has_postgres
        or has_document_store
        or has_file_persist
        or has_redis
        or has_pickle
    )
    seen_ids: set[str] = set()

    def add(candidate: DurableBoundaryCandidate) -> None:
        if candidate.candidate_id in seen_ids:
            return
        seen_ids.add(candidate.candidate_id)
        found.append(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name in (
                "persist_package_contract_authority",
                "get_package_contract_authority",
                "persist_candidate_revision",
                "save_checkpoint",
                "export_durable_continuation_state",
                "execution_continuation_state_store_from_durable_export",
                "restore_durable_continuation_backing",
            ):
                cid = f"persist.api.def::{module_path}::{node.name}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=None,
                        function_name=node.name,
                        storage_signal=DurableStorageSignal.EXPLICIT_PERSIST_API,
                        write_or_read=DurableAccessDirection.WRITE
                        if not node.name.startswith("get_")
                        else DurableAccessDirection.READ,
                        payload_symbol=None,
                        evidence=(f"def {node.name}",),
                    )
                )
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name is None:
            continue
        lineno = getattr(node, "lineno", 0)
        fn = _enclosing_function(tree, lineno)

        if call_name in (
            "persist_package_contract_authority",
            "get_package_contract_authority",
            "save_checkpoint",
            "persist_candidate_revision",
            "export_durable_continuation_state",
            "execution_continuation_state_store_from_durable_export",
            "restore_durable_continuation_backing",
        ):
            cid = f"persist.api::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.EXPLICIT_PERSIST_API,
                    write_or_read=DurableAccessDirection.WRITE
                    if "get_" not in call_name
                    else DurableAccessDirection.READ,
                    payload_symbol=None,
                    evidence=(f"{call_name}@{lineno}",),
                )
            )

        if call_name in _DURABLE_CALL_ATTRS and has_sqlite:
            cid = f"sqlite.dml::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.SQLITE_DML,
                    write_or_read=DurableAccessDirection.WRITE,
                    payload_symbol=None,
                    evidence=(f"sqlite3.{call_name}@{lineno}",),
                )
            )

        if call_name in ("put", "put_if_absent", "replace_if_match") and has_document_store:
            cid = f"document_store.write::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.DOCUMENT_STORE_WRITE,
                    write_or_read=DurableAccessDirection.WRITE,
                    payload_symbol=None,
                    evidence=(f"DocumentStore.{call_name}@{lineno}",),
                )
            )

        if call_name in ("write_text", "write_bytes", "read_text"):
            if has_file_persist:
                direction = (
                    DurableAccessDirection.READ
                    if call_name == "read_text"
                    else DurableAccessDirection.WRITE
                )
                cid = f"file.io::{module_path}::{lineno}::{call_name}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=None,
                        function_name=fn,
                        storage_signal=DurableStorageSignal.FILE_WRITE,
                        write_or_read=direction,
                        payload_symbol=None,
                        evidence=(f"file_io.{call_name}@{lineno}",),
                    )
                )

        if call_name == "replace" and isinstance(node.func, ast.Attribute):
            if has_file_persist and node.func.attr == "replace":
                cid = f"file.atomic_replace::{module_path}::{lineno}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=None,
                        function_name=fn,
                        storage_signal=DurableStorageSignal.FILE_WRITE,
                        write_or_read=DurableAccessDirection.WRITE,
                        payload_symbol=None,
                        evidence=(f"Path.replace@{lineno}",),
                    )
                )

        if call_name in _REDIS_WRITE_ATTRS and has_redis:
            signal = (
                DurableStorageSignal.REDIS_HASH_WRITE
                if call_name == "hset"
                else DurableStorageSignal.REDIS_KV_WRITE
            )
            cid = f"redis.write::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=signal,
                    write_or_read=DurableAccessDirection.WRITE,
                    payload_symbol=None,
                    evidence=(f"redis.{call_name}@{lineno}",),
                )
            )

        if call_name in _REDIS_READ_ATTRS and has_redis:
            cid = f"redis.read::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.REDIS_READ,
                    write_or_read=DurableAccessDirection.READ,
                    payload_symbol=None,
                    evidence=(f"redis.{call_name}@{lineno}",),
                )
            )

        if call_name == "register_script" and has_redis:
            cid = f"redis.script::{module_path}::{lineno}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.REDIS_SCRIPTED_STATE,
                    write_or_read=DurableAccessDirection.READ_WRITE,
                    payload_symbol=None,
                    evidence=(f"register_script@{lineno}",),
                )
            )

        if call_name in ("dumps", "loads") and has_pickle:
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("dumps", "loads"):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "pickle":
                    if wire_without_durable:
                        cid = f"pickle.wire::{module_path}::{lineno}::{call_name}"
                        add(
                            DurableBoundaryCandidate(
                                candidate_id=cid,
                                module_path=module_path,
                                class_name=None,
                                function_name=fn,
                                storage_signal=DurableStorageSignal.PICKLE_DURABLE_WIRE,
                                write_or_read=DurableAccessDirection.WRITE
                                if call_name == "dumps"
                                else DurableAccessDirection.READ,
                                payload_symbol=None,
                                evidence=(f"pickle.{call_name}@{lineno}",),
                            )
                        )

        if call_name == "dump" and isinstance(node.func, ast.Attribute):
            if node.func.attr == "dump" and has_file_persist:
                cid = f"file.json_dump::{module_path}::{lineno}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=None,
                        function_name=fn,
                        storage_signal=DurableStorageSignal.FILE_WRITE,
                        write_or_read=DurableAccessDirection.WRITE,
                        payload_symbol=None,
                        evidence=(f"json.dump@{lineno}",),
                    )
                )

        if call_name == "model_validate" and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and wire_without_durable:
                payload_sym = node.func.value.id
                cid = f"wire.model_validate::{module_path}::{lineno}::{payload_sym}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=payload_sym,
                        function_name=fn,
                        storage_signal=DurableStorageSignal.WIRE_TO_DURABLE_STORE,
                        write_or_read=DurableAccessDirection.READ,
                        payload_symbol=payload_sym,
                        evidence=(f"{payload_sym}.model_validate@{lineno}",),
                    )
                )

        if call_name == "parse_suspended_execution_operation_descriptor" and wire_without_durable:
            cid = f"wire.parse_suspended_descriptor::{module_path}::{lineno}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name="SuspendedExecutionOperationDescriptor",
                    function_name=fn,
                    storage_signal=DurableStorageSignal.WIRE_TO_DURABLE_STORE,
                    write_or_read=DurableAccessDirection.READ,
                    payload_symbol="SuspendedExecutionOperationDescriptor",
                    evidence=(f"parse_suspended_execution_operation_descriptor@{lineno}",),
                )
            )

        if call_name in _WIRE_NAMES and wire_without_durable:
            payload_sym: str | None = None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                payload_sym = node.func.value.id
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                    payload_sym = arg.func.id
                elif isinstance(arg, ast.Name):
                    payload_sym = arg.id
            cid = f"wire.durable::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.WIRE_TO_DURABLE_STORE,
                    write_or_read=DurableAccessDirection.WRITE,
                    payload_symbol=payload_sym,
                    evidence=(f"{call_name}→durable_module@{lineno}",),
                )
            )

        if call_name == "model_validate_json" and has_sqlite:
            model_sym: str | None = None
            if isinstance(node.func, ast.Attribute):
                model_sym = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            cid = f"sqlite.read.payload::{module_path}::{lineno}::{model_sym or 'model'}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=model_sym,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.SQLITE_DML,
                    write_or_read=DurableAccessDirection.READ,
                    payload_symbol=model_sym,
                    evidence=(f"model_validate_json@{lineno}",),
                )
            )

        if has_postgres and call_name in ("execute", "executemany"):
            cid = f"postgresql.dml::{module_path}::{lineno}::{call_name}"
            add(
                DurableBoundaryCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    class_name=None,
                    function_name=fn,
                    storage_signal=DurableStorageSignal.POSTGRESQL_DML,
                    write_or_read=DurableAccessDirection.WRITE,
                    payload_symbol=None,
                    evidence=(f"postgresql.{call_name}@{lineno}",),
                )
            )

    if "DocumentRecord(" in source and has_document_store:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) == "DocumentRecord":
                lineno = getattr(node, "lineno", 0)
                cid = f"document.record::{module_path}::{lineno}"
                add(
                    DurableBoundaryCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        class_name=None,
                        function_name=_enclosing_function(tree, lineno),
                        storage_signal=DurableStorageSignal.DOCUMENT_STORE_WRITE,
                        write_or_read=DurableAccessDirection.WRITE,
                        payload_symbol=None,
                        evidence=(f"DocumentRecord()@{lineno}",),
                    )
                )

    return found


@lru_cache(maxsize=1)
def discover_all_production_durable_candidates() -> tuple[DurableBoundaryCandidate, ...]:
    all_candidates: list[DurableBoundaryCandidate] = []
    seen: set[str] = set()
    for module_path in iter_production_python_modules():
        full = _REPO_ROOT / module_path
        try:
            source = full.read_text(encoding="utf-8")
        except OSError:
            continue
        for candidate in discover_durable_candidates_in_source(module_path, source):
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                all_candidates.append(candidate)
    return tuple(all_candidates)


def discover_agent_distribution_payload_tables_from_sql(source: str) -> frozenset[str]:
    tables: set[str] = set()
    for match in _CREATE_TABLE_PAYLOAD_RE.finditer(source):
        tables.add(match.group(1))
    return frozenset(tables)
