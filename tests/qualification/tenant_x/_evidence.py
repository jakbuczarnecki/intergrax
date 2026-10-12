# © Artur Czarnecki. All rights reserved.

"""AST/source-backed tenant semantic evidence for TENANT-X discovery."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.tenant_x._types import TenantAstEvidence, TenantDiscoveryEvidence

_TENANT_NAMES: frozenset[str] = frozenset(
    {"tenant_id", "tenant", "tenant_scope", "principal_tenant_id"}
)

_SQL_EXECUTE_NAMES: frozenset[str] = frozenset({"execute", "executemany", "run", "fetchone", "fetchall"})

_VALIDATOR_CALL_NAMES: frozenset[str] = frozenset(
    {"require_tenant_id", "validate_tenant", "_validate_tenant_id", "require_tenant_id_from_exec_ctx"}
)

_TENANT_SCOPED_CONTAINER_NAMES: frozenset[str] = frozenset(
    {
        "store",
        "cache",
        "state",
        "memory",
        "index",
        "repository",
        "repo",
        "tenant_store",
        "tenant_cache",
        "tenant_state",
        "session_store",
        "kv",
        "bucket",
        "persistence",
    }
)

_TENANT_SCOPED_TYPE_MARKERS: frozenset[str] = frozenset(
    {
        "TenantScopedStore",
        "TenantScopedCache",
        "TenantScopedState",
        "TenantScopedResource",
        "TenantStateStore",
    }
)

_GLOBAL_SQL_TABLES: frozenset[str] = frozenset(
    {
        "countries",
        "country",
        "regions",
        "region",
        "currencies",
        "currency",
        "languages",
        "language",
        "locales",
        "locale",
        "timezones",
        "timezone",
        "platform_config",
        "platform_metadata",
        "reference_data",
        "global_config",
    }
)

_TENANT_OWNED_SQL_TABLES: frozenset[str] = frozenset(
    {
        "items",
        "tenants",
        "tenant",
        "sessions",
        "documents",
        "records",
        "checkpoints",
        "artifacts",
        "memories",
        "vectors",
        "embeddings",
    }
)

_SQL_FROM_RE = re.compile(r"\bfrom\s+([`\"']?)(\w+)\1", re.IGNORECASE)

_AMBIENT_DEFAULT_RE = re.compile(
    r'tenant_id\s*:\s*str\s*=\s*["\']default["\']|tenant_id\s*=\s*["\']default["\']'
)


class TenantAstEvidenceKind(StrEnum):
    TENANT_FIELD_DECLARED = "TENANT_FIELD_DECLARED"
    TENANT_SCOPED_RESOURCE = "TENANT_SCOPED_RESOURCE"
    TENANT_IN_RESOURCE_KEY = "TENANT_IN_RESOURCE_KEY"
    TENANT_MISSING_FROM_TENANT_SCOPED_RESOURCE_KEY = "TENANT_MISSING_FROM_TENANT_SCOPED_RESOURCE_KEY"
    TENANT_IN_SQL_PREDICATE = "TENANT_IN_SQL_PREDICATE"
    TENANT_MISSING_IN_TENANT_SCOPED_SQL = "TENANT_MISSING_IN_TENANT_SCOPED_SQL"
    SQL_GLOBAL_REFERENCE_TABLE = "SQL_GLOBAL_REFERENCE_TABLE"
    SQL_UNKNOWN_TABLE_OWNERSHIP = "SQL_UNKNOWN_TABLE_OWNERSHIP"
    DEFAULT_FLOWS_TO_PROTECTED_OP = "DEFAULT_FLOWS_TO_PROTECTED_OP"
    VALIDATOR_MARKER_PRESENT = "VALIDATOR_MARKER_PRESENT"
    VALIDATOR_INVOKED_BEFORE_LOOKUP = "VALIDATOR_INVOKED_BEFORE_LOOKUP"
    VALIDATOR_AFTER_PROTECTED_OPERATION = "VALIDATOR_AFTER_PROTECTED_OPERATION"
    PROTECTED_LOOKUP_WITHOUT_VALIDATOR = "PROTECTED_LOOKUP_WITHOUT_VALIDATOR"
    GLOBAL_NON_TENANT_SYMBOL = "GLOBAL_NON_TENANT_SYMBOL"
    APPLICATION_LOCAL_MODULE = "APPLICATION_LOCAL_MODULE"
    COMPATIBILITY_MODULE = "COMPATIBILITY_MODULE"


@dataclass(frozen=True, slots=True)
class _ModuleEvidenceIndex:
    by_enclosing: dict[str, tuple[TenantAstEvidence, ...]]
    module_level: tuple[TenantAstEvidence, ...]


def _excerpt(source: str, line_no: int, radius: int = 2) -> str:
    lines = source.splitlines()
    if not lines:
        return ""
    idx = max(0, min(line_no - 1, len(lines) - 1))
    start = max(0, idx - radius)
    end = min(len(lines), idx + radius + 1)
    return "\n".join(lines[start:end])


def _name_refs_tenant(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in _TENANT_NAMES:
        return True
    if isinstance(node, ast.Attribute) and node.attr in _TENANT_NAMES:
        return True
    if isinstance(node, ast.Tuple | ast.List):
        return any(_name_refs_tenant(elt) for elt in node.elts)
    return False


def _sql_literal_has_tenant_predicate(sql: str) -> bool:
    lowered = sql.lower()
    return "tenant_id" in lowered or " tenant " in f" {lowered} "


def _sql_primary_tables(sql: str) -> tuple[str, ...]:
    return tuple(m.group(2).lower() for m in _SQL_FROM_RE.finditer(sql))


def _sql_table_ownership(table: str) -> str:
    if table in _GLOBAL_SQL_TABLES:
        return "global"
    if table in _TENANT_OWNED_SQL_TABLES or table.startswith("tenant_") or table.endswith("_by_tenant"):
        return "tenant"
    return "unknown"


def _sql_resource_ownership(sql: str) -> str:
    tables = _sql_primary_tables(sql)
    if not tables:
        return "unknown"
    kinds = {_sql_table_ownership(t) for t in tables}
    if "tenant" in kinds:
        return "tenant"
    if kinds == {"global"}:
        return "global"
    if "unknown" in kinds:
        return "unknown"
    return "unknown"


def _container_symbol(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _name_suggests_tenant_scoped_resource(name: str, *, on_instance_field: bool = False) -> bool:
    lowered = name.lower()
    if "tenant" in lowered and any(
        marker in lowered for marker in ("store", "cache", "state", "memory", "index", "bucket", "kv")
    ):
        return True
    if on_instance_field and lowered in _TENANT_SCOPED_CONTAINER_NAMES:
        return True
    return lowered in {
        "tenant_store",
        "tenant_cache",
        "tenant_state",
        "tenant_memory",
        "tenant_index",
        "session_store",
    }


def _subscript_container_tenant_scoped(
    node: ast.Subscript,
    *,
    class_has_tenant_field: bool,
    class_scoped_fields: frozenset[str],
) -> bool:
    symbol = _container_symbol(node.value)
    if symbol is None:
        return False
    if isinstance(node.value, ast.Name):
        return _name_suggests_tenant_scoped_resource(symbol)
    if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
        if node.value.value.id == "self" and class_has_tenant_field:
            if symbol in class_scoped_fields or _name_suggests_tenant_scoped_resource(
                symbol, on_instance_field=True
            ):
                return True
    return _name_suggests_tenant_scoped_resource(symbol)


def _class_tenant_scoped_field_names(class_def: ast.ClassDef) -> frozenset[str]:
    scoped: set[str] = set()
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if _name_suggests_tenant_scoped_resource(name, on_instance_field=True):
                scoped.add(name)
            if stmt.annotation is not None:
                ann_src = ast.unparse(stmt.annotation)
                if any(marker in ann_src for marker in _TENANT_SCOPED_TYPE_MARKERS):
                    scoped.add(name)
    return frozenset(scoped)


def _is_protected_execute(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Attribute) and call.func.attr in _SQL_EXECUTE_NAMES:
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            return True
    return False


def _analyze_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    class_name: str | None,
    *,
    class_has_tenant_field: bool = False,
    class_scoped_fields: frozenset[str] = frozenset(),
) -> tuple[TenantAstEvidence, ...]:
    enclosing = f"{class_name}.{func.name}" if class_name else func.name
    evidence: list[TenantAstEvidence] = []
    validator_lines: list[int] = []
    protected_lines: list[int] = []
    tenant_key_vars: set[str] = set()

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Tuple | ast.List) and _name_refs_tenant(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tenant_key_vars.add(target.id)

    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _VALIDATOR_CALL_NAMES:
                validator_lines.append(node.lineno)
            if isinstance(node.func, ast.Attribute) and node.func.attr in _VALIDATOR_CALL_NAMES:
                validator_lines.append(node.lineno)
            if _is_protected_execute(node):
                sql = node.args[0].value  # type: ignore[union-attr]
                ownership = _sql_resource_ownership(sql)
                if ownership == "tenant":
                    protected_lines.append(node.lineno)
                    evidence.append(
                        TenantAstEvidence(
                            kind=TenantAstEvidenceKind.TENANT_SCOPED_RESOURCE.value,
                            detail=f"{enclosing}: SQL targets tenant-scoped table",
                            line_no=node.lineno,
                        )
                    )
                    if _sql_literal_has_tenant_predicate(sql):
                        evidence.append(
                            TenantAstEvidence(
                                kind=TenantAstEvidenceKind.TENANT_IN_SQL_PREDICATE.value,
                                detail=f"{enclosing}: SQL predicate references tenant",
                                line_no=node.lineno,
                            )
                        )
                    else:
                        evidence.append(
                            TenantAstEvidence(
                                kind=TenantAstEvidenceKind.TENANT_MISSING_IN_TENANT_SCOPED_SQL.value,
                                detail=f"{enclosing}: tenant-scoped SQL without tenant predicate",
                                line_no=node.lineno,
                            )
                        )
                elif ownership == "global":
                    evidence.append(
                        TenantAstEvidence(
                            kind=TenantAstEvidenceKind.SQL_GLOBAL_REFERENCE_TABLE.value,
                            detail=f"{enclosing}: SQL targets global/reference table",
                            line_no=node.lineno,
                        )
                    )
                else:
                    evidence.append(
                        TenantAstEvidence(
                            kind=TenantAstEvidenceKind.SQL_UNKNOWN_TABLE_OWNERSHIP.value,
                            detail=f"{enclosing}: SQL table ownership unknown",
                            line_no=node.lineno,
                        )
                    )

        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Load):
            slice_has_tenant = _name_refs_tenant(node.slice)
            if isinstance(node.slice, ast.Name) and node.slice.id in tenant_key_vars:
                slice_has_tenant = True
            if slice_has_tenant:
                evidence.append(
                    TenantAstEvidence(
                        kind=TenantAstEvidenceKind.TENANT_IN_RESOURCE_KEY.value,
                        detail=f"{enclosing}: resource key includes tenant",
                        line_no=node.lineno,
                    )
                )
                if _subscript_container_tenant_scoped(
                    node,
                    class_has_tenant_field=class_has_tenant_field,
                    class_scoped_fields=class_scoped_fields,
                ):
                    protected_lines.append(node.lineno)
                continue
            if not _subscript_container_tenant_scoped(
                node,
                class_has_tenant_field=class_has_tenant_field,
                class_scoped_fields=class_scoped_fields,
            ):
                continue
            protected_lines.append(node.lineno)
            evidence.append(
                TenantAstEvidence(
                    kind=TenantAstEvidenceKind.TENANT_SCOPED_RESOURCE.value,
                    detail=f"{enclosing}: subscript on tenant-scoped resource",
                    line_no=node.lineno,
                )
            )
            if isinstance(node.slice, ast.Name) and not _name_refs_tenant(node.slice):
                evidence.append(
                    TenantAstEvidence(
                        kind=TenantAstEvidenceKind.TENANT_MISSING_FROM_TENANT_SCOPED_RESOURCE_KEY.value,
                        detail=f"{enclosing}: tenant-scoped resource key lacks tenant",
                        line_no=node.lineno,
                    )
                )

    if protected_lines:
        all_guarded = all(any(v < p for v in validator_lines) for p in protected_lines)
        any_validator = bool(validator_lines)
        if all_guarded and any_validator:
            evidence.append(
                TenantAstEvidence(
                    kind=TenantAstEvidenceKind.VALIDATOR_INVOKED_BEFORE_LOOKUP.value,
                    detail=f"{enclosing}: validator precedes protected operation(s)",
                    line_no=min(validator_lines),
                )
            )
        elif any_validator and any(v > min(protected_lines) for v in validator_lines):
            evidence.append(
                TenantAstEvidence(
                    kind=TenantAstEvidenceKind.VALIDATOR_AFTER_PROTECTED_OPERATION.value,
                    detail=f"{enclosing}: validator occurs after protected operation",
                    line_no=max(validator_lines),
                )
            )
        else:
            module_has_validator = "require_tenant_id" in source or "validate_tenant" in source
            if module_has_validator:
                evidence.append(
                    TenantAstEvidence(
                        kind=TenantAstEvidenceKind.PROTECTED_LOOKUP_WITHOUT_VALIDATOR.value,
                        detail=f"{enclosing}: protected access without validator before operation",
                        line_no=func.lineno,
                    )
                )

    return tuple(evidence)


def _class_field_declarations(class_def: ast.ClassDef) -> tuple[TenantAstEvidence, ...]:
    out: list[TenantAstEvidence] = []
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id in _TENANT_NAMES:
                out.append(
                    TenantAstEvidence(
                        kind=TenantAstEvidenceKind.TENANT_FIELD_DECLARED.value,
                        detail=f"{class_def.name}.{stmt.target.id} field declared",
                        line_no=stmt.lineno,
                    )
                )
    return tuple(out)


def _class_has_tenant_field(class_def: ast.ClassDef) -> bool:
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id in _TENANT_NAMES:
                return True
    return False


def _ambient_default_flows_to_protected(source: str, tree: ast.Module) -> bool:
    if not _AMBIENT_DEFAULT_RE.search(source):
        return False
    default_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in _TENANT_NAMES:
                default_names.add(node.target.id)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _TENANT_NAMES:
                    default_names.add(target.id)
    if not default_names:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_protected_execute(node):
            for arg in node.args[1:]:
                if isinstance(arg, ast.Name) and arg.id in default_names:
                    return True
                if isinstance(arg, ast.Attribute) and arg.attr in _TENANT_NAMES:
                    return True
                if isinstance(arg, ast.Tuple):
                    if any(
                        isinstance(elt, ast.Name) and elt.id in default_names
                        or isinstance(elt, ast.Attribute) and elt.attr in _TENANT_NAMES
                        for elt in arg.elts
                    ):
                        return True
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if _name_refs_tenant(node.slice):
                return True
    return False


def build_module_evidence_index(module_path: str, source: str) -> _ModuleEvidenceIndex:
    try:
        tree = ast.parse(source, filename=module_path)
    except SyntaxError:
        return _ModuleEvidenceIndex(by_enclosing={}, module_level=())

    by_enclosing: dict[str, list[TenantAstEvidence]] = {}
    module_level: list[TenantAstEvidence] = []

    if module_path.startswith("applications/"):
        module_level.append(
            TenantAstEvidence(
                kind=TenantAstEvidenceKind.APPLICATION_LOCAL_MODULE.value,
                detail="module under applications/",
                line_no=1,
            )
        )
    if "compat" in module_path or "legacy" in module_path:
        module_level.append(
            TenantAstEvidence(
                kind=TenantAstEvidenceKind.COMPATIBILITY_MODULE.value,
                detail="compatibility/legacy path segment",
                line_no=1,
            )
        )

    for marker in ("require_tenant_id", "validate_tenant", "_validate_tenant_id"):
        if marker in source:
            module_level.append(
                TenantAstEvidence(
                    kind=TenantAstEvidenceKind.VALIDATOR_MARKER_PRESENT.value,
                    detail=f"module contains {marker}",
                    line_no=1,
                )
            )
            break

    if _ambient_default_flows_to_protected(source, tree):
        module_level.append(
            TenantAstEvidence(
                kind=TenantAstEvidenceKind.DEFAULT_FLOWS_TO_PROTECTED_OP.value,
                detail="ambient tenant default participates in protected operation",
                line_no=1,
            )
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            scoped_fields = _class_tenant_scoped_field_names(node)
            has_tenant = _class_has_tenant_field(node)
            by_enclosing[node.name] = list(_class_field_declarations(node))
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    fn_key = f"{node.name}.{stmt.name}"
                    fn_ev = _analyze_function(
                        stmt,
                        source,
                        node.name,
                        class_has_tenant_field=has_tenant,
                        class_scoped_fields=scoped_fields,
                    )
                    by_enclosing[fn_key] = list(fn_ev)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            by_enclosing[node.name] = list(_analyze_function(node, source, None))

    return _ModuleEvidenceIndex(
        by_enclosing={k: tuple(v) for k, v in by_enclosing.items()},
        module_level=tuple(module_level),
    )


def discovery_evidence_for(
    module_path: str,
    source: str,
    *,
    enclosing_symbol: str,
    line_no: int,
    index: _ModuleEvidenceIndex | None = None,
) -> TenantDiscoveryEvidence:
    idx = index if index is not None else build_module_evidence_index(module_path, source)
    collected: list[TenantAstEvidence] = list(idx.module_level)

    if enclosing_symbol in idx.by_enclosing:
        collected.extend(idx.by_enclosing[enclosing_symbol])
    for key, ev_tuple in idx.by_enclosing.items():
        if key.endswith(f".{enclosing_symbol}") and key != enclosing_symbol:
            collected.extend(ev_tuple)
    method_prefix = f"{enclosing_symbol}."
    for key, ev_tuple in idx.by_enclosing.items():
        if key.startswith(method_prefix):
            collected.extend(ev_tuple)
    if enclosing_symbol not in idx.by_enclosing:
        class_prefix = enclosing_symbol.split(".")[0]
        if class_prefix in idx.by_enclosing:
            collected.extend(idx.by_enclosing[class_prefix])
            for key, ev_tuple in idx.by_enclosing.items():
                if key.startswith(f"{class_prefix}.") and key != enclosing_symbol:
                    collected.extend(ev_tuple)

    for kind in (TenantAstEvidenceKind.GLOBAL_NON_TENANT_SYMBOL.value,):
        if any(
            m in enclosing_symbol.lower()
            for m in ("health", "liveness", "readiness", "metrics_registry", "platform_version")
        ):
            collected.append(
                TenantAstEvidence(
                    kind=kind,
                    detail=f"symbol {enclosing_symbol} matches global non-tenant marker",
                    line_no=line_no,
                )
            )

    return TenantDiscoveryEvidence(
        source_excerpt=_excerpt(source, line_no),
        enclosing_symbol=enclosing_symbol,
        ast_evidence=tuple(collected),
    )


def evidence_kinds(evidence: TenantDiscoveryEvidence | None) -> frozenset[str]:
    if evidence is None:
        return frozenset()
    return frozenset(e.kind for e in evidence.ast_evidence)


def has_kind(evidence: TenantDiscoveryEvidence | None, kind: str) -> bool:
    return kind in evidence_kinds(evidence)
