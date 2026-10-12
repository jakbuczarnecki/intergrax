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

_AMBIENT_DEFAULT_RE = re.compile(
    r'tenant_id\s*:\s*str\s*=\s*["\']default["\']|tenant_id\s*=\s*["\']default["\']'
)


class _EvidenceKind(StrEnum):
    TENANT_FIELD_DECLARED = "TENANT_FIELD_DECLARED"
    TENANT_IN_SQL_PREDICATE = "TENANT_IN_SQL_PREDICATE"
    TENANT_MISSING_IN_SQL_QUERY = "TENANT_MISSING_IN_SQL_QUERY"
    TENANT_IN_SUBSCRIPT_KEY = "TENANT_IN_SUBSCRIPT_KEY"
    TENANT_MISSING_IN_SUBSCRIPT = "TENANT_MISSING_IN_SUBSCRIPT"
    DEFAULT_FLOWS_TO_PROTECTED_OP = "DEFAULT_FLOWS_TO_PROTECTED_OP"
    VALIDATOR_MARKER_PRESENT = "VALIDATOR_MARKER_PRESENT"
    VALIDATOR_INVOKED_BEFORE_LOOKUP = "VALIDATOR_INVOKED_BEFORE_LOOKUP"
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


def _is_protected_execute(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Attribute) and call.func.attr in _SQL_EXECUTE_NAMES:
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            return True
    return False


def _analyze_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    class_name: str | None,
) -> tuple[TenantAstEvidence, ...]:
    enclosing = f"{class_name}.{func.name}" if class_name else func.name
    evidence: list[TenantAstEvidence] = []
    validator_invoked = False
    protected_calls: list[tuple[int, ast.Call, str]] = []
    tenant_key_vars: set[str] = set()

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Tuple | ast.List) and _name_refs_tenant(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tenant_key_vars.add(target.id)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _VALIDATOR_CALL_NAMES:
                validator_invoked = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in _VALIDATOR_CALL_NAMES:
                validator_invoked = True
            if _is_protected_execute(node):
                sql = node.args[0].value  # type: ignore[union-attr]
                protected_calls.append((node.lineno, node, sql))

        if isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Store | ast.Load):
                slice_has_tenant = _name_refs_tenant(node.slice)
                if isinstance(node.slice, ast.Name) and node.slice.id in tenant_key_vars:
                    slice_has_tenant = True
                if slice_has_tenant:
                    evidence.append(
                        TenantAstEvidence(
                            kind=_EvidenceKind.TENANT_IN_SUBSCRIPT_KEY.value,
                            detail=f"{enclosing}: subscript key includes tenant",
                            line_no=node.lineno,
                        )
                    )
                elif isinstance(node.slice, ast.Name) and not _name_refs_tenant(node.slice):
                    has_tenant_param = any(a.arg in _TENANT_NAMES for a in func.args.args)
                    has_class_tenant = class_name is not None
                    if has_tenant_param or has_class_tenant:
                        evidence.append(
                            TenantAstEvidence(
                                kind=_EvidenceKind.TENANT_MISSING_IN_SUBSCRIPT.value,
                                detail=f"{enclosing}: subscript without tenant in key",
                                line_no=node.lineno,
                            )
                        )

    for lineno, _call, sql in protected_calls:
        if _sql_literal_has_tenant_predicate(sql):
            evidence.append(
                TenantAstEvidence(
                    kind=_EvidenceKind.TENANT_IN_SQL_PREDICATE.value,
                    detail=f"{enclosing}: SQL predicate references tenant",
                    line_no=lineno,
                )
            )
        else:
            has_tenant_param = any(a.arg in _TENANT_NAMES for a in func.args.args)
            if has_tenant_param or class_name is not None:
                evidence.append(
                    TenantAstEvidence(
                        kind=_EvidenceKind.TENANT_MISSING_IN_SQL_QUERY.value,
                        detail=f"{enclosing}: SQL query without tenant predicate",
                        line_no=lineno,
                    )
                )

    if protected_calls:
        if validator_invoked:
            evidence.append(
                TenantAstEvidence(
                    kind=_EvidenceKind.VALIDATOR_INVOKED_BEFORE_LOOKUP.value,
                    detail=f"{enclosing}: validator invoked in function with protected access",
                    line_no=func.lineno,
                )
            )
        else:
            module_has_validator = "require_tenant_id" in source or "validate_tenant" in source
            if module_has_validator:
                evidence.append(
                    TenantAstEvidence(
                        kind=_EvidenceKind.PROTECTED_LOOKUP_WITHOUT_VALIDATOR.value,
                        detail=f"{enclosing}: protected access without validator invocation in flow",
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
                        kind=_EvidenceKind.TENANT_FIELD_DECLARED.value,
                        detail=f"{class_def.name}.{stmt.target.id} field declared",
                        line_no=stmt.lineno,
                    )
                )
    return tuple(out)


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
                kind=_EvidenceKind.APPLICATION_LOCAL_MODULE.value,
                detail="module under applications/",
                line_no=1,
            )
        )
    if "compat" in module_path or "legacy" in module_path:
        module_level.append(
            TenantAstEvidence(
                kind=_EvidenceKind.COMPATIBILITY_MODULE.value,
                detail="compatibility/legacy path segment",
                line_no=1,
            )
        )

    for marker in ("require_tenant_id", "validate_tenant", "_validate_tenant_id"):
        if marker in source:
            module_level.append(
                TenantAstEvidence(
                    kind=_EvidenceKind.VALIDATOR_MARKER_PRESENT.value,
                    detail=f"module contains {marker}",
                    line_no=1,
                )
            )
            break

    if _ambient_default_flows_to_protected(source, tree):
        module_level.append(
            TenantAstEvidence(
                kind=_EvidenceKind.DEFAULT_FLOWS_TO_PROTECTED_OP.value,
                detail="ambient tenant default participates in protected operation",
                line_no=1,
            )
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            by_enclosing[node.name] = list(_class_field_declarations(node))
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    fn_key = f"{node.name}.{stmt.name}"
                    fn_ev = _analyze_function(stmt, source, node.name)
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

    for kind in (_EvidenceKind.GLOBAL_NON_TENANT_SYMBOL.value,):
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
