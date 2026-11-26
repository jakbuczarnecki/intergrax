# © Artur Czarnecki. All rights reserved.

"""Mechanical seam ledger load, import scan, and validation (PT-G01 seam ledger gate)."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LEDGER_PATH = Path("governance/architecture/seam_ledger.yaml")
_MISSING_SEAM_PRODUCT_ROOTS: tuple[str, ...] = ("agents", "intergrax", "applications")


class SeamFindingKind(StrEnum):
    MISSING_SEAM = "missing_seam"
    ALLOWED = "allowed"
    STALE_ENTRY = "stale_entry"
    DUPLICATE = "duplicate"
    OWNER_MISMATCH = "owner_mismatch"
    FORBIDDEN_DIRECTION = "forbidden_direction"


@dataclass(frozen=True, slots=True)
class SeamFinding:
    kind: SeamFindingKind
    message: str
    rule_id: str | None = None
    path: str | None = None
    line: int | None = None

    @property
    def is_error(self) -> bool:
        return self.kind is not SeamFindingKind.ALLOWED


def _module_prefix_match(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _import_module_symbol(dotted: str, symbol: str) -> Any:
    mod = importlib.import_module(dotted)
    return getattr(mod, symbol)


def load_ledger(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"ledger root must be a mapping: {path}")
    return raw


def validate_ledger_structure(ledger: Mapping[str, Any], *, repo_root: Path) -> list[SeamFinding]:
    findings: list[SeamFinding] = []
    seam_types = ledger.get("seam_types")
    if not isinstance(seam_types, dict) or not seam_types:
        findings.append(SeamFinding(SeamFindingKind.STALE_ENTRY, "seam_types missing or empty"))
        return findings

    owners_by_type: dict[str, set[str]] = {}
    for name, entry in seam_types.items():
        if not isinstance(entry, dict):
            continue
        owners: set[str] = set()
        primary = entry.get("owner")
        if isinstance(primary, str) and primary:
            owners.add(primary)
        for co in entry.get("co_owners") or []:
            if isinstance(co, str) and co:
                owners.add(co)
        owners_by_type[name] = owners

    seen_ids: dict[str, str] = {}
    for collection_key in ("import_rules", "external_registries", "symbol_anchors"):
        rows = ledger.get(collection_key)
        if not isinstance(rows, list):
            findings.append(SeamFinding(SeamFindingKind.STALE_ENTRY, f"{collection_key} must be a list"))
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                findings.append(SeamFinding(SeamFindingKind.STALE_ENTRY, f"{collection_key} row missing id"))
                continue
            if row_id in seen_ids:
                findings.append(
                    SeamFinding(
                        SeamFindingKind.DUPLICATE,
                        f"duplicate ledger id {row_id!r} in {collection_key} and {seen_ids[row_id]}",
                        rule_id=row_id,
                    ),
                )
            else:
                seen_ids[row_id] = collection_key

            seam_type = row.get("seam_type")
            row_owner = row.get("owner")
            if isinstance(seam_type, str) and isinstance(row_owner, str):
                expected = owners_by_type.get(seam_type, set())
                if expected and row_owner not in expected:
                    findings.append(
                        SeamFinding(
                            SeamFindingKind.OWNER_MISMATCH,
                            f"{row_id}: owner {row_owner!r} not in seam_types.{seam_type} owners {sorted(expected)!r}",
                            rule_id=row_id,
                        ),
                    )

    import_rules = ledger.get("import_rules", [])
    if isinstance(import_rules, list):
        for rule in import_rules:
            if not isinstance(rule, dict):
                continue
            rule_id = rule.get("id", "")
            for root_name in rule.get("scan_roots", []) or []:
                if isinstance(root_name, str) and not (repo_root / root_name).is_dir():
                    findings.append(
                        SeamFinding(
                            SeamFindingKind.STALE_ENTRY,
                            f"scan root {root_name!r} does not exist",
                            rule_id=str(rule_id),
                        ),
                    )
            for rel in rule.get("exclude_paths", []) or []:
                if isinstance(rel, str) and not (repo_root / rel).is_file():
                    findings.append(
                        SeamFinding(
                            SeamFindingKind.STALE_ENTRY,
                            f"exclude_paths entry {rel!r} is not a file",
                            rule_id=str(rule_id),
                            path=rel,
                        ),
                    )

    symbol_anchors = ledger.get("symbol_anchors", [])
    if isinstance(symbol_anchors, list):
        for anchor in symbol_anchors:
            if not isinstance(anchor, dict):
                continue
            anchor_id = str(anchor.get("id", ""))
            mod_name = anchor.get("module")
            sym_name = anchor.get("symbol")
            if not isinstance(mod_name, str) or not isinstance(sym_name, str):
                findings.append(
                    SeamFinding(
                        SeamFindingKind.STALE_ENTRY,
                        f"symbol anchor {anchor_id!r} missing module or symbol",
                        rule_id=anchor_id,
                    ),
                )
                continue
            try:
                _import_module_symbol(mod_name, sym_name)
            except (ImportError, AttributeError) as exc:
                findings.append(
                    SeamFinding(
                        SeamFindingKind.STALE_ENTRY,
                        f"symbol anchor {anchor_id!r} unresolved: {mod_name}.{sym_name} ({exc})",
                        rule_id=anchor_id,
                    ),
                )

    external = ledger.get("external_registries", [])
    if isinstance(external, list):
        for reg in external:
            if not isinstance(reg, dict):
                continue
            reg_id = str(reg.get("id", ""))
            mod_name = reg.get("module")
            sym_name = reg.get("symbol")
            if not isinstance(mod_name, str) or not isinstance(sym_name, str):
                findings.append(
                    SeamFinding(
                        SeamFindingKind.STALE_ENTRY,
                        f"external registry {reg_id!r} missing module or symbol",
                        rule_id=reg_id,
                    ),
                )
                continue
            try:
                entries = _import_module_symbol(mod_name, sym_name)
            except (ImportError, AttributeError) as exc:
                findings.append(
                    SeamFinding(
                        SeamFindingKind.STALE_ENTRY,
                        f"external registry {reg_id!r} unloadable: {exc}",
                        rule_id=reg_id,
                    ),
                )
                continue
            paths_seen: set[str] = set()
            for item in entries:
                if not isinstance(item, tuple) or len(item) < 1:
                    continue
                rel_path = str(item[0])
                if rel_path in paths_seen:
                    findings.append(
                        SeamFinding(
                            SeamFindingKind.DUPLICATE,
                            f"duplicate path {rel_path!r} in registry {reg_id!r}",
                            rule_id=reg_id,
                            path=rel_path,
                        ),
                    )
                paths_seen.add(rel_path)
                if not (repo_root / rel_path).is_file():
                    findings.append(
                        SeamFinding(
                            SeamFindingKind.STALE_ENTRY,
                            f"registry path {rel_path!r} is not a file",
                            rule_id=reg_id,
                            path=rel_path,
                        ),
                    )

    return findings


def _iter_source_modules(repo_root: Path, scan_roots: Sequence[str]) -> Iterable[tuple[str, Path]]:
    for root_name in scan_roots:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            yield rel, path


def _extract_imports(source: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            base = node.module
            for alias in node.names:
                if alias.name == "*":
                    result.append((node.lineno, base))
                else:
                    result.append((node.lineno, f"{base}.{alias.name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result.append((node.lineno, alias.name))
    return result


def _rule_denies_import(module: str, rule: Mapping[str, Any]) -> bool:
    for exact in rule.get("denied_module_exact") or []:
        if isinstance(exact, str) and module == exact:
            return True
    for prefix in rule.get("denied_module_prefixes") or []:
        if isinstance(prefix, str) and _module_prefix_match(module, prefix):
            allowed = rule.get("allowed_module_prefixes") or []
            if any(isinstance(p, str) and _module_prefix_match(module, p) for p in allowed):
                return False
            return True
    return False


def _rule_allows_import(module: str, rule: Mapping[str, Any]) -> bool:
    if _rule_denies_import(module, rule):
        return False
    allowed = rule.get("allowed_module_prefixes") or []
    if allowed:
        return any(isinstance(p, str) and _module_prefix_match(module, p) for p in allowed)
    return True


def classify_import_for_rule(
    *,
    module: str,
    source_path: str,
    rule: Mapping[str, Any],
) -> SeamFinding | None:
    """Return a finding for one import against one rule, or None when the rule does not apply."""
    rule_id = str(rule.get("id", ""))
    excludes = {str(p) for p in rule.get("exclude_paths") or []}
    if source_path in excludes:
        return None

    denied = rule.get("denied_module_prefixes") or rule.get("denied_module_exact")
    allowed = rule.get("allowed_module_prefixes") or []
    touches_denied = bool(denied) and _rule_denies_import(module, rule)
    touches_allowed = bool(allowed) and any(isinstance(p, str) and _module_prefix_match(module, p) for p in allowed)
    if not touches_denied and not touches_allowed:
        return None

    if _rule_allows_import(module, rule):
        return SeamFinding(
            SeamFindingKind.ALLOWED,
            f"import {module!r} allowed by rule {rule_id}",
            rule_id=rule_id,
            path=source_path,
        )
    return SeamFinding(
        SeamFindingKind.FORBIDDEN_DIRECTION,
        f"import {module!r} forbidden by rule {rule_id} ({rule.get('direction')})",
        rule_id=rule_id,
        path=source_path,
    )


def collect_import_rule_findings(ledger: Mapping[str, Any], *, repo_root: Path) -> list[SeamFinding]:
    findings: list[SeamFinding] = []
    rules = ledger.get("import_rules")
    if not isinstance(rules, list):
        return findings

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id", ""))
        scan_roots = [str(r) for r in rule.get("scan_roots") or [] if isinstance(r, str)]
        for rel_path, path in _iter_source_modules(repo_root, scan_roots):
            try:
                imports = _extract_imports(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                findings.append(
                    SeamFinding(
                        SeamFindingKind.FORBIDDEN_DIRECTION,
                        f"syntax error in {rel_path}: {exc}",
                        rule_id=rule_id,
                        path=rel_path,
                    ),
                )
                continue
            for line, module in imports:
                finding = classify_import_for_rule(module=module, source_path=rel_path, rule=rule)
                if finding is None:
                    continue
                findings.append(replace(finding, line=line))

    return findings


def collect_missing_seam_findings(ledger: Mapping[str, Any], *, repo_root: Path) -> list[SeamFinding]:
    rules = ledger.get("import_rules")
    if not isinstance(rules, list):
        return []
    if not any(isinstance(rule, dict) and rule.get("report_missing_seam_outside_scan_roots") for rule in rules):
        return []
    findings: list[SeamFinding] = []
    for rel_path, path in _iter_source_modules(repo_root, _MISSING_SEAM_PRODUCT_ROOTS):
        try:
            imports = _extract_imports(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for line, module in imports:
            missing = detect_unregistered_cross_boundary_import(
                module=module,
                source_path=rel_path,
                ledger=ledger,
            )
            if missing is not None:
                findings.append(replace(missing, line=line))
    return findings


def _path_in_scan_roots(rel_path: str, scan_roots: Sequence[str]) -> bool:
    return any(rel_path == root or rel_path.startswith(root + "/") for root in scan_roots)


def _path_under_prefixes(rel_path: str, prefixes: Sequence[str]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(f"{prefix}/") for prefix in prefixes)


def _missing_seam_source_in_scope(rule: Mapping[str, Any], source_path: str) -> bool:
    scan_roots = [str(r) for r in rule.get("scan_roots") or [] if isinstance(r, str)]
    if _path_in_scan_roots(source_path, scan_roots):
        return False
    jurisdiction = rule.get("missing_seam_jurisdiction_roots")
    if jurisdiction is None:
        return True
    if not isinstance(jurisdiction, list) or not jurisdiction:
        return False
    jurisdiction_roots = [str(r) for r in jurisdiction if isinstance(r, str)]
    if not _path_under_prefixes(source_path, jurisdiction_roots):
        return False
    excludes = rule.get("missing_seam_jurisdiction_exclude_prefixes") or []
    exclude_prefixes = [str(e) for e in excludes if isinstance(e, str)]
    return not _path_under_prefixes(source_path, exclude_prefixes)


def detect_unregistered_cross_boundary_import(
    *,
    module: str,
    source_path: str,
    ledger: Mapping[str, Any],
) -> SeamFinding | None:
    """Return MISSING_SEAM when a denied namespace is tracked but no rule governs this source file."""
    rules = ledger.get("import_rules")
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if not rule.get("report_missing_seam_outside_scan_roots"):
            continue
        denied = rule.get("denied_module_prefixes") or []
        if not any(isinstance(p, str) and _module_prefix_match(module, p) for p in denied):
            continue
        if not _missing_seam_source_in_scope(rule, source_path):
            continue
        governed = False
        for other in rules:
            if not isinstance(other, dict):
                continue
            other_roots = [str(r) for r in other.get("scan_roots") or [] if isinstance(r, str)]
            if not _path_in_scan_roots(source_path, other_roots):
                continue
            if classify_import_for_rule(module=module, source_path=source_path, rule=other) is not None:
                governed = True
                break
        if not governed:
            return SeamFinding(
                SeamFindingKind.MISSING_SEAM,
                f"import {module!r} in {source_path!r} matches denied namespace of rule {rule.get('id')!r} "
                "but no import_rule governs this source file",
                rule_id=str(rule.get("id", "")),
                path=source_path,
            )
    return None


def collect_external_registry_findings(ledger: Mapping[str, Any]) -> list[SeamFinding]:
    findings: list[SeamFinding] = []
    external = ledger.get("external_registries")
    if not isinstance(external, list):
        return findings
    for reg in external:
        if not isinstance(reg, dict):
            continue
        reg_id = str(reg.get("id", ""))
        scan_mod = reg.get("provider_import_scan_module")
        scan_callable = reg.get("provider_import_scan_callable")
        mod_name = reg.get("module")
        sym_name = reg.get("symbol")
        if not all(isinstance(x, str) for x in (scan_mod, scan_callable, mod_name, sym_name)):
            continue
        try:
            closed = _import_module_symbol(mod_name, sym_name)
            scan_fn = _import_module_symbol(scan_mod, scan_callable)
        except (ImportError, AttributeError) as exc:
            findings.append(
                SeamFinding(
                    SeamFindingKind.STALE_ENTRY,
                    f"registry {reg_id!r}: cannot load symbols ({exc})",
                    rule_id=reg_id,
                ),
            )
            continue
        if not callable(scan_fn):
            findings.append(
                SeamFinding(
                    SeamFindingKind.STALE_ENTRY,
                    f"registry {reg_id!r}: scan callable is not callable",
                    rule_id=reg_id,
                ),
            )
            continue
        allowed = {str(item[0]) for item in closed if isinstance(item, tuple) and item}
        supplement = reg.get("supplemental_closed_paths") or []
        allowed |= {str(path) for path in supplement if isinstance(path, str)}
        observed: set[str] = set(scan_fn())
        outside = sorted(observed - allowed)
        if outside:
            findings.append(
                SeamFinding(
                    SeamFindingKind.FORBIDDEN_DIRECTION,
                    "provider package imports outside the closed composition-root list: " + ", ".join(outside),
                    rule_id=reg_id,
                ),
            )
        for rel in sorted(allowed):
            findings.append(
                SeamFinding(
                    SeamFindingKind.ALLOWED,
                    f"closed provider composition root {rel!r}",
                    rule_id=reg_id,
                    path=rel,
                ),
            )
    return findings


def run_seam_ledger_check(
    *,
    repo_root: Path,
    ledger_path: Path | None = None,
) -> list[SeamFinding]:
    ledger_file = ledger_path or (repo_root / DEFAULT_LEDGER_PATH)
    ledger = load_ledger(ledger_file)
    findings: list[SeamFinding] = []
    findings.extend(validate_ledger_structure(ledger, repo_root=repo_root))
    findings.extend(collect_import_rule_findings(ledger, repo_root=repo_root))
    findings.extend(collect_missing_seam_findings(ledger, repo_root=repo_root))
    findings.extend(collect_external_registry_findings(ledger))
    return findings


def error_findings(findings: Sequence[SeamFinding]) -> list[SeamFinding]:
    return [f for f in findings if f.is_error]
