# © Artur Czarnecki. All rights reserved.

"""EBH-6 — FRZ-TYP behavioral evidence for contracts × F and application_services × F."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_6._closed_world_discovery import discover_contracts_surface_paths

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_SHARED_PREFIX: Final[str] = "intergrax.applications._shared"


@dataclass(frozen=True, slots=True)
class ApplicationServicesTypingViolation:
    source_path: str
    source_module: str
    rule_id: str
    line: int
    symbol: str
    detail: str

    def as_message(self) -> str:
        return f"{self.source_path}:{self.line}: {self.symbol} [{self.rule_id}] {self.detail}"


@dataclass(frozen=True, slots=True)
class ApplicationServicesTypingDebtEntry:
    finding_id: str
    source_path: str
    line: int
    rule_id: str
    symbol: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ApplicationServicesTypingGateResult:
    unregistered_violations: tuple[ApplicationServicesTypingViolation, ...]
    stale_debt_entries: tuple[ApplicationServicesTypingDebtEntry, ...]
    registry_validation_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.unregistered_violations and not self.stale_debt_entries and not self.registry_validation_errors


# Production-effective @ HEAD = Tier-3 `applications/**/*_wiring.py` direct `_shared`
# imports plus non-harness `intergrax/applications/_shared/*_wiring.py` seeds.
# Transitive `_shared` helpers are internal composition — excluded from class-F public seam census.
APPLICATION_SERVICES_TYPING_DEBT: Final[tuple[ApplicationServicesTypingDebtEntry, ...]] = (
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-01",
        source_path="intergrax/applications/_shared/guardrail_runtime_bridge.py",
        line=76,
        rule_id="generic_object_public_seam",
        symbol="apply_guardrail_profiles_to_runtime_config",
        rationale="RuntimeConfig attachment uses duck-typed config until wiring boundary narrowed.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-02",
        source_path="intergrax/applications/_shared/harness_host_composition.py",
        line=91,
        rule_id="generic_object_public_seam",
        symbol="resolve_harness_host_orchestration_topology_submission_port",
        rationale="Host orchestration port resolver pending typed port export.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-03",
        source_path="intergrax/applications/_shared/scenario_runtime_profiles.py",
        line=68,
        rule_id="semantic_any_public_seam",
        symbol="build_scenario_lab_runtime",
        rationale="Scenario lab runtime builder — diagnostic profile seam on wiring path.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-04",
        source_path="intergrax/applications/_shared/scenario_runtime_profiles.py",
        line=114,
        rule_id="semantic_any_public_seam",
        symbol="build_scenario_production_runtime",
        rationale="Scenario production runtime builder — typed host export deferred.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-05",
        source_path="intergrax/applications/_shared/business_agent_certification.py",
        line=72,
        rule_id="semantic_any_public_seam",
        symbol="certify_business_agents_for_deploy",
        rationale="Deploy certification factories map pending typed factory protocol.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-06",
        source_path="intergrax/applications/_shared/environment_snapshot_wiring.py",
        line=30,
        rule_id="semantic_any_public_seam",
        symbol="stable_digest_hex",
        rationale="Snapshot digest helper accepts dynamic payload until canonical snapshot typing.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-07",
        source_path="intergrax/applications/_shared/fastapi_lifespan.py",
        line=35,
        rule_id="semantic_any_public_seam",
        symbol="make_scheduler_lifespan",
        rationale="FastAPI scheduler hook typed as Any pending hosting scheduler contract.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-08",
        source_path="intergrax/applications/_shared/harness_auth.py",
        line=137,
        rule_id="generic_object_public_seam",
        symbol="resolve_harness_auth_state_from_app",
        rationale="ASGI app handle remains object until hosting app protocol export.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-09",
        source_path="intergrax/applications/_shared/migration_wiring.py",
        line=123,
        rule_id="semantic_any_public_seam",
        symbol="apply_profile_migration",
        rationale="Profile migration entry accepts dynamic profile until migration contract narrowing.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-10",
        source_path="intergrax/applications/_shared/reasoning_wiring.py",
        line=39,
        rule_id="semantic_any_public_seam",
        symbol="resolve_reasoning_task_metadata",
        rationale="Reasoning metadata resolver return pending typed metadata export.",
    ),
    ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-11",
        source_path="intergrax/applications/_shared/recovery_contract_wiring.py",
        line=123,
        rule_id="semantic_any_public_seam",
        symbol="attach_recovery_contract_to_task_metadata",
        rationale="Task metadata attachment seam pending typed metadata contract.",
    ),
)


def _module_to_path(repo_root: Path, module: str) -> Path | None:
    rel = module.replace(".", "/") + ".py"
    candidate = repo_root / rel
    if candidate.is_file():
        return candidate
    init = repo_root / module.replace(".", "/") / "__init__.py"
    return init if init.is_file() else None


def _shared_imports_in_file(path: Path) -> frozenset[str]:
    modules: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return frozenset()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(_SHARED_PREFIX):
            modules.add(node.module)
    return frozenset(modules)


def _annotation_uses_semantic_any(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Name) and node.id == "object":
        return True
    if isinstance(node, ast.Constant) and node.value is object:
        return True
    if isinstance(node, ast.Subscript):
        return _annotation_uses_semantic_any(node.slice)
    if isinstance(node, ast.Tuple):
        return any(_annotation_uses_semantic_any(elt) for elt in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_uses_semantic_any(node.left) or _annotation_uses_semantic_any(node.right)
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    return False


def _public_function_violations(
    *,
    rel_path: str,
    source_module: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ApplicationServicesTypingViolation]:
    if func.name.startswith("_"):
        return []
    violations: list[ApplicationServicesTypingViolation] = []
    if func.returns is None:
        violations.append(
            ApplicationServicesTypingViolation(
                source_path=rel_path,
                source_module=source_module,
                rule_id="missing_public_return_annotation",
                line=func.lineno,
                symbol=func.name,
                detail="public wiring function missing return annotation",
            ),
        )
    elif _annotation_uses_semantic_any(func.returns):
        rule = "semantic_any_public_seam"
        if isinstance(func.returns, ast.Name) and func.returns.id == "object":
            rule = "generic_object_public_seam"
        violations.append(
            ApplicationServicesTypingViolation(
                source_path=rel_path,
                source_module=source_module,
                rule_id=rule,
                line=func.lineno,
                symbol=func.name,
                detail="public wiring function return uses Any/object semantic seam",
            ),
        )
    for arg in func.args.args:
        if arg.arg.startswith("_"):
            continue
        if _annotation_uses_semantic_any(arg.annotation):
            rule = "semantic_any_public_seam"
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "object":
                rule = "generic_object_public_seam"
            violations.append(
                ApplicationServicesTypingViolation(
                    source_path=rel_path,
                    source_module=source_module,
                    rule_id=rule,
                    line=func.lineno,
                    symbol=func.name,
                    detail=f"public parameter {arg.arg!r} uses Any/object semantic seam",
                ),
            )
    return violations


def _violations_for_shared_module(repo_root: Path, module: str) -> tuple[ApplicationServicesTypingViolation, ...]:
    path = _module_to_path(repo_root, module)
    if path is None:
        return ()
    rel_path = path.relative_to(repo_root).as_posix()
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=rel_path)
    hits: list[ApplicationServicesTypingViolation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits.extend(
                _public_function_violations(
                    rel_path=rel_path,
                    source_module=module,
                    func=node,
                ),
            )
    return tuple(hits)


def _debt_covers(
    violation: ApplicationServicesTypingViolation,
    entry: ApplicationServicesTypingDebtEntry,
) -> bool:
    return (
        violation.source_path == entry.source_path and violation.line == entry.line and violation.symbol == entry.symbol
    )


def _validate_debt_registry(
    debt: tuple[ApplicationServicesTypingDebtEntry, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for entry in debt:
        key = (entry.source_path, entry.line, entry.symbol)
        if key in seen:
            errors.append(f"duplicate debt key: {entry.finding_id}")
        seen.add(key)
        if not entry.finding_id.startswith("EBH6-DEBT-APP-TYP-"):
            errors.append(f"{entry.finding_id}: finding_id prefix")
    return tuple(errors)


def _find_stale_debt(
    violations: tuple[ApplicationServicesTypingViolation, ...],
    debt: tuple[ApplicationServicesTypingDebtEntry, ...],
) -> tuple[ApplicationServicesTypingDebtEntry, ...]:
    stale = [entry for entry in debt if not any(_debt_covers(v, entry) for v in violations)]
    return tuple(stale)


@lru_cache(maxsize=1)
def discover_production_effective_application_shared_modules() -> frozenset[str]:
    seeds: set[str] = set()
    applications_root = _REPO_ROOT / "applications"
    if applications_root.is_dir():
        for wiring in applications_root.rglob("*_wiring.py"):
            if wiring.is_file():
                seeds.update(_shared_imports_in_file(wiring))
    shared_root = _REPO_ROOT / "intergrax" / "applications" / "_shared"
    if shared_root.is_dir():
        for wiring in shared_root.glob("*_wiring.py"):
            if wiring.name.startswith("harness_"):
                continue
            seeds.update(_shared_imports_in_file(wiring))
    assert seeds, "direct application wiring → _shared imports must be non-empty"
    return frozenset(seeds)


def evaluate_ebh6_contracts_layer_surface_purity(
    repo_root: Path,
) -> tuple[bool, str]:
    from testing_support.architecture.public_contract_boundary import (
        evaluate_contract_surface_purity,
        format_contract_surface_purity_failure,
    )

    paths = tuple(repo_root / rel for rel in sorted(discover_contracts_surface_paths()))
    result = evaluate_contract_surface_purity(repo_root, paths=paths)
    if result.passed:
        return True, ""
    return False, format_contract_surface_purity_failure(result)


def evaluate_application_services_production_seam_typing(
    repo_root: Path,
    *,
    debt_entries: tuple[ApplicationServicesTypingDebtEntry, ...] | None = None,
) -> ApplicationServicesTypingGateResult:
    registry = debt_entries if debt_entries is not None else APPLICATION_SERVICES_TYPING_DEBT
    registry_errors = _validate_debt_registry(registry)
    if registry_errors:
        return ApplicationServicesTypingGateResult(
            unregistered_violations=(),
            stale_debt_entries=(),
            registry_validation_errors=registry_errors,
        )
    violations: list[ApplicationServicesTypingViolation] = []
    for module in sorted(discover_production_effective_application_shared_modules()):
        violations.extend(_violations_for_shared_module(repo_root, module))
    all_violations = tuple(violations)
    unregistered = [
        violation for violation in all_violations if not any(_debt_covers(violation, entry) for entry in registry)
    ]
    unregistered.sort(key=lambda item: (item.source_path, item.line, item.rule_id))
    stale = _find_stale_debt(all_violations, registry)
    return ApplicationServicesTypingGateResult(
        unregistered_violations=tuple(unregistered),
        stale_debt_entries=stale,
    )


def format_application_services_typing_failure(result: ApplicationServicesTypingGateResult) -> str:
    parts: list[str] = []
    if result.registry_validation_errors:
        parts.append("registry validation:")
        parts.extend(f"  - {err}" for err in result.registry_validation_errors)
    if result.stale_debt_entries:
        parts.append("stale debt entries (no matching violation @ HEAD):")
        for entry in result.stale_debt_entries:
            parts.append(f"  - {entry.finding_id} @ {entry.source_path}:{entry.line}")
    if result.unregistered_violations:
        parts.append("unregistered FRZ-TYP public seam violations:")
        for violation in result.unregistered_violations:
            parts.append(f"  - {violation.as_message()}")
    return "\n".join(parts) if parts else "application services typing gate failed"


def evaluate_application_services_typing_on_source(
    *,
    source: str,
    source_path: str,
    source_module: str,
) -> tuple[ApplicationServicesTypingViolation, ...]:
    tree = ast.parse(source, filename=source_path)
    hits: list[ApplicationServicesTypingViolation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits.extend(
                _public_function_violations(
                    rel_path=source_path,
                    source_module=source_module,
                    func=node,
                ),
            )
    return tuple(hits)
