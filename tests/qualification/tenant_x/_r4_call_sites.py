# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 bounded production call-site inventory (AST closed-world discovery)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present, read_production
from tests.qualification.tenant_x._r4_call_site_scanner import (
    R4DiscoveredProductionCallSite,
    call_expression_at_line,
    discover_r4_calls_for_callee,
)
from tests.qualification.tenant_x._r4_types import (
    R4ContextAssemblyPathClass,
    R4LlmCallTenantParamClass,
)

_CONTEXT_RECORDING: Final[str] = "intergrax/runtime/events/context_skill_recording.py"
_LLM_RECORDING: Final[str] = "intergrax/runtime/events/llm_call_recording.py"
_USAGE_LOG: Final[str] = "intergrax/llm_adapters/base/usage_log.py"

_REQUIRED_PRODUCTION_PATHS: Final[tuple[str, ...]] = (
    "intergrax/runtime/nexus/context/context_manager.py",
    _CONTEXT_RECORDING,
    _LLM_RECORDING,
    _USAGE_LOG,
)


@dataclass(frozen=True, slots=True)
class R4ContextAssemblyCallSite:
    module: str
    containing_symbol: str
    path_class: R4ContextAssemblyPathClass
    tenant_source_expression: str
    validation_before_helper: str


@dataclass(frozen=True, slots=True)
class R4LlmCallCallSite:
    module: str
    containing_symbol: str
    symbol: str
    tenant_param_class: R4LlmCallTenantParamClass
    notes: str


@dataclass(frozen=True, slots=True)
class R4ContextAssemblyCallerInventory:
    production_callers_total: int
    canonically_bound: int
    caller_controlled_unsafe: int
    unknown: int
    call_sites: tuple[R4ContextAssemblyCallSite, ...]
    helper_path_class: R4ContextAssemblyPathClass


@dataclass(frozen=True, slots=True)
class R4LlmCallCallerInventory:
    production_callers_total: int
    canonical_tenant: int
    validated_explicit_tenant: int
    untrusted_explicit_tenant: int
    unknown: int
    ambient_fallback_only: int
    call_sites: tuple[R4LlmCallCallSite, ...]


@dataclass(frozen=True, slots=True)
class R4MaybeRecordLlmCallInventory:
    production_callers_total: int
    canonical_tenant: int
    validated_explicit_tenant: int
    untrusted_explicit_tenant: int
    unknown: int
    call_sites: tuple[R4LlmCallCallSite, ...]


def _is_task_tenant_id(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "task"
        and expr.attr == "tenant_id"
    )


def _is_evidence_tenant_id(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "evidence"
        and expr.attr == "tenant_id"
    )


def classify_record_context_assembly_call(call: ast.Call) -> R4ContextAssemblyPathClass:
    metadata_kw: ast.expr | None = None
    for kw in call.keywords:
        if kw.arg == "metadata":
            metadata_kw = kw.value
            break
    if metadata_kw is None:
        return R4ContextAssemblyPathClass.UNKNOWN_CALLSITE
    if not isinstance(metadata_kw, ast.Dict):
        return R4ContextAssemblyPathClass.UNKNOWN_CALLSITE
    has_bundle_spread = False
    tenant_expr: ast.expr | None = None
    for key, value in zip(metadata_kw.keys, metadata_kw.values, strict=False):
        if key is None:
            unparsed = ast.unparse(value)
            if "bundle_metadata" in unparsed:
                has_bundle_spread = True
            continue
        if isinstance(key, ast.Constant) and key.value == "tenant_id":
            tenant_expr = value
    if has_bundle_spread and tenant_expr is not None and _is_task_tenant_id(tenant_expr):
        return R4ContextAssemblyPathClass.PRODUCTION_CALLSITE_CANONICALLY_BOUND
    if tenant_expr is not None and not _is_task_tenant_id(tenant_expr):
        return R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
    return R4ContextAssemblyPathClass.UNKNOWN_CALLSITE


def classify_record_llm_call_tenant_param(call: ast.Call) -> R4LlmCallTenantParamClass:
    tenant_expr: ast.expr | None = None
    for kw in call.keywords:
        if kw.arg == "tenant_id":
            tenant_expr = kw.value
            break
    if tenant_expr is None:
        return R4LlmCallTenantParamClass.AMBIENT_RUNTIME_TENANT
    if _is_evidence_tenant_id(tenant_expr):
        return R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT
    unparsed = ast.unparse(tenant_expr)
    if "peek_active_runtime_event_tenant_id" in unparsed:
        return R4LlmCallTenantParamClass.AMBIENT_RUNTIME_TENANT
    if "validate" in unparsed and "tenant" in unparsed:
        return R4LlmCallTenantParamClass.VALIDATED_BOUND_TENANT
    if "metadata" in unparsed or "caller" in unparsed:
        return R4LlmCallTenantParamClass.UNTRUSTED_CALLER_TENANT
    return R4LlmCallTenantParamClass.UNKNOWN


def classify_maybe_record_llm_call_caller(
    site: R4DiscoveredProductionCallSite,
    *,
    wrapper_structurally_bounded: bool,
) -> R4LlmCallTenantParamClass:
    if site.resolution != "resolved":
        return R4LlmCallTenantParamClass.UNKNOWN
    if not wrapper_structurally_bounded:
        return R4LlmCallTenantParamClass.UNKNOWN
    return R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT


def _classify_discovered_context_site(site: R4DiscoveredProductionCallSite) -> R4ContextAssemblyCallSite:
    if site.resolution != "resolved":
        return R4ContextAssemblyCallSite(
            module=site.module,
            containing_symbol=site.containing_symbol,
            path_class=R4ContextAssemblyPathClass.UNKNOWN_CALLSITE,
            tenant_source_expression=f"resolution={site.resolution}",
            validation_before_helper="AST resolution incomplete",
        )
    source = read_production(site.module)
    call = call_expression_at_line(source, site.line_no, site.module)
    if call is None:
        return R4ContextAssemblyCallSite(
            module=site.module,
            containing_symbol=site.containing_symbol,
            path_class=R4ContextAssemblyPathClass.UNKNOWN_CALLSITE,
            tenant_source_expression="call AST not found at line",
            validation_before_helper="mechanical re-parse failed",
        )
    path_class = classify_record_context_assembly_call(call)
    tenant_expr = "unknown"
    for kw in call.keywords:
        if kw.arg == "metadata":
            tenant_expr = ast.unparse(kw.value)
            break
    validation = (
        "Task.tenant_id on execution path; metadata spread then canonical tenant override"
        if path_class == R4ContextAssemblyPathClass.PRODUCTION_CALLSITE_CANONICALLY_BOUND
        else "no structural canonical tenant binding proof"
    )
    return R4ContextAssemblyCallSite(
        module=site.module,
        containing_symbol=site.containing_symbol,
        path_class=path_class,
        tenant_source_expression=tenant_expr,
        validation_before_helper=validation,
    )


def _classify_discovered_llm_direct_site(site: R4DiscoveredProductionCallSite) -> R4LlmCallCallSite:
    if site.resolution != "resolved":
        return R4LlmCallCallSite(
            module=site.module,
            containing_symbol=site.containing_symbol,
            symbol="record_llm_call_runtime_event",
            tenant_param_class=R4LlmCallTenantParamClass.UNKNOWN,
            notes=f"resolution={site.resolution}",
        )
    source = read_production(site.module)
    call = call_expression_at_line(source, site.line_no, site.module)
    if call is None:
        return R4LlmCallCallSite(
            module=site.module,
            containing_symbol=site.containing_symbol,
            symbol="record_llm_call_runtime_event",
            tenant_param_class=R4LlmCallTenantParamClass.UNKNOWN,
            notes="call AST not found at line",
        )
    tenant_class = classify_record_llm_call_tenant_param(call)
    notes = ast.unparse(call)
    if tenant_class == R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT:
        notes = "tenant_id=evidence.tenant_id from active execution evidence context"
    return R4LlmCallCallSite(
        module=site.module,
        containing_symbol=site.containing_symbol,
        symbol="record_llm_call_runtime_event",
        tenant_param_class=tenant_class,
        notes=notes[:240],
    )


def reconcile_context_assembly_inventory(
    call_sites: tuple[R4ContextAssemblyCallSite, ...],
) -> R4ContextAssemblyCallerInventory:
    canonical = sum(
        1 for s in call_sites if s.path_class == R4ContextAssemblyPathClass.PRODUCTION_CALLSITE_CANONICALLY_BOUND
    )
    unsafe = sum(
        1
        for s in call_sites
        if s.path_class == R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
    )
    unknown = sum(1 for s in call_sites if s.path_class == R4ContextAssemblyPathClass.UNKNOWN_CALLSITE)
    total = len(call_sites)
    if total != canonical + unsafe + unknown:
        unknown += total - (canonical + unsafe + unknown)
    helper_checks: tuple[tuple[str, str], ...] = (
        (_CONTEXT_RECORDING, 'metadata.get("tenant_id")'),
    )
    helper_ok = bool(all_structural_patterns_present(helper_checks))
    return R4ContextAssemblyCallerInventory(
        production_callers_total=total,
        canonically_bound=canonical,
        caller_controlled_unsafe=unsafe,
        unknown=unknown,
        call_sites=call_sites,
        helper_path_class=(
            R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
            if helper_ok
            else R4ContextAssemblyPathClass.UNKNOWN_CALLSITE
        ),
    )


def reconcile_llm_call_inventory(
    call_sites: tuple[R4LlmCallCallSite, ...],
) -> R4LlmCallCallerInventory:
    canonical = sum(1 for s in call_sites if s.tenant_param_class == R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT)
    validated = sum(1 for s in call_sites if s.tenant_param_class == R4LlmCallTenantParamClass.VALIDATED_BOUND_TENANT)
    untrusted = sum(1 for s in call_sites if s.tenant_param_class == R4LlmCallTenantParamClass.UNTRUSTED_CALLER_TENANT)
    ambient = sum(1 for s in call_sites if s.tenant_param_class == R4LlmCallTenantParamClass.AMBIENT_RUNTIME_TENANT)
    unknown = sum(1 for s in call_sites if s.tenant_param_class == R4LlmCallTenantParamClass.UNKNOWN)
    total = len(call_sites)
    classified = canonical + validated + untrusted + ambient + unknown
    if total != classified:
        unknown += total - classified
    return R4LlmCallCallerInventory(
        production_callers_total=total,
        canonical_tenant=canonical,
        validated_explicit_tenant=validated,
        untrusted_explicit_tenant=untrusted,
        unknown=unknown,
        ambient_fallback_only=ambient,
        call_sites=call_sites,
    )


def reconcile_maybe_record_llm_inventory(
    call_sites: tuple[R4LlmCallCallSite, ...],
) -> R4MaybeRecordLlmCallInventory:
    inv = reconcile_llm_call_inventory(call_sites)
    return R4MaybeRecordLlmCallInventory(
        production_callers_total=inv.production_callers_total,
        canonical_tenant=inv.canonical_tenant,
        validated_explicit_tenant=inv.validated_explicit_tenant,
        untrusted_explicit_tenant=inv.untrusted_explicit_tenant,
        unknown=inv.unknown,
        call_sites=inv.call_sites,
    )


def _required_paths_present() -> bool:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    return all((root / rel).is_file() for rel in _REQUIRED_PRODUCTION_PATHS)


def build_record_context_assembly_inventory() -> R4ContextAssemblyCallerInventory:
    discovered = discover_r4_calls_for_callee("record_context_assembly")
    classified = tuple(_classify_discovered_context_site(s) for s in discovered)
    inv = reconcile_context_assembly_inventory(classified)
    if not _required_paths_present():
        return reconcile_context_assembly_inventory(
            (
                *classified,
                R4ContextAssemblyCallSite(
                    module="<missing-required-path>",
                    containing_symbol="<module>",
                    path_class=R4ContextAssemblyPathClass.UNKNOWN_CALLSITE,
                    tenant_source_expression="required production path missing",
                    validation_before_helper="fail-closed",
                ),
            )
        )
    return inv


def build_record_llm_call_inventory() -> R4LlmCallCallerInventory:
    discovered = discover_r4_calls_for_callee("record_llm_call_runtime_event")
    classified = tuple(_classify_discovered_llm_direct_site(s) for s in discovered)
    inv = reconcile_llm_call_inventory(classified)
    if not _required_paths_present():
        return reconcile_llm_call_inventory(
            (
                *classified,
                R4LlmCallCallSite(
                    module="<missing-required-path>",
                    containing_symbol="<module>",
                    symbol="record_llm_call_runtime_event",
                    tenant_param_class=R4LlmCallTenantParamClass.UNKNOWN,
                    notes="required production path missing",
                ),
            )
        )
    return inv


def build_maybe_record_llm_call_from_usage_end_inventory() -> R4MaybeRecordLlmCallInventory:
    wrapper_bounded = all_structural_patterns_present(
        ((_LLM_RECORDING, "tenant_id=evidence.tenant_id"),)
    )
    discovered = discover_r4_calls_for_callee("maybe_record_llm_call_from_usage_end")
    sites: list[R4LlmCallCallSite] = []
    for site in discovered:
        tenant_class = classify_maybe_record_llm_call_caller(
            site,
            wrapper_structurally_bounded=bool(wrapper_bounded),
        )
        notes = (
            "production adapter usage_end → bounded runtime recorder"
            if site.module == _USAGE_LOG
            else "caller delegates tenant authority to bounded wrapper"
        )
        sites.append(
            R4LlmCallCallSite(
                module=site.module,
                containing_symbol=site.containing_symbol,
                symbol="maybe_record_llm_call_from_usage_end",
                tenant_param_class=tenant_class,
                notes=notes,
            )
        )
    inv = reconcile_maybe_record_llm_inventory(tuple(sites))
    if not _required_paths_present():
        return reconcile_maybe_record_llm_inventory(
            (
                *inv.call_sites,
                R4LlmCallCallSite(
                    module="<missing-required-path>",
                    containing_symbol="<module>",
                    symbol="maybe_record_llm_call_from_usage_end",
                    tenant_param_class=R4LlmCallTenantParamClass.UNKNOWN,
                    notes="required production path missing",
                ),
            )
        )
    return inv


def context_assembly_production_path_closed() -> bool:
    inv = build_record_context_assembly_inventory()
    return (
        inv.unknown == 0
        and inv.caller_controlled_unsafe == 0
        and inv.production_callers_total > 0
        and inv.canonically_bound == inv.production_callers_total
        and inv.helper_path_class == R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
        and inv.production_callers_total
        == inv.canonically_bound + inv.caller_controlled_unsafe + inv.unknown
    )


def llm_call_production_tenant_closed() -> bool:
    direct = build_record_llm_call_inventory()
    wrapper = build_maybe_record_llm_call_from_usage_end_inventory()
    direct_ok = (
        direct.unknown == 0
        and direct.untrusted_explicit_tenant == 0
        and direct.production_callers_total
        == direct.canonical_tenant
        + direct.validated_explicit_tenant
        + direct.untrusted_explicit_tenant
        + direct.unknown
        + direct.ambient_fallback_only
    )
    wrapper_ok = (
        wrapper.unknown == 0
        and wrapper.untrusted_explicit_tenant == 0
        and wrapper.production_callers_total > 0
        and wrapper.production_callers_total
        == wrapper.canonical_tenant
        + wrapper.validated_explicit_tenant
        + wrapper.untrusted_explicit_tenant
        + wrapper.unknown
    )
    return direct_ok and wrapper_ok
