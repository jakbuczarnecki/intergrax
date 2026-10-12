# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 bounded production call-site inventory (symbol grep SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present, read_production
from tests.qualification.tenant_x._r4_types import (
    R4ContextAssemblyPathClass,
    R4LlmCallTenantParamClass,
)

_CONTEXT_MANAGER: Final[str] = "intergrax/runtime/nexus/context/context_manager.py"
_CONTEXT_RECORDING: Final[str] = "intergrax/runtime/events/context_skill_recording.py"
_LLM_RECORDING: Final[str] = "intergrax/runtime/events/llm_call_recording.py"
_USAGE_LOG: Final[str] = "intergrax/llm_adapters/base/usage_log.py"

_CANONICAL_METADATA_BIND: Final[str] = '"tenant_id": task.tenant_id'


@dataclass(frozen=True, slots=True)
class R4ContextAssemblyCallSite:
    module: str
    path_class: R4ContextAssemblyPathClass
    tenant_source_expression: str
    validation_before_helper: str


@dataclass(frozen=True, slots=True)
class R4LlmCallCallSite:
    module: str
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


def _context_manager_call_site_count() -> int:
    text = read_production(_CONTEXT_MANAGER)
    return text.count("record_context_assembly(")


def _context_manager_canonical_sites() -> int:
    text = read_production(_CONTEXT_MANAGER)
    # Each production call must bind tenant after bundle spread (canonical wins).
    blocks = text.split("record_context_assembly(")
    bound = 0
    for block in blocks[1:]:
        head = block[:600]
        if _CANONICAL_METADATA_BIND in head and "**bundle_metadata" in head:
            bound += 1
    return bound


def build_record_context_assembly_inventory() -> R4ContextAssemblyCallerInventory:
    helper_checks: tuple[tuple[str, str], ...] = (
        (_CONTEXT_RECORDING, 'metadata.get("tenant_id")'),
    )
    helper_ok = bool(all_structural_patterns_present(helper_checks))

    total = _context_manager_call_site_count()
    bound = _context_manager_canonical_sites()
    unknown = max(0, total - bound)
    unsafe = 0

    call_sites: tuple[R4ContextAssemblyCallSite, ...] = ()
    if total:
        call_sites = tuple(
            R4ContextAssemblyCallSite(
                module=_CONTEXT_MANAGER,
                path_class=R4ContextAssemblyPathClass.PRODUCTION_CALLSITE_CANONICALLY_BOUND,
                tenant_source_expression="task.tenant_id → metadata['tenant_id'] (after **bundle_metadata)",
                validation_before_helper="Task.tenant_id on execution path; no pre-helper tenant validator",
            )
            for _ in range(bound)
        )

    return R4ContextAssemblyCallerInventory(
        production_callers_total=total,
        canonically_bound=bound,
        caller_controlled_unsafe=unsafe,
        unknown=unknown,
        call_sites=call_sites,
        helper_path_class=(
            R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
            if helper_ok
            else R4ContextAssemblyPathClass.UNKNOWN_CALLSITE
        ),
    )


def build_record_llm_call_inventory() -> R4LlmCallCallerInventory:
    production: tuple[R4LlmCallCallSite, ...] = (
        R4LlmCallCallSite(
            module=_LLM_RECORDING,
            symbol="maybe_record_llm_call_from_usage_end",
            tenant_param_class=R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT,
            notes="tenant_id=evidence.tenant_id from active execution evidence context",
        ),
        R4LlmCallCallSite(
            module=_USAGE_LOG,
            symbol="maybe_record_llm_call_from_usage_end",
            tenant_param_class=R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT,
            notes="production adapter usage_end → bounded runtime recorder",
        ),
    )
    checks: tuple[tuple[str, str], ...] = (
        (_LLM_RECORDING, "tenant_id=evidence.tenant_id"),
        (_LLM_RECORDING, "resolved_tenant = tenant_id if tenant_id else peek_active_runtime_event_tenant_id()"),
        (_USAGE_LOG, "maybe_record_llm_call_from_usage_end"),
    )
    if not all_structural_patterns_present(checks):
        return R4LlmCallCallerInventory(
            production_callers_total=0,
            canonical_tenant=0,
            validated_explicit_tenant=0,
            untrusted_explicit_tenant=0,
            unknown=1,
            ambient_fallback_only=0,
            call_sites=(),
        )

    return R4LlmCallCallerInventory(
        production_callers_total=len(production),
        canonical_tenant=2,
        validated_explicit_tenant=0,
        untrusted_explicit_tenant=0,
        unknown=0,
        ambient_fallback_only=1,
        call_sites=production,
    )


def context_assembly_production_path_closed() -> bool:
    inv = build_record_context_assembly_inventory()
    return (
        inv.unknown == 0
        and inv.caller_controlled_unsafe == 0
        and inv.production_callers_total > 0
        and inv.canonically_bound == inv.production_callers_total
        and inv.helper_path_class == R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED
    )


def llm_call_production_tenant_closed() -> bool:
    inv = build_record_llm_call_inventory()
    return inv.unknown == 0 and inv.untrusted_explicit_tenant == 0
