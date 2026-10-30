# © Artur Czarnecki. All rights reserved.
# ruff: noqa: UP042

"""COMPAT-X-R3-P0 closed-world RuntimeEvent evolution inventory (qualification evidence only)."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final

from intergrax.contracts.runtime_event import parse_runtime_event_payload
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    bootstrap_canonical_payload_registry,
    list_registered_payload_schema_ids,
    register_payload_schema,
)
from intergrax.runtime.events.payloads import CANONICAL_PAYLOAD_TYPES, RuntimeEventPayload
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextAssemblyPayloadV3,
    ContextAssemblyPayloadV4,
    ContextCandidatePayloadV1,
    DecisionPayloadV1,
    DelegationGrantedPayloadV1,
    ExecutionFailurePayloadV1,
    ExternalOperationFailurePayloadV1,
    GraphNodePayloadV1,
    HandoffPayloadV1,
    HumanPayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    LlmCallPayloadV2,
    LlmCallPayloadV3,
    SkillResolvedPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    TraceBridgePayloadV1,
    ValidationPayloadV1,
)
from intergrax.runtime.events.payloads.platform_families import (
    DiagnosticSubsystemFailurePayloadV1,
    PlatformHookViolationPayloadV1,
)
from intergrax.runtime.events.payloads.spine_families import (
    BudgetSignalPayloadV1,
    CancellationLifecyclePayloadV1,
    GraphBackpressurePayloadV1,
    GuardrailBlockedPayloadV1,
    HumanTimeoutPayloadV1,
    IntegrationConfigurationProvenanceRequirementPayloadV1,
    MemoryAccessPayloadV1,
    OperationalAlertPayloadV1,
    PauseLifecyclePayloadV1,
    PlanLifecyclePayloadV1,
    PolicyDecisionSpinePayloadV1,
    RetryLifecyclePayloadV1,
    TaskProgressPayloadV1,
    TracePersistedPayloadV1,
)
from intergrax.runtime.events.runtime_event_payload_policy import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    PayloadWriteMode,
    RuntimeEventPayloadPolicyEntry,
    RuntimeEventTypeClassification,
    iter_runtime_event_payload_policies,
)
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY, validate_schema_version

_QUALIFICATION_PREFIX: Final = "tests/qualification/compat_x/"
_COMPAT_X_ARCH_POLICY: Final = "docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md"
_RUNTIME_EVENT_CONTRACT: Final = "intergrax/contracts/runtime_event.py"
_RUNTIME_SCHEMA_REGISTRY: Final = "intergrax/runtime/schema/registry.py"
_PAYLOAD_REGISTRY: Final = "intergrax/runtime/events/payload_registry.py"
_PAYLOAD_MODELS: Final = "intergrax/runtime/events/payloads/"
_WRITE_POLICY: Final = "intergrax/runtime/events/runtime_event_payload_policy.py"
_SCHEMA_GUARD: Final = "intergrax/runtime/events/schema_guard.py"
_EVENT_KIND_REGISTRY: Final = "intergrax/runtime/events/event_kind_registry.py"
_PERSISTENCE_CONTRACT: Final = "intergrax/runtime/events/persistence_contract.py"
_VALIDATING_STORE: Final = "intergrax/runtime/events/stores/validating_runtime_event_store.py"
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_PRODUCTION_SCAN_ROOTS: Final = ("intergrax", "applications")
_PRODUCTION_SCAN_EXCLUDE: Final = frozenset({"tests", "testing_support", "docs", "scripts", "__pycache__"})


class RuntimeEventWritePathDisposition(str, Enum):
    CANONICAL_VALIDATING_PATH = "canonical_validating_path"
    RAW_BACKEND_INTERNAL = "raw_backend_internal"
    TEST_LAB_ONLY = "test_lab_only"
    NON_EVENT_APPEND = "non_event_append"
    UNCLASSIFIED = "unclassified"


class RuntimeEventCallSiteClassificationEvidence(str, Enum):
    """Typed positive evidence backing a disposition (default ``UNSPECIFIED`` = unknown)."""

    UNSPECIFIED = "unspecified"
    SECURITY_AUDIT_TRAIL_WIRING_MULTI_REGION = "security_audit_trail_wiring_multi_region"
    SECURITY_AUDIT_TRAIL_INNER_PRIMARY_DELEGATE = "security_audit_trail_inner_primary_delegate"
    PLATFORM_RUNTIME_EVENT_HISTORY_BUFFER = "platform_runtime_event_history_buffer"
    RUNTIME_EVENT_BUS_VALIDATING_PERSISTENCE = "runtime_event_bus_validating_persistence"
    GOVERNANCE_VALIDATING_EVIDENCE_PERSISTENCE = "governance_validating_evidence_persistence"
    VALIDATING_RUNTIME_EVENT_STORE_OUTER_APPEND = "validating_runtime_event_store_outer_append"
    VALIDATING_EVIDENCE_PERSISTENCE_PORT_OUTER = "validating_evidence_persistence_port_outer"
    RUNTIME_EVENT_STORE_BACKEND_IMPLEMENTATION = "runtime_event_store_backend_implementation"
    PERSISTENCE_CONTRACT_PROTOCOL_LAYER = "persistence_contract_protocol_layer"
    VALIDATING_WRAPPER_INNER_BACKEND_DELEGATE = "validating_wrapper_inner_backend_delegate"
    EVIDENCE_PERSISTENCE_ADAPTER_INNER_DELEGATE = "evidence_persistence_adapter_inner_delegate"
    PERSISTENCE_CONFORMANCE_LAB_HARNESS = "persistence_conformance_lab_harness"
    LAB_OR_POC_APPLICATION_SCOPE = "lab_or_poc_application_scope"


_NON_EVENT_APPEND_EVIDENCE: Final = frozenset(
    {
        RuntimeEventCallSiteClassificationEvidence.SECURITY_AUDIT_TRAIL_WIRING_MULTI_REGION,
        RuntimeEventCallSiteClassificationEvidence.SECURITY_AUDIT_TRAIL_INNER_PRIMARY_DELEGATE,
        RuntimeEventCallSiteClassificationEvidence.PLATFORM_RUNTIME_EVENT_HISTORY_BUFFER,
    }
)

_CANONICAL_VALIDATING_PATH_EVIDENCE: Final = frozenset(
    {
        RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_BUS_VALIDATING_PERSISTENCE,
        RuntimeEventCallSiteClassificationEvidence.GOVERNANCE_VALIDATING_EVIDENCE_PERSISTENCE,
        RuntimeEventCallSiteClassificationEvidence.VALIDATING_RUNTIME_EVENT_STORE_OUTER_APPEND,
        RuntimeEventCallSiteClassificationEvidence.VALIDATING_EVIDENCE_PERSISTENCE_PORT_OUTER,
    }
)

_RAW_BACKEND_INTERNAL_EVIDENCE: Final = frozenset(
    {
        RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_STORE_BACKEND_IMPLEMENTATION,
        RuntimeEventCallSiteClassificationEvidence.PERSISTENCE_CONTRACT_PROTOCOL_LAYER,
        RuntimeEventCallSiteClassificationEvidence.VALIDATING_WRAPPER_INNER_BACKEND_DELEGATE,
        RuntimeEventCallSiteClassificationEvidence.EVIDENCE_PERSISTENCE_ADAPTER_INNER_DELEGATE,
    }
)

_TEST_LAB_ONLY_EVIDENCE: Final = frozenset(
    {
        RuntimeEventCallSiteClassificationEvidence.PERSISTENCE_CONFORMANCE_LAB_HARNESS,
        RuntimeEventCallSiteClassificationEvidence.LAB_OR_POC_APPLICATION_SCOPE,
    }
)

_RAW_BACKEND_INTERNAL_MODULE_MARKERS: Final = (
    "runtime/events/stores/sqlite_runtime_event_store.py",
    "runtime/events/stores/memory_runtime_event_store.py",
    "runtime/events/stores/document_backed_runtime_event_store.py",
    "runtime/events/persistence_contract.py",
    "runtime/events/stores/validating_runtime_event_store.py",
    "runtime/events/validating_evidence_persistence_port.py",
    "runtime/events/evidence_persistence_adapter.py",
)


@dataclass(frozen=True, slots=True)
class RuntimeEventAppendCallSiteClassification:
    disposition: RuntimeEventWritePathDisposition
    evidence: RuntimeEventCallSiteClassificationEvidence


@dataclass(frozen=True, slots=True)
class RuntimeEventPersistenceCallSite:
    module_path: str
    lineno: int
    qualname: str
    callee_expr: str
    disposition: RuntimeEventWritePathDisposition
    classification_evidence: RuntimeEventCallSiteClassificationEvidence = (
        RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED
    )


class EventEvolutionSurfaceKind(str, Enum):
    ENVELOPE = "envelope"
    CANONICAL_PAYLOAD = "canonical_payload"
    EXTENSION_PAYLOAD = "extension_payload"
    WRITE_POLICY = "write_policy"
    EVENT_TYPE = "event_type"


class EventVersionDisposition(str, Enum):
    CURRENT = "current"
    HISTORICAL_SUPPORTED = "historical_supported"
    RETIRED = "retired"
    UNKNOWN = "unknown"


class EventCompatibilityDisposition(str, Enum):
    CURRENT_WRITE_AND_READ = "current_write_and_read"
    SUPPORTED_READ_ONLY = "supported_read_only"
    RETIRED_FAIL_CLOSED = "retired_fail_closed"
    EXTENSION_OWNED_NEW_ID_REQUIRED = "extension_owned_new_id_required"
    UNKNOWN_FAIL_CLOSED = "unknown_fail_closed"


@dataclass(frozen=True, slots=True)
class EventEnvelopeEvolutionRecord:
    family_id: str
    schema_version: str
    version_disposition: EventVersionDisposition
    compatibility_disposition: EventCompatibilityDisposition
    semantic_owner_path: str
    migration_required: bool
    adaptation_required: bool


@dataclass(frozen=True, slots=True)
class EventPayloadVersionRecord:
    payload_type: type[RuntimeEventPayload]
    schema_id: str
    payload_schema_version: int
    compatibility_disposition: EventCompatibilityDisposition
    migration_required: bool
    adaptation_required: bool


@dataclass(frozen=True, slots=True)
class EventPayloadEvolutionFamily:
    family_id: str
    versions: tuple[EventPayloadVersionRecord, ...]
    semantic_owner_path: str
    composition_owner_path: str
    compatibility_semantics_owner_path: str
    allows_multiple_active_write_versions: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ExtensionEventEvolutionContract:
    rules: tuple[str, ...]
    enforcement_owner_path: str
    registration_owner_path: str


@dataclass(frozen=True, slots=True)
class EventEvolutionOwnerRow:
    concern: str
    semantic_owner_path: str
    composition_owner_path: str
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventEvolutionInventory:
    envelope_records: tuple[EventEnvelopeEvolutionRecord, ...]
    payload_families: tuple[EventPayloadEvolutionFamily, ...]
    extension_contract: ExtensionEventEvolutionContract
    owner_matrix: tuple[EventEvolutionOwnerRow, ...]
    production_persistence_guard_paths: tuple[str, ...]
    production_persistence_call_sites: tuple[RuntimeEventPersistenceCallSite, ...]


@dataclass(frozen=True, slots=True)
class EventEvolutionR3P0Metrics:
    runtime_event_type_count: int
    canonical_payload_class_count: int
    payload_schema_id_count: int
    single_version_family_count: int
    multi_version_family_count: int
    supported_read_version_count: int
    current_write_version_count: int
    retired_envelope_version_count: int
    strict_production_write_policies: int
    extension_event_kind_policies: int
    unclassified_payload_classes: int
    duplicate_schema_ids: int
    duplicate_family_membership: int
    duplicate_write_ownership: int


_PAYLOAD_CLASS_TO_FAMILY: dict[type[RuntimeEventPayload], str] = {
    DiagnosticSubsystemFailurePayloadV1: "platform_diagnostic_subsystem_failure",
    PlatformHookViolationPayloadV1: "platform_hook_violation",
    DelegationGrantedPayloadV1: "delegation_granted",
    DecisionPayloadV1: "decision",
    ToolPayloadV1: "tool",
    ValidationPayloadV1: "validation",
    InterruptPayloadV1: "interrupt",
    HumanPayloadV1: "human",
    HandoffPayloadV1: "handoff",
    AgentSelectionPayloadV1: "agent_selection",
    GraphNodePayloadV1: "graph_node",
    LlmCallPayloadV1: "llm_call",
    LlmCallPayloadV2: "llm_call",
    LlmCallPayloadV3: "llm_call",
    TraceBridgePayloadV1: "trace_bridge",
    SkillResolvedPayloadV1: "skill_resolved",
    ContextAssemblyPayloadV1: "context_assembly",
    ContextAssemblyPayloadV2: "context_assembly",
    ContextAssemblyPayloadV3: "context_assembly",
    ContextAssemblyPayloadV4: "context_assembly",
    ContextCandidatePayloadV1: "context_candidate",
    ExecutionFailurePayloadV1: "execution_failure",
    ExternalOperationFailurePayloadV1: "external_operation_failure",
    TaskLifecyclePayloadV1: "task_lifecycle",
    PlanLifecyclePayloadV1: "plan_lifecycle",
    PauseLifecyclePayloadV1: "pause_lifecycle",
    RetryLifecyclePayloadV1: "retry_lifecycle",
    CancellationLifecyclePayloadV1: "cancellation_lifecycle",
    MemoryAccessPayloadV1: "memory_access",
    OperationalAlertPayloadV1: "operational_alert",
    HumanTimeoutPayloadV1: "human_timeout",
    PolicyDecisionSpinePayloadV1: "policy_decision_spine",
    BudgetSignalPayloadV1: "budget_signal",
    IntegrationConfigurationProvenanceRequirementPayloadV1: ("integration_configuration_provenance_requirement"),
    GraphBackpressurePayloadV1: "graph_backpressure",
    GuardrailBlockedPayloadV1: "guardrail_blocked",
    TracePersistedPayloadV1: "trace_persisted",
    TaskProgressPayloadV1: "task_progress",
}

_MULTI_WRITE_FAMILIES: frozenset[str] = frozenset({"context_assembly"})

_PREFERRED_SCHEMA_IDS: frozenset[str] = frozenset(EVENT_TYPE_PREFERRED_SCHEMA.values())


def _compatibility_for_schema(schema_id: str) -> EventCompatibilityDisposition:
    if schema_id in _PREFERRED_SCHEMA_IDS:
        return EventCompatibilityDisposition.CURRENT_WRITE_AND_READ
    return EventCompatibilityDisposition.SUPPORTED_READ_ONLY


def _build_payload_families() -> tuple[EventPayloadEvolutionFamily, ...]:
    by_family: dict[str, list[EventPayloadVersionRecord]] = {}
    for payload_type in CANONICAL_PAYLOAD_TYPES:
        family_id = _PAYLOAD_CLASS_TO_FAMILY.get(payload_type)
        if family_id is None:
            continue
        schema_id = payload_type.schema_id
        record = EventPayloadVersionRecord(
            payload_type=payload_type,
            schema_id=schema_id,
            payload_schema_version=int(payload_type.schema_version),
            compatibility_disposition=_compatibility_for_schema(schema_id),
            migration_required=False,
            adaptation_required=False,
        )
        by_family.setdefault(family_id, []).append(record)

    families: list[EventPayloadEvolutionFamily] = []
    for family_id in sorted(by_family):
        versions = tuple(sorted(by_family[family_id], key=lambda r: r.schema_id))
        families.append(
            EventPayloadEvolutionFamily(
                family_id=family_id,
                versions=versions,
                semantic_owner_path=_PAYLOAD_REGISTRY,
                composition_owner_path=_EVENT_KIND_REGISTRY,
                compatibility_semantics_owner_path=_COMPAT_X_ARCH_POLICY,
                allows_multiple_active_write_versions=family_id in _MULTI_WRITE_FAMILIES,
                notes=(
                    "Multiple event types select different current-write payload versions."
                    if family_id in _MULTI_WRITE_FAMILIES
                    else ""
                ),
            )
        )
    return tuple(families)


def _envelope_records() -> tuple[EventEnvelopeEvolutionRecord, ...]:
    current = RUNTIME_SCHEMA_REGISTRY["runtime_event"]
    return (
        EventEnvelopeEvolutionRecord(
            family_id="runtime_event",
            schema_version=current,
            version_disposition=EventVersionDisposition.CURRENT,
            compatibility_disposition=EventCompatibilityDisposition.CURRENT_WRITE_AND_READ,
            semantic_owner_path=_RUNTIME_SCHEMA_REGISTRY,
            migration_required=False,
            adaptation_required=False,
        ),
        EventEnvelopeEvolutionRecord(
            family_id="runtime_event",
            schema_version="runtime_event.v1",
            version_disposition=EventVersionDisposition.RETIRED,
            compatibility_disposition=EventCompatibilityDisposition.RETIRED_FAIL_CLOSED,
            semantic_owner_path=_RUNTIME_EVENT_CONTRACT,
            migration_required=False,
            adaptation_required=False,
        ),
    )


def _extension_contract() -> ExtensionEventEvolutionContract:
    return ExtensionEventEvolutionContract(
        rules=(
            "Platform canonical payload schema IDs cannot be overridden by extension registration.",
            "Duplicate schema_id bound to a different class fails closed at registration.",
            "Unknown extension payload_schema_id fails closed on validation.",
            "Extension evolution requires a new explicit schema identity; no implicit ID mutation.",
            "Platform does not infer backward compatibility for extension schemas.",
            "Extension registration does not grant Execution or Governance authority.",
            "Event compatibility never widens tenant scope.",
        ),
        enforcement_owner_path=_SCHEMA_GUARD,
        registration_owner_path=_PAYLOAD_REGISTRY,
    )


def _owner_matrix() -> tuple[EventEvolutionOwnerRow, ...]:
    return (
        EventEvolutionOwnerRow(
            concern="event_evolution_policy",
            semantic_owner_path=_COMPAT_X_ARCH_POLICY,
            composition_owner_path=_COMPAT_X_ARCH_POLICY,
            evidence_paths=(_COMPAT_X_ARCH_POLICY,),
        ),
        EventEvolutionOwnerRow(
            concern="runtime_event_envelope_current_version",
            semantic_owner_path=_RUNTIME_SCHEMA_REGISTRY,
            composition_owner_path=_RUNTIME_EVENT_CONTRACT,
            evidence_paths=(_RUNTIME_SCHEMA_REGISTRY, _RUNTIME_EVENT_CONTRACT),
        ),
        EventEvolutionOwnerRow(
            concern="payload_schema_identity",
            semantic_owner_path=_PAYLOAD_REGISTRY,
            composition_owner_path=_PAYLOAD_MODELS,
            evidence_paths=(_PAYLOAD_REGISTRY, _PAYLOAD_MODELS),
        ),
        EventEvolutionOwnerRow(
            concern="canonical_payload_write_selection",
            semantic_owner_path=_WRITE_POLICY,
            composition_owner_path=_WRITE_POLICY,
            evidence_paths=(_WRITE_POLICY,),
        ),
        EventEvolutionOwnerRow(
            concern="event_kind_composition",
            semantic_owner_path=_EVENT_KIND_REGISTRY,
            composition_owner_path=_EVENT_KIND_REGISTRY,
            evidence_paths=(_EVENT_KIND_REGISTRY,),
        ),
        EventEvolutionOwnerRow(
            concern="event_schema_enforcement",
            semantic_owner_path=_SCHEMA_GUARD,
            composition_owner_path=_SCHEMA_GUARD,
            evidence_paths=(_SCHEMA_GUARD, _VALIDATING_STORE),
        ),
    )


def _repo_relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _iter_production_python_sources() -> list[Path]:
    paths: list[Path] = []
    for root_name in _PRODUCTION_SCAN_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for file_path in root.rglob("*.py"):
            rel_parts = file_path.relative_to(_REPO_ROOT).parts
            if any(part in _PRODUCTION_SCAN_EXCLUDE for part in rel_parts):
                continue
            # Generated Docker image context; mirrors canonical ``intergrax/`` — not an independent SSOT tree.
            if "docker" in rel_parts and "runtime-context" in rel_parts:
                continue
            paths.append(file_path)
    return sorted(paths)


def count_production_python_files_scanned() -> int:
    return len(_iter_production_python_sources())


def _enclosing_qualname(tree: ast.Module, lineno: int) -> str:
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno):
                if best is None or node.lineno >= best[0]:
                    best = (node.lineno, node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.lineno <= lineno <= getattr(item, "end_lineno", item.lineno):
                        qual = f"{node.name}.{item.name}"
                        if best is None or item.lineno >= best[0]:
                            best = (item.lineno, qual)
    return best[1] if best else "<module>"


def _callee_expr(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        base = ast.unparse(node.func.value) if hasattr(ast, "unparse") else "?"
        return f"{base}.{node.func.attr}"
    if isinstance(node.func, ast.Name):
        return node.func.id
    return "?"


def classify_runtime_event_append_call_site(
    module_path: str,
    qualname: str,
    callee_expr: str,
) -> RuntimeEventAppendCallSiteClassification:
    """Classify discovered ``.append(..., tenant_id=...)`` sites; unknown → ``UNCLASSIFIED``."""
    path = module_path.replace("\\", "/")
    unspecified = RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED
    if "runtime/events/stores/sqlite_runtime_event_store.py" in path:
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
            RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_STORE_BACKEND_IMPLEMENTATION,
        )
    if "runtime/events/stores/memory_runtime_event_store.py" in path:
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
            RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_STORE_BACKEND_IMPLEMENTATION,
        )
    if "runtime/events/stores/document_backed_runtime_event_store.py" in path:
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
            RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_STORE_BACKEND_IMPLEMENTATION,
        )
    if path.endswith("runtime/events/persistence_contract.py"):
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
            RuntimeEventCallSiteClassificationEvidence.PERSISTENCE_CONTRACT_PROTOCOL_LAYER,
        )
    if path.endswith("runtime/events/stores/validating_runtime_event_store.py"):
        if "_inner" in callee_expr:
            return RuntimeEventAppendCallSiteClassification(
                RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
                RuntimeEventCallSiteClassificationEvidence.VALIDATING_WRAPPER_INNER_BACKEND_DELEGATE,
            )
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH,
            RuntimeEventCallSiteClassificationEvidence.VALIDATING_RUNTIME_EVENT_STORE_OUTER_APPEND,
        )
    if "runtime/events/validating_evidence_persistence_port.py" in path:
        if "_inner" in callee_expr:
            return RuntimeEventAppendCallSiteClassification(
                RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
                RuntimeEventCallSiteClassificationEvidence.VALIDATING_WRAPPER_INNER_BACKEND_DELEGATE,
            )
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH,
            RuntimeEventCallSiteClassificationEvidence.VALIDATING_EVIDENCE_PERSISTENCE_PORT_OUTER,
        )
    if path.endswith("runtime/events/evidence_persistence_adapter.py"):
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL,
            RuntimeEventCallSiteClassificationEvidence.EVIDENCE_PERSISTENCE_ADAPTER_INNER_DELEGATE,
        )
    if path.endswith("runtime/events/event_bus.py") and "_persistence" in callee_expr:
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH,
            RuntimeEventCallSiteClassificationEvidence.RUNTIME_EVENT_BUS_VALIDATING_PERSISTENCE,
        )
    if path.endswith("runtime/governance/governance_evidence_persistence.py"):
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH,
            RuntimeEventCallSiteClassificationEvidence.GOVERNANCE_VALIDATING_EVIDENCE_PERSISTENCE,
        )
    if path.endswith("runtime/observability/persistence_conformance.py"):
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.TEST_LAB_ONLY,
            RuntimeEventCallSiteClassificationEvidence.PERSISTENCE_CONFORMANCE_LAB_HARNESS,
        )
    if "lab_application/" in path or "poc_template_application/" in path:
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.TEST_LAB_ONLY,
            RuntimeEventCallSiteClassificationEvidence.LAB_OR_POC_APPLICATION_SCOPE,
        )
    if path.endswith("runtime/events/runtime_event_history.py"):
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.NON_EVENT_APPEND,
            RuntimeEventCallSiteClassificationEvidence.PLATFORM_RUNTIME_EVENT_HISTORY_BUFFER,
        )
    if path.endswith("applications/_shared/security_audit_trail_wiring.py") and callee_expr == "trail.append":
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.NON_EVENT_APPEND,
            RuntimeEventCallSiteClassificationEvidence.SECURITY_AUDIT_TRAIL_WIRING_MULTI_REGION,
        )
    if path.endswith("runtime/security/multi_region_audit_trail.py") and callee_expr == "self._primary.append":
        return RuntimeEventAppendCallSiteClassification(
            RuntimeEventWritePathDisposition.NON_EVENT_APPEND,
            RuntimeEventCallSiteClassificationEvidence.SECURITY_AUDIT_TRAIL_INNER_PRIMARY_DELEGATE,
        )
    return RuntimeEventAppendCallSiteClassification(
        RuntimeEventWritePathDisposition.UNCLASSIFIED,
        unspecified,
    )


def _module_is_raw_backend_internal_delegate(module_path: str) -> bool:
    path = module_path.replace("\\", "/")
    return any(marker in path for marker in _RAW_BACKEND_INTERNAL_MODULE_MARKERS)


@lru_cache(maxsize=1)
def discover_runtime_event_persistence_call_sites() -> tuple[RuntimeEventPersistenceCallSite, ...]:
    """AST scan for production ``.append(..., tenant_id=...)`` runtime-event persistence writes."""
    discovered: list[RuntimeEventPersistenceCallSite] = []
    for file_path in _iter_production_python_sources():
        module_path = _repo_relative(file_path)
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module_path)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
                continue
            has_tenant_kw = any(kw.arg == "tenant_id" for kw in node.keywords)
            if not has_tenant_kw:
                continue
            callee = _callee_expr(node)
            qualname = _enclosing_qualname(tree, node.lineno)
            classification = classify_runtime_event_append_call_site(module_path, qualname, callee)
            discovered.append(
                RuntimeEventPersistenceCallSite(
                    module_path=module_path,
                    lineno=node.lineno,
                    qualname=qualname,
                    callee_expr=callee,
                    disposition=classification.disposition,
                    classification_evidence=classification.evidence,
                )
            )
    return tuple(sorted(discovered, key=lambda s: (s.module_path, s.lineno)))


_CANONICAL_COMPOSITION_ROOTS: Final = (
    "intergrax/runtime/events/store.py",
    "intergrax/runtime/persistence/integration_profile_wiring.py",
    "intergrax/runtime/nexus/observability_wiring.py",
    "intergrax/runtime/events/evidence_persistence_adapter.py",
)


def _composition_root_references_validating_wrapper(module_path: str) -> bool:
    path = _REPO_ROOT / module_path
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8")
    return (
        "ValidatingRuntimeEventPersistence" in source
        or "ValidatingEvidencePersistencePort" in source
        or "resolve_runtime_event_persistence" in source
        or "open_runtime_event_store_from_profile" in source
    )


def validate_canonical_production_guard_chains() -> tuple[str, ...]:
    violations: list[str] = []
    for root in _CANONICAL_COMPOSITION_ROOTS:
        if not _composition_root_references_validating_wrapper(root):
            violations.append(f"composition root missing validating wrapper reference: {root}")
    return tuple(violations)


def validate_production_persistence_call_site_classification(
    call_sites: Iterable[RuntimeEventPersistenceCallSite] | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    sites = tuple(call_sites or discover_runtime_event_persistence_call_sites())
    for site in sites:
        if site.disposition == RuntimeEventWritePathDisposition.UNCLASSIFIED:
            violations.append(f"unclassified persistence append {site.module_path}:{site.lineno} {site.callee_expr}")
    violations.extend(validate_non_event_append_positive_evidence(sites))
    violations.extend(validate_call_site_disposition_evidence_alignment(sites))
    return tuple(violations)


def validate_non_event_append_positive_evidence(
    call_sites: Iterable[RuntimeEventPersistenceCallSite],
) -> tuple[str, ...]:
    violations: list[str] = []
    for site in call_sites:
        if site.disposition != RuntimeEventWritePathDisposition.NON_EVENT_APPEND:
            continue
        if site.classification_evidence not in _NON_EVENT_APPEND_EVIDENCE:
            violations.append(
                f"NON_EVENT_APPEND without positive evidence {site.module_path}:{site.lineno} {site.callee_expr}"
            )
    return tuple(violations)


def validate_call_site_disposition_evidence_alignment(
    call_sites: Iterable[RuntimeEventPersistenceCallSite],
) -> tuple[str, ...]:
    violations: list[str] = []
    for site in call_sites:
        evidence = site.classification_evidence
        disposition = site.disposition
        if disposition == RuntimeEventWritePathDisposition.UNCLASSIFIED:
            if evidence != RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED:
                violations.append(
                    f"UNCLASSIFIED site must not carry positive evidence: {site.module_path}:{site.lineno}"
                )
            continue
        if evidence == RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED:
            violations.append(f"classified site missing evidence {site.module_path}:{site.lineno} {disposition.value}")
            continue
        allowed: frozenset[RuntimeEventCallSiteClassificationEvidence]
        if disposition == RuntimeEventWritePathDisposition.NON_EVENT_APPEND:
            allowed = _NON_EVENT_APPEND_EVIDENCE
        elif disposition == RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH:
            allowed = _CANONICAL_VALIDATING_PATH_EVIDENCE
        elif disposition == RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL:
            allowed = _RAW_BACKEND_INTERNAL_EVIDENCE
        elif disposition == RuntimeEventWritePathDisposition.TEST_LAB_ONLY:
            allowed = _TEST_LAB_ONLY_EVIDENCE
        else:
            continue
        if evidence not in allowed:
            violations.append(
                f"evidence {evidence.value} incompatible with {disposition.value} at {site.module_path}:{site.lineno}"
            )
    return tuple(violations)


def count_unclassified_persistence_call_sites(
    call_sites: Iterable[RuntimeEventPersistenceCallSite] | None = None,
) -> int:
    sites = tuple(call_sites or discover_runtime_event_persistence_call_sites())
    return sum(1 for site in sites if site.disposition == RuntimeEventWritePathDisposition.UNCLASSIFIED)


def count_non_event_append_without_evidence(
    call_sites: Iterable[RuntimeEventPersistenceCallSite] | None = None,
) -> int:
    sites = tuple(call_sites or discover_runtime_event_persistence_call_sites())
    return sum(
        1
        for site in sites
        if site.disposition == RuntimeEventWritePathDisposition.NON_EVENT_APPEND
        and site.classification_evidence not in _NON_EVENT_APPEND_EVIDENCE
    )


def summarize_persistence_call_site_dispositions(
    call_sites: Iterable[RuntimeEventPersistenceCallSite] | None = None,
) -> dict[RuntimeEventWritePathDisposition, int]:
    sites = tuple(call_sites or discover_runtime_event_persistence_call_sites())
    counts = {disp: 0 for disp in RuntimeEventWritePathDisposition}
    for site in sites:
        counts[site.disposition] += 1
    return counts


def count_canonical_production_raw_store_bypasses(
    call_sites: Iterable[RuntimeEventPersistenceCallSite] | None = None,
) -> int:
    """Production persistence writes that bypass validating composition (not backend-internal delegates)."""
    sites = tuple(call_sites or discover_runtime_event_persistence_call_sites())
    bypass = 0
    for site in sites:
        disposition = site.disposition
        if disposition == RuntimeEventWritePathDisposition.UNCLASSIFIED:
            bypass += 1
            continue
        if disposition == RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH:
            if site.classification_evidence not in _CANONICAL_VALIDATING_PATH_EVIDENCE:
                bypass += 1
            continue
        if disposition == RuntimeEventWritePathDisposition.TEST_LAB_ONLY:
            continue
        if disposition == RuntimeEventWritePathDisposition.NON_EVENT_APPEND:
            if site.classification_evidence not in _NON_EVENT_APPEND_EVIDENCE:
                bypass += 1
            continue
        if disposition == RuntimeEventWritePathDisposition.RAW_BACKEND_INTERNAL:
            if site.classification_evidence not in _RAW_BACKEND_INTERNAL_EVIDENCE:
                bypass += 1
                continue
            if not _module_is_raw_backend_internal_delegate(site.module_path):
                bypass += 1
            continue
        bypass += 1
    return bypass


def detect_raw_runtime_event_persistence_without_validator(
    persistence: object,
) -> tuple[str, ...]:
    """Return violations when a composition root would expose unwrapped durable runtime-event persistence."""
    if persistence is None:
        return ()
    if isinstance(persistence, ValidatingRuntimeEventPersistence):
        return ()
    append = getattr(persistence, "append", None)
    if callable(append):
        return ("raw RuntimeEventPersistence without ValidatingRuntimeEventPersistence",)
    return ()


def synthetic_canonical_composition_raw_store_bypass_probe() -> tuple[str, ...]:
    """Adversarial: production composition exposing raw ``RuntimeEventPersistence`` without validator."""
    from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore

    return detect_raw_runtime_event_persistence_without_validator(InMemoryRuntimeEventStore())


def validate_strict_write_policies_against_registry(
    policies: Iterable[tuple[RuntimeEventType, RuntimeEventPayloadPolicyEntry]],
    *,
    registered_schema_ids: frozenset[str] | None = None,
) -> tuple[str, ...]:
    reg = registered_schema_ids or registered_canonical_payload_schema_ids()
    violations: list[str] = []
    for event_type, entry in policies:
        if entry.write_mode == PayloadWriteMode.STRICT_SPINE_SCHEMA and entry.schema_id:
            if entry.schema_id not in reg:
                violations.append(f"strict policy {event_type.value} references unregistered {entry.schema_id!r}")
    return tuple(violations)


def synthetic_write_policy_unregistered_schema_probe() -> tuple[str, ...]:
    """Probe I: inject write policy pointing at a schema id absent from the canonical registry."""
    phantom_id = "phantom.compat_x_probe.write_policy.v1"
    reg = registered_canonical_payload_schema_ids()
    if phantom_id in reg:
        return ("probe: phantom schema unexpectedly registered",)
    synthetic_policy = (
        (
            RuntimeEventType.TASK_CREATED,
            RuntimeEventPayloadPolicyEntry(
                classification=RuntimeEventTypeClassification.CANONICAL_PRODUCTION,
                write_mode=PayloadWriteMode.STRICT_SPINE_SCHEMA,
                schema_id=phantom_id,
            ),
        ),
    )
    return validate_strict_write_policies_against_registry(synthetic_policy, registered_schema_ids=reg)


def _payload_envelope_version_fail_closed_probe() -> tuple[str, ...]:
    from intergrax.runtime.events.payload_registry import RuntimeEventPayloadError, validate_payload_envelope

    failures: list[str] = []
    base = {
        "payload_schema_id": "tool.v1",
        "data": {"tool_name": "t", "status": "requested"},
    }
    cases: list[tuple[str, dict[str, object]]] = [
        ("missing_version", dict(base)),
        ("wrong_numeric", {**base, "payload_schema_version": 99}),
        ("wrong_type_str", {**base, "payload_schema_version": "1"}),
        ("wrong_type_bool", {**base, "payload_schema_version": True}),
        ("wrong_type_none", {**base, "payload_schema_version": None}),
    ]
    for label, envelope in cases:
        try:
            validate_payload_envelope(envelope)  # type: ignore[arg-type]
        except RuntimeEventPayloadError:
            continue
        except Exception as exc:
            failures.append(f"{label}: unexpected {type(exc).__name__}")
        else:
            failures.append(f"{label}: unexpected success")
    return tuple(failures)


def validate_payload_version_inventory_consistency(
    inventory: EventEvolutionInventory | None = None,
) -> tuple[str, ...]:
    inv = inventory or event_evolution_inventory()
    violations: list[str] = []
    for family in inv.payload_families:
        for version in family.versions:
            payload_type = version.payload_type
            if not getattr(payload_type, "schema_id", None):
                violations.append(f"missing schema_id on {payload_type.__name__}")
            class_version = getattr(payload_type, "schema_version", None)
            if type(class_version) is not int:
                violations.append(f"missing or non-int schema_version on {payload_type.__name__}")
            elif version.payload_schema_version != class_version:
                violations.append(
                    f"inventory version mismatch for {payload_type.__name__}: "
                    f"{version.payload_schema_version} vs class {class_version}"
                )
            if version.schema_id != payload_type.schema_id:
                violations.append(
                    f"inventory schema_id mismatch for {payload_type.__name__}: "
                    f"{version.schema_id!r} vs {payload_type.schema_id!r}"
                )
    return tuple(violations)


@lru_cache(maxsize=1)
def event_evolution_inventory() -> EventEvolutionInventory:
    """Authoritative R3-P0 closed-world inventory (qualification evidence; not runtime authority)."""
    bootstrap_canonical_payload_registry()
    return EventEvolutionInventory(
        envelope_records=_envelope_records(),
        payload_families=_build_payload_families(),
        extension_contract=_extension_contract(),
        owner_matrix=_owner_matrix(),
        production_persistence_guard_paths=_CANONICAL_COMPOSITION_ROOTS,
        production_persistence_call_sites=discover_runtime_event_persistence_call_sites(),
    )


def compute_event_evolution_r3_p0_metrics(
    inventory: EventEvolutionInventory | None = None,
) -> EventEvolutionR3P0Metrics:
    inv = inventory or event_evolution_inventory()
    all_versions: list[EventPayloadVersionRecord] = [v for family in inv.payload_families for v in family.versions]
    supported_read = sum(
        1
        for v in all_versions
        if v.compatibility_disposition
        in (
            EventCompatibilityDisposition.CURRENT_WRITE_AND_READ,
            EventCompatibilityDisposition.SUPPORTED_READ_ONLY,
        )
    )
    current_write = sum(
        1 for v in all_versions if v.compatibility_disposition == EventCompatibilityDisposition.CURRENT_WRITE_AND_READ
    )
    multi = sum(1 for f in inv.payload_families if len(f.versions) > 1)
    single = len(inv.payload_families) - multi
    strict = sum(
        1
        for _event_type, entry in iter_runtime_event_payload_policies()
        if entry.write_mode == PayloadWriteMode.STRICT_SPINE_SCHEMA and entry.schema_id
    )
    extension = sum(
        1
        for _event_type, entry in iter_runtime_event_payload_policies()
        if entry.write_mode == PayloadWriteMode.EXTENSION_EVENT_KIND
    )
    classified = {v.payload_type for v in all_versions}
    unclassified = len([c for c in CANONICAL_PAYLOAD_TYPES if c not in classified])
    schema_ids = [v.schema_id for v in all_versions]
    dup_schema = len(schema_ids) - len(set(schema_ids))
    family_by_class: dict[type[RuntimeEventPayload], str] = {}
    dup_membership = 0
    for family in inv.payload_families:
        for version in family.versions:
            prior = family_by_class.get(version.payload_type)
            if prior is not None and prior != family.family_id:
                dup_membership += 1
            else:
                family_by_class[version.payload_type] = family.family_id
    write_owner_dup = _duplicate_write_policy_violations()
    return EventEvolutionR3P0Metrics(
        runtime_event_type_count=len(RuntimeEventType),
        canonical_payload_class_count=len(CANONICAL_PAYLOAD_TYPES),
        payload_schema_id_count=len(list_registered_payload_schema_ids(include_extensions=False)),
        single_version_family_count=single,
        multi_version_family_count=multi,
        supported_read_version_count=supported_read,
        current_write_version_count=current_write,
        retired_envelope_version_count=sum(
            1
            for r in inv.envelope_records
            if r.compatibility_disposition == EventCompatibilityDisposition.RETIRED_FAIL_CLOSED
        ),
        strict_production_write_policies=strict,
        extension_event_kind_policies=extension,
        unclassified_payload_classes=unclassified,
        duplicate_schema_ids=dup_schema,
        duplicate_family_membership=dup_membership,
        duplicate_write_ownership=len(write_owner_dup),
    )


def _duplicate_write_policy_violations() -> tuple[str, ...]:
    seen: dict[RuntimeEventType, str] = {}
    duplicates: list[str] = []
    for event_type, entry in iter_runtime_event_payload_policies():
        key = entry.schema_id or entry.write_mode.value
        prior = seen.get(event_type)
        if prior is not None and prior != key:
            duplicates.append(event_type.value)
        else:
            seen[event_type] = key
    return tuple(sorted(duplicates))


def inventory_payload_schema_ids(inventory: EventEvolutionInventory) -> frozenset[str]:
    return frozenset(v.schema_id for family in inventory.payload_families for v in family.versions)


def registered_canonical_payload_schema_ids() -> frozenset[str]:
    bootstrap_canonical_payload_registry()
    return frozenset(list_registered_payload_schema_ids(include_extensions=False))


def envelope_fail_closed_probes() -> tuple[str, ...]:
    """Runtime probe outcomes; returns error message fragments for fail-closed cases."""
    failures: list[str] = []
    for version in (None, "runtime_event.v1", "runtime_event.v99"):
        data = {
            "schema_version": version,
            "task_id": "task_test",
            "run_id": "run_test",
            "attempt_id": "attempt_test",
            "execution_id": "execution_test",
            "event_type": "task_created",
            "phase": "planning",
        }
        if version is not None:
            data["schema_version"] = version
        else:
            data.pop("schema_version", None)
        try:
            parse_runtime_event_payload(data)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            failures.append(f"unexpected success for version={version!r}")
    return tuple(failures)


def validate_inventory_closed_world(inventory: EventEvolutionInventory | None = None) -> tuple[str, ...]:
    inv = inventory or event_evolution_inventory()
    violations: list[str] = []

    inv_ids = inventory_payload_schema_ids(inv)
    reg_ids = registered_canonical_payload_schema_ids()
    if inv_ids != reg_ids:
        violations.append(f"inventory/registry schema_id mismatch: {sorted(inv_ids ^ reg_ids)}")

    classified: dict[type[RuntimeEventPayload], str] = {}
    for family in inv.payload_families:
        for version in family.versions:
            if version.payload_type in classified:
                violations.append(
                    f"duplicate family for {version.payload_type.__name__}: "
                    f"{classified[version.payload_type]} vs {family.family_id}"
                )
            classified[version.payload_type] = family.family_id

    for payload_type in CANONICAL_PAYLOAD_TYPES:
        if payload_type not in classified:
            violations.append(f"unclassified payload class: {payload_type.__name__}")
        if payload_type not in _PAYLOAD_CLASS_TO_FAMILY:
            violations.append(f"missing explicit family map: {payload_type.__name__}")

    for event_type in RuntimeEventType:
        policies = [et for et, _ in iter_runtime_event_payload_policies() if et == event_type]
        if len(policies) != 1:
            violations.append(f"write policy count for {event_type.value}: {len(policies)}")

    for event_type, entry in iter_runtime_event_payload_policies():
        if entry.write_mode == PayloadWriteMode.STRICT_SPINE_SCHEMA and entry.schema_id:
            if entry.schema_id not in reg_ids:
                violations.append(f"strict policy {event_type.value} references unregistered {entry.schema_id!r}")

    current_envelope = RUNTIME_SCHEMA_REGISTRY["runtime_event"]
    if not validate_schema_version("runtime_event", current_envelope):
        violations.append("runtime_event registry version fails validate_schema_version")

    for family in inv.payload_families:
        if len(family.versions) <= 1:
            continue
        active_writes = [
            v.schema_id
            for v in family.versions
            if v.compatibility_disposition == EventCompatibilityDisposition.CURRENT_WRITE_AND_READ
        ]
        if not family.allows_multiple_active_write_versions and len(active_writes) != 1:
            violations.append(
                f"family {family.family_id} requires exactly one active write version; got {active_writes}"
            )
        for version in family.versions:
            if version.compatibility_disposition == EventCompatibilityDisposition.UNKNOWN_FAIL_CLOSED:
                violations.append(f"undispositioned version {version.schema_id}")

    owner_concerns = [row.concern for row in inv.owner_matrix]
    if len(owner_concerns) != len(set(owner_concerns)):
        violations.append("duplicate owner matrix concerns")

    if any(_QUALIFICATION_PREFIX in row.semantic_owner_path for row in inv.owner_matrix):
        violations.append("qualification module listed as semantic version owner")

    violations.extend(validate_payload_version_inventory_consistency(inv))
    violations.extend(validate_production_persistence_call_site_classification(inv.production_persistence_call_sites))
    violations.extend(validate_canonical_production_guard_chains())
    if count_canonical_production_raw_store_bypasses(inv.production_persistence_call_sites) != 0:
        violations.append("inventory persistence call sites include canonical production bypasses")

    return tuple(violations)


def synthetic_inventory_missing_classification() -> tuple[str, ...]:
    """Probe A: orphan canonical class without family map."""

    class _SyntheticOrphan(RuntimeEventPayload):
        schema_id = "synthetic.orphan_probe.v1"
        schema_version = 1

    return validate_inventory_closed_world_for_extra_class(_SyntheticOrphan)


def validate_inventory_closed_world_for_extra_class(
    extra: type[RuntimeEventPayload],
) -> tuple[str, ...]:
    violations = list(validate_inventory_closed_world())
    if extra not in _PAYLOAD_CLASS_TO_FAMILY:
        violations.append(f"probe: unclassified {extra.__name__}")
    return tuple(violations)


def synthetic_inventory_phantom_schema_id() -> str:
    """Probe B: schema id present in inventory check but not registered."""
    return "phantom.compat_x_probe.nonexistent.v1"


def synthetic_inventory_phantom_schema_row_probe() -> tuple[str, ...]:
    """Probe B (adversarial): closed-world validator must reject phantom inventory row."""
    phantom_id = synthetic_inventory_phantom_schema_id()
    inv = event_evolution_inventory()
    phantom_record = EventPayloadVersionRecord(
        payload_type=ToolPayloadV1,
        schema_id=phantom_id,
        payload_schema_version=int(ToolPayloadV1.schema_version),
        compatibility_disposition=EventCompatibilityDisposition.SUPPORTED_READ_ONLY,
        migration_required=False,
        adaptation_required=False,
    )
    mutated_families: list[EventPayloadEvolutionFamily] = []
    injected = False
    for family in inv.payload_families:
        if family.family_id == "tool":
            mutated_families.append(
                EventPayloadEvolutionFamily(
                    family_id=family.family_id,
                    versions=family.versions + (phantom_record,),
                    semantic_owner_path=family.semantic_owner_path,
                    composition_owner_path=family.composition_owner_path,
                    compatibility_semantics_owner_path=family.compatibility_semantics_owner_path,
                    allows_multiple_active_write_versions=family.allows_multiple_active_write_versions,
                    notes=family.notes,
                )
            )
            injected = True
        else:
            mutated_families.append(family)
    if not injected:
        return ("probe: could not inject phantom row into tool family",)
    probe_inv = EventEvolutionInventory(
        envelope_records=inv.envelope_records,
        payload_families=tuple(mutated_families),
        extension_contract=inv.extension_contract,
        owner_matrix=inv.owner_matrix,
        production_persistence_guard_paths=inv.production_persistence_guard_paths,
        production_persistence_call_sites=inv.production_persistence_call_sites,
    )
    violations = validate_inventory_closed_world(probe_inv)
    if violations:
        return violations
    return ("probe: phantom inventory row did not fail closed",)


def synthetic_two_current_write_versions_for_family() -> tuple[str, ...]:
    """Probe C: family with two active writes when disallowed."""
    inv = event_evolution_inventory()
    mutated: list[EventPayloadEvolutionFamily] = []
    for family in inv.payload_families:
        if family.family_id == "llm_call":
            versions = tuple(
                EventPayloadVersionRecord(
                    payload_type=v.payload_type,
                    schema_id=v.schema_id,
                    payload_schema_version=v.payload_schema_version,
                    compatibility_disposition=EventCompatibilityDisposition.CURRENT_WRITE_AND_READ,
                    migration_required=False,
                    adaptation_required=False,
                )
                for v in family.versions
            )
            mutated.append(
                EventPayloadEvolutionFamily(
                    family_id=family.family_id,
                    versions=versions,
                    semantic_owner_path=family.semantic_owner_path,
                    composition_owner_path=family.composition_owner_path,
                    compatibility_semantics_owner_path=family.compatibility_semantics_owner_path,
                    allows_multiple_active_write_versions=False,
                )
            )
        else:
            mutated.append(family)
    probe_inv = EventEvolutionInventory(
        envelope_records=inv.envelope_records,
        payload_families=tuple(mutated),
        extension_contract=inv.extension_contract,
        owner_matrix=inv.owner_matrix,
        production_persistence_guard_paths=inv.production_persistence_guard_paths,
        production_persistence_call_sites=inv.production_persistence_call_sites,
    )
    return validate_inventory_closed_world(probe_inv)


def synthetic_undispositioned_historical_version() -> tuple[str, ...]:
    """Probe D."""
    inv = event_evolution_inventory()
    mutated: list[EventPayloadEvolutionFamily] = []
    for family in inv.payload_families:
        if family.family_id != "llm_call":
            mutated.append(family)
            continue
        versions: list[EventPayloadVersionRecord] = []
        for version in family.versions:
            disp = (
                EventCompatibilityDisposition.UNKNOWN_FAIL_CLOSED
                if version.schema_id == "llm_call.v1"
                else version.compatibility_disposition
            )
            versions.append(
                EventPayloadVersionRecord(
                    payload_type=version.payload_type,
                    schema_id=version.schema_id,
                    payload_schema_version=version.payload_schema_version,
                    compatibility_disposition=disp,
                    migration_required=version.migration_required,
                    adaptation_required=version.adaptation_required,
                )
            )
        mutated.append(
            EventPayloadEvolutionFamily(
                family_id=family.family_id,
                versions=tuple(versions),
                semantic_owner_path=family.semantic_owner_path,
                composition_owner_path=family.composition_owner_path,
                compatibility_semantics_owner_path=family.compatibility_semantics_owner_path,
                allows_multiple_active_write_versions=family.allows_multiple_active_write_versions,
            )
        )
    probe_inv = EventEvolutionInventory(
        envelope_records=inv.envelope_records,
        payload_families=tuple(mutated),
        extension_contract=inv.extension_contract,
        owner_matrix=inv.owner_matrix,
        production_persistence_guard_paths=inv.production_persistence_guard_paths,
        production_persistence_call_sites=inv.production_persistence_call_sites,
    )
    return validate_inventory_closed_world(probe_inv)


def synthetic_second_evolution_authority_owner() -> tuple[str, ...]:
    """Probe J: duplicate policy owner concern."""
    inv = event_evolution_inventory()
    extra_row = EventEvolutionOwnerRow(
        concern="event_evolution_policy",
        semantic_owner_path="intergrax/runtime/events/payload_registry.py",
        composition_owner_path="intergrax/runtime/events/payload_registry.py",
        evidence_paths=(_PAYLOAD_REGISTRY,),
    )
    probe_inv = EventEvolutionInventory(
        envelope_records=inv.envelope_records,
        payload_families=inv.payload_families,
        extension_contract=inv.extension_contract,
        owner_matrix=inv.owner_matrix + (extra_row,),
        production_persistence_guard_paths=inv.production_persistence_guard_paths,
        production_persistence_call_sites=inv.production_persistence_call_sites,
    )
    return validate_inventory_closed_world(probe_inv)


def collect_event_evolution_r3_p0_gate_violations(
    *,
    extra_persistence_call_sites: tuple[RuntimeEventPersistenceCallSite, ...] = (),
) -> tuple[str, ...]:
    """All violations for the combined R3-P0 gate (optionally includes synthetic call sites)."""
    base_sites = discover_runtime_event_persistence_call_sites()
    sites = base_sites + extra_persistence_call_sites
    violations: list[str] = list(validate_inventory_closed_world())
    violations.extend(_payload_envelope_version_fail_closed_probe())
    violations.extend(validate_strict_write_policies_against_registry(iter_runtime_event_payload_policies()))
    violations.extend(validate_production_persistence_call_site_classification(sites))
    violations.extend(validate_canonical_production_guard_chains())
    if count_unclassified_persistence_call_sites(sites) != 0:
        violations.append("persistence discovery includes UNCLASSIFIED append call sites")
    if count_non_event_append_without_evidence(sites) != 0:
        violations.append("NON_EVENT_APPEND call sites without positive evidence")
    bypass_count = count_canonical_production_raw_store_bypasses(sites)
    if bypass_count != 0:
        violations.append(f"canonical production persistence bypass count {bypass_count} (expected 0)")
    return tuple(violations)


def event_evolution_r3_p0_gate_passes() -> tuple[bool, tuple[str, ...]]:
    """
    Combined R3-P0 closure verdict (qualification authority; not runtime enforcement).

    Requires inventory closed-world, payload version contract, envelope fail-closed probes,
    mechanical persistence discovery classification, and canonical composition guard evidence.
    """
    violations = collect_event_evolution_r3_p0_gate_violations()
    return (not violations, violations)


def synthetic_unknown_persistence_append_call_site() -> RuntimeEventPersistenceCallSite:
    """Probe K/L helper: synthetic previously unknown ``.append(..., tenant_id=...)`` call."""
    classification = classify_runtime_event_append_call_site(
        "intergrax/runtime/new_component.py",
        "SomeNewComponent.persist",
        "raw_runtime_store.append",
    )
    return RuntimeEventPersistenceCallSite(
        module_path="intergrax/runtime/new_component.py",
        lineno=1,
        qualname="SomeNewComponent.persist",
        callee_expr="raw_runtime_store.append",
        disposition=classification.disposition,
        classification_evidence=classification.evidence,
    )


def synthetic_misleading_audit_append_call_site() -> RuntimeEventPersistenceCallSite:
    """Probe L: benign-looking ``audit.append`` without contract evidence."""
    classification = classify_runtime_event_append_call_site(
        "intergrax/runtime/unknown_audit_module.py",
        "record",
        "audit.append",
    )
    return RuntimeEventPersistenceCallSite(
        module_path="intergrax/runtime/unknown_audit_module.py",
        lineno=1,
        qualname="record",
        callee_expr="audit.append",
        disposition=classification.disposition,
        classification_evidence=classification.evidence,
    )


def synthetic_evidence_less_non_event_append_call_site() -> RuntimeEventPersistenceCallSite:
    """Probe N: incorrectly labeled harmless append without positive evidence."""
    return RuntimeEventPersistenceCallSite(
        module_path="intergrax/runtime/synthetic_probe_n.py",
        lineno=1,
        qualname="probe",
        callee_expr="mystery_store.append",
        disposition=RuntimeEventWritePathDisposition.NON_EVENT_APPEND,
        classification_evidence=RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED,
    )


def synthetic_probe_k_unknown_persistence_append_gate() -> tuple[str, ...]:
    site = synthetic_unknown_persistence_append_call_site()
    if site.disposition != RuntimeEventWritePathDisposition.UNCLASSIFIED:
        return (f"probe K: expected UNCLASSIFIED, got {site.disposition.value}",)
    passed, _ = event_evolution_r3_p0_gate_passes()
    if not passed:
        return ("probe K: production gate must pass before unknown injection",)
    gate_with_unknown = collect_event_evolution_r3_p0_gate_violations(extra_persistence_call_sites=(site,))
    if not gate_with_unknown:
        return ("probe K: combined gate unexpectedly passed with unknown call site",)
    return ()


def synthetic_probe_l_misleading_audit_append() -> tuple[str, ...]:
    site = synthetic_misleading_audit_append_call_site()
    if site.disposition != RuntimeEventWritePathDisposition.UNCLASSIFIED:
        return (f"probe L: expected UNCLASSIFIED, got {site.disposition.value}",)
    gate_violations = collect_event_evolution_r3_p0_gate_violations(extra_persistence_call_sites=(site,))
    if not gate_violations:
        return ("probe L: gate unexpectedly passed with misleading audit append",)
    return ()


def synthetic_probe_m_raw_production_composition_bypass_gate() -> tuple[str, ...]:
    composition_violations = synthetic_canonical_composition_raw_store_bypass_probe()
    if not composition_violations:
        return ("probe M: raw store without validator not detected",)
    if not any("ValidatingRuntimeEventPersistence" in v for v in composition_violations):
        return ("probe M: composition probe missing validator bypass message",)
    raw_site = RuntimeEventPersistenceCallSite(
        module_path="intergrax/runtime/persistence/integration_profile_wiring.py",
        lineno=1,
        qualname="synthetic_probe_m_composition",
        callee_expr="InMemoryRuntimeEventStore().append",
        disposition=RuntimeEventWritePathDisposition.UNCLASSIFIED,
        classification_evidence=RuntimeEventCallSiteClassificationEvidence.UNSPECIFIED,
    )
    if count_canonical_production_raw_store_bypasses((raw_site,)) != 1:
        return ("probe M: bypass counter did not detect synthetic production raw append",)
    gate_violations = collect_event_evolution_r3_p0_gate_violations(extra_persistence_call_sites=(raw_site,))
    if not gate_violations:
        return ("probe M: combined gate passed with synthetic raw production append",)
    return ()


def synthetic_probe_n_evidence_less_non_event_append() -> tuple[str, ...]:
    site = synthetic_evidence_less_non_event_append_call_site()
    violations = validate_non_event_append_positive_evidence((site,))
    if not violations:
        return ("probe N: evidence-less NON_EVENT_APPEND was accepted",)
    return ()


def probe_extension_duplicate_canonical_override() -> str | None:
    """Probe H: returns error message when override fails closed."""
    from intergrax.runtime.events.payload_registry import RuntimeEventPayloadError

    class _AltDecision(RuntimeEventPayload):
        schema_id = DecisionPayloadV1.schema_id
        schema_version = 1

    try:
        register_payload_schema(_AltDecision, extension=True)
    except RuntimeEventPayloadError as exc:
        return str(exc)
    return None


__all__ = [
    "EventCompatibilityDisposition",
    "EventEnvelopeEvolutionRecord",
    "EventEvolutionInventory",
    "EventEvolutionOwnerRow",
    "EventEvolutionR3P0Metrics",
    "EventEvolutionSurfaceKind",
    "EventPayloadEvolutionFamily",
    "EventPayloadVersionRecord",
    "EventVersionDisposition",
    "ExtensionEventEvolutionContract",
    "RuntimeEventAppendCallSiteClassification",
    "RuntimeEventCallSiteClassificationEvidence",
    "RuntimeEventPersistenceCallSite",
    "RuntimeEventWritePathDisposition",
    "classify_runtime_event_append_call_site",
    "collect_event_evolution_r3_p0_gate_violations",
    "compute_event_evolution_r3_p0_metrics",
    "count_canonical_production_raw_store_bypasses",
    "count_non_event_append_without_evidence",
    "count_production_python_files_scanned",
    "count_unclassified_persistence_call_sites",
    "detect_raw_runtime_event_persistence_without_validator",
    "discover_runtime_event_persistence_call_sites",
    "envelope_fail_closed_probes",
    "event_evolution_inventory",
    "event_evolution_r3_p0_gate_passes",
    "inventory_payload_schema_ids",
    "probe_extension_duplicate_canonical_override",
    "registered_canonical_payload_schema_ids",
    "summarize_persistence_call_site_dispositions",
    "synthetic_canonical_composition_raw_store_bypass_probe",
    "synthetic_evidence_less_non_event_append_call_site",
    "synthetic_inventory_missing_classification",
    "synthetic_inventory_phantom_schema_id",
    "synthetic_inventory_phantom_schema_row_probe",
    "synthetic_misleading_audit_append_call_site",
    "synthetic_probe_k_unknown_persistence_append_gate",
    "synthetic_probe_l_misleading_audit_append",
    "synthetic_probe_m_raw_production_composition_bypass_gate",
    "synthetic_probe_n_evidence_less_non_event_append",
    "synthetic_second_evolution_authority_owner",
    "synthetic_two_current_write_versions_for_family",
    "synthetic_undispositioned_historical_version",
    "synthetic_unknown_persistence_append_call_site",
    "synthetic_write_policy_unregistered_schema_probe",
    "validate_call_site_disposition_evidence_alignment",
    "validate_canonical_production_guard_chains",
    "validate_inventory_closed_world",
    "validate_inventory_closed_world_for_extra_class",
    "validate_non_event_append_positive_evidence",
    "validate_production_persistence_call_site_classification",
    "validate_strict_write_policies_against_registry",
]
