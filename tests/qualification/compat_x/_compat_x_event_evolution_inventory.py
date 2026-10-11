# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R3-P0 closed-world RuntimeEvent evolution inventory (qualification evidence only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
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
from intergrax.runtime.events.payloads.platform_families import DiagnosticSubsystemFailurePayloadV1
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
    iter_runtime_event_payload_policies,
)
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY, validate_schema_version

_QUALIFICATION_PREFIX: Final = "tests/qualification/compat_x/"
_COMPAT_X_ARCH_POLICY: Final = (
    "docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md"
)
_RUNTIME_EVENT_CONTRACT: Final = "intergrax/contracts/runtime_event.py"
_RUNTIME_SCHEMA_REGISTRY: Final = "intergrax/runtime/schema/registry.py"
_PAYLOAD_REGISTRY: Final = "intergrax/runtime/events/payload_registry.py"
_PAYLOAD_MODELS: Final = "intergrax/runtime/events/payloads/"
_WRITE_POLICY: Final = "intergrax/runtime/events/runtime_event_payload_policy.py"
_SCHEMA_GUARD: Final = "intergrax/runtime/events/schema_guard.py"
_EVENT_KIND_REGISTRY: Final = "intergrax/runtime/events/event_kind_registry.py"
_PERSISTENCE_CONTRACT: Final = "intergrax/runtime/events/persistence_contract.py"
_VALIDATING_STORE: Final = "intergrax/runtime/events/stores/validating_runtime_event_store.py"


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
    IntegrationConfigurationProvenanceRequirementPayloadV1: (
        "integration_configuration_provenance_requirement"
    ),
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


@lru_cache(maxsize=1)
def event_evolution_inventory() -> EventEvolutionInventory:
    """Authoritative R3-P0 closed-world inventory (qualification evidence; not runtime authority)."""
    bootstrap_canonical_payload_registry()
    return EventEvolutionInventory(
        envelope_records=_envelope_records(),
        payload_families=_build_payload_families(),
        extension_contract=_extension_contract(),
        owner_matrix=_owner_matrix(),
        production_persistence_guard_paths=(
            _VALIDATING_STORE,
            "intergrax/runtime/events/store.py",
            "intergrax/runtime/persistence/integration_profile_wiring.py",
        ),
    )


def compute_event_evolution_r3_p0_metrics(
    inventory: EventEvolutionInventory | None = None,
) -> EventEvolutionR3P0Metrics:
    inv = inventory or event_evolution_inventory()
    preferred = set(EVENT_TYPE_PREFERRED_SCHEMA.values())
    all_versions: list[EventPayloadVersionRecord] = [
        v for family in inv.payload_families for v in family.versions
    ]
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
        1
        for v in all_versions
        if v.compatibility_disposition == EventCompatibilityDisposition.CURRENT_WRITE_AND_READ
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
                violations.append(
                    f"strict policy {event_type.value} references unregistered {entry.schema_id!r}"
                )

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
    )
    return validate_inventory_closed_world(probe_inv)


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
    "compute_event_evolution_r3_p0_metrics",
    "envelope_fail_closed_probes",
    "event_evolution_inventory",
    "inventory_payload_schema_ids",
    "probe_extension_duplicate_canonical_override",
    "registered_canonical_payload_schema_ids",
    "synthetic_inventory_missing_classification",
    "synthetic_inventory_phantom_schema_id",
    "synthetic_second_evolution_authority_owner",
    "synthetic_two_current_write_versions_for_family",
    "synthetic_undispositioned_historical_version",
    "validate_inventory_closed_world",
    "validate_inventory_closed_world_for_extra_class",
]
