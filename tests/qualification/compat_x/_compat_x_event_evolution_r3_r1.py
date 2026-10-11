# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R3-R1 mechanical gates — envelope version authority (qualification only)."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    parse_runtime_event_payload,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.runtime_event import RuntimeEvent as RuntimeRuntimeEvent
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.schema.registry import (
    PREVIEW_RUNTIME_SCHEMA_VERSIONS,
    RUNTIME_SCHEMA_REGISTRY,
    validate_schema_version,
)

from testing_support.runtime_events import with_preferred_canonical_payload

from tests.qualification.compat_x._compat_x_event_evolution_inventory import (
    _enclosing_qualname,
    _iter_production_python_sources,
    _repo_relative,
    collect_event_evolution_r3_p0_gate_violations,
    count_canonical_production_raw_store_bypasses,
    count_unclassified_persistence_call_sites,
    discover_runtime_event_persistence_call_sites,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_ROOT = _REPO_ROOT / "intergrax"
_OBSERVABILITY_ROOT = _PRODUCTION_ROOT / "runtime" / "observability"
_CANONICAL_OWNER = _PRODUCTION_ROOT / "contracts" / "runtime_event.py"

_V2_LITERAL = "runtime_event.v2"
_V1_LITERAL = "runtime_event.v1"

_RUNTIME_EVENT_STORE_MARKERS: Final = (
    "runtime/events/stores/sqlite_runtime_event_store.py",
    "runtime/events/stores/document_backed_runtime_event_store.py",
    "runtime/events/stores/memory_runtime_event_store.py",
)


class RuntimeEventReaderDisposition(str, Enum):
    CANONICAL_PARSER = "canonical_parser"
    NON_PERSISTED_CONSTRUCTION = "non_persisted_construction"
    INTERNAL_ALREADY_TYPED_FLOW = "internal_already_typed_flow"
    TEST_LAB_ONLY = "test_lab_only"
    UNCLASSIFIED = "unclassified"


class RuntimeEventReaderClassificationEvidence(str, Enum):
    UNSPECIFIED = "unspecified"
    RUNTIME_EVENT_STORE_SQLITE_WIRE_DECODE = "runtime_event_store_sqlite_wire_decode"
    RUNTIME_EVENT_STORE_DOCUMENT_WIRE_DECODE = "runtime_event_store_document_wire_decode"
    CANONICAL_PARSER_POST_VERSION_GATE = "canonical_parser_post_version_gate"
    PLATFORM_HISTORY_BUFFER_PROBE = "platform_history_buffer_probe"
    TYPED_EVENT_PRODUCER = "typed_event_producer"
    PERSISTENCE_CONFORMANCE_LAB = "persistence_conformance_lab"
    LAB_OR_POC_APPLICATION_SCOPE = "lab_or_poc_application_scope"


@dataclass(frozen=True, slots=True)
class RuntimeEventReaderSite:
    module_path: str
    line: int
    qualname: str
    operation: str
    disposition: RuntimeEventReaderDisposition
    classification_evidence: RuntimeEventReaderClassificationEvidence


def _iter_production_python_files() -> tuple[Path, ...]:
    return tuple(Path(p) for p in _iter_production_python_sources())


def production_runtime_event_v2_literal_sites() -> tuple[tuple[str, int], ...]:
    """Every production file/line containing the exact v2 string literal."""
    sites: list[tuple[str, int]] = []
    for path in _iter_production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _V2_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def production_runtime_event_v1_literal_sites() -> tuple[tuple[str, int], ...]:
    sites: list[tuple[str, int]] = []
    for path in _iter_production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _V1_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def count_runtime_event_v2_semantic_authority_literals() -> int:
    """Lines that define the canonical current envelope version string in the contract owner."""
    owner_text = _CANONICAL_OWNER.read_text(encoding="utf-8")
    tree = ast.parse(owner_text)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RUNTIME_EVENT_SCHEMA_VERSION":
                if (
                    isinstance(node.value, ast.Constant)
                    and node.value.value == _V2_LITERAL
                ):
                    count += 1
    return count


def observability_runtime_event_v2_literal_sites() -> tuple[tuple[str, int], ...]:
    sites: list[tuple[str, int]] = []
    if not _OBSERVABILITY_ROOT.is_dir():
        return ()
    for path in sorted(_OBSERVABILITY_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _V2_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def validate_runtime_event_registry_parity(
    registry: Mapping[str, str],
    *,
    canonical_version: str = RUNTIME_EVENT_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Injectable registry parity check (real and synthetic drift probes share this validator)."""
    violations: list[str] = []
    mapped = registry.get("runtime_event")
    if mapped != canonical_version:
        violations.append(
            "RUNTIME_SCHEMA_REGISTRY['runtime_event'] != RUNTIME_EVENT_SCHEMA_VERSION "
            f"(registry={mapped!r}, canonical={canonical_version!r})",
        )
    return tuple(violations)


def registry_runtime_event_version_parity_violations() -> tuple[str, ...]:
    return validate_runtime_event_registry_parity(RUNTIME_SCHEMA_REGISTRY)


def synthetic_registry_drift_violations() -> tuple[str, ...]:
    """Probe B: synthetic registry drift must surface parity violations (non-empty)."""
    drifted = {**RUNTIME_SCHEMA_REGISTRY, "runtime_event": "runtime_event.v99"}
    return validate_runtime_event_registry_parity(drifted)


def synthetic_second_v2_literal_owner(
    extra_site: tuple[str, int] | None = None,
) -> tuple[str, ...]:
    """Probe A: second current-version literal outside canonical owner."""
    sites = list(production_runtime_event_v2_literal_sites())
    if extra_site is not None:
        sites.append(extra_site)
    if len(sites) != 1:
        return (f"expected 1 production v2 literal site, got {len(sites)}: {sites}",)
    rel, _ = sites[0]
    if rel != "intergrax/contracts/runtime_event.py":
        return (f"canonical v2 literal not in contract owner: {rel}",)
    return ()


def _runtime_event_call_kind(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        if node.func.id == "parse_runtime_event_payload":
            return "parse_runtime_event_payload"
        if node.func.id == "RuntimeEvent":
            return "runtime_event_constructor"
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in {"model_validate", "model_validate_json"}:
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "RuntimeEvent":
                return (
                    "runtime_event_model_validate_json"
                    if node.func.attr == "model_validate_json"
                    else "runtime_event_model_validate"
                )
    return None


def _operation_label(site_kind: str) -> str:
    if site_kind == "parse_runtime_event_payload":
        return "parse_runtime_event_payload"
    if site_kind == "runtime_event_model_validate":
        return "RuntimeEvent.model_validate"
    if site_kind == "runtime_event_model_validate_json":
        return "RuntimeEvent.model_validate_json"
    if site_kind == "runtime_event_constructor":
        return "RuntimeEvent(...)"
    return site_kind


def classify_runtime_event_reader_site(
    module_path: str,
    qualname: str,
    operation: str,
    site_kind: str,
) -> tuple[RuntimeEventReaderDisposition, RuntimeEventReaderClassificationEvidence]:
    """Classify a discovered or synthetic reader site; unknown → ``UNCLASSIFIED`` (fail closed)."""
    path = module_path.replace("\\", "/")
    unspecified = RuntimeEventReaderClassificationEvidence.UNSPECIFIED

    if site_kind == "unknown_wire_decode":
        return RuntimeEventReaderDisposition.UNCLASSIFIED, unspecified

    if site_kind == "parse_runtime_event_payload":
        if path.endswith("runtime/events/stores/sqlite_runtime_event_store.py"):
            return (
                RuntimeEventReaderDisposition.CANONICAL_PARSER,
                RuntimeEventReaderClassificationEvidence.RUNTIME_EVENT_STORE_SQLITE_WIRE_DECODE,
            )
        if path.endswith("runtime/events/stores/document_backed_runtime_event_store.py"):
            return (
                RuntimeEventReaderDisposition.CANONICAL_PARSER,
                RuntimeEventReaderClassificationEvidence.RUNTIME_EVENT_STORE_DOCUMENT_WIRE_DECODE,
            )
        return RuntimeEventReaderDisposition.UNCLASSIFIED, unspecified

    if site_kind in {"runtime_event_model_validate", "runtime_event_model_validate_json"}:
        if path.endswith("contracts/runtime_event.py"):
            return (
                RuntimeEventReaderDisposition.INTERNAL_ALREADY_TYPED_FLOW,
                RuntimeEventReaderClassificationEvidence.CANONICAL_PARSER_POST_VERSION_GATE,
            )
        if path.endswith("runtime/events/runtime_event_history_validation.py"):
            return (
                RuntimeEventReaderDisposition.INTERNAL_ALREADY_TYPED_FLOW,
                RuntimeEventReaderClassificationEvidence.PLATFORM_HISTORY_BUFFER_PROBE,
            )
        if "runtime/events/stores/" in path:
            return RuntimeEventReaderDisposition.UNCLASSIFIED, unspecified
        return RuntimeEventReaderDisposition.UNCLASSIFIED, unspecified

    if site_kind == "runtime_event_constructor":
        if path.endswith("runtime/observability/persistence_conformance.py"):
            return (
                RuntimeEventReaderDisposition.TEST_LAB_ONLY,
                RuntimeEventReaderClassificationEvidence.PERSISTENCE_CONFORMANCE_LAB,
            )
        if "lab_application/" in path or "poc_template_application/" in path:
            return (
                RuntimeEventReaderDisposition.TEST_LAB_ONLY,
                RuntimeEventReaderClassificationEvidence.LAB_OR_POC_APPLICATION_SCOPE,
            )
        return (
            RuntimeEventReaderDisposition.NON_PERSISTED_CONSTRUCTION,
            RuntimeEventReaderClassificationEvidence.TYPED_EVENT_PRODUCER,
        )

    return RuntimeEventReaderDisposition.UNCLASSIFIED, unspecified


@lru_cache(maxsize=1)
def discover_runtime_event_reader_sites() -> tuple[RuntimeEventReaderSite, ...]:
    """AST discovery of production RuntimeEvent deserialization / construction surfaces."""
    discovered: list[RuntimeEventReaderSite] = []
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
            site_kind = _runtime_event_call_kind(node)
            if site_kind is None:
                continue
            qualname = _enclosing_qualname(tree, node.lineno)
            operation = _operation_label(site_kind)
            disposition, evidence = classify_runtime_event_reader_site(
                module_path,
                qualname,
                operation,
                site_kind,
            )
            discovered.append(
                RuntimeEventReaderSite(
                    module_path=module_path,
                    line=node.lineno,
                    qualname=qualname,
                    operation=operation,
                    disposition=disposition,
                    classification_evidence=evidence,
                )
            )
    return tuple(sorted(discovered, key=lambda s: (s.module_path, s.line, s.operation)))


def discover_persisted_runtime_event_read_paths() -> tuple[str, ...]:
    """Modules with canonical parser call sites (mechanical; not the sole reader proof)."""
    modules = sorted(
        {
            site.module_path
            for site in discover_runtime_event_reader_sites()
            if site.disposition == RuntimeEventReaderDisposition.CANONICAL_PARSER
        }
    )
    return tuple(modules)


def validate_runtime_event_reader_site_classification(
    sites: tuple[RuntimeEventReaderSite, ...] | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    reader_sites = sites if sites is not None else discover_runtime_event_reader_sites()
    for site in reader_sites:
        if site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED:
            violations.append(
                f"unclassified runtime event reader {site.module_path}:{site.line} "
                f"{site.qualname} {site.operation}",
            )
            continue
        if site.classification_evidence == RuntimeEventReaderClassificationEvidence.UNSPECIFIED:
            violations.append(
                f"classified reader missing evidence {site.module_path}:{site.line} "
                f"{site.disposition.value}",
            )
    return tuple(violations)


def count_unclassified_runtime_event_reader_sites(
    sites: tuple[RuntimeEventReaderSite, ...] | None = None,
) -> int:
    reader_sites = sites if sites is not None else discover_runtime_event_reader_sites()
    return sum(
        1 for site in reader_sites if site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    )


def count_persisted_reader_canonical_version_bypasses(
    sites: tuple[RuntimeEventReaderSite, ...] | None = None,
) -> int:
    """Direct ``RuntimeEvent.model_validate*`` on store wire paths without canonical parser delegation."""
    reader_sites = sites if sites is not None else discover_runtime_event_reader_sites()
    bypass = 0
    for site in reader_sites:
        if site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED:
            bypass += 1
            continue
        if site.operation not in {
            "RuntimeEvent.model_validate",
            "RuntimeEvent.model_validate_json",
        }:
            continue
        if site.disposition == RuntimeEventReaderDisposition.INTERNAL_ALREADY_TYPED_FLOW:
            continue
        if "runtime/events/stores/" in site.module_path.replace("\\", "/"):
            bypass += 1
    return bypass


def summarize_runtime_event_reader_dispositions(
    sites: tuple[RuntimeEventReaderSite, ...] | None = None,
) -> dict[RuntimeEventReaderDisposition, int]:
    reader_sites = sites if sites is not None else discover_runtime_event_reader_sites()
    counts = {disp: 0 for disp in RuntimeEventReaderDisposition}
    for site in reader_sites:
        counts[site.disposition] += 1
    return counts


def synthetic_persisted_runtime_event_model_validate_bypass_site() -> RuntimeEventReaderSite:
    """Probe I: persisted store wire decode via direct model_validate (must fail closed)."""
    disposition, evidence = classify_runtime_event_reader_site(
        "intergrax/runtime/events/stores/synthetic_bypass_probe.py",
        "SyntheticBypassStore._load_row",
        "RuntimeEvent.model_validate",
        "runtime_event_model_validate",
    )
    return RuntimeEventReaderSite(
        module_path="intergrax/runtime/events/stores/synthetic_bypass_probe.py",
        line=2,
        qualname="SyntheticBypassStore._load_row",
        operation="RuntimeEvent.model_validate",
        disposition=disposition,
        classification_evidence=evidence,
    )


def synthetic_unknown_persisted_reader_site() -> RuntimeEventReaderSite:
    """Probe J: unknown persisted reader path without positive classification evidence."""
    disposition, evidence = classify_runtime_event_reader_site(
        "intergrax/runtime/events/stores/new_backend.py",
        "NewStore.read",
        "deserialize_event",
        "unknown_wire_decode",
    )
    return RuntimeEventReaderSite(
        module_path="intergrax/runtime/events/stores/new_backend.py",
        line=1,
        qualname="NewStore.read",
        operation="deserialize_event",
        disposition=disposition,
        classification_evidence=evidence,
    )


def minimal_runtime_event_payload_dict(
    schema_version: object,
) -> dict[str, object]:
    data: dict[str, object] = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "event_type": RuntimeEventType.TASK_CREATED.value,
        "phase": ExecutionPhase.INTAKE.value,
    }
    if schema_version is not None:
        data["schema_version"] = schema_version
    return data


def runtime_event_default_schema_version() -> str:
    return RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    ).schema_version


def _envelope_version_read_write_violations() -> tuple[str, ...]:
    violations: list[str] = []
    try:
        parse_runtime_event_payload(
            minimal_runtime_event_payload_dict(RUNTIME_EVENT_SCHEMA_VERSION),
        )
    except ValueError as exc:
        violations.append(f"current envelope read rejected: {exc}")

    for version, label in (("runtime_event.v1", "v1"), ("runtime_event.v99", "future")):
        try:
            parse_runtime_event_payload(minimal_runtime_event_payload_dict(version))
        except ValueError:
            pass
        else:
            violations.append(f"{label} envelope read accepted")

    try:
        parse_runtime_event_payload(minimal_runtime_event_payload_dict(None))
    except ValueError:
        pass
    else:
        violations.append("missing schema_version read accepted")

    current_event = with_preferred_canonical_payload(
        RuntimeRuntimeEvent(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.TASK_CREATED,
            phase=ExecutionPhase.INTAKE,
        ),
    )
    try:
        assert_runtime_event_schema(current_event)
    except RuntimeEventSchemaError as exc:
        violations.append(f"current envelope write rejected: {exc}")

    for version, label in (("runtime_event.v1", "v1"), ("runtime_event.v99", "future")):
        mutated = current_event.model_copy(update={"schema_version": version})
        try:
            assert_runtime_event_schema(mutated)
        except RuntimeEventSchemaError:
            pass
        else:
            violations.append(f"{label} envelope write accepted")

    if validate_schema_version("runtime_event", "runtime_event.v99"):
        violations.append("preview registry widened runtime_event v99 acceptance")

    return tuple(violations)


def collect_event_evolution_r3_r1_violations(
    *,
    registry: Mapping[str, str] | None = None,
    extra_reader_sites: tuple[RuntimeEventReaderSite, ...] = (),
) -> tuple[str, ...]:
    """Combined R3-R1 qualification violations (optional synthetic registry / reader injection)."""
    violations: list[str] = []
    reg = registry if registry is not None else RUNTIME_SCHEMA_REGISTRY

    violations.extend(validate_runtime_event_registry_parity(reg))
    violations.extend(synthetic_second_v2_literal_owner())
    if count_runtime_event_v2_semantic_authority_literals() != 1:
        violations.append("semantic current-version authority count != 1")
    if runtime_event_default_schema_version() != RUNTIME_EVENT_SCHEMA_VERSION:
        violations.append("RuntimeEvent default schema_version != RUNTIME_EVENT_SCHEMA_VERSION")
    if observability_runtime_event_v2_literal_sites():
        violations.append("observability tree contains runtime_event.v2 literal")
    violations.extend(_envelope_version_read_write_violations())

    reader_sites = discover_runtime_event_reader_sites() + extra_reader_sites
    violations.extend(validate_runtime_event_reader_site_classification(reader_sites))
    if count_unclassified_runtime_event_reader_sites(reader_sites) != 0:
        violations.append("runtime event reader discovery includes UNCLASSIFIED sites")
    bypass = count_persisted_reader_canonical_version_bypasses(reader_sites)
    if bypass != 0:
        violations.append(f"persisted reader canonical-version bypass count {bypass} (expected 0)")

    violations.extend(collect_event_evolution_r3_p0_gate_violations())
    persistence_sites = discover_runtime_event_persistence_call_sites()
    if count_unclassified_persistence_call_sites(persistence_sites) != 0:
        violations.append("persistence discovery includes UNCLASSIFIED append call sites")
    if count_canonical_production_raw_store_bypasses(persistence_sites) != 0:
        violations.append("canonical production persistence bypass count != 0")

    return tuple(violations)


def event_evolution_r3_r1_gate_passes(
    *,
    registry: Mapping[str, str] | None = None,
    extra_reader_sites: tuple[RuntimeEventReaderSite, ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    violations = collect_event_evolution_r3_r1_violations(
        registry=registry,
        extra_reader_sites=extra_reader_sites,
    )
    return (not violations, violations)


@dataclass(frozen=True, slots=True)
class EventEvolutionR3R1Metrics:
    production_runtime_event_v2_literals: int
    semantic_current_version_authorities: int
    observability_v2_literals: int
    persisted_reader_candidates: int
    reader_canonical_parser: int
    reader_non_persisted_construction: int
    reader_internal_already_typed_flow: int
    reader_test_lab_only: int
    reader_unclassified: int
    reader_bypass_count: int
    writer_candidate_count: int
    writer_unclassified: int
    writer_bypass_count: int
    registry_parity_violations: int


def compute_event_evolution_r3_r1_metrics() -> EventEvolutionR3R1Metrics:
    reader_sites = discover_runtime_event_reader_sites()
    reader_counts = summarize_runtime_event_reader_dispositions(reader_sites)
    persistence_sites = discover_runtime_event_persistence_call_sites()
    return EventEvolutionR3R1Metrics(
        production_runtime_event_v2_literals=len(production_runtime_event_v2_literal_sites()),
        semantic_current_version_authorities=count_runtime_event_v2_semantic_authority_literals(),
        observability_v2_literals=len(observability_runtime_event_v2_literal_sites()),
        persisted_reader_candidates=len(reader_sites),
        reader_canonical_parser=reader_counts[RuntimeEventReaderDisposition.CANONICAL_PARSER],
        reader_non_persisted_construction=reader_counts[
            RuntimeEventReaderDisposition.NON_PERSISTED_CONSTRUCTION
        ],
        reader_internal_already_typed_flow=reader_counts[
            RuntimeEventReaderDisposition.INTERNAL_ALREADY_TYPED_FLOW
        ],
        reader_test_lab_only=reader_counts[RuntimeEventReaderDisposition.TEST_LAB_ONLY],
        reader_unclassified=reader_counts[RuntimeEventReaderDisposition.UNCLASSIFIED],
        reader_bypass_count=count_persisted_reader_canonical_version_bypasses(reader_sites),
        writer_candidate_count=len(persistence_sites),
        writer_unclassified=count_unclassified_persistence_call_sites(persistence_sites),
        writer_bypass_count=count_canonical_production_raw_store_bypasses(persistence_sites),
        registry_parity_violations=len(registry_runtime_event_version_parity_violations()),
    )


__all__ = [
    "EventEvolutionR3R1Metrics",
    "RUNTIME_EVENT_SCHEMA_VERSION",
    "RuntimeEventReaderClassificationEvidence",
    "RuntimeEventReaderDisposition",
    "RuntimeEventReaderSite",
    "collect_event_evolution_r3_r1_violations",
    "compute_event_evolution_r3_r1_metrics",
    "count_persisted_reader_canonical_version_bypasses",
    "count_runtime_event_v2_semantic_authority_literals",
    "count_unclassified_runtime_event_reader_sites",
    "discover_persisted_runtime_event_read_paths",
    "discover_runtime_event_reader_sites",
    "event_evolution_r3_r1_gate_passes",
    "minimal_runtime_event_payload_dict",
    "observability_runtime_event_v2_literal_sites",
    "parse_runtime_event_payload",
    "production_runtime_event_v1_literal_sites",
    "production_runtime_event_v2_literal_sites",
    "registry_runtime_event_version_parity_violations",
    "runtime_event_default_schema_version",
    "summarize_runtime_event_reader_dispositions",
    "synthetic_persisted_runtime_event_model_validate_bypass_site",
    "synthetic_registry_drift_violations",
    "synthetic_second_v2_literal_owner",
    "synthetic_unknown_persisted_reader_site",
    "validate_runtime_event_registry_parity",
    "validate_runtime_event_reader_site_classification",
    "validate_schema_version",
    "PREVIEW_RUNTIME_SCHEMA_VERSIONS",
]
