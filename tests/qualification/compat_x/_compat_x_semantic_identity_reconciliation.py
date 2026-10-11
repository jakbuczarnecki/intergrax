# © Artur Czarnecki. All rights reserved.

"""Evidence-only exact historical P0 class-field semantic identity projection (COMPAT-X-R1-R5)."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tests.qualification.compat_x._compat_x_ast_signals import parse_module
from tests.qualification.compat_x._compat_x_closed_world import ClosedWorldReport, build_closed_world_report
from tests.qualification.compat_x._compat_x_types import DiscoveryCandidate

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CLASS_FIELD_PREFIX = "class.field:"
_LEGACY_LITERAL_PREFIX = "schema.literal:"

ACCEPTED_P0_RAW_CANDIDATES = 7293
ACCEPTED_P0_SEMANTIC_SURFACES = 1901
ACCEPTED_P0_EXCLUSIONS = 5294

_UNRESOLVED_VERSION_MARKERS = frozenset({"unknown", "MISSING", ""})

_NON_CLASS_FIELD_KINDS_PRESERVED: frozenset[str] = frozenset(
    {
        "registry.contracts",
        "registry.runtime",
        "event.payload",
        "migration.mechanism",
        "compat.shim",
        "mechanism.policy_owner",
        "defect.persisted_without_version",
        "version.constant",
    }
)


class HistoricalP0ProjectionError(Exception):
    """Fail-closed historical AST projection (ambiguous or missing class field)."""


@dataclass(frozen=True, slots=True)
class SemanticIdentityReconciliation:
    current_identity: str
    legacy_identity: str
    owner_module_path: str
    class_name: str
    field_name: str
    current_version: str | None


@dataclass(frozen=True, slots=True)
class LegacySemanticCollisionGroup:
    legacy_identity: str
    members: tuple[SemanticIdentityReconciliation, ...]

    @property
    def member_count(self) -> int:
        return len(self.members)


@dataclass(frozen=True, slots=True)
class ClassFieldSurfaceTuple:
    owner_module_path: str
    class_name: str
    field_name: str


@dataclass(frozen=True, slots=True)
class SemanticIdentityReconciliationClosure:
    reconciliations: tuple[SemanticIdentityReconciliation, ...]
    collision_groups: tuple[LegacySemanticCollisionGroup, ...]
    legacy_projected_semantic_count: int
    current_semantic_count: int
    semantic_split_delta: int
    collision_group_count: int
    collision_group_member_total: int
    unexplained_added_semantic_identities: frozenset[str]
    unexplained_removed_semantic_identities: frozenset[str]
    semantic_surface_counts_by_kind: dict[str, int]


def historical_p0_literal_version(node: ast.AST | None) -> str | None:
    """Frozen accepted-P0 @ 0612c2a7 literal extraction from assignment value nodes only."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return str(node.value)
    if isinstance(node, ast.Subscript):
        return historical_p0_literal_version(node.slice)
    return None


def _normalize_current_version(raw: str | None) -> str | None:
    if raw is None or raw in _UNRESOLVED_VERSION_MARKERS:
        return None
    return raw


def parse_class_field_semantic_identity(semantic_identity: str) -> ClassFieldSurfaceTuple:
    if not semantic_identity.startswith(_CLASS_FIELD_PREFIX):
        raise ValueError(f"not a class.field semantic identity: {semantic_identity}")
    body = semantic_identity[len(_CLASS_FIELD_PREFIX) :]
    module_path, class_name, field_name = body.rsplit(":", 2)
    return ClassFieldSurfaceTuple(module_path, class_name, field_name)


def _read_module_source(
    module_path: str,
    source_by_module: Mapping[str, str] | None,
) -> str:
    if source_by_module is not None and module_path in source_by_module:
        return source_by_module[module_path]
    full = _REPO_ROOT / module_path
    if not full.is_file():
        raise HistoricalP0ProjectionError(f"no source for module {module_path}")
    return full.read_text(encoding="utf-8")


def _class_def_in_module(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _class_field_value_nodes(class_def: ast.ClassDef, field_name: str) -> list[ast.AST | None]:
    nodes: list[ast.AST | None] = []
    for item in class_def.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == field_name:
            nodes.append(item.value)
        elif (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == field_name
        ):
            nodes.append(item.value)
    return nodes


def locate_class_field_value_node(
    module_path: str,
    class_name: str,
    field_name: str,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> ast.AST | None:
    """Return the sole assignment value node for a module-level class field (fail closed if ambiguous)."""
    source = _read_module_source(module_path, source_by_module)
    tree = parse_module(module_path, source)
    class_def = _class_def_in_module(tree, class_name)
    if class_def is None:
        raise HistoricalP0ProjectionError(f"class {class_name} not found in {module_path}")
    value_nodes = _class_field_value_nodes(class_def, field_name)
    if len(value_nodes) == 0:
        raise HistoricalP0ProjectionError(
            f"field {class_name}.{field_name} not found in {module_path}"
        )
    if len(value_nodes) > 1:
        raise HistoricalP0ProjectionError(
            f"ambiguous field assignments for {class_name}.{field_name} in {module_path}"
        )
    return value_nodes[0]


def project_legacy_p0_class_field_semantic_identity_from_ast(
    surface: ClassFieldSurfaceTuple,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> str:
    """Derive accepted-P0 identity by applying historical literal rules to the field's AST value node."""
    class_field_identity = (
        f"{_CLASS_FIELD_PREFIX}{surface.owner_module_path}:{surface.class_name}:{surface.field_name}"
    )
    value_node = locate_class_field_value_node(
        surface.owner_module_path,
        surface.class_name,
        surface.field_name,
        source_by_module=source_by_module,
    )
    literal = historical_p0_literal_version(value_node)
    # Accepted P0 @ 0612c2a7 used `if signal.version_literal:` (falsy "" does not collapse).
    if literal:
        return f"{_LEGACY_LITERAL_PREFIX}{literal}"
    return class_field_identity


def project_legacy_p0_class_field_semantic_identity(
    current_identity: str,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> str:
    """Reproduce accepted-P0 class-field identity from current source AST (evidence only)."""
    surface = parse_class_field_semantic_identity(current_identity)
    return project_legacy_p0_class_field_semantic_identity_from_ast(
        surface, source_by_module=source_by_module
    )


def project_legacy_p0_semantic_identity_from_candidate(
    candidate: DiscoveryCandidate,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> str:
    if candidate.discovery_kind != "class.field.version":
        raise ValueError(f"expected class.field.version candidate, got {candidate.discovery_kind}")
    return project_legacy_p0_class_field_semantic_identity(
        candidate.semantic_identity,
        source_by_module=source_by_module,
    )


def _reconciliation_from_candidate(
    candidate: DiscoveryCandidate,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> SemanticIdentityReconciliation:
    parsed = parse_class_field_semantic_identity(candidate.semantic_identity)
    return SemanticIdentityReconciliation(
        current_identity=candidate.semantic_identity,
        legacy_identity=project_legacy_p0_class_field_semantic_identity_from_ast(
            parsed, source_by_module=source_by_module
        ),
        owner_module_path=parsed.owner_module_path,
        class_name=parsed.class_name,
        field_name=parsed.field_name,
        current_version=_normalize_current_version(candidate.current_version),
    )


def build_class_field_reconciliations_from_report(
    report: ClosedWorldReport,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> tuple[SemanticIdentityReconciliation, ...]:
    class_field_candidates = [
        c for c in report.raw_candidates if c.discovery_kind == "class.field.version"
    ]
    reconciled = [
        _reconciliation_from_candidate(c, source_by_module=source_by_module)
        for c in class_field_candidates
    ]
    return tuple(sorted(reconciled, key=lambda r: r.current_identity))


def legacy_projected_semantic_identity_set(
    report: ClosedWorldReport,
    *,
    source_by_module: Mapping[str, str] | None = None,
) -> frozenset[str]:
    projected: set[str] = set()
    class_field_legacy = {
        row.current_identity: row.legacy_identity
        for row in build_class_field_reconciliations_from_report(
            report, source_by_module=source_by_module
        )
    }
    for surface in report.semantic_surfaces:
        if surface.discovery_kind == "class.field.version":
            projected.add(class_field_legacy[surface.semantic_identity])
        else:
            projected.add(surface.semantic_identity)
    return frozenset(projected)


def current_semantic_identity_set(report: ClosedWorldReport) -> frozenset[str]:
    return frozenset(s.semantic_identity for s in report.semantic_surfaces)


def build_legacy_semantic_collision_groups(
    reconciliations: tuple[SemanticIdentityReconciliation, ...],
) -> tuple[LegacySemanticCollisionGroup, ...]:
    buckets: dict[str, list[SemanticIdentityReconciliation]] = defaultdict(list)
    for row in reconciliations:
        if row.legacy_identity.startswith(_LEGACY_LITERAL_PREFIX):
            buckets[row.legacy_identity].append(row)
    groups: list[LegacySemanticCollisionGroup] = []
    for legacy_id in sorted(buckets):
        members = tuple(sorted(buckets[legacy_id], key=lambda m: m.current_identity))
        if len(members) > 1:
            groups.append(LegacySemanticCollisionGroup(legacy_identity=legacy_id, members=members))
    return tuple(groups)


def derived_semantic_split_delta(collision_groups: tuple[LegacySemanticCollisionGroup, ...]) -> int:
    return sum(group.member_count - 1 for group in collision_groups)


def semantic_surface_counts_by_discovery_kind(report: ClosedWorldReport) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for surface in report.semantic_surfaces:
        counts[surface.discovery_kind] += 1
    return dict(sorted(counts.items()))


def class_field_surface_tuples(
    reconciliations: tuple[SemanticIdentityReconciliation, ...],
) -> frozenset[ClassFieldSurfaceTuple]:
    return frozenset(
        ClassFieldSurfaceTuple(r.owner_module_path, r.class_name, r.field_name) for r in reconciliations
    )


def _unexplained_identity_deltas(
    report: ClosedWorldReport,
    reconciliations: tuple[SemanticIdentityReconciliation, ...],
) -> tuple[frozenset[str], frozenset[str]]:
    current_ids = current_semantic_identity_set(report)
    legacy_ids = legacy_projected_semantic_identity_set(report)

    non_class_current = frozenset(
        s.semantic_identity
        for s in report.semantic_surfaces
        if s.discovery_kind != "class.field.version"
    )
    non_class_legacy = frozenset(
        identity
        for identity in legacy_ids
        if not identity.startswith(_CLASS_FIELD_PREFIX)
        and not identity.startswith(_LEGACY_LITERAL_PREFIX)
    )
    if non_class_current != non_class_legacy:
        raise AssertionError("non-class-field semantic identities changed under legacy projection")

    explained_current = non_class_current | frozenset(r.current_identity for r in reconciliations)
    unexplained_added = frozenset(current_ids - explained_current)
    unexplained_removed: frozenset[str] = frozenset()
    return unexplained_added, unexplained_removed


def verify_class_field_surface_coverage(
    report: ClosedWorldReport,
    reconciliations: tuple[SemanticIdentityReconciliation, ...],
) -> None:
    raw_tuples = frozenset(
        parse_class_field_semantic_identity(c.semantic_identity)
        for c in report.raw_candidates
        if c.discovery_kind == "class.field.version"
    )
    semantic_tuples = frozenset(
        parse_class_field_semantic_identity(s.semantic_identity)
        for s in report.semantic_surfaces
        if s.discovery_kind == "class.field.version"
    )
    reconciled_tuples = class_field_surface_tuples(reconciliations)
    if raw_tuples != reconciled_tuples:
        raise AssertionError("raw class-field candidates != reconciliation tuples")
    if semantic_tuples != reconciled_tuples:
        raise AssertionError("semantic class-field surfaces != reconciliation tuples")
    if len(reconciliations) != sum(
        1 for s in report.semantic_surfaces if s.discovery_kind == "class.field.version"
    ):
        raise AssertionError("class-field semantic surface cardinality != reconciliation rows")


def build_semantic_identity_reconciliation_closure(
    report: ClosedWorldReport | None = None,
) -> SemanticIdentityReconciliationClosure:
    closed = report or build_closed_world_report()
    reconciliations = build_class_field_reconciliations_from_report(closed)
    collision_groups = build_legacy_semantic_collision_groups(reconciliations)
    verify_class_field_surface_coverage(closed, reconciliations)
    for group in collision_groups:
        verify_collision_group_invariants(group)
    verify_class_field_legacy_mapping_bijection(reconciliations)
    legacy_count = len(legacy_projected_semantic_identity_set(closed))
    current_count = len(closed.semantic_surfaces)
    split_delta = derived_semantic_split_delta(collision_groups)
    if split_delta != current_count - legacy_count:
        raise AssertionError("semantic split delta != current - legacy projected counts")
    added, removed = _unexplained_identity_deltas(closed, reconciliations)
    member_total = sum(group.member_count for group in collision_groups)
    return SemanticIdentityReconciliationClosure(
        reconciliations=reconciliations,
        collision_groups=collision_groups,
        legacy_projected_semantic_count=legacy_count,
        current_semantic_count=current_count,
        semantic_split_delta=split_delta,
        collision_group_count=len(collision_groups),
        collision_group_member_total=member_total,
        unexplained_added_semantic_identities=added,
        unexplained_removed_semantic_identities=removed,
        semantic_surface_counts_by_kind=semantic_surface_counts_by_discovery_kind(closed),
    )


@lru_cache(maxsize=1)
def build_deterministic_semantic_identity_reconciliation() -> SemanticIdentityReconciliationClosure:
    return build_semantic_identity_reconciliation_closure(build_closed_world_report())


def verify_collision_group_invariants(group: LegacySemanticCollisionGroup) -> None:
    current_ids = [m.current_identity for m in group.members]
    tuples = [
        ClassFieldSurfaceTuple(m.owner_module_path, m.class_name, m.field_name) for m in group.members
    ]
    if len(current_ids) != len(set(current_ids)):
        raise AssertionError(f"duplicate current identities in {group.legacy_identity}")
    if len(tuples) != len(set(tuples)):
        raise AssertionError(f"duplicate module/class/field in {group.legacy_identity}")
    legacy_version = group.legacy_identity.removeprefix(_LEGACY_LITERAL_PREFIX)
    for member in group.members:
        if member.legacy_identity != group.legacy_identity:
            raise AssertionError("member legacy identity mismatch")
        if member.current_version is not None and member.current_version != legacy_version:
            raise AssertionError(
                f"current_version {member.current_version} incompatible with {group.legacy_identity}"
            )


def verify_class_field_legacy_mapping_bijection(
    reconciliations: tuple[SemanticIdentityReconciliation, ...],
) -> None:
    by_current: dict[str, SemanticIdentityReconciliation] = {}
    by_tuple: dict[ClassFieldSurfaceTuple, SemanticIdentityReconciliation] = {}
    for row in reconciliations:
        if row.current_identity in by_current:
            raise AssertionError(f"duplicate current identity {row.current_identity}")
        by_current[row.current_identity] = row
        key = ClassFieldSurfaceTuple(row.owner_module_path, row.class_name, row.field_name)
        if key in by_tuple:
            raise AssertionError(f"duplicate class-field tuple {key}")
        by_tuple[key] = row
