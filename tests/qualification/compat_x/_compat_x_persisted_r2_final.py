# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-FINAL current-HEAD persisted schema / migration reconciliation."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import get_type_hints

from tests.qualification.compat_x._compat_x_legacy_obligations import (
    legacy_persisted_compatibility_obligations,
)
from tests.qualification.compat_x._compat_x_persisted_contract_authority import (
    build_contract_authority_durability_evidence,
)
from tests.qualification.compat_x._compat_x_persisted_discovery import (
    build_persisted_mechanical_discovery_report,
    inventory_family_ids,
    mechanical_closed_world_family_ids,
)
from tests.qualification.compat_x._compat_x_persisted_gates import (
    build_persisted_r2_p0_qualification_report,
    persisted_r2_p0_inventory_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    iter_production_python_modules,
)
from tests.qualification.compat_x._compat_x_persisted_r2_r1_gates import (
    build_persisted_r2_r1_qualification_report,
)
from tests.qualification.compat_x._compat_x_persisted_r2_r1_r1_gates import (
    build_idempotency_schema_gate_report,
    persisted_r2_r1_r1_schema_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_r2_r2_gates import (
    compute_persisted_r2_r2_metrics,
    persisted_r2_r2_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    LegacyPersistedCompatibilityObligation,
    PersistedSchemaFamily,
)
from tests.qualification.compat_x._compat_x_r2_r2_r1_implicit_compatibility import (
    build_implicit_compatibility_report_cached,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "compat_x" / "r2_final_accepted_baseline.json"
_REPLAY_R2_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "compat_x" / "r2_final_accepted_baseline_replay_05549f9.json"
)
REPLAY_X_R2_LINEAGE_HEAD_SHA = "05549f9acb9ecedf910c7470d442bba5d0a27337"

_REDIS_LIVE_EVIDENCE_DEBT = (
    "TRACKED FREEZE DEBT: live Redis behavioral matrix not evidenced in R2-FINAL; "
    "static Lua/source gating only unless an environment-backed live test is executed."
)

_FIXTURE_TOP_LEVEL_KEYS = frozenset(
    {
        "provenance",
        "accepted_persisted_family_ids",
        "accepted_family_fingerprints",
        "accepted_legacy_obligation_fingerprints",
    }
)

_PROVENANCE_KEYS = frozenset(
    {
        "family_identity_baseline_sha",
        "schema_gate_accepted_sha",
        "closure_fingerprint_baseline_sha",
        "bookkeeping_task_baseline_sha",
    }
)

_FAMILY_FINGERPRINT_KEYS = frozenset(
    {
        "family_id",
        "version_disposition",
        "current_version",
        "version_owner_path",
        "migration_disposition",
        "migration_owner_path",
        "writer_paths",
        "reader_paths",
        "storage_kind",
        "canonical_payload_type",
        "owner_module_path",
    }
)

_OBLIGATION_FINGERPRINT_KEYS = frozenset(
    {
        "obligation_key",
        "family_id",
        "legacy_shape_id",
        "disposition",
        "semantic_owner_path",
        "migration_owner_path",
        "reader_path",
        "sanctioned_reader_symbols",
        "migration_boundary_helper_symbols",
    }
)

_FORBIDDEN_SEMANTIC_ANNOTATION_MARKERS = (
    "Any",
    "dict[str, Any]",
    "Dict[str, Any]",
    "Mapping[str, Any]",
    "Mapping[str, Mapping[str, Any]]",
)

_ACCEPTED_FAMILY_COUNT = 28
_REPLAY_ACCEPTED_FAMILY_COUNT = 29
REPLAY_ACCEPTED_FAMILY_COUNT = _REPLAY_ACCEPTED_FAMILY_COUNT
_ACCEPTED_OBLIGATION_COUNT = 16


def accepted_r2_baseline_fixture_path() -> Path:
    if _REPLAY_R2_FIXTURE_PATH.is_file():
        return _REPLAY_R2_FIXTURE_PATH
    return _FIXTURE_PATH


def accepted_r2_baseline_family_count_for_fixture(path: Path) -> int:
    if path.resolve() == _REPLAY_R2_FIXTURE_PATH.resolve():
        return _REPLAY_ACCEPTED_FAMILY_COUNT
    return _ACCEPTED_FAMILY_COUNT


@dataclass(frozen=True, slots=True)
class AcceptedR2BaselineProvenance:
    family_identity_baseline_sha: str
    schema_gate_accepted_sha: str
    closure_fingerprint_baseline_sha: str
    bookkeeping_task_baseline_sha: str


EXPECTED_ACCEPTED_R2_BASELINE_PROVENANCE = AcceptedR2BaselineProvenance(
    family_identity_baseline_sha="78bd502b5da5284b1bdaae2e321b7076fa22227d",
    schema_gate_accepted_sha="bb5900cd9a502f3e967b88ee54e4e18f6ff101cb",
    closure_fingerprint_baseline_sha="2e7d6173cc13bd1fb5ae48ef02292ee76bd22137",
    bookkeeping_task_baseline_sha="f526d97e6fb56bf032d60fd9166fb1c6fa9ecf42",
)


@dataclass(frozen=True, slots=True)
class AcceptedR2Baseline:
    provenance: AcceptedR2BaselineProvenance
    accepted_persisted_family_ids: frozenset[str]
    accepted_family_fingerprints: Mapping[str, PersistedFamilyFingerprint]
    accepted_legacy_obligation_fingerprints: Mapping[str, LegacyObligationFingerprint]


@dataclass(frozen=True, slots=True)
class PersistedFamilyFingerprint:
    family_id: str
    version_disposition: str
    current_version: str | None
    version_owner_path: str | None
    migration_disposition: str
    migration_owner_path: str | None
    writer_paths: tuple[str, ...]
    reader_paths: tuple[str, ...]
    storage_kind: str
    canonical_payload_type: str
    owner_module_path: str


@dataclass(frozen=True, slots=True)
class LegacyObligationFingerprint:
    obligation_key: str
    family_id: str
    legacy_shape_id: str
    disposition: str
    semantic_owner_path: str
    migration_owner_path: str | None
    reader_path: str
    sanctioned_reader_symbols: tuple[str, ...]
    migration_boundary_helper_symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R2FinalDrift:
    family_additions: tuple[str, ...]
    family_removals: tuple[str, ...]
    family_fingerprint_drift: tuple[str, ...]
    obligation_additions: tuple[str, ...]
    obligation_removals: tuple[str, ...]
    obligation_fingerprint_drift: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedR2FinalReport:
    production_modules_scanned: int
    raw_durable_candidates: int
    persisted_members: int
    evidence_backed_exclusions: int
    unclassified_candidates: int
    mechanical_family_ids: frozenset[str]
    inventory_family_ids: frozenset[str]
    closed_world_discovery_parity: bool
    missing_version_semantic_families: int
    opaque_carrier_families: int
    duplicate_version_authority_count: int
    duplicate_migration_authority_count: int
    qualification_as_version_authority: int
    writer_without_reader_violations: tuple[str, ...]
    reader_without_writer_violations: tuple[str, ...]
    migration_required_but_missing: int
    migration_owner_missing: int
    legacy_obligation_count: int
    unclassified_legacy_obligations: int
    implicit_compatibility_blockers: int
    orphan_migration_readers: int
    strong_typing_violations: tuple[str, ...]
    module_wide_sanctioned_exemptions: int
    sqlite_idempotency_schema_gate: bool
    redis_static_idempotency_schema_gate: bool
    package_agent_contract_authority_durability_blocker: bool
    redis_live_evidence_note: str
    tenant_isolation_verdict: str
    accepted_baseline: AcceptedR2Baseline
    drift: R2FinalDrift
    baseline_provenance_valid: bool
    r2_final_baseline_semantic_typing_violations: tuple[str, ...]
    r2_p0_gate: bool
    r2_r1_gate: bool
    r2_r2_gate: bool
    idempotency_r1_r1_gate: bool


def _reject_unknown_keys(mapping: Mapping[str, object], allowed: frozenset[str], context: str) -> None:
    extra = frozenset(mapping) - allowed
    if extra:
        raise ValueError(f"unknown keys in {context}: {sorted(extra)}")


def _require_str(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be str, got {type(value).__name__}")
    return value


def _require_str_or_null(mapping: Mapping[str, object], key: str, context: str) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be str or null, got {type(value).__name__}")
    return value


def _require_str_list(mapping: Mapping[str, object], key: str, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be list, got {type(value).__name__}")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}.{key}[{index}] must be str, got {type(item).__name__}")
        items.append(item)
    return tuple(items)


def _parse_provenance(raw: object) -> AcceptedR2BaselineProvenance:
    if not isinstance(raw, dict):
        raise ValueError("provenance must be object")
    _reject_unknown_keys(raw, _PROVENANCE_KEYS, "provenance")
    for key in _PROVENANCE_KEYS:
        if key not in raw:
            raise ValueError(f"provenance missing required field: {key}")
    return AcceptedR2BaselineProvenance(
        family_identity_baseline_sha=_require_str(raw, "family_identity_baseline_sha", "provenance"),
        schema_gate_accepted_sha=_require_str(raw, "schema_gate_accepted_sha", "provenance"),
        closure_fingerprint_baseline_sha=_require_str(raw, "closure_fingerprint_baseline_sha", "provenance"),
        bookkeeping_task_baseline_sha=_require_str(raw, "bookkeeping_task_baseline_sha", "provenance"),
    )


def _parse_family_fingerprint(raw: object, context: str) -> PersistedFamilyFingerprint:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be object")
    _reject_unknown_keys(raw, _FAMILY_FINGERPRINT_KEYS, context)
    for key in _FAMILY_FINGERPRINT_KEYS:
        if key not in raw:
            raise ValueError(f"{context} missing required field: {key}")
    return PersistedFamilyFingerprint(
        family_id=_require_str(raw, "family_id", context),
        version_disposition=_require_str(raw, "version_disposition", context),
        current_version=_require_str_or_null(raw, "current_version", context),
        version_owner_path=_require_str_or_null(raw, "version_owner_path", context),
        migration_disposition=_require_str(raw, "migration_disposition", context),
        migration_owner_path=_require_str_or_null(raw, "migration_owner_path", context),
        writer_paths=_require_str_list(raw, "writer_paths", context),
        reader_paths=_require_str_list(raw, "reader_paths", context),
        storage_kind=_require_str(raw, "storage_kind", context),
        canonical_payload_type=_require_str(raw, "canonical_payload_type", context),
        owner_module_path=_require_str(raw, "owner_module_path", context),
    )


def _parse_obligation_fingerprint(raw: object, context: str) -> LegacyObligationFingerprint:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be object")
    _reject_unknown_keys(raw, _OBLIGATION_FINGERPRINT_KEYS, context)
    for key in _OBLIGATION_FINGERPRINT_KEYS:
        if key not in raw:
            raise ValueError(f"{context} missing required field: {key}")
    return LegacyObligationFingerprint(
        obligation_key=_require_str(raw, "obligation_key", context),
        family_id=_require_str(raw, "family_id", context),
        legacy_shape_id=_require_str(raw, "legacy_shape_id", context),
        disposition=_require_str(raw, "disposition", context),
        semantic_owner_path=_require_str(raw, "semantic_owner_path", context),
        migration_owner_path=_require_str_or_null(raw, "migration_owner_path", context),
        reader_path=_require_str(raw, "reader_path", context),
        sanctioned_reader_symbols=_require_str_list(raw, "sanctioned_reader_symbols", context),
        migration_boundary_helper_symbols=_require_str_list(raw, "migration_boundary_helper_symbols", context),
    )


def _parse_obligation_fingerprints(raw: object) -> Mapping[str, LegacyObligationFingerprint]:
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((index, entry) for index, entry in enumerate(raw))
    else:
        raise ValueError(f"accepted_legacy_obligation_fingerprints must be object or list, got {type(raw).__name__}")
    parsed: dict[str, LegacyObligationFingerprint] = {}
    for key_or_index, entry in items:
        context = (
            f"accepted_legacy_obligation_fingerprints[{key_or_index}]"
            if isinstance(raw, list)
            else f"accepted_legacy_obligation_fingerprints.{key_or_index}"
        )
        fingerprint = _parse_obligation_fingerprint(entry, context)
        if fingerprint.obligation_key in parsed:
            raise ValueError(f"duplicate obligation_key in fixture: {fingerprint.obligation_key}")
        parsed[fingerprint.obligation_key] = fingerprint
    return types.MappingProxyType(parsed)


def validate_accepted_r2_baseline_integrity(
    baseline: AcceptedR2Baseline,
    *,
    expected_family_count: int | None = None,
) -> None:
    family_ids = baseline.accepted_persisted_family_ids
    family_fps = baseline.accepted_family_fingerprints
    obligation_fps = baseline.accepted_legacy_obligation_fingerprints

    expected_families = expected_family_count if expected_family_count is not None else len(family_ids)
    if len(family_ids) != expected_families:
        raise ValueError(f"accepted family count must be {expected_families}, got {len(family_ids)}")
    if len(obligation_fps) != _ACCEPTED_OBLIGATION_COUNT:
        raise ValueError(f"accepted obligation count must be {_ACCEPTED_OBLIGATION_COUNT}, got {len(obligation_fps)}")
    if frozenset(family_fps) != family_ids:
        raise ValueError("accepted family IDs must equal keys(accepted family fingerprints)")
    for family_id, fingerprint in family_fps.items():
        if fingerprint.family_id != family_id:
            raise ValueError(f"family fingerprint key {family_id!r} != fingerprint.family_id {fingerprint.family_id!r}")
    for obligation_key, fingerprint in obligation_fps.items():
        if fingerprint.obligation_key != obligation_key:
            raise ValueError(
                f"obligation fingerprint key {obligation_key!r} != "
                f"fingerprint.obligation_key {fingerprint.obligation_key!r}"
            )


def accepted_r2_baseline_provenance_valid(provenance: AcceptedR2BaselineProvenance) -> bool:
    return provenance == EXPECTED_ACCEPTED_R2_BASELINE_PROVENANCE


def discover_r2_final_baseline_semantic_typing_violations() -> tuple[str, ...]:
    violations: list[str] = []
    for cls in (
        AcceptedR2BaselineProvenance,
        AcceptedR2Baseline,
        PersistedFamilyFingerprint,
        LegacyObligationFingerprint,
    ):
        for field_name, hint in get_type_hints(cls).items():
            text = str(hint).replace("typing.", "")
            for marker in _FORBIDDEN_SEMANTIC_ANNOTATION_MARKERS:
                if marker in text:
                    violations.append(f"{cls.__name__}.{field_name}: {text}")
                    break
    return tuple(violations)


def load_accepted_r2_baseline(path: Path | None = None) -> AcceptedR2Baseline:
    fixture_path = path or accepted_r2_baseline_fixture_path()
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("baseline fixture root must be object")
    _reject_unknown_keys(raw, _FIXTURE_TOP_LEVEL_KEYS, "baseline fixture")

    for key in _FIXTURE_TOP_LEVEL_KEYS:
        if key not in raw:
            raise ValueError(f"baseline fixture missing required field: {key}")

    family_raw = raw["accepted_family_fingerprints"]
    if not isinstance(family_raw, dict):
        raise ValueError("accepted_family_fingerprints must be object")
    family_parsed: dict[str, PersistedFamilyFingerprint] = {}
    for family_id, entry in family_raw.items():
        if not isinstance(family_id, str):
            raise ValueError("accepted_family_fingerprints keys must be str")
        family_parsed[family_id] = _parse_family_fingerprint(entry, f"accepted_family_fingerprints.{family_id}")

    ids_raw = raw["accepted_persisted_family_ids"]
    if not isinstance(ids_raw, list):
        raise ValueError("accepted_persisted_family_ids must be list")
    accepted_ids: list[str] = []
    for index, item in enumerate(ids_raw):
        if not isinstance(item, str):
            raise ValueError(f"accepted_persisted_family_ids[{index}] must be str, got {type(item).__name__}")
        accepted_ids.append(item)

    baseline = AcceptedR2Baseline(
        provenance=_parse_provenance(raw["provenance"]),
        accepted_persisted_family_ids=frozenset(accepted_ids),
        accepted_family_fingerprints=types.MappingProxyType(family_parsed),
        accepted_legacy_obligation_fingerprints=_parse_obligation_fingerprints(
            raw["accepted_legacy_obligation_fingerprints"]
        ),
    )
    validate_accepted_r2_baseline_integrity(
        baseline,
        expected_family_count=accepted_r2_baseline_family_count_for_fixture(fixture_path),
    )
    return baseline


def family_fingerprint_from_inventory(family: PersistedSchemaFamily) -> PersistedFamilyFingerprint:
    return PersistedFamilyFingerprint(
        family_id=family.family_id,
        version_disposition=family.version_disposition.value,
        current_version=family.current_version,
        version_owner_path=family.version_owner_path,
        migration_disposition=family.migration_disposition.value,
        migration_owner_path=family.migration_owner_path,
        writer_paths=family.writer_paths,
        reader_paths=family.reader_paths,
        storage_kind=family.storage_kind.value,
        canonical_payload_type=family.canonical_payload_type,
        owner_module_path=family.owner_module_path,
    )


def obligation_fingerprint_from_obligation(
    obligation: LegacyPersistedCompatibilityObligation,
) -> LegacyObligationFingerprint:
    key = f"{obligation.family_id}::{obligation.legacy_shape_id}"
    return LegacyObligationFingerprint(
        obligation_key=key,
        family_id=obligation.family_id,
        legacy_shape_id=obligation.legacy_shape_id,
        disposition=obligation.disposition.value,
        semantic_owner_path=obligation.semantic_owner_path,
        migration_owner_path=obligation.migration_owner_path,
        reader_path=obligation.reader_path,
        sanctioned_reader_symbols=obligation.sanctioned_reader_symbols,
        migration_boundary_helper_symbols=obligation.migration_boundary_helper_symbols,
    )


def current_family_fingerprints() -> dict[str, PersistedFamilyFingerprint]:
    return {family.family_id: family_fingerprint_from_inventory(family) for family in PERSISTED_SCHEMA_FAMILIES}


def current_obligation_fingerprints() -> dict[str, LegacyObligationFingerprint]:
    result: dict[str, LegacyObligationFingerprint] = {}
    for obligation in legacy_persisted_compatibility_obligations():
        fp = obligation_fingerprint_from_obligation(obligation)
        if fp.obligation_key in result:
            raise ValueError(f"duplicate obligation key: {fp.obligation_key}")
        result[fp.obligation_key] = fp
    return result


def compute_r2_final_drift(
    baseline: AcceptedR2Baseline,
    current_family_ids: frozenset[str],
    current_families: Mapping[str, PersistedFamilyFingerprint],
    current_obligations: Mapping[str, LegacyObligationFingerprint],
) -> R2FinalDrift:
    accepted_ids = baseline.accepted_persisted_family_ids
    additions = tuple(sorted(current_family_ids - accepted_ids))
    removals = tuple(sorted(accepted_ids - current_family_ids))

    drift_families: list[str] = []
    for family_id in sorted(accepted_ids & current_family_ids):
        accepted = baseline.accepted_family_fingerprints[family_id]
        current = current_families[family_id]
        if accepted != current:
            drift_families.append(family_id)

    accepted_obl_keys = frozenset(baseline.accepted_legacy_obligation_fingerprints)
    current_obl_keys = frozenset(current_obligations)
    obl_additions = tuple(sorted(current_obl_keys - accepted_obl_keys))
    obl_removals = tuple(sorted(accepted_obl_keys - current_obl_keys))

    obl_drift: list[str] = []
    for key in sorted(accepted_obl_keys & current_obl_keys):
        if baseline.accepted_legacy_obligation_fingerprints[key] != current_obligations[key]:
            obl_drift.append(key)

    return R2FinalDrift(
        family_additions=additions,
        family_removals=removals,
        family_fingerprint_drift=tuple(drift_families),
        obligation_additions=obl_additions,
        obligation_removals=obl_removals,
        obligation_fingerprint_drift=tuple(obl_drift),
    )


def _tenant_isolation_verdict() -> str:
    return (
        "N/A — WITH EVIDENCE: R2-FINAL is qualification-only; no production tenant semantics "
        "changed; migration/read-compat paths do not mint tenant authority; TENANT-X remains mandatory."
    )


def build_persisted_r2_final_report(
    *,
    baseline: AcceptedR2Baseline | None = None,
    family_fingerprints: Mapping[str, PersistedFamilyFingerprint] | None = None,
    obligation_fingerprints: Mapping[str, LegacyObligationFingerprint] | None = None,
) -> PersistedR2FinalReport:
    accepted = baseline or load_accepted_r2_baseline()
    validate_accepted_r2_baseline_integrity(accepted)
    families = dict(family_fingerprints or current_family_fingerprints())
    obligations = dict(obligation_fingerprints or current_obligation_fingerprints())

    discovery = build_persisted_mechanical_discovery_report()
    p0 = build_persisted_r2_p0_qualification_report()
    r1 = build_persisted_r2_r1_qualification_report()
    r2_metrics = compute_persisted_r2_r2_metrics()
    implicit = build_implicit_compatibility_report_cached()
    idempotency = build_idempotency_schema_gate_report()
    authority = build_contract_authority_durability_evidence()

    mechanical_ids = mechanical_closed_world_family_ids()
    inventory_ids = inventory_family_ids()
    current_family_ids = frozenset(families)

    drift = compute_r2_final_drift(accepted, current_family_ids, families, obligations)

    redis_static = idempotency.redis_lua_scripts_gated and idempotency.redis_external_effect_read_gated

    return PersistedR2FinalReport(
        production_modules_scanned=len(iter_production_python_modules()),
        raw_durable_candidates=len(discovery.raw_candidates),
        persisted_members=discovery.persisted_member_count,
        evidence_backed_exclusions=discovery.exclusion_count,
        unclassified_candidates=len(discovery.unclassified_candidate_ids),
        mechanical_family_ids=mechanical_ids,
        inventory_family_ids=inventory_ids,
        closed_world_discovery_parity=mechanical_ids == inventory_ids
        and len(discovery.unclassified_candidate_ids) == 0
        and discovery.persisted_member_count + discovery.exclusion_count == len(discovery.raw_candidates),
        missing_version_semantic_families=r1.missing_version_semantic_family_count,
        opaque_carrier_families=r1.opaque_carrier_family_count,
        duplicate_version_authority_count=p0.metrics.duplicate_version_authority_count,
        duplicate_migration_authority_count=p0.metrics.duplicate_migration_authority_count,
        qualification_as_version_authority=p0.qualification_as_version_authority,
        writer_without_reader_violations=p0.writer_without_reader_violations,
        reader_without_writer_violations=p0.reader_without_writer_violations,
        migration_required_but_missing=p0.metrics.migration_required_count,
        migration_owner_missing=r2_metrics.migration_owner_missing,
        legacy_obligation_count=r2_metrics.legacy_compatibility_obligations,
        unclassified_legacy_obligations=r2_metrics.unclassified_legacy_obligations,
        implicit_compatibility_blockers=len(implicit.blockers),
        orphan_migration_readers=r2_metrics.orphan_migration_readers,
        strong_typing_violations=implicit.strong_typing_violations,
        module_wide_sanctioned_exemptions=implicit.module_wide_sanctioned_exemptions,
        sqlite_idempotency_schema_gate=idempotency.sqlite_mutation_methods_gated,
        redis_static_idempotency_schema_gate=redis_static,
        package_agent_contract_authority_durability_blocker=authority.in_scope_blocker,
        redis_live_evidence_note=_REDIS_LIVE_EVIDENCE_DEBT,
        tenant_isolation_verdict=_tenant_isolation_verdict(),
        accepted_baseline=accepted,
        drift=drift,
        baseline_provenance_valid=accepted_r2_baseline_provenance_valid(accepted.provenance),
        r2_final_baseline_semantic_typing_violations=discover_r2_final_baseline_semantic_typing_violations(),
        r2_p0_gate=persisted_r2_p0_inventory_gate_passes(),
        r2_r1_gate=r1.missing_version_semantic_family_count == 0
        and r1.opaque_carrier_family_count == 1
        and not r1.sqlite_contract_authority_blocker,
        r2_r2_gate=persisted_r2_r2_gate_passes(),
        idempotency_r1_r1_gate=persisted_r2_r1_r1_schema_gate_passes(),
    )


def persisted_r2_final_gate_passes(report: PersistedR2FinalReport | None = None) -> bool:
    item = report or build_persisted_r2_final_report()
    if not item.baseline_provenance_valid:
        return False
    if item.r2_final_baseline_semantic_typing_violations:
        return False
    if not item.closed_world_discovery_parity:
        return False
    if item.unclassified_candidates != 0:
        return False
    if item.mechanical_family_ids != item.inventory_family_ids:
        return False
    if item.missing_version_semantic_families != 0:
        return False
    if item.opaque_carrier_families != 1:
        return False
    if item.duplicate_version_authority_count != 0:
        return False
    if item.duplicate_migration_authority_count != 0:
        return False
    if item.qualification_as_version_authority != 0:
        return False
    if item.writer_without_reader_violations or item.reader_without_writer_violations:
        return False
    if item.migration_required_but_missing != 0:
        return False
    if item.migration_owner_missing != 0:
        return False
    if item.legacy_obligation_count != 16:
        return False
    if item.unclassified_legacy_obligations != 0:
        return False
    if item.implicit_compatibility_blockers != 0:
        return False
    if item.orphan_migration_readers != 0:
        return False
    if item.strong_typing_violations:
        return False
    if item.module_wide_sanctioned_exemptions != 0:
        return False
    if not item.sqlite_idempotency_schema_gate:
        return False
    if not item.redis_static_idempotency_schema_gate:
        return False
    if item.package_agent_contract_authority_durability_blocker:
        return False
    if not item.r2_p0_gate or not item.r2_r1_gate or not item.r2_r2_gate:
        return False
    if not item.idempotency_r1_r1_gate:
        return False
    drift = item.drift
    if drift.family_additions or drift.family_removals or drift.family_fingerprint_drift:
        return False
    if drift.obligation_additions or drift.obligation_removals or drift.obligation_fingerprint_drift:
        return False
    return True


@lru_cache(maxsize=1)
def build_persisted_r2_final_report_cached() -> PersistedR2FinalReport:
    return build_persisted_r2_final_report()
