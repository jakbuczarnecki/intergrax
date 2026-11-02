# © Artur Czarnecki. All rights reserved.

"""W3-01/W3-03: COMPAT-policy contract / persisted-schema / event-schema diff (JSON-schema level)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as R


@dataclass(frozen=True)
class SchemaDiffRecord:
    result: R
    reasons: tuple[str, ...]


def _type_key(spec: dict[str, Any]) -> Any:
    if "type" in spec:
        return spec["type"]
    for key in ("anyOf", "oneOf"):
        if key in spec:
            return tuple(sorted(str(_type_key(s)) for s in spec[key]))
    return spec.get("$ref")


def diff_schema_record(before: dict[str, Any] | None, after: dict[str, Any] | None) -> SchemaDiffRecord:
    """Return result + reasons. UNKNOWN when either side is not a comparable object schema (fail-closed)."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return SchemaDiffRecord(R.UNKNOWN, ("schema not comparable",))
    before_props: dict[str, Any] = before.get("properties") or {}
    after_props: dict[str, Any] = after.get("properties") or {}
    before_required = set(before.get("required") or ())
    after_required = set(after.get("required") or ())
    incompatible: list[str] = []
    migration: list[str] = []
    for name in sorted(before_required - after_required):
        if name in after_props:
            incompatible.append(f"required->optional:{name}")
        else:
            incompatible.append(f"removed required:{name}")
    for name, spec in sorted(before_props.items()):
        if name not in after_props:
            if name not in before_required:
                incompatible.append(f"removed optional field (consumers may read it):{name}")
            continue
        if _type_key(spec) != _type_key(after_props[name]):
            incompatible.append(f"type changed:{name}")
        b_enum, a_enum = spec.get("enum"), after_props[name].get("enum")
        if b_enum and a_enum and not set(b_enum) <= set(a_enum):
            incompatible.append(f"enum narrowed:{name}")
        elif b_enum and not a_enum:
            migration.append(f"enum constraint removed:{name}")
    for name in sorted(after_required - before_required):
        migration.append(f"new required:{name}")
    for name in sorted(set(after_props) - set(before_props)):
        if name in after_required:
            migration.append(f"new required field:{name}")
    if incompatible:
        return SchemaDiffRecord(R.INCOMPATIBLE, tuple(incompatible))
    if migration:
        return SchemaDiffRecord(R.MIGRATION_REQUIRED, tuple(sorted(set(migration))))
    return SchemaDiffRecord(R.COMPATIBLE, ())


def diff_json_field_schemas(before: dict[str, Any], after: dict[str, Any]) -> R:
    return diff_schema_record(before, after).result


def assess_persisted_schema_change(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    migration_registered: bool,
) -> SchemaDiffRecord:
    """W3-03: persisted/event schema change. A breaking or migration-needing change without a registered migration is
    never reported as safe: INCOMPATIBLE (silent break) vs MIGRATION_REQUIRED (declared migration present)."""
    record = diff_schema_record(before, after)
    if record.result in {R.COMPATIBLE, R.UNKNOWN}:
        return record
    if record.result is R.MIGRATION_REQUIRED and migration_registered:
        return record
    if record.result is R.INCOMPATIBLE and migration_registered:
        return SchemaDiffRecord(R.MIGRATION_REQUIRED, (*record.reasons, "migration registered"))
    return SchemaDiffRecord(R.INCOMPATIBLE, (*record.reasons, "no migration registered (silent break)"))


def diff_event_schema(before: dict[str, Any] | None, after: dict[str, Any] | None) -> SchemaDiffRecord:
    """Events are consumed by unknown handlers: removed field / type change is INCOMPATIBLE (field rules)."""
    return diff_schema_record(before, after)
