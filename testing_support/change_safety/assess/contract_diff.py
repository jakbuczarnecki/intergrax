# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any

from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult


def diff_json_field_schemas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> CompatibilityAtChangeResult:
    before_props = before.get("properties") or {}
    after_props = after.get("properties") or {}
    before_required = set(before.get("required") or ())
    after_required = set(after.get("required") or ())
    removed_required = before_required - after_required
    if removed_required:
        return CompatibilityAtChangeResult.INCOMPATIBLE
    for name, spec in before_props.items():
        if name not in after_props:
            if name in before_required:
                return CompatibilityAtChangeResult.INCOMPATIBLE
            continue
        if spec.get("type") != after_props[name].get("type"):
            return CompatibilityAtChangeResult.INCOMPATIBLE
    if before_props != after_props and after_required - before_required:
        return CompatibilityAtChangeResult.MIGRATION_REQUIRED
    return CompatibilityAtChangeResult.COMPATIBLE
