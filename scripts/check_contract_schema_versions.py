#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Verify contract schema_version literals match registry (architecture §40.11 · ACP-PROD-11)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY  # noqa: E402


def _schema_version_field(model: object) -> object | None:
    """Pydantic ``ModelMetaclass`` exposes ``model_fields`` only via normal lookup."""
    if isinstance(model, type) and hasattr(model, "model_fields"):
        return model.model_fields.get("schema_version")
    fields = attribute_access.optional(model, "model_fields", {})
    if hasattr(fields, "get"):
        return fields.get("schema_version")
    return None


def main() -> int:
    violations: list[str] = []
    for entry in CONTRACT_SCHEMA_REGISTRY:
        module = importlib.import_module(entry.module_path)
        model = attribute_access.optional(module, entry.contract_name, None)
        if model is None:
            violations.append(f"missing model {entry.contract_name} in {entry.module_path}")
            continue
        schema_field = _schema_version_field(model)
        if schema_field is None:
            module_default = attribute_access.optional(module, "ACP_STATE_SCHEMA_VERSION", None)
            if module_default != entry.current_version:
                violations.append(
                    f"{entry.contract_name}: expected {entry.current_version}, got {module_default}"
                )
            continue
        default = schema_field.default
        if default != entry.current_version:
            violations.append(
                f"{entry.contract_name}: expected {entry.current_version}, got {default!r}"
            )

    artifact = (REPO_ROOT / "intergrax" / "contracts" / "artifact_ref.py").read_text(encoding="utf-8")
    if 'schema_version: Literal["artifact_ref.v1"]' not in artifact:
        violations.append("ArtifactRef schema_version literal missing in artifact_ref.py")

    if violations:
        print("Contract schema version violations:")
        print("\n".join(violations))
        return 1

    print("Contract schema versions: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
