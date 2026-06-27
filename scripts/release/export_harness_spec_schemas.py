#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Export JSON Schema for harness environment specs (Phase DX-7.1)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.applications.contracts.manifest import ApplicationManifest


def main() -> int:
    out = Path(__file__).resolve().parents[2] / "build" / "harness_specs"
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "ApplicationEnvironmentProfile": ApplicationEnvironmentProfile.model_json_schema(),
        "ApplicationManifest": ApplicationManifest.model_json_schema(),
        "ApplicationGraphSpec": ApplicationGraphSpec.model_json_schema(),
    }
    for name, schema in specs.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
