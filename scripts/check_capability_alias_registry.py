#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — capability alias registry on Tier-3 hosts (APP-EVOL-3)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import importlib
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.capability_alias_wiring import (  # noqa: E402
    check_environment_capability_aliases,
)
from intergrax.applications.contracts.manifest import ApplicationManifest  # noqa: E402


def _load_manifest(package: str) -> ApplicationManifest | None:
    try:
        module = importlib.import_module(f"{package}.manifest")
    except ImportError:
        return None
    for value in module.__dict__.values():
        if isinstance(value, ApplicationManifest):
            return value
    for name in dir(module):
        if not (name.startswith("build_") and "manifest" in name.lower()):
            continue
        builder = attribute_access.optional(module, name, None)
        if not callable(builder) or inspect.isclass(builder):
            continue
        try:
            signature = inspect.signature(builder)
        except (TypeError, ValueError):
            continue
        required = [
            param
            for param in signature.parameters.values()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            continue
        manifest = builder()
        if isinstance(manifest, ApplicationManifest):
            return manifest
    return None


def main() -> int:
    violations: list[str] = []
    for package_dir in sorted(APPLICATIONS_ROOT.glob("*_application")):
        manifest = _load_manifest(package_dir.name)
        if manifest is None:
            continue
        env = manifest.resolved_environment()
        violations.extend(
            check_environment_capability_aliases(package_dir.name, manifest, env.capability_governance_profile),
        )

    if violations:
        print("capability alias registry gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("capability alias registry gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
