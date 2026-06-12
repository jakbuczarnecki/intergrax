#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ApplicationPackage closure for STRICT product hosts (APP-EVOL-7)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.environment_wiring import wire_application_environment  # noqa: E402
from intergrax.applications._shared.package_wiring import (  # noqa: E402
    build_application_package,
    load_application_package,
    package_gate_environment,
    validate_application_package_closure,
)
from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_strict_product_manifests,
)


def check_strict_product_packages() -> list[str]:
    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        gate_env = package_gate_environment(manifest.resolved_environment())
        try:
            wiring = wire_application_environment(manifest, gate_env, conformance_check=False)
        except Exception as exc:
            violations.append(f"{product_id}: wire_application_environment failed: {exc}")
            continue

        package = build_application_package(manifest, gate_env)
        violations.extend(
            validate_application_package_closure(
                package,
                manifest,
                gate_env,
                wiring.registry_snapshot,
                capability_graph=wiring.capability_graph,
            ),
        )
        if not package.distribution.checksum:
            violations.append(f"{product_id}: package checksum must be populated")
        if package.package_id != f"com.intergrax.{manifest.app_id}":
            violations.append(f"{product_id}: unexpected package_id {package.package_id!r}")
    return violations


def check_scaffold_package_roundtrip() -> list[str]:
    import tempfile

    from intergrax.scaffold.agent_catalog import resolve_agent_specs
    from intergrax.scaffold.application_names import ScaffoldApplicationNames
    from intergrax.scaffold.new_agent import create_agent
    from intergrax.scaffold.new_application import create_application

    violations: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        slug = "pkg_gate_stack"
        create_agent(name=slug, capabilities=[f"{slug}.basic"], root=root, minimal=True, force=True)
        resolve_agent_specs([slug])
        create_application(
            name=slug,
            agents=[slug],
            profile="lab",
            root=root,
            force=True,
            minimal=True,
        )
        names = ScaffoldApplicationNames.resolve(slug)
        package_path = root / "applications" / names.pkg / "package.json"
        if not package_path.is_file():
            return ["scaffold package.json was not emitted by new-stack/new-application"]
        try:
            package = load_application_package(package_path)
        except Exception as exc:
            return [f"scaffold package.json parse failed: {exc}"]
        if package.app_id != names.short:
            violations.append("scaffold package app_id mismatch")
        if not package.dependencies:
            violations.append("scaffold package must declare dependencies")
    return violations


def main() -> int:
    violations = check_strict_product_packages()
    violations.extend(check_scaffold_package_roundtrip())

    if violations:
        print("application package gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application package gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
