#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when scaffolded factories use legacy NexusLoop wiring (Phase DX-8.3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SNIPPETS = (
    "from {pkg}.host.integration_wiring import wire_",
    "nexus_loop = NexusLoop(",
    "wire_{short}_integrations(",
)


def _check_factory(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if "build_harness_host_runtime" not in text and "create_" in path.name:
        errors.append(f"{path}: missing build_harness_host_runtime")
    if "nexus_loop = NexusLoop(" in text:
        errors.append(f"{path}: direct NexusLoop() construction")
    if ".host.integration_wiring import" in text and "factory" in path.name:
        errors.append(f"{path}: imports integration_wiring in factory")
    return errors


def main() -> int:
    errors: list[str] = []
    scaffold_gen = ROOT / "intergrax" / "scaffold" / "new_application.py"
    if scaffold_gen.is_file():
        body = scaffold_gen.read_text(encoding="utf-8")
        if "build_harness_host_runtime" not in body:
            errors.append("new_application.py factory template missing build_harness_host_runtime")
    for path in (ROOT / "applications").glob("*_application/host/factory.py"):
        errors.extend(_check_factory(path))
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("scaffold harness alignment: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
