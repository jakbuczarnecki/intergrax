#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-11.3 — oversized-tool lint enforcement in CI."""

from __future__ import annotations

import sys

from intergrax.tools.lint.oversized_tool_lint import lint_shipped_tool_contracts
from intergrax.tools.registry.bootstrap import register_default_tools
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile


def main() -> int:
    register_default_tools()
    registry = build_registry_from_profile(ToolProfile(register_all_catalog_bundles=True))
    contracts = [item.contract for item in registry.list()]
    if not contracts:
        print("no shipped tool contracts to lint", file=sys.stderr)
        return 1

    violations = lint_shipped_tool_contracts(contracts)
    if violations:
        for violation in violations[:10]:
            print(f"{violation.tool_id}: {violation.reason}", file=sys.stderr)
        print(f"oversized tool lint failed ({len(violations)} violations)", file=sys.stderr)
        return 1

    print(f"OK: oversized-tool lint ({len(contracts)} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
