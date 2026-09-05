#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ECC-9 — Code Craft layer catalog, wiring, and health probe gate."""

from __future__ import annotations

import sys
from pathlib import Path

from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.tools.providers.codecraft.service import CODECRAFT_TOOL_IDS
from intergrax.tools.providers.health.category_probes import HEALTH_CHECK_CODECRAFT_TOOL_ID


def main() -> int:
    errors: list[str] = []

    if not hasattr(TraceComponent, "CODECRAFT"):
        errors.append("TraceComponent.CODECRAFT missing")

    health_bundle = Path("intergrax/tools/providers/health/bundle.py").read_text(encoding="utf-8")
    if "HEALTH_CHECK_CODECRAFT_TOOL_ID" not in health_bundle:
        errors.append("HEALTH_CHECK_CODECRAFT_TOOL_ID not wired in health bundle")
    if "HealthCheckCodecraftHandler" not in health_bundle:
        errors.append("HealthCheckCodecraftHandler not registered in health bundle")

    codecraft_bundle = Path("intergrax/tools/providers/codecraft/bundle.py").read_text(encoding="utf-8")
    if "register_codecraft_tools" not in codecraft_bundle:
        errors.append("register_codecraft_tools missing from codecraft bundle")

    for tool_id in CODECRAFT_TOOL_IDS:
        if tool_id not in Path("intergrax/tools/providers/codecraft/service.py").read_text(encoding="utf-8"):
            errors.append(f"{tool_id} missing from codecraft service")
    wiring_source = Path("intergrax/applications/_shared/codecraft_wiring.py").read_text(encoding="utf-8")
    if "wire_application_codecraft" not in wiring_source:
        errors.append("wire_application_codecraft missing from codecraft_wiring.py")

    shipped_source = Path("intergrax/tools/registry/shipped_plugins.py").read_text(encoding="utf-8")
    if "codecraft" not in shipped_source:
        errors.append("codecraft tool plugin not in shipped_plugins")

    if HEALTH_CHECK_CODECRAFT_TOOL_ID != "health.check_codecraft":
        errors.append("HEALTH_CHECK_CODECRAFT_TOOL_ID slug mismatch")

    trace_source = Path("intergrax/runtime/codecraft/trace.py").read_text(encoding="utf-8")
    for step in (
        "codecraft.generation",
        "codecraft.test",
        "codecraft.iteration_verdict",
        "codecraft.hitl_requested",
        "codecraft.promoted",
    ):
        if step not in trace_source:
            errors.append(f"trace step {step!r} missing from trace.py")

    service_source = Path("intergrax/tools/providers/codecraft/service.py").read_text(encoding="utf-8")
    if (
        "resolve_craft_sandbox_session" not in service_source
        and "resolve_craft_sandbox_with_evidence" not in service_source
    ):
        errors.append("codecraft.run must use resolve_craft_sandbox_session")

    if errors:
        print("check_codecraft_layer: FAIL", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("OK: codecraft layer gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
