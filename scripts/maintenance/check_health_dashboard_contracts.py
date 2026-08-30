#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-21.2 / DIAG-FOUNDATION-2 — health dashboard contracts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.health_dashboard_wiring import (
    resolve_health_dashboard_wiring_from_runtime,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.contracts.execution_mode import ExecutionMode


def _build_product_runtime_for_gate() -> object:
    import tempfile

    from echo.echo_agent import EchoAgent
    from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    environment = ApplicationEnvironmentProfile.product_defaults(profile_id="health.dashboard.gate")
    environment.execution_mode = ExecutionMode.BALANCED
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"otel_enabled": False},
    )
    manifest = ApplicationManifest.lab(
        app_id="health_dashboard_gate",
        name="Health Dashboard Gate Host",
        route_prefix="/v1/health_dashboard_gate",
        env_prefix="HEALTH_DASHBOARD_GATE_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=environment,
    )
    registry = AgentRegistry()
    registry.register(EchoAgent())
    temp_dir = Path(tempfile.mkdtemp(prefix="health-dashboard-gate-"))
    return build_harness_host_runtime(
        manifest,
        environment,
        registry=registry,
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=InMemoryDocumentStore(),
        trace_db_path=temp_dir / "trace.db",
        runtime_events_db_path=temp_dir / "events.db",
    )


def main() -> int:
    runtime = _build_product_runtime_for_gate()
    wiring = resolve_health_dashboard_wiring_from_runtime(
        runtime,
        diagnostic_read_side_ready=True,
    )
    if not wiring.enabled:
        print("product host must enable health dashboard contracts", file=sys.stderr)
        return 1
    contract = wiring.contract
    if contract is None:
        print("health dashboard contract missing", file=sys.stderr)
        return 1
    if contract.governance.tenant_isolation_verified is not True:
        print("governance health must verify tenant isolation on product hosts", file=sys.stderr)
        return 1
    auditability = contract.auditability
    if not auditability.auditability_ready:
        print("product host auditability must be ready when diagnostics are attached", file=sys.stderr)
        return 1
    if not auditability.diagnostics_attached:
        print("product host health must reflect attached diagnostic wiring", file=sys.stderr)
        return 1
    print("OK: health dashboard contracts (quality/governance/cost/auditability)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
