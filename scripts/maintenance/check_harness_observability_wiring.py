#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 observability wiring from environment profile (Phase OBS-3 / DIAG-FOUNDATION-2)."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.auditability_health_wiring import (
    assert_host_auditability_health_valid,
    project_host_auditability_health_facts_from_runtime,
)
from intergrax.applications._shared.diagnostic_read_wiring import resolve_host_diagnostic_read_service
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
    close_harness_host_runtime,
)
from intergrax.applications._shared.observability_assembly_resolver import (
    assert_observability_assembly_valid,
)
from intergrax.applications._shared.observability_wiring import wire_application_observability
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore

from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest


def _audit_lab_host() -> int:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    if env is None:
        print("lab manifest missing environment profile")
        return 1

    env = env.model_copy(
        update={
            "observability_profile": env.observability_profile.model_copy(
                update={"otlp_export_endpoint": "http://127.0.0.1:0/v1/logs"},
            ),
        },
    )
    manifest = manifest.model_copy(update={"environment": env})

    wiring = wire_application_observability(env)
    assert_observability_assembly_valid(wiring, env)

    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    if not isinstance(runtime.observability.trace_store, (SQLiteRunTraceStore,)):
        if env.observability_profile.trace_sqlite_enabled:
            print("lab host must use SQLite trace store when trace_sqlite_enabled")
            return 1

    if runtime.observability.runtime_event_store is None:
        if env.observability_profile.trace_sqlite_enabled:
            print("lab host must wire runtime event journal when trace_sqlite_enabled")
            return 1

    return 0


def _audit_product_host_auditability() -> int:
    import os
    import secrets
    import tempfile

    from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
    from intergrax.contracts.execution_mode import ExecutionMode
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from tests.fixtures.g09_tenant_evaluation_registry import CI_SMOKE_PRODUCT_TENANT

    from echo.echo_agent import EchoAgent

    cursor_secret_env = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
    prior_cursor_secret = os.environ.get(cursor_secret_env)
    os.environ[cursor_secret_env] = secrets.token_hex(32)

    environment = ApplicationEnvironmentProfile.product_defaults(profile_id="obs.auditability.product")
    environment.execution_mode = ExecutionMode.BALANCED
    environment.observability_profile = environment.observability_profile.model_copy(
        update={
            "otel_enabled": False,
            "otlp_export_endpoint": "http://127.0.0.1:0/v1/logs",
        },
    )
    manifest = ApplicationManifest.lab(
        app_id="obs_auditability_product",
        name="Observability Auditability Product Host",
        route_prefix="/v1/obs_auditability",
        env_prefix="OBS_AUDITABILITY_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
        environment=environment,
    )
    registry = AgentRegistry()
    registry.register(EchoAgent())
    try:
        with tempfile.TemporaryDirectory(prefix="obs-auditability-gate-") as gate_tmp:
            temp_dir = Path(gate_tmp)
            if environment.sandbox is not None:
                environment = environment.model_copy(
                    update={"sandbox": environment.sandbox.model_copy(update={"root": temp_dir / "sandbox"})},
                )
            runtime = build_harness_host_runtime(
                manifest,
                environment,
                registry=registry,
                tenant_id=CI_SMOKE_PRODUCT_TENANT,
                registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
                document_store=InMemoryDocumentStore(),
                trace_db_path=temp_dir / "trace.db",
                runtime_events_db_path=temp_dir / "events.db",
            )
            read_side_ready = True
            if environment.observability_profile.diagnostics_pane_enabled:
                try:
                    resolve_host_diagnostic_read_service(runtime)
                except ValueError:
                    read_side_ready = False
            facts = project_host_auditability_health_facts_from_runtime(
                runtime,
                diagnostic_read_side_ready=read_side_ready,
            )
            try:
                assert_host_auditability_health_valid(facts, environment)
            except Exception as exc:
                print(str(exc))
                close_harness_host_runtime(runtime)
                return 1
            if not runtime.diagnostic_wiring.attached:
                print("product host must attach central diagnostics for auditability conformance")
                close_harness_host_runtime(runtime)
                return 1
            close_harness_host_runtime(runtime)
            return 0
    finally:
        if prior_cursor_secret is None:
            os.environ.pop(cursor_secret_env, None)
        else:
            os.environ[cursor_secret_env] = prior_cursor_secret


def main() -> int:
    lab_result = _audit_lab_host()
    if lab_result != 0:
        return lab_result
    product_result = _audit_product_host_auditability()
    if product_result != 0:
        return product_result
    print("harness observability wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
