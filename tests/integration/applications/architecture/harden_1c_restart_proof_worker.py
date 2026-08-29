# © Artur Czarnecki. All rights reserved.

"""HARDEN-1C subprocess worker — platform document-store restart proof phases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.registry.factory import resolve as resolve_integration
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus
from intergrax.runtime.registry.agent_registry import AgentRegistry

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_SKIP = 2

_DOCUMENT_PARTITION_PREFIX = "intergrax.diagnostic_problem.v1"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_1c"


def _json_default(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if is_dataclass(value):
        return {key: _json_default(item) for key, item in asdict(value).items()}
    raise TypeError(f"unsupported JSON value type: {type(value)!r}")


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=_json_default))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, *, code: int = _EXIT_ERROR) -> None:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.flush()
    raise SystemExit(code)


def _install_llm_stub() -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    import intergrax.applications._shared.llm_resolver as llm_resolver

    llm_resolver.resolve_llm_adapter = _resolve  # type: ignore[method-assign]


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _product_manifest(*, integration_profile: IntegrationProfile) -> ApplicationManifest:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="harden.1c.product")
    env.application_profile = ApplicationProfile.PRODUCT
    env.integration_profile = integration_profile
    env.observability_profile = ObservabilityProfile(
        trace_sqlite_enabled=True,
        otel_enabled=False,
        metrics_plugins_enabled=True,
    )
    return ApplicationManifest.lab(
        app_id="harden_1c_product",
        name="HARDEN-1C Product Host",
        route_prefix="/v1/harden_1c",
        env_prefix="HARDEN_1C_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=env,
        integration_profile=integration_profile,
    )


def _resolve_platform_document_store() -> ConditionalDocumentStore:
    bootstrap_application_integration_catalog()
    integration_profile = IntegrationProfile(document_store="mongodb")
    store = resolve_integration(
        IntegrationCategory.DOCUMENT_STORE,
        profile=integration_profile,
    )
    return assert_conditional_document_store(store)


def _build_product_runtime(work_dir: Path) -> tuple[HarnessHostRuntime, ConditionalDocumentStore]:
    bootstrap_application_integration_catalog()
    integration_profile = IntegrationProfile(document_store="mongodb")
    manifest = _product_manifest(integration_profile=integration_profile)
    env = manifest.environment
    if env is None:
        _fail("manifest environment is required")

    work_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        trace_db_path=work_dir / "trace.db",
        runtime_events_db_path=work_dir / "events.db",
    )
    wiring_context = runtime.env_wiring.build_context.tool_wiring_context
    if wiring_context is None or wiring_context.document_store is None:
        _fail("document store wiring is required")
    store = assert_conditional_document_store(wiring_context.document_store)
    return runtime, store


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def _purge_tenant_documents(store: ConditionalDocumentStore, tenant_id: str) -> None:
    partition_key = _document_partition(tenant_id)
    cursor: str | None = None
    while True:
        page = store.query(partition_key, limit=5000, cursor=cursor)
        for document in page.documents:
            store.delete(document.partition_key, document.row_key)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor


def _run_probe() -> None:
    try:
        store = _resolve_platform_document_store()
        store.close()
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        _fail(
            "MongoDB backend unavailable for HARDEN-1C: "
            f"{type(exc).__name__}: {exc}. "
            "Start infra/docker/mongodb/docker-compose.yml or set INTERGRAX_MONGODB_URI.",
            code=_EXIT_SKIP,
        )
    _emit({"ok": True, "pid": os.getpid(), "phase": "probe"})


def _run_phase_a(*, work_dir: Path, tenant_id: str) -> None:
    _install_llm_stub()
    runtime, store = _build_product_runtime(work_dir)
    runtime_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
    )
    if runtime_deps is None:
        _fail("runtime diagnostic dependencies are required")
    if not isinstance(runtime_deps.problem_persistence, DocumentStoreProblemPersistence):
        _fail("expected DocumentStoreProblemPersistence for phase A write")

    record = runtime_deps.problem_persistence.create(sample_problem(tenant_id=tenant_id))
    payload = {
        "ok": True,
        "pid": os.getpid(),
        "phase": "a",
        "problem_id": str(record.problem_id),
        "tenant_id": record.tenant_id,
        "status": record.status,
        "occurrence_count": record.occurrence_count,
        "grouping_provenance": grouping_provenance_from_problem_provenance(record.provenance),
        "occurrences": record.occurrences,
        "store_object_id": id(store),
        "runtime_object_id": id(runtime),
    }
    store.close()
    del runtime
    del store
    del runtime_deps
    _emit(payload)


def _run_phase_b(
    *,
    work_dir: Path,
    tenant_id: str,
    other_tenant_id: str,
    expect: dict[str, Any],
) -> None:
    _install_llm_stub()
    runtime, store = _build_product_runtime(work_dir)
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    if not isinstance(read_deps.problem_persistence, DocumentStoreProblemPersistence):
        _fail("expected DocumentStoreProblemPersistence for phase B read")
    read_service = build_diagnostic_read_service(read_deps)

    problem_id = expect["problem_id"]
    expected_status = ProblemStatus(expect["status"])
    expected_grouping_payload = expect["grouping_provenance"]
    expected_occurrences = expect["occurrences"]
    expected_occurrence_count = expect["occurrence_count"]

    listed = read_service.list_problems(tenant_id=tenant_id)
    if listed.total_count != 1:
        _fail(f"expected exactly one problem for tenant {tenant_id!r}, got {listed.total_count}")
    summary = listed.problems[0]
    if summary.problem_id != problem_id:
        _fail(f"problem_id mismatch: {summary.problem_id!r} != {problem_id!r}")
    if summary.tenant_id != tenant_id:
        _fail(f"tenant mismatch: {summary.tenant_id!r} != {tenant_id!r}")
    if summary.status != expected_status:
        _fail(f"status mismatch: {summary.status!r} != {expected_status!r}")
    if summary.occurrence_count != expected_occurrence_count:
        _fail(
            "occurrence_count mismatch: "
            f"{summary.occurrence_count!r} != {expected_occurrence_count!r}",
        )

    detail = read_service.get_problem(tenant_id=tenant_id, problem_id=problem_id)
    if detail is None:
        _fail(f"problem {problem_id!r} not found for tenant {tenant_id!r}")

    actual_grouping = json.loads(json.dumps(detail.grouping_provenance, default=_json_default))
    if actual_grouping != expected_grouping_payload:
        _fail("grouping provenance mismatch after restart")

    if detail.total_occurrence_count != len(expected_occurrences):
        _fail("total occurrence count mismatch after restart")

    observed_at = datetime.fromisoformat(expected_occurrences[0]["observed_at"])
    if detail.occurrences[0].observed_at != observed_at:
        _fail("first occurrence observed_at mismatch after restart")

    other_tenant = read_service.list_problems(tenant_id=other_tenant_id)
    if other_tenant.total_count != 0 or other_tenant.problems != ():
        _fail(f"tenant isolation failed for {other_tenant_id!r}")

    payload = {
        "ok": True,
        "pid": os.getpid(),
        "phase": "b",
        "problem_id": problem_id,
        "tenant_id": tenant_id,
        "store_object_id": id(store),
        "runtime_object_id": id(runtime),
    }
    store.close()
    del runtime
    del store
    del read_deps
    _emit(payload)


def _run_cleanup(*, tenant_ids: tuple[str, ...]) -> None:
    store = _resolve_platform_document_store()
    for tenant_id in tenant_ids:
        _purge_tenant_documents(store, tenant_id)
    store.close()
    _emit({"ok": True, "pid": os.getpid(), "phase": "cleanup"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HARDEN-1C process restart proof worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="verify MongoDB via platform document-store resolution")

    phase_a = subparsers.add_parser("phase-a", help="write durable Problem and exit process")
    phase_a.add_argument("--work-dir", type=Path, required=True)
    phase_a.add_argument("--tenant-id", required=True)

    phase_b = subparsers.add_parser("phase-b", help="read durable Problem in a fresh process")
    phase_b.add_argument("--work-dir", type=Path, required=True)
    phase_b.add_argument("--tenant-id", required=True)
    phase_b.add_argument("--other-tenant-id", required=True)
    phase_b.add_argument("--expect-file", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup", help="purge proof tenant documents via store API")
    cleanup.add_argument("--tenant-id", action="append", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "probe":
        _run_probe()
        return
    if args.command == "phase-a":
        _run_phase_a(work_dir=args.work_dir, tenant_id=args.tenant_id)
        return
    if args.command == "phase-b":
        expect = json.loads(args.expect_file.read_text(encoding="utf-8"))
        _run_phase_b(
            work_dir=args.work_dir,
            tenant_id=args.tenant_id,
            other_tenant_id=args.other_tenant_id,
            expect=expect,
        )
        return
    if args.command == "cleanup":
        _run_cleanup(tenant_ids=tuple(args.tenant_id))
        return
    _fail(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
