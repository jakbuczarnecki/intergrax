# © Artur Czarnecki. All rights reserved.

"""OBS-DG005-R1 — provider-neutral qualification harness contract proofs."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_port import (
    EvidencePersistencePort,
)
from intergrax.contracts.task_runtime_event_runs import TaskRuntimeEventRuns
from intergrax.runtime.events.evidence_persistence_adapter import (
    RuntimeEventPersistenceEvidenceAdapter,
)
from dataclasses import replace

from testing_support.obs_distributed_topology.provider_composition import (
    DEFAULT_DG005_EVIDENCE_PROVIDER_RESOLVER,
)
from testing_support.obs_distributed_topology.provider_contract import (
    EvidenceProviderDescriptor,
    QualificationEvidenceProviderResolver,
)
from testing_support.obs_distributed_topology.providers.sqlite_file import (
    SQLITE_FILE_PROVIDER_ID,
    sqlite_file_evidence_provider_factory,
)
from testing_support.obs_distributed_topology.scenario_builder import (
    build_dg005_scenario,
)
from testing_support.obs_distributed_topology.scenario_io import (
    read_scenario,
    scenario_from_dict,
    scenario_to_dict,
    write_scenario,
)
from testing_support.obs_distributed_topology.worker_ops import run_writer_role

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HARNESS_ROOT = _REPO_ROOT / "testing_support" / "obs_distributed_topology"


class _CustomQualificationPort(EvidencePersistencePort):
    """Contract proof — not RuntimeEventPersistenceEvidenceAdapter."""

    def append(self, event, *, tenant_id: str):
        raise NotImplementedError

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through=None,
        after=None,
    ):
        return []

    def list_for_task(self, task_id: str, *, tenant_id: str, limit: int = 1000):
        return []

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return TaskRuntimeEventRuns(runs=())

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return None

    def list_positioned_through(self, boundary, *, tenant_id: str, limit: int = 1000):
        return []


def _custom_factory(_: EvidenceProviderDescriptor) -> EvidencePersistencePort:
    return _CustomQualificationPort()


def test_r1_custom_port_accepted_without_runtime_adapter() -> None:
    resolver = QualificationEvidenceProviderResolver(
        providers={"custom-test": _custom_factory},
    )
    port = resolver.resolve(
        EvidenceProviderDescriptor(provider_id="custom-test", config={}),
    )
    assert isinstance(port, EvidencePersistencePort)
    assert not isinstance(port, RuntimeEventPersistenceEvidenceAdapter)


def test_r1_unknown_provider_fails_explicitly() -> None:
    descriptor = EvidenceProviderDescriptor(
        provider_id="postgres",
        config={"dsn": "example"},
    )
    with pytest.raises(ValueError, match="unknown qualification evidence provider"):
        DEFAULT_DG005_EVIDENCE_PROVIDER_RESOLVER.resolve(descriptor)


def test_r1_sqlite_malformed_config_fails_explicitly(tmp_path: Path) -> None:
    descriptor = EvidenceProviderDescriptor(
        provider_id=SQLITE_FILE_PROVIDER_ID,
        config={},
    )
    with pytest.raises(ValueError, match="db_path"):
        sqlite_file_evidence_provider_factory(descriptor)


def test_r1_scenario_provider_serialization_roundtrip(tmp_path: Path) -> None:
    scenario = build_dg005_scenario(
        qualification_sha="abc123",
        sqlite_db_path=tmp_path / "events.db",
    )
    path = tmp_path / "scenario.json"
    write_scenario(path, scenario)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["provider"]["provider_id"] == SQLITE_FILE_PROVIDER_ID
    assert "db_path" in raw["provider"]["config"]
    restored = read_scenario(path)
    assert restored.provider == scenario.provider
    assert scenario_to_dict(restored) == scenario_to_dict(scenario)


def test_r1_scenario_io_rejects_legacy_sqlite_only_provider_blob() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        scenario_from_dict(
            {
                "qualification_sha": "x",
                "provider": {"db_path": "/tmp/x.db"},
                "primary_tenant": "t",
                "foreign_tenant": "f",
                "primary_task_id": "task",
                "primary_run_id": "run",
                "isolated_run_id": "iso",
                "foreign_run_id": "for",
                "primary_attempt_id": "att",
                "primary_execution_id": "exe",
                "primary_events": [],
                "isolated_run_events": [],
                "foreign_tenant_events": [],
                "diagnostics_task_id": "dt",
                "diagnostics_run_id": "dr",
                "diagnostics_attempt_id": "da",
                "idempotent_event": {
                    "event_id": "e",
                    "tenant_id": "t",
                    "task_id": "task",
                    "run_id": "run",
                    "attempt_id": "att",
                    "execution_id": "exe",
                    "event_type": "step.started",
                    "timestamp_iso": "2026-01-01T00:00:00+00:00",
                },
                "reconstruction_initial_limit": 5,
                "as_of_position_index": 1,
            },
        )


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize(
    "relative_path",
    (
        "models.py",
        "qualification_harness.py",
        "worker_ops.py",
        "provider_contract.py",
    ),
)
def test_r1_architecture_gate_no_sqlite_in_neutral_modules(relative_path: str) -> None:
    path = _HARNESS_ROOT / relative_path
    imports = _module_imports(path)
    forbidden = (
        "intergrax.runtime.events.stores.sqlite_runtime_event_store",
        "intergrax.runtime.events.evidence_persistence_adapter",
    )
    for module in forbidden:
        assert module not in imports, f"{relative_path} imports {module}"


def test_r1_models_do_not_define_sqlite_config_type() -> None:
    text = (_HARNESS_ROOT / "models.py").read_text(encoding="utf-8")
    assert "SqliteEvidenceProviderConfig" not in text


def test_r1_worker_role_accepts_custom_resolver(tmp_path: Path) -> None:
    scenario = build_dg005_scenario(
        qualification_sha="sha",
        sqlite_db_path=tmp_path / "noop.db",
    )
    resolver = QualificationEvidenceProviderResolver(
        providers={"custom-test": _custom_factory},
    )
    scenario = replace(
        scenario,
        provider=EvidenceProviderDescriptor(provider_id="custom-test", config={}),
    )
    with pytest.raises(NotImplementedError):
        run_writer_role(scenario, provider_resolver=resolver)
