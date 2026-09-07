# © Artur Czarnecki. All rights reserved.

"""Unit tests for DG-001B R5-R1 qualification helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.proof.dg001b_r5_qualification_contracts import (
    ControlledFailingBackgroundWorkerConstructor,
    qualification_secret_sentinel,
)
from scripts.proof.dg001b_r5_qualification_support import (
    build_qualification_environment,
    evaluate_prerequisites,
    spawn_worker_child,
)


def test_controlled_failing_constructor_raises_type_error_with_sentinel() -> None:
    constructor = ControlledFailingBackgroundWorkerConstructor()
    with pytest.raises(TypeError) as exc_info:
        constructor(
            kv_store=MagicMock(),
            execution_registry=MagicMock(),
            idempotency_store=None,
            causal_evidence_persistence=MagicMock(),
        )
    assert qualification_secret_sentinel() in str(exc_info.value)


def test_build_qualification_environment_sets_observability_without_fault_flag(
    tmp_path: Path,
) -> None:
    environment = build_qualification_environment(
        marker="dg001b-r5-r1-test",
        attempt_data_home=tmp_path,
        base_environment={},
    )
    assert "LOCAL_WORKSPACE_WORKER_CONSTRUCTION_FAULT" not in environment
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] == "true"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] == "elasticsearch"
    assert environment["LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND"] == "mongodb"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT"] == "dg001b-r5-r1-test"


def test_evaluate_prerequisites_reports_backend_classes() -> None:
    status = evaluate_prerequisites(
        mongodb_uri="mongodb://example.invalid:27018/db",
        mongodb_database="intergrax_proofs",
        mongodb_collection="lkw_managed_workspaces",
        elasticsearch_url="http://127.0.0.1:9200",
        redis_url="redis://127.0.0.1:6379/0",
    )
    assert status.mongodb_uri_class == "mongodb+srv_or_standard_local"
    assert status.observability_backend == "elasticsearch"
    assert status.mongodb_database == "intergrax_proofs"
    assert status.mongodb_collection_authority == "lkw_managed_workspaces"
    assert isinstance(status.environment_ready, bool)


def test_qualification_environment_json_is_secret_free(tmp_path: Path) -> None:
    environment = build_qualification_environment(
        marker="dg001b-r5-r1-test",
        attempt_data_home=tmp_path,
        base_environment={},
    )
    serialized = json.dumps(environment)
    assert qualification_secret_sentinel() not in serialized


def test_spawn_worker_child_uses_qualification_child_script(tmp_path: Path) -> None:
    captured_command: list[str] = []

    def _runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="TypeError: create_kafka_worker composition failure",
        )

    evidence = spawn_worker_child(
        build_qualification_environment(
            marker="dg001b-r5-r1-test",
            attempt_data_home=tmp_path,
            base_environment={},
        ),
        subprocess_runner=_runner,
    )
    assert captured_command
    assert captured_command[1].replace("\\", "/").endswith("scripts/proof/dg001b_r5_worker_child.py")
    assert evidence.exit_code == 1
