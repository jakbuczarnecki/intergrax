# © Artur Czarnecki. All rights reserved.

"""Unit tests for DG-001D R4 supervisor pre-engine failure qualification helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.proof.dg001d_r4_qualification_contracts import (
    ControlledFailingHostedApplicationEngineFactory,
    qualification_secret_sentinel,
)
from scripts.proof.dg001d_r4_qualification_support import (
    build_qualification_environment,
    evaluate_prerequisites,
    spawn_supervisor_child,
)


def test_controlled_failing_engine_factory_raises_with_sentinel() -> None:
    factory = ControlledFailingHostedApplicationEngineFactory()
    with pytest.raises(RuntimeError) as exc_info:
        factory(MagicMock())
    assert qualification_secret_sentinel() in str(exc_info.value)


def test_build_qualification_environment_sets_observability_and_mongo(tmp_path: Path) -> None:
    environment = build_qualification_environment(
        marker="dg001d-r4-test",
        attempt_data_home=tmp_path,
        base_environment={},
        instance_id="dg001d-r4-instance-001",
    )
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] == "true"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] == "elasticsearch"
    assert environment["LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND"] == "mongodb"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT"] == "dg001d-r4-test"
    assert environment["DG001D_R4_INSTANCE_ID"] == "dg001d-r4-instance-001"


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
        marker="dg001d-r4-test",
        attempt_data_home=tmp_path,
        base_environment={},
    )
    serialized = json.dumps(environment)
    assert qualification_secret_sentinel() not in serialized


def test_spawn_supervisor_child_uses_qualification_child_script(tmp_path: Path) -> None:
    captured_command: list[str] = []

    def _runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="SUPERVISOR_EXIT_KIND=supervisor_error\nSUPERVISOR_INSTANCE_ID=dg001d-r4-instance-001\n",
            stderr="",
        )

    evidence = spawn_supervisor_child(
        build_qualification_environment(
            marker="dg001d-r4-test",
            attempt_data_home=tmp_path,
            base_environment={},
        ),
        subprocess_runner=_runner,
    )
    assert captured_command
    assert captured_command[1].replace("\\", "/").endswith(
        "scripts/proof/dg001d_r4_supervisor_child.py",
    )
    assert evidence.exit_code == 1
    assert evidence.instance_id == "dg001d-r4-instance-001"
