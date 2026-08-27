# © Artur Czarnecki. All rights reserved.

"""Static/unit guardrails for LKW.7C1/C2 watcher-triggered E2E proof."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_DOCKER_DIR = _LKW_ROOT / "docker"
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.bat"
_WATCHER_COMPOSE = _DOCKER_DIR / "file-watcher-e2e.compose.yml"

_FORBIDDEN_TRIGGER_PATTERNS = (
    "/proof/background-task/enqueue",
    "local.workspace.index",
    "enqueue_background_ingest_job",
    "build_file_watcher_ingest_job",
    "LocalIndexerAgent",
    "run_index_job",
    "rag.ingest_document",
    "RAG_INGEST_TOOL_ID",
    "qdrant_client",
    "QdrantClient",
    "upsert(",
    "insert(",
)

_REQUIRED_PROOF_CONCEPTS = (
    "local.workspace.search",
    "lkw.search_summary.v1",
    "source_refs",
    "confluent_kafka",
    "docker compose",
    "lkw-file-watcher",
    "lkw-background-worker",
    "local_workspace",
    "qdrant",
    "embedding_readiness",
)

_REQUIRED_PASS_FIELDS = (
    "trigger",
    "manual_index_command",
    "direct_enqueue",
    "direct_indexer_call",
    "message_bus_provider",
    "worker_execution",
    "vector_store_provider",
    "persistent_index",
    "watcher_checkpoint_ready",
    "watcher_restored_after_restart",
    "tenant_id",
    "workspace_id",
    "collection_id",
    "marker",
    "proof_filename",
    "container_source_path",
    "task_topic",
    "task_count_before_file",
    "task_count_after_file",
    "task_topic_increased",
    "search_results_before_restart",
    "source_ref_found_before_restart",
    "restart_mode",
    "volumes_removed",
    "source_file_modified_after_index",
    "reindexed_after_restart",
    "task_count_before_restart",
    "task_count_after_restart",
    "duplicate_enqueue_after_restart",
    "search_results_after_restart",
    "source_ref_found_after_restart",
    "embedding_warmup_completed",
    "reviewer_rerun_required",
    "proof_receipt_recorded",
    "proof_receipt_verified",
    "proof_receipt_query_verified",
    "proof_receipt_store",
    "document_store_provider",
    "document_store_integration",
    "proof_receipt_id",
    "proof_receipt_run_id",
    "proof_receipt_result",
    "proof_receipt_application_id",
    "proof_receipt_task",
    "mongo_express_url",
    "markdown_source_of_truth",
    "direct_mongodb_write",
    "direct_pymongo_from_lkw",
    "manual_evidence_injection",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_proof_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lkw_file_watcher_e2e_proof", _PROOF_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_verified_receipt(proof: Any) -> Any:
    return proof.ProofReceipt(
        proof_id="local_workspace:file_watcher_persistent_search:run-1",
        proof_kind="file_watcher_persistent_search",
        application_id="local_workspace",
        result=proof.ProofReceiptResult.PASS,
        run_id="run-1",
    )


def _sample_workload_evidence(proof: Any, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "marker": "LKW_FILE_WATCHER_E2E_20260719T120000Z_ab12cd34",
        "proof_filename": "lkw_file_watcher_e2e_20260719T120000Z_ab12cd34.txt",
        "container_source_path": (
            "/data/user_docs/lkw_file_watcher_e2e_20260719T120000Z_ab12cd34.txt"
        ),
        "watcher_checkpoint_ready": True,
        "embedding_warmup_completed": True,
        "task_count_before_file": 1,
        "task_count_after_file": 2,
        "search_results_before_restart": 1,
        "source_ref_found_before_restart": True,
        "task_count_before_restart": 2,
        "task_count_after_restart": 2,
        "search_results_after_restart": 1,
        "source_ref_found_after_restart": True,
        "watcher_restored_after_restart": True,
        "watcher_final_checkpoint_saved": True,
        "source_file_modified_after_index": False,
        "restart_mode": "non_destructive",
        "volumes_removed": False,
    }
    values.update(overrides)
    return proof.FileWatcherE2EWorkloadEvidence(**values)


def test_overlay_filename_guardrail() -> None:
    assert _WATCHER_COMPOSE.name == "file-watcher-e2e.compose.yml"
    assert not re.fullmatch(r"docker-compose\..+\.yml", _WATCHER_COMPOSE.name)


def test_compose_overlay_watcher_service_contract() -> None:
    payload = yaml.safe_load(_read(_WATCHER_COMPOSE))
    services = payload["services"]
    assert "lkw-file-watcher" in services
    watcher = services["lkw-file-watcher"]
    assert watcher["command"] == [
        "python",
        "-m",
        "local_workspace_application.file_watcher",
    ]
    env = watcher["environment"]
    assert env["LOCAL_WORKSPACE_FILE_WATCHER_ENABLED"] == "true"
    assert env["LOCAL_WORKSPACE_FILE_WATCHER_TENANT_ID"] == "lkw-file-watcher-e2e"
    assert env["LOCAL_WORKSPACE_FILE_WATCHER_WORKSPACE_ID"] == "lkw-file-watcher-e2e"
    assert env["LOCAL_WORKSPACE_FILE_WATCHER_COLLECTION_ID"] == "lkw-file-watcher-e2e"
    assert env["INTERGRAX_ALLOWED_READ_ROOTS"] == "/data/user_docs"
    assert env["LOCAL_WORKSPACE_DATA_HOME"] == "/data/file_watcher_state"
    assert env["LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS"] == "true"
    assert env["LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS"] == "true"
    assert env["INTERGRAX_KAFKA_BOOTSTRAP_SERVERS"] == "lkw-kafka:9092"
    assert env["INTERGRAX_REDIS_URL"] == "redis://lkw-redis:6379/0"
    volumes = watcher["volumes"]
    assert "../.proof_docs:/data/user_docs:rw" in volumes
    assert "../.file_watcher_e2e_state:/data/file_watcher_state:rw" in volumes
    assert watcher["restart"] == "no"
    health_test = " ".join(watcher["healthcheck"]["test"])
    assert "/data/file_watcher_state/data/file_watcher/checkpoint.json" in health_test
    depends_on = watcher["depends_on"]
    assert "lkw-kafka" in depends_on
    assert "lkw-kafka-topics" in depends_on
    assert "lkw-redis" in depends_on
    assert "local_workspace" not in depends_on
    text = _read(_WATCHER_COMPOSE)
    assert "background_worker_main" not in text
    assert "uvicorn" not in text
    assert "mongo" not in text.lower()
    assert "ProofReceipt" not in text


def test_proof_script_trigger_guardrails() -> None:
    text = _read(_PROOF_SCRIPT)
    for pattern in _FORBIDDEN_TRIGGER_PATTERNS:
        assert pattern not in text, pattern
    for concept in _REQUIRED_PROOF_CONCEPTS:
        assert concept in text, concept


def test_four_file_compose_command_includes_mongodb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    command = proof.build_compose_command(
        "ps",
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    assert command == [
        "docker",
        "compose",
        "-p",
        "lkw-file-watcher-e2e-proof",
        "-f",
        "base.yml",
        "-f",
        "kafka.yml",
        "-f",
        "watcher.yml",
        "-f",
        "mongodb.yml",
        "ps",
    ]
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "lkw-core-platform-proof")
    command_with_project = proof.build_compose_command(
        "ps",
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    assert command_with_project[:5] == [
        "docker",
        "compose",
        "-p",
        "lkw-file-watcher-e2e-proof",
        "-f",
    ]
    assert "--mongodb-compose" in _read(_PROOF_SCRIPT)
    assert "docker-compose.mongodb.yml" in _read(_PROOF_SCRIPT)


def test_search_diagnostic_parsing_success_and_rejection() -> None:
    proof = _load_proof_module()
    expected = "/data/user_docs/example.txt"
    good = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "num_results": 2,
                        "evidence_count": 2,
                        "source_refs": [expected, "/data/user_docs/other.txt"],
                        "raw_tool_reason": "ok",
                        "used": True,
                        "reason": "retrieve_complete",
                    }
                }
            }
        }
    }
    diagnostics = proof.extract_search_diagnostics(good)
    assert diagnostics is not None
    assert diagnostics.num_results == 2
    assert diagnostics.evidence_count == 2
    assert diagnostics.raw_tool_reason == "ok"
    assert diagnostics.used is True
    assert diagnostics.reason is proof.SearchSummaryReason.RETRIEVE_COMPLETE
    assert proof.search_attempt_succeeded(diagnostics, expected_source_path=expected)

    wrong_source = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "num_results": 1,
                        "evidence_count": 1,
                        "source_refs": ["/data/user_docs/other.txt"],
                    }
                }
            }
        }
    }
    wrong = proof.extract_search_diagnostics(wrong_source)
    assert not proof.search_attempt_succeeded(wrong, expected_source_path=expected)

    missing = proof.extract_search_diagnostics({"metadata": {}})
    assert missing is None
    assert not proof.search_attempt_succeeded(missing, expected_source_path=expected)

    non_list_refs = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "num_results": 1,
                        "evidence_count": 1,
                        "source_refs": "not-a-list",
                        "raw_tool_reason": "safe",
                    }
                }
            }
        }
    }
    parsed = proof.extract_search_diagnostics(non_list_refs)
    assert parsed is not None
    assert parsed.source_refs == ()
    assert parsed.raw_tool_reason == "safe"
    assert not proof.search_attempt_succeeded(parsed, expected_source_path=expected)

    evidence_only = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "num_results": 0,
                        "evidence_count": 1,
                        "source_refs": [expected],
                    }
                }
            }
        }
    }
    evidence_diagnostics = proof.extract_search_diagnostics(evidence_only)
    assert proof.search_attempt_succeeded(
        evidence_diagnostics, expected_source_path=expected
    )


def test_embedding_readiness_probe_code_uses_production_pipeline() -> None:
    proof = _load_proof_module()
    code = proof.build_embedding_readiness_probe_code(
        probe_text="LKW_FILE_WATCHER_E2E_EMBEDDING_READINESS"
    )
    assert "create_default_embedding_pipeline" in code
    assert "embedding_profile_from_env" in code
    assert "embed_texts" in code
    assert "local.workspace.search" not in code
    assert "rag.retrieve" not in code
    assert "qdrant" not in code.lower()


def test_parse_embedding_readiness_output_success_and_failure() -> None:
    proof = _load_proof_module()
    success = proof.parse_embedding_readiness_output(
        'noise\n{"provider": "ollama", "model": "nomic-embed-text", "dimension": 768, "ok": true}\n'
    )
    assert success.ready is True
    assert success.provider == "ollama"
    assert success.model == "nomic-embed-text"
    assert success.dimension == 768
    assert success.failure_reason is None

    invalid_vector = proof.parse_embedding_readiness_output(
        '{"provider": "ollama", "model": "", "dimension": 0, "ok": false}'
    )
    assert invalid_vector.ready is False
    assert invalid_vector.failure_reason == "embedding_readiness_invalid_vector"

    invalid_output = proof.parse_embedding_readiness_output("not-json")
    assert invalid_output.ready is False
    assert invalid_output.failure_reason == "embedding_readiness_invalid_output"


def test_embedding_readiness_does_not_call_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    retrieval_calls: list[object] = []

    def fake_request_json(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        retrieval_calls.append("request_json")
        return {}

    monkeypatch.setattr(proof, "request_json", fake_request_json)
    monkeypatch.setattr(
        proof,
        "bootstrap_run_command",
        lambda *_a, **_k: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": '{"provider":"ollama","model":"m","dimension":768,"ok":true}'},
        )(),
    )
    result = proof.run_embedding_readiness(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
        cwd=Path("."),
        timeout_seconds=30,
    )
    assert result.ready is True
    assert retrieval_calls == []


def test_embedding_readiness_failure_contract() -> None:
    proof = _load_proof_module()
    assert "embedding_readiness_failed" in _read(_PROOF_SCRIPT)
    assert "run_persistence_embedding_warmup" not in _read(_PROOF_SCRIPT)
    assert "run_embedding_warmup" not in _read(_PROOF_SCRIPT)


def test_readiness_before_kafka_and_file_ordering() -> None:
    source = _read(_PROOF_SCRIPT)
    main_start = source.index("def main(")
    main_body = source[main_start:]
    bootstrap_idx = main_body.index("ensure_embedding_model_bootstrap_if_configured")
    readiness_idx = main_body.index("run_embedding_readiness")
    kafka_before_idx = main_body.index("task_count_before_file = inspect_kafka")
    create_doc_idx = main_body.index("create_proof_document")
    assert bootstrap_idx < readiness_idx < kafka_before_idx < create_doc_idx
    assert "run_persistence_embedding_warmup" not in main_body
    assert "run_embedding_warmup" not in main_body


def test_warmup_before_kafka_and_file_ordering() -> None:
    test_readiness_before_kafka_and_file_ordering()


def test_timeout_defaults() -> None:
    proof = _load_proof_module()
    args = proof._parse_args([])
    assert args.timeout_seconds == 600
    assert args.warmup_timeout_seconds == 300


def test_kafka_delta_helpers() -> None:
    proof = _load_proof_module()
    assert proof.kafka_topic_increased(before=1, after=2) is True
    assert proof.kafka_topic_increased(before=2, after=2) is False
    assert proof.kafka_topic_regression(before=2, after=1) is True
    assert proof.kafka_topic_regression(before=2, after=2) is False
    assert proof.duplicate_enqueue_detected(before_restart=3, after_restart=3) is False
    assert proof.duplicate_enqueue_detected(before_restart=3, after_restart=4) is True
    assert proof.kafka_topic_regression(before=3, after=2) is True


def test_proof_document_generation() -> None:
    proof = _load_proof_module()
    with tempfile.TemporaryDirectory() as tmp:
        proof_docs_dir = Path(tmp)
        document = proof.create_proof_document(proof_docs_dir)
        assert document.marker.startswith("LKW_FILE_WATCHER_E2E_")
        assert document.filename.startswith("lkw_file_watcher_e2e_")
        assert document.host_path.parent == proof_docs_dir
        assert document.host_path.exists()
        content = document.host_path.read_text(encoding="utf-8")
        assert document.marker in content
        assert document.container_source_path == (
            f"/data/user_docs/{document.filename}"
        )
        assert "after" in content.lower()
        assert "manual" in content.lower()


def test_search_request_shape() -> None:
    proof = _load_proof_module()
    request = proof.build_search_request("MARKER_ABC")
    assert request["capability"] == "local.workspace.search"
    assert request["tenant_id"] == "lkw-file-watcher-e2e"
    assert request["workspace_id"] == "lkw-file-watcher-e2e"
    assert request["user_id"] == "lkw.file_watcher"
    metadata = request["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["collection_id"] == "lkw-file-watcher-e2e"
    assert metadata["user_id"] == "lkw.file_watcher"
    assert metadata["query"] == "MARKER_ABC"
    assert metadata["top_k"] == 5
    serialized = str(request)
    assert "local.workspace.index" not in serialized
    assert "source_paths" not in serialized


def test_restart_command_targets() -> None:
    proof = _load_proof_module()
    command = proof.build_restart_command(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    joined = " ".join(command)
    assert command[:2] == ["docker", "compose"]
    assert "restart" in command
    assert "mongodb.yml" in command
    for service in (
        "lkw-file-watcher",
        "lkw-background-worker",
        "local_workspace",
        "qdrant",
    ):
        assert service in command
    assert "down" not in joined
    assert "down -v" not in joined
    assert "volume rm" not in joined
    assert "system prune" not in joined
    assert "lkw-kafka" not in command[command.index("restart") + 1 :]
    assert "lkw-redis" not in command[command.index("restart") + 1 :]
    assert "lkw-mongodb" not in command[command.index("restart") + 1 :]


def test_bat_runner_ordering_and_services() -> None:
    text = _read(_PROOF_BAT)
    reset_idx = text.lower().index("resetting dedicated watcher proof checkpoint state")
    materialize_idx = text.lower().index(
        "materializing minimal runtime context for local_workspace_application"
    )
    validate_idx = text.lower().index("validating docker compose merge")
    start_idx = text.lower().index("starting watcher e2e proof stack")
    health_idx = text.lower().index("waiting for lkw health")
    watcher_ready_idx = text.lower().index("waiting for watcher baseline checkpoint")
    mongo_idx = text.lower().index("waiting for mongodb health")
    mongo_express_idx = text.lower().index("waiting for mongo express")
    kafka_ui_idx = text.lower().index("waiting for kafka ui")
    invoke_idx = text.lower().index("invoking python watcher e2e proof workload")
    assert reset_idx < materialize_idx < validate_idx < start_idx < health_idx < watcher_ready_idx
    assert watcher_ready_idx < mongo_idx < mongo_express_idx < kafka_ui_idx < invoke_idx
    assert "watcher_container_running=true" in text
    assert "watcher_checkpoint_ready=true" in text
    assert "mongodb_container_healthy=true" in text
    assert "mongo_express_available=true" in text
    assert "kafka_ui=ok" in text
    for service in (
        "local_workspace",
        "lkw-background-worker",
        "lkw-file-watcher",
        "lkw-kafka",
        "lkw-kafka-topics",
        "lkw-kafka-ui",
        "lkw-redis",
        "qdrant",
        "ollama",
        "lkw-mongodb",
        "lkw-mongo-express",
    ):
        assert service in text
    assert "docker-compose.mongodb.yml" in text
    assert "--project applications/local_workspace_application" in text
    assert "INTERGRAX_MONGODB_URI=" in text
    assert "INTERGRAX_MONGODB_DATABASE=" in text
    assert "INTERGRAX_MONGODB_COLLECTION=" in text
    assert "--mongodb-compose" in text
    assert "--mongo-express" in text
    assert "echo %INTERGRAX_MONGODB_URI%" not in text
    assert "echo INTERGRAX_MONGODB_URI" not in text
    assert "docker compose down" not in text
    assert "down -v" not in text
    assert "volume rm" not in text
    assert "system prune" not in text


def test_bat_runner_materializes_runtime_context_before_compose() -> None:
    text = _read(_PROOF_BAT)
    pushd_idx = text.index('pushd "%REPO_ROOT%"')
    source_roots_idx = text.index("lkw_tier3_source_roots.py")
    materialize_idx = text.index("build_application_image.py")
    assert pushd_idx < source_roots_idx < materialize_idx
    assert "source_import_context=ready" in text
    assert "failure_reason=source_import_context_failed" in text
    assert '--format windows-path-list' in text
    compose_config_idx = text.lower().index("validating docker compose merge")
    compose_up_idx = text.lower().index("starting watcher e2e proof stack")
    ownership_idx = text.index('set "LKW_COMPOSE_OWNERSHIP_ENTERED=true"')
    assert materialize_idx < compose_config_idx < compose_up_idx < ownership_idx
    assert "--application local_workspace_application" in text
    assert (
        "--context-dir applications/local_workspace_application/docker/runtime-context"
        in text
    )
    assert "--materialize-only" in text
    assert "runtime_context_materialization=PASS" in text
    assert "runtime_context_materialization_failed" in text


def test_bat_runner_materialization_failure_gates_compose() -> None:
    text = _read(_PROOF_BAT)
    failure_idx = text.index("failure_reason=runtime_context_materialization_failed")
    ownership_idx = text.index('set "LKW_COMPOSE_OWNERSHIP_ENTERED=true"')
    compose_up_idx = text.lower().index("up -d --build")
    assert failure_idx < ownership_idx < compose_up_idx
    materialize_block = text[failure_idx:ownership_idx]
    assert "goto proof_fail" in materialize_block
    assert "xcopy" not in text.lower()
    assert "robocopy" not in text.lower()


def test_bat_runner_materialization_remains_sync_capable() -> None:
    text = _read(_PROOF_BAT)
    materialize_line = next(
        line
        for line in text.splitlines()
        if "build_application_image.py" in line and "uv run" in line
    )
    assert "--materialize-only" in materialize_line
    assert "--no-sync" not in materialize_line
    assert materialize_line.strip().startswith(
        "uv run python scripts/build/build_application_image.py"
    )


def test_bat_runner_workload_uses_no_sync() -> None:
    text = _read(_PROOF_BAT)
    assert 'set "PROOF=%SCRIPT_DIR%run-lkw-file-watcher-e2e-proof.py"' in text
    workload_line = next(
        line
        for line in text.splitlines()
        if 'python "%PROOF%"' in line and "uv run" in line
    )
    assert (
        "uv run --no-sync --project applications/local_workspace_application"
        in workload_line
    )


def test_bat_runner_teardown_uses_no_sync() -> None:
    text = _read(_PROOF_BAT)
    assert (
        'set "LIFECYCLE=%SCRIPT_DIR%lkw_proof_compose_lifecycle.py"' in text
    )
    teardown_line = next(
        line
        for line in text.splitlines()
        if 'python "%LIFECYCLE%"' in line and "teardown" in line
    )
    assert (
        "uv run --no-sync --project applications/local_workspace_application"
        in teardown_line
    )


def test_bat_runner_exactly_two_no_sync_boundaries() -> None:
    text = _read(_PROOF_BAT)
    no_sync_lines = [line for line in text.splitlines() if "--no-sync" in line]
    assert len(no_sync_lines) == 2
    assert all(
        "uv run --no-sync --project applications/local_workspace_application"
        in line
        for line in no_sync_lines
    )
    materialize_block_end = text.index("runtime_context_materialization=PASS")
    assert "--no-sync" not in text[:materialize_block_end]


def test_pass_output_fields() -> None:
    proof = _load_proof_module()
    receipt = _fake_verified_receipt(proof)
    workload = _sample_workload_evidence(
        proof,
        marker="MARKER",
        proof_filename="file.txt",
        container_source_path="/data/user_docs/file.txt",
    )
    evidence = proof.build_pass_evidence(
        workload_evidence=workload,
        verified_receipt=receipt,
        integration_class="MongoDBDocumentStoreIntegration",
        mongo_express_url="http://127.0.0.1:8086",
    )
    for field in _REQUIRED_PASS_FIELDS:
        assert field in evidence
    assert evidence["watcher_restored_after_restart"] is (
        workload.watcher_restored_after_restart
    )
    assert evidence["source_file_modified_after_index"] is (
        workload.source_file_modified_after_index
    )
    assert evidence["embedding_warmup_completed"] is (
        workload.embedding_warmup_completed
    )
    assert evidence["reviewer_rerun_required"] is workload.reviewer_rerun_required
    assert evidence["proof_receipt_recorded"] is True
    assert evidence["proof_receipt_verified"] is True
    assert evidence["proof_receipt_query_verified"] is True
    assert evidence["proof_receipt_task"] == "LKW.7C2"
    rendered = proof.format_pass_output(evidence)
    assert "proof_result=PASS" in rendered
    assert "watcher_restored_after_restart=true" in rendered
    assert "source_file_modified_after_index=false" in rendered
    assert "embedding_warmup_completed=true" in rendered
    assert "reviewer_rerun_required=false" in rendered
    assert "proof_receipt_recorded=true" in rendered
    assert "proof_receipt_task=LKW.7C2" in rendered
    assert "redis://" not in rendered.lower()
    assert "password" not in rendered.lower()
    assert "This file was created" not in rendered


def test_pass_evidence_maps_typed_workload_object() -> None:
    proof = _load_proof_module()
    receipt = _fake_verified_receipt(proof)
    workload = _sample_workload_evidence(proof)
    measured = proof.build_pass_evidence(
        workload_evidence=workload,
        verified_receipt=receipt,
        integration_class="MongoDBDocumentStoreIntegration",
        mongo_express_url="http://127.0.0.1:8086",
    )
    assert measured["watcher_checkpoint_ready"] is workload.watcher_checkpoint_ready
    assert (
        measured["watcher_restored_after_restart"]
        is workload.watcher_restored_after_restart
    )
    assert measured["task_topic_increased"] is workload.task_topic_increased
    assert (
        measured["source_ref_found_before_restart"]
        is workload.source_ref_found_before_restart
    )
    assert (
        measured["source_file_modified_after_index"]
        is workload.source_file_modified_after_index
    )
    assert measured["reindexed_after_restart"] is workload.reindexed_after_restart
    assert (
        measured["duplicate_enqueue_after_restart"]
        is workload.duplicate_enqueue_after_restart
    )
    assert (
        measured["source_ref_found_after_restart"]
        is workload.source_ref_found_after_restart
    )
    assert measured["embedding_warmup_completed"] is workload.embedding_warmup_completed
    assert measured["reviewer_rerun_required"] is workload.reviewer_rerun_required

    invalid = _sample_workload_evidence(
        proof,
        watcher_restored_after_restart=False,
    )
    with pytest.raises(ValueError, match="watcher_restore_not_proven"):
        proof.build_pass_evidence(
            workload_evidence=invalid,
            verified_receipt=receipt,
            integration_class="MongoDBDocumentStoreIntegration",
            mongo_express_url="http://127.0.0.1:8086",
        )

    source = _read(_PROOF_SCRIPT)
    fn_start = source.index("def build_pass_evidence(")
    fn_end = source.index("\ndef ", fn_start + 1)
    body = source[fn_start:fn_end]
    assert '"watcher_restored_after_restart": True' not in body
    assert '"source_file_modified_after_index": False' not in body
    assert "validate_file_watcher_e2e_workload_evidence" in body


def test_proof_file_stat_unchanged_comparisons() -> None:
    proof = _load_proof_module()
    same = proof.ProofFileStat(size_bytes=10, modified_time_ns=100)
    assert (
        proof.proof_file_stat_unchanged(
            before=same,
            after=proof.ProofFileStat(size_bytes=10, modified_time_ns=100),
        )
        is True
    )
    assert (
        proof.proof_file_stat_unchanged(
            before=same,
            after=proof.ProofFileStat(size_bytes=11, modified_time_ns=100),
        )
        is False
    )
    assert (
        proof.proof_file_stat_unchanged(
            before=same,
            after=proof.ProofFileStat(size_bytes=10, modified_time_ns=101),
        )
        is False
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proof.txt"
        path.write_text("hello", encoding="utf-8")
        captured = proof.capture_proof_file_stat(path)
        assert captured.size_bytes == path.stat().st_size
        assert captured.modified_time_ns == path.stat().st_mtime_ns


def test_extract_last_file_watcher_sidecar_result() -> None:
    proof = _load_proof_module()
    valid = {
        "schema_version": "lkw.file_watcher_sidecar_result.v1",
        "exit_kind": "clean_stop",
        "exit_code": 0,
        "restored_from_checkpoint": True,
        "cycles_completed": 5,
        "last_cycle_status": "idle",
        "final_checkpoint_saved": True,
        "error_id": None,
    }
    valid_json = json.dumps(valid)
    assert proof.extract_last_file_watcher_sidecar_result(valid_json) == valid

    multi = "\n".join(
        [
            "INFO starting",
            json.dumps({**valid, "cycles_completed": 1}),
            "not-json",
            "{broken",
            json.dumps({"schema_version": "other.v1"}),
            json.dumps({"hello": "world"}),
            f"lkw-file-watcher  | {json.dumps({**valid, 'cycles_completed': 9})}",
        ]
    )
    last = proof.extract_last_file_watcher_sidecar_result(multi)
    assert last is not None
    assert last["cycles_completed"] == 9
    assert proof.extract_last_file_watcher_sidecar_result("INFO only\n") is None


def test_sidecar_result_proves_checkpoint_restore() -> None:
    proof = _load_proof_module()
    valid = {
        "schema_version": "lkw.file_watcher_sidecar_result.v1",
        "exit_kind": "clean_stop",
        "exit_code": 0,
        "restored_from_checkpoint": True,
        "final_checkpoint_saved": True,
        "error_id": None,
    }
    assert proof.sidecar_result_proves_checkpoint_restore(valid) is True
    assert proof.sidecar_result_proves_checkpoint_restore(None) is False
    assert (
        proof.sidecar_result_proves_checkpoint_restore(
            {**valid, "restored_from_checkpoint": False}
        )
        is False
    )
    assert (
        proof.sidecar_result_proves_checkpoint_restore(
            {**valid, "exit_kind": "startup_failed"}
        )
        is False
    )
    assert (
        proof.sidecar_result_proves_checkpoint_restore(
            {**valid, "exit_kind": "checkpoint_failed"}
        )
        is False
    )
    assert (
        proof.sidecar_result_proves_checkpoint_restore({**valid, "exit_code": 1})
        is False
    )
    assert (
        proof.sidecar_result_proves_checkpoint_restore(
            {**valid, "final_checkpoint_saved": False}
        )
        is False
    )
    assert (
        proof.sidecar_result_proves_checkpoint_restore(
            {**valid, "error_id": "checkpoint_failed"}
        )
        is False
    )


def test_watcher_evidence_compose_commands() -> None:
    proof = _load_proof_module()
    stop_cmd = proof.build_watcher_graceful_stop_command(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    logs_cmd = proof.build_watcher_logs_command(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    resume_cmd = proof.build_watcher_resume_command(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
    )
    assert stop_cmd[-4:] == ["stop", "--timeout", "30", "lkw-file-watcher"]
    assert "mongodb.yml" in stop_cmd
    assert logs_cmd[-6:] == [
        "logs",
        "--no-color",
        "--no-log-prefix",
        "--tail",
        "200",
        "lkw-file-watcher",
    ]
    assert resume_cmd[-3:] == ["up", "-d", "lkw-file-watcher"]
    text = _read(_PROOF_SCRIPT)
    assert "docker kill" not in text
    assert "docker rm" not in text
    assert "docker compose down" not in text
    assert "down -v" not in text


def test_main_flow_evidence_ordering() -> None:
    source = _read(_PROOF_SCRIPT)
    main_start = source.index("def main(")
    main_body = source[main_start:]
    first_poll = main_body.index("_poll_search_until_indexed")
    second_poll = main_body.index("_poll_search_until_indexed", first_poll + 1)
    markers = [
        ("bootstrap", main_body.index("ensure_embedding_model_bootstrap_if_configured")),
        ("embedding_readiness", main_body.index("run_embedding_readiness")),
        (
            "kafka_before_file",
            main_body.index("task_count_before_file = inspect_kafka"),
        ),
        ("create_document", main_body.index("create_proof_document")),
        ("first_search", first_poll),
        (
            "source_stat_after_index",
            main_body.index("source_stat_after_index = capture_proof_file_stat"),
        ),
        ("component_restart", main_body.index("*_RESTART_SERVICES")),
        ("duplicate_enqueue", main_body.index("duplicate_enqueue_detected")),
        ("post_restart_search", second_poll),
        (
            "source_stat_after_restart",
            main_body.index("source_stat_after_restart = capture_proof_file_stat"),
        ),
        ("compare_source_stat", main_body.index("proof_file_stat_unchanged")),
        ("graceful_stop", main_body.index('"stop"')),
        (
            "read_watcher_result",
            main_body.index("extract_last_file_watcher_sidecar_result"),
        ),
        (
            "validate_restore",
            main_body.index("sidecar_result_proves_checkpoint_restore"),
        ),
        ("resume_watcher", main_body.index('"up"')),
        (
            "record_receipt",
            main_body.index("record_file_watcher_e2e_proof_receipt"),
        ),
        ("pass_evidence", main_body.index("build_pass_evidence")),
        ("pass_output", main_body.index("format_pass_output")),
    ]
    positions = [index for _, index in markers]
    assert positions == sorted(positions)
    assert main_body.index("search_after_restart_failed") < main_body.index('"stop"')


def test_ensure_embedding_model_bootstrap_ollama_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    calls: list[str] = []

    def fake_ensure(**_kwargs: Any) -> str:
        calls.append("bootstrap")
        return "resolved-model"

    monkeypatch.setattr(proof, "_ensure_ollama_embedding_model_if_configured", fake_ensure)
    resolved = proof.ensure_embedding_model_bootstrap_if_configured(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
        cwd=Path("."),
        timeout_seconds=30,
    )
    assert resolved == "resolved-model"
    assert calls == ["bootstrap"]


def test_ensure_embedding_model_bootstrap_non_ollama_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    monkeypatch.setattr(
        proof,
        "_ensure_ollama_embedding_model_if_configured",
        lambda **_kwargs: None,
    )
    resolved = proof.ensure_embedding_model_bootstrap_if_configured(
        base_compose=Path("base.yml"),
        kafka_compose=Path("kafka.yml"),
        watcher_compose=Path("watcher.yml"),
        mongodb_compose=Path("mongodb.yml"),
        cwd=Path("."),
        timeout_seconds=30,
    )
    assert resolved is None


def test_ensure_embedding_model_bootstrap_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()

    def fake_ensure(**_kwargs: Any) -> str:
        raise proof.OllamaEmbeddingBootstrapError("embedding_model_pull_failed")

    monkeypatch.setattr(proof, "_ensure_ollama_embedding_model_if_configured", fake_ensure)
    with pytest.raises(proof.OllamaEmbeddingBootstrapError) as exc:
        proof.ensure_embedding_model_bootstrap_if_configured(
            base_compose=Path("base.yml"),
            kafka_compose=Path("kafka.yml"),
            watcher_compose=Path("watcher.yml"),
            mongodb_compose=Path("mongodb.yml"),
            cwd=Path("."),
            timeout_seconds=30,
        )
    assert exc.value.reason == "embedding_model_pull_failed"


def test_main_bootstrap_before_readiness_and_failure_skips_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    events: list[str] = []
    readiness = proof.EmbeddingReadinessResult(
        ready=True,
        provider="ollama",
        model="resolved-model",
        dimension=768,
        failure_reason=None,
    )

    monkeypatch.setattr(proof, "wait_for_health", lambda *_a, **_k: True)
    monkeypatch.setattr(proof, "watcher_container_running", lambda **_k: True)
    monkeypatch.setattr(proof, "watcher_checkpoint_ready", lambda **_k: True)

    def bootstrap_ok(**_kwargs: Any) -> str:
        events.append("bootstrap")
        return "resolved-model"

    monkeypatch.setattr(
        proof,
        "ensure_embedding_model_bootstrap_if_configured",
        bootstrap_ok,
    )

    def readiness_ok(**_kwargs: Any) -> Any:
        events.append("embedding_readiness")
        return readiness

    monkeypatch.setattr(proof, "run_embedding_readiness", readiness_ok)
    monkeypatch.setattr(
        proof,
        "inspect_kafka_topic_message_count",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("stop_after_readiness")),
    )

    exit_code = proof.main(
        [
            "--repo-root",
            str(_PROJECT_ROOT),
            "--proof-docs-dir",
            str(_LKW_ROOT / ".proof_docs"),
        ]
    )
    assert exit_code != 0
    assert events == ["bootstrap", "embedding_readiness"]

    events.clear()

    def bootstrap_fail(**_kwargs: Any) -> None:
        events.append("bootstrap")
        raise proof.OllamaEmbeddingBootstrapError("embedding_model_pull_failed")

    monkeypatch.setattr(
        proof,
        "ensure_embedding_model_bootstrap_if_configured",
        bootstrap_fail,
    )
    monkeypatch.setattr(
        proof,
        "run_embedding_readiness",
        lambda **_k: (_ for _ in ()).throw(AssertionError("readiness must not run")),
    )

    exit_code = proof.main(
        [
            "--repo-root",
            str(_PROJECT_ROOT),
            "--proof-docs-dir",
            str(_LKW_ROOT / ".proof_docs"),
        ]
    )
    assert exit_code == 1
    assert events == ["bootstrap"]


def test_main_embedding_readiness_failure_gates_kafka_and_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    events: list[str] = []

    monkeypatch.setattr(proof, "wait_for_health", lambda *_a, **_k: True)
    monkeypatch.setattr(proof, "watcher_container_running", lambda **_k: True)
    monkeypatch.setattr(proof, "watcher_checkpoint_ready", lambda **_k: True)
    monkeypatch.setattr(
        proof,
        "ensure_embedding_model_bootstrap_if_configured",
        lambda **_k: "resolved-model",
    )
    monkeypatch.setattr(
        proof,
        "run_embedding_readiness",
        lambda **_k: proof.EmbeddingReadinessResult(
            ready=False,
            provider="ollama",
            model="resolved-model",
            dimension=None,
            failure_reason="embedding_readiness_probe_failed",
        ),
    )
    monkeypatch.setattr(
        proof,
        "inspect_kafka_topic_message_count",
        lambda **_k: (_ for _ in ()).throw(AssertionError("kafka must not run")),
    )
    monkeypatch.setattr(
        proof,
        "create_proof_document",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("file must not run")),
    )

    exit_code = proof.main(
        [
            "--repo-root",
            str(_PROJECT_ROOT),
            "--proof-docs-dir",
            str(_LKW_ROOT / ".proof_docs"),
        ]
    )
    assert exit_code == 1
    assert events == []


def test_main_non_ollama_bootstrap_skips_to_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof_module()
    events: list[str] = []
    readiness = proof.EmbeddingReadinessResult(
        ready=True,
        provider="openai",
        model="text-embedding-3-large",
        dimension=3072,
        failure_reason=None,
    )

    monkeypatch.setattr(proof, "wait_for_health", lambda *_a, **_k: True)
    monkeypatch.setattr(proof, "watcher_container_running", lambda **_k: True)
    monkeypatch.setattr(proof, "watcher_checkpoint_ready", lambda **_k: True)

    def bootstrap_skip(**_kwargs: Any) -> None:
        events.append("bootstrap")

    monkeypatch.setattr(
        proof,
        "ensure_embedding_model_bootstrap_if_configured",
        bootstrap_skip,
    )
    monkeypatch.setattr(
        proof,
        "run_embedding_readiness",
        lambda **_k: (events.append("embedding_readiness"), readiness)[1],
    )
    monkeypatch.setattr(
        proof,
        "inspect_kafka_topic_message_count",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("stop_after_readiness")),
    )

    exit_code = proof.main(
        [
            "--repo-root",
            str(_PROJECT_ROOT),
            "--proof-docs-dir",
            str(_LKW_ROOT / ".proof_docs"),
        ]
    )
    assert exit_code != 0
    assert events == ["bootstrap", "embedding_readiness"]


def test_main_bootstrap_before_warmup_and_failure_skips_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_main_bootstrap_before_readiness_and_failure_skips_readiness(monkeypatch)


def test_main_non_ollama_bootstrap_skips_to_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_main_non_ollama_bootstrap_skips_to_readiness(monkeypatch)


def test_destructive_compose_absent_from_python_proof() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "docker compose down" not in text
    assert "down -v" not in text
    assert "volume rm" not in text
    assert "system prune" not in text
    assert "docker kill" not in text
    assert "docker rm" not in text
