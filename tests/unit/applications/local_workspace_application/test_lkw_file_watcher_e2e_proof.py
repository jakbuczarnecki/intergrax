# © Artur Czarnecki. All rights reserved.

"""Static/unit guardrails for LKW.7C1 watcher-triggered E2E proof."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

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
)

_FORBIDDEN_RECEIPT_PATTERNS = (
    "ProofReceipt",
    "ProofReceiptStore",
    "record_and_verify_proof_receipt",
    "create_mongodb_integration",
    "MongoDBDocumentStoreIntegration",
    "pymongo",
    "MongoClient",
    "mongo_express",
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
    "proof_receipt_recorded",
    "proof_receipt_task",
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


def test_proof_script_receipt_boundary() -> None:
    text = _read(_PROOF_SCRIPT)
    compose = _read(_WATCHER_COMPOSE)
    for pattern in _FORBIDDEN_RECEIPT_PATTERNS:
        assert pattern not in text, pattern
        assert pattern not in compose, pattern


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
    )
    joined = " ".join(command)
    assert command[:2] == ["docker", "compose"]
    assert "restart" in command
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


def test_bat_runner_ordering_and_services() -> None:
    text = _read(_PROOF_BAT)
    reset_idx = text.lower().index("resetting dedicated watcher proof checkpoint state")
    validate_idx = text.lower().index("validating docker compose merge")
    start_idx = text.lower().index("starting watcher e2e proof stack")
    health_idx = text.lower().index("waiting for lkw health")
    watcher_ready_idx = text.lower().index("waiting for watcher baseline checkpoint")
    invoke_idx = text.lower().index("invoking python watcher e2e proof workload")
    assert reset_idx < validate_idx < start_idx < health_idx < watcher_ready_idx
    assert watcher_ready_idx < invoke_idx
    assert "watcher_container_running=true" in text
    assert "watcher_checkpoint_ready=true" in text
    for service in (
        "local_workspace",
        "lkw-background-worker",
        "lkw-file-watcher",
        "lkw-kafka",
        "lkw-kafka-topics",
        "lkw-redis",
        "qdrant",
        "ollama",
    ):
        assert service in text
    assert "lkw-mongodb" not in text
    assert "lkw-mongo-express" not in text
    assert "docker compose down" not in text
    assert "down -v" not in text
    assert "volume rm" not in text
    assert "system prune" not in text


def test_pass_output_fields() -> None:
    proof = _load_proof_module()
    evidence = proof.build_pass_evidence(
        marker="MARKER",
        filename="file.txt",
        container_source_path="/data/user_docs/file.txt",
        task_count_before_file=1,
        task_count_after_file=2,
        search_results_before_restart=1,
        task_count_before_restart=2,
        task_count_after_restart=2,
        search_results_after_restart=1,
    )
    for field in _REQUIRED_PASS_FIELDS:
        assert field in evidence
    assert evidence["proof_receipt_recorded"] is False
    assert evidence["proof_receipt_task"] == "LKW.7C2"
    rendered = proof.format_pass_output(evidence)
    assert "proof_result=PASS" in rendered
    assert "proof_receipt_recorded=false" in rendered
    assert "proof_receipt_task=LKW.7C2" in rendered
    assert "mongodb" not in rendered.lower()
    assert "redis://" not in rendered.lower()
    assert "password" not in rendered.lower()
    assert "embedding" not in rendered.lower()
    assert "This file was created" not in rendered


def test_destructive_compose_absent_from_python_proof() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "docker compose down" not in text
    assert "down -v" not in text
    assert "volume rm" not in text
    assert "system prune" not in text
