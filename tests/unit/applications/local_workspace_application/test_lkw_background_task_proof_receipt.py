# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-background-task-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-background-task-proof.bat"
_PUBLIC_PLATFORM_PROOF = _PROJECT_ROOT / "docs" / "project" / "proofs" / "LKW_PLATFORM_PROOF.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_proof_module():
    spec = importlib.util.spec_from_file_location("lkw_background_task_proof", _PROOF_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_background_task_proof_receipt_maps_live_evidence() -> None:
    proof = _load_proof_module()
    receipt = proof.build_background_task_proof_receipt(
        run_id="run-abc",
        correlation_id="corr-abc",
        task_id="task-abc",
        provider="kafka",
        final_status="SUCCEEDED",
        search_results=3,
        marker="MARKER_123",
        collection_id="local_workspace",
        tenant_id="tenant-1",
        kafka_messages=2,
        mongo_express_url="http://127.0.0.1:8086",
    )

    assert receipt.proof_id == "local_workspace:platform_background_task:run-abc"
    assert receipt.proof_kind == "platform_background_task"
    assert receipt.application_id == "local_workspace"
    assert receipt.result.value == "PASS"
    assert receipt.run_id == "run-abc"
    assert receipt.correlation_id == "corr-abc"
    assert receipt.task_id == "task-abc"
    assert receipt.provider_evidence["message_bus_provider"] == "kafka"
    assert receipt.provider_evidence["enqueue_mode"] == "real_provider"
    assert receipt.provider_evidence["worker_execution"] == "asynchronous"
    assert receipt.provider_evidence["task_status"] == "SUCCEEDED"
    assert receipt.provider_evidence["task_result_available"] is True
    assert receipt.provider_evidence["handler_resolved"] is True
    assert receipt.provider_evidence["worker_runtime_received"] is True
    assert receipt.provider_evidence["kafka_topic_messages"] == 2
    assert receipt.provider_evidence["kafka_topic_inspection_available"] is True
    assert receipt.provider_evidence["kafka_topics"] == [
        "intergrax.tasks",
        "intergrax.task-events",
        "intergrax.task-status",
        "intergrax.task-results",
    ]
    assert receipt.domain_evidence["task_name"] == "lkw.background_ingest.v1"
    assert receipt.domain_evidence["index_ingested"] == 1
    assert receipt.domain_evidence["search_results"] == 3
    assert receipt.domain_evidence["evidence_marker_found"] is True
    assert receipt.domain_evidence["marker"] == "MARKER_123"
    assert receipt.domain_evidence["collection_id"] == "local_workspace"
    assert receipt.domain_evidence["tenant_id"] == "tenant-1"
    assert receipt.guardrails["mock_queue"] is False
    assert receipt.guardrails["inmemory_bypass"] is False
    assert receipt.guardrails["direct_handler_call"] is False
    assert receipt.guardrails["direct_indexer_call"] is False
    assert receipt.guardrails["direct_mongodb_write"] is False
    assert receipt.guardrails["direct_pymongo_from_lkw"] is False
    assert receipt.guardrails["markdown_source_of_truth"] is False
    assert receipt.metadata["proof_runner"] == "run-lkw-background-task-proof.py"
    assert receipt.metadata["receipt_task"] == "PROOF-RECEIPTS-1E"
    assert receipt.metadata["mongo_express_url"] == "http://127.0.0.1:8086"
    assert receipt.metadata["recorded_from_live_run"] is True


def test_build_background_task_proof_receipt_marks_kafka_inspection_unavailable() -> None:
    proof = _load_proof_module()
    receipt = proof.build_background_task_proof_receipt(
        run_id="run-abc",
        correlation_id="corr-abc",
        task_id="task-abc",
        provider="kafka",
        final_status="SUCCEEDED",
        search_results=1,
        marker="MARKER_123",
        collection_id="local_workspace",
        tenant_id="tenant-1",
        kafka_messages=-1,
    )

    assert receipt.provider_evidence["kafka_topic_messages"] is None
    assert receipt.provider_evidence["kafka_topic_inspection_available"] is False


def test_build_background_task_proof_receipt_has_no_credentials() -> None:
    proof = _load_proof_module()
    receipt = proof.build_background_task_proof_receipt(
        run_id="run-abc",
        correlation_id="corr-abc",
        task_id="task-abc",
        provider="kafka",
        final_status="SUCCEEDED",
        search_results=1,
        marker="MARKER_123",
        collection_id="local_workspace",
        tenant_id="tenant-1",
        kafka_messages=1,
    )
    serialized = receipt.model_dump_json()

    assert "mongodb://" not in serialized
    assert "authSource" not in serialized
    assert "intergrax-local-dev-only" not in serialized


def test_background_task_bat_runner_composes_kafka_and_mongodb_overlays() -> None:
    text = _read(_PROOF_BAT)
    assert "docker-compose.yml" in text
    assert "docker-compose.kafka.yml" in text
    assert "docker-compose.mongodb.yml" in text
    assert "lkw-mongodb" in text
    assert "lkw-mongo-express" in text
    assert "lkw-background-worker" in text
    assert "lkw-kafka-ui" in text
    assert "mongodb_container_healthy=true" in text
    assert "mongo_express_available=true" in text
    assert "--project applications/local_workspace_application" in text
    assert "INTERGRAX_MONGODB_URI=" in text
    assert "INTERGRAX_MONGODB_DATABASE=" in text
    assert "INTERGRAX_MONGODB_COLLECTION=" in text


def test_background_task_proof_script_uses_platform_receipt_recording() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "ProofReceipt" in text
    assert "record_and_verify_proof_receipt" in text
    assert "create_mongodb_integration" in text
    assert "MongoDBDocumentStoreIntegration" in text
    assert "proof_receipt_recorded=true" in text
    assert "proof_receipt_verified=true" in text
    assert "proof_receipt_recording_failed" in text
    assert "DocumentRecord" not in text
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    assert pymongo_import not in text
    assert pymongo_from not in text


def test_background_task_proof_pass_prints_after_receipt_verification() -> None:
    text = _read(_PROOF_SCRIPT)
    receipt_record_index = text.index("record_background_task_proof_receipt")
    pass_output_index = text.index('print("proof_result=PASS")')
    assert receipt_record_index < pass_output_index


def test_background_task_proof_failure_reports_receipt_recording_failure() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "failure_reason=proof_receipt_recording_failed" in text
    assert "proof_workload_result=PASS" in text
    assert "proof_receipt_verified=false" in text


def test_public_reviewer_document_contains_step_9_and_receipt_fields() -> None:
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "## Step 9 — Inspect the structured ProofReceipt in Mongo Express" in text
    assert "proof_receipt_recorded=true" in text
    assert "proof_receipt_verified=true" in text
    assert "proof_receipt_query_verified=true" in text
    assert "markdown_source_of_truth=false" in text
    assert "Mongo Express   http://127.0.0.1:8086" in text
    step_8_block = text.split("## Step 8", 1)[1].split("## Step 9", 1)[0]
    assert "Latest recorded live result" not in step_8_block
    assert "Use this document as the source of truth." not in text
    assert "Structured ProofReceipt documents persisted through the platform DocumentStore" in text


def test_no_direct_pymongo_import_in_lkw_application_python_files() -> None:
    app_root = _LKW_ROOT
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    for path in app_root.rglob("*.py"):
        text = _read(path)
        assert pymongo_import not in text
        assert pymongo_from not in text


def test_background_task_proof_script_has_no_direct_mongodb_collection_operations() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "insert_one" not in text
    assert "update_one" not in text
    assert "delete_one" not in text
    assert "MongoClient" not in text
    assert re.search(r"\.collection\(", text) is None
