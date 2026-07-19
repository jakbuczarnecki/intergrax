# © Artur Czarnecki. All rights reserved.

"""Unit guardrails for LKW.7C2 file-watcher E2E ProofReceipt mapping."""

from __future__ import annotations

import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.bat"
_PUBLIC_PLATFORM_PROOF = (
    _PROJECT_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
)
_VERIFICATION_DOC = _LKW_ROOT / "docs" / "LKW_7_FILE_WATCHER_VERIFICATION.md"
_ARCHITECTURE = _LKW_ROOT / "docs" / "ARCHITECTURE.md"
_IMPLEMENTATION_PLAN = _LKW_ROOT / "docs" / "IMPLEMENTATION_PLAN.md"
_RUNTIME_ARCH = _PROJECT_ROOT / "docs" / "intergrax_runtime_architecture.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_proof_module():
    spec = importlib.util.spec_from_file_location(
        "lkw_file_watcher_e2e_proof_receipt", _PROOF_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_workload_evidence() -> dict[str, object]:
    return {
        "marker": "LKW_FILE_WATCHER_E2E_20260719T120000Z_ab12cd34",
        "proof_filename": "lkw_file_watcher_e2e_20260719T120000Z_ab12cd34.txt",
        "container_source_path": (
            "/data/user_docs/lkw_file_watcher_e2e_20260719T120000Z_ab12cd34.txt"
        ),
        "task_count_before_file": 3,
        "task_count_after_file": 4,
        "search_results_before_restart": 1,
        "task_count_before_restart": 4,
        "task_count_after_restart": 4,
        "search_results_after_restart": 1,
        "watcher_restored_after_restart": True,
        "watcher_final_checkpoint_saved": True,
        "source_file_modified_after_index": False,
        "embedding_warmup_completed": True,
        "reviewer_rerun_required": False,
    }


def test_build_file_watcher_proof_id_valid_and_blank() -> None:
    proof = _load_proof_module()
    run_id = "LKW_FILE_WATCHER_E2E_20260719T120000Z_ab12cd34"
    assert (
        proof.build_file_watcher_proof_id(run_id)
        == f"local_workspace:file_watcher_persistent_search:{run_id}"
    )
    with pytest.raises(ValueError):
        proof.build_file_watcher_proof_id("")
    with pytest.raises(ValueError):
        proof.build_file_watcher_proof_id("   ")


def test_build_file_watcher_e2e_proof_receipt_maps_live_evidence() -> None:
    proof = _load_proof_module()
    run_id = "LKW_FILE_WATCHER_E2E_20260719T120000Z_ab12cd34"
    evidence = _sample_workload_evidence()
    receipt = proof.build_file_watcher_e2e_proof_receipt(
        run_id=run_id,
        workload_evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )

    assert receipt.schema_version == "intergrax.proof_receipt.v1"
    assert (
        receipt.proof_id == f"local_workspace:file_watcher_persistent_search:{run_id}"
    )
    assert receipt.proof_kind == "file_watcher_persistent_search"
    assert receipt.application_id == "local_workspace"
    assert receipt.result.value == "PASS"
    assert receipt.run_id == run_id
    assert receipt.task_id is None
    assert receipt.correlation_id is None

    provider = receipt.provider_evidence
    assert provider["message_bus_provider"] == "kafka"
    assert provider["worker_execution"] == "asynchronous"
    assert provider["enqueue_trigger"] == "filesystem_create"
    assert provider["watcher_process"] == "foreground_sidecar"
    assert provider["watcher_checkpoint_store"] == "json_file"
    assert provider["checkpoint_restore_verified"] is True
    assert provider["watcher_final_checkpoint_saved"] is True
    assert provider["vector_store_provider"] == "qdrant"
    assert provider["persistent_index"] is True
    assert provider["document_store_provider"] == "mongodb"
    assert provider["kafka_task_topic"] == "intergrax.tasks"
    assert provider["task_count_before_file"] == 3
    assert provider["task_count_after_file"] == 4
    assert provider["task_topic_increased"] is True
    assert provider["task_count_before_restart"] == 4
    assert provider["task_count_after_restart"] == 4
    assert provider["duplicate_enqueue_after_restart"] is False
    assert provider["restart_services"] == [
        "lkw-file-watcher",
        "lkw-background-worker",
        "local_workspace",
        "qdrant",
    ]

    domain = receipt.domain_evidence
    assert domain["trigger"] == "filesystem_create"
    assert domain["tenant_id"] == "lkw-file-watcher-e2e"
    assert domain["workspace_id"] == "lkw-file-watcher-e2e"
    assert domain["collection_id"] == "lkw-file-watcher-e2e"
    assert domain["marker"] == run_id
    assert domain["proof_filename"] == evidence["proof_filename"]
    assert domain["container_source_path"] == evidence["container_source_path"]
    assert domain["embedding_warmup_completed"] is True
    assert domain["reviewer_rerun_required"] is False
    assert domain["watcher_checkpoint_ready"] is True
    assert domain["watcher_restored_after_restart"] is True
    assert domain["search_results_before_restart"] == 1
    assert domain["source_ref_found_before_restart"] is True
    assert domain["restart_mode"] == "non_destructive"
    assert domain["volumes_removed"] is False
    assert domain["source_file_modified_after_index"] is False
    assert domain["reindexed_after_restart"] is False
    assert domain["search_results_after_restart"] == 1
    assert domain["source_ref_found_after_restart"] is True

    guardrails = receipt.guardrails
    assert guardrails["manual_index_command"] is False
    assert guardrails["direct_enqueue"] is False
    assert guardrails["direct_handler_call"] is False
    assert guardrails["direct_indexer_call"] is False
    assert guardrails["direct_ingest_call"] is False
    assert guardrails["mock_queue"] is False
    assert guardrails["inmemory_bypass"] is False
    assert guardrails["direct_qdrant_write"] is False
    assert guardrails["direct_mongodb_write"] is False
    assert guardrails["direct_pymongo_from_lkw"] is False
    assert guardrails["markdown_source_of_truth"] is False
    assert guardrails["manual_evidence_injection"] is False

    metadata = receipt.metadata
    assert metadata["proof_runner"] == "run-lkw-file-watcher-e2e-proof.py"
    assert metadata["receipt_task"] == "LKW.7C2"
    assert metadata["mongo_express_url"] == "http://127.0.0.1:8086"
    assert metadata["recorded_from_live_run"] is True
    assert metadata["reviewer_guide"] == "docs/public-adoption/LKW_PLATFORM_PROOF.md"
    assert (
        metadata["verification_document"]
        == "applications/local_workspace_application/docs/"
        "LKW_7_FILE_WATCHER_VERIFICATION.md"
    )


def test_file_watcher_receipt_has_no_credentials_or_content() -> None:
    proof = _load_proof_module()
    receipt = proof.build_file_watcher_e2e_proof_receipt(
        run_id="LKW_FILE_WATCHER_E2E_20260719T120000Z_ab12cd34",
        workload_evidence=_sample_workload_evidence(),
    )
    serialized = receipt.model_dump_json()
    assert "mongodb://" not in serialized
    assert "authSource" not in serialized
    assert "intergrax-local-dev-only" not in serialized
    assert "password" not in serialized.lower()
    assert "redis://" not in serialized
    assert "No manual indexing" not in serialized
    assert "embedding" not in serialized.lower() or "embedding_warmup" in serialized
    assert "D:/" not in serialized
    assert "C:\\" not in serialized


def test_file_watcher_proof_script_uses_platform_receipt_recording() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "ProofReceipt" in text
    assert "record_and_verify_proof_receipt" in text
    assert "create_mongodb_integration" in text
    assert "MongoDBDocumentStoreIntegration" in text
    assert "proof_receipt_recorded=true" in text
    assert "proof_receipt_verified=true" in text
    assert "proof_receipt_query_verified=true" in text
    assert "DocumentRecord" not in text
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    assert pymongo_import not in text
    assert pymongo_from not in text
    assert "MongoClient" not in text
    assert "insert_one" not in text
    assert "update_one" not in text
    assert "delete_one" not in text
    assert re.search(r"\.collection\(", text) is None


def test_file_watcher_pass_prints_after_receipt_verification() -> None:
    text = _read(_PROOF_SCRIPT)
    receipt_record_index = text.index("record_file_watcher_e2e_proof_receipt")
    pass_output_index = text.index("print(format_pass_output(evidence))")
    assert receipt_record_index < pass_output_index
    fn_start = text.index("def build_pass_evidence(")
    fn_end = text.index("\ndef ", fn_start + 1)
    body = text[fn_start:fn_end]
    assert "verified_receipt" in body


def test_file_watcher_receipt_failure_reports_safe_fields() -> None:
    proof = _load_proof_module()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = proof.fail_receipt_recording(RuntimeError("secret mongodb://x"))
    assert code == 1
    output = buffer.getvalue()
    assert "failure_reason=proof_receipt_recording_failed" in output
    assert "proof_workload_result=PASS" in output
    assert "proof_receipt_recorded=false" in output
    assert "proof_receipt_verified=false" in output
    assert "receipt_error_type=RuntimeError" in output
    assert "mongodb://" not in output
    assert "secret" not in output


def test_file_watcher_bat_runner_composes_mongodb_overlay() -> None:
    text = _read(_PROOF_BAT)
    assert "docker-compose.mongodb.yml" in text
    assert "lkw-mongodb" in text
    assert "lkw-mongo-express" in text
    assert "mongodb_container_healthy=true" in text
    assert "mongo_express_available=true" in text
    assert "lkw-kafka-ui" in text
    assert "kafka_ui=ok" in text
    assert "--extra integrations-mongodb" in text
    assert "INTERGRAX_MONGODB_URI=" in text
    assert "INTERGRAX_MONGODB_DATABASE=" in text
    assert "INTERGRAX_MONGODB_COLLECTION=" in text
    assert "--mongodb-compose" in text
    assert "--mongo-express" in text
    assert "echo %INTERGRAX_MONGODB_URI%" not in text


def test_public_reviewer_document_contains_steps_14_and_15() -> None:
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "## Step 14 — Run the File Watcher E2E proof" in text
    assert "## Step 15 — Inspect the File Watcher ProofReceipt in Mongo Express" in text
    assert "proof_kind = file_watcher_persistent_search" in text
    assert "row_key =\n  proof/file_watcher_persistent_search/<run_id>" in text or (
        "proof/file_watcher_persistent_search/<run_id>" in text
    )
    assert "ProofReceipt is authoritative" in text or (
        "MongoDB-backed ProofReceipt is the authoritative" in text
    )
    shortcut = text.split("## Reviewer shortcut", 1)[1]
    assert "run-lkw-file-watcher-e2e-proof.bat" in shortcut
    assert "proof_kind=file_watcher_persistent_search" in shortcut


def test_verification_document_authority_and_boundaries() -> None:
    text = _read(_VERIFICATION_DOC)
    assert "LKW.7 File Watcher E2E Verification" in text
    assert "Status: Closed" in text
    assert "run-lkw-file-watcher-e2e-proof.bat" in text
    assert "authoritative" in text.lower()
    assert "Known boundaries" in text or "known boundaries" in text.lower()
    assert "Non-authoritative reviewer convenience" in text
    assert "Markdown is the source of truth" not in text
    assert "markdown is the source of truth" not in text.lower()


def test_plan_status_documents_agree_lkw7_closed() -> None:
    architecture = _read(_ARCHITECTURE)
    plan = _read(_IMPLEMENTATION_PLAN)
    runtime = _read(_RUNTIME_ARCH)
    for text in (architecture, plan, runtime):
        assert re.search(
            r"\|\s*(?:\*\*)?LKW\.7(?:\*\*)?\s*\|[^\n]*\*\*Closed\*\*",
            text,
        ), "LKW.7 Closed missing"
        assert re.search(
            r"\|\s*(?:\*\*)?LKW\.7C(?:\*\*)?\s*\|[^\n]*\*\*Closed\*\*",
            text,
        ), "LKW.7C Closed missing"
        assert re.search(
            r"\|\s*(?:\*\*)?LKW\.7C1(?:\*\*)?\s*\|[^\n]*\*\*Done\*\*",
            text,
        ), "LKW.7C1 Done missing"
        assert re.search(
            r"\|\s*(?:\*\*)?LKW\.7C2(?:\*\*)?\s*\|[^\n]*\*\*Done\*\*",
            text,
        ), "LKW.7C2 Done missing"
        assert "LKW.7C2 Planned" not in text
        assert "LKW.7C2 next" not in text
        assert "LKW.7C In progress" not in text
        assert "LKW.7 — **In progress**" not in text
        assert "LKW.7 | File watcher + incremental index | **In progress**" not in text
