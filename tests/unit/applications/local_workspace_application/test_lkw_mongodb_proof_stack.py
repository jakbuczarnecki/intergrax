# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_DOCKER_DIR = _LKW_ROOT / "docker"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_MONGODB_OVERLAY = _DOCKER_DIR / "docker-compose.mongodb.yml"
_VERIFY_SCRIPT = _SCRIPTS_DIR / "verify_lkw_mongodb_stack.py"
_RUNNER_BAT = _SCRIPTS_DIR / "run-lkw-mongodb-proof-stack.bat"
_PUBLIC_PLATFORM_PROOF = _PROJECT_ROOT / "docs" / "project" / "proofs" / "LKW_PLATFORM_PROOF.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_mongodb_overlay_exists() -> None:
    assert _MONGODB_OVERLAY.exists()


def test_mongodb_overlay_defines_services_and_volume() -> None:
    text = _read(_MONGODB_OVERLAY)
    assert "lkw-mongodb:" in text
    assert "lkw-mongo-express:" in text
    assert "lkw_mongodb_data:" in text
    assert "mongo:7.0.14" in text
    assert "mongo-express:1.0.2-20" in text


def test_mongodb_overlay_configures_local_workspace_provider_env() -> None:
    text = _read(_MONGODB_OVERLAY)
    assert "INTERGRAX_MONGODB_URI:" in text
    assert "INTERGRAX_MONGODB_DATABASE:" in text
    assert "INTERGRAX_MONGODB_COLLECTION:" in text
    assert "lkw-mongodb:27017" in text
    assert "authSource=admin" in text
    assert "${LKW_MONGODB_DATABASE:-intergrax_proofs}" in text
    assert "${LKW_MONGODB_COLLECTION:-proof_receipts}" in text


def test_local_workspace_depends_on_healthy_mongodb() -> None:
    text = _read(_MONGODB_OVERLAY)
    assert re.search(
        r"local_workspace:[\s\S]*depends_on:[\s\S]*lkw-mongodb:[\s\S]*condition:\s*service_healthy",
        text,
    )


def test_mongo_express_depends_on_mongodb_and_exposes_reviewer_port() -> None:
    text = _read(_MONGODB_OVERLAY)
    assert re.search(
        r"lkw-mongo-express:[\s\S]*depends_on:[\s\S]*lkw-mongodb:[\s\S]*condition:\s*service_healthy",
        text,
    )
    assert "${LKW_MONGO_EXPRESS_PORT:-8086}:8081" in text
    assert "local_workspace" not in text.split("lkw-mongo-express:", 1)[1].split("volumes:", 1)[0]


def test_mongodb_overlay_does_not_make_local_workspace_depend_on_mongo_express() -> None:
    local_workspace_block = _read(_MONGODB_OVERLAY).split("local_workspace:", 1)[1].split(
        "lkw-mongodb:", 1
    )[0]
    assert "lkw-mongo-express" not in local_workspace_block


def test_no_direct_pymongo_import_in_lkw_application_python_files() -> None:
    app_root = _LKW_ROOT
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    for path in app_root.rglob("*.py"):
        if "scripts" in path.parts and path.name.startswith("verify_lkw_mongodb"):
            continue
        text = _read(path)
        assert pymongo_import not in text
        assert pymongo_from not in text


def test_verify_script_uses_platform_integration_path() -> None:
    assert _VERIFY_SCRIPT.exists()
    text = _read(_VERIFY_SCRIPT)
    assert "create_mongodb_integration" in text
    assert "as_document_store" in text
    assert "MongoDBDocumentStoreIntegration" in text
    assert "proof_receipt_recording" in text
    assert "ProofReceiptStore" not in text
    assert "ProofReceipt" not in text
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    assert pymongo_import not in text
    assert pymongo_from not in text
    assert 'SMOKE_PARTITION_KEY = "platform_smoke"' in text


def test_runner_does_not_use_mongosh_writes_or_proof_receipt_store() -> None:
    assert _RUNNER_BAT.exists()
    text = _read(_RUNNER_BAT)
    lowered = text.lower()
    assert "proof_receipt_recording=false" in lowered
    assert "verify_lkw_mongodb_stack.py" in text
    assert "docker-compose.mongodb.yml" in text
    assert "mongosh" not in lowered or "insert" not in lowered
    assert "ProofReceiptStore" not in text
    assert "restart lkw-mongodb" in lowered


def test_public_step_9_documents_receipt_inspection() -> None:
    assert _PUBLIC_PLATFORM_PROOF.exists()
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "## Step 9 — Inspect the structured ProofReceipt in Mongo Express" in text
    assert "proof_receipts/local_workspace" in text


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_lkw_mongodb_stack", _VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_module_smoke_constants_are_infrastructure_only() -> None:
    module = _load_verify_module()
    assert module.SMOKE_PARTITION_KEY == "platform_smoke"
    assert module.SMOKE_ROW_KEY == "mongodb_document_store"
    assert module.SMOKE_DATA["proof_kind"] == "infrastructure_connectivity"
    assert module.SMOKE_DATA["task"] == "PROOF-RECEIPTS-1D"
