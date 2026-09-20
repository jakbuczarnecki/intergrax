# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 / X5A — vendor boundary, inventory integrity, and config-leak gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from testing_support.obs_diag_provider_qualification.discovery import (
    discover_obs_diag_provider_surfaces,
)
from testing_support.obs_diag_provider_qualification.inventory import (
    OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    OBS_DIAG_X5_PROVIDER_INVENTORY,
)
from testing_support.obs_diag_provider_qualification.reconciliation import (
    obs_diag_anti_drift_delta,
    obs_diag_qualified_external_without_proof,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_x5]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIAG_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_OBS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts"
_APP_CONTRACTS = _REPO_ROOT / "intergrax" / "applications" / "contracts"

_FORBIDDEN_VENDOR_PREFIXES = (
    "confluent_kafka",
    "pymongo",
    "motor",
    "boto3",
    "redis",
    "opentelemetry",
    "datadog",
    "sentry_sdk",
)

_VENDOR_CONFIG_PATTERNS = re.compile(
    r"\b(mongo_uri|datadog_api_key|sentry_dsn|kafka_bootstrap_servers)\s*:",
    re.IGNORECASE,
)


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _collect_vendor_import_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == prefix or alias.name.startswith(f"{prefix}.")
                        for prefix in _FORBIDDEN_VENDOR_PREFIXES
                    ):
                        violations.append(f"{rel}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if any(
                    node.module == prefix or node.module.startswith(f"{prefix}.")
                    for prefix in _FORBIDDEN_VENDOR_PREFIXES
                ):
                    violations.append(f"{rel}:{node.lineno}:{node.module}")
    return violations


def _proof_path_exists(module: str) -> bool:
    path = _REPO_ROOT / module
    if path.suffix == ".py":
        return path.is_file()
    return path.exists()


def test_obs_diag_x5_inventory_proof_modules_exist() -> None:
    missing: list[str] = []
    for row in OBS_DIAG_X5_PROVIDER_INVENTORY:
        for module in (row.live_proof_module, row.failure_recovery_proof_module):
            if module is None:
                continue
            if not _proof_path_exists(module):
                missing.append(f"{row.provider_id}:{module}")
    assert missing == []


def test_obs_diag_x5_manifest_discovery_matches_external_classifications() -> None:
    discovered = discover_obs_diag_provider_surfaces()
    missing, stale = obs_diag_anti_drift_delta(
        discovered=discovered,
        classifications=OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    )
    assert missing == []
    assert stale == []


def test_obs_diag_x5_qualified_external_providers_have_proof_linkage() -> None:
    assert obs_diag_qualified_external_without_proof(OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS) == []


def test_obs_diag_x5_core_has_no_direct_vendor_imports() -> None:
    """Semantic OBS/DIAG core — export adapter trees under ``observability/exporters/`` excluded."""
    paths: list[Path] = []
    paths.extend(_python_files(_DIAG_ROOT))
    if _OBS_ROOT.is_dir():
        for path in _python_files(_OBS_ROOT):
            rel = path.relative_to(_OBS_ROOT).as_posix()
            if rel.startswith("exporters/"):
                continue
            paths.append(path)
    for root in (_CONTRACTS_ROOT, _APP_CONTRACTS):
        if root.is_dir():
            paths.extend(_python_files(root))
    violations = _collect_vendor_import_violations(paths)
    assert violations == []


def test_obs_diag_x5_contracts_have_no_vendor_specific_config_fields() -> None:
    violations: list[str] = []
    for root in (_CONTRACTS_ROOT, _APP_CONTRACTS):
        if not root.is_dir():
            continue
        for path in _python_files(root):
            if _VENDOR_CONFIG_PATTERNS.search(path.read_text(encoding="utf-8")):
                violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert violations == []


def test_obs_diag_x5_no_provider_branching_in_diagnostic_orchestrator() -> None:
    orchestrator = _DIAG_ROOT / "diagnostic_orchestrator.py"
    text = orchestrator.read_text(encoding="utf-8")
    assert 'if provider == "' not in text
    assert "elif provider ==" not in text
