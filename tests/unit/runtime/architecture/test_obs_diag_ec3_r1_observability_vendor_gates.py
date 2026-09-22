# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC3-R1 — observability vendor isolation and qualification evidence gates."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_type_hints

import pytest

from testing_support.obs_diag_observability_vendor_qualification.descriptor import (
    ObservabilityPlatformIsolationEvidence,
    ObservabilityVendorQualificationEvidence,
    PlatformIsolationProofReference,
    VendorQualificationProofReference,
)

from testing_support.obs_diag_observability_vendor_qualification.inventory import (
    OBSERVABILITY_QUALIFIED_PATHS,
    OBSERVABILITY_VENDOR_INVENTORY,
)
from testing_support.obs_diag_observability_vendor_qualification.reconciliation import (
    observability_qualified_path_without_evidence,
    observability_vendor_live_qualified_without_evidence,
    observability_vendor_rows_borrowing_platform_canonical_isolation,
)
from testing_support.obs_diag_provider_qualification.discovery import (
    discover_obs_diag_provider_surfaces,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PARSER_TRACE_EXPORTER = (
    _REPO_ROOT / "intergrax" / "rag" / "document_loaders" / "observability" / "parser_trace_exporter.py"
)
_FORBIDDEN_PARSER_VENDOR_MARKERS = (
    "sentry_sdk",
    "langfuse",
    "/api/public/ingestion",
    'slug == "sentry"',
    'slug == "langfuse"',
)


def test_ec3_observability_vendor_inventory_matches_manifest_discovery() -> None:
    discovered = {
        row.provider_id
        for row in discover_obs_diag_provider_surfaces()
        if row.domain.value == "telemetry"
    }
    inventory_ids = {row.provider_id for row in OBSERVABILITY_VENDOR_INVENTORY}
    assert inventory_ids == discovered


def test_ec3_no_catalog_observability_vendor_live_qualified_without_full_evidence() -> None:
    assert observability_vendor_live_qualified_without_evidence(OBSERVABILITY_VENDOR_INVENTORY) == []


def test_ec3_qualified_paths_live_entries_have_full_evidence() -> None:
    assert observability_qualified_path_without_evidence(OBSERVABILITY_QUALIFIED_PATHS) == []


def test_ec3_vendor_evidence_fields_are_vendor_proof_references_by_type() -> None:
    hints = get_type_hints(ObservabilityVendorQualificationEvidence)
    for field in (
        "normal_delivery",
        "failure_isolation",
        "recovery",
        "canonical_truth_isolation",
        "privacy",
    ):
        assert hints[field] == VendorQualificationProofReference | None
    platform_hints = get_type_hints(ObservabilityPlatformIsolationEvidence)
    assert platform_hints["canonical_truth_isolation"] is PlatformIsolationProofReference


def test_ec3_vendor_rows_do_not_borrow_platform_canonical_isolation_proof() -> None:
    assert (
        observability_vendor_rows_borrowing_platform_canonical_isolation(
            OBSERVABILITY_VENDOR_INVENTORY,
        )
        == []
    )


def test_ec3_parser_trace_exporter_has_no_direct_vendor_delivery() -> None:
    source = _PARSER_TRACE_EXPORTER.read_text(encoding="utf-8")
    for marker in _FORBIDDEN_PARSER_VENDOR_MARKERS:
        assert marker not in source


def test_ec3_parser_trace_exporter_has_no_vendor_sdk_imports() -> None:
    tree = ast.parse(_PARSER_TRACE_EXPORTER.read_text(encoding="utf-8"))
    forbidden_roots = {"sentry_sdk", "httpx", "langfuse"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden_roots
