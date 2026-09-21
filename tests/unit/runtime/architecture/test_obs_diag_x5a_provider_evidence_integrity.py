# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5A — qualification evidence integrity, anti-drift, and lifecycle gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderClass,
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderDomain,
    ObsDiagProviderSupportStatus,
)
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
    obs_diag_telemetry_vendor_falsely_qualified,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_x5a]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_MANIFEST_DIR = (
    _REPO_ROOT
    / "testing_support"
    / "obs_diag_provider_qualification"
    / "fixture_manifests"
    / "unclassified_fake"
)
_QUALIFICATION_TEST_ROOT = _REPO_ROOT / "tests" / "integration" / "providers" / "obs_diag"


def test_obs_diag_x5a_discovered_equals_classified_external_providers() -> None:
    discovered = discover_obs_diag_provider_surfaces()
    missing, stale = obs_diag_anti_drift_delta(
        discovered=discovered,
        classifications=OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    )
    assert missing == []
    assert stale == []


def test_obs_diag_x5a_new_manifest_without_classification_fails_anti_drift() -> None:
    discovered = discover_obs_diag_provider_surfaces(
        supplemental_manifest_dirs=(_FIXTURE_MANIFEST_DIR,),
    )
    missing, _stale = obs_diag_anti_drift_delta(
        discovered=discovered,
        classifications=OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    )
    assert "x5a_unclassified_fake" in missing


def test_obs_diag_x5a_stale_classification_detected() -> None:
    discovered = discover_obs_diag_provider_surfaces()
    stale_row = ObsDiagProviderQualificationDescriptor(
        provider_id="stale-x5a-provider",
        domain=ObsDiagProviderDomain.PERSISTENCE,
        provider_class=ObsDiagProviderClass.EXTERNAL_VENDOR,
        contract="stale",
        integration_status="STABLE",
        adapter_exists=True,
        live_proof_module=None,
        failure_recovery_proof_module=None,
        declared_status=ObsDiagProviderSupportStatus.ADAPTER_ONLY,
        discovery_source="stale",
    )
    classifications = (*OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS, stale_row)
    _missing, stale = obs_diag_anti_drift_delta(
        discovered=discovered,
        classifications=classifications,
    )
    assert "stale-x5a-provider" in stale


def test_obs_diag_x5a_qualified_external_requires_proof_modules() -> None:
    violations = obs_diag_qualified_external_without_proof(OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS)
    assert violations == []


def test_obs_diag_x5a_qualified_external_without_proof_detected() -> None:
    bad = ObsDiagProviderQualificationDescriptor(
        provider_id="bad-qualified",
        domain=ObsDiagProviderDomain.TRANSPORT,
        provider_class=ObsDiagProviderClass.EXTERNAL_VENDOR,
        contract="bad",
        integration_status="STABLE",
        adapter_exists=True,
        live_proof_module=None,
        failure_recovery_proof_module=None,
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
        discovery_source="test",
    )
    violations = obs_diag_qualified_external_without_proof((bad,))
    assert violations == ["bad-qualified"]


def test_obs_diag_x5a_no_external_telemetry_vendor_qualified() -> None:
    violations = obs_diag_telemetry_vendor_falsely_qualified(
        OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    )
    assert violations == []


def test_obs_diag_x5a_no_private_consumer_reach_through_in_qualification_tests() -> None:
    violations: list[str] = []
    for path in _QUALIFICATION_TEST_ROOT.glob("test_x5*.py"):
        text = path.read_text(encoding="utf-8")
        if "._consumer" in text:
            violations.append(path.name)
    assert violations == []


def test_obs_diag_x5a_platform_export_semantics_qualified_separately_from_otel_catalog() -> None:
    by_id = {row.provider_id: row for row in OBS_DIAG_X5_PROVIDER_INVENTORY}
    export_semantics = by_id["observability_export_semantics"]
    otel_catalog = by_id["otel"]
    assert export_semantics.declared_status == ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED
    assert export_semantics.provider_class == ObsDiagProviderClass.PLATFORM_EXPORT_SEMANTICS
    assert otel_catalog.declared_status == ObsDiagProviderSupportStatus.ADAPTER_ONLY
    assert otel_catalog.provider_class == ObsDiagProviderClass.EXTERNAL_VENDOR
