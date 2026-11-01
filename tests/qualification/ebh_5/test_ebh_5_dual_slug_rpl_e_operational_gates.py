# © Artur Czarnecki. All rights reserved.

"""RPL-E operational probe fail-closed — adversarial negatives (F07)."""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import HealthStatus
from tests.qualification.ebh_5._dual_slug_rpl_e_operational import (
    assert_rpl_e_category_operational_surface,
    assert_strict_qualification_probe,
    operational_evidence_policy,
)
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import INTEGRATION_DUAL_SLUG_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


class _ProbeAlwaysFail:
    def health(self) -> None:
        raise RuntimeError("probe failed")

    def ping(self) -> None:
        raise ValueError("ping failed")


class _Unhealthy:
    def health(self) -> HealthStatus:
        return HealthStatus(slug="x", healthy=False, detail="down")


class _Healthy:
    def health(self) -> HealthStatus:
        return HealthStatus(slug="x", healthy=True, detail="ok")


class _BackendA:
    schema_id = "message_bus_integration_contract.v1"
    integration_kind = "message_bus"
    capabilities = ()
    provider_id = "a"

    def publish(self, topic: str, payload: bytes) -> None:
        return None

    def close(self) -> None:
        return None


class _BackendBMissingPublish:
    schema_id = "message_bus_integration_contract.v1"
    integration_kind = "message_bus"
    capabilities = ()
    provider_id = "b"

    def close(self) -> None:
        return None


def test_rpl_e_strict_probe_all_methods_raise_fails() -> None:
    with pytest.raises(RuntimeError, match="probe failed"):
        assert_strict_qualification_probe(_ProbeAlwaysFail(), "health", concern_id="probe-test")


def test_rpl_e_strict_probe_missing_method_fails() -> None:
    with pytest.raises(AssertionError, match="not available"):
        assert_strict_qualification_probe(object(), "health", concern_id="probe-test")


def test_rpl_e_strict_probe_unhealthy_status_fails() -> None:
    with pytest.raises(AssertionError, match="unhealthy"):
        assert_strict_qualification_probe(_Unhealthy(), "health", concern_id="probe-test")


def test_rpl_e_strict_probe_healthy_passes() -> None:
    assert_strict_qualification_probe(_Healthy(), "health", concern_id="probe-test")


def test_rpl_e_capabilities_mismatch_fails() -> None:
    entry = next(e for e in INTEGRATION_DUAL_SLUG_CATALOG if e.concern_id == "integration.message_bus")
    a = _BackendA()
    b = _BackendBMissingPublish()
    a.capabilities = ("connect", "write")
    b.capabilities = ("connect",)
    with pytest.raises(AssertionError, match="capabilities"):
        assert_rpl_e_category_operational_surface(entry, a, b)


def test_rpl_e_required_operation_missing_on_one_backend_fails() -> None:
    entry = next(e for e in INTEGRATION_DUAL_SLUG_CATALOG if e.concern_id == "integration.message_bus")
    with pytest.raises(AssertionError, match="publish"):
        assert_rpl_e_category_operational_surface(entry, _BackendA(), _BackendBMissingPublish())


def test_rpl_e_structural_policy_no_connect_side_effect() -> None:
    entry = next(e for e in INTEGRATION_DUAL_SLUG_CATALOG if e.concern_id == "integration.document_store")
    policy = operational_evidence_policy(entry)
    assert policy.kind.value == "structural_contract"


def test_rpl_e_matrix_catalog_all_structural_or_relational() -> None:
    from tests.qualification.ebh_5._dual_slug_rpl_ef_matrix import assert_dual_slug_rpl_ef_matrix_complete

    assert_dual_slug_rpl_ef_matrix_complete()
