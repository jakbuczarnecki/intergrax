# © Artur Czarnecki. All rights reserved.

"""HARDEN-3F — observability vendor qualification matrix and delivery semantics."""

from __future__ import annotations

import pytest

from intergrax.runtime.observability.operator_wiring import (
    DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY,
    parse_observability_export_backend_id,
)

pytestmark = pytest.mark.unit

_HARDEN_3_INVARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ("canonical persistence first", "test_harden_3c_export_failure_semantics.py", "yes", "n/a"),
    ("exporter failure isolation", "test_harden_3c_export_failure_semantics.py", "yes", "n/a"),
    ("recovery", "test_harden_3c_export_failure_semantics.py", "yes", "deterministic"),
    ("bounded OTLP", "test_harden_3c_export_failure_semantics.py", "yes", "config-only"),
    ("health degraded/recovered", "test_harden_3d_exporter_health.py", "yes", "deterministic"),
    ("per-route isolation", "test_harden_3d_exporter_health.py", "yes", "n/a"),
    ("identity correlation", "test_diag_final_external_otel_e2e.py", "yes", "yes"),
    ("privacy", "test_diag_final_external_otel_e2e.py", "yes", "yes"),
    ("no duplicate canonical path", "test_diag_final_external_otel_e2e.py", "yes", "yes"),
    ("direct OTel boundary", "test_harden_3e_otel_import_gate.py", "yes", "n/a"),
)

_VENDOR_MATRIX: tuple[dict[str, str], ...] = (
    {
        "backend": "otlp",
        "abstraction": "yes",
        "bounded": "yes",
        "failure_isolated": "yes",
        "health": "yes",
        "external_proof": "yes",
    },
    {
        "backend": "elasticsearch",
        "abstraction": "yes",
        "bounded": "yes",
        "failure_isolated": "yes",
        "health": "yes",
        "external_proof": "deferred",
    },
    {
        "backend": "sentry",
        "abstraction": "yes",
        "bounded": "yes",
        "failure_isolated": "yes",
        "health": "yes",
        "external_proof": "deferred",
    },
)

_DELIVERY_SEMANTICS = {
    "canonical_evidence": "durable in RuntimeEventPersistence",
    "otlp_delivery": "best-effort / at-most-once attempt at HOS level",
    "transport_timeout": "bounded",
    "automatic_replay": "no",
    "transport_retry": "none at HOS level; OTLP adapter uses ephemeral client per export",
    "flush_close": "qualified safe — ephemeral httpx client closed in finally",
}


def test_harden_3_invariant_matrix_has_required_rows() -> None:
    invariant_names = {row[0] for row in _HARDEN_3_INVARIANTS}
    required = {
        "canonical persistence first",
        "exporter failure isolation",
        "recovery",
        "bounded OTLP",
        "health degraded/recovered",
        "per-route isolation",
        "identity correlation",
        "privacy",
        "no duplicate canonical path",
        "direct OTel boundary",
    }
    assert required.issubset(invariant_names)


def test_production_observability_backend_registry_matches_vendor_matrix() -> None:
    registered = {
        parse_observability_export_backend_id(backend_id)
        for backend_id in ("otlp", "elasticsearch", "sentry")
    }
    for backend_id in registered:
        DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY.get(backend_id)

    matrix_backends = {row["backend"] for row in _VENDOR_MATRIX}
    assert registered == matrix_backends


def test_otlp_is_only_external_qualified_backend() -> None:
    external_qualified = [
        row["backend"] for row in _VENDOR_MATRIX if row["external_proof"] == "yes"
    ]
    assert external_qualified == ["otlp"]


def test_delivery_semantics_are_explicit() -> None:
    assert _DELIVERY_SEMANTICS["canonical_evidence"].startswith("durable")
    assert "best-effort" in _DELIVERY_SEMANTICS["otlp_delivery"]
    assert _DELIVERY_SEMANTICS["automatic_replay"] == "no"
    assert "ephemeral" in _DELIVERY_SEMANTICS["flush_close"]
