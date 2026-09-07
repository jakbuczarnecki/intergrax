# © Artur Czarnecki. All rights reserved.

"""Unit tests for DG-001B R5 qualification helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_workspace_application.host.worker_construction_fault import (
    maybe_raise_worker_construction_fault,
    parse_worker_construction_fault_mode,
    qualification_secret_sentinel,
)
from scripts.proof.dg001b_r5_qualification_support import (
    build_qualification_environment,
    evaluate_prerequisites,
)


def test_parse_worker_construction_fault_mode_defaults_to_none() -> None:
    assert parse_worker_construction_fault_mode("") == "none"
    assert parse_worker_construction_fault_mode("none") == "none"


def test_parse_worker_construction_fault_mode_accepts_typed_bootstrap_exception() -> None:
    assert (
        parse_worker_construction_fault_mode("typed_bootstrap_exception")
        == "typed_bootstrap_exception"
    )


def test_parse_worker_construction_fault_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_WORKER_CONSTRUCTION_FAULT"):
        parse_worker_construction_fault_mode("unexpected")


def test_maybe_raise_worker_construction_fault_is_noop_by_default() -> None:
    maybe_raise_worker_construction_fault("none")


def test_maybe_raise_worker_construction_fault_raises_type_error_with_sentinel() -> None:
    with pytest.raises(TypeError) as exc_info:
        maybe_raise_worker_construction_fault("typed_bootstrap_exception")
    assert qualification_secret_sentinel() in str(exc_info.value)


def test_build_qualification_environment_sets_fault_and_observability(tmp_path: Path) -> None:
    environment = build_qualification_environment(
        marker="dg001b-r5-test",
        attempt_data_home=tmp_path,
        base_environment={},
    )
    assert environment["LOCAL_WORKSPACE_WORKER_CONSTRUCTION_FAULT"] == "typed_bootstrap_exception"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] == "true"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] == "elasticsearch"
    assert environment["LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND"] == "mongodb"
    assert environment["LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT"] == "dg001b-r5-test"


def test_evaluate_prerequisites_reports_backend_classes() -> None:
    status = evaluate_prerequisites(
        mongodb_uri="mongodb://example.invalid:27018/db",
        mongodb_database="intergrax_proofs",
        mongodb_collection="lkw_managed_workspaces",
        elasticsearch_url="http://127.0.0.1:9200",
        redis_url="redis://127.0.0.1:6379/0",
    )
    assert status.mongodb_uri_class == "mongodb+srv_or_standard_local"
    assert status.observability_backend == "elasticsearch"
    assert status.mongodb_database == "intergrax_proofs"
    assert status.mongodb_collection_authority == "lkw_managed_workspaces"
    assert isinstance(status.environment_ready, bool)


def test_qualification_environment_json_is_secret_free(tmp_path: Path) -> None:
    environment = build_qualification_environment(
        marker="dg001b-r5-test",
        attempt_data_home=tmp_path,
        base_environment={},
    )
    serialized = json.dumps(environment)
    assert qualification_secret_sentinel() not in serialized
