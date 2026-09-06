# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DISCOVERY_ROOT = Path("intergrax/runtime/diagnostics")
_DISCOVERY_FILES = (
    _DISCOVERY_ROOT / "diagnostic_scope_discovery_models.py",
    _DISCOVERY_ROOT / "diagnostic_scope_discovery_provider.py",
    _DISCOVERY_ROOT / "diagnostic_scope_discovery_service.py",
    _DISCOVERY_ROOT / "providers" / "problem_scope_provider.py",
    _DISCOVERY_ROOT / "providers" / "causal_transport_scope_provider.py",
    _DISCOVERY_ROOT / "providers" / "runtime_event_scope_provider.py",
)

_SERVICE_FILE = _DISCOVERY_ROOT / "diagnostic_scope_discovery_service.py"

_FORBIDDEN_IMPORT_TOKENS = (
    "kafka",
    "celery",
    "rabbitmq",
    "pymongo",
    "qdrant",
    "opentelemetry",
    "platform_proofs",
    "applications",
    "worker",
    "background_worker_factory",
)

_SERVICE_FORBIDDEN_IMPORT_TOKENS = (
    "problem_persistence",
    "problem_occurrence_persistence",
    "runtime_event_persistence",
    "causal_evidence_persistence",
    "pymongo",
    "DocumentStore",
    "kafka",
    "celery",
)


def test_discovery_core_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _DISCOVERY_FILES:
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_IMPORT_TOKENS:
            if token in source:
                violations.append(f"{path}: {token}")
    assert not violations, f"forbidden imports: {violations}"


def test_discovery_service_has_no_concrete_persistence_imports() -> None:
    source = _SERVICE_FILE.read_text(encoding="utf-8")
    violations = [
        token
        for token in _SERVICE_FORBIDDEN_IMPORT_TOKENS
        if token in source
    ]
    assert not violations, f"service concrete persistence imports: {violations}"
