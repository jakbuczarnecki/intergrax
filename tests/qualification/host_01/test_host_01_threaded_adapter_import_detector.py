# © Artur Czarnecki. All rights reserved.

"""Unit proof for HOST-01 legacy ThreadedExecutionAdapter import detector."""

from __future__ import annotations

import pytest

from tests.qualification.host_01.threaded_adapter_import_detector import (
    THREADED_ADAPTER_MODULE,
    threaded_adapter_import_violations_from_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_detector_flags_module_import() -> None:
    source = f"import {THREADED_ADAPTER_MODULE}\n"
    violations = threaded_adapter_import_violations_from_source(source, "fake/composition.py")
    assert any("import" in v and THREADED_ADAPTER_MODULE in v for v in violations)


def test_detector_flags_from_import() -> None:
    source = f"from {THREADED_ADAPTER_MODULE} import ThreadedExecutionAdapter\n"
    violations = threaded_adapter_import_violations_from_source(source, "fake/composition.py")
    assert any("ThreadedExecutionAdapter" in v for v in violations)


def test_detector_flags_aliased_module_import() -> None:
    source = f"import {THREADED_ADAPTER_MODULE} as legacy_execution\n"
    violations = threaded_adapter_import_violations_from_source(source, "fake/composition.py")
    assert violations


def test_detector_ignores_unrelated_imports() -> None:
    source = "from intergrax.runtime.execution.host_task import HostTaskExecutionPort\n"
    violations = threaded_adapter_import_violations_from_source(source, "fake/composition.py")
    assert violations == []
