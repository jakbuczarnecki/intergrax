# © Artur Czarnecki. All rights reserved.

"""Re-export fake executor from testing_support (unit test import path)."""

from testing_support.execution_qualification.fake_executor import (
    FakeExecutorProbe,
    FakeQualificationSuiteExecutor,
)

__all__ = ["FakeExecutorProbe", "FakeQualificationSuiteExecutor"]
