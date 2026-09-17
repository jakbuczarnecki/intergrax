# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — Evidence and Diagnostics non-authority qualification."""

from __future__ import annotations

import pytest

from testing_support.mp4r7_enterprise_integration.composition import (
    open_mp4r7_enterprise_integration_composition,
)
from testing_support.mp4r7_enterprise_integration.contracts import Mp4R7ProtectedOperationError
from testing_support.mp4r7_enterprise_integration.scenario import (
    Mp4R7EnterpriseIntegrationScenarioExecutor,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_scenario_evidence_failure_does_not_create_allow_or_success() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_evidence_failure()
    assert result.primary_error_code == Mp4R7ProtectedOperationError.__name__
    assert result.secondary_evidence_error_code == "FunctionalEvidencePersistenceError"
    assert result.protected_operation_completed is False
