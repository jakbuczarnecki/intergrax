# © Artur Czarnecki. All rights reserved.

"""Contract tests for application execution stage signal (OBS-P1B / P1B-R1)."""

from __future__ import annotations

import pytest

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionCorrelation,
    ApplicationExecutionStageSignalError,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = pytest.mark.gate


def _valid_correlation_kwargs(*, tenant_id: str = "tenant-vpi") -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "scenario_execution_correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    }


def test_application_execution_correlation_accepts_canonical_tenant_id() -> None:
    correlation = ApplicationExecutionCorrelation(**_valid_correlation_kwargs(tenant_id="tenant-vpi"))
    assert correlation.tenant_id == "tenant-vpi"


def test_application_execution_correlation_rejects_non_str_tenant_id() -> None:
    with pytest.raises(ApplicationExecutionStageSignalError, match="tenant_id must be str"):
        ApplicationExecutionCorrelation(**{**_valid_correlation_kwargs(), "tenant_id": 42})


def test_application_execution_correlation_rejects_empty_tenant_id() -> None:
    with pytest.raises(ApplicationExecutionStageSignalError, match="tenant_id must be non-empty"):
        ApplicationExecutionCorrelation(**_valid_correlation_kwargs(tenant_id=""))


def test_application_execution_correlation_rejects_whitespace_only_tenant_id() -> None:
    with pytest.raises(ApplicationExecutionStageSignalError, match="tenant_id must be non-empty"):
        ApplicationExecutionCorrelation(**_valid_correlation_kwargs(tenant_id="   "))


def test_stage_signal_contract_module_has_no_vpi_imports() -> None:
    import intergrax.contracts.application_execution_stage_signal as module

    source_path = module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "verified_product_identification" not in text
    assert "platform_proofs" not in text
