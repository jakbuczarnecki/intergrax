# © Artur Czarnecki. All rights reserved.

"""Every prediction run and signal must carry audit identity."""

from __future__ import annotations

import inspect

import pytest

from intergrax.contracts import predictive_risk as predictive_risk_module
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction import (
    FailurePatternAnalyzer,
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_PAYLOAD_FIELDS = (
    "raw_exception",
    "full_log_dump",
    "secret_payload",
    "customer_data_copy",
)


def test_predictive_audit_contract() -> None:
    result = PredictionEngine(
        registry=PredictiveAnalyzerRegistry(
            (LatencyTrendAnalyzer(), FailurePatternAnalyzer()),
        ),
    ).analyze(crm_agent_showcase_context())

    assert result.audit.prediction_id.startswith("prun_")
    assert result.audit.tenant_id == "tenant-demo"
    assert result.audit.input_snapshot_id == "showcase_crm_agent"
    assert result.audit.context_snapshot_id
    assert result.audit.quality_assessment is not None
    assert 0.0 <= result.audit.quality_assessment.governed_confidence <= 1.0

    for signal in result.signals:
        assert signal.prediction_run_id == result.audit.prediction_id
        assert signal.prediction_run_id.startswith("prun_")
        assert signal.analyzer_metadata.analyzer_id
        assert signal.analyzer_metadata.analyzer_version
        assert signal.evidence_refs
        assert signal.generated_at is not None

    field_names = {f.name for f in PredictiveRiskSignal.__dataclass_fields__.values()}
    for forbidden in _FORBIDDEN_PAYLOAD_FIELDS:
        assert forbidden not in field_names

    source = inspect.getsource(predictive_risk_module.PredictiveRiskSignal)
    for forbidden in _FORBIDDEN_PAYLOAD_FIELDS:
        assert forbidden in source
