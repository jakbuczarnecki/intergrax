# © Artur Czarnecki. All rights reserved.

"""RI-01 qualification gate — foundation packs + composition proof."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomain,
    RuntimeInvariantOverallStatus,
    RuntimeInvariantRule,
    RuntimeInvariantStatus,
)
from intergrax.runtime.invariants.evaluation_id import MonotonicRuntimeInvariantEvaluationIdFactory
from intergrax.runtime.invariants.foundation_composition import (
    compose_foundation_runtime_invariant_service,
    foundation_runtime_invariant_rule_packs,
)
from intergrax.runtime.invariants.service import RuntimeInvariantService
from intergrax.runtime.invariants.clock import SystemRuntimeInvariantEvaluationClock

pytestmark = pytest.mark.unit


def test_qualification_composes_three_domain_packs() -> None:
    packs = foundation_runtime_invariant_rule_packs()
    assert len(packs) == 3
    domains = {pack.domain for pack in packs}
    assert domains == {
        RuntimeInvariantDomain.EXECUTION,
        RuntimeInvariantDomain.DELEGATED_PROVIDER,
        RuntimeInvariantDomain.GOVERNANCE,
    }


def test_qualification_run_conformant() -> None:
    report = compose_foundation_runtime_invariant_service().evaluate()
    assert report.overall_status is RuntimeInvariantOverallStatus.CONFORMANT


def test_qualification_custom_pack_replaces_foundation_subset() -> None:
    foundation = compose_foundation_runtime_invariant_service()
    baseline_ids = {r.rule_id for r in foundation.evaluate().results}

    class ReplacementPack:
        pack_id = "qual-replacement"
        pack_version = "1.0.0"
        domain = RuntimeInvariantDomain.EXECUTION

        @property
        def rules(self) -> tuple[RuntimeInvariantRule, ...]:
            from intergrax.contracts.runtime_invariants import (
                RuntimeInvariantEvaluationContext,
                RuntimeInvariantResult,
                RuntimeInvariantSeverity,
            )

            class _AlwaysPass:
                rule_id = "QUAL-REPLACE-001"
                domain = RuntimeInvariantDomain.EXECUTION
                rule_version = "1.0.0"
                severity = RuntimeInvariantSeverity.LOW

                def evaluate(
                    self,
                    context: RuntimeInvariantEvaluationContext,
                ) -> RuntimeInvariantResult:
                    return RuntimeInvariantResult(
                        rule_id=self.rule_id,
                        domain=self.domain,
                        rule_version=self.rule_version,
                        severity=self.severity,
                        status=RuntimeInvariantStatus.PASS,
                        summary="replacement",
                        evaluation_id=context.evaluation_id,
                        correlation_id=context.correlation_id,
                    )

            return (_AlwaysPass(),)

    service = RuntimeInvariantService(
        rule_packs=(ReplacementPack(),),
        clock=SystemRuntimeInvariantEvaluationClock(),
        evaluation_id_factory=MonotonicRuntimeInvariantEvaluationIdFactory(),
    )
    report = service.evaluate()
    assert "QUAL-REPLACE-001" in {r.rule_id for r in report.results}
    assert "EE-INV-001" in baseline_ids
