# © Artur Czarnecki. All rights reserved.

"""RI-01 qualification gate — foundation packs + composition proof."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomains,
    RuntimeInvariantOverallStatus,
    RuntimeInvariantRule,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantStatus,
)
from intergrax.runtime.invariants.composition import compose_default_runtime_invariant_runner
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
        RuntimeInvariantDomains.EXECUTION,
        RuntimeInvariantDomains.DELEGATED_PROVIDER,
        RuntimeInvariantDomains.GOVERNANCE,
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
        domain = RuntimeInvariantDomains.EXECUTION

        @property
        def rules(self) -> tuple[RuntimeInvariantRule, ...]:
            from intergrax.contracts.runtime_invariants import (
                RuntimeInvariantEvaluationContext,
                RuntimeInvariantSeverity,
            )

            class _AlwaysPass:
                rule_id = "QUAL-REPLACE-001"
                domain = RuntimeInvariantDomains.EXECUTION
                rule_version = "1.0.0"
                severity = RuntimeInvariantSeverity.LOW

                def evaluate(
                    self,
                    context: RuntimeInvariantEvaluationContext,
                ) -> RuntimeInvariantRuleEvaluation:
                    return RuntimeInvariantRuleEvaluation(
                        status=RuntimeInvariantStatus.PASS,
                        summary="replacement",
                    )

            return (_AlwaysPass(),)

    packs = (ReplacementPack(),)
    service = RuntimeInvariantService(
        rule_packs=packs,
        clock=SystemRuntimeInvariantEvaluationClock(),
        evaluation_id_factory=MonotonicRuntimeInvariantEvaluationIdFactory(),
        runner=compose_default_runtime_invariant_runner(packs),
    )
    report = service.evaluate()
    assert "QUAL-REPLACE-001" in {r.rule_id for r in report.results}
    assert "EE-INV-001" in baseline_ids
