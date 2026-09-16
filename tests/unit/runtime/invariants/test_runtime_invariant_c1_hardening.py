# © Artur Czarnecki. All rights reserved.

"""RI-C1 — runner contract, domain extensibility, result trust hardening."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantCompositionError,
    RuntimeInvariantDomain,
    RuntimeInvariantDomainError,
    RuntimeInvariantDomains,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantEvaluationMode,
    RuntimeInvariantEvaluationRequest,
    RuntimeInvariantOverallStatus,
    RuntimeInvariantReport,
    RuntimeInvariantReportSummary,
    RuntimeInvariantResult,
    RuntimeInvariantRule,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantRulePack,
    RuntimeInvariantRunner,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.invariants.composition import (
    compose_default_runtime_invariant_runner,
    validate_runtime_invariant_rule_packs,
)
from intergrax.runtime.invariants.default_runner import DefaultRuntimeInvariantRunner
from intergrax.runtime.invariants.evaluation_id import MonotonicRuntimeInvariantEvaluationIdFactory
from intergrax.runtime.invariants.service import RuntimeInvariantService

pytestmark = pytest.mark.unit

_FIXED_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def now(self) -> datetime:
        return _FIXED_NOW


class _SingleRulePack:
    __slots__ = ("_domain", "_pack_id", "_rules")

    def __init__(
        self,
        *,
        pack_id: str,
        rules: tuple[RuntimeInvariantRule, ...],
        domain: RuntimeInvariantDomain,
    ) -> None:
        self._pack_id = pack_id
        self._domain = domain
        self._rules = rules

    @property
    def pack_id(self) -> str:
        return self._pack_id

    @property
    def pack_version(self) -> str:
        return "1.0.0"

    @property
    def domain(self) -> RuntimeInvariantDomain:
        return self._domain

    @property
    def rules(self) -> tuple[RuntimeInvariantRule, ...]:
        return self._rules


def _service(*packs: RuntimeInvariantRulePack) -> RuntimeInvariantService:
    rule_packs = packs
    return RuntimeInvariantService(
        rule_packs=rule_packs,
        clock=_FixedClock(),
        evaluation_id_factory=MonotonicRuntimeInvariantEvaluationIdFactory(),
        runner=compose_default_runtime_invariant_runner(rule_packs),
    )


def test_ri_c1_t1_custom_injected_runner() -> None:
    class ExternalRuntimeInvariantRunner:
        def evaluate(
            self,
            request: RuntimeInvariantEvaluationRequest,
            *,
            context: RuntimeInvariantEvaluationContext,
            evaluated_at: datetime,
        ) -> RuntimeInvariantReport:
            return RuntimeInvariantReport(
                evaluation_id=context.evaluation_id,
                correlation_id=context.correlation_id,
                evaluated_at=evaluated_at,
                mode=context.mode,
                packs=(),
                results=(),
                summary=RuntimeInvariantReportSummary(0, 0, 0, 0),
                overall_status=RuntimeInvariantOverallStatus.CONFORMANT,
                execution_id=context.execution_id,
            )

    runner = ExternalRuntimeInvariantRunner()
    assert isinstance(runner, RuntimeInvariantRunner)
    service = RuntimeInvariantService(
        rule_packs=(),
        clock=_FixedClock(),
        evaluation_id_factory=MonotonicRuntimeInvariantEvaluationIdFactory(),
        runner=runner,
    )
    report = service.evaluate()
    assert report.results == ()
    assert report.overall_status is RuntimeInvariantOverallStatus.CONFORMANT


def test_ri_c1_t2_custom_external_domain_allowed() -> None:
    domain = RuntimeInvariantDomain("external.vendor.example")
    rule = _PassRule(rule_id="EXT-2", domain=domain)
    report = _service(_SingleRulePack(pack_id="ext", rules=(rule,), domain=domain)).evaluate()
    assert report.results[0].domain == domain


def test_ri_c1_t3_invalid_domain_rejected() -> None:
    with pytest.raises(RuntimeInvariantDomainError):
        RuntimeInvariantDomain("")
    with pytest.raises(RuntimeInvariantDomainError):
        RuntimeInvariantDomain("  ")
    with pytest.raises(RuntimeInvariantDomainError):
        RuntimeInvariantDomain("INVALID_UPPER")
    with pytest.raises(RuntimeInvariantDomainError):
        RuntimeInvariantDomain("bad space")


def test_ri_c1_t4_canonical_domains_preserved() -> None:
    assert RuntimeInvariantDomains.EXECUTION.value == "execution"
    assert RuntimeInvariantDomains.DELEGATED_PROVIDER.value == "delegated_provider"
    assert RuntimeInvariantDomains.GOVERNANCE.value == "governance"


class _PassRule:
    __slots__ = ("_domain", "_rule_id")

    def __init__(self, *, rule_id: str, domain: RuntimeInvariantDomain) -> None:
        self._rule_id = rule_id
        self._domain = domain

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def domain(self) -> RuntimeInvariantDomain:
        return self._domain

    @property
    def rule_version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> RuntimeInvariantSeverity:
        return RuntimeInvariantSeverity.LOW

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        return RuntimeInvariantRuleEvaluation(
            status=RuntimeInvariantStatus.PASS,
            summary="ok",
        )


def test_ri_c1_t5_rule_cannot_spoof_rule_id_in_final_result() -> None:
    domain = RuntimeInvariantDomains.EXECUTION
    rule = _PassRule(rule_id="AUTHORITATIVE-ID", domain=domain)
    report = _service(_SingleRulePack(pack_id="p", rules=(rule,), domain=domain)).evaluate()
    assert report.results[0].rule_id == "AUTHORITATIVE-ID"


def test_ri_c1_t6_rule_cannot_spoof_domain() -> None:
    domain = RuntimeInvariantDomains.EXECUTION
    rule = _PassRule(rule_id="D", domain=domain)
    report = _service(_SingleRulePack(pack_id="p", rules=(rule,), domain=domain)).evaluate()
    assert report.results[0].domain == domain


def test_ri_c1_t7_rule_cannot_spoof_rule_version() -> None:
    domain = RuntimeInvariantDomains.EXECUTION

    class VersionedRule(_PassRule):
        @property
        def rule_version(self) -> str:
            return "9.9.9"

    report = _service(
        _SingleRulePack(pack_id="p", rules=(VersionedRule(rule_id="V", domain=domain),), domain=domain),
    ).evaluate()
    assert report.results[0].rule_version == "9.9.9"


def test_ri_c1_t8_rule_cannot_spoof_severity() -> None:
    domain = RuntimeInvariantDomains.EXECUTION

    class SevereRule(_PassRule):
        @property
        def severity(self) -> RuntimeInvariantSeverity:
            return RuntimeInvariantSeverity.CRITICAL

    report = _service(
        _SingleRulePack(pack_id="p", rules=(SevereRule(rule_id="S", domain=domain),), domain=domain),
    ).evaluate()
    assert report.results[0].severity is RuntimeInvariantSeverity.CRITICAL


def test_ri_c1_t9_rule_cannot_spoof_evaluation_id() -> None:
    domain = RuntimeInvariantDomains.EXECUTION
    rule = _PassRule(rule_id="E", domain=domain)
    report = _service(_SingleRulePack(pack_id="p", rules=(rule,), domain=domain)).evaluate()
    assert report.results[0].evaluation_id == report.evaluation_id
    assert report.evaluation_id.startswith("ri-eval-")


def test_ri_c1_t10_rule_cannot_spoof_correlation_id() -> None:
    domain = RuntimeInvariantDomains.EXECUTION
    rule = _PassRule(rule_id="C", domain=domain)
    report = _service(_SingleRulePack(pack_id="p", rules=(rule,), domain=domain)).evaluate(
        RuntimeInvariantEvaluationRequest(correlation_id="corr-c1"),
    )
    assert report.results[0].correlation_id == "corr-c1"


def test_ri_c1_t11_pack_rule_domain_mismatch_fail_fast() -> None:
    pack_domain = RuntimeInvariantDomains.EXECUTION
    rule = _PassRule(rule_id="M", domain=RuntimeInvariantDomains.GOVERNANCE)
    pack = _SingleRulePack(pack_id="mismatch", rules=(rule,), domain=pack_domain)
    with pytest.raises(RuntimeInvariantCompositionError):
        validate_runtime_invariant_rule_packs((pack,))


def test_ri_c1_t12_deterministic_pack_order() -> None:
    domain_a = RuntimeInvariantDomains.EXECUTION
    domain_b = RuntimeInvariantDomains.GOVERNANCE
    pack_a = _SingleRulePack(
        pack_id="pack-a",
        rules=(_PassRule(rule_id="A", domain=domain_a),),
        domain=domain_a,
    )
    pack_b = _SingleRulePack(
        pack_id="pack-b",
        rules=(_PassRule(rule_id="B", domain=domain_b),),
        domain=domain_b,
    )
    forward = _service(pack_a, pack_b).evaluate()
    reverse = _service(pack_b, pack_a).evaluate()
    assert forward.packs == reverse.packs


def test_ri_c1_t13_invalid_rule_output_type_fail_closed() -> None:
    domain = RuntimeInvariantDomains.EXECUTION

    class BadOutputRule(_PassRule):
        def evaluate(
            self,
            context: RuntimeInvariantEvaluationContext,
        ) -> RuntimeInvariantResult:
            return RuntimeInvariantResult(
                rule_id="spoof",
                domain=RuntimeInvariantDomains.GOVERNANCE,
                rule_version="0",
                severity=RuntimeInvariantSeverity.LOW,
                status=RuntimeInvariantStatus.PASS,
                summary="bad",
                evaluation_id="evil",
                correlation_id="evil",
            )

    report = _service(
        _SingleRulePack(pack_id="bad", rules=(BadOutputRule(rule_id="BAD", domain=domain),), domain=domain),
    ).evaluate()
    result = report.results[0]
    assert result.status is RuntimeInvariantStatus.EVALUATION_ERROR
    assert result.diagnostic_code == "RI_RULE_OUTPUT_TYPE_MISMATCH"


def test_ri_c1_t14_default_runner_conforms_to_contract() -> None:
    runner = compose_default_runtime_invariant_runner(())
    assert isinstance(runner, RuntimeInvariantRunner)
    assert isinstance(runner, DefaultRuntimeInvariantRunner)


def test_ri_c1_t15_service_has_no_concrete_runner_import() -> None:
    import ast
    from pathlib import Path

    service_path = Path(__file__).resolve().parents[4] / "intergrax" / "runtime" / "invariants" / "service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "default_runner" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "default_runner" not in alias.name


def test_ri_c1_rule_returning_evaluation_error_status_fail_closed() -> None:
    domain = RuntimeInvariantDomains.EXECUTION

    class ErrorStatusRule(_PassRule):
        def evaluate(
            self,
            context: RuntimeInvariantEvaluationContext,
        ) -> RuntimeInvariantRuleEvaluation:
            return RuntimeInvariantRuleEvaluation(
                status=RuntimeInvariantStatus.EVALUATION_ERROR,
                summary="rule tried to own evaluation error",
            )

    report = _service(
        _SingleRulePack(pack_id="e", rules=(ErrorStatusRule(rule_id="ERR", domain=domain),), domain=domain),
    ).evaluate()
    result = report.results[0]
    assert result.status is RuntimeInvariantStatus.EVALUATION_ERROR
    assert result.diagnostic_code == "RI_RESULT_CONTRACT_MISMATCH"
    assert result.summary == "invariant result contract mismatch"
