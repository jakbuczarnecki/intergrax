# © Artur Czarnecki. All rights reserved.

"""RI-T1 … RI-T15 — runtime invariant foundation matrix."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantCompositionError,
    RuntimeInvariantDomain,
    RuntimeInvariantDomains,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantEvaluationMode,
    RuntimeInvariantEvaluationRequest,
    RuntimeInvariantOverallStatus,
    RuntimeInvariantPackRef,
    RuntimeInvariantRule,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantRulePack,
    RuntimeInvariantSelection,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.invariants.clock import SystemRuntimeInvariantEvaluationClock
from intergrax.runtime.invariants.composition import (
    compose_default_runtime_invariant_runner,
    validate_runtime_invariant_rule_packs,
)
from intergrax.runtime.invariants.evaluation_id import MonotonicRuntimeInvariantEvaluationIdFactory
from intergrax.runtime.invariants.service import RuntimeInvariantService

pytestmark = pytest.mark.unit

_FIXED_NOW = datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc)


class _FixedClock:
    def now(self) -> datetime:
        return _FIXED_NOW


class _BoomRule:
    rule_id = "TEST-RULE-BOOM"
    domain = RuntimeInvariantDomains.EXECUTION
    rule_version = "1.0.0"
    severity = RuntimeInvariantSeverity.LOW

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        raise RuntimeError("secret-token-xyz")


class _StaticRule:
    __slots__ = ("_domain", "_rule_id", "_status", "_summary")

    def __init__(
        self,
        *,
        rule_id: str,
        status: RuntimeInvariantStatus,
        summary: str,
        domain: RuntimeInvariantDomain = RuntimeInvariantDomains.EXECUTION,
    ) -> None:
        self._rule_id = rule_id
        self._status = status
        self._summary = summary
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
        return RuntimeInvariantSeverity.MEDIUM

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        return RuntimeInvariantRuleEvaluation(status=self._status, summary=self._summary)


class _SingleRulePack:
    __slots__ = ("_pack_id", "_domain", "_rules")

    def __init__(
        self,
        *,
        pack_id: str,
        rules: tuple[RuntimeInvariantRule, ...],
        domain: RuntimeInvariantDomain = RuntimeInvariantDomains.EXECUTION,
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


def test_ri_t1_empty_packs_conformant() -> None:
    service = _service()
    report = service.evaluate()
    assert report.overall_status is RuntimeInvariantOverallStatus.CONFORMANT
    assert report.results == ()
    assert report.summary.pass_count == 0


def test_ri_t2_one_pass() -> None:
    pack = _SingleRulePack(
        pack_id="p-pass",
        rules=(
            _StaticRule(
                rule_id="RI-T2",
                status=RuntimeInvariantStatus.PASS,
                summary="ok",
            ),
        ),
    )
    report = _service(pack).evaluate()
    assert report.results[0].status is RuntimeInvariantStatus.PASS


def test_ri_t3_one_violation_non_conformant() -> None:
    pack = _SingleRulePack(
        pack_id="p-violation",
        rules=(
            _StaticRule(
                rule_id="RI-T3",
                status=RuntimeInvariantStatus.VIOLATION,
                summary="bad",
            ),
        ),
    )
    report = _service(pack).evaluate()
    assert report.overall_status is RuntimeInvariantOverallStatus.NON_CONFORMANT


def test_ri_t4_not_applicable_still_conformant() -> None:
    pack = _SingleRulePack(
        pack_id="p-na",
        rules=(
            _StaticRule(
                rule_id="RI-T4",
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="na",
            ),
        ),
    )
    report = _service(pack).evaluate()
    assert report.overall_status is RuntimeInvariantOverallStatus.CONFORMANT


def test_ri_t5_evaluation_exception_indeterminate() -> None:
    pack = _SingleRulePack(pack_id="p-boom", rules=(_BoomRule(),))
    report = _service(pack).evaluate()
    assert report.results[0].status is RuntimeInvariantStatus.EVALUATION_ERROR
    assert report.overall_status is RuntimeInvariantOverallStatus.INDETERMINATE


def test_ri_t6_deterministic_order_independent_of_pack_order() -> None:
    rule_a = _StaticRule(rule_id="A-RULE", status=RuntimeInvariantStatus.PASS, summary="a")
    rule_b = _StaticRule(
        rule_id="B-RULE",
        status=RuntimeInvariantStatus.PASS,
        summary="b",
        domain=RuntimeInvariantDomains.GOVERNANCE,
    )
    pack_one = _SingleRulePack(pack_id="pack-one", rules=(rule_a,))
    pack_two = _SingleRulePack(
        pack_id="pack-two",
        rules=(rule_b,),
        domain=RuntimeInvariantDomains.GOVERNANCE,
    )
    forward = _service(pack_one, pack_two).evaluate()
    reverse = _service(pack_two, pack_one).evaluate()
    assert tuple(r.rule_id for r in forward.results) == tuple(
        r.rule_id for r in reverse.results
    )


def test_ri_t7_duplicate_rule_id_fail_fast() -> None:
    rule = _StaticRule(rule_id="DUP", status=RuntimeInvariantStatus.PASS, summary="x")
    pack_a = _SingleRulePack(pack_id="pack-a", rules=(rule,))
    pack_b = _SingleRulePack(pack_id="pack-b", rules=(rule,))
    with pytest.raises(RuntimeInvariantCompositionError, match="duplicate runtime invariant rule_id"):
        validate_runtime_invariant_rule_packs((pack_a, pack_b))


def test_ri_t8_duplicate_pack_id_fail_fast() -> None:
    rule_a = _StaticRule(rule_id="R1", status=RuntimeInvariantStatus.PASS, summary="a")
    rule_b = _StaticRule(rule_id="R2", status=RuntimeInvariantStatus.PASS, summary="b")
    pack_a = _SingleRulePack(pack_id="same-pack", rules=(rule_a,))
    pack_b = _SingleRulePack(pack_id="same-pack", rules=(rule_b,))
    with pytest.raises(RuntimeInvariantCompositionError, match="duplicate runtime invariant pack_id"):
        validate_runtime_invariant_rule_packs((pack_a, pack_b))


def test_ri_t9_custom_external_pack_without_runner_changes() -> None:
    external_domain = RuntimeInvariantDomain("external.vendor.example")

    class ExternalPack:
        pack_id = "external-qual-pack"
        pack_version = "9.9.9"
        domain = external_domain

        @property
        def rules(self) -> tuple[RuntimeInvariantRule, ...]:
            return (
                _StaticRule(
                    rule_id="EXT-INV-001",
                    status=RuntimeInvariantStatus.PASS,
                    summary="external",
                    domain=external_domain,
                ),
            )

    report = _service(ExternalPack()).evaluate()
    assert report.packs == (
        RuntimeInvariantPackRef(
            pack_id="external-qual-pack",
            pack_version="9.9.9",
            domain=external_domain,
        ),
    )
    assert report.results[0].rule_id == "EXT-INV-001"
    assert report.results[0].domain == external_domain


def test_ri_t10_domain_selection() -> None:
    exec_rule = _StaticRule(
        rule_id="EXEC-ONLY",
        status=RuntimeInvariantStatus.PASS,
        summary="e",
        domain=RuntimeInvariantDomains.EXECUTION,
    )
    gov_rule = _StaticRule(
        rule_id="GOV-ONLY",
        status=RuntimeInvariantStatus.PASS,
        summary="g",
        domain=RuntimeInvariantDomains.GOVERNANCE,
    )
    packs = (
        _SingleRulePack(pack_id="p1", rules=(exec_rule,)),
        _SingleRulePack(
            pack_id="p2",
            rules=(gov_rule,),
            domain=RuntimeInvariantDomains.GOVERNANCE,
        ),
    )
    request = RuntimeInvariantEvaluationRequest(
        selection=RuntimeInvariantSelection(domains=frozenset({RuntimeInvariantDomains.GOVERNANCE})),
    )
    report = _service(*packs).evaluate(request)
    assert [r.rule_id for r in report.results] == ["GOV-ONLY"]


def test_ri_t11_rule_id_selection() -> None:
    rules = (
        _StaticRule(rule_id="KEEP", status=RuntimeInvariantStatus.PASS, summary="k"),
        _StaticRule(rule_id="SKIP", status=RuntimeInvariantStatus.VIOLATION, summary="s"),
    )
    pack = _SingleRulePack(pack_id="sel", rules=rules)
    request = RuntimeInvariantEvaluationRequest(
        selection=RuntimeInvariantSelection(rule_ids=frozenset({"KEEP"})),
    )
    report = _service(pack).evaluate(request)
    assert len(report.results) == 1
    assert report.results[0].rule_id == "KEEP"


def test_ri_t12_immutable_report() -> None:
    pack = _SingleRulePack(
        pack_id="imm",
        rules=(_StaticRule(rule_id="IMM", status=RuntimeInvariantStatus.PASS, summary="i"),),
    )
    report = _service(pack).evaluate()
    with pytest.raises(FrozenInstanceError):
        report.summary.pass_count = 1  # type: ignore[misc]


def test_ri_t13_error_sanitization() -> None:
    pack = _SingleRulePack(pack_id="san", rules=(_BoomRule(),))
    report = _service(pack).evaluate()
    assert "secret-token" not in report.results[0].summary


def test_ri_t14_evaluation_correlation() -> None:
    pack = _SingleRulePack(
        pack_id="corr",
        rules=(_StaticRule(rule_id="C", status=RuntimeInvariantStatus.PASS, summary="c"),),
    )
    request = RuntimeInvariantEvaluationRequest(correlation_id="corr-42")
    report = _service(pack).evaluate(request)
    assert report.correlation_id == "corr-42"
    assert report.evaluation_id.startswith("ri-eval-")
    assert report.results[0].correlation_id == "corr-42"


def test_ri_t15_version_propagation() -> None:
    class VersionedPack:
        pack_id = "versioned"
        pack_version = "2.3.4"
        domain = RuntimeInvariantDomains.EXECUTION

        @property
        def rules(self) -> tuple[RuntimeInvariantRule, ...]:
            return (
                _StaticRule(rule_id="VER", status=RuntimeInvariantStatus.PASS, summary="v"),
            )

    report = _service(VersionedPack()).evaluate()
    assert report.packs[0].pack_version == "2.3.4"
    assert report.results[0].rule_version == "1.0.0"
    assert report.mode is RuntimeInvariantEvaluationMode.AD_HOC
