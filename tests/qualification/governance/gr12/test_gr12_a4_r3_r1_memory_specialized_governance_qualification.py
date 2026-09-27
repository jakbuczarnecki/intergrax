# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R3-R1 — Memory specialized governance enterprise qualification gates."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import user_memory_scope
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryGovernanceRecordSnapshot,
    MemoryRetentionAction,
    MemoryRetentionDecision,
    MemorySecurityContext,
    MemorySecurityStrategySet,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
)
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    memory_control_scope_from_entity_scope,
)
from intergrax.memory.strategies.defaults.memory_security_governance import (
    DefaultMemoryAdmissionPolicy,
    DefaultMemoryAuthorizationPolicy,
    DefaultMemoryGovernancePolicy,
    DefaultMemoryRetentionPolicy,
    DefaultMemoryTrustEvaluationPolicy,
    build_default_memory_security_strategy_set,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R3_MEMORY_ADR_PATH,
    GR12_A4_R3_R1_EXECUTION_PROOF_NODES,
    GR12_A4_R3_R1_QUALIFICATION_PROOF,
    GR12_A4_R3_R1_TASK_STATUS,
    GR12_CONTROL_PLANE_SURFACES,
    GR12_MEMORY_FUTURE_LIVE_OPERATOR_RULE,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.gr12_a4_r3_memory_architecture_decision import (
    GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION,
    GR12_MEMORY_GOVERNANCE_SERVICE,
    GR12_MEMORY_MUTATION_SURFACES,
    GR12_MEMORY_POLICY_EVALUATOR_PORTS,
    GR12_MEMORY_R3_R1_QUALIFICATION_PROOF,
    Gr12MemoryMutationContextClass,
    Gr12MemoryMutationGovernanceBinding,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_proof_nodes_registered,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MEMORY_ROOT = _REPO_ROOT / "intergrax" / "memory"

_FORBIDDEN_CLA04_SYMBOLS = (
    "ControlPlaneMutationAuthorizationBoundary",
    "ControlPlaneMutationDecision",
    "ControlPlaneMutationRequest",
)

_MUTATION_ENTRYPOINT_GOVERNANCE_MARKERS: tuple[tuple[str, str], ...] = (
    ("default_memory_control_plane.py", "_enforce_governance"),
    ("long_horizon_memory_service.py", "enforce_specialized_memory_mutation"),
    ("procedural_memory_service.py", "enforce_specialized_memory_mutation"),
    ("entity_memory_indexing.py", "enforce_specialized_memory_mutation"),
    ("procedural_memory_indexing.py", "enforce_specialized_memory_mutation"),
)

_OPERATOR_SURFACE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"memory\s+admin",
        r"delete\s+memory",
        r"compact\s+memory",
        r"promote\s+memory",
        r"supersede\s+memory",
        r"operator\s+memory",
        r"memory\s+mutation",
    )
)

_LIVE_OPERATOR_SCAN_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "intergrax" / "applications",
)

_TENANT = "tenant-gr12-mem"
_USER = "user-gr12-mem"


def _identity() -> RequestIdentity:
    return RequestIdentity(tenant_id=_TENANT, user_id=_USER)


def _remember_request() -> MemoryGovernanceEvaluationRequest:
    scope = user_memory_scope(_identity())
    entry = UserProfileMemoryEntry(
        content="gr12 memory qualification",
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        governance=MemoryRecordGovernance(
            data_classification=DataClassification.INTERNAL
        ),
    )
    return MemoryGovernanceEvaluationRequest(
        context=MemorySecurityContext(
            identity=_identity(),
            scope=scope,
            operation=MemoryGovernanceOperation.REMEMBER,
        ),
        proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
    )


def _memory_catalog_row():
    return next(
        row
        for row in GR12_CONTROL_PLANE_SURFACES
        if row.path_id == "CP-MEM-SPECIALIZED-MUTATION"
    )


def _memory_adr_text() -> str:
    return (_REPO_ROOT / GR12_A4_R3_MEMORY_ADR_PATH).read_text(encoding="utf-8")


def _iter_memory_py_files() -> list[Path]:
    return sorted(_MEMORY_ROOT.rglob("*.py"))


def _memory_imports_forbidden_cla04(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "control_plane_mutation" not in module:
                continue
            for alias in node.names:
                if alias.name.startswith("ControlPlaneMutation"):
                    hits.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{alias.name}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ControlPlaneMutation"):
                    hits.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{alias.name}"
                    )
    return hits


def test_gr12_a4_r3_r1_m1_memory_adr_accepted() -> None:
    adr = _memory_adr_text()
    assert "Accepted" in adr
    assert "GR-12-A4-R3" in adr


def test_gr12_a4_r3_r1_m2_cp_mem_not_applicable() -> None:
    row = _memory_catalog_row()
    assert row.applicability is Gr12Applicability.NOT_APPLICABLE
    assert row.coverage is Gr12CoverageStatus.NOT_APPLICABLE
    assert GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION.cp_mem_catalog_applicability is (
        Gr12Applicability.NOT_APPLICABLE
    )


def test_gr12_a4_r3_r1_m3_single_production_permission_authority() -> None:
    service_path = (
        _REPO_ROOT
        / f"{GR12_MEMORY_GOVERNANCE_SERVICE.rsplit('.', 1)[0].replace('.', '/')}.py"
    )
    assert service_path.is_file()
    merge_sources = 0
    for path in _iter_memory_py_files():
        text = path.read_text(encoding="utf-8")
        if "_merge_decisions" in text or "strictest" in text.lower():
            merge_sources += 1
    assert merge_sources == 1, (
        "expected exactly one policy merge orchestrator in memory domain"
    )


def test_gr12_a4_r3_r1_m4_no_cla04_authority_in_memory_domain() -> None:
    violations: list[str] = []
    for path in _iter_memory_py_files():
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_CLA04_SYMBOLS:
            if symbol in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{symbol}")
        violations.extend(_memory_imports_forbidden_cla04(path))
    assert violations == []


def test_gr12_a4_r3_r1_m5_no_live_operator_memory_mutation_surface() -> None:
    skip_fragments = ("/test", "tests/", "runtime-context/", "/docker/", "sample_docs/")
    live_hits: list[str] = []
    for root in _LIVE_OPERATOR_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if any(fragment in rel for fragment in skip_fragments):
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in _OPERATOR_SURFACE_PATTERNS:
                if pattern.search(text):
                    live_hits.append(rel)
                    break
    assert live_hits == []


def test_gr12_a4_r3_r1_m6_inventoried_mutation_paths_enforce_governance() -> None:
    enforced = [
        surface
        for surface in GR12_MEMORY_MUTATION_SURFACES
        if surface.governance
        is Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED
    ]
    assert len(enforced) >= 5
    for rel_name, marker in _MUTATION_ENTRYPOINT_GOVERNANCE_MARKERS:
        path = _MEMORY_ROOT / rel_name
        assert path.is_file(), rel_name
        assert marker in path.read_text(encoding="utf-8")


def test_gr12_a4_r3_r1_m7_deny_zero_mutation_enforcement_helper() -> None:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="gr12.test.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class _DenyAll:
        policy_id: str = "gr12.test.deny"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return denied

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=_DenyAll(),
            trust=base.trust,
            admission=base.admission,
            governance=base.governance,
            retention=base.retention,
        )
    )
    request = _remember_request()
    with pytest.raises(MemoryGovernanceDenied):
        enforce_specialized_memory_mutation(service, request)


def test_gr12_a4_r3_r1_m8_require_review_zero_mutation() -> None:
    review = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
        reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
        policy_id="gr12.test.review",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class _ReviewGovernance:
        policy_id: str = "gr12.test.review"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return review

    strategies = MemorySecurityStrategySet(
        authorization=DefaultMemoryAuthorizationPolicy(),
        trust=DefaultMemoryTrustEvaluationPolicy(),
        admission=DefaultMemoryAdmissionPolicy(),
        governance=_ReviewGovernance(),
        retention=DefaultMemoryRetentionPolicy(),
    )
    service = MemorySecurityGovernanceService(strategies=strategies)
    with pytest.raises(MemoryGovernanceDenied):
        enforce_specialized_memory_mutation(service, _remember_request())


def test_gr12_a4_r3_r1_m9_missing_strategy_fail_closed() -> None:
    service = MemorySecurityGovernanceService(
        strategies=build_default_memory_security_strategy_set()
    )
    service.strategies = None  # type: ignore[assignment]
    decision = service.evaluate(_remember_request())
    assert decision.outcome is MemoryGovernanceOutcome.DENY
    assert decision.reason_code is MemoryGovernanceReasonCode.POLICY_MISSING


def test_gr12_a4_r3_r1_m10_policy_exception_fail_closed() -> None:
    class _Broken:
        policy_id = "broken"
        policy_version = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            raise RuntimeError("boom")

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=base.authorization,
            trust=base.trust,
            admission=_Broken(),
            governance=base.governance,
            retention=base.retention,
        )
    )
    decision = service.evaluate(_remember_request())
    assert decision.outcome is MemoryGovernanceOutcome.DENY
    assert decision.reason_code is MemoryGovernanceReasonCode.POLICY_FAILURE


def test_gr12_a4_r3_r1_m11_strictest_merge_preserves_narrowing() -> None:
    allow = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.ALLOW,
        reason_code=MemoryGovernanceReasonCode.ALLOWED,
        policy_id="gr12.allow",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )
    deny = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="gr12.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )
    review = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
        reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
        policy_id="gr12.review",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class _FixedAuth:
        decision: MemoryGovernanceDecision
        policy_id: str = "gr12.auth"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return self.decision

    @dataclass(frozen=True, slots=True)
    class _FixedGov:
        decision: MemoryGovernanceDecision
        policy_id: str = "gr12.gov"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return self.decision

    base = build_default_memory_security_strategy_set()
    service_allow_deny = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=_FixedAuth(allow),
            trust=base.trust,
            admission=base.admission,
            governance=_FixedGov(deny),
            retention=base.retention,
        )
    )
    merged_deny = service_allow_deny.evaluate(_remember_request())
    assert merged_deny.outcome is MemoryGovernanceOutcome.DENY

    service_allow_review = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=_FixedAuth(allow),
            trust=base.trust,
            admission=base.admission,
            governance=_FixedGov(review),
            retention=base.retention,
        )
    )
    merged_review = service_allow_review.evaluate(_remember_request())
    assert merged_review.outcome is MemoryGovernanceOutcome.REQUIRE_REVIEW


def test_gr12_a4_r3_r1_m11_retention_delete_block_on_mutation() -> None:
    @dataclass(frozen=True, slots=True)
    class _DeleteRetention:
        policy_id: str = "gr12.retention.delete"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryRetentionDecision:
            return MemoryRetentionDecision(
                action=MemoryRetentionAction.DELETE,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
            )

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=base.authorization,
            trust=base.trust,
            admission=base.admission,
            governance=base.governance,
            retention=_DeleteRetention(),
        )
    )
    decision = service.evaluate(_remember_request())
    assert decision.outcome is MemoryGovernanceOutcome.DENY
    assert decision.reason_code is MemoryGovernanceReasonCode.RETENTION_BLOCK


def test_gr12_a4_r3_r1_m12_external_policy_strategies_replace_defaults() -> None:
    for port in GR12_MEMORY_POLICY_EVALUATOR_PORTS:
        module_path = _REPO_ROOT / f"{port.rsplit('.', 1)[0].replace('.', '/')}.py"
        assert module_path.is_file(), port
    assert inspect.isclass(MemorySecurityGovernanceService)
    assert (
        "provider"
        not in inspect.getsource(MemorySecurityGovernanceService.evaluate).lower()
    )


def test_gr12_a4_r3_r1_m13_custom_plugin_cannot_widen_deny() -> None:
    deny = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.AUTHORIZATION_DENY,
        policy_id="gr12.auth.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class _DenyAuth:
        policy_id: str = "gr12.auth.deny"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return deny

    @dataclass(frozen=True, slots=True)
    class _AllowGovernance:
        policy_id: str = "gr12.plugin.allow"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return MemoryGovernanceDecision(
                outcome=MemoryGovernanceOutcome.ALLOW,
                reason_code=MemoryGovernanceReasonCode.ALLOWED,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                operation=request.context.operation,
            )

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=_DenyAuth(),
            trust=base.trust,
            admission=base.admission,
            governance=_AllowGovernance(),
            retention=base.retention,
        )
    )
    decision = service.evaluate(_remember_request())
    assert decision.outcome is MemoryGovernanceOutcome.DENY


def test_gr12_a4_r3_r1_m13_custom_plugin_cannot_widen_require_review() -> None:
    review = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
        reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
        policy_id="gr12.review",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class _ReviewAdmission:
        policy_id: str = "gr12.review"
        policy_version: str = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return review

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=DefaultMemoryAuthorizationPolicy(),
            trust=base.trust,
            admission=_ReviewAdmission(),
            governance=DefaultMemoryGovernancePolicy(),
            retention=base.retention,
        )
    )
    decision = service.evaluate(_remember_request())
    assert decision.outcome is MemoryGovernanceOutcome.REQUIRE_REVIEW


def test_gr12_a4_r3_r1_m14_missing_user_scope_fails_closed() -> None:
    from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope

    scope = EntityMemoryScope(tenant_id=_TENANT, user_id="")
    with pytest.raises(MemoryGovernanceDenied) as exc:
        memory_control_scope_from_entity_scope(scope)
    assert exc.value.decision.reason_code is MemoryGovernanceReasonCode.CROSS_SCOPE


def test_gr12_a4_r3_r1_m15_execution_proof_nodes_registered() -> None:
    assert_proof_nodes_registered(GR12_A4_R3_R1_EXECUTION_PROOF_NODES)


def test_gr12_a4_r3_r1_m16_no_provider_storage_permission_authority() -> None:
    forbidden = (
        "MemoryGovernanceOutcome.ALLOW",
        "MemoryGovernanceOutcome.DENY",
        "authorize_mutation",
        "permission_authority",
    )
    store_roots = (
        _MEMORY_ROOT / "stores",
        _REPO_ROOT / "intergrax" / "integrations" / "providers",
    )
    hits: list[str] = []
    for root in store_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "MemoryGovernance" not in text:
                continue
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(_REPO_ROOT)}:{token}")
    assert hits == []


def test_gr12_a4_r3_r1_m17_memory_domain_ast_cla04_import_gate() -> None:
    import_violations: list[str] = []
    for path in _iter_memory_py_files():
        import_violations.extend(_memory_imports_forbidden_cla04(path))
    assert import_violations == []


def test_gr12_a4_r3_r1_m18_no_second_memory_permission_engine() -> None:
    merge_hits = [
        path
        for path in _iter_memory_py_files()
        if "_merge_decisions" in path.read_text(encoding="utf-8")
    ]
    assert merge_hits == [_MEMORY_ROOT / "memory_security_governance_service.py"]


def test_gr12_a4_r3_r1_m19_diagnostics_do_not_grant_permission() -> None:
    from intergrax.memory import memory_observability_support

    source = inspect.getsource(memory_observability_support.emit_governance_diagnostic)
    assert "permits_mutation" not in source
    assert "MemoryGovernanceOutcome.ALLOW" not in source


def test_gr12_a4_r3_r1_m20_future_operator_rule_explicit() -> None:
    assert "architecture revisit" in GR12_MEMORY_FUTURE_LIVE_OPERATOR_RULE.lower()
    assert "dual" in GR12_MEMORY_FUTURE_LIVE_OPERATOR_RULE.lower()
    assert "CLA-04" in GR12_MEMORY_FUTURE_LIVE_OPERATOR_RULE


def test_gr12_a4_r3_r1_ready_for_audit_ssot() -> None:
    assert GR12_A4_R3_R1_TASK_STATUS == "READY FOR AUDIT"
    assert GR12_A4_R3_R1_QUALIFICATION_PROOF == GR12_MEMORY_R3_R1_QUALIFICATION_PROOF
    assert (_REPO_ROOT / GR12_A4_R3_R1_QUALIFICATION_PROOF).is_file()
    live = [
        s
        for s in GR12_MEMORY_MUTATION_SURFACES
        if s.context_class is Gr12MemoryMutationContextClass.LIVE_OPERATOR_CONTROL_PLANE
    ]
    assert len(live) == 1
    assert live[0].governance is Gr12MemoryMutationGovernanceBinding.NOT_PRODUCTION
    assert not GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION.dual_independent_authority


def test_gr12_a4_r3_r1_vector_qualified_no_reopen() -> None:
    vector = next(
        row
        for row in GR12_CONTROL_PLANE_SURFACES
        if row.path_id == "CP-VECTOR-INDEX-ADMIN"
    )
    assert vector.coverage is Gr12CoverageStatus.QUALIFIED


def test_gr12_a4_r3_r1_weak_boundary_scan_tracked_not_blocking() -> None:
    """Classify weak seams; qualification does not silently PASS FRZ-TYP."""
    contract_path = _MEMORY_ROOT / "contracts" / "memory_security_governance.py"
    text = contract_path.read_text(encoding="utf-8")
    assert ": Any" not in text
    assert "type: ignore" not in text
