# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2-R1 vector operator CLA-04 proofs (VEC-GOV-1–VEC-GOV-16)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from intergrax.applications._shared.vector_index_admin_governance import (
    MUTATION_TYPE_VECTOR_INDEX_PREPARE,
    VECTOR_INDEX_RESOURCE_TYPE,
    build_vector_index_prepare_mutation_request,
    vector_index_resource_id,
    vector_index_resource_scope,
)
from intergrax.applications._shared.vector_index_admin_service import (
    BLOCKER_INVALID_IDENTITY,
    BLOCKER_MISSING_BOUNDARY,
    BLOCKER_MISSING_PRINCIPAL,
    BLOCKER_POLICY,
    BLOCKER_POST_AUTH_STALE,
    BLOCKER_COMPATIBILITY,
    BLOCKER_TENANT_MISMATCH,
    BLOCKER_UNPROJECTABLE_CURRENT_STATE,
    VectorIndexAdminService,
)
from intergrax.applications._shared.vector_index_admin_wiring import resolve_vector_index_admin_wiring
from intergrax.applications._shared.vector_index_configuration_projection import (
    VECTOR_INDEX_ABSENT_REVISION,
    current_revision_from_description,
    target_revision_from_spec,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.vector_index_operator import VectorIndexPrepareOperatorRequest
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexCompatibilityError,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorIndexPrepareResult,
    VectorIndexSpec,
    VectorSearchCapability,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_PRINCIPAL = RequestIdentity(
    tenant_id=_TENANT,
    user_id="operator-1",
    principal_type=PrincipalType.USER,
    auth_subject="operator-1",
)


def _identity() -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name="catalog", tenant_id=_TENANT)


def _spec() -> VectorIndexSpec:
    return VectorIndexSpec(
        identity=_identity(),
        dense=DenseVectorChannelSpec(
            channel_name="dense",
            dimension=1024,
            metric="cosine",
        ),
        required_capabilities=frozenset(
            {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
        ),
        sparse_lexical=SparseLexicalChannelSpec(channel_name="sparse"),
    )


def _description(
    *,
    exists: bool = False,
    dimension: int = 1024,
    metric: str = "cosine",
) -> VectorIndexDescription:
    caps = {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
    return VectorIndexDescription(
        identity=_identity(),
        exists=exists,
        reachable=True,
        point_count=0,
        dense_dimension=dimension if exists else None,
        dense_metric=metric if exists else None,
        present_capabilities=frozenset(caps) if exists else frozenset(),
        dense_channel_name="dense" if exists else None,
        sparse_lexical_channel_name="sparse" if exists else None,
    )


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.allow",
            decision_id="dec-allow",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


@dataclass
class _FakeVectorIndexAdmin:
    descriptions: list[VectorIndexDescription]
    prepare_calls: int = 0
    prepare_outcome: VectorIndexPrepareOutcome = VectorIndexPrepareOutcome.CREATED
    prepare_error: Exception | None = None
    _describe_index: int = 0
    describe_calls: int = 0
    described_identities: list[VectorIndexIdentity] = field(default_factory=list)
    prepared_specs: list[VectorIndexSpec] = field(default_factory=list)

    def probe(self) -> HealthStatus:
        return HealthStatus.HEALTHY

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        self.describe_calls += 1
        self.described_identities.append(identity)
        index = min(self._describe_index, len(self.descriptions) - 1)
        self._describe_index += 1
        return self.descriptions[index]

    def prepare_index(self, spec: VectorIndexSpec) -> VectorIndexPrepareResult:
        self.prepare_calls += 1
        self.prepared_specs.append(spec)
        if self.prepare_error is not None:
            raise self.prepare_error
        return VectorIndexPrepareResult(
            outcome=self.prepare_outcome,
            description=_description(exists=True),
        )

    def close(self) -> None:
        return None


def _service(
    admin: _FakeVectorIndexAdmin,
    evaluator: _RecordingEvaluator,
) -> VectorIndexAdminService:
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    return VectorIndexAdminService(
        vector_index_administration=admin,
        mutation_authorization_boundary=boundary,
    )


def _request(mutation_id: str = "mut-vec-1") -> VectorIndexPrepareOperatorRequest:
    return VectorIndexPrepareOperatorRequest(mutation_id=mutation_id, spec=_spec())


def test_vec_gov_1_allow_create() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 1
    assert result.changed is True
    assert result.outcome is VectorIndexPrepareOutcome.CREATED
    assert result.before_revision == VECTOR_INDEX_ABSENT_REVISION


def test_vec_gov_2_deny() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.deny",
            decision_id="dec-deny",
        )
    )
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 0
    assert result.blocker_code == BLOCKER_POLICY


def test_vec_gov_3_require_human() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="hitl",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.hitl",
            decision_id="dec-hitl",
        )
    )
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 0
    assert result.policy_action == PolicyAction.REQUIRE_HUMAN.value


def test_vec_gov_4_escalate() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason="escalate",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.escalate",
            decision_id="dec-escalate",
        )
    )
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 0


def test_vec_gov_5_missing_principal() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    result = VectorIndexAdminService(
        vector_index_administration=admin,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=evaluator
        ),
    ).prepare(_request(), principal=None)
    assert admin.prepare_calls == 0
    assert evaluator.calls == []
    assert result.blocker_code == BLOCKER_MISSING_PRINCIPAL


def test_vec_gov_6_invalid_identity() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    bad_spec = MagicMock()
    bad_spec.identity = VectorIndexIdentity(logical_name="  ", tenant_id=_TENANT)
    result = _service(admin, evaluator).prepare(
        VectorIndexPrepareOperatorRequest(mutation_id="mut-bad", spec=bad_spec),
        principal=_PRINCIPAL,
    )
    assert evaluator.calls == []
    assert admin.prepare_calls == 0
    assert result.blocker_code == BLOCKER_INVALID_IDENTITY


def test_vec_gov_7_missing_boundary() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    result = VectorIndexAdminService(
        vector_index_administration=admin,
        mutation_authorization_boundary=None,
    ).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 0
    assert result.blocker_code == BLOCKER_MISSING_BOUNDARY


def test_vec_gov_8_stale_after_authorization() -> None:
    admin = _FakeVectorIndexAdmin(
        descriptions=[
            _description(exists=False),
            _description(exists=True),
        ]
    )
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 0
    assert result.blocker_code == BLOCKER_POST_AUTH_STALE


def test_vec_gov_9_external_evaluator_receives_cla04_request() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="capture",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.capture",
            decision_id="dec-capture",
        )
    )
    _service(admin, evaluator).prepare(_request(mutation_id="mut-ext"), principal=_PRINCIPAL)
    assert len(evaluator.calls) == 1
    captured = evaluator.calls[0]
    assert captured.mutation_type == MUTATION_TYPE_VECTOR_INDEX_PREPARE
    assert captured.current_revision == VECTOR_INDEX_ABSENT_REVISION
    assert captured.target_revision.startswith("sha256:")


def test_vec_gov_10_resource_mapping() -> None:
    identity = _identity()
    assert vector_index_resource_id(identity) == f"{_TENANT}/catalog"
    assert vector_index_resource_scope(_TENANT) == f"vector_index.tenant/{_TENANT}"
    request = build_vector_index_prepare_mutation_request(
        mutation_id="mut-map",
        principal=_PRINCIPAL,
        identity=identity,
        current_revision=VECTOR_INDEX_ABSENT_REVISION,
        target_revision="sha256:abc",
    )
    assert request.resource_type == VECTOR_INDEX_RESOURCE_TYPE
    assert request.resource_id == f"{_TENANT}/catalog"
    assert request.resource_scope == f"vector_index.tenant/{_TENANT}"


def test_vec_gov_11_absent_semantics() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert result.before_revision == VECTOR_INDEX_ABSENT_REVISION


def test_vec_gov_12_deterministic_digest_via_target_revision() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False), _description(exists=False)])
    evaluator = _RecordingEvaluator()
    svc = _service(admin, evaluator)
    r1 = svc.prepare(_request(mutation_id="m1"), principal=_PRINCIPAL)
    admin._describe_index = 0
    r2 = svc.prepare(_request(mutation_id="m2"), principal=_PRINCIPAL)
    assert evaluator.calls[0].target_revision == evaluator.calls[1].target_revision
    assert r1.after_revision == r2.after_revision


def test_vec_gov_15_already_compatible() -> None:
    existing = _description(exists=True)
    admin = _FakeVectorIndexAdmin(
        descriptions=[existing, existing],
        prepare_outcome=VectorIndexPrepareOutcome.ALREADY_COMPATIBLE,
    )
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert result.changed is False
    assert result.outcome is VectorIndexPrepareOutcome.ALREADY_COMPATIBLE
    assert result.before_revision != VECTOR_INDEX_ABSENT_REVISION


def test_vec_gov_16_composition_does_not_execute_prepare() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    wiring = resolve_vector_index_admin_wiring(
        vector_index_administration=admin,
        mutation_authorization_boundary=boundary,
    )
    assert admin.prepare_calls == 0
    assert isinstance(wiring.service, VectorIndexAdminService)
    assert wiring.service.vector_index_administration is admin
    assert wiring.service.mutation_authorization_boundary is boundary


def test_vec_gov_identity_principal_propagation() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="identity",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.identity",
            decision_id="dec-id",
        )
    )
    _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    principal = evaluator.calls[0].principal
    assert principal.tenant_id == _PRINCIPAL.tenant_id
    assert principal.user_id == _PRINCIPAL.user_id
    assert principal.auth_subject == _PRINCIPAL.auth_subject
    assert principal.principal_type == _PRINCIPAL.principal_type


def test_vec_gov_compatibility_error_zero_prepare() -> None:
    admin = _FakeVectorIndexAdmin(
        descriptions=[_description(exists=True), _description(exists=True)],
        prepare_error=VectorIndexCompatibilityError("incompatible"),
    )
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.prepare_calls == 1
    assert result.blocker_code == BLOCKER_COMPATIBILITY
    assert result.changed is False


def test_vec_gov_17_tenant_mismatch_fail_closed() -> None:
    """VEC-GOV-17 / TEN-2–TEN-6: permissive ALLOW cannot bypass tenant isolation."""
    other_tenant_spec = VectorIndexSpec(
        identity=VectorIndexIdentity(logical_name="catalog", tenant_id="tenant-b"),
        dense=_spec().dense,
        required_capabilities=_spec().required_capabilities,
        sparse_lexical=_spec().sparse_lexical,
    )
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(
        VectorIndexPrepareOperatorRequest(mutation_id="mut-tenant", spec=other_tenant_spec),
        principal=_PRINCIPAL,
    )
    assert admin.describe_calls == 0
    assert admin.prepare_calls == 0
    assert evaluator.calls == []
    assert result.changed is False
    assert result.outcome is None
    assert result.authorization_evidence is None
    assert result.blocker_code == BLOCKER_TENANT_MISMATCH
    assert result.policy_action == "tenant_mismatch"
    assert result.before_revision == VECTOR_INDEX_ABSENT_REVISION
    assert result.after_revision == VECTOR_INDEX_ABSENT_REVISION


def test_vec_gov_18_same_tenant_reaches_cla04() -> None:
    """VEC-GOV-18 / TEN-1: same-tenant principal reaches policy evaluation."""
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator()
    _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert len(evaluator.calls) == 1
    assert admin.describe_calls >= 1


def test_ten_7_principal_propagation_same_tenant_unchanged() -> None:
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="capture",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.capture",
            decision_id="dec-capture",
        )
    )
    _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert evaluator.calls[0].principal is _PRINCIPAL


def test_ten_request_identity_rejects_whitespace_only_tenant() -> None:
    with pytest.raises(ValidationError, match="tenant_id must be non-empty"):
        RequestIdentity(
            tenant_id="   ",
            user_id="operator-1",
            principal_type=PrincipalType.USER,
            auth_subject="operator-1",
        )


def test_vec_gov_19_canonical_identity_authority_to_execution_binding() -> None:
    """VEC-GOV-19 / BIND-1–BIND-8: canonical spec binds authority to execution."""
    padded_spec = VectorIndexSpec(
        identity=VectorIndexIdentity(logical_name=" catalog ", tenant_id=" tenant-a "),
        dense=_spec().dense,
        required_capabilities=_spec().required_capabilities,
        sparse_lexical=_spec().sparse_lexical,
    )
    canonical_identity = VectorIndexIdentity(logical_name="catalog", tenant_id=_TENANT)
    normalized_spec = VectorIndexSpec(
        identity=canonical_identity,
        dense=padded_spec.dense,
        required_capabilities=padded_spec.required_capabilities,
        sparse_lexical=padded_spec.sparse_lexical,
    )
    admin = _FakeVectorIndexAdmin(
        descriptions=[_description(exists=False), _description(exists=False)]
    )
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(
        VectorIndexPrepareOperatorRequest(mutation_id="mut-bind", spec=padded_spec),
        principal=_PRINCIPAL,
    )
    assert result.changed is True
    assert admin.prepare_calls == 1
    captured = evaluator.calls[0]
    assert captured.resource_id == f"{_TENANT}/catalog"
    assert captured.target_revision == target_revision_from_spec(normalized_spec)
    assert captured.target_revision != target_revision_from_spec(padded_spec)
    assert admin.described_identities == [canonical_identity, canonical_identity]
    prepared = admin.prepared_specs[0]
    assert prepared.identity == canonical_identity
    assert prepared.identity.logical_name == "catalog"
    assert prepared.identity.tenant_id == _TENANT
    assert admin.described_identities[0] == admin.described_identities[1] == prepared.identity


def test_vec_gov_identity_strips_whitespace_for_authority_mapping() -> None:
    padded_spec = VectorIndexSpec(
        identity=VectorIndexIdentity(logical_name=" catalog ", tenant_id=" tenant-a "),
        dense=_spec().dense,
        required_capabilities=_spec().required_capabilities,
        sparse_lexical=_spec().sparse_lexical,
    )
    admin = _FakeVectorIndexAdmin(descriptions=[_description(exists=False)])
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="map",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="vector.prepare.map",
            decision_id="dec-map",
        )
    )
    _service(admin, evaluator).prepare(
        VectorIndexPrepareOperatorRequest(mutation_id="mut-strip", spec=padded_spec),
        principal=_PRINCIPAL,
    )
    captured = evaluator.calls[0]
    assert captured.resource_id == f"{_TENANT}/catalog"


def test_vec_gov_proj_1_unprojectable_current_state() -> None:
    malformed = VectorIndexDescription(
        identity=_identity(),
        exists=True,
        reachable=True,
        point_count=0,
        dense_dimension=1024,
        dense_metric=None,
        present_capabilities=frozenset({VectorSearchCapability.DENSE}),
        dense_channel_name="dense",
        sparse_lexical_channel_name=None,
    )
    admin = _FakeVectorIndexAdmin(descriptions=[malformed])
    evaluator = _RecordingEvaluator()
    result = _service(admin, evaluator).prepare(_request(), principal=_PRINCIPAL)
    assert admin.describe_calls == 1
    assert evaluator.calls == []
    assert admin.prepare_calls == 0
    assert result.changed is False
    assert result.blocker_code == BLOCKER_UNPROJECTABLE_CURRENT_STATE
    assert result.authorization_evidence is None
