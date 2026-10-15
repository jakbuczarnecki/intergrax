# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P3-INTEGRATION-GATE — IG-F01 (strict composition repeat dispatch negative).

Coverage class: **strict production composition boundary** (lifecycle-owned SQLite PIS +
``build_governed_external_work_production_runtime`` + ``ExternalWorkAdapter._tracked_execute``).
**HTTP-level re-entry with stable invocation_id: NOT PROVEN** (orchestrator mints fresh IDs).

Status: IMPLEMENTED — NOT EXECUTED — AWAITING OPERATOR AUTHORIZATION.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    RecordingAdmissionPort,
    _policy_bundle,
)
from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    load_d4_launcher_module,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_NODE_ID: Final = (
    "tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative.py"
    "::test_strict_session_external_work_no_second_provider_mutation_same_invocation_id"
)

_T0: Final = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)
_TENANT: Final = "ig-f01-tenant"
_WORKSPACE: Final = "ig-f01-workspace"
_PRINCIPAL: Final = "ig-f01-principal"
_PROVIDER: Final = "gec3_deterministic_fake"
_SCOPE_DIGEST: Final = "sha256:" + ("cd" * 32)
_INVOCATION_ID: Final = "inv-ig-f01-fixed-identity"
_IDEMPOTENCY_KEY: Final = "idem-ig-f01-fixed"


def _external_work_meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "ig-f01 scope",
        META_SCOPE_DIGEST: _SCOPE_DIGEST,
        META_IDEMPOTENCY_KEY: _IDEMPOTENCY_KEY,
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("25.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
        "external_work.workspace_ref": _WORKSPACE,
    }


def _fixed_invocation(task_id: str, run_id: str) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id=_INVOCATION_ID,
        provider_id=_PROVIDER,
        operation="create_work",
        task_id=task_id,
        run_id=run_id,
        request_digest="sha256:" + ("ef" * 32),
        started_at=_T0,
    )


def test_strict_session_external_work_no_second_provider_mutation_same_invocation_id(
    tmp_path,
) -> None:
    """IG-F01: second ordinary dispatch with same invocation_id is denied; provider called once."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"

    storage, adapter = d4._storage_and_adapter(tmp_path)
    environment = d4._ac3_environment()
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    composition_builder = d4._composition_builder(tmp_path)

    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )

    from governed_contractor_application.host.stores import (
        InMemoryContinuationStateStore,
        InMemoryGovernedExecutionStore,
        InMemoryPolicyBundleArtifactStore,
        InMemoryProofReceiptStore,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        pis = lifecycle.composition.provider_invocation_store
        assert isinstance(pis, SQLiteProviderInvocationStore)

        production_runtime = build_governed_external_work_production_runtime(
            fake,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
            task_scope=StaticActiveTaskScope(task_id),
            capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
            policy_bundle=_policy_bundle(),
            collaborative_work_repositories=cw,
            execution_store=InMemoryGovernedExecutionStore(),
            receipt_store=InMemoryProofReceiptStore(),
            bundle_store=InMemoryPolicyBundleArtifactStore(),
            continuation_store=InMemoryContinuationStateStore(),
            provider_invocation_store=pis,
            reliability_bridge=bridge,
            clock=lambda: _T0,
        )
        external_adapter = production_runtime.adapter
        invocation = _fixed_invocation(str(task_id), str(run_id))
        request = external_adapter.build_create_request(
            task_id=str(task_id),
            run_id=str(run_id),
            metadata=_external_work_meta(),
        )

        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            task_id=task_id,
            governance_tenant_id=_TENANT,
            governance_workspace_id=_WORKSPACE,
            governance_principal_id=_PRINCIPAL,
        ):
            first = external_adapter.create_and_map(
                request,
                principal_id=_PRINCIPAL,
                tenant_id=_TENANT,
                provider_invocation=invocation,
            )
            assert first.used is True
            assert fake.create_calls == 1

            stored = pis.get_invocation(_INVOCATION_ID)
            assert stored is not None
            assert stored.invocation_id == _INVOCATION_ID
            assert pis.get_outcome(_INVOCATION_ID) is None

            second = external_adapter.create_and_map(
                request,
                principal_id=_PRINCIPAL,
                tenant_id=_TENANT,
                provider_invocation=invocation,
            )

        assert fake.create_calls == 1
        assert second.used is False
        assert second.error_message is not None
        assert "dispatch_not_admitted" in second.error_message
        assert pis.get_invocation(_INVOCATION_ID) is not None
        assert pis.get_outcome(_INVOCATION_ID) is None
        assert pis.get_invocation(_INVOCATION_ID).invocation_id == _INVOCATION_ID

    assert _NODE_ID.endswith(
        "test_strict_session_external_work_no_second_provider_mutation_same_invocation_id",
    )
