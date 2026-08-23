# © Artur Czarnecki. All rights reserved.

"""AP-11-FIX-2 composite application-environment scope coexistence proof."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    RollbackRuntimeRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.application_environment_identity import (
    ApplicationEnvironmentIdentity,
)
from intergrax.agent_distribution.errors import AgentDistributionNotFoundError
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _ARTIFACT,
    _build_request,
    _install_request,
    build_admin_stack,
    admin_test_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP_A = "app-a"
_APP_B = "app-b"
_ENV = "prod"

app_a_scope = (_APP_A, _ENV)
app_b_scope = (_APP_B, _ENV)


def _scope(application_id: str, environment_id: str) -> ApplicationEnvironmentIdentity:
    return ApplicationEnvironmentIdentity(
        application_id=application_id,
        application_environment_id=environment_id,
    )


def _setup_application(stack, *, application_id: str, binding_id: str, slot_id: str) -> None:
    principal = admin_test_principal()
    install = _install_request()
    install = install.model_copy(
        update={
            "installation_id": f"inst-{application_id}",
            "installation_slot_id": slot_id,
        }
    )
    stack.service.install_agent(
        application_id=application_id,
        application_environment_id=_ENV,
        request=install,
        principal=principal,
    )
    bind = BindAgentRequest(
        mutation_id=f"mut-bind-{application_id}",
        application_binding_id=binding_id,
        logical_agent_id="researcher",
        installation_slot_id=slot_id,
    )
    stack.service.bind_agent(
        application_id=application_id,
        application_environment_id=_ENV,
        request=bind,
        principal=principal,
    )
    stack.service.enable_binding(
        application_id=application_id,
        application_environment_id=_ENV,
        application_binding_id=binding_id,
        request=SetAgentEnablementRequest(
            mutation_id=f"mut-enable-{application_id}",
            expected_revision=0,
        ),
        principal=principal,
    )


def _build_and_activate(
    stack,
    *,
    application_id: str,
    revision_id: str,
    pointer: int = 0,
    expected_prior: str | None = None,
) -> None:
    built = stack.service.build_application_revision(
        application_id=application_id,
        application_environment_id=_ENV,
        request=_build_request(revision_id),
        principal=admin_test_principal(),
    )
    stack.service.activate_revision(
        application_id=application_id,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id=f"mut-{revision_id}",
            runtime_revision_id=revision_id,
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=pointer,
            expected_prior_traffic_revision_id=expected_prior,
        ),
    )


def test_app_a_and_app_b_prod_coexist_activate_and_rollback_isolation() -> None:
    stack = build_admin_stack()
    _setup_application(
        stack,
        application_id=_APP_A,
        binding_id="bind-a",
        slot_id="slot-a",
    )
    _setup_application(
        stack,
        application_id=_APP_B,
        binding_id="bind-b",
        slot_id="slot-b",
    )

    bindings_a = stack.service.list_bindings(
        application_id=_APP_A,
        application_environment_id=_ENV,
    ).bindings
    bindings_b = stack.service.list_bindings(
        application_id=_APP_B,
        application_environment_id=_ENV,
    ).bindings
    assert len(bindings_a) == 1
    assert len(bindings_b) == 1
    assert bindings_a[0].application_binding_id == "bind-a"
    assert bindings_b[0].application_binding_id == "bind-b"

    _build_and_activate(stack, application_id=_APP_A, revision_id="rev-a-1")
    _build_and_activate(stack, application_id=_APP_B, revision_id="rev-b-1")

    serving_a = stack.service.inspect_serving(
        application_id=_APP_A,
        application_environment_id=_ENV,
    )
    serving_b = stack.service.inspect_serving(
        application_id=_APP_B,
        application_environment_id=_ENV,
    )
    assert serving_a.traffic_serving_revision_id == "rev-a-1"
    assert serving_b.traffic_serving_revision_id == "rev-b-1"
    assert serving_a.serving_pointer_revision == 1
    assert serving_b.serving_pointer_revision == 1

    _build_and_activate(
        stack,
        application_id=_APP_A,
        revision_id="rev-a-2",
        pointer=1,
        expected_prior="rev-a-1",
    )

    serving_a_after = stack.service.inspect_serving(
        application_id=_APP_A,
        application_environment_id=_ENV,
    )
    serving_b_after = stack.service.inspect_serving(
        application_id=_APP_B,
        application_environment_id=_ENV,
    )
    assert serving_a_after.traffic_serving_revision_id == "rev-a-2"
    assert serving_b_after.traffic_serving_revision_id == "rev-b-1"
    assert serving_b_after.serving_pointer_revision == 1

    stack.service.rollback_revision(
        application_id=_APP_A,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=RollbackRuntimeRevisionRequest(
            mutation_id="mut-rollback-a",
            expected_current_traffic_revision_id="rev-a-2",
            expected_serving_pointer_revision=2,
        ),
    )

    serving_a_restored = stack.service.inspect_serving(
        application_id=_APP_A,
        application_environment_id=_ENV,
    )
    serving_b_restored = stack.service.inspect_serving(
        application_id=_APP_B,
        application_environment_id=_ENV,
    )
    assert serving_a_restored.traffic_serving_revision_id == "rev-a-1"
    assert serving_b_restored.traffic_serving_revision_id == "rev-b-1"
    assert serving_b_restored.serving_pointer_revision == 1

    assert _scope(*app_a_scope) in stack.state.serving_records
    assert _scope(*app_b_scope) in stack.state.serving_records
    assert stack.state.serving_records[_scope(*app_a_scope)].traffic_serving_revision_id == "rev-a-1"
    assert stack.state.serving_records[_scope(*app_b_scope)].traffic_serving_revision_id == "rev-b-1"


def test_cross_scope_revision_inspection_rejected() -> None:
    stack = build_admin_stack()
    _setup_application(
        stack,
        application_id=_APP_A,
        binding_id="bind-a",
        slot_id="slot-a",
    )
    _build_and_activate(stack, application_id=_APP_A, revision_id="rev-a-only")
    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.inspect_revision(
            application_id=_APP_B,
            application_environment_id=_ENV,
            runtime_revision_id="rev-a-only",
        )
