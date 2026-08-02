# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 safe serialization tests."""

from __future__ import annotations

import pytest

from intergrax.context.planning import (
    ContextArtifactLookupInputs,
    ContextArtifactRequirement,
    ContextBudgetClass,
    ContextMinimumPreservationRequirements,
    ContextPlan,
    ContextSourceBudgetAllocation,
    ContextSourceGroup,
)
from intergrax.context.contracts import ContextFragmentSource
from intergrax.context.serialization import (
    serialize_context_artifact_lookup_inputs_safe,
    serialize_context_plan_safe,
    serialize_session_history_snapshot_safe,
)
from intergrax.context.session_history import build_session_history_snapshot
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_snapshot_safe_serialization_excludes_content() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant-secret",
        context_scope_id="scope-secret",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="TOP_SECRET", entry_id="m1")],
    )
    payload = serialize_session_history_snapshot_safe(snapshot)
    assert payload["raw_content_included"] is False
    assert "TOP_SECRET" not in str(payload)
    assert "tenant-secret" not in str(payload)
    assert "scope-secret" not in str(payload)


def test_lookup_inputs_safe_serialization() -> None:
    inputs = ContextArtifactLookupInputs(
        tenant_id="tenant",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash",
        compression_target=ArtifactCompressionTarget(target_tokens=10),
        lossiness_profile="lossless",
        source_refs=("m1",),
    )
    payload = serialize_context_artifact_lookup_inputs_safe(inputs)
    assert payload["raw_content_included"] is False
    assert "tenant" not in str(payload)


def test_plan_safe_serialization() -> None:
    group = ContextSourceGroup(
        group_id="grp-1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash",
        token_estimate=5,
    )
    plan = ContextPlan(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
        resolved_global_budget_tokens=100,
        estimated_total_tokens=5,
        source_groups=(group,),
        source_allocations=(
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=5,
                selected_group_ids=("grp-1",),
            ),
        ),
        selected_group_ids=("grp-1",),
        excluded_group_ids=(),
        required_group_ids=(),
        protected_group_ids=(),
        compressible_group_ids=(),
        droppable_group_ids=(),
        trim_safe_group_ids=(),
        optimization_required=False,
        artifact_requirement=None,
        final_validation_requirements=("preserve_message_order",),
    )
    payload = serialize_context_plan_safe(plan)
    assert payload["raw_content_included"] is False
