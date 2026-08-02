# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 planning contract tests."""

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
    artifact_lookup_key_kwargs_from_context_inputs,
    budget_class_for_execution_scope,
)
from intergrax.context.contracts import ContextFragmentSource
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)
from intergrax.runtime.context_lifecycle.serialization import compute_artifact_lookup_key_hash

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_budget_class_mapping() -> None:
    assert (
        budget_class_for_execution_scope(ModelCallExecutionScope.PRIMARY_MODEL_CALL)
        is ContextBudgetClass.PRIMARY_MODEL_INPUT
    )
    assert (
        budget_class_for_execution_scope(ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL)
        is ContextBudgetClass.INTERNAL_OPTIMIZATION_INPUT
    )


def test_lookup_inputs_reject_both_locators() -> None:
    from intergrax.runtime.context_lifecycle.contracts import ArtifactSourceRange

    with pytest.raises(ValueError, match="exactly one"):
        ContextArtifactLookupInputs(
            tenant_id="t1",
            context_scope_id="scope",
            artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
            source_content_hash="abc",
            compression_target=ArtifactCompressionTarget(target_tokens=100),
            lossiness_profile="lossless",
            source_refs=("m1",),
            source_range=ArtifactSourceRange(0, 1),
        )


def test_lookup_key_kwargs_exclude_strategy_versions() -> None:
    inputs = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash-1",
        compression_target=ArtifactCompressionTarget(target_tokens=128),
        lossiness_profile="lossless",
        source_refs=("m1", "m2"),
    )
    kwargs = artifact_lookup_key_kwargs_from_context_inputs(inputs)
    assert "strategy_id" not in kwargs
    assert "policy_version" not in kwargs
    assert kwargs["tenant_id"] == "t1"


def test_nexus_builds_equal_lookup_keys_from_same_inputs() -> None:
    inputs = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash-1",
        compression_target=ArtifactCompressionTarget(target_tokens=128),
        lossiness_profile="lossless",
        source_refs=("m1", "m2"),
    )
    partial = artifact_lookup_key_kwargs_from_context_inputs(inputs)
    key_a = ArtifactLookupKey(
        **partial,
        strategy_id="message_sequence.summary.v1",
        strategy_version="1",
        policy_version="pol-1",
        validation_contract_version="val-1",
    )
    key_b = ArtifactLookupKey(
        **partial,
        strategy_id="message_sequence.summary.v1",
        strategy_version="1",
        policy_version="pol-1",
        validation_contract_version="val-1",
    )
    assert compute_artifact_lookup_key_hash(key_a) == compute_artifact_lookup_key_hash(key_b)


def test_lookup_key_hash_changes_with_source_refs() -> None:
    base = dict(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash-1",
        compression_target=ArtifactCompressionTarget(target_tokens=128),
        lossiness_profile="lossless",
        strategy_id="message_sequence.summary.v1",
        strategy_version="1",
        policy_version="pol-1",
        validation_contract_version="val-1",
    )
    key_a = ArtifactLookupKey(source_refs=("m1", "m2"), **base)
    key_b = ArtifactLookupKey(source_refs=("m2", "m1"), **base)
    assert compute_artifact_lookup_key_hash(key_a) != compute_artifact_lookup_key_hash(key_b)


def test_context_plan_optimization_invariant() -> None:
    group = ContextSourceGroup(
        group_id="grp-1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash",
        token_estimate=10,
        compressible=True,
    )
    lookup = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash",
        compression_target=ArtifactCompressionTarget(target_tokens=5),
        lossiness_profile="lossless",
        source_refs=("m1",),
    )
    requirement = ContextArtifactRequirement(
        lookup_inputs=lookup,
        source_group_ids=("grp-1",),
        allowed_strategy_ids=("message_sequence.summary.v1",),
        minimum_preservation=ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=2,
            required_group_ids=(),
            protected_group_ids=(),
        ),
    )
    plan = ContextPlan(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
        resolved_global_budget_tokens=100,
        estimated_total_tokens=120,
        source_groups=(group,),
        source_allocations=(
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=10,
                selected_group_ids=("grp-1",),
            ),
        ),
        selected_group_ids=("grp-1",),
        excluded_group_ids=(),
        required_group_ids=(),
        protected_group_ids=(),
        compressible_group_ids=("grp-1",),
        droppable_group_ids=(),
        trim_safe_group_ids=(),
        optimization_required=True,
        artifact_requirement=requirement,
        final_validation_requirements=("preserve_message_order",),
    )
    assert plan.optimization_required
    assert plan.artifact_requirement is not None


def _minimal_plan_kwargs(**overrides: object) -> dict[str, object]:
    group = ContextSourceGroup(
        group_id="grp-1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash",
        token_estimate=10,
        compressible=True,
    )
    lookup = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash",
        compression_target=ArtifactCompressionTarget(target_tokens=5),
        lossiness_profile="lossless",
        source_refs=("m1",),
    )
    requirement = ContextArtifactRequirement(
        lookup_inputs=lookup,
        source_group_ids=("grp-1",),
        allowed_strategy_ids=("message_sequence.summary.v1",),
        minimum_preservation=ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=2,
            required_group_ids=(),
            protected_group_ids=(),
        ),
    )
    base: dict[str, object] = {
        "execution_scope": ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        "budget_class": ContextBudgetClass.PRIMARY_MODEL_INPUT,
        "resolved_global_budget_tokens": 100,
        "estimated_total_tokens": 120,
        "source_groups": (group,),
        "source_allocations": (
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=10,
                selected_group_ids=("grp-1",),
            ),
        ),
        "selected_group_ids": ("grp-1",),
        "excluded_group_ids": (),
        "required_group_ids": (),
        "protected_group_ids": (),
        "compressible_group_ids": ("grp-1",),
        "droppable_group_ids": (),
        "trim_safe_group_ids": (),
        "optimization_required": True,
        "artifact_requirement": requirement,
        "final_validation_requirements": ("preserve_message_order",),
    }
    base.update(overrides)
    return base


def test_strict_bool_group_flags_rejected() -> None:
    with pytest.raises(ValueError, match="required must be a bool"):
        ContextSourceGroup(
            group_id="grp-1",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_refs=("m1",),
            source_content_hash="hash",
            token_estimate=10,
            required=1,  # type: ignore[arg-type]
        )


def test_strict_optimization_required_rejected() -> None:
    with pytest.raises(ValueError, match="optimization_required must be a bool"):
        ContextPlan(**_minimal_plan_kwargs(optimization_required=1))


def test_wrong_source_groups_item_rejected() -> None:
    with pytest.raises(ValueError, match="source_groups\\[0\\] must be ContextSourceGroup"):
        ContextPlan(**_minimal_plan_kwargs(source_groups=("not-a-group",)))


def test_unknown_compressible_group_id_rejected() -> None:
    with pytest.raises(ValueError, match="compressible_group_ids must match group compressible flags"):
        ContextPlan(**_minimal_plan_kwargs(compressible_group_ids=("missing",)))


def test_classification_lists_must_match_group_flags() -> None:
    with pytest.raises(ValueError, match="required_group_ids must match group required flags"):
        ContextPlan(**_minimal_plan_kwargs(required_group_ids=("grp-1",)))


def test_selected_and_excluded_must_cover_all_groups() -> None:
    with pytest.raises(ValueError, match="selected and excluded group IDs must cover all source groups"):
        ContextPlan(**_minimal_plan_kwargs(selected_group_ids=(), excluded_group_ids=()))


def test_allocation_source_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="allocation source mismatch"):
        ContextPlan(
            **_minimal_plan_kwargs(
                source_allocations=(
                    ContextSourceBudgetAllocation(
                        source=ContextFragmentSource.RAG,
                        allocated_tokens=10,
                        selected_group_ids=("grp-1",),
                    ),
                ),
            )
        )


def test_allocation_missing_group_rejected() -> None:
    with pytest.raises(ValueError, match="allocation missing group"):
        ContextPlan(
            **_minimal_plan_kwargs(
                source_allocations=(
                    ContextSourceBudgetAllocation(
                        source=ContextFragmentSource.SESSION_HISTORY,
                        allocated_tokens=0,
                        selected_group_ids=(),
                    ),
                ),
            )
        )


def test_empty_strategy_id_rejected() -> None:
    lookup = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash",
        compression_target=ArtifactCompressionTarget(target_tokens=5),
        lossiness_profile="lossless",
        source_refs=("m1",),
    )
    with pytest.raises(ValueError, match="allowed_strategy_ids item"):
        ContextArtifactRequirement(
            lookup_inputs=lookup,
            source_group_ids=("grp-1",),
            allowed_strategy_ids=("",),
            minimum_preservation=ContextMinimumPreservationRequirements(
                preserve_message_order=True,
                preserve_roles=True,
                preserve_message_ids=True,
                preserve_tool_call_links=True,
                preserve_recent_tail_messages=2,
            ),
        )


def test_preservation_ids_match_plan_required_protected() -> None:
    protected_group = ContextSourceGroup(
        group_id="grp-protected",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash",
        token_estimate=10,
        required=True,
        protected=True,
        compressible=False,
    )
    artifact_group = ContextSourceGroup(
        group_id="grp-artifact",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m2",),
        source_content_hash="hash2",
        token_estimate=8,
        required=False,
        protected=False,
        compressible=True,
    )
    lookup = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash2",
        compression_target=ArtifactCompressionTarget(target_tokens=5),
        lossiness_profile="lossless",
        source_refs=("m2",),
    )
    requirement = ContextArtifactRequirement(
        lookup_inputs=lookup,
        source_group_ids=("grp-artifact",),
        allowed_strategy_ids=("message_sequence.summary.v1",),
        minimum_preservation=ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=2,
            required_group_ids=(),
            protected_group_ids=("grp-protected",),
        ),
    )
    with pytest.raises(ValueError, match="preservation required_group_ids must match plan required_group_ids"):
        ContextPlan(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
            resolved_global_budget_tokens=100,
            estimated_total_tokens=120,
            source_groups=(protected_group, artifact_group),
            source_allocations=(
                ContextSourceBudgetAllocation(
                    source=ContextFragmentSource.SESSION_HISTORY,
                    allocated_tokens=18,
                    selected_group_ids=("grp-protected", "grp-artifact"),
                ),
            ),
            selected_group_ids=("grp-protected", "grp-artifact"),
            excluded_group_ids=(),
            required_group_ids=("grp-protected",),
            protected_group_ids=("grp-protected",),
            compressible_group_ids=("grp-artifact",),
            droppable_group_ids=(),
            trim_safe_group_ids=(),
            optimization_required=True,
            artifact_requirement=requirement,
            final_validation_requirements=("preserve_message_order",),
        )


def test_preservation_ids_disjoint_from_artifact_target() -> None:
    group = ContextSourceGroup(
        group_id="grp-1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash",
        token_estimate=10,
        required=True,
        protected=False,
        compressible=True,
    )
    lookup = ContextArtifactLookupInputs(
        tenant_id="t1",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash="hash",
        compression_target=ArtifactCompressionTarget(target_tokens=5),
        lossiness_profile="lossless",
        source_refs=("m1",),
    )
    requirement = ContextArtifactRequirement(
        lookup_inputs=lookup,
        source_group_ids=("grp-1",),
        allowed_strategy_ids=("message_sequence.summary.v1",),
        minimum_preservation=ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=2,
            required_group_ids=("grp-1",),
            protected_group_ids=(),
        ),
    )
    with pytest.raises(ValueError, match="preservation IDs must be disjoint from artifact source_group_ids"):
        ContextPlan(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
            resolved_global_budget_tokens=100,
            estimated_total_tokens=120,
            source_groups=(group,),
            source_allocations=(
                ContextSourceBudgetAllocation(
                    source=ContextFragmentSource.SESSION_HISTORY,
                    allocated_tokens=10,
                    selected_group_ids=("grp-1",),
                ),
            ),
            selected_group_ids=("grp-1",),
            excluded_group_ids=(),
            required_group_ids=("grp-1",),
            protected_group_ids=(),
            compressible_group_ids=("grp-1",),
            droppable_group_ids=(),
            trim_safe_group_ids=(),
            optimization_required=True,
            artifact_requirement=requirement,
            final_validation_requirements=("preserve_message_order",),
        )
