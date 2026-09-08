"""Reduction metrics and policy experiment helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.builder import (
    SemanticRepresentationBuilder,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.contracts import (
    RepresentationReductionMetrics,
    SemanticRepresentationResult,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.policy import (
    RepresentationPolicyProfile,
    resolve_semantic_representation_policy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.ports import (
    TokenEstimatorPort,
)


@dataclass(frozen=True, slots=True)
class RepresentationPolicyExperimentResult:
    """One profile experiment outcome for a single catalog record."""

    profile: str
    result: SemanticRepresentationResult
    reduction: RepresentationReductionMetrics


def build_representation_reduction_metrics(
    *,
    before_chars: int,
    after_chars: int,
    before_tokens: int,
    after_tokens: int,
) -> RepresentationReductionMetrics:
    if before_chars <= 0:
        reduction_ratio = 0.0
    else:
        reduction_ratio = max(0.0, 1.0 - (after_chars / float(before_chars)))
    return RepresentationReductionMetrics(
        before_chars=before_chars,
        after_chars=after_chars,
        reduction_ratio=reduction_ratio,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
    )


def reduction_metrics_for_result(
    result: SemanticRepresentationResult,
    *,
    before_tokens: int,
) -> RepresentationReductionMetrics:
    return build_representation_reduction_metrics(
        before_chars=result.original_character_count,
        after_chars=result.final_character_count,
        before_tokens=before_tokens,
        after_tokens=result.estimated_tokens,
    )


def run_representation_policy_experiment(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
    profile: str,
    token_estimator: TokenEstimatorPort,
    legacy_semantic_text: str,
) -> RepresentationPolicyExperimentResult:
    policy = resolve_semantic_representation_policy(profile)
    if policy is None:
        before_tokens = token_estimator.estimate_tokens(legacy_semantic_text)
        result = SemanticRepresentationResult(
            source_ref=source_ref,
            original_character_count=len(legacy_semantic_text),
            final_character_count=len(legacy_semantic_text),
            estimated_tokens=before_tokens,
            truncated=False,
            preserved_field_count=0,
            representation_text=legacy_semantic_text,
        )
        reduction = build_representation_reduction_metrics(
            before_chars=len(legacy_semantic_text),
            after_chars=len(legacy_semantic_text),
            before_tokens=before_tokens,
            after_tokens=before_tokens,
        )
        return RepresentationPolicyExperimentResult(
            profile=profile,
            result=result,
            reduction=reduction,
        )

    builder = SemanticRepresentationBuilder(policy, token_estimator=token_estimator)
    build_output = builder.build(source_offer, source_ref=source_ref)
    result = build_output.result
    before_tokens = token_estimator.estimate_tokens(legacy_semantic_text)
    reduction = build_representation_reduction_metrics(
        before_chars=len(legacy_semantic_text),
        after_chars=result.final_character_count,
        before_tokens=before_tokens,
        after_tokens=result.estimated_tokens,
    )
    return RepresentationPolicyExperimentResult(
        profile=profile,
        result=result,
        reduction=reduction,
    )


def run_all_representation_policy_experiments(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
    token_estimator: TokenEstimatorPort,
    legacy_semantic_text: str,
    profiles: Sequence[str] | None = None,
) -> tuple[RepresentationPolicyExperimentResult, ...]:
    selected_profiles = profiles or RepresentationPolicyProfile.all_profiles()
    return tuple(
        run_representation_policy_experiment(
            source_offer,
            source_ref=source_ref,
            profile=profile,
            token_estimator=token_estimator,
            legacy_semantic_text=legacy_semantic_text,
        )
        for profile in selected_profiles
    )


def aggregate_reduction_metrics(
    experiments: Sequence[RepresentationPolicyExperimentResult],
) -> RepresentationReductionMetrics | None:
    if not experiments:
        return None
    before_chars = sum(item.reduction.before_chars for item in experiments)
    after_chars = sum(item.reduction.after_chars for item in experiments)
    before_tokens = sum(item.reduction.before_tokens for item in experiments)
    after_tokens = sum(item.reduction.after_tokens for item in experiments)
    return build_representation_reduction_metrics(
        before_chars=before_chars,
        after_chars=after_chars,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
    )
