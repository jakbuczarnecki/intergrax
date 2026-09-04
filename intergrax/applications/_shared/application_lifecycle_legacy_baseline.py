# © Artur Czarnecki. All rights reserved.

"""Bounded legacy allowances for application lifecycle conformance gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ApplicationLifecycleRuleId(StrEnum):
    AGENT_LIFECYCLE_BYPASS = "APPLICATION_ARCH_AGENT_LIFECYCLE_BYPASS"
    BUILD_APPLICATION_REGISTRY_BYPASS = "APPLICATION_ARCH_BUILD_APPLICATION_REGISTRY_BYPASS"
    MUTABLE_REGISTRY_TYPE_EXPOSURE = "APPLICATION_ARCH_MUTABLE_REGISTRY_TYPE_EXPOSURE"


@dataclass(frozen=True, slots=True)
class ApplicationLifecycleLegacyAllowance:
    path: str
    rule_id: ApplicationLifecycleRuleId
    symbol: str
    reason: str


_APPLICATION_LIFECYCLE_LEGACY_ALLOWANCES: tuple[ApplicationLifecycleLegacyAllowance, ...] = (
    ApplicationLifecycleLegacyAllowance(
        path="applications/attestation_demo/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/dispute_sim_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/governed_contractor_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/intergrax_assistant_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/lab_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/legal_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/local_workspace_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/poc_template_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/research_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
        symbol="build_application_registry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/attestation_demo/host/factory.py",
        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
        symbol="AgentRegistry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/intergrax_assistant_application/host/factory.py",
        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
        symbol="AgentRegistry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/lab_application/host/factory.py",
        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
        symbol="AgentRegistry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/poc_template_application/host/factory.py",
        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
        symbol="AgentRegistry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
    ApplicationLifecycleLegacyAllowance(
        path="applications/local_workspace_application/host/wiring.py",
        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
        symbol="AgentRegistry",
        reason="pre-canonical lifecycle; migrate in Stage 12",
    ),
)


def application_lifecycle_legacy_allowances() -> tuple[ApplicationLifecycleLegacyAllowance, ...]:
    return _APPLICATION_LIFECYCLE_LEGACY_ALLOWANCES


def is_legacy_application_lifecycle_violation_allowed(
    *,
    relative_path: str,
    rule_id: ApplicationLifecycleRuleId,
    symbol: str,
) -> bool:
    normalized = relative_path.replace("\\", "/")
    return any(
        allowance.path == normalized
        and allowance.rule_id is rule_id
        and allowance.symbol == symbol
        for allowance in _APPLICATION_LIFECYCLE_LEGACY_ALLOWANCES
    )
