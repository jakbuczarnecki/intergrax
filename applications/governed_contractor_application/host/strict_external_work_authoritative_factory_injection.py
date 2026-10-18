# © Artur Czarnecki. All rights reserved.

"""Wrap artifact-authorized factories with strict GCA host injection (R3-F01)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from external_contractor_adapter.strict_host_dependency_injection import (
    StrictExternalWorkHostAgentInjectionError,
    materialize_with_strict_external_work_host_dependencies,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkOperationPort,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings


@dataclass(frozen=True, slots=True)
class StrictExternalWorkHostAgentInjection:
    """Frozen host ports applied only after authoritative factory materialization."""

    external_work: ExternalWorkIntegration
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None
    sanctioned_operation_port: SanctionedExternalWorkOperationPort
    require_sanctioned_external_work_mutations: bool = True


def strict_external_work_host_agent_injection_from_runtime(
    host_runtime: GovernedContractorHostRuntimeComposition,
) -> StrictExternalWorkHostAgentInjection:
    if not host_runtime.require_sanctioned_external_work_mutations:
        raise ValueError(
            "strict external work injection requires require_sanctioned_external_work_mutations",
        )
    port = host_runtime.sanctioned_external_work_operation_port
    if port is None:
        raise ValueError(
            "strict external work injection requires sanctioned_external_work_operation_port",
        )
    integration = host_runtime.external_work_integration
    if integration is None:
        raise ValueError(
            "strict external work injection requires external_work_integration",
        )
    return StrictExternalWorkHostAgentInjection(
        external_work=integration,
        authorization_boundary=host_runtime.meaningful_side_effect_authorization_boundary,
        sanctioned_operation_port=port,
        require_sanctioned_external_work_mutations=True,
    )


def build_strict_external_work_authoritative_factory_wrapper(
    injection: StrictExternalWorkHostAgentInjection,
) -> Callable[[CanonicalAgentFactory], CanonicalAgentFactory]:
    """Return a wrapper that preserves authoritative factory invocation semantics."""

    def _wrap(
        authoritative: CanonicalAgentFactory,
    ) -> CanonicalAgentFactory:
        def _wrapped(
            ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
            binding: AgentBinding,
        ):
            try:
                agent = authoritative(ctx, binding)
            except Exception as exc:
                raise RuntimeAgentFactoryResolutionError(
                    f"authoritative factory materialization failed: {exc}",
                ) from exc
            try:
                return materialize_with_strict_external_work_host_dependencies(
                    agent,
                    external_work=injection.external_work,
                    authorization_boundary=injection.authorization_boundary,
                    sanctioned_operation_port=injection.sanctioned_operation_port,
                    require_sanctioned_external_work_mutations=(
                        injection.require_sanctioned_external_work_mutations
                    ),
                )
            except StrictExternalWorkHostAgentInjectionError as exc:
                raise RuntimeAgentFactoryResolutionError(str(exc)) from exc

        return _wrapped

    return _wrap


__all__ = [
    "StrictExternalWorkHostAgentInjection",
    "build_strict_external_work_authoritative_factory_wrapper",
    "strict_external_work_host_agent_injection_from_runtime",
]
