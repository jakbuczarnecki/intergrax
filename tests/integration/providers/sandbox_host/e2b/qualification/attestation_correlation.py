# © Artur Czarnecki. All rights reserved.

"""Provider attestation correlation — requested vs attested vs observed scope."""

from __future__ import annotations

from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressAllowlist,
    canonicalize_network_egress_allowlist,
)

from .models import (
    HostProbeEvidence,
    ObservedNetworkScope,
    ProviderAttestationCorrelationEvidence,
    ProviderAttestationEvidence,
)


class ProviderAttestationCorrelation:
    """Evaluate requested policy against provider attestation and runtime probes."""

    @staticmethod
    def _scope_tuple(allowlist: NetworkEgressAllowlist) -> tuple[str, ...]:
        return tuple(host.canonical_form() for host in allowlist.hosts)

    @classmethod
    def evaluate(
        cls,
        *,
        requested_allowlist: NetworkEgressAllowlist,
        provider_attestation: ProviderAttestationEvidence | None,
        allowed_probe: HostProbeEvidence,
        denied_probe: HostProbeEvidence,
    ) -> ProviderAttestationCorrelationEvidence:
        """Correlate requested scope, provider attestation, and observed execution."""
        requested_scope = cls._scope_tuple(requested_allowlist)

        attested_scope: tuple[str, ...] | None = None
        attestation_verified = False
        if (
            provider_attestation is not None
            and provider_attestation.network_egress_allowlist_enforced is True
            and provider_attestation.enforced_network_hosts is not None
        ):
            attested_allowlist = canonicalize_network_egress_allowlist(
                provider_attestation.enforced_network_hosts,
            )
            attested_scope = cls._scope_tuple(attested_allowlist)
            attestation_verified = attested_scope == requested_scope

        observed_scope = ObservedNetworkScope(
            allowed_target=allowed_probe.target,
            allowed_reachable=allowed_probe.result.reachable,
            denied_target=denied_probe.target,
            denied_reachable=denied_probe.result.reachable,
        )
        execution_verified = (
            observed_scope.allowed_reachable and not observed_scope.denied_reachable
        )
        correlation_result = (
            "PASS" if attestation_verified and execution_verified else "DENIED"
        )

        return ProviderAttestationCorrelationEvidence(
            requested_scope=requested_scope,
            attested_scope=attested_scope,
            observed_scope=observed_scope,
            attestation_verified=attestation_verified,
            execution_verified=execution_verified,
            correlation_result=correlation_result,
        )
