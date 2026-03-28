# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Ready-made :class:`LegalExecutionPolicyPort` implementations for SaaS-style wiring.

* :class:`TenantRegistryLegalExecutionPolicy` — caps per ``tenant_id`` (in-memory map or DB-backed facade).
* :class:`RequestMetadataLegalExecutionPolicy` — per-request overrides via :attr:`RuntimeRequest.metadata`.
* :class:`ChainedLegalExecutionPolicy` — AND caps from several policies (tenant + subscription + request).

Replace the registry with your own adapter that loads from a store; keep ``resolve_nexus_layer_caps`` fast.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.governance.legal_platform_policy_governance import (
    LegalExecutionPolicyPort,
    LegalNexusLayerCaps,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def _and_caps(a: LegalNexusLayerCaps, b: LegalNexusLayerCaps) -> LegalNexusLayerCaps:
    return LegalNexusLayerCaps(
        allow_rag=a.allow_rag and b.allow_rag,
        allow_websearch=a.allow_websearch and b.allow_websearch,
        allow_tools=a.allow_tools and b.allow_tools,
    )


@dataclass(slots=True)
class ChainedLegalExecutionPolicy:
    """
    Combine policies with AND on each layer (most restrictive wins).
    """

    policies: tuple[LegalExecutionPolicyPort, ...]

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("ChainedLegalExecutionPolicy requires at least one policy.")

    def resolve_nexus_layer_caps(
        self,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalNexusLayerCaps:
        acc = LegalNexusLayerCaps()
        for p in self.policies:
            acc = _and_caps(acc, p.resolve_nexus_layer_caps(state=state, legal_config=legal_config))
        return acc


@dataclass(slots=True)
class TenantRegistryLegalExecutionPolicy:
    """
    Lookup :class:`LegalNexusLayerCaps` by ``state.request.tenant_id``.

    * ``tenant_id`` missing or ``None`` → ``default_caps``.
    * unknown tenant → ``default_caps`` (or set ``strict_tenant`` to raise ``KeyError``).
    """

    by_tenant: Mapping[str, LegalNexusLayerCaps]
    default_caps: LegalNexusLayerCaps | None = None
    strict_tenant: bool = False

    def resolve_nexus_layer_caps(
        self,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalNexusLayerCaps:
        tid = state.request.tenant_id
        if not tid:
            return self.default_caps if self.default_caps is not None else LegalNexusLayerCaps()
        if tid not in self.by_tenant:
            if self.strict_tenant:
                raise KeyError(f"Unknown tenant_id for legal execution policy: {tid!r}")
            return self.default_caps if self.default_caps is not None else LegalNexusLayerCaps()
        return self.by_tenant[tid]


@dataclass(slots=True)
class RequestMetadataLegalExecutionPolicy:
    """
    Optional per-request caps from ``RuntimeRequest.metadata[metadata_key]``.

    Expected value: ``dict`` with optional keys ``allow_rag``, ``allow_websearch``, ``allow_tools``
    (booleans). Omitted keys default to ``True`` (no extra restriction from metadata).
    """

    metadata_key: str = "legal_nexus_layer_caps"

    def resolve_nexus_layer_caps(
        self,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalNexusLayerCaps:
        raw = (state.request.metadata or {}).get(self.metadata_key)
        if not isinstance(raw, dict):
            return LegalNexusLayerCaps()

        def _b(name: str) -> bool:
            if name not in raw:
                return True
            v = raw[name]
            return bool(v)

        return LegalNexusLayerCaps(
            allow_rag=_b("allow_rag"),
            allow_websearch=_b("allow_websearch"),
            allow_tools=_b("allow_tools"),
        )
