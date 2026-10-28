# © Artur Czarnecki. All rights reserved.

"""Roadmap domain N/A — WITH EVIDENCE (G06-01 bounded static discovery)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._vendor_backend_types import RoadmapVendorDomain

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class RoadmapDomainDisposition:
    domain: RoadmapVendorDomain
    kind: str  # COVERED elsewhere | N/A — WITH EVIDENCE
    covered_concern_ids: tuple[str, ...]
    discovery_paths: tuple[str, ...]
    production_entrypoints: tuple[str, ...]
    discovery_limitations: str
    rationale: str


_DOMAIN_DISPOSITIONS: Final[tuple[RoadmapDomainDisposition, ...]] = (
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.CACHE,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=("integration.key_value_cache",),
        discovery_paths=(
            "intergrax/integrations/contracts/base.py",
            "tests/qualification/config_x/_config_x_concern_inventory.py",
        ),
        production_entrypoints=(
            "intergrax/integrations/registry/factory.py",
            "intergrax/applications/_shared/memory_wiring.py",
        ),
        discovery_limitations=(
            "IntegrationCategory enum + CONFIG-X inventory only; does not assert absence of cache tech "
            "outside integration profile selection."
        ),
        rationale=(
            "No standalone roadmap domain-4 IntegrationCategory; production KV/cache selection "
            "is concern integration.key_value_cache (domain KEY_VALUE), not a duplicate cache catalog row."
        ),
    ),
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.DISTRIBUTED_COORDINATION,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=(),
        discovery_paths=(
            "intergrax/runtime/observability/exporters/distributed/distributed_configuration.py",
            "intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py",
        ),
        production_entrypoints=("intergrax/runtime/observability/exporters/distributed/distributed_configuration.py",),
        discovery_limitations="Static file review only; does not prove absence of future coordination vendors.",
        rationale=(
            "Coordination is runtime/distributed contracts and composition wiring — "
            "no separate vendor/backend catalog concern in EBH-5 closed-world inventory."
        ),
    ),
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.LOGGING,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=("observability.otlp_transport",),
        discovery_paths=(
            "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
            "tests/qualification/config_x/_config_x_concern_inventory.py",
        ),
        production_entrypoints=("intergrax/applications/_shared/runtime_event_delivery_wiring.py",),
        discovery_limitations=("CONFIG-X has no LOGGING domain row; logging via observability concern only."),
        rationale=(
            "Logging sinks are wired via observability/event delivery profile — "
            "not an isolated IntegrationCategory row in CONFIG-X inventory."
        ),
    ),
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.METRICS,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=("observability.otlp_transport",),
        discovery_paths=(
            "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
            "intergrax/integrations/contracts/base.py",
        ),
        production_entrypoints=("intergrax/applications/_shared/runtime_event_delivery_wiring.py",),
        discovery_limitations="Same observability profile path as logging; no separate metrics IntegrationCategory.",
        rationale=(
            "Metrics export uses observability profile (OTLP/diagnostics backends) — "
            "no isolated metrics-vendor concern in EBH-5 inventory."
        ),
    ),
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.NETWORK_HTTP,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=(),
        discovery_paths=(
            "intergrax/integrations/contracts/base.py",
            "tests/qualification/ebh_5/_vendor_backend_inventory.py",
        ),
        production_entrypoints=(
            "intergrax/integrations/registry/factory.py",
            "intergrax/integrations/execution_bound_integration_resolution.py",
        ),
        discovery_limitations=(
            "HTTP selection absorbed by integration categories (EXTERNAL_MCP etc.); "
            "no domain-28 duplicate inventory row."
        ),
        rationale=(
            "HTTP client surfaces map to EXTERNAL_MCP / integration adapter concerns — "
            "roadmap domain 28 is not duplicated as its own inventory category."
        ),
    ),
    RoadmapDomainDisposition(
        domain=RoadmapVendorDomain.EVAL_ANALYTICS,
        kind="N/A — WITH EVIDENCE",
        covered_concern_ids=(),
        discovery_paths=(
            "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
            "intergrax/applications/_shared/memory_wiring.py",
            "applications/lab_application/host/factory.py",
        ),
        production_entrypoints=(),
        discovery_limitations=("Composition-root bounded scan only; EVOL-X/ENT-AUDIT-X may add eval vendors later."),
        rationale=(
            "Bounded scan: no production-effective eval-analytics vendor selection module "
            "in intergrax/applications composition roots; deferred to EVOL-X / ENT-AUDIT-X — "
            "not asserted as absent platform-wide beyond scanned paths."
        ),
    ),
)


def roadmap_domain_na_dispositions() -> tuple[RoadmapDomainDisposition, ...]:
    return _DOMAIN_DISPOSITIONS


def roadmap_domain_na_validation_errors() -> tuple[str, ...]:
    from tests.qualification.ebh_5._vendor_backend_inventory import inventory_concern_ids

    errors: list[str] = []
    known_concerns = inventory_concern_ids()
    by_domain = {d.domain: d for d in _DOMAIN_DISPOSITIONS}
    for domain in (
        RoadmapVendorDomain.CACHE,
        RoadmapVendorDomain.DISTRIBUTED_COORDINATION,
        RoadmapVendorDomain.LOGGING,
        RoadmapVendorDomain.METRICS,
        RoadmapVendorDomain.NETWORK_HTTP,
        RoadmapVendorDomain.EVAL_ANALYTICS,
    ):
        disp = by_domain.get(domain)
        if disp is None:
            errors.append(f"missing disposition record for {domain.value}")
            continue
        if not disp.kind.startswith("N/A"):
            errors.append(f"{domain.value}: expected N/A disposition")
        if not disp.discovery_limitations.strip():
            errors.append(f"{domain.value}: missing discovery_limitations")
        for cid in disp.covered_concern_ids:
            if cid not in known_concerns:
                errors.append(f"{domain.value}: covered concern {cid} not in inventory")
        for rel in disp.discovery_paths:
            path = _REPO_ROOT / rel
            if rel.endswith("/"):
                if not path.is_dir():
                    errors.append(f"{domain.value}: discovery path missing dir {rel}")
            elif not path.is_file() and not path.is_dir():
                errors.append(f"{domain.value}: discovery path missing {rel}")
        for rel in disp.production_entrypoints:
            path = _REPO_ROOT / rel
            if rel.endswith("/"):
                if not path.is_dir():
                    errors.append(f"{domain.value}: entrypoint dir missing {rel}")
            elif not path.is_file():
                errors.append(f"{domain.value}: entrypoint missing {rel}")
    return tuple(errors)
