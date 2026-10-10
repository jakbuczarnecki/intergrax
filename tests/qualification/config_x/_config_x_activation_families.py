# © Artur Czarnecki. All rights reserved.

"""FRZ-CFG-06 — sanctioned activation owner families (registration ≠ activation)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from tests.qualification.config_x._config_x_path_evidence import (
    primary_repo_paths_from_text,
    repo_root,
)

@dataclass(frozen=True, slots=True)
class ActivationFamilyRecord:
    family_id: str
    owner_paths: tuple[str, ...]
    guard_markers: tuple[str, ...]
    summary: str


ACTIVATION_FAMILY_RECORDS: Final[tuple[ActivationFamilyRecord, ...]] = (
    ActivationFamilyRecord(
        family_id="integration_catalog",
        owner_paths=("intergrax/integrations/registry/catalog.py",),
        guard_markers=("def get_entry",),
        summary="Catalog lookup — slug registration does not imply configured activation",
    ),
    ActivationFamilyRecord(
        family_id="integration_profile_resolution",
        owner_paths=("intergrax/integrations/registry/factory.py",),
        guard_markers=(
            "def resolve_slug",
            "IntegrationConfigurationError",
        ),
        summary="Profile/env/explicit slug resolution — unconfigured category fails closed",
    ),
    ActivationFamilyRecord(
        family_id="llm_adapter_registry",
        owner_paths=("intergrax/llm_adapters/llm_provider_registry.py",),
        guard_markers=("class LLMAdapterRegistry", "def create(cls, provider"),
        summary="Explicit provider id required — registration alone not effective",
    ),
    ActivationFamilyRecord(
        family_id="tokenizer_registry",
        owner_paths=("intergrax/tokenizers/registry/tokenizer_registry.py",),
        guard_markers=("_default_tokenizer_id", "No default tokenizer configured"),
        summary="Explicit default_tokenizer_id — no first-registered silent default",
    ),
    ActivationFamilyRecord(
        family_id="observability_role_binding",
        owner_paths=("intergrax/tools/providers/observability/resolve.py",),
        guard_markers=(
            "observability_backend_not_configured",
            "observability_role_backend_not_configured",
        ),
        summary="Role-bound observability backends — no slug-order probing",
    ),
    ActivationFamilyRecord(
        family_id="observability_profile_materialization",
        owner_paths=("intergrax/tools/registry/wiring.py",),
        guard_markers=(
            "observability_role_backends",
            "from_integration_profile",
        ),
        summary="IntegrationProfile observability role materialization into ToolWiringContext",
    ),
    ActivationFamilyRecord(
        family_id="execution_bound_integration_resolution",
        owner_paths=("intergrax/integrations/execution_bound_integration_resolution.py",),
        guard_markers=("class ExecutionBoundIntegrationResolution",),
        summary="Execution-bound configured/effective adoption — TRACE-X-P5-R2 owner",
    ),
    ActivationFamilyRecord(
        family_id="configure_existing_realization",
        owner_paths=("intergrax/integrations/existing_capability_configuration_service.py",),
        guard_markers=("class ExistingCapabilityConfigurationRealizationService",),
        summary="CONFIGURE_EXISTING realization — governed service, not ambient provider ctor",
    ),
    ActivationFamilyRecord(
        family_id="marketplace_tool_activation",
        owner_paths=("intergrax/tools/qualified_marketplace_tool_activation_resolver.py",),
        guard_markers=("QualifiedMarketplaceToolActivationResolver",),
        summary="Qualified marketplace activation — explicit governed resolver",
    ),
    ActivationFamilyRecord(
        family_id="llm_adapter_profile",
        owner_paths=("intergrax/llm_adapters/registry/profile.py",),
        guard_markers=("LLMProfile",),
        summary="LLM profile composition owner — typed adapter selection surface",
    ),
)


@dataclass(frozen=True, slots=True)
class ActivationBypassFinding:
    family_id: str
    repo_path: str
    detail: str


def _path_has_markers(rel: str, markers: tuple[str, ...]) -> bool:
    path = repo_root() / rel
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in markers)


@lru_cache(maxsize=1)
def discover_activation_bypass_findings() -> tuple[ActivationBypassFinding, ...]:
    findings: list[ActivationBypassFinding] = []
    for family in ACTIVATION_FAMILY_RECORDS:
        for rel in family.owner_paths:
            if not (repo_root() / rel).is_file():
                findings.append(
                    ActivationBypassFinding(
                        family_id=family.family_id,
                        repo_path=rel,
                        detail="sanctioned activation owner path missing on disk",
                    ),
                )
                continue
            if not _path_has_markers(rel, family.guard_markers):
                findings.append(
                    ActivationBypassFinding(
                        family_id=family.family_id,
                        repo_path=rel,
                        detail=f"missing guard markers for {family.family_id}",
                    ),
                )
    return tuple(findings)


@lru_cache(maxsize=1)
def discover_unsanctioned_activation_owner_paths() -> frozenset[str]:
    """Paths outside inventory/owner SSOT that expose alternate activation factories."""
    from tests.qualification.config_x._config_x_discovery import (
        discover_composition_root_paths,
    )
    from tests.qualification.config_x._config_x_path_evidence import (
        sanctioned_composition_owner_paths_from_inventory,
    )

    sanctioned = sanctioned_composition_owner_paths_from_inventory()
    unsanctioned: set[str] = set()
    activation_markers = (
        "def resolve_from_profile",
        "class LLMAdapterRegistry",
        "class ExecutionBoundIntegrationResolution",
        "QualifiedMarketplaceToolActivationResolver",
    )
    for rel in discover_composition_root_paths():
        path = repo_root() / rel
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not any(marker in text for marker in activation_markers):
            continue
        if rel in sanctioned:
            continue
        if rel.endswith("integration_wiring.py") or rel.endswith("tool_wiring.py"):
            continue
        unsanctioned.add(rel)
    return frozenset(unsanctioned)
