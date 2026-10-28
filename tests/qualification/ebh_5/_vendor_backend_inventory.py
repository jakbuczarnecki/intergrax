# © Artur Czarnecki. All rights reserved.

"""VENDOR_BACKEND_DEPENDENCY_INVENTORY SSOT (EBH-5)."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory
from tests.qualification.config_x._config_x_concern_inventory import (
    CONFIG_X_CONCERN_INVENTORY,
    ConfigConcernRecord,
)
from tests.qualification.config_x._config_x_types import ConfigClassification
from tests.qualification.ebh_5._behavioral_proof_registry import (
    QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE,
)
from tests.qualification.ebh_5._runtime_platform_concerns import (
    RUNTIME_PLATFORM_CONCERNS,
    roadmap_domain_for_integration_category,
)
from tests.qualification.ebh_5._vendor_backend_types import (
    ConcernConclusion,
    CoreChangeToSwitch,
    ExtensibilityClassification,
    ExtensibilityPromised,
    RoadmapVendorDomain,
    VendorBackendConcernRecord,
)

_CONFIG_DOMAIN_MAP: dict[str, RoadmapVendorDomain] = {
    "INTEGRATIONS": RoadmapVendorDomain.DISCOVERED,
    "LLM_AI": RoadmapVendorDomain.LLM_EMBEDDING,
    "PERSISTENCE": RoadmapVendorDomain.RELATIONAL_PERSISTENCE,
    "OBSERVABILITY": RoadmapVendorDomain.TELEMETRY_OTLP,
    "PLATFORM_ACTIVATION": RoadmapVendorDomain.CONFIG_PROFILE,
}


def _extensibility_from_config(row: ConfigConcernRecord) -> ExtensibilityPromised:
    if row.classification == ConfigClassification.H_REFERENCE_LAB_TEST_ONLY:
        return ExtensibilityPromised.NO
    if row.classification == ConfigClassification.G_COMPATIBILITY_ADAPTER:
        return ExtensibilityPromised.PARTIAL
    if row.concern_id.startswith("integration."):
        return ExtensibilityPromised.YES
    return ExtensibilityPromised.YES


def _classification_from_config(row: ConfigConcernRecord) -> ExtensibilityClassification:
    if row.classification == ConfigClassification.H_REFERENCE_LAB_TEST_ONLY:
        return ExtensibilityClassification.INTERNAL
    if row.classification == ConfigClassification.F_ENVIRONMENT_DEPLOYMENT_CONSTANT:
        return ExtensibilityClassification.REPLACEABLE_NOT_YET
    if row.concern_id == "integration.key_value_cache":
        return ExtensibilityClassification.REPLACEABLE_QUALIFIED
    if row.concern_id.startswith("integration."):
        return ExtensibilityClassification.REPLACEABLE_NOT_YET
    if row.concern_id == "persistence.relational_sqlite_lab":
        return ExtensibilityClassification.REPLACEABLE_QUALIFIED
    if row.concern_id == "int_config.configure_existing":
        return ExtensibilityClassification.REPLACEABLE_QUALIFIED
    if row.concern_id == "observability.otlp_transport":
        return ExtensibilityClassification.REPLACEABLE_QUALIFIED
    if row.concern_id == "tokenizer.selection":
        return ExtensibilityClassification.REPLACEABLE_QUALIFIED
    return ExtensibilityClassification.REPLACEABLE_NOT_YET


def _roadmap_domain_for_config(row: ConfigConcernRecord) -> RoadmapVendorDomain:
    if row.concern_id.startswith("integration."):
        cat_value = row.concern_id.removeprefix("integration.")
        try:
            cat = IntegrationCategory(cat_value)
        except ValueError:
            return RoadmapVendorDomain.DISCOVERED
        return roadmap_domain_for_integration_category(cat)
    return _CONFIG_DOMAIN_MAP.get(row.domain, RoadmapVendorDomain.DISCOVERED)


def _frz_for_config(row: ConfigConcernRecord) -> tuple[str, ...]:
    base = ("FRZ-CFG-01", "FRZ-PLG-04", "FRZ-CTR-02")
    if row.concern_id.startswith("integration."):
        return base + ("FRZ-RPL-01", "FRZ-RPL-04", "FRZ-PLG-03")
    if row.domain == "OBSERVABILITY":
        return base + ("FRZ-OBS-04", "FRZ-CFG-08")
    if row.domain == "LLM_AI":
        return base + ("FRZ-RPL-02",)
    return base


def vendor_record_from_config_x(row: ConfigConcernRecord) -> VendorBackendConcernRecord:
    ext = _extensibility_from_config(row)
    rpl_h = _classification_from_config(row)
    core_switch = CoreChangeToSwitch.CONFIG_ONLY if ext != ExtensibilityPromised.NO else CoreChangeToSwitch.ADAPTER_ONLY
    behavioral = (
        "PENDING — EBH_5 Package C"
        if rpl_h == ExtensibilityClassification.REPLACEABLE_NOT_YET
        else QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE.get(
            row.concern_id,
            "PENDING — EBH_5 Package C",
        )
    )
    return VendorBackendConcernRecord(
        concern_id=row.concern_id,
        domain=_roadmap_domain_for_config(row),
        semantic_owner=row.effective_resolution_owner,
        canonical_contract_or_port=row.configuration_contract,
        composition_selection_owner=row.composition_owner,
        concrete_implementation=row.provider_surface,
        extensibility_promised=ext,
        configuration_profile_source=row.configuration_contract,
        lifecycle_activation_authority=row.effective_resolution_owner,
        vendor_code_location=row.provider_surface,
        core_consumers=(row.composition_owner,),
        core_change_required_to_switch=core_switch,
        fallback_behavior="fail-closed — CONFIG-X classification " + row.classification.value,
        structural_replaceability_evidence_id="CROSS_LAYER_REPLACEABILITY_MATRIX",
        behavioral_qualification_evidence_id=behavioral,
        frz_criteria=_frz_for_config(row),
        extensibility_classification=rpl_h,
        conclusion=ConcernConclusion.ACCEPTED,
        outstanding_debt=(
            "" if rpl_h != ExtensibilityClassification.REPLACEABLE_NOT_YET else "RPL-B/E L3 proof pending Package C"
        ),
        config_x_concern_id=row.concern_id,
        production_effective=row.classification != ConfigClassification.H_REFERENCE_LAB_TEST_ONLY,
    )


@lru_cache(maxsize=1)
def vendor_backend_concern_inventory() -> tuple[VendorBackendConcernRecord, ...]:
    from_config = tuple(vendor_record_from_config_x(row) for row in CONFIG_X_CONCERN_INVENTORY)
    extra_ids = {r.concern_id for r in from_config}
    extras = tuple(r for r in RUNTIME_PLATFORM_CONCERNS if r.concern_id not in extra_ids)
    return from_config + extras


VENDOR_BACKEND_DEPENDENCY_INVENTORY: Final[tuple[VendorBackendConcernRecord, ...]] = vendor_backend_concern_inventory()


def inventory_concern_ids() -> frozenset[str]:
    return frozenset(row.concern_id for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY)


def count_unclassified() -> int:
    return sum(
        1
        for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY
        if row.extensibility_classification == ExtensibilityClassification.ARCHITECTURE_BLOCKER
        and row.conclusion == ConcernConclusion.BLOCKER
    )


def concerns_with_extensibility_yes() -> tuple[VendorBackendConcernRecord, ...]:
    return tuple(
        row
        for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY
        if row.extensibility_promised in (ExtensibilityPromised.YES, ExtensibilityPromised.PARTIAL)
    )
