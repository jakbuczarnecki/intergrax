# © Artur Czarnecki. All rights reserved.

"""Current-HEAD mechanical CONFIG-X concern classification sweep (54/54)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from intergrax.integrations.contracts.base import IntegrationCategory

from tests.qualification.config_x._config_x_activation_families import (
    discover_activation_bypass_findings,
    discover_unsanctioned_activation_owner_paths,
)
from tests.qualification.config_x._config_x_types import ConfigBlockerRecord
from tests.qualification.config_x._config_x_concern_inventory import (
    CONFIG_X_CONCERN_INVENTORY,
    ConfigConcernRecord,
)
from tests.qualification.config_x._config_x_discovery import (
    _ACTIVE_BLOCKER_FORBIDDEN_MARKERS,
    discover_class_names_defining_resolver,
)
from tests.qualification.config_x._config_x_owner_discovery import (
    CONFIG_X_OWNER_EXPECTATIONS,
    discover_owner_paths,
)
from tests.qualification.config_x._config_x_path_evidence import (
    expand_inventory_glob_pattern,
    expand_provider_surface_glob,
    primary_repo_paths_from_text,
    repo_root,
)
from tests.qualification.config_x._config_x_semantic_production_scan import (
    discover_semantic_i_blocker_paths,
)
from tests.qualification.config_x._config_x_types import (
    BLOCKER_CLASSIFICATIONS,
    ConfigClassification,
)

_SANCTIONED_RESOLVER_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "ExecutionBoundIntegrationResolution",
        "QualifiedMarketplaceToolActivationResolver",
    },
)

_INTEGRATION_FACTORY_MARKERS: tuple[str, ...] = (
    "def resolve_slug",
    "def resolve_from_profile",
)


@dataclass(frozen=True, slots=True)
class ConcernClassificationEvidence:
    concern_id: str
    inventory_classification: ConfigClassification
    mechanical_classification: ConfigClassification
    evidence_notes: tuple[str, ...]


@lru_cache(maxsize=1)
def _integration_categories_with_provider_evidence() -> frozenset[str]:
    """Category values with on-disk provider tree or manifest registration evidence."""
    found: set[str] = set()
    providers_root = repo_root() / "intergrax/integrations/providers"
    if providers_root.is_dir():
        for child in providers_root.iterdir():
            if child.is_dir() and not child.name.startswith("_"):
                found.add(child.name)
    integrations_root = repo_root() / "intergrax/integrations"
    for path in integrations_root.rglob("*.py"):
        rel = path.relative_to(repo_root()).as_posix()
        if "/tests/" in rel or "docker/runtime-context" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "categories=" not in text or "IntegrationCategory." not in text:
            continue
        for category in IntegrationCategory:
            if f"IntegrationCategory.{category.name}" in text:
                found.add(category.value)
    return frozenset(found)


def _integration_category_has_provider_evidence(category: IntegrationCategory) -> bool:
    return category.value in _integration_categories_with_provider_evidence()


def _file_exists(rel: str) -> bool:
    return (repo_root() / rel).is_file()


def _file_has_markers(rel: str, markers: tuple[str, ...]) -> bool:
    path = repo_root() / rel
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in markers)


def _mechanical_evidence_for_concern(row: ConfigConcernRecord) -> ConcernClassificationEvidence:
    notes: list[str] = []
    mechanical = row.classification

    if row.classification in BLOCKER_CLASSIFICATIONS:
        return ConcernClassificationEvidence(
            concern_id=row.concern_id,
            inventory_classification=row.classification,
            mechanical_classification=row.classification,
            evidence_notes=("inventory row marked active blocker classification",),
        )

    if not row.configuration_contract.strip():
        return ConcernClassificationEvidence(
            concern_id=row.concern_id,
            inventory_classification=row.classification,
            mechanical_classification=ConfigClassification.L_UNCLEAR,
            evidence_notes=("empty configuration_contract",),
        )

    owner_paths = primary_repo_paths_from_text(row.composition_owner)
    owner_glob = expand_inventory_glob_pattern(row.composition_owner)
    if not owner_paths and owner_glob:
        owner_paths = tuple(sorted(owner_glob))[:1]
    if not owner_paths:
        notes.append("no extractable composition_owner path")
        mechanical = ConfigClassification.L_UNCLEAR
    else:
        missing = [p for p in owner_paths if not _file_exists(p)]
        if missing and not owner_glob:
            notes.append(f"composition_owner missing: {', '.join(missing)}")
            mechanical = ConfigClassification.L_UNCLEAR

    if row.concern_id.startswith("integration."):
        cat_value = row.concern_id.removeprefix("integration.")
        try:
            category = IntegrationCategory(cat_value)
        except ValueError:
            mechanical = ConfigClassification.L_UNCLEAR
            notes.append(f"unknown integration category {cat_value!r}")
        else:
            provider_dir = repo_root() / "intergrax/integrations/providers" / cat_value
            if not provider_dir.is_dir() and not _integration_category_has_provider_evidence(
                category,
            ):
                mechanical = ConfigClassification.L_UNCLEAR
                notes.append(
                    f"no provider tree and no catalog entry for integration.{cat_value}",
                )
            if not _file_has_markers(
                "intergrax/integrations/registry/factory.py",
                _INTEGRATION_FACTORY_MARKERS,
            ):
                mechanical = ConfigClassification.L_UNCLEAR
                notes.append("integration factory markers missing")
    else:
        surfaces = expand_provider_surface_glob(row.provider_surface)
        if not surfaces and row.provider_surface.strip():
            primary = primary_repo_paths_from_text(row.provider_surface)
            if primary and all(_file_exists(p) for p in primary):
                notes.append("provider_surface resolved via primary path")
            elif row.classification not in (
                ConfigClassification.F_ENVIRONMENT_DEPLOYMENT_CONSTANT,
                ConfigClassification.H_REFERENCE_LAB_TEST_ONLY,
            ):
                notes.append(f"empty provider_surface expansion for {row.provider_surface!r}")
                mechanical = ConfigClassification.L_UNCLEAR

    if mechanical not in BLOCKER_CLASSIFICATIONS:
        for rel in owner_paths:
            forbidden = _ACTIVE_BLOCKER_FORBIDDEN_MARKERS.get(rel)
            if forbidden is None:
                continue
            text = (repo_root() / rel).read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                mechanical = ConfigClassification.J_SILENT_FALLBACK
                if "tenant_id" in forbidden[0] or "default" in forbidden[0]:
                    mechanical = ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION
                notes.append(f"wave-1 forbidden marker still present in {rel}")

    return ConcernClassificationEvidence(
        concern_id=row.concern_id,
        inventory_classification=row.classification,
        mechanical_classification=mechanical,
        evidence_notes=tuple(notes) if notes else ("mechanical evidence satisfied",),
    )


@lru_cache(maxsize=1)
def sweep_concern_classification_evidence() -> tuple[ConcernClassificationEvidence, ...]:
    return tuple(_mechanical_evidence_for_concern(row) for row in CONFIG_X_CONCERN_INVENTORY)


@lru_cache(maxsize=1)
def discover_duplicate_configuration_authority_paths() -> frozenset[str]:
    """K-class: extra integration slug resolution owners beyond sanctioned factory."""
    sanctioned = frozenset({"intergrax/integrations/registry/factory.py"})
    duplicates: set[str] = set()
    for root in (repo_root() / "intergrax", repo_root() / "applications", repo_root() / "agents"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root()).as_posix()
            if (
                "/tests/" in rel
                or "/qualification/" in rel
                or "docker/runtime-context" in rel
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "def resolve_slug" in text and "def resolve_from_profile" in text:
                if rel not in sanctioned:
                    duplicates.add(rel)
    return frozenset(duplicates)


@lru_cache(maxsize=1)
def discover_unsanctioned_resolver_class_names() -> frozenset[str]:
    discovered = discover_class_names_defining_resolver()
    return frozenset(
        name for name in discovered if name not in _SANCTIONED_RESOLVER_CLASS_NAMES
    )


def _blocker_record(
    blocker_id: str,
    classification: ConfigClassification,
    paths: tuple[str, ...],
    summary: str,
) -> ConfigBlockerRecord:
    return ConfigBlockerRecord(
        blocker_id=blocker_id,
        classification=classification,
        paths=paths,
        summary=summary,
        child_stage="CONFIG-X-FINAL-R1",
        remediation_lineage="current-head mechanical discovery",
    )


@lru_cache(maxsize=1)
def derive_config_x_active_blocker_records() -> tuple[ConfigBlockerRecord, ...]:
    records: list[ConfigBlockerRecord] = []

    for evidence in sweep_concern_classification_evidence():
        if evidence.mechanical_classification not in BLOCKER_CLASSIFICATIONS:
            continue
        records.append(
            _blocker_record(
                blocker_id=f"CONFIG-X-ACTIVE-{evidence.concern_id}",
                classification=evidence.mechanical_classification,
                paths=tuple(
                    primary_repo_paths_from_text(
                        next(
                            row.composition_owner
                            for row in CONFIG_X_CONCERN_INVENTORY
                            if row.concern_id == evidence.concern_id
                        ),
                    ),
                )
                or ("<unmapped>",),
                summary="; ".join(evidence.evidence_notes),
            ),
        )

    for rel in sorted(discover_semantic_i_blocker_paths()):
        records.append(
            _blocker_record(
                blocker_id=f"CONFIG-X-ACTIVE-SEM-I-{rel.replace('/', '-')}",
                classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
                paths=(rel,),
                summary="FRZ-CFG-05 semantic hard-coded production selection",
            ),
        )

    for rel in sorted(discover_duplicate_configuration_authority_paths()):
        records.append(
            _blocker_record(
                blocker_id=f"CONFIG-X-ACTIVE-K-DUP-{rel.replace('/', '-')}",
                classification=ConfigClassification.K_DUPLICATE_CONFIGURATION_AUTHORITY,
                paths=(rel,),
                summary="duplicate integration resolution authority",
            ),
        )

    for rel in sorted(discover_unsanctioned_activation_owner_paths()):
        records.append(
            _blocker_record(
                blocker_id=f"CONFIG-X-ACTIVE-UNSANC-ACT-{rel.replace('/', '-')}",
                classification=ConfigClassification.K_DUPLICATE_CONFIGURATION_AUTHORITY,
                paths=(rel,),
                summary="unsanctioned activation owner outside closed-world inventory",
            ),
        )

    for finding in discover_activation_bypass_findings():
        records.append(
            _blocker_record(
                blocker_id=f"CONFIG-X-ACTIVE-FRZ06-{finding.family_id}",
                classification=ConfigClassification.J_SILENT_FALLBACK,
                paths=(finding.repo_path,),
                summary=finding.detail,
            ),
        )

    for rel, forbidden in _ACTIVE_BLOCKER_FORBIDDEN_MARKERS.items():
        path = repo_root() / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            records.append(
                _blocker_record(
                    blocker_id=f"CONFIG-X-ACTIVE-WAVE1-{rel.replace('/', '-')}",
                    classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
                    paths=(rel,),
                    summary="wave-1 forbidden marker regression",
                ),
            )

    for concern in CONFIG_X_OWNER_EXPECTATIONS:
        discovered = discover_owner_paths(concern)
        expected = CONFIG_X_OWNER_EXPECTATIONS[concern]
        if discovered != expected:
            records.append(
                _blocker_record(
                    blocker_id=f"CONFIG-X-ACTIVE-OWNER-{concern}",
                    classification=ConfigClassification.K_DUPLICATE_CONFIGURATION_AUTHORITY,
                    paths=tuple(sorted(discovered | expected)),
                    summary=f"owner gate mismatch for {concern}: {discovered!r} vs {expected!r}",
                ),
            )

    return tuple(records)
