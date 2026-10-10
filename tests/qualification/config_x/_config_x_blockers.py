# © Artur Czarnecki. All rights reserved.

"""CONFIG-X closed-world blocker SSOT (historical vs active I/J/K/L)."""

from __future__ import annotations

from typing import Final

from tests.qualification.config_x._config_x_current_classification import (
    derive_config_x_active_blocker_records,
)
from tests.qualification.config_x._config_x_types import (
    ConfigBlockerRecord,
    ConfigClassification,
)

CONFIG_X_HISTORICAL_BLOCKER_RECORDS: Final[tuple[ConfigBlockerRecord, ...]] = (
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-OBS-TOOL-01",
        classification=ConfigClassification.J_SILENT_FALLBACK,
        paths=("intergrax/tools/providers/observability/resolve.py",),
        summary=(
            "Wave-1: implicit observability backend selection via slug-order probing. "
            "REMEDIATED (R1-R1 / R1-R1-R1) — explicit IntegrationProfile.observability_roles "
            "and role materialization; no longer an active blocker."
        ),
        child_stage="CONFIG-X-BLK-OBS-TOOL-01",
        remediation_lineage=(
            "R1-R1 @ 4b8961e4a4376a95969384f76083d019eede9903 · "
            "R1-R1-R1 @ 4fa6f42f2cc24e9edd5c043a352b70f2668687bf"
        ),
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-TOK-01",
        classification=ConfigClassification.J_SILENT_FALLBACK,
        paths=("intergrax/tokenizers/registry/tokenizer_registry.py",),
        summary=(
            "Wave-1: TokenizerRegistry selected first registered tokenizer when default unset. "
            "REMEDIATED (R1) — explicit default_tokenizer_id; no longer an active blocker."
        ),
        child_stage="CONFIG-X-BLK-TOK-01",
        remediation_lineage="R1 @ d7183eb31d19967330687a5bc5345774c230e3aa",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-HARNESS-HTTP-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=(
            "intergrax/applications/_shared/harness_task_routes.py",
            "intergrax/applications/_shared/trace_explorer_routes.py",
        ),
        summary=(
            "Wave-1: harness/trace HTTP ambient tenant literals. "
            "REMEDIATED (R1) — principal-scoped / required tenant_id; no longer an active blocker."
        ),
        child_stage="CONFIG-X-BLK-HARNESS-HTTP-01",
        remediation_lineage="R1 @ d7183eb31d19967330687a5bc5345774c230e3aa",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-MM-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=("intergrax/multimedia/image_smart_loader.py",),
        summary=(
            "Wave-1: ImageSmartLoader default tenant_id literal. "
            "REMEDIATED (R1) — required tenant_id parameter; no longer an active blocker."
        ),
        child_stage="CONFIG-X-BLK-MM-01",
        remediation_lineage="R1 @ d7183eb31d19967330687a5bc5345774c230e3aa",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-INT-P3-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=("intergrax/integrations/_shared/p3/configs.py",),
        summary=(
            "Wave-1: VectorIntegrationConfig ambient tenant default. "
            "REMEDIATED (R1) — require_tenant_id / from_env fail-closed; no longer an active blocker."
        ),
        child_stage="CONFIG-X-BLK-INT-P3-01",
        remediation_lineage="R1 @ d7183eb31d19967330687a5bc5345774c230e3aa",
    ),
)

CONFIG_X_ACTIVE_BLOCKER_RECORDS: Final[tuple[ConfigBlockerRecord, ...]] = (
    derive_config_x_active_blocker_records()
)

# Backward-compatible alias: wave-1 historical inventory (not current active blockers).
CONFIG_X_BLOCKER_RECORDS: Final[tuple[ConfigBlockerRecord, ...]] = (
    CONFIG_X_HISTORICAL_BLOCKER_RECORDS
)

WAVE1_HISTORICAL_BLOCKER_IDS: Final[frozenset[str]] = frozenset(
    row.blocker_id for row in CONFIG_X_HISTORICAL_BLOCKER_RECORDS
)


def _counts_for(
    records: tuple[ConfigBlockerRecord, ...],
) -> dict[ConfigClassification, int]:
    counts: dict[ConfigClassification, int] = dict.fromkeys(ConfigClassification, 0)
    for row in records:
        counts[row.classification] += 1
    return counts


def historical_blocker_counts_by_classification() -> dict[ConfigClassification, int]:
    return _counts_for(CONFIG_X_HISTORICAL_BLOCKER_RECORDS)


def active_blocker_counts_by_classification() -> dict[ConfigClassification, int]:
    return _counts_for(CONFIG_X_ACTIVE_BLOCKER_RECORDS)


def blocker_counts_by_classification() -> dict[ConfigClassification, int]:
    """Current exit gate counts (active unresolved blockers only)."""
    return active_blocker_counts_by_classification()
