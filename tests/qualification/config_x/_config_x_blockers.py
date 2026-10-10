# © Artur Czarnecki. All rights reserved.

"""CONFIG-X closed-world blocker SSOT (I/J/K/L)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.config_x._config_x_types import ConfigClassification

_EVIDENCE = "test_config_x_qualification_gates.py::test_cx_q05_blocker_inventory_parity"


@dataclass(frozen=True, slots=True)
class ConfigBlockerRecord:
    blocker_id: str
    classification: ConfigClassification
    paths: tuple[str, ...]
    summary: str
    child_stage: str


CONFIG_X_BLOCKER_RECORDS: Final[tuple[ConfigBlockerRecord, ...]] = (
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-OBS-TOOL-01",
        classification=ConfigClassification.J_SILENT_FALLBACK,
        paths=("intergrax/tools/providers/observability/resolve.py",),
        summary=(
            "Tool observability backend resolution uses slug-order probing and "
            "`next(iter(backends.values()))` when role/default does not match — "
            "implicit backend selection without explicit configuration policy."
        ),
        child_stage="CONFIG-X-BLK-OBS-TOOL-01",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-TOK-01",
        classification=ConfigClassification.J_SILENT_FALLBACK,
        paths=("intergrax/tokenizers/registry/tokenizer_registry.py",),
        summary=(
            "TokenizerRegistry.get(name=None) and default() select first registered tokenizer "
            "— import/registration order becomes effective selection."
        ),
        child_stage="CONFIG-X-BLK-TOK-01",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-HARNESS-HTTP-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=(
            "intergrax/applications/_shared/harness_task_routes.py",
            "intergrax/applications/_shared/trace_explorer_routes.py",
        ),
        summary=(
            "Harness HTTP route models embed production-default tenant/user literals "
            "(e.g. tenant_id='default') — tracked EBH-4 freeze debt; not sanctioned "
            "typed deployment configuration boundary."
        ),
        child_stage="CONFIG-X-BLK-HARNESS-HTTP-01",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-MM-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=("intergrax/multimedia/image_smart_loader.py",),
        summary=(
            "Multimedia loader API defaults tenant_id to literal 'default' — "
            "ambient tenant selection (EBH-4 tracked CONFIG-X debt)."
        ),
        child_stage="CONFIG-X-BLK-MM-01",
    ),
    ConfigBlockerRecord(
        blocker_id="CONFIG-X-BLK-INT-P3-01",
        classification=ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        paths=("intergrax/integrations/_shared/p3/configs.py",),
        summary=(
            "Integration P3 shared config dataclass defaults tenant_id='default' — "
            "configuration contract encodes ambient tenant literal."
        ),
        child_stage="CONFIG-X-BLK-INT-P3-01",
    ),
)


def blocker_counts_by_classification() -> dict[ConfigClassification, int]:
    counts: dict[ConfigClassification, int] = dict.fromkeys(ConfigClassification, 0)
    for row in CONFIG_X_BLOCKER_RECORDS:
        counts[row.classification] += 1
    return counts
