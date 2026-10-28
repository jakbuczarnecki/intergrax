# © Artur Czarnecki. All rights reserved.

"""CONFIG-X @ fac9a43f historical reconciliation vs EBH-5 VBN inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CONFIG_X_HISTORICAL_PASS_SHA: Final[str] = "fac9a43f3b128fde63541d2b56ebd8f4a751e8fb"
CONFIG_X_CONCERN_COUNT: Final[int] = 54


@dataclass(frozen=True, slots=True)
class ConfigXReconciliationRow:
    config_x_concern_id: str
    vendor_concern_id: str
    equivalence_claim: str
    current_head_obligation: str


def config_x_reconciliation_rows() -> tuple[ConfigXReconciliationRow, ...]:
    from tests.qualification.config_x._config_x_concern_inventory import (
        CONFIG_X_CONCERN_INVENTORY,
    )

    rows: list[ConfigXReconciliationRow] = []
    for cx in CONFIG_X_CONCERN_INVENTORY:
        vendor_id = cx.concern_id
        rows.append(
            ConfigXReconciliationRow(
                config_x_concern_id=cx.concern_id,
                vendor_concern_id=vendor_id,
                equivalence_claim=(
                    "Historical CONFIG-X closed-world concern — FRZ-CFG PASS @ "
                    f"{CONFIG_X_HISTORICAL_PASS_SHA[:12]}… does not prove current-HEAD RPL"
                ),
                current_head_obligation=("EBH-5 VBN record + matrix row where extensibility_promised=yes"),
            ),
        )
    return tuple(rows)


def assert_config_x_coverage_complete(vendor_ids_with_config_x_lineage: frozenset[str]) -> None:
    from tests.qualification.config_x._config_x_concern_inventory import (
        CONFIG_X_CONCERN_INVENTORY,
    )

    expected = {row.concern_id for row in CONFIG_X_CONCERN_INVENTORY}
    missing = expected - vendor_ids_with_config_x_lineage
    assert not missing, f"CONFIG-X concerns missing from VBN inventory: {sorted(missing)}"
