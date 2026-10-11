# © Artur Czarnecki. All rights reserved.

"""R2 remediation waves derived from R2-P0 persisted-family evidence."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.compat_x._compat_x_legacy_obligations import (
    legacy_r2_r2_remediation_family_ids,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import (
    MigrationDisposition,
    PersistedRemediationItem,
    PersistedRemediationWave,
    PersistedVersionDisposition,
)


def _remediation_item(family_id: str) -> PersistedRemediationItem:
    family = next(f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == family_id)
    writers = ", ".join(family.writer_paths)
    readers = ", ".join(family.reader_paths)
    version_state = (
        f"{family.version_disposition.value}"
        f" current={family.current_version!r}"
    )
    if family.version_disposition == PersistedVersionDisposition.MISSING_VERSION:
        required = "Add explicit durable schema_version + fail-closed reader gate on SQLite/document adapters."
    elif family.migration_disposition == MigrationDisposition.MIGRATION_REQUIRED_BUT_MISSING:
        required = "Document and implement domain-owned migration or current-only fail-closed policy."
    elif "IN-SCOPE BLOCKER" in family.notes:
        required = "Persist PackageAgentContractAuthorityRecord in SQLite durable bundle or narrow store protocol."
    else:
        required = "R2-R2 migration/read-compat certification."
    migration = family.migration_disposition.value
    tests = f"tests/unit coverage for {family.canonical_payload_type} durable round-trip + unknown version fail-closed"
    return PersistedRemediationItem(
        family_id=family_id,
        canonical_owner=family.owner_module_path,
        current_writer=writers,
        current_reader=readers,
        format_version_state=version_state,
        required_change=required,
        migration_behavior_required=migration,
        tests_required=tests,
        frz_mapping=family.frz_cmp_tags,
    )


@lru_cache(maxsize=1)
def persisted_r2_remediation_waves() -> tuple[PersistedRemediationWave, ...]:
    missing_version_ids = tuple(
        sorted(
            f.family_id
            for f in PERSISTED_SCHEMA_FAMILIES
            if f.version_disposition == PersistedVersionDisposition.MISSING_VERSION
        )
    )
    migration_gap_ids = legacy_r2_r2_remediation_family_ids()
    blocker_ids = tuple(
        f.family_id
        for f in PERSISTED_SCHEMA_FAMILIES
        if "IN-SCOPE BLOCKER" in f.notes
    )
    wave_r2_r1_ids = tuple(dict.fromkeys((*missing_version_ids, *blocker_ids)))
    return (
        PersistedRemediationWave(
            wave_id="COMPAT-X-R2-R1",
            title="Missing durable schema identities + strict reader gates",
            items=tuple(_remediation_item(fid) for fid in wave_r2_r1_ids),
        ),
        PersistedRemediationWave(
            wave_id="COMPAT-X-R2-R2",
            title="Migration/read-compat policy closure",
            items=tuple(_remediation_item(fid) for fid in migration_gap_ids),
        ),
        PersistedRemediationWave(
            wave_id="COMPAT-X-R2-FINAL",
            title="Current-HEAD reconciliation + FRZ-CMP-03/06 qualification",
            items=(),
        ),
    )
