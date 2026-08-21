# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from platform_proofs.tools.iterative_sql_investigation.dataset import (
    DEFAULT_SEED,
    PROOF_ROW_COUNT,
)

PROOF_ID = "TOOLS-ITERATIVE-SQL-INVESTIGATION"
DATASET_ID = "TOOLS-ITERATIVE-SQL-INVESTIGATION-DATASET"
DATASET_VERSION = "v1"
GROUND_TRUTH_VERSION = "A1-B1-C1"
SCHEMA_IDENTITY = "proof.parcel_events/v1"


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    dataset_version: str
    seed: int
    row_count: int
    ground_truth_version: str
    schema_identity: str

    @classmethod
    def canonical(cls) -> DatasetIdentity:
        return cls(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            seed=DEFAULT_SEED,
            row_count=PROOF_ROW_COUNT,
            ground_truth_version=GROUND_TRUTH_VERSION,
            schema_identity=SCHEMA_IDENTITY,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            "row_count": self.row_count,
            "ground_truth_version": self.ground_truth_version,
            "schema_identity": self.schema_identity,
        }


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    identity: DatasetIdentity
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"identity": self.identity.as_dict(), "sha256": self.sha256}


def compute_dataset_fingerprint(identity: DatasetIdentity | None = None) -> DatasetFingerprint:
    resolved = identity or DatasetIdentity.canonical()
    payload = json.dumps(resolved.as_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return DatasetFingerprint(identity=resolved, sha256=digest)
