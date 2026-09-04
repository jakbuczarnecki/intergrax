# © Artur Czarnecki. All rights reserved.

"""Subprocess recovery reader for DIAG-FUNCTIONAL-READ-R1-R3 Mongo qualification."""

from __future__ import annotations

import json
import os
import sys

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
)

_CURSOR_SECRET = b"diag-functional-read-r1r3-qualification-secret"


def main() -> int:
    uri = os.environ["DIAG_R1R3_MONGO_URI"]
    collection = os.environ["DIAG_R1R3_COLLECTION"]
    tenant_id = os.environ["DIAG_R1R3_TENANT"]
    task_id = validate_task_id(os.environ["DIAG_R1R3_TASK"])
    run_id = validate_run_id(os.environ["DIAG_R1R3_RUN"])
    page_size = int(os.environ["DIAG_R1R3_PAGE_SIZE"])
    expected_count = int(os.environ["DIAG_R1R3_EXPECTED_COUNT"])
    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1r3",
            collection_name=collection,
        ),
    )
    persistence = DocumentStoreFunctionalEvidencePersistence(
        inner,
        cursor_secret=_CURSOR_SECRET,
    )
    recovered = collect_all_evidence(
        persistence,
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        page_size=page_size,
    )
    _ = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            page_size=page_size,
        ),
    )
    inner.close()
    sys.stdout.write(
        json.dumps(
            {
                "recovered_count": len(recovered),
                "expected_count": expected_count,
                "pid": os.getpid(),
            },
        ),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
