# © Artur Czarnecki. All rights reserved.

from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
    validate_background_execution_admission_durability,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    BackgroundExecutionTenantMismatchError,
    bootstrap_background_execution,
    resolve_background_execution,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentry,
    BackgroundExecutionReentryAdmissionError,
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.identity_admission import (
    BackgroundExecutionIdentityMismatchError,
    assert_handler_run_id_matches_identity,
    assert_payload_run_id_consistent,
    assert_payload_task_id_consistent,
)
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
    DocumentStoreBackgroundExecutionIdentityPersistence,
    KvBackgroundExecutionIdentityPersistence,
    PersistedBackgroundExecutionIdentity,
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    EvidenceDurabilityClass,
    REQUIRED_BACKGROUND_CAUSAL_RELATIONS,
    RequiredAuditEvidencePersistenceError,
    admit_background_execution_handler,
    build_transport_triggered_execution_evidence,
    persist_required_audit_evidence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)

__all__ = [
    "BackgroundExecutionAdmissionDependencies",
    "BackgroundExecutionReentry",
    "BackgroundExecutionReentryAdmissionError",
    "BackgroundExecutionReentryDisposition",
    "BackgroundExecutionIdentity",
    "BackgroundExecutionIdentityMismatchError",
    "BackgroundExecutionIdentityPersistence",
    "BackgroundExecutionTenantMismatchError",
    "PersistedBackgroundExecutionIdentity",
    "assert_handler_run_id_matches_identity",
    "assert_payload_run_id_consistent",
    "assert_payload_task_id_consistent",
    "BackgroundTransportExecutionRef",
    "DocumentStoreBackgroundExecutionIdentityPersistence",
    "KvBackgroundExecutionIdentityPersistence",
    "wire_background_execution_identity_persistence",
    "admit_background_execution_reentry",
    "bootstrap_background_execution",
    "resolve_background_execution",
    "wire_background_execution_admission_dependencies",
    "validate_background_execution_admission_durability",
    "EvidenceDurabilityClass",
    "REQUIRED_BACKGROUND_CAUSAL_RELATIONS",
    "RequiredAuditEvidencePersistenceError",
    "admit_background_execution_handler",
    "build_transport_triggered_execution_evidence",
    "persist_required_audit_evidence",
]
