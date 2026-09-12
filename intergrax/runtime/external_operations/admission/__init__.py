# © Artur Czarnecki. All rights reserved.

"""External operation admission runtime (R1)."""

from intergrax.runtime.external_operations.admission.audit_chain import (
    ExternalOperationAuditChain,
    InMemoryExternalOperationAuditChain,
)
from intergrax.runtime.external_operations.admission.governance_bridge import (
    resolve_governance_admission,
)
from intergrax.runtime.external_operations.admission.local_admission import (
    PolicyExternalOperationAdmission,
)
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)

__all__ = [
    "ExternalOperationAuditChain",
    "ExternalOperationExecutionGate",
    "InMemoryExternalOperationAuditChain",
    "PolicyExternalOperationAdmission",
    "resolve_governance_admission",
]
