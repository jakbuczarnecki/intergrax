"""Lifecycle ownership boundaries for ERL-QUAL-004 provisioning (documentation module).

Ownership model (normative for this scenario):

| Component | Owns |
| --- | --- |
| **Provisioning** | Preparation, materialization (provision), state availability checks, cleanup lifecycle. |
| **Scenario application** | Business workflow (order → payment → UNKNOWN → reconciliation path). |
| **Platform (Integrax runtime / ERL)** | Reliability execution on the observability spine. |
| **Proof harness** | Orchestrates variant selection, invokes provisioning lifecycle, projects evidence; does not own business rules. |

Dependency direction: runtime and application depend on ports; provisioning depends on dataset
semantics and target capabilities; the canonical dataset package does not depend on any provisioner.
"""

PROVISIONING_OWNS_LIFECYCLE_PHASES: frozenset[str] = frozenset(
    {
        "PREPARE",
        "PROVISION",
        "STATE_AVAILABILITY",
        "CLEANUP",
    }
)

SCENARIO_APPLICATION_OWNS: frozenset[str] = frozenset({"business_flow"})

PLATFORM_OWNS: frozenset[str] = frozenset({"reliability_execution"})
