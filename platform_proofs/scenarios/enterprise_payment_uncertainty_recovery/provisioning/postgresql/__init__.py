"""PostgreSQL provisioning adapter — one replaceable implementation of the scenario port."""

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.provisioner import (
    PostgreSqlScenarioProvisioner,
)

__all__ = ["PostgreSqlScenarioProvisioner"]
