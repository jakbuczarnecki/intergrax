"""Deprecated import path — use scripts.dataset.operator_lifecycle.run_vpi_data_pack_resume."""

from platform_proofs.scenarios.verified_product_identification.scripts.dataset.operator_lifecycle.run_vpi_data_pack_resume import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.dataset.operator_lifecycle.run_vpi_data_pack_resume")
    raise SystemExit(_m.main())
