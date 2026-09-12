"""Deprecated import path — use scripts.dataset.dataset_validation.run_data_pack_validation."""

from platform_proofs.scenarios.verified_product_identification.scripts.dataset.dataset_validation.run_data_pack_validation import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.dataset.dataset_validation.run_data_pack_validation")
    raise SystemExit(_m.main())
