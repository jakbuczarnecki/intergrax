"""Deprecated import path — use scripts.dataset.dataset_generation.sample_wdc_dataset."""

from platform_proofs.scenarios.verified_product_identification.scripts.dataset.dataset_generation.sample_wdc_dataset import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.dataset.dataset_generation.sample_wdc_dataset")
    raise SystemExit(_m.main())
